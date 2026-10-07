"""Zero-shot TTS pipeline: text and prompt audio in, waveform out.

DELTA-TTS replaces only the autoregressive LLM of CosyVoice 3. Everything around it is the
original CosyVoice 3 code, loaded from the submodule in third_party/CosyVoice.

One call of `DeltaTTSPipeline.synthesize`
    1. CosyVoice 3 frontend   text tokens, prompt speech tokens, prompt mel, speaker embedding
    2. Output length          estimated from the prompt (delta_tts/duration.py)
    3. DELTA-TTS              speech tokens by iterative unmasking (delta_tts/generate.py)
    4. CosyVoice 3 decoder    flow matching + vocoder, conditioned on the prompt
"""

import logging
import sys
import uuid
from pathlib import Path

import torch

from delta_tts.duration import estimate_speech_len
from delta_tts.generate import generate
from delta_tts.model import DeltaTTS

COSYVOICE_DIR = Path(__file__).resolve().parent.parent / "third_party" / "CosyVoice"
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402

__all__ = ["DeltaTTSPipeline"]

logger = logging.getLogger(__name__)

# Token ids of "You are a helpful assistant.<|endofprompt|>", the instruction CosyVoice 3 is
# trained with. It is placed before the prompt transcript, as in CosyVoice 3.
INSTRUCT_TOKENS = [2610, 525, 264, 10950, 17847, 13, 151646]


class DeltaTTSPipeline:
    """Text and prompt audio in, waveform out.

    Args:
        backbone_dir: CosyVoice 3 model directory (Fun-CosyVoice3-0.5B-2512).
        checkpoint: Path to the DELTA-TTS checkpoint `delta_tts.pt`.
        device: Torch device for the DELTA-TTS model.
    """

    def __init__(self, backbone_dir: str, checkpoint: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.model = DeltaTTS(backbone_dir).to(self.device).float().eval()
        self.model.load_checkpoint(checkpoint)
        self.cosyvoice = CosyVoice3(backbone_dir)
        self.sample_rate = self.cosyvoice.sample_rate  # 24 kHz

    @torch.no_grad()
    def synthesize(self, text: str, prompt_wav: str, prompt_text: str, speech_len: int | None = None, **decoding) -> torch.Tensor:
        """Synthesize `text` in the voice of `prompt_wav`.

        Args:
            text: Text to synthesize.
            prompt_wav: Path to the prompt audio.
            prompt_text: Transcript of the prompt audio.
            speech_len: Output length in speech tokens (25 per second). Estimated from the
                prompt when None.
            **decoding: Overrides for `delta_tts.generate.generate` (num_steps, top_p, ...).
        Returns:
            Waveform at `self.sample_rate`. Shape: (1, num_samples)
        """
        frontend = self.cosyvoice.frontend

        # -----------------------------
        # 1. CosyVoice 3 frontend
        # -----------------------------
        model_input = frontend.frontend_zero_shot(
            frontend.text_normalize(text, split=False, text_frontend=True),
            frontend.text_normalize(prompt_text, split=False, text_frontend=True),
            prompt_wav, self.sample_rate, "",
        )
        # Text sequence: [instruction] [prompt transcript] [target text]
        target_text_token = model_input["text"].to(self.device)  # (1, target_text_len)
        prompt_text_token = model_input["prompt_text"][:, :int(model_input["prompt_text_len"])].to(self.device)  # (1, prompt_text_len)
        instruct = torch.tensor([INSTRUCT_TOKENS], dtype=target_text_token.dtype, device=self.device)  # (1, 7)
        text_token = torch.cat([instruct, prompt_text_token, target_text_token], dim=1)  # (1, text_len)
        prompt_speech_token = model_input["llm_prompt_speech_token"].to(self.device)  # (1, prompt_len)

        # -----------------------------
        # 2. Output length
        # -----------------------------
        if speech_len is None:
            # The frontend trims the prompt tokens to match the mel length; the length rule
            # uses the untrimmed token count.
            _, prompt_speech_len = frontend._extract_speech_token(prompt_wav)
            speech_len = estimate_speech_len(prompt_text, text, int(prompt_speech_len))

        # -----------------------------
        # 3. DELTA-TTS: speech tokens by iterative unmasking
        # -----------------------------
        speech_token = generate(self.model, text_token, prompt_speech_token, speech_len, **decoding)  # (1, speech_len)

        # -----------------------------
        # 4. CosyVoice 3 flow decoder and vocoder
        # -----------------------------
        return self._token2wav(speech_token, model_input)

    def _token2wav(self, speech_token: torch.Tensor, model_input: dict) -> torch.Tensor:
        """Decode speech tokens to a waveform with the prompt as acoustic reference.

        Args:
            speech_token: Generated speech tokens. Shape: (1, speech_len)
            model_input: Output of the CosyVoice 3 zero-shot frontend.
        Returns:
            Waveform. Shape: (1, num_samples)
        """
        model = self.cosyvoice.model
        embedding = model_input["flow_embedding"]
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # (1, speaker_dim)
        uid = str(uuid.uuid1())
        model.hift_cache_dict[uid] = None
        wav = model.token2wav(
            token=speech_token.to(self.device, dtype=torch.int32),
            prompt_token=model_input["flow_prompt_speech_token"].to(self.device, dtype=torch.int32),
            prompt_feat=model_input["prompt_speech_feat"].to(self.device),
            embedding=embedding.to(self.device),
            token_offset=0,
            uuid=uid,
            finalize=True,
        )  # (1, num_samples)
        model.hift_cache_dict.pop(uid, None)
        return wav.cpu()
