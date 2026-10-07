"""DELTA-TTS model.

What this file contains
    Constants        token ids and sizes of CosyVoice 3, LoRA and convolution settings
    ConvModule       local-mixing block added after every transformer block
    BidirectionalEncoder
                     runs the Qwen2 backbone with full (non-causal) attention
    DeltaTTS         the whole speech-token model; `logits()` is its only inference entry point

How the model sees one utterance
    [SOS] [text tokens] [TASK] [prompt speech tokens] [MASK ... MASK] [EOS]

    The text tokens are [instruction, prompt transcript, target text]. Every position attends
    to every other position. As in the original autoregressive model, the logit at position i
    predicts the token at position i + 1.

Which weights come from where
    Frozen, from CosyVoice 3     Qwen2 backbone, speech-token embedding, output layer
    Trained, in delta_tts.pt     LoRA adapters, ConvModules, [MASK] embedding
"""

import logging
import os

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import Qwen2ForCausalLM

__all__ = ["DeltaTTS"]

logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
# CosyVoice 3 keeps its special tokens in the speech-embedding table, right after the codebook.
SPEECH_TOKEN_SIZE = 6561  # speech codebook size; ids 0..6560 are speech tokens
SOS_TOKEN = 6561
EOS_TOKEN = 6562
TASK_TOKEN = 6563
NUM_OUTPUT_TOKENS = 6761  # vocab_size: rows of the speech embedding and of the output layer
MASK_TOKEN = 6761  # one past the table; embedded with the learned [MASK] vector
HIDDEN_DIM = 896

LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
}
CONV_KERNEL_SIZE = 31  # 31 tokens = about 1.2 s at 25 tokens per second


class ConvModule(nn.Module):
    """Conformer-style convolution block, added as a residual after each transformer block.

    Speech tokens are strongly correlated with their neighbours. The depth-wise convolution
    mixes this local context so that attention can focus on long-range structure.

    Args:
        hidden_dim: Hidden size.
        kernel_size: Kernel size of the depth-wise convolution (odd).
        dropout: Dropout before the residual connection.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, kernel_size: int = CONV_KERNEL_SIZE, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.pw1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.glu = nn.GLU(dim=-1)
        self.dw = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.dw_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.pw2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix local context into the hidden states.

        Args:
            x: Hidden states. Shape: (batch_size, seq_len, hidden_dim)
        Returns:
            Hidden states with the residual added. Shape: (batch_size, seq_len, hidden_dim)
        """
        residual = x  # (batch_size, seq_len, hidden_dim)
        x = self.norm(x)  # (batch_size, seq_len, hidden_dim)
        x = self.pw1(x)  # (batch_size, seq_len, 2 * hidden_dim)
        x = self.glu(x)  # (batch_size, seq_len, hidden_dim)
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        x = self.dw(x)  # (batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        x = self.dw_norm(x)  # (batch_size, seq_len, hidden_dim)
        x = self.act(x)  # (batch_size, seq_len, hidden_dim)
        x = self.pw2(x)  # (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)  # (batch_size, seq_len, hidden_dim)
        return residual + x  # (batch_size, seq_len, hidden_dim)


class BidirectionalEncoder(nn.Module):
    """Runs the LoRA-wrapped Qwen2 backbone with full bidirectional attention.

    Qwen2 builds a causal mask by default. When it receives a 4D attention mask instead, it uses
    that mask as given, so passing an all-zero mask (0 = may attend) removes the causal
    restriction without changing any Qwen2 code.

    Args:
        qwen2_model: Qwen2ForCausalLM wrapped by PEFT (LoRA).
    """

    def __init__(self, qwen2_model: nn.Module) -> None:
        super().__init__()
        self.qwen2_model = qwen2_model

    def embed_tokens(self, text_token: torch.Tensor) -> torch.Tensor:
        """Embed text tokens with the Qwen2 vocabulary.

        Args:
            text_token: Text token ids. Shape: (batch_size, text_len)
        Returns:
            Text embeddings. Shape: (batch_size, text_len, hidden_dim)
        """
        return self.qwen2_model.get_base_model().model.embed_tokens(text_token)  # (batch_size, text_len, hidden_dim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of embeddings.

        Args:
            xs: Input embeddings. Shape: (batch_size, seq_len, hidden_dim)
        Returns:
            Hidden states of the last layer. Shape: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = xs.shape
        full_attention = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=xs.dtype, device=xs.device)  # (batch_size, 1, seq_len, seq_len)
        out = self.qwen2_model.get_base_model().model(inputs_embeds=xs, attention_mask=full_attention, return_dict=True)
        return out.last_hidden_state  # (batch_size, seq_len, hidden_dim)


class DeltaTTS(nn.Module):
    """DELTA-TTS speech-token model.

    Args:
        backbone_dir: CosyVoice 3 model directory (Fun-CosyVoice3-0.5B-2512), which provides
            `CosyVoice-BlankEN/` (Qwen2 config) and `llm.pt` (LLM weights).
    """

    def __init__(self, backbone_dir: str) -> None:
        super().__init__()

        # 1. CosyVoice 3 LLM: Qwen2 backbone, speech-token embedding and output layer (frozen).
        qwen2 = Qwen2ForCausalLM.from_pretrained(os.path.join(backbone_dir, "CosyVoice-BlankEN"))
        self.speech_embedding = nn.Embedding(NUM_OUTPUT_TOKENS, HIDDEN_DIM)
        self.llm_decoder = nn.Linear(HIDDEN_DIM, NUM_OUTPUT_TOKENS, bias=False)
        _load_cosyvoice_llm(os.path.join(backbone_dir, "llm.pt"), qwen2, self.speech_embedding, self.llm_decoder)

        # 2. LoRA adapters on every attention and feed-forward projection.
        peft_model = get_peft_model(qwen2, LoraConfig(bias="none", task_type=None, **LORA_CONFIG))

        # 3. Bidirectional attention.
        self.bidirectional_encoder = BidirectionalEncoder(peft_model)

        # 4. A ConvModule after every transformer block. A forward hook applies it to the block's
        #    output, so the Qwen2 implementation is used as is.
        layers = peft_model.get_base_model().model.layers
        self.conv_modules = nn.ModuleList([ConvModule() for _ in layers])
        for layer, conv in zip(layers, self.conv_modules):
            layer.register_forward_hook(_apply_after_block(conv))

        # 5. Learned [MASK] embedding (loaded from delta_tts.pt).
        self.mask_embedding = nn.Parameter(torch.zeros(1, HIDDEN_DIM))  # (1, hidden_dim)

        self.requires_grad_(False)  # inference only

    def load_checkpoint(self, path: str) -> None:
        """Load the trained DELTA-TTS weights: LoRA, ConvModules and the [MASK] embedding.

        Args:
            path: Path to `delta_tts.pt`.
        """
        state = torch.load(path, map_location="cpu", weights_only=True)
        _, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise KeyError(f"Unexpected keys in {path}: {unexpected[:5]}")

        trained = {name for name, _ in self.named_parameters()
                   if "lora_" in name or name.startswith(("conv_modules.", "mask_embedding"))}
        missing = trained - set(state)
        if missing:
            raise KeyError(f"Missing keys in {path}: {sorted(missing)[:5]}")
        logger.info("Loaded %d DELTA-TTS tensors from %s", len(state), path)

    @torch.no_grad()
    def logits(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor:
        """Output logits for the sequence [SOS, text, TASK, speech].

        Args:
            text_token: Text token ids. Shape: (1, text_len)
            speech_token: Speech token ids, MASK_TOKEN where masked. Shape: (1, speech_seq_len)
        Returns:
            Logits; the logit at position i predicts the token at position i + 1.
            Shape: (1, seq_len, vocab_size) with seq_len = 1 + text_len + 1 + speech_seq_len
        """
        # Text: Qwen2 token embeddings.
        text_emb = self.bidirectional_encoder.embed_tokens(text_token)  # (1, text_len, hidden_dim)

        # Speech: speech-token embeddings, with the [MASK] vector at masked positions.
        is_mask = speech_token == MASK_TOKEN  # (1, speech_seq_len)
        speech_emb = self.speech_embedding(speech_token.masked_fill(is_mask, 0))  # (1, speech_seq_len, hidden_dim)
        speech_emb[is_mask] = self.mask_embedding.squeeze(0)

        # [SOS] text [TASK] speech
        sos = self.speech_embedding.weight[SOS_TOKEN].view(1, 1, -1)  # (1, 1, hidden_dim)
        task = self.speech_embedding.weight[TASK_TOKEN].view(1, 1, -1)  # (1, 1, hidden_dim)
        sequence = torch.cat([sos, text_emb, task, speech_emb], dim=1)  # (1, seq_len, hidden_dim)

        hidden = self.bidirectional_encoder(sequence)  # (1, seq_len, hidden_dim)
        return self.llm_decoder(hidden)  # (1, seq_len, vocab_size)


def _apply_after_block(conv: ConvModule):
    """Forward hook that replaces a transformer block's hidden-state output with conv(output)."""

    def hook(module: nn.Module, inputs: tuple, output: tuple) -> tuple:
        hidden_states = output[0]  # (batch_size, seq_len, hidden_dim); the rest of the tuple is passed through
        return (conv(hidden_states),) + tuple(output[1:])

    return hook


def _load_cosyvoice_llm(path: str, qwen2: nn.Module, speech_embedding: nn.Module, llm_decoder: nn.Module) -> None:
    """Load the pretrained CosyVoice 3 LLM weights (`llm.pt`) before LoRA is attached.

    Args:
        path: Path to CosyVoice 3 `llm.pt`.
        qwen2: Qwen2ForCausalLM; receives the `llm.model.*` weights.
        speech_embedding: Receives `speech_embedding.*`.
        llm_decoder: Output layer; receives `llm_decoder.*`.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)

    def weights_under(prefix: str) -> dict:
        return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}

    qwen2.load_state_dict(weights_under("llm.model."), strict=False)
    speech_embedding.load_state_dict(weights_under("speech_embedding."))
    llm_decoder.load_state_dict(weights_under("llm_decoder."))
