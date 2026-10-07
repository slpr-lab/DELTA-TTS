# DELTA-TTS

**Adapting Autoregressive Model into Diffusion Language Model for Text-to-Speech**

DELTA-TTS converts a pretrained autoregressive TTS model into a discrete diffusion language model with LoRA and a convolution module, so that speech tokens are decoded in confidence order instead of left to right.

<!-- TODO: OpenReview badge -> NeurIPS 2026 forum link -->
[![Demo](https://img.shields.io/badge/Demo-Page-blue)](https://slpr-lab.github.io/DELTA-TTS/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.04140-B31B1B)](https://arxiv.org/abs/2607.04140)
[![OpenReview](https://img.shields.io/badge/OpenReview-Coming_Soon-9F9F9F)](#)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/slpr-lab/DELTA-TTS)

## News

- ⭐ **[2026.09.25]** DELTA-TTS is accepted to NeurIPS 2026.
- ⭐ **[2026.05.24]** DELTA-TTS is accepted to the SPIGM Workshop at ICML 2026.

## Installation

```bash
git clone --recursive https://github.com/slpr-lab/DELTA-TTS.git
cd DELTA-TTS
uv sync
```

FFmpeg must be installed on the system, since torchaudio reads and writes audio through it (e.g. `apt install ffmpeg`).

## Checkpoints

| Model | Backbone | Trainable modules | Training data | Download |
|---|---|---|---|---|
| DELTA-TTS | Fun-CosyVoice3-0.5B-2512 | LoRA (r=64) and depth-wise convolution (k=31), 94M | LibriTTS, 585 h | [Hugging Face](https://huggingface.co/slpr-lab/DELTA-TTS) |

```bash
uv run huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir pretrained_models/Fun-CosyVoice3-0.5B-2512
uv run huggingface-cli download slpr-lab/DELTA-TTS --local-dir pretrained_models/DELTA-TTS
```

## Inference

Synthesize one sentence in the voice of a prompt:

```bash
uv run python infer.py \
    --prompt_wav prompt.wav \
    --prompt_text "Transcript of the prompt audio." \
    --text "Text to synthesize." \
    --output output.wav
```

Synthesize a test set given as a Seed-TTS style `meta.lst` (`utt_id|prompt_text|prompt_wav|target_text`):

```bash
uv run python infer.py --meta third_party/F5-TTS/data/seedtts_testset/en/meta.lst --output_dir outputs/seedtts_en
```

The output length is estimated with the length rule of [OmniVoice](https://github.com/k2-fsa/OmniVoice).

## Evaluation

WER and SIM are computed with the [evaluation code of F5-TTS](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/eval), included as a submodule and used without modification.
As F5-TTS expects, place the Seed-TTS test set at `third_party/F5-TTS/data/seedtts_testset/en` and the WavLM checkpoint at `third_party/checkpoints/UniSpeech/wavlm_large_finetune.pth`, then run:

```bash
bash eval.sh outputs/seedtts_en
```

## Acknowledgement

This work builds on [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (backbone, speech tokenizer, flow-matching decoder and vocoder), [F5-TTS](https://github.com/SWivid/F5-TTS) (evaluation code) and [OmniVoice](https://github.com/k2-fsa/OmniVoice) (length estimation rule).
We thank the authors for releasing their work.

## Citation

```bibtex
@inproceedings{moon2026deltatts,
  title     = {{DELTA-TTS}: Adapting Autoregressive Model into Diffusion Language Model for Text-to-Speech},
  author    = {Moon, Junwon and Lee, Yejin and Kim, Seungbeom and Ahn, Hoseong and Park, Sewoong and Kim, Heeseung and Shim, Kyuhong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code is released under the Apache-2.0 license.
The model weights are built on CosyVoice 3 and follow its Apache-2.0 license.
