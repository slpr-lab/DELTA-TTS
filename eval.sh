#!/usr/bin/env bash
# Evaluate generated speech on Seed-TTS test-en with the evaluation code of F5-TTS,
# used without modification (third_party/F5-TTS).
#
#   bash eval.sh <gen_wav_dir> [num_gpus]
#
# <gen_wav_dir> holds <utt_id>.wav files, e.g. the output of `infer.py --meta`.
# As in F5-TTS, the following are expected:
#   third_party/F5-TTS/data/seedtts_testset/en/                  Seed-TTS test set
#   third_party/checkpoints/UniSpeech/wavlm_large_finetune.pth   WavLM-large speaker verification model
# The F5-TTS evaluation runs in its own environment, created on the first call.
# A few dependency versions are overridden so that the environment installs and runs; the
# evaluation code itself is unchanged:
#   faster-whisper 1.2.1 instead of 0.10.1, whose PyAV 10 has no Linux wheels,
#   torch 2.8, because the s3prl WavLM loader used for SIM calls torchaudio.set_audio_backend,
#     which newer torchaudio no longer has, and
#   pyarrow 20, because the datasets release F5-TTS resolves to fails with newer pyarrow.
set -euo pipefail

GEN_DIR=$(realpath "$1")
NUM_GPUS=${2:-1}
F5_DIR=$(cd "$(dirname "$0")" && pwd)/third_party/F5-TTS

if [ ! -x "$F5_DIR/.venv/bin/python" ]; then
    uv venv --python 3.11 "$F5_DIR/.venv"
    printf "faster-whisper==1.2.1\ntorch==2.8.0\ntorchaudio==2.8.0\ntorchcodec==0.7.0\npyarrow==20.0.0\n" > "$F5_DIR/.venv/overrides.txt"
    uv pip install --python "$F5_DIR/.venv/bin/python" --override "$F5_DIR/.venv/overrides.txt" -e "$F5_DIR[eval]"
fi

# F5-TTS resolves data/ and ../checkpoints/ relative to its own directory.
cd "$F5_DIR"
for task in wer sim; do
    "$F5_DIR/.venv/bin/python" src/f5_tts/eval/eval_seedtts_testset.py -e "$task" -l en -g "$GEN_DIR" -n "$NUM_GPUS"
done
