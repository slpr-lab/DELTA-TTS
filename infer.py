"""DELTA-TTS inference.

Single sentence:
    python infer.py --prompt_wav prompt.wav --prompt_text "..." --text "..." --output output.wav

Test set in Seed-TTS format (one `utt_id|prompt_text|prompt_wav|target_text` per line;
prompt_wav is relative to the list file). Outputs are written as `<output_dir>/<utt_id>.wav`:
    python infer.py --meta data/seedtts_testset/en/meta.lst --output_dir outputs/seedtts_en
"""

import argparse
import logging
from pathlib import Path

import torchaudio
from tqdm import tqdm

from delta_tts import DeltaTTSPipeline

logger = logging.getLogger("delta_tts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone_dir", default="pretrained_models/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--checkpoint", default="pretrained_models/DELTA-TTS/delta_tts.pt")
    # single sentence
    parser.add_argument("--prompt_wav")
    parser.add_argument("--prompt_text")
    parser.add_argument("--text")
    parser.add_argument("--output", default="output.wav")
    # test set
    parser.add_argument("--meta", help="Seed-TTS style list file")
    parser.add_argument("--output_dir")
    parser.add_argument("--shard", default="0/1", help="i/N: process every N-th line starting at i (one shard per GPU)")
    # decoding
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 takes the most likely token; above 0 samples")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # -----------------------------
    # Load DELTA-TTS and CosyVoice 3
    # -----------------------------
    tts = DeltaTTSPipeline(args.backbone_dir, args.checkpoint, args.device)
    decoding = {"num_steps": args.num_steps, "temperature": args.temperature}

    # -----------------------------
    # Single sentence
    # -----------------------------
    if args.meta is None:
        if not (args.prompt_wav and args.prompt_text and args.text):
            raise SystemExit("Give --prompt_wav, --prompt_text and --text, or --meta.")
        wav = tts.synthesize(args.text, args.prompt_wav, args.prompt_text, **decoding)
        torchaudio.save(args.output, wav, tts.sample_rate)
        logger.info("Saved %s", args.output)
        return

    # -----------------------------
    # Test set
    # -----------------------------
    meta = Path(args.meta)
    rows = [line.strip().split("|") for line in meta.read_text(encoding="utf-8").splitlines() if line.strip()]
    shard, num_shards = (int(x) for x in args.shard.split("/"))
    rows = rows[shard::num_shards]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for utt_id, prompt_text, prompt_wav, target_text in tqdm(rows, desc=f"shard {args.shard}"):
        out_path = out_dir / f"{utt_id}.wav"
        if out_path.exists():
            continue
        wav = tts.synthesize(target_text, str(meta.parent / prompt_wav), prompt_text, **decoding)
        torchaudio.save(str(out_path), wav, tts.sample_rate)
    logger.info("Wrote %d utterances to %s", len(rows), out_dir)


if __name__ == "__main__":
    main()
