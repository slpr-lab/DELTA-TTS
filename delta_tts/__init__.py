"""DELTA-TTS: Adapting Autoregressive Model into Diffusion Language Model for Text-to-Speech."""

from delta_tts.model import DeltaTTS
from delta_tts.duration import estimate_speech_len
from delta_tts.generate import generate
from delta_tts.pipeline import DeltaTTSPipeline

__all__ = ["DeltaTTS", "DeltaTTSPipeline", "estimate_speech_len", "generate"]
