"""Output length estimation.

A masked-token model needs the number of output tokens before decoding starts. We follow the
length rule of OmniVoice (RuleDurationEstimator in omnivoice/utils/duration.py, Apache-2.0):

    1. Each character gets a weight that reflects how long it takes to say.
    2. The prompt gives the speaking rate: prompt weight per prompt speech token.
    3. The target length is the target weight divided by that rate.
    4. Estimates below 2 s are lengthened with a power curve, since a proportional rule
       under-allocates short utterances.

Only the weights that apply to English text are kept here; letters of other scripts count as
1.0, whereas OmniVoice weights them per script.
"""

import unicodedata

__all__ = ["estimate_speech_len"]

# -----------------------------
# Character weights of OmniVoice, relative to one Latin letter
# -----------------------------
LATIN = 1.0
SPACE = 0.2
PUNCTUATION = 0.5  # punctuation and symbols
DIGIT = 3.5  # a digit is usually read as a whole word
MARK = 0.0  # combining diacritics
DEFAULT = 1.0

# -----------------------------
# Short-utterance correction (OmniVoice defaults, in speech tokens at 25 per second)
# -----------------------------
SHORT_THRESHOLD = 50  # 2 s
SHORT_BOOST = 3.0

# OmniVoice's fallback when no prompt transcript is given.
FALLBACK_PROMPT_TEXT = "Nice to meet you."
FALLBACK_PROMPT_LEN = 25


def char_weight(c: str) -> float:
    """Weight of one character."""
    code = ord(c)
    if 65 <= code <= 90 or 97 <= code <= 122:  # A-Z, a-z
        return LATIN
    if code == 32:
        return SPACE
    category = unicodedata.category(c)
    if category.startswith("M"):
        return MARK
    if category.startswith(("P", "S")):
        return PUNCTUATION
    if category.startswith("Z"):
        return SPACE
    if category.startswith("N"):
        return DIGIT
    return DEFAULT


def estimate_speech_len(prompt_text: str, target_text: str, prompt_speech_len: int) -> int:
    """Estimate the number of speech tokens for `target_text`.

    Args:
        prompt_text: Transcript of the prompt audio.
        target_text: Text to synthesize.
        prompt_speech_len: Number of speech tokens of the prompt audio.
    Returns:
        Estimated length in speech tokens (at least 1).
    """
    if not prompt_text:
        prompt_text, prompt_speech_len = FALLBACK_PROMPT_TEXT, FALLBACK_PROMPT_LEN

    prompt_weight = sum(char_weight(c) for c in prompt_text)
    if prompt_weight == 0 or prompt_speech_len <= 0:
        return 1
    rate = prompt_weight / prompt_speech_len  # weight per speech token
    estimate = sum(char_weight(c) for c in target_text) / rate

    if estimate < SHORT_THRESHOLD:
        estimate = SHORT_THRESHOLD * (estimate / SHORT_THRESHOLD) ** (1 / SHORT_BOOST)
    return max(1, int(estimate))
