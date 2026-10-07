"""Confidence-ordered iterative unmasking.

What this file contains
    generate          generates the target speech tokens for one utterance
    unmask_schedule   how many positions are committed at each step
    top_p_filter      nucleus filtering of the logits

How generation works
    The target starts as [MASK] tokens after the prompt speech tokens:

        [prompt speech] [MASK MASK ... MASK] [EOS]

    At every step the model predicts all masked positions in parallel and the most confident
    predictions are committed. The time-shifted schedule commits few positions in early steps,
    when little context is available, and more positions in later steps.
"""

import torch
import torch.nn.functional as F

from delta_tts.model import EOS_TOKEN, MASK_TOKEN, SPEECH_TOKEN_SIZE, DeltaTTS

__all__ = ["generate", "unmask_schedule", "top_p_filter"]

MAX_REPEAT = 3


@torch.no_grad()
def generate(
    model: DeltaTTS,
    text_token: torch.Tensor,
    prompt_speech_token: torch.Tensor,
    speech_len: int,
    num_steps: int = 16,
    top_p: float = 0.8,
    t_shift: float = 0.3,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Generate `speech_len` speech tokens by iterative unmasking.

    Args:
        model: DELTA-TTS model in eval mode.
        text_token: Text token ids of [instruction, prompt transcript, target text]. Shape: (1, text_len)
        prompt_speech_token: Speech tokens of the prompt audio. Shape: (1, prompt_len)
        speech_len: Number of speech tokens to generate (25 tokens per second).
        num_steps: Number of unmasking steps.
        top_p: Nucleus filtering applied before computing confidences.
        t_shift: Time shift of the unmasking schedule; below 1 commits fewer positions early.
        temperature: 0 takes the most likely token at each position; above 0 samples.
    Returns:
        Generated speech tokens. Shape: (1, speech_len)
    """
    device = text_token.device
    prompt_len = prompt_speech_token.shape[1]
    speech_seq_len = prompt_len + speech_len + 1  # prompt + target + EOS

    # -----------------------------
    # Initial sequence: [prompt speech] [MASK x speech_len] [EOS]
    # -----------------------------
    speech_tokens = torch.full((1, speech_seq_len), EOS_TOKEN, dtype=torch.long, device=device)  # (1, speech_seq_len)
    speech_tokens[0, :prompt_len] = prompt_speech_token[0]
    target = slice(prompt_len, prompt_len + speech_len)  # positions to generate
    speech_tokens[0, target] = MASK_TOKEN
    is_masked = speech_tokens[0] == MASK_TOKEN  # (speech_seq_len,)

    # The logit at position i predicts position i + 1. In [SOS, text, TASK, prompt, target],
    # the logits for the target therefore start one position before the target.
    logit_start = 1 + text_token.shape[1] + 1 + prompt_len - 1

    for n_commit in unmask_schedule(speech_len, num_steps, t_shift):
        if n_commit == 0 or not is_masked.any():
            continue

        # -----------------------------
        # 1. Predict every masked position in parallel
        # -----------------------------
        logits = model.logits(text_token, speech_tokens)  # (1, seq_len, vocab_size)
        logits = logits[0, logit_start:logit_start + speech_len]  # (speech_len, vocab_size)
        logits[:, SPEECH_TOKEN_SIZE:] = float("-inf")  # only speech tokens can be generated
        if temperature > 0:
            logits = logits / temperature
        if top_p < 1.0:
            logits = top_p_filter(logits, top_p)  # (speech_len, vocab_size)
        probs = F.softmax(logits, dim=-1)  # (speech_len, vocab_size)
        if temperature > 0:
            predicted = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (speech_len,)
        else:
            predicted = logits.argmax(dim=-1)  # (speech_len,)
        confidence = probs.gather(1, predicted.unsqueeze(-1)).squeeze(-1)  # (speech_len,)

        # -----------------------------
        # 2. Pick the most confident masked positions
        # -----------------------------
        confidence[~is_masked[target]] = float("-inf")
        n_commit = min(n_commit, int(is_masked[target].sum()))
        candidates = confidence.topk(n_commit).indices.tolist()  # most confident first

        # -----------------------------
        # 3. Commit, limiting repeats of the same token
        # -----------------------------
        committed, count = [], {}
        for pos in candidates:
            token = int(predicted[pos])
            if _run_length(speech_tokens, is_masked, target, pos, token) >= MAX_REPEAT:
                for alternative in logits[pos].argsort(descending=True)[1:].tolist():
                    if alternative < SPEECH_TOKEN_SIZE and _run_length(speech_tokens, is_masked, target, pos, alternative) < MAX_REPEAT:
                        token = alternative
                        break
                predicted[pos] = token
            if count.get(token, 0) < MAX_REPEAT:
                count[token] = count.get(token, 0) + 1
                committed.append(pos)
        committed = torch.tensor(committed, dtype=torch.long, device=device)  # (n_committed,)

        speech_tokens[0, prompt_len + committed] = predicted[committed]
        is_masked[prompt_len + committed] = False

    # -----------------------------
    # Positions still masked after the last step get their most likely token
    # -----------------------------
    if is_masked[target].any():
        logits = model.logits(text_token, speech_tokens)  # (1, seq_len, vocab_size)
        logits = logits[0, logit_start:logit_start + speech_len]  # (speech_len, vocab_size)
        remaining = is_masked[target].nonzero(as_tuple=True)[0]  # (n_remaining,)
        remaining_logits = logits[remaining]  # (n_remaining, vocab_size)
        remaining_logits[:, SPEECH_TOKEN_SIZE:] = float("-inf")
        speech_tokens[0, prompt_len + remaining] = remaining_logits.argmax(dim=-1)

    return speech_tokens[:, target]  # (1, speech_len)


def unmask_schedule(total: int, num_steps: int, t_shift: float = 1.0) -> list[int]:
    """Number of positions to commit at each step.

    After a fraction s of the steps, a fraction f(s) = t_shift * s / (1 + (t_shift - 1) * s) of the
    positions is committed. t_shift = 1 gives a linear schedule.

    Args:
        total: Number of masked positions.
        num_steps: Number of steps.
        t_shift: Time-shift parameter.
    Returns:
        Positions per step, summing to `total`. Length: num_steps
    """
    if num_steps >= total:
        return [1] * total + [0] * (num_steps - total)

    def committed_fraction(s: float) -> float:
        if t_shift == 1.0:
            return s
        denom = 1 + (t_shift - 1) * s
        return t_shift * s / denom if denom > 0 else s

    schedule, assigned = [], 0
    for step in range(num_steps):
        n = round(committed_fraction((step + 1) / num_steps) * total) - assigned
        n = max(0, min(n, total - assigned))
        schedule.append(n)
        assigned += n
    schedule[-1] += total - assigned
    return schedule


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Set logits outside the top-p nucleus to -inf.

    Args:
        logits: Logits. Shape: (num_positions, vocab_size)
        top_p: Cumulative probability kept.
    Returns:
        Filtered logits. Shape: (num_positions, vocab_size)
    """
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)  # (num_positions, vocab_size)
    remove = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1) > top_p  # (num_positions, vocab_size)
    remove[:, 1:] = remove[:, :-1].clone()  # keep the token that crosses top_p
    remove[:, 0] = False
    sorted_logits[remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)  # (num_positions, vocab_size)


def _run_length(speech_tokens: torch.Tensor, is_masked: torch.Tensor, target: slice, pos: int, token: int) -> int:
    """Length of the run of `token` that placing it at target position `pos` would create.

    Only committed neighbours inside the target are counted.

    Args:
        speech_tokens: Current speech tokens. Shape: (1, speech_seq_len)
        is_masked: Masked positions. Shape: (speech_seq_len,)
        target: Positions of the target in `speech_tokens`.
        pos: Position inside the target.
        token: Candidate token id.
    """
    run = 1
    for direction in (-1, 1):
        p = target.start + pos + direction
        while target.start <= p < target.stop and not is_masked[p] and int(speech_tokens[0, p]) == token:
            run += 1
            p += direction
    return run
