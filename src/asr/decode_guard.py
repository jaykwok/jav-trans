"""Stop a decode that has fallen into a repetition loop.

Measured on an RTX 4060 Ti with the 1.7B model, batch 8, 20 s chunks: one chunk
decoded `んじゅるるるるるる…` and never emitted EOS, so it ran to the 128-token
cap while the median sequence finished at 59. `generate` returns when *every*
sequence is done, so that one chunk held the other seven hostage and **53% of
all decode steps produced padding**. Per-step cost is flat in batch size (55.5
ms at batch 1, 58.0 ms at batch 8), so those wasted steps are the whole loss.

What makes this safe to cut is that nothing is lost by cutting it. The sequence
being stopped is already being truncated - it is running into `max_new_tokens`
by definition - and `postgate` already flags exactly this shape as
`FLAG_RUNAWAY`. The text after the loop starts is the same token repeating; the
only question is how many copies of it get paid for.

Thresholds are set far above anything this domain produces legitimately.
`すー…すー…すー` and `あっ…あっ…あっ` are ordinary here and repeat three times;
the guard needs six consecutive copies and at least ten tokens of pure
repetition before it fires.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cost is the whole point
    import torch

# Longest repeating unit the guard looks for. Beyond four tokens a "loop" is
# more likely to be a genuine refrain than a decoder failure.
MAX_NGRAM = 4
# Consecutive copies of that unit before the sequence is called finished.
MIN_REPEATS = 6
# ...and the copies must together cover at least this many tokens, so a single
# repeating token needs ten of them rather than six.
MIN_REPEATED_TOKENS = 10


def loop_guard_enabled() -> bool:
    return os.getenv("ASR_DECODE_LOOP_GUARD", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def loop_guard_config() -> tuple[int, int, int]:
    return (
        _env_int("ASR_DECODE_LOOP_MAX_NGRAM", MAX_NGRAM),
        _env_int("ASR_DECODE_LOOP_MIN_REPEATS", MIN_REPEATS),
        _env_int("ASR_DECODE_LOOP_MIN_TOKENS", MIN_REPEATED_TOKENS),
    )


def _windows(max_ngram: int, min_repeats: int, min_tokens: int) -> list[tuple[int, int]]:
    """`(ngram, repeats)` pairs to test, smallest window first."""
    pairs = []
    for ngram in range(1, max_ngram + 1):
        repeats = max(min_repeats, -(-min_tokens // ngram))
        pairs.append((ngram, repeats))
    return pairs


def detect_repetition_loop(
    suffix: "torch.Tensor",
    *,
    max_ngram: int = MAX_NGRAM,
    min_repeats: int = MIN_REPEATS,
    min_tokens: int = MIN_REPEATED_TOKENS,
) -> "torch.Tensor":
    """Per-sequence: does this end in an n-gram repeated to the threshold?

    `suffix` is `(batch, generated_tokens)`. Compared entirely on-device - a
    `.tolist()` here would force a host sync on every decode step, which is the
    cost this whole module exists to avoid paying twice.
    """
    import torch

    batch, length = suffix.shape
    done = torch.zeros(batch, dtype=torch.bool, device=suffix.device)
    for ngram, repeats in _windows(max_ngram, min_repeats, min_tokens):
        window = ngram * repeats
        if length < window:
            continue
        block = suffix[:, length - window :].reshape(batch, repeats, ngram)
        done |= (block == block[:, -1:, :]).all(dim=2).all(dim=1)
    return done


def build_stopping_criteria(prompt_length: int):
    """A `StoppingCriteriaList` with the loop guard, or None if switched off.

    Returns None rather than an empty list so the caller can leave
    `stopping_criteria` out of the `generate` call entirely when disabled.
    """
    if not loop_guard_enabled():
        return None

    from transformers import StoppingCriteria, StoppingCriteriaList

    max_ngram, min_repeats, min_tokens = loop_guard_config()

    class RepetitionLoopCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.stopped = 0

        def __call__(self, input_ids, scores, **kwargs):
            import torch

            suffix = input_ids[:, prompt_length:]
            if suffix.shape[1] < min_tokens:
                return torch.zeros(
                    input_ids.shape[0], dtype=torch.bool, device=input_ids.device
                )
            return detect_repetition_loop(
                suffix,
                max_ngram=max_ngram,
                min_repeats=min_repeats,
                min_tokens=min_tokens,
            )

    return StoppingCriteriaList([RepetitionLoopCriteria()])
