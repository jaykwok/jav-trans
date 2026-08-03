"""Bound an ASR decode by what the audio can physically contain.

`generate` returns when *every* sequence in the batch is done, so one sequence
that never emits EOS makes the whole batch pay for its budget. Measured on an
RTX 4060 Ti with the 1.7B model, batch 8, 20 s chunks: one chunk decoded
`んじゅるるるるるる…` to the 128-token cap while the median sequence finished at
59, and **53% of all decode steps produced padding**. Per-step cost is flat in
batch size (55.5 ms at batch 1, 58.0 ms at batch 8), so those wasted steps are
the whole loss. Something has to stop such a sequence.

--- Why the budget is derived from the chunk's duration ----------------------

The first version of this module stopped sequences on *repetition*: an n-gram
repeated enough times was called a loop. That test does not generalise, and the
attempt to make it generalise by measuring is what falsified it.

A village-ritual film decoded on 2026-08-03, 283 chunks, tail repetition scored
against whether the model terminated on its own
(`tools/asr/measure_repetition_budget.py`):

    self-terminated (repetition is real audio)   longest tail 65 token, 2.18 tok/s
    stopped by the guard (repetition unknown)    2.42 - 3.05 tok/s, all *lower bounds*

The two populations are adjacent, and the second column is pinned by the bar that
produced it - a 72-token bar divided by ~27 s chunks *is* 2.6 tok/s, so the
apparent separation is the guard measuring itself. Nine chunks of that film were
cut while genuinely chanting; one lost the sentence that followed the chant,
`綾様、お召し上がりください`, and a chunk is decoded once, so the loss is
permanent. No share-of-budget threshold separates a crowd chanting `ありがたやぁ`
from `んふぅ` looping forever, because as text they are the same thing.

What does separate them is arithmetic. Japanese runs at 6-8 mora/s in
conversation and tops out near 10 mora/s in fast speech, and on this checkpoint
one token is one mora - measured over the same 283 chunks, the median chunk has
exactly 1.00 characters per token. So a chunk of `duration_s` seconds cannot
contain more than `duration_s * TOKENS_PER_SECOND_CEILING` tokens of speech, no
matter what is being said. A sequence that emits more is not transcribing; it is
generating. Stopping it there cannot cost content, and the bound is a property of
human articulation and the tokenizer rather than of any film.

The same 283 chunks, decoded with a 384-token budget so nothing was censored,
came in at p50 1.38, p95 3.38, max 4.45 tok/s. The ceiling below leaves 2.2x
headroom over the fastest chunk measured, which is the margin a *dialogue-dense*
film is allowed to eat into.

--- What the repetition guard is now ----------------------------------------

It is a cost optimisation, not a truncation policy. Its bar is a share of the
duration-derived budget, which makes the bar a *rate*: at fraction 0.5 and the
default ceiling, a sequence is stopped once one unit has repeated at 5 tok/s for
the whole chunk. Genuine sustained repetition was measured at 2.18 tok/s, so the
bar sits 2.3x above the fastest real chanting seen - and any sequence it lets
through still stops at the arithmetic bound, which is why erring high here costs
decode steps and nothing else.

`MAX_NGRAM` is derived rather than guessed: a unit too long to repeat
`MIN_REPEATS` times inside the bar cannot be judged a loop on this evidence, and
the honest outcome for it is to run to the arithmetic bound and be reported as
`decode_cap_truncations`.

--- What the flag means now --------------------------------------------------

Because the budget tracks duration, reaching it is no longer ambiguous. Under the
old flat 128 it meant "either a loop or a chunk with more speech than 128 tokens"
- and at 30 s chunks, 128 tokens is 4.27 tok/s, *below* the 4.45 tok/s this film
actually reached, so it was silently amputating real dialogue. Under a
duration-derived budget it can only mean the model generated faster than anyone
can speak. `decode_cap_truncations` is a loop counter.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - import cost is the whole point
    import torch

# Tokens per second of audio that no real speech can exceed. Japanese fast
# speech is ~10 mora/s and this checkpoint emits ~1 token per mora (measured
# median 1.00 chars/token), plus headroom for the punctuation tokens the output
# format adds. See the module docstring for the measured distribution.
TOKENS_PER_SECOND_CEILING = 10.0
# Non-speech tokens every sequence pays regardless of duration: the language tag
# and the structure `parse_output` reads back. Without it a 2 s chunk gets a
# budget too small to say anything at all.
STRUCTURE_TOKENS = 16
# Floor on the budget, so a short tail chunk is never starved.
MIN_TOKEN_BUDGET = 64
# Share of a chunk's own budget that has to be one repeating unit before the
# sequence is called finished. Because the budget is duration-derived this is a
# repetition *rate* ceiling, which is what makes it portable across chunk
# lengths and domains.
LOOP_BUDGET_FRACTION = 0.5
# Floor on consecutive copies, and the only number here that is a judgement
# rather than a measurement: two copies of a phrase is a person repeating
# themselves, and no budget argument should override that.
MIN_REPEATS = 3
# Chunk length to assume when a caller asks for thresholds without saying how
# long the audio is. Mirrors `chunking._FEATURE_CHUNK_S`, the encoder window that
# every chunk is padded up to, so the fallback is the common case rather than an
# arbitrary constant.
DEFAULT_BUDGET_SECONDS = 30.0


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


def _env_float(name: str, default: float, *, upper: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if value <= 0.0 or (upper is not None and value > upper):
        return default
    return value


def tokens_per_second_ceiling() -> float:
    return _env_float("ASR_DECODE_TOKENS_PER_SECOND", TOKENS_PER_SECOND_CEILING)


def loop_budget_fraction() -> float:
    return _env_float("ASR_DECODE_LOOP_BUDGET_FRACTION", LOOP_BUDGET_FRACTION, upper=1.0)


def explicit_token_cap() -> int | None:
    """`ASR_MAX_NEW_TOKENS` as a hard ceiling, or None for "follow the audio".

    Kept as an escape hatch for bounding decode cost on a slow card. It can only
    lower the derived budget, and when it bites the pipeline says so - a flat cap
    is the thing that was silently truncating dialogue, so it must not be the
    default.
    """
    raw = os.getenv("ASR_MAX_NEW_TOKENS", "").strip().lower()
    if raw in {"", "auto", "0", "none", "off"}:
        return None
    try:
        return max(MIN_TOKEN_BUDGET, int(raw))
    except (TypeError, ValueError):
        return None


def plausible_token_budget(duration_s: float) -> int:
    """The most tokens `duration_s` of speech can produce.

    Not a guess at how much this chunk will say - an upper bound on how much any
    chunk of this length *could* say. Passing it to `generate` as
    `max_new_tokens` therefore cannot cut a transcription short.
    """
    try:
        seconds = max(0.0, float(duration_s))
    except (TypeError, ValueError):
        seconds = 0.0
    budget = math.ceil(seconds * tokens_per_second_ceiling()) + STRUCTURE_TOKENS
    budget = max(MIN_TOKEN_BUDGET, budget)
    cap = explicit_token_cap()
    return min(budget, cap) if cap is not None else budget


def loop_guard_config(token_budget: int | None = None) -> tuple[int, int, int]:
    """`(max_ngram, min_repeats, min_tokens)` for a given token budget.

    Derived from the budget, then overridable. The overrides exist for
    measurement runs - the derivation is what production uses, so that a change
    in chunk length or in the rate ceiling moves the bar with it instead of
    leaving a threshold calibrated against a different budget.
    """
    budget = int(token_budget or plausible_token_budget(DEFAULT_BUDGET_SECONDS))
    min_repeats = _env_int("ASR_DECODE_LOOP_MIN_REPEATS", MIN_REPEATS)
    derived_tokens = max(min_repeats, math.ceil(loop_budget_fraction() * budget))
    min_tokens = _env_int("ASR_DECODE_LOOP_MIN_TOKENS", derived_tokens)
    derived_ngram = max(1, min_tokens // min_repeats)
    max_ngram = _env_int("ASR_DECODE_LOOP_MAX_NGRAM", derived_ngram)
    return max_ngram, min_repeats, min_tokens


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
    max_ngram: int | None = None,
    min_repeats: int | None = None,
    min_tokens: int | None = None,
) -> "torch.Tensor":
    """Per-sequence: does this end in an n-gram repeated to the threshold?

    `suffix` is `(batch, generated_tokens)`. Compared entirely on-device - a
    `.tolist()` here would force a host sync on every decode step, which is the
    cost this whole module exists to avoid paying twice.

    Defaults come from `loop_guard_config()`, so a caller that passes nothing
    tests the same thresholds production uses.
    """
    import torch

    if max_ngram is None or min_repeats is None or min_tokens is None:
        derived = loop_guard_config()
        max_ngram = derived[0] if max_ngram is None else max_ngram
        min_repeats = derived[1] if min_repeats is None else min_repeats
        min_tokens = derived[2] if min_tokens is None else min_tokens

    batch, length = suffix.shape
    done = torch.zeros(batch, dtype=torch.bool, device=suffix.device)
    for ngram, repeats in _windows(max_ngram, min_repeats, min_tokens):
        window = ngram * repeats
        if length < window:
            continue
        block = suffix[:, length - window :].reshape(batch, repeats, ngram)
        done |= (block == block[:, -1:, :]).all(dim=2).all(dim=1)
    return done


def build_stopping_criteria(
    prompt_length: int,
    token_budgets: Sequence[int] | None = None,
):
    """Per-row arithmetic bound, plus the repetition guard when it is enabled.

    `token_budgets` is one budget per row, from `plausible_token_budget`. The
    batch's `max_new_tokens` has to be the largest of them, so without a per-row
    stop a 6 s chunk batched with a 30 s one would be free to generate 30 s worth
    of tokens. That per-row stop is content-safe by construction and stays on
    even with the guard switched off; returning None means neither applies.
    """
    budgets = [int(value) for value in (token_budgets or [])]
    guard_on = loop_guard_enabled()
    if not budgets and not guard_on:
        return None

    from transformers import StoppingCriteria, StoppingCriteriaList

    criteria: list[StoppingCriteria] = []

    if budgets:

        class TokenBudgetCriteria(StoppingCriteria):
            """Stop each row once it has emitted more than its audio can hold."""

            def __init__(self) -> None:
                self._budgets = None

            def __call__(self, input_ids, scores, **kwargs):
                import torch

                rows = input_ids.shape[0]
                if rows != len(budgets):
                    # Beam search or `num_return_sequences` would break the
                    # row-to-chunk mapping. Failing open leaves the batch cap in
                    # charge rather than stopping the wrong sequence.
                    return torch.zeros(
                        rows, dtype=torch.bool, device=input_ids.device
                    )
                if self._budgets is None or self._budgets.device != input_ids.device:
                    self._budgets = torch.tensor(
                        budgets, dtype=torch.long, device=input_ids.device
                    )
                emitted = input_ids.shape[1] - prompt_length
                return self._budgets <= emitted

        criteria.append(TokenBudgetCriteria())

    if guard_on:
        max_ngram, min_repeats, min_tokens = loop_guard_config(
            max(budgets) if budgets else None
        )

        class RepetitionLoopCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                import torch

                suffix = input_ids[:, prompt_length:]
                # Nothing can reach the bar before this, and skipping the check
                # keeps the first half of every decode free of the guard's
                # tensor work.
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

        criteria.append(RepetitionLoopCriteria())

    return StoppingCriteriaList(criteria)
