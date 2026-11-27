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
from typing import TYPE_CHECKING, Sequence

from core.typed_config import FALL_BACK, env_bool, env_float, env_int, env_text

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

# Rates below this are not slow settings, they are unusable ones: at 1e-6 tokens
# per second a 30s chunk gets a budget of zero.
_SMALLEST_USEFUL_RATE = 1e-3


def loop_guard_enabled() -> bool:
    return env_bool("ASR_DECODE_LOOP_GUARD", True)


def tokens_per_second_ceiling() -> float:
    # A rate, so a value at or below zero falls back rather than clamping: a
    # ceiling of ~0 tokens per second truncates every line in the film.
    return env_float(
        "ASR_DECODE_TOKENS_PER_SECOND",
        TOKENS_PER_SECOND_CEILING,
        minimum=_SMALLEST_USEFUL_RATE,
        out_of_range=FALL_BACK,
    )


def loop_budget_fraction() -> float:
    return env_float(
        "ASR_DECODE_LOOP_BUDGET_FRACTION",
        LOOP_BUDGET_FRACTION,
        minimum=_SMALLEST_USEFUL_RATE,
        maximum=1.0,
        out_of_range=FALL_BACK,
    )


def explicit_token_cap() -> int | None:
    """`ASR_MAX_NEW_TOKENS` as a hard ceiling, or None for "follow the audio".

    Kept as an escape hatch for bounding decode cost on a slow card. It can only
    lower the derived budget, and when it bites the pipeline says so - a flat cap
    is the thing that was silently truncating dialogue, so it must not be the
    default.
    """
    raw = env_text("ASR_MAX_NEW_TOKENS", "auto", lower=True)
    if raw in {"auto", "0", "none", "off"}:
        return None
    cap = env_int("ASR_MAX_NEW_TOKENS", 0, minimum=MIN_TOKEN_BUDGET)
    # 0 only happens when the value did not parse, which is recorded as a config
    # problem. "Follow the audio" is the safe reading of an unusable cap.
    return cap or None


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
    min_repeats = env_int("ASR_DECODE_LOOP_MIN_REPEATS", MIN_REPEATS, minimum=1)
    derived_tokens = max(min_repeats, math.ceil(loop_budget_fraction() * budget))
    min_tokens = env_int("ASR_DECODE_LOOP_MIN_TOKENS", derived_tokens, minimum=1)
    derived_ngram = max(1, min_tokens // min_repeats)
    max_ngram = env_int("ASR_DECODE_LOOP_MAX_NGRAM", derived_ngram, minimum=1)
    return max_ngram, min_repeats, min_tokens


def _windows(max_ngram: int, min_repeats: int, min_tokens: int) -> list[tuple[int, int]]:
    """`(ngram, repeats)` pairs to test, smallest window first."""
    pairs = []
    for ngram in range(1, max_ngram + 1):
        repeats = max(min_repeats, -(-min_tokens // ngram))
        pairs.append((ngram, repeats))
    return pairs


def _coerce_row_param(
    value: int | Sequence[int],
    batch: int,
    *,
    name: str,
) -> list[int]:
    if isinstance(value, int):
        return [int(value)] * batch
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        raw = value.tolist()
        if isinstance(raw, int):
            return [int(raw)] * batch
        items = [int(item) for item in raw]
    else:
        items = [int(item) for item in value]
    if len(items) != batch:
        raise ValueError(
            f"{name} length ({len(items)}) must match batch size ({batch})"
        )
    return items


def _detect_repetition_loop_vectorized(
    suffix: "torch.Tensor",
    *,
    row_max_ngrams: list[int],
    min_repeats: int,
    row_min_tokens: list[int],
    cache: dict | None = None,
) -> "torch.Tensor":
    """Vectorized periodicity test across all active n-grams in one batched pass.

    A tail of length ``W = n * R`` consists of ``R`` copies of its final
    ``n``-gram if and only if ``suffix[:, t] == suffix[:, t + n]`` for every
    ``t`` in ``[L - W, L - n)``. Reversing only the needed tail slice maps all
    periods ``n`` onto one ``(B, M, K)`` comparison, cutting ~210 per-step CUDA
    kernel launches down to a handful with zero host syncs.
    """
    import torch

    batch, length = int(suffix.shape[0]), int(suffix.shape[1])
    device = suffix.device
    if batch == 0 or length == 0:
        return torch.zeros(batch, dtype=torch.bool, device=device)

    min_repeats_int = max(1, int(min_repeats))
    uniform = (
        len(set(row_min_tokens)) == 1 and len(set(row_max_ngrams)) == 1
    )

    if uniform:
        mt = max(1, int(row_min_tokens[0]))
        mn = max(0, int(row_max_ngrams[0]))
        if mn <= 0 or length < mt:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        active_n: list[int] = []
        active_req: list[int] = []
        max_window = 0
        for ngram in range(1, mn + 1):
            repeats = max(min_repeats_int, -(-mt // ngram))
            window = ngram * repeats
            if window <= length:
                active_n.append(ngram)
                active_req.append(window - ngram)
                if window > max_window:
                    max_window = window
        if not active_n:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        max_k = max(active_req)
        if max_k == 0:
            return torch.ones(batch, dtype=torch.bool, device=device)

        cache_key = (
            "uniform",
            device,
            tuple(active_n),
            max_window,
            max_k,
            mn,
            mt,
            min_repeats_int,
        )
        cached = cache.get(cache_key) if cache is not None else None
        if cached is None:
            n_t = torch.tensor(active_n, dtype=torch.long, device=device)
            req_t = torch.tensor(active_req, dtype=torch.long, device=device)
            k_row = torch.arange(max_k, dtype=torch.long, device=device)
            shifted_idx = (n_t[:, None] + k_row[None, :]).clamp_max(max_window - 1)
            ignore_mask = (k_row[None, :] >= req_t[:, None]).unsqueeze(0)
            cached = (shifted_idx, ignore_mask)
            if cache is not None:
                cache[cache_key] = cached
        else:
            shifted_idx, ignore_mask = cached

        rev = suffix[:, length - max_window :].flip(dims=(1,))
        lhs = rev[:, :max_k].unsqueeze(1)
        rhs = rev[:, shifted_idx]
        return ((lhs == rhs) | ignore_mask).all(dim=2).any(dim=1)

    # Per-row thresholds: keep only n-grams feasible for at least one row.
    max_mn = max((max(0, v) for v in row_max_ngrams), default=0)
    if max_mn <= 0 or length < min(max(1, v) for v in row_min_tokens):
        return torch.zeros(batch, dtype=torch.bool, device=device)

    active_n = []
    feasible_cols: list[list[bool]] = []
    req_cols: list[list[int]] = []
    max_window = 0
    max_k = 0
    for ngram in range(1, max_mn + 1):
        col_feasible: list[bool] = []
        col_req: list[int] = []
        any_row = False
        for row_mt, row_mn in zip(row_min_tokens, row_max_ngrams):
            if ngram > row_mn:
                col_feasible.append(False)
                col_req.append(0)
                continue
            repeats = max(min_repeats_int, -(-max(1, row_mt) // ngram))
            window = ngram * repeats
            if window <= length:
                any_row = True
                col_feasible.append(True)
                req = window - ngram
                col_req.append(req)
                if window > max_window:
                    max_window = window
                if req > max_k:
                    max_k = req
            else:
                col_feasible.append(False)
                col_req.append(0)
        if any_row:
            active_n.append(ngram)
            feasible_cols.append(col_feasible)
            req_cols.append(col_req)

    if not active_n:
        return torch.zeros(batch, dtype=torch.bool, device=device)

    # Transpose (M, B) -> (B, M)
    feasible_rows = [
        [feasible_cols[col][row] for col in range(len(active_n))]
        for row in range(batch)
    ]
    if max_k == 0:
        return torch.tensor(feasible_rows, dtype=torch.bool, device=device).any(dim=1)

    req_rows = [
        [req_cols[col][row] for col in range(len(active_n))]
        for row in range(batch)
    ]
    per_row_key = (
        "per_row",
        device,
        tuple(active_n),
        max_window,
        max_k,
        tuple(tuple(row) for row in feasible_rows),
        tuple(row_min_tokens),
        tuple(row_max_ngrams),
        min_repeats_int,
    )
    cached_per_row = cache.get(per_row_key) if cache is not None else None
    if cached_per_row is None:
        n_t = torch.tensor(active_n, dtype=torch.long, device=device)
        k_row = torch.arange(max_k, dtype=torch.long, device=device)
        shifted_idx = (n_t[:, None] + k_row[None, :]).clamp_max(max_window - 1)
        feasible_t = torch.tensor(feasible_rows, dtype=torch.bool, device=device)
        req_t = torch.tensor(req_rows, dtype=torch.long, device=device)
        ignore_mask = k_row[None, None, :] >= req_t[:, :, None]
        cached_per_row = (shifted_idx, feasible_t, ignore_mask)
        if cache is not None:
            cache[per_row_key] = cached_per_row
    else:
        shifted_idx, feasible_t, ignore_mask = cached_per_row

    rev = suffix[:, length - max_window :].flip(dims=(1,))
    lhs = rev[:, :max_k].unsqueeze(1)
    rhs = rev[:, shifted_idx]
    period_hit = ((lhs == rhs) | ignore_mask).all(dim=2) & feasible_t
    return period_hit.any(dim=1)


def detect_repetition_loop(
    suffix: "torch.Tensor",
    *,
    max_ngram: int | Sequence[int] | None = None,
    min_repeats: int | None = None,
    min_tokens: int | Sequence[int] | None = None,
) -> "torch.Tensor":
    """Per-sequence: does this end in an n-gram repeated to the threshold?

    `suffix` is `(batch, generated_tokens)`. Compared entirely on-device - a
    `.tolist()` here would force a host sync on every decode step, which is the
    cost this whole module exists to avoid paying twice.

    ``min_tokens`` (and ``max_ngram``) may be either a scalar int shared across
    the batch or a per-row sequence matching ``suffix.shape[0]``.
    """
    batch = int(suffix.shape[0])
    per_row_min_tokens = (
        min_tokens is not None and not isinstance(min_tokens, int)
    )
    per_row_max_ngram = (
        max_ngram is not None and not isinstance(max_ngram, int)
    )

    if per_row_min_tokens or per_row_max_ngram:
        derived = loop_guard_config()
        resolved_min_repeats = (
            derived[1] if min_repeats is None else int(min_repeats)
        )
        row_min_tokens = (
            [derived[2]] * batch
            if min_tokens is None
            else _coerce_row_param(min_tokens, batch, name="min_tokens")
        )
        if max_ngram is None:
            row_max_ngrams = [
                max(1, mt // max(1, resolved_min_repeats))
                for mt in row_min_tokens
            ]
        else:
            row_max_ngrams = _coerce_row_param(
                max_ngram, batch, name="max_ngram"
            )
    else:
        if max_ngram is None or min_repeats is None or min_tokens is None:
            derived = loop_guard_config()
            max_ngram = derived[0] if max_ngram is None else max_ngram
            min_repeats = derived[1] if min_repeats is None else min_repeats
            min_tokens = derived[2] if min_tokens is None else min_tokens
        resolved_min_repeats = int(min_repeats)
        row_min_tokens = [int(min_tokens)] * batch
        row_max_ngrams = [int(max_ngram)] * batch

    return _detect_repetition_loop_vectorized(
        suffix,
        row_max_ngrams=row_max_ngrams,
        min_repeats=resolved_min_repeats,
        row_min_tokens=row_min_tokens,
    )


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
        if budgets:
            per_row_configs = [loop_guard_config(budget) for budget in budgets]
            row_max_ngrams = [cfg[0] for cfg in per_row_configs]
            min_repeats = per_row_configs[0][1]
            row_min_tokens = [cfg[2] for cfg in per_row_configs]
            min_activation_tokens = min(row_min_tokens)
        else:
            scalar_max_ngram, min_repeats, scalar_min_tokens = loop_guard_config(None)
            row_max_ngrams = None
            row_min_tokens = None
            min_activation_tokens = scalar_min_tokens

        class RepetitionLoopCriteria(StoppingCriteria):
            def __init__(self) -> None:
                self._index_cache: dict = {}

            def __call__(self, input_ids, scores, **kwargs):
                import torch

                rows = int(input_ids.shape[0])
                suffix = input_ids[:, prompt_length:]
                # Nothing can reach the bar before this, and skipping the check
                # keeps the first half of every decode free of the guard's
                # tensor work.
                if int(suffix.shape[1]) < min_activation_tokens:
                    return torch.zeros(
                        rows, dtype=torch.bool, device=input_ids.device
                    )
                if row_min_tokens is not None and row_max_ngrams is not None:
                    if rows != len(row_min_tokens):
                        return torch.zeros(
                            rows, dtype=torch.bool, device=input_ids.device
                        )
                    return _detect_repetition_loop_vectorized(
                        suffix,
                        row_max_ngrams=row_max_ngrams,
                        min_repeats=min_repeats,
                        row_min_tokens=row_min_tokens,
                        cache=self._index_cache,
                    )
                return _detect_repetition_loop_vectorized(
                    suffix,
                    row_max_ngrams=[scalar_max_ngram] * rows,
                    min_repeats=min_repeats,
                    row_min_tokens=[scalar_min_tokens] * rows,
                    cache=self._index_cache,
                )

        criteria.append(RepetitionLoopCriteria())

    return StoppingCriteriaList(criteria)
