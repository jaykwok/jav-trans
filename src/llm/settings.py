"""Translation-side settings: env-derived constants and normalizers.

Single source of truth for TRANSLATION_* tuning knobs. translator.py aliases
these into its own namespace during the migration so tests that monkeypatch
`translator.TRANSLATION_*` keep working until the engine reads them directly.
"""

from __future__ import annotations

import os

from core.config import (
    DEFAULT_REASONING_EFFORT,
    escalated_reasoning_effort,
    load_config,
    normalize_reasoning_effort,
    recognized_reasoning_effort,
)

load_config()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int_clamped(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


DEFAULT_TARGET_LANG = "简体中文"

OPENAI_COMPATIBILITY_BASE_URL = (
    os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip() or None
)
API_KEY = os.getenv("API_KEY", "").strip() or None
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "").strip()
LLM_REASONING_EFFORT = (
    os.getenv("LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()
    or DEFAULT_REASONING_EFFORT
)
# Which output constraint the endpoint gets, and how hard to insist on it.
# Empty is the rule in `_structured_output_mode`: ask for a strict `json_schema`
# everywhere except DeepSeek's own API, which has no such thing, and accept
# whatever the endpoint does with it. `json_object` is the escape hatch for a
# relay that proxies a provider without strict structured output, since the
# endpoint is then some private domain nothing can detect. `json_schema` is the
# opposite pin: on OpenRouter it also adds `provider.require_parameters`, so a
# model whose upstreams cannot enforce the schema fails instead of quietly
# answering without one. Env only: it is a property of the deployment, not a
# per-job choice.
LLM_STRUCTURED_OUTPUT = os.getenv("LLM_STRUCTURED_OUTPUT", "").strip().lower()

# Fallback ceiling on `max_tokens`, used only until the endpoint tells us its
# real one (`llm.max_tokens_limits`). Not a target either: every real request is
# sized by `response_token_budget` and only ever `min()`ed against this.
#
# 384000 was not a legal value everywhere: it is exactly what the default
# OpenRouter deployment's deepseek-v4-flash accepts, and it was carried over to
# an endpoint capping the parameter at 131072, which rejects the request
# outright. That broke the prefix warmup - the one call that passes no budget -
# on every run from 2026-09-04, silently costing each film its prompt-cache
# priming. 65536 is accepted by both and still far above what a batch asks for:
# 200 cues at the default reasoning allowance computes to about 46k (32000
# thinking + 28/item structure + 1.5x source chars), so the fallback does not
# bind at defaults. Raising it is what the env override is for; raise it too far
# and the endpoint's refusal teaches the real number.
TRANSLATION_MAX_TOKENS = _env_int_clamped(
    "TRANSLATION_MAX_TOKENS", 65536, 1024, 1_000_000
)
# Arithmetic bound on how long a reply may legitimately get, so a model that
# falls into a repetition loop stops at the bound instead of at
# TRANSLATION_MAX_TOKENS (a ceiling that only has to be legal at the endpoint,
# i.e. no bound at all locally). Measured 2026-08-04 over 1098 clean translations from three local
# models: output/source character ratio p50 0.69, p95 0.88, p99 0.97, max 1.27.
# Chinese is denser than Japanese kana, so a translation is essentially never
# longer than its source; the ratio below leaves margin over the observed max
# while still being far under a loop. Validated against 20 real 12-line batches:
# none would have been cut, tightest margin 1.76x. Raising it only costs time on
# runaway replies; lowering it below ~1.3 starts truncating real translations.
TRANSLATION_OUTPUT_CHAR_RATIO = _env_float("TRANSLATION_OUTPUT_CHAR_RATIO", 1.5)
# One escalation when a reply is cut off at the budget above. Hitting an
# arithmetic bound on a legitimate translation means one of two things, and the
# transport cannot tell them apart: the bound was too tight for this batch, or
# the model is looping. Retrying once at this multiple settles the first; the
# second costs one extra request and then fails anyway. Before this existed the
# failure was terminal, so a single cut reply killed a whole film - sample-b on
# 2026-08-13 died with 1,310 of 1,701 cues already translated and paid for.
TRANSLATION_TRUNCATION_RETRY_FACTOR = _env_float(
    "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0
)
# Room for the thinking the answer is not made of. The ratio above models the
# visible reply, but it is sent as `max_tokens`, which on a reasoning request
# also has to cover that stream. The `none` tier omits this allowance entirely -
# that is most of why it is cheap - and every other tier includes it.
#
# Measured 2026-08-13/14 on deepseek-v4-flash, reasoning characters by batch:
#   low     8 cues/142 src chars -> 7,860   24/396 -> 14,034   54/826 ->  9,383
#   medium  8 cues/142 src chars -> 2,058   24/396 -> 18,393   54/826 -> 20,231
#   max     8 cues/142 src chars -> 6,321   24/396 -> 18,917   54/826 -> 53,388
# Read those rows as low / high / max: `medium` was not a value DeepSeek
# accepted, so that arm ran at the API's default of `high` (see
# core.config.REASONING_EFFORTS). The shipped budgets for those batches were
# 469 / 1,298 / 2,783 tokens, i.e. short by one to two orders of magnitude.
# Reasoning grows with the batch but nothing like proportionally to the source
# (`high` spends about the same on 24 and 54 cues, and `low` is not even
# monotonic), so this is an allowance sized to cover the worst measured case,
# not a second ratio: a character-proportional term would keep starving the
# small batches, which already have the least room. Chars are used as a
# pessimistic stand-in for tokens because the stream reports characters.
#
# One allowance covers both thinking tiers. The `low`/`high` rows above overlap
# and neither is reliably the lighter one, and the tier that was clearly heavier
# (`max`, 53,388 characters on 54 cues) no longer exists.
#
# Over-allowing costs time on a runaway and nothing else: the token budget is
# not the only loop guard - `bounded_schema` caps each translation's length
# independently, and that is the guard that actually catches 嗯嗯嗯….
TRANSLATION_REASONING_TOKEN_ALLOWANCE = _env_int_clamped(
    "TRANSLATION_REASONING_TOKEN_ALLOWANCE", 32000, 0, 200000
)
TRANSLATION_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.6)
TRANSLATION_TOP_P = 0.9
# Cues per request, and since the worker coupling was removed on 2026-08-24 this
# is the operating point rather than a ceiling - the only number deciding how
# many requests a film costs. Env override (restart required); clamped so a bad
# value never produces 0-length batches.
#
# Bigger is cheaper: reasoning is a per-request cost that barely scales with the
# batch (18,393 reasoning chars on 24 cues vs 20,231 on 54), so halving this
# nearly doubles what a film spends thinking. Bigger is also less reliable, and
# not because of the token budget: at 200, four full sample-v runs had 7 of 32
# requests come back with a dropped contiguous tail (9, 50, 100, 100 and 184
# missing) or an out-of-range id, while the output budget was 42,495 tokens
# against a 31,486 worst case. The model stops early, more often the more items
# it holds, and the abandoned request's thinking is charged in full.
#
# 200 is where those two meet on the films measured so far. The per-line quality
# that a smaller batch used to protect is now covered by `bounded_schema`, which
# caps each translation independently of the batch, and by the repair pass.
TRANSLATION_BATCH_SIZE = _env_int_clamped("TRANSLATION_BATCH_SIZE", 200, 8, 400)
COMPACT_SYSTEM_PROMPT = False
TRANSLATION_API_RETRIES = 4
TRANSLATION_BATCH_REPAIR_RETRIES = 2
# Hard cap on requests a single batch may issue. The repair loop resets its
# retry budget whenever the missing set shrinks, which can otherwise let a
# pathological model (one-at-a-time progress) loop indefinitely. Hitting this
# cap fails the batch via the normal failure path (best-effort partial results
# are kept in batch_results but not persisted), bounding cost.
#
# 12 was sized for a loop that always reissued the whole pending set, so one
# request covered one attempt. Halving the span on failure changes that
# arithmetic: a 54-cue batch that descends 54 -> 27 -> 13 spends 1 + 1 failure
# plus ceil(54/13) = 5 covering requests, and `TRANSLATION_API_RETRIES` allows
# four such descents. 12 would abort mid-descent - i.e. exactly when the batch
# was about to succeed at a size the model can handle.
TRANSLATION_BATCH_MAX_REQUESTS = 24
TRANSLATION_API_BACKOFF_BASE_S = 1.5
TRANSLATION_API_BACKOFF_MAX_S = 20.0
TRANSLATION_PREFIX_WARMUP = True
TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS = 180000
# How many flagged cues the repair pass may reissue. This was 12 when the pass
# was one request chasing length-ratio outliers on an already-good translation.
# It is now also the second half of the cost cascade, where the base pass runs
# at a cheaper tier and the detectors are expected to fire: thinking-off left
# 171 of sample-b's 1,700 cues (10.1%) echoing the Japanese source, so a cap of
# 12 would have left 159 of them untranslated. Sized to clear that rate on a
# long film with headroom, and it still bounds cost - the pass groups ids into
# `TRANSLATION_BATCH_SIZE`-sized requests rather than one request per cue.
TRANSLATION_REPAIR_MAX_IDS = _env_int_clamped(
    "TRANSLATION_REPAIR_MAX_IDS", 400, 0, 4000
)
# Widened 2026-09-01 alongside the none-first repair cascade: a none-tier
# repair request has to lean on local context instead of reasoning, so it
# gets a couple more lines of already-translated dialogue on each side.
TRANSLATION_REPAIR_CONTEXT_RADIUS = 2
TRANSLATION_REPAIR_LENGTH_RATIO_MIN = 0.25
TRANSLATION_REPAIR_LENGTH_RATIO_MAX = 4.0
# Which tier the repair pass reissues at. Empty follows the rule in
# `_repair_reasoning_effort` (the base tier, floored at `low`); a tier name pins
# it instead, which is how a job buys back the old always-escalate behaviour
# (`high`) for a film where the base pass is struggling.
TRANSLATION_REPAIR_REASONING_EFFORT = os.getenv(
    "TRANSLATION_REPAIR_REASONING_EFFORT", ""
).strip().lower()


_normalize_reasoning_effort = normalize_reasoning_effort
_escalated_reasoning_effort = escalated_reasoning_effort


def _repair_reasoning_effort(base_effort: str | None) -> str:
    """The base tier, floored at `low` - or the pin, if one is set.

    Escalation is only kept where it is proven necessary. `none` has to escalate:
    thinking-off left 171 of sample-b's 1,700 cues echoing the Japanese source,
    and this pass ends in a gate that fails the job over exactly that. Above
    `none` it was not earning its price. Measured on sample-v with a `low` base:
    repairing at `high` spent 22,585 reasoning tokens in one request - 25% of the
    film's bill, against 11 base requests costing twice that between them -
    while repairing at `low` spent 7,673 and still fixed 146 of 146 flagged
    cues, leaving source echo, residual kana and glossary compliance identical
    (0, 0, 100%). Film price ¥0.886 -> ¥0.752, wall 315s -> 170s.

    What it does cost is 12 length-ratio outliers in the finished file against 6,
    both diagnostics rather than defects, and both inside the run-to-run spread
    of this measurement. A pin of `none` is refused for the reason above.
    """
    pinned = recognized_reasoning_effort(TRANSLATION_REPAIR_REASONING_EFFORT)
    if pinned is not None and pinned != "none":
        return pinned
    base = _normalize_reasoning_effort(base_effort)
    return base if base != "none" else _escalated_reasoning_effort(base)
