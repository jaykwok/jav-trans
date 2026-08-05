"""Translation-side settings: env-derived constants and normalizers.

Single source of truth for TRANSLATION_* tuning knobs. translator.py aliases
these into its own namespace during the migration so tests that monkeypatch
`translator.TRANSLATION_*` keep working until the engine reads them directly.
"""

from __future__ import annotations

import os

from core.config import (
    REASONING_EFFORTS,
    load_config,
    normalize_reasoning_effort,
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
LLM_API_FORMAT = os.getenv("LLM_API_FORMAT", "chat").strip().lower() or "chat"
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "medium").strip() or "medium"

TRANSLATION_MAX_TOKENS = 384000
# Arithmetic bound on how long a reply may legitimately get, so a model that
# falls into a repetition loop stops at the bound instead of at
# TRANSLATION_MAX_TOKENS (a ceiling sized for API models, i.e. no bound at all
# locally). Measured 2026-08-04 over 1098 clean translations from three local
# models: output/source character ratio p50 0.69, p95 0.88, p99 0.97, max 1.27.
# Chinese is denser than Japanese kana, so a translation is essentially never
# longer than its source; the ratio below leaves margin over the observed max
# while still being far under a loop. Validated against 20 real 12-line batches:
# none would have been cut, tightest margin 1.76x. Raising it only costs time on
# runaway replies; lowering it below ~1.3 starts truncating real translations.
TRANSLATION_OUTPUT_CHAR_RATIO = _env_float("TRANSLATION_OUTPUT_CHAR_RATIO", 1.5)
TRANSLATION_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.6)
TRANSLATION_TOP_P = 0.9
# Worker-independent request granularity. Env override (restart required);
# clamped to a sane range so a bad value never produces 0-length or huge batches.
TRANSLATION_BATCH_SIZE = _env_int_clamped("TRANSLATION_BATCH_SIZE", 64, 8, 400)
COMPACT_SYSTEM_PROMPT = False
TRANSLATION_API_RETRIES = 4
TRANSLATION_BATCH_REPAIR_RETRIES = 2
# Hard cap on requests a single batch may issue. The repair loop resets its
# retry budget whenever the missing set shrinks, which can otherwise let a
# pathological model (one-at-a-time progress) loop indefinitely. Hitting this
# cap fails the batch via the normal failure path (best-effort partial results
# are kept in batch_results but not persisted), bounding cost.
TRANSLATION_BATCH_MAX_REQUESTS = 12
TRANSLATION_API_BACKOFF_BASE_S = 1.5
TRANSLATION_API_BACKOFF_MAX_S = 20.0
TRANSLATION_PREFIX_WARMUP = True
TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS = 180000
TRANSLATION_REPAIR_MAX_IDS = 12
TRANSLATION_REPAIR_CONTEXT_RADIUS = 1
TRANSLATION_REPAIR_LENGTH_RATIO_MIN = 0.25
TRANSLATION_REPAIR_LENGTH_RATIO_MAX = 4.0


def _normalize_llm_api_format(value: str | None, fallback: str = "chat") -> str:
    normalized = (value or fallback or "chat").strip().lower()
    return normalized if normalized in {"chat", "responses"} else "chat"


def _llm_api_format(api_format: str | None = None) -> str:
    if api_format is not None:
        return _normalize_llm_api_format(api_format, LLM_API_FORMAT)
    value = os.getenv("LLM_API_FORMAT", LLM_API_FORMAT)
    return _normalize_llm_api_format(value)


_REASONING_EFFORTS = set(REASONING_EFFORTS)
_normalize_reasoning_effort = normalize_reasoning_effort
