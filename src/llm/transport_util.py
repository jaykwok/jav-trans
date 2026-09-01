"""Shared transport utilities for the translation engine and backends.

Cancellation, usage metrics, retry classification/backoff, retry-event
scoping, and progress emission. Nothing here imports llm.translator — this is
the layer both backends and the orchestration engine stand on.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable

from llm import settings
from llm.errors import (
    RetryableTranslationFormatError,
    TranslationCancelledError,
)


# --- cancellation -------------------------------------------------------------


def _cancel_requested(cancel_event: threading.Event | None) -> bool:
    try:
        return bool(cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if _cancel_requested(cancel_event):
        raise TranslationCancelledError("任务已取消")


# --- usage metrics ------------------------------------------------------------


def _get_nested_value(value, *path: str):
    current = value
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _coerce_optional_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _first_present(usage, *paths: tuple[str, ...]):
    """First path that resolves to something, so one reader serves both APIs."""
    for path in paths:
        value = _coerce_optional_int(_get_nested_value(usage, *path))
        if value is not None:
            return value
    return None


def _extract_usage_metrics(usage) -> dict:
    # Both names are still read although only Responses is called: relays serve
    # Responses while reporting usage in the older Chat spelling, and the two
    # surfaces share only `total_tokens`. Reading one spelling once meant a run
    # recorded `total_tokens` and nothing else - every prompt-cache field came
    # back null, which read as "the provider does not cache" when in fact the
    # accounting was simply never parsed. Verified 2026-08-02 against DeepSeek:
    # a repeated prefix reports 3200/3230 cached, and it was invisible here.
    cached_tokens = _first_present(
        usage,
        ("prompt_tokens_details", "cached_tokens"),   # Chat Completions
        ("input_tokens_details", "cached_tokens"),    # Responses
    )
    cache_hit_tokens = _coerce_optional_int(
        _get_nested_value(usage, "prompt_cache_hit_tokens")
    )
    cache_miss_tokens = _coerce_optional_int(
        _get_nested_value(usage, "prompt_cache_miss_tokens")
    )
    # The reported fields are never synthesised: hit/miss is a billing split the
    # provider owns, and a derived number sitting in the same key as a reported
    # one is indistinguishable from it in the timings.
    metrics = {
        "cached_tokens": cached_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
    }
    prompt_tokens = _first_present(usage, ("prompt_tokens",), ("input_tokens",))
    if prompt_tokens is not None:
        metrics["prompt_tokens"] = prompt_tokens
    # `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` are a DeepSeek Chat
    # extension; Responses reports only `cached_tokens`. Making Responses the
    # default on 2026-08-24 therefore left every run's cost uncomputable from
    # its own timings - the two fields the input bill is priced from were both
    # null.
    #
    # Recorded under separate keys rather than filling the reported ones. The
    # split is arithmetic, not an estimate: on the Chat arm of sample-v,
    # `cache_hit_tokens` equalled `cached_tokens` exactly and
    # 1,796,224 + 125,959 = 1,922,183 = `prompt_tokens`.
    if (
        cache_hit_tokens is None
        and cache_miss_tokens is None
        and cached_tokens is not None
        and prompt_tokens is not None
    ):
        metrics["cache_hit_tokens_derived"] = cached_tokens
        metrics["cache_miss_tokens_derived"] = max(0, prompt_tokens - cached_tokens)
    completion_tokens = _first_present(
        usage, ("completion_tokens",), ("output_tokens",)
    )
    if completion_tokens is not None:
        metrics["completion_tokens"] = completion_tokens
    total_tokens = _coerce_optional_int(_get_nested_value(usage, "total_tokens"))
    if total_tokens is not None:
        metrics["total_tokens"] = total_tokens
    # The line the bill actually turns on, and until 2026-08-24 it was the one
    # number never recorded: output is ~91% of a DeepSeek film's cost and most
    # of the output is reasoning, but establishing that took reconstructing it
    # from streamed `reasoning_chars` progress events across interleaved
    # workers. A provider that does not report it leaves this None rather than
    # having it guessed at.
    reasoning_tokens = _first_present(
        usage,
        ("output_tokens_details", "reasoning_tokens"),      # Responses
        ("completion_tokens_details", "reasoning_tokens"),  # OpenAI Chat
    )
    if reasoning_tokens is not None:
        metrics["reasoning_tokens"] = reasoning_tokens
    return metrics


def _emit_usage(on_usage: Callable[[dict], None] | None, usage) -> None:
    if on_usage is None or usage is None:
        return
    metrics = _extract_usage_metrics(usage)
    if not any(value is not None for value in metrics.values()):
        return
    try:
        on_usage(metrics)
    except Exception:
        return


def _merge_usage_metrics(usages: list[dict]) -> dict:
    merged = {
        "cached_tokens": None,
        "cache_hit_tokens": None,
        "cache_miss_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "cache_hit_tokens_derived": None,
        "cache_miss_tokens_derived": None,
    }
    for usage in usages:
        for key in merged:
            value = _coerce_optional_int(usage.get(key))
            if value is None:
                continue
            merged[key] = value if merged[key] is None else merged[key] + value
    return merged


# --- retry events -------------------------------------------------------------

# Worker threads bind their own event list by assigning `_RETRY_CONTEXT.events`
# directly (see llm/engine.py, llm/translator.py); the engine merges the lists
# on the main thread. threading.local means an unbound thread simply records
# nothing.
_RETRY_CONTEXT = threading.local()


def _current_retry_events() -> list[dict] | None:
    events = getattr(_RETRY_CONTEXT, "events", None)
    return events if isinstance(events, list) else None


# --- retry classification / backoff -------------------------------------------


def _is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    name = type(exc).__name__.lower()
    return any(
        marker in name
        for marker in (
            "ratelimit",
            "timeout",
            "connection",
            "serviceunavailable",
            "internalserver",
            "protocol",
        )
    )


def _is_stream_interrupted_error(exc: Exception) -> bool:
    message = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "protocol" in name
        or "incomplete chunked read" in message
        or "peer closed connection" in message
        or "incomplete message body" in message
    )


def _stream_interrupted_format_error(exc: Exception) -> RetryableTranslationFormatError:
    return RetryableTranslationFormatError(
        "LLM stream interrupted before complete JSON content: "
        f"{type(exc).__name__}: {exc}"
    )


def _request_backoff_delay(attempt: int) -> float:
    return min(
        settings.TRANSLATION_API_BACKOFF_MAX_S,
        settings.TRANSLATION_API_BACKOFF_BASE_S * (2**attempt),
    )


def _record_api_retry_event(
    exc: Exception,
    attempt: int,
    delay_s: float,
    *,
    note: str = "",
) -> None:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    event = {
        "attempt": attempt + 1,
        "delay_s": delay_s,
        "status_code": status_code,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
    }
    if note:
        event["note"] = note
    local_events = _current_retry_events()
    if local_events is not None:
        local_events.append(event)


def _interruptible_sleep(
    total_s: float,
    cancel_event: threading.Event | None = None,
) -> None:
    remaining = max(0.0, float(total_s))
    while remaining > 0:
        if _cancel_requested(cancel_event):
            return
        sleep_for = min(0.1, remaining)
        time.sleep(sleep_for)
        remaining -= sleep_for


def _request_backoff_sleep(
    attempt: int, exc: Exception, cancel_event: threading.Event | None = None
) -> None:
    delay = _request_backoff_delay(attempt)
    _record_api_retry_event(exc, attempt, delay)
    _interruptible_sleep(delay, cancel_event)
    _raise_if_cancelled(cancel_event)


# --- progress -----------------------------------------------------------------


def _emit_progress(
    on_progress: Callable[[dict], None] | None,
    payload: dict,
) -> None:
    if on_progress is None:
        return
    try:
        on_progress(payload)
    except Exception:
        pass


def _count_translation_markers(
    *,
    piece: str,
    id_scan_tail: str,
    id_marker: str,
) -> tuple[int, str]:
    scan_text = id_scan_tail + piece
    tail_len = len(id_scan_tail)
    count = sum(
        1
        for match in re.finditer(re.escape(id_marker), scan_text)
        if match.end() > tail_len
    )
    return count, scan_text[-(len(id_marker) - 1) :]


def _emit_stream_content_progress(
    *,
    piece: str,
    state: dict,
    expected_count: int,
    maybe_emit: Callable[[dict], None],
) -> None:
    state["final_content"].append(piece)
    state["content_chars"] += len(piece)
    count, state["id_scan_tail"] = _count_translation_markers(
        piece=piece,
        id_scan_tail=state["id_scan_tail"],
        id_marker=state["id_marker"],
    )
    state["translated_count"] += count
    maybe_emit(
        {
            "phase": "translating",
            "translated": state["translated_count"],
            "expected": expected_count,
            "content_chars": state["content_chars"],
        }
    )
