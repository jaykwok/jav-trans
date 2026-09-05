"""OpenAI-compatible backend: the canonical Responses streaming transport.

This module owns the OpenAI wire protocol end to end (client cache, request
construction, streaming loops, retry-with-backoff, usage/progress emission).
Nothing here imports llm.translator — the orchestration layer calls into the
backend, never the other way around.

Responses is the only surface, since 2026-08-24. Chat Completions was the
second one and it paid for itself twice over in branching: it has no `none`
effort, so switching thinking off was a separate field with a different
spelling per provider (`thinking.type` for DeepSeek, `reasoning.enabled` for
OpenRouter), and it reports no `reasoning_tokens` at all - the number that is
~85% of this pipeline's bill. Responses spells the whole axis as
`reasoning.effort`, `none` included, for every provider.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable
from urllib.parse import urlsplit

from openai import OpenAI

from core.stage_errors import MISSING_MODEL
from llm import settings as llm_settings
from llm import transport_util
from llm.backends.base import BaseTranslationBackend
from llm.errors import (
    ContentPolicyRefusalError,
    MaxTokensRejectedError,
    ResponseTruncatedError,
    RetryableTranslationFormatError,
)
from llm.preflight import require_translation_config

_raise_if_cancelled = transport_util._raise_if_cancelled
_emit_progress = transport_util._emit_progress
_emit_usage = transport_util._emit_usage
_get_nested_value = transport_util._get_nested_value
_is_retryable_api_error = transport_util._is_retryable_api_error
_stream_interrupted_format_error = transport_util._stream_interrupted_format_error
_emit_stream_content_progress = transport_util._emit_stream_content_progress

_JSON_OUTPUT_LABEL = "LLM JSON"

# Default structured-output schema when the caller does not pass one. The
# orchestration layer normally supplies the profile's schema explicitly.
_DEFAULT_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        },
    },
    "required": ["translations"],
}

_CLIENT: OpenAI | None = None
_CLIENT_KEY: tuple[str, str] = ("", "")
_CLIENT_LOCK = threading.Lock()

# The two endpoints this module knows anything about beyond "OpenAI-compatible".
# Both are host comparisons, and both exist for a documented divergence rather
# than for a vendor: DeepSeek's API has no `json_schema`, and OpenRouter is a
# router whose upstream endpoint has to be constrained per request.
_OFFICIAL_DEEPSEEK_HOST = "api.deepseek.com"
_OPENROUTER_HOST = "openrouter.ai"


def _base_url_host(base_url: str | None) -> str:
    """The hostname of a base URL, lowercased, or "" if there is none.

    Provider detection compares this, never a string prefix: with `startswith`
    an attacker-controlled `https://api.deepseek.com.example/v1` reads as the
    official DeepSeek endpoint, and so does any path that merely begins with
    one we know.
    """
    raw = (base_url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"//{raw}"
    return (urlsplit(raw).hostname or "").lower()


def _normalize_openai_compat_base_url(base_url: str | None) -> str | None:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return None
    lower = normalized.lower()
    for suffix in ("/chat/completions", "/responses", "/completions", "/models"):
        if lower.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip("/")
            lower = normalized.lower()
            break
    if lower.endswith("/v1") or "/v1/" in lower:
        return normalized
    # DeepSeek's own OpenAI-compatible base URL carries no version path -- it is
    # `https://api.deepseek.com`, and the other documented one is
    # `https://api.deepseek.com/beta`, which appending `/v1` would turn into a
    # path that exists nowhere.
    if _base_url_host(normalized) == _OFFICIAL_DEEPSEEK_HOST:
        return normalized
    return f"{normalized}/v1"


def _is_official_deepseek_base_url(base_url: str | None) -> bool:
    return _base_url_host(base_url) == _OFFICIAL_DEEPSEEK_HOST


def _is_openrouter_base_url(base_url: str | None) -> bool:
    return _base_url_host(base_url) == _OPENROUTER_HOST


def _structured_output_mode(base_url: str | None) -> str:
    """`json_schema` or `json_object`, decided by endpoint rather than model.

    The model name says nothing about which output constraint the endpoint in
    front of it accepts: `deepseek/deepseek-v4-flash` on OpenRouter takes a
    strict schema, the same weights on DeepSeek's own API do not. Only the
    official DeepSeek host is special-cased; everything else gets the strict
    schema, and a relay that cannot take one sets `LLM_STRUCTURED_OUTPUT`.
    """
    pinned = llm_settings.LLM_STRUCTURED_OUTPUT
    if pinned in ("json_schema", "json_object"):
        return pinned
    return "json_object" if _is_official_deepseek_base_url(base_url) else "json_schema"


def _get_client() -> OpenAI:
    global _CLIENT, _CLIENT_KEY
    # Ask first, so a forgotten API key reads as the setting it is instead of
    # the SDK's "The api_key client option must be set ..." from __init__.
    require_translation_config("openai")
    current_key = os.getenv("API_KEY", "").strip() or None
    current_url = _normalize_openai_compat_base_url(
        os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip()
    )
    key_tuple = (current_key or "", current_url or "")
    with _CLIENT_LOCK:
        if _CLIENT is None or key_tuple != _CLIENT_KEY:
            _CLIENT = OpenAI(api_key=current_key, base_url=current_url)
            _CLIENT_KEY = key_tuple
    return _CLIENT


def _responses_text_format(
    base_url: str | None = None,
    *,
    schema: dict | None = None,
    schema_name: str = "subtitle_translations",
) -> dict:
    if _structured_output_mode(base_url) == "json_object":
        return {"format": {"type": "json_object"}}
    return {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": schema or _DEFAULT_OUTPUT_SCHEMA,
        }
    }


def _openrouter_provider_fields(base_url: str | None) -> dict:
    """Refuse upstream endpoints that would silently drop the strict schema.

    OpenRouter picks a provider per request and structured-output support is a
    property of that endpoint rather than of the model, so without
    `require_parameters` it may route to one that ignores `response_format`.

    Off unless `LLM_STRUCTURED_OUTPUT=json_schema` asks for it, because the
    measurement says the default cannot afford it: the filter reads the
    `structured_outputs` capability, not whether the parameter was sent, and a
    model that only declares `response_format` (`stealth/ox-alpha`, 2026-08-24)
    gets 404 `No endpoints found that can handle the requested parameters` -
    while the same model, asked for the same schema without this constraint,
    translated sample-c's 609 cues with zero missing ids. Being unenforced is
    not the same as being unchecked either: the batch parser validates the ids
    it asked for and reissues what is missing. So this is the switch for a
    caller who wants the contract enforced by the provider rather than obeyed
    by a cooperative model, and `stage_errors` maps that 404 to both remedies.
    """
    if not _is_openrouter_base_url(base_url):
        return {}
    if llm_settings.LLM_STRUCTURED_OUTPUT != "json_schema":
        return {}
    return {"provider": {"require_parameters": True}}


def _merge_extra_body(request: dict, extra: dict) -> None:
    """Fold vendor extensions into one `extra_body`, since several add to it."""
    if not extra:
        return
    merged = dict(request.get("extra_body") or {})
    merged.update(extra)
    request["extra_body"] = merged


def _backoff_sleep(attempt: int, exc: Exception, cancel_event=None) -> None:
    transport_util._request_backoff_sleep(attempt, exc, cancel_event=cancel_event)


def _create_response(
    request: dict,
    cancel_event: threading.Event | None = None,
):
    last_error: Exception | None = None

    for attempt in range(llm_settings.TRANSLATION_API_RETRIES):
        _raise_if_cancelled(cancel_event)
        try:
            return _get_client().responses.create(**request)
        except Exception as exc:
            last_error = exc
            if not _is_retryable_api_error(exc):
                raise

            if attempt < llm_settings.TRANSLATION_API_RETRIES - 1:
                _backoff_sleep(attempt, exc, cancel_event=cancel_event)

    if last_error is not None:
        raise last_error
    raise RuntimeError("response creation failed without an exception")


def _call_create_response(
    request: dict,
    cancel_event: threading.Event | None = None,
):
    if cancel_event is None:
        return _create_response(request)
    try:
        return _create_response(request, cancel_event=cancel_event)
    except TypeError as exc:
        if "cancel_event" not in str(exc):
            raise
        return _create_response(request)


def _build_responses_input(
    messages: list[dict],
) -> list[dict]:
    response_input: list[dict] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        response_input.append(
            {
                "role": role,
                "content": [
                    {
                        "type": "input_text",
                        "text": str(content),
                    }
                ],
            }
        )
    return response_input


def _extract_response_output_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "".join(parts)


def _response_event_type(event) -> str:
    value = getattr(event, "type", "")
    if value:
        return str(value)
    if isinstance(event, dict):
        return str(event.get("type", ""))
    return ""


def _response_event_delta(event) -> str:
    value = getattr(event, "delta", None)
    if value is None and isinstance(event, dict):
        value = event.get("delta")
    return value if isinstance(value, str) else ""


def _response_event_response(event):
    value = getattr(event, "response", None)
    if value is None and isinstance(event, dict):
        value = event.get("response")
    return value


def _response_error(response) -> tuple[str, str]:
    """`(code, message)` off a failed Responses API event; empty when absent."""
    error = getattr(response, "error", None)
    if error is None and isinstance(response, dict):
        error = response.get("error")
    if error is None:
        return "", ""
    if isinstance(error, dict):
        code, message = error.get("code"), error.get("message")
    else:
        code, message = getattr(error, "code", None), getattr(error, "message", None)
    return str(code or ""), str(message or "")


# Codes that mean "we read what you sent and refused it", as opposed to "we
# mishandled it". Only what has actually been observed on the wire belongs
# here - a retryable failure misfiled as terminal kills a film that would have
# recovered. Zhipu/GLM answers `cyber_policy`; add others as they show up.
_CONTENT_POLICY_ERROR_CODES = frozenset({"cyber_policy"})

# The three spellings of the same parameter across OpenAI-compatible endpoints.
_MAX_TOKENS_PARAM_NAMES = ("max_tokens", "max_output_tokens", "max_completion_tokens")

# An inclusive range, as in `限制数值范围[1,131072]` or `(1, 131072)`. The upper
# bound is the second number.
_RANGE_RE = re.compile(r"[\[(]\s*\d+\s*[,，]\s*(\d+)\s*[\])]")
# A comparison or a named bound. Only a short run of non-digits may sit between
# the phrase and its number, so the number belongs to the phrase rather than to
# whatever the message says next.
# Only phrases that can *only* introduce an upper bound. `must be ...` is
# deliberately absent: it introduces `must be <= 131072` and `must be at least
# 100` alike, and reading the second as a ceiling would teach the endpoint's
# floor as its cap.
_UPPER_BOUND_RE = re.compile(
    r"(?:<=?|≤|不得超过|不能超过|不超过|至多|最多|最大(?:值|为)?|上限(?:为|是)?|"
    r"less than or equal to|no more than|at most|maximum(?:\s+of)?|max(?:imum)?"
    r"(?:\s+value)?|limit(?:ed)?(?:\s+to|\s+of)?)"
    r"\D{0,4}(\d+)",
    re.IGNORECASE,
)


# A floor, not a ceiling. Same shape as `_UPPER_BOUND_RE` and read for the
# opposite reason: to recognise the message and *stop*, because the retry ladder
# only ever goes down and down is the wrong direction for these.
_LOWER_BOUND_RE = re.compile(
    r"(?:>=?|≥|at least|not less than|no less than|greater than|"
    r"minimum(?:\s+of)?|min(?:imum)?(?:\s+value)?|"
    r"不少于|不小于|不能小于|至少|最小(?:值|为)?|下限(?:为|是)?)"
    r"\D{0,4}(\d+)",
    re.IGNORECASE,
)
# The parameter is not a parameter here. No number helps, so neither does the
# ladder.
_UNSUPPORTED_PARAM_RE = re.compile(
    r"(?:not supported|unsupported parameter|unknown parameter|unrecognized|"
    r"not permitted|不支持)",
    re.IGNORECASE,
)
# The number in the message bounds input *plus* output. A smaller budget still
# helps this request, so the retry is worth making - but the number is a
# property of this batch's prompt, not of the endpoint, and writing it down as
# an output ceiling under-books every shorter batch for the next 30 days.
_COMBINED_BUDGET_RE = re.compile(
    r"(?:input|prompt|context|combined|together with|sum of|plus|total)"
    r"|输入|上下文|提示词|总和",
    re.IGNORECASE,
)
# "The number you sent is too big", without a number attached. Enough to retry
# on, and - unlike a message that merely names the parameter - enough to write
# down, because it does claim the magnitude was the problem.
_TOO_LARGE_RE = re.compile(
    r"(?:out of range|too (?:large|big|high|long)|exceed(?:s|ed|ing)?|"
    r"over the limit|beyond the limit)"
    r"|超出|超过|过大|范围",
    re.IGNORECASE,
)


def _mentions_max_tokens(message: str) -> bool:
    """Whether the message is about the `max_tokens` parameter at all.

    Keyed on the parameter name rather than on an error code, because the codes
    are not portable (`invalid_request` here, `invalid_request_error` and bare
    `400` elsewhere) while the parameter name is the parameter name.
    """
    lowered = message.lower()
    return any(name in lowered for name in _MAX_TOKENS_PARAM_NAMES)


def _max_tokens_refusal_kind(message: str) -> str:
    """Which of the refusals that name this parameter it actually is.

    Four answers. `"output"` is the endpoint's own output cap and the only one
    worth remembering; `"request"` is a limit on input plus output, real but
    sized by this prompt; `"unclear"` names the parameter without ever claiming
    the number was too big; `""` is the ones no smaller number can satisfy.
    Naming the parameter says it is involved, nothing more, and each of the
    other three costs something specific when it is treated as `"output"`:

    * `max_tokens must be at least 100000` is a floor. The ladder halves away
      from the only value that could have worked, three times, and then writes
      `rejected_at` for numbers the endpoint never refused.
    * `max_output_tokens is not supported for this model` is answered the same
      way by every number, so the three retries are pure round trips.
    * `input tokens plus max_output_tokens must be <= 131072` names a real
      number, but it is this prompt's ceiling, not the endpoint's. Learned as
      `exact_ceiling` it would clamp a 200-cue batch by what a 1500-cue one
      could not have.
    * `max_tokens is invalid` claims nothing at all. Halving is still the only
      move available, but the 30-day `rejected_at` it would leave behind is an
      inference from a message that never said the value was too large.

    Which is the asymmetry the default turns on. Not learning costs one refused
    round trip on the next run, and that request generates nothing; learning a
    wrong bracket clamps every budget to a midpoint for a month, and a clamped
    budget is paid for in generated tokens the moment a reply is cut off. So
    persistence needs positive evidence of magnitude - a bound, a range, or a
    phrase that says too large - and everything else is retried in memory.

    Upper-bound evidence is read first: a message can say both `unsupported
    value` and `must be <= 131072`, and the second is the one that is actionable.
    """
    if _COMBINED_BUDGET_RE.search(message):
        return "request"
    if _RANGE_RE.search(message) or _UPPER_BOUND_RE.search(message):
        return "output"
    if _UNSUPPORTED_PARAM_RE.search(message):
        return ""
    if _LOWER_BOUND_RE.search(message):
        return ""
    if _TOO_LARGE_RE.search(message):
        return "output"
    return "unclear"


def _accepted_max_tokens(message: str, sent: int) -> int | None:
    """The ceiling the endpoint named, read out of its refusal - or None.

    Provider-agnostic but not credulous. An earlier version took the largest
    integer below what was sent, which reads `131072` correctly out of
    `限制数值范围[1,131072]` and just as happily reads `12345` out of
    `max_tokens invalid; request id req_12345`. So a number only counts when it
    sits inside a range or immediately after a bound phrase, and anything else
    returns None - the caller then halves, which is slower but never learns a
    request id as a token ceiling.
    """
    candidates = [
        int(match)
        for pattern in (_RANGE_RE, _UPPER_BOUND_RE)
        for match in pattern.findall(message)
    ]
    usable = [value for value in candidates if 0 < value < sent]
    return max(usable) if usable else None


def _provider_error_reason(code: str, message: str, sent: int) -> Exception | None:
    """The terminal/learnable meaning of a provider error, if it has one.

    One classifier for both shapes a refusal arrives in - a `response.failed`
    event mid-stream, and an exception raised while the stream is still being
    opened. Splitting them is how an endpoint that answers 400 before the first
    SSE frame used to bypass every rule below.
    """
    if code in _CONTENT_POLICY_ERROR_CODES:
        return ContentPolicyRefusalError(
            "OpenAI Responses API refused this request on content policy "
            f"(code={code}): {message}"
        )
    if sent > 0 and _mentions_max_tokens(message):
        kind = _max_tokens_refusal_kind(message)
        if kind == "output":
            return MaxTokensRejectedError(
                f"OpenAI Responses API rejected max_tokens={sent} "
                f"(code={code}): {message}",
                sent=sent,
                limit=_accepted_max_tokens(message, sent),
            )
        if kind == "request":
            # Retried, never recorded: the ladder can find a budget that fits
            # this prompt, but the number that bound is this prompt's, and the
            # cache is read by every other batch of the film.
            return MaxTokensRejectedError(
                f"OpenAI Responses API rejected max_tokens={sent} against a "
                f"combined input+output limit (code={code}): {message}",
                sent=sent,
                learnable=False,
            )
        if kind == "unclear":
            # Same treatment for the same reason: bisect now, remember nothing.
            # The words are carried along so the retry log keeps them - this
            # classifier is built out of a handful of observed messages, and the
            # ones it cannot place are exactly the ones worth collecting.
            return MaxTokensRejectedError(
                f"OpenAI Responses API rejected max_tokens={sent} without "
                f"saying the value was too large (code={code}): {message}",
                sent=sent,
                learnable=False,
            )
    # Floors, unsupported parameters and everything unrecognised fall through to
    # the generic retry rather than into the ladder. Reissuing is not useful for
    # any of them either, but it is at least not *wrong*, and it leaves nothing
    # behind in the capability cache.
    return None


def _exception_error_fields(exc: BaseException) -> tuple[str, str]:
    """`(code, message)` off an SDK exception, best effort.

    The parsed body wins over the exception's own attributes, which is the whole
    point of looking at it: `exc.message` is often the SDK's own summary
    (`Error code: 400`) while `body["error"]` holds what the provider actually
    said - and the rules above read the provider's words.
    """
    code = ""
    message = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "")
            message = str(error.get("message") or "")
    code = code or str(getattr(exc, "code", "") or "")
    message = message or str(getattr(exc, "message", "") or "")
    return code, message or str(exc)


def _response_incomplete_reason(response) -> str:
    details = getattr(response, "incomplete_details", None)
    if isinstance(details, dict):
        return str(details.get("reason", ""))
    return str(getattr(details, "reason", "") or "")


def _chat_responses(
    messages: list[dict],
    expected_count: int = 0,
    on_progress: Callable[[dict], None] | None = None,
    reasoning_effort: str | None = None,
    on_usage: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    response_schema: dict | None = None,
    response_schema_name: str = "subtitle_translations",
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> str:
    _raise_if_cancelled(cancel_event)
    model_name = os.getenv("LLM_MODEL_NAME", llm_settings.LLM_MODEL_NAME).strip()
    if not model_name:
        raise RuntimeError(MISSING_MODEL)

    effective_reasoning_effort = llm_settings._normalize_reasoning_effort(
        reasoning_effort
        or os.getenv("LLM_REASONING_EFFORT", llm_settings.LLM_REASONING_EFFORT)
    )
    effective_temperature = (
        llm_settings.TRANSLATION_TEMPERATURE if temperature is None else temperature
    )
    effective_top_p = llm_settings.TRANSLATION_TOP_P if top_p is None else top_p
    effective_max_tokens = (
        llm_settings.TRANSLATION_MAX_TOKENS if max_tokens is None else max_tokens
    )
    base_url = os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "")
    request = {
        "model": model_name,
        "input": _build_responses_input(messages),
        "stream": True,
        "reasoning": {"effort": effective_reasoning_effort},
        "text": _responses_text_format(
            base_url,
            schema=response_schema,
            schema_name=response_schema_name,
        ),
        "temperature": effective_temperature,
        "top_p": effective_top_p,
    }
    _merge_extra_body(request, _openrouter_provider_fields(base_url))
    if effective_max_tokens > 0:
        request["max_output_tokens"] = effective_max_tokens

    try:
        response_stream = _call_create_response(request, cancel_event=cancel_event)
    except Exception as exc:
        # An endpoint that validates before opening the stream answers 400 here
        # instead of with a `response.failed` frame, and the same refusal has to
        # mean the same thing on both paths.
        reason = _provider_error_reason(
            *_exception_error_fields(exc), effective_max_tokens
        )
        if reason is not None:
            raise reason from exc
        raise

    completed_response = None
    incomplete_response = None
    failed_error = None
    reasoning_chars = 0
    last_emit = 0.0
    debounce_s = 0.25
    stream_state = {
        "final_content": [],
        "content_chars": 0,
        "translated_count": 0,
        "id_scan_tail": "",
        "id_marker": '"id":',
    }

    def maybe_emit(payload: dict, *, force: bool = False) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if not force and now - last_emit < debounce_s:
            return
        last_emit = now
        _emit_progress(on_progress, payload)

    try:
        for event in response_stream:
            _raise_if_cancelled(cancel_event)
            event_type = _response_event_type(event)
            if event_type == "response.output_text.delta":
                piece = _response_event_delta(event)
                if piece:
                    _emit_stream_content_progress(
                        piece=piece,
                        state=stream_state,
                        expected_count=expected_count,
                        maybe_emit=maybe_emit,
                    )
                continue

            if event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            }:
                reasoning_piece = _response_event_delta(event)
                if reasoning_piece:
                    reasoning_chars += len(reasoning_piece)
                    maybe_emit(
                        {"phase": "thinking", "reasoning_chars": reasoning_chars}
                    )
                continue

            if event_type == "response.completed":
                completed_response = _response_event_response(event)
                _emit_usage(on_usage, _get_nested_value(completed_response, "usage"))
                continue

            if event_type == "response.incomplete":
                incomplete_response = _response_event_response(event)
                continue

            if event_type in {"response.failed", "response.error"}:
                failed_error = event
            _raise_if_cancelled(cancel_event)
    except Exception as exc:
        if _is_retryable_api_error(exc):
            raise _stream_interrupted_format_error(exc) from exc
        raise

    _raise_if_cancelled(cancel_event)
    if failed_error is not None:
        error_code, error_message = _response_error(
            _response_event_response(failed_error)
        )
        reason = _provider_error_reason(
            error_code, error_message or str(failed_error), effective_max_tokens
        )
        if reason is not None:
            raise reason
        raise RetryableTranslationFormatError(
            f"OpenAI Responses API failed: {failed_error}"
        )

    if incomplete_response is not None:
        reason = _response_incomplete_reason(incomplete_response)
        if reason == "max_output_tokens":
            # Reissuing the identical request is pointless, a larger budget is
            # not, so the caller gets the limit it actually hit.
            raise ResponseTruncatedError(
                "OpenAI Responses API response was cut off at "
                f"{effective_max_tokens} output tokens.",
                limit=effective_max_tokens,
            )
        raise RetryableTranslationFormatError(
            f"OpenAI Responses API returned incomplete response: {reason or 'unknown'}"
        )

    final_content_str = "".join(stream_state["final_content"])
    if not final_content_str.strip() and completed_response is not None:
        final_content_str = _extract_response_output_text(completed_response)
    if not final_content_str.strip():
        raise RetryableTranslationFormatError("OpenAI Responses API returned empty content.")

    _emit_progress(
        on_progress,
        {
            "phase": "done",
            "translated": stream_state["translated_count"],
            "expected": expected_count,
        },
    )
    return final_content_str.strip()


class OpenAICompatBackend(BaseTranslationBackend):
    def name(self) -> str:
        return "openai"

    def cache_identity(self) -> str:
        model_name = os.getenv("LLM_MODEL_NAME", "").strip()
        base_url = _normalize_openai_compat_base_url(
            os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip()
        )
        return f"openai:{base_url or 'default'}:{model_name}"

    def supports_json_schema(self) -> bool:
        return (
            _structured_output_mode(os.getenv("OPENAI_COMPATIBILITY_BASE_URL", ""))
            == "json_schema"
        )

    def supports_reasoning(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 65536,
        response_format: dict | None = None,
        stream: bool = True,
        reasoning_effort: str | None = None,
        expected_count: int = 0,
        cancel_event=None,
        on_progress: Callable[[dict], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
    ) -> str:
        # The canonical OpenAI transport is streaming. ``stream`` remains in
        # the cross-backend signature but direct calls still return final text.
        del stream
        self._raise_if_cancelled(cancel_event)

        return _chat_responses(
            messages=messages,
            expected_count=expected_count,
            on_progress=on_progress,
            reasoning_effort=reasoning_effort,
            on_usage=on_usage,
            cancel_event=cancel_event,
            response_schema=response_format,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
