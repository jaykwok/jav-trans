"""OpenAI-compatible backend: the canonical Chat/Responses streaming transport.

This module owns the OpenAI wire protocol end to end (client cache, request
construction, streaming loops, retry-with-backoff, usage/progress emission).
Nothing here imports llm.translator — the orchestration layer calls into the
backend, never the other way around.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

from openai import OpenAI

from core.stage_errors import MISSING_MODEL
from llm import settings as llm_settings
from llm import transport_util
from llm.backends.base import BaseTranslationBackend
from llm.errors import ResponseTruncatedError, RetryableTranslationFormatError
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


def _is_deepseek_model(model_name: str | None) -> bool:
    return "deepseek" in (model_name or "").lower()


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
    return f"{normalized}/v1"


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


def _chat_response_format(
    model_name: str | None = None,
    *,
    schema: dict | None = None,
    schema_name: str = "subtitle_translations",
) -> dict:
    if _is_deepseek_model(model_name):
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema or _DEFAULT_OUTPUT_SCHEMA,
        },
    }


def _responses_text_format(
    model_name: str | None = None,
    *,
    schema: dict | None = None,
    schema_name: str = "subtitle_translations",
) -> dict:
    if _is_deepseek_model(model_name):
        return {"format": {"type": "json_object"}}
    return {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": schema or _DEFAULT_OUTPUT_SCHEMA,
        }
    }


def _backoff_sleep(attempt: int, exc: Exception, cancel_event=None) -> None:
    transport_util._request_backoff_sleep(attempt, exc, cancel_event=cancel_event)


def _create_chat_completion(
    request: dict,
    cancel_event: threading.Event | None = None,
):
    last_error: Exception | None = None

    for attempt in range(llm_settings.TRANSLATION_API_RETRIES):
        _raise_if_cancelled(cancel_event)
        try:
            return _get_client().chat.completions.create(**request)
        except Exception as exc:
            last_error = exc
            if not _is_retryable_api_error(exc):
                raise

            if attempt < llm_settings.TRANSLATION_API_RETRIES - 1:
                _backoff_sleep(attempt, exc, cancel_event=cancel_event)

    if last_error is not None:
        raise last_error
    raise RuntimeError("chat completion failed without an exception")


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


def _call_create_chat_completion(
    request: dict,
    cancel_event: threading.Event | None = None,
):
    # Tolerates replacement callables (tests, wrappers) that do not accept
    # cancel_event.
    if cancel_event is None:
        return _create_chat_completion(request)
    try:
        return _create_chat_completion(request, cancel_event=cancel_event)
    except TypeError as exc:
        if "cancel_event" not in str(exc):
            raise
        return _create_chat_completion(request)


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


def _response_incomplete_reason(response) -> str:
    details = getattr(response, "incomplete_details", None)
    if isinstance(details, dict):
        return str(details.get("reason", ""))
    return str(getattr(details, "reason", "") or "")


def _chat_reasoning_fields(effort: str) -> dict:
    """Translate one thinking tier into the Chat Completions wire fields.

    The only place the tier names meet the API, which is why the mismatch that
    caused them survived so long. DeepSeek's Chat surface accepts `low`, `high`
    and `max` for `reasoning_effort` and nothing else; an unrecognised value is
    not rejected, it is ignored, and the request falls back to the documented
    default of `high`. `core.config.REASONING_EFFORTS` is therefore restricted
    to values that survive this trip - passing a tier through verbatim is only
    safe because that tuple is now wire-accurate.

    Switching thinking off is a separate field (`extra_body.thinking.type`)
    rather than an effort value on this surface, so `none` sets that and omits
    `reasoning_effort` entirely: providers keyed on the extra_body convention
    ignore the effort, providers keyed on the effort ignore the extra_body, and
    sending "disabled" alongside a nonzero effort asks the two conventions for
    opposite things.
    """
    if effort == "none":
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {
        "reasoning_effort": effort,
        "extra_body": {"thinking": {"type": "enabled"}},
    }


def _chat_completions(
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
    effective_effort = llm_settings._normalize_reasoning_effort(
        reasoning_effort
        or os.getenv("LLM_REASONING_EFFORT", llm_settings.LLM_REASONING_EFFORT)
    )
    request = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "response_format": _chat_response_format(
            model_name,
            schema=response_schema,
            schema_name=response_schema_name,
        ),
        "stream_options": {"include_usage": True},
        "temperature": (
            llm_settings.TRANSLATION_TEMPERATURE if temperature is None else temperature
        ),
        "top_p": llm_settings.TRANSLATION_TOP_P if top_p is None else top_p,
    }
    request.update(_chat_reasoning_fields(effective_effort))
    effective_max_tokens = (
        llm_settings.TRANSLATION_MAX_TOKENS if max_tokens is None else max_tokens
    )
    if effective_max_tokens > 0:
        request["max_tokens"] = effective_max_tokens

    try:
        response_stream = _call_create_chat_completion(request, cancel_event=cancel_event)
    except Exception as exc:
        if "stream_options" not in request or "stream_options" not in str(exc):
            raise
        request = dict(request)
        request.pop("stream_options", None)
        response_stream = _call_create_chat_completion(request, cancel_event=cancel_event)

    finish_reason = None
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
        for chunk in response_stream:
            _raise_if_cancelled(cancel_event)
            _emit_usage(on_usage, getattr(chunk, "usage", None))
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                reasoning_chars += len(reasoning_content)
                maybe_emit(
                    {"phase": "thinking", "reasoning_chars": reasoning_chars}
                )
            if hasattr(delta, "content") and delta.content:
                _emit_stream_content_progress(
                    piece=delta.content,
                    state=stream_state,
                    expected_count=expected_count,
                    maybe_emit=maybe_emit,
                )
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            _raise_if_cancelled(cancel_event)
    except Exception as exc:
        if _is_retryable_api_error(exc):
            raise _stream_interrupted_format_error(exc) from exc
        raise

    _raise_if_cancelled(cancel_event)
    if finish_reason == "length":
        # Not retryable as-is - the identical request truncates identically -
        # but `translator._chat` may reissue once with a larger budget, so the
        # limit travels with the error. The number below is THIS request's
        # budget, which for the JSON profile is
        # `source_chars * TRANSLATION_OUTPUT_CHAR_RATIO + structure`; the old
        # message told the reader to raise `TRANSLATION_MAX_TOKENS`, a ceiling
        # the profile budget is always orders of magnitude under, so following
        # it changed nothing (sample-b, 2026-08-13).
        raise ResponseTruncatedError(
            f"{_JSON_OUTPUT_LABEL} response was cut off at "
            f"{effective_max_tokens} output tokens.",
            limit=effective_max_tokens,
        )

    final_content_str = "".join(stream_state["final_content"])
    if not final_content_str.strip():
        raise RetryableTranslationFormatError(f"{_JSON_OUTPUT_LABEL} returned empty content.")
    _emit_progress(
        on_progress,
        {
            "phase": "done",
            "translated": stream_state["translated_count"],
            "expected": expected_count,
        },
    )
    return final_content_str.strip()


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
    request = {
        "model": model_name,
        "input": _build_responses_input(messages),
        "stream": True,
        "reasoning": {"effort": effective_reasoning_effort},
        "text": _responses_text_format(
            model_name,
            schema=response_schema,
            schema_name=response_schema_name,
        ),
        "temperature": effective_temperature,
        "top_p": effective_top_p,
    }
    if effective_max_tokens > 0:
        request["max_output_tokens"] = effective_max_tokens

    response_stream = _call_create_response(request, cancel_event=cancel_event)

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
        raise RetryableTranslationFormatError(
            f"OpenAI Responses API failed: {failed_error}"
        )

    if incomplete_response is not None:
        reason = _response_incomplete_reason(incomplete_response)
        if reason == "max_output_tokens":
            # Same contract as the Chat path above: identical reissue is
            # pointless, a larger budget is not, so the caller gets the limit.
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
        return "deepseek" not in os.getenv("LLM_MODEL_NAME", "").lower()

    def supports_reasoning(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    @staticmethod
    def _normalize_base_url(base_url: str | None) -> str | None:
        return _normalize_openai_compat_base_url(base_url)

    def chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 384000,
        response_format: dict | None = None,
        stream: bool = True,
        reasoning_effort: str | None = None,
        api_format: str | None = None,
        expected_count: int = 0,
        cancel_event=None,
        on_progress: Callable[[dict], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
    ) -> str:
        # The canonical OpenAI transport is streaming. ``stream`` remains in
        # the cross-backend signature but direct calls still return final text.
        del stream
        self._raise_if_cancelled(cancel_event)

        kwargs = {
            "messages": messages,
            "expected_count": expected_count,
            "on_progress": on_progress,
            "reasoning_effort": reasoning_effort,
            "on_usage": on_usage,
            "cancel_event": cancel_event,
            "response_schema": response_format,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if llm_settings._llm_api_format(api_format) == "responses":
            return _chat_responses(**kwargs)
        return _chat_completions(**kwargs)
