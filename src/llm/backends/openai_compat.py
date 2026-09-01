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
import threading
import time
from typing import Callable
from urllib.parse import urlsplit

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
        max_tokens: int = 384000,
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
