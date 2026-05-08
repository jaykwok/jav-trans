from __future__ import annotations

import json
import os
from typing import Callable

import httpx


MICU_BASE_URL_MARKERS = ("micuapi.ai",)
_REASONING_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high",
}


def is_micu_grok_responses_request(
    *,
    model_name: str,
    api_format: str,
    base_url: str | None = None,
) -> bool:
    if (api_format or "").strip().lower() != "responses":
        return False
    if "grok" not in (model_name or "").lower():
        return False

    resolved_base_url = (
        base_url
        if base_url is not None
        else os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "")
    )
    normalized_base_url = (resolved_base_url or "").strip().lower()
    return any(marker in normalized_base_url for marker in MICU_BASE_URL_MARKERS)


def build_micu_grok_responses_input(messages: list[dict]) -> list[dict]:
    merged = "\n\n".join(
        f"{str(message.get('role') or 'user').upper()}:\n{message.get('content', '')}"
        for message in messages
    )
    return [{"role": "user", "content": merged.strip()}]


def build_micu_grok_responses_request(
    *,
    messages: list[dict],
    model_name: str,
    max_tokens: int,
    reasoning_effort: str,
) -> dict:
    effort = _REASONING_EFFORT_MAP.get(
        (reasoning_effort or "").strip().lower(),
        "high",
    )
    return {
        "model": model_name,
        "input": build_micu_grok_responses_input(messages),
        "stream": True,
        "reasoning": {"effort": effort},
        "include_reasoning": True,
        "max_tokens": max(16000, max_tokens or 0),
    }


def _responses_endpoint_url(base_url: str) -> str:
    normalized_base_url = (base_url or "").strip().rstrip("/")
    if normalized_base_url.endswith("/responses"):
        return normalized_base_url
    return f"{normalized_base_url}/responses"


def _parse_sse_json_event(event_type: str, data_lines: list[str]) -> dict | None:
    if not data_lines:
        return None
    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        return {"type": "__done__"} if data == "[DONE]" else None

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if event_type and not payload.get("type"):
        payload["type"] = event_type
    return payload


def iter_sse_json_events(lines):
    event_type = ""
    data_lines: list[str] = []
    for raw_line in lines:
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else str(raw_line)
        )
        line = line.rstrip("\r\n")
        if not line:
            event = _parse_sse_json_event(event_type, data_lines)
            if event and event.get("type") == "__done__":
                return
            if event:
                yield event
            event_type = ""
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        if line.startswith("{") and not data_lines:
            event = _parse_sse_json_event("", [line])
            if event:
                yield event
            continue

        field, sep, value = line.partition(":")
        if not sep:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_type = value
        elif field == "data":
            data_lines.append(value)

    event = _parse_sse_json_event(event_type, data_lines)
    if event and event.get("type") != "__done__":
        yield event


def create_micu_grok_response_stream(
    request: dict,
    *,
    api_key: str,
    base_url: str,
    api_retries: int,
    is_retryable_api_error: Callable[[Exception], bool],
    backoff_sleep: Callable[[int, Exception], None],
):
    def stream_events():
        last_error: Exception | None = None

        for attempt in range(api_retries):
            emitted = False
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                timeout = httpx.Timeout(
                    connect=20.0,
                    read=None,
                    write=60.0,
                    pool=20.0,
                )
                with httpx.Client(timeout=timeout) as client:
                    with client.stream(
                        "POST",
                        _responses_endpoint_url(base_url),
                        headers=headers,
                        json=request,
                    ) as response:
                        response.raise_for_status()
                        for event in iter_sse_json_events(response.iter_lines()):
                            emitted = True
                            yield event
                        return
            except Exception as exc:
                last_error = exc
                if emitted or not is_retryable_api_error(exc):
                    raise

                if attempt < api_retries - 1:
                    backoff_sleep(attempt, exc)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Micu Grok response creation failed without an exception")

    return stream_events()
