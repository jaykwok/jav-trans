"""OpenAI-compatible audio-teacher transport.

Streaming chat-completions against providers that accept inline audio, plus the
small pieces every teacher run needs: env-file loading, JSON extraction from a
model that likes to wrap objects in prose, and an ffmpeg clip cutter.

Provider differences are confined to two request contracts (`input_audio` for
Qwen, `input_audio_raw` for OpenRouter) and one reasoning block, because those
are the only places the providers actually disagree. Sampling parameters are
deliberately not sent: teacher runs are for labelling, and a temperature knob
would make two runs of the same audio incomparable.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_API_KEY_ENV_CANDIDATES = (
    "OMNI_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
    "QWEN_API_KEY",
    "API_KEY",
)


DEFAULT_BASE_URL_ENV_CANDIDATES = (
    "OMNI_BASE_URL",
    "OPENAI_BASE_URL",
    "DASHSCOPE_BASE_URL",
    "OPENAI_COMPATIBILITY_BASE_URL",
)


OMNI_PROVIDER_PROFILES = ("qwen", "openrouter")


GEMINI_THINKING_LEVELS = (
    "minimal",
    "low",
    "medium",
    "high",
)


def normalize_openai_compat_base_url(value: str) -> str:
    """Return an OpenAI SDK base URL, accepting a pasted endpoint URL too.

    OpenRouter and some provider dashboards expose the full
    ``.../chat/completions`` endpoint.  ``openai.OpenAI(base_url=...)`` expects
    the parent API root, otherwise it appends the endpoint a second time.
    """
    result = str(value or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if result.lower().endswith(suffix):
            return result[: -len(suffix)].rstrip("/")
    return result


def audio_content_mode_for_profile(profile: str) -> str:
    """Resolve the only two provider-specific audio request contracts."""
    normalized = str(profile or "").strip().lower()
    try:
        return {
            "qwen": "input_audio",
            "openrouter": "input_audio_raw",
        }[normalized]
    except KeyError as error:
        raise ValueError(
            f"unsupported omni provider profile: {profile!r}; "
            f"expected one of {OMNI_PROVIDER_PROFILES}"
        ) from error


def reasoning_extra_body_for_profile(
    *,
    profile: str,
    enable_thinking: bool,
    thinking_budget: int,
    reasoning_effort: str,
    exclude_reasoning: bool = False,
) -> dict[str, Any]:
    """Build provider-specific reasoning fields without compatibility aliases."""
    normalized = str(profile or "").strip().lower()
    if normalized == "qwen":
        result: dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
        if enable_thinking and thinking_budget > 0:
            result["thinking_budget"] = int(thinking_budget)
        return result
    if normalized == "openrouter":
        effort = (
            str(reasoning_effort or "").strip().lower()
            if enable_thinking
            else "none"
        )
        if effort not in (*GEMINI_THINKING_LEVELS, "none"):
            raise ValueError(
                f"unsupported Gemini thinking level: {reasoning_effort!r}; "
                f"expected one of {GEMINI_THINKING_LEVELS}"
            )
        reasoning: dict[str, Any] = {"effort": effort}
        if exclude_reasoning:
            reasoning["exclude"] = True
        return {"reasoning": reasoning}
    raise ValueError(
        f"unsupported omni provider profile: {profile!r}; "
        f"expected one of {OMNI_PROVIDER_PROFILES}"
    )


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(payload))
    return rows


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    for marker in (" #", "\t#"):
        if marker in value:
            value = value.split(marker, 1)[0].rstrip()
    return value


def load_env_file(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line_number, line in enumerate(env_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            raise ValueError(f"invalid env line without '=': {env_path}:{line_number}")
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid env key {key!r}: {env_path}:{line_number}")
        parsed = _parse_env_value(value)
        loaded[key] = parsed
        os.environ.setdefault(key, parsed)
    return loaded


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def slice_audio_clip(
    *,
    source_audio: Path,
    row: Mapping[str, Any],
    output_path: Path,
    fmt: str,
    bitrate: str,
    sample_rate: int,
    force: bool,
) -> Path:
    if output_path.exists() and not force:
        return output_path
    start = max(0.0, _safe_float(row.get("start")))
    end = max(start, _safe_float(row.get("end"), start))
    if end <= start:
        end = start + max(0.05, _safe_float(row.get("duration_s"), 0.05))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-to",
        f"{end:.6f}",
        "-i",
        str(source_audio),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
    ]
    if fmt == "mp3":
        command.extend(["-codec:a", "libmp3lame", "-b:a", bitrate])
    elif fmt == "m4a":
        command.extend(["-codec:a", "aac", "-b:a", bitrate])
    elif fmt == "ogg":
        command.extend(
            [
                "-codec:a",
                "libopus",
                "-b:a",
                bitrate,
                "-application",
                "voip",
                "-vbr",
                "on",
                "-compression_level",
                "10",
            ]
        )
    elif fmt == "wav":
        command.extend(["-codec:a", "pcm_s16le"])
    else:
        raise ValueError(f"unsupported audio format: {fmt}")
    command.append(str(output_path))
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    return output_path


def extract_json_value(text: str, *, require_object: bool) -> Any:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        if not require_object:
            raise
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if require_object and not isinstance(payload, Mapping):
        raise ValueError("omni response JSON must be an object")
    return dict(payload) if isinstance(payload, Mapping) else payload


def data_uri_for_audio(path: Path, fmt: str) -> str:
    mime = {
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "ogg": "audio/ogg",
        "wav": "audio/wav",
    }[fmt]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def audio_content_part(path: Path, *, fmt: str, mode: str) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    if mode == "input_audio":
        # Qwen-Omni's OpenAI-compatible API documents local audio as
        # input_audio.data="data:;base64,<payload>" rather than OpenAI's raw
        # base64 input_audio.data shape.
        return {"type": "input_audio", "input_audio": {"data": f"data:;base64,{encoded}", "format": fmt}}
    if mode == "input_audio_raw":
        return {"type": "input_audio", "input_audio": {"data": encoded, "format": fmt}}
    uri = data_uri_for_audio(path, fmt)
    if mode == "audio_url":
        return {"type": "audio_url", "audio_url": {"url": uri}}
    if mode == "audio":
        return {"type": "audio", "audio": uri}
    if mode == "video_url":
        return {"type": "video_url", "video_url": {"url": uri}}
    if mode == "video":
        return {"type": "video", "video": uri}
    raise ValueError(f"unsupported audio content mode: {mode}")


def build_omni_request_body(
    *,
    audio_path: Path,
    fmt: str,
    audio_content_mode: str,
    model: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    enable_thinking: bool | None,
    thinking_budget: int,
    provider_profile: str,
    reasoning_effort: str,
    exclude_reasoning: bool = False,
    require_provider_parameters: bool = False,
    response_format: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build SDK request kwargs plus the provider fields merged on the wire."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                audio_content_part(audio_path, fmt=fmt, mode=audio_content_mode),
            ],
        }
    )
    extra_body: dict[str, Any] = {}
    if provider_profile:
        extra_body = reasoning_extra_body_for_profile(
            profile=provider_profile,
            enable_thinking=bool(enable_thinking),
            thinking_budget=thinking_budget,
            reasoning_effort=reasoning_effort,
            exclude_reasoning=exclude_reasoning,
        )
    elif enable_thinking is not None:
        # Legacy callers without an explicit profile retain the direct-Qwen
        # request contract. New provider-aware callers must pass a profile.
        extra_body = {"enable_thinking": bool(enable_thinking)}
        if enable_thinking and thinking_budget > 0:
            extra_body["thinking_budget"] = int(thinking_budget)
    if require_provider_parameters:
        # OpenRouter may otherwise route to an upstream variant that accepts the
        # request but does not actually honor the requested reasoning controls.
        # This is a top-level OpenRouter provider preference, not a Gemini alias.
        extra_body["provider"] = {"require_parameters": True}
    request_body: dict[str, Any] = {
        "model": model,
        "max_tokens": max(1, int(max_tokens)),
        "messages": messages,
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if response_format:
        request_body["response_format"] = dict(response_format)
    return request_body, extra_body


def redact_omni_request_preview(
    *,
    request_body: Mapping[str, Any],
    extra_body: Mapping[str, Any],
    provider_profile: str,
    base_url: str,
) -> dict[str, Any]:
    """Return the actual wire shape with only embedded audio bytes redacted."""
    preview = json.loads(json.dumps(request_body))
    audio_payload = preview["messages"][-1]["content"][-1]
    input_audio = audio_payload.get("input_audio") or {}
    data = str(input_audio.get("data") or "")
    if data.startswith("data:;base64,"):
        input_audio["data"] = (
            f"data:;base64,<redacted {len(data) - len('data:;base64,')} chars>"
        )
    elif data:
        input_audio["data"] = f"<redacted {len(data)} base64 chars>"
    preview.update(extra_body)
    return {
        "provider_profile": provider_profile or "legacy",
        "endpoint": normalize_openai_compat_base_url(base_url).rstrip("/")
        + "/chat/completions",
        "body": preview,
        "omitted_sampling_parameters": ["temperature", "top_p", "top_k"],
    }


def call_omni(
    *,
    audio_path: Path,
    fmt: str,
    audio_content_mode: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    store_stream_chunks: bool,
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 256,
    enable_thinking: bool | None = None,
    thinking_budget: int = 0,
    provider_profile: str = "",
    reasoning_effort: str = "medium",
    exclude_reasoning: bool = False,
    require_provider_parameters: bool = False,
    response_format: Mapping[str, Any] | None = None,
    require_object: bool = True,
    print_request: bool = False,
) -> tuple[Any, dict[str, Any]]:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
    normalized_base_url = normalize_openai_compat_base_url(base_url)
    if normalized_base_url:
        client_kwargs["base_url"] = normalized_base_url
    client = OpenAI(**client_kwargs)
    request_body, extra_body = build_omni_request_body(
        audio_path=audio_path,
        fmt=fmt,
        audio_content_mode=audio_content_mode,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        provider_profile=provider_profile,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=exclude_reasoning,
        require_provider_parameters=require_provider_parameters,
        response_format=response_format,
    )
    request_kwargs: dict[str, Any] = {}
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    if print_request:
        print(
            "omni_request_preview="
            + json.dumps(
                redact_omni_request_preview(
                    request_body=request_body,
                    extra_body=extra_body,
                    provider_profile=provider_profile,
                    base_url=normalized_base_url,
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
    stream = client.chat.completions.create(
        **request_body,
        **request_kwargs,
    )
    text_parts: list[str] = []
    chunks: list[dict[str, Any]] = []
    chunk_count = 0
    response_ids: set[str] = set()
    response_models: set[str] = set()
    finish_reasons: list[str] = []
    usage_payload: dict[str, Any] | None = None
    reasoning_signature_count = 0
    reasoning_text_chunk_count = 0
    reasoning_character_count = 0
    reasoning_formats: set[str] = set()
    for chunk in stream:
        chunk_count += 1
        chunk_payload = chunk.model_dump(mode="json")
        if store_stream_chunks:
            chunks.append(chunk_payload)
        if chunk_payload.get("id"):
            response_ids.add(str(chunk_payload.get("id")))
        if chunk_payload.get("model"):
            response_models.add(str(chunk_payload.get("model")))
        if chunk_payload.get("usage"):
            usage_payload = chunk_payload.get("usage")
        payload_choices = chunk_payload.get("choices") or []
        if payload_choices and isinstance(payload_choices[0], Mapping):
            delta_payload = payload_choices[0].get("delta") or {}
            if isinstance(delta_payload, Mapping):
                # OpenRouter may surface reasoning as delta.reasoning and/or
                # delta.reasoning_details[].text/signature (Gemini: google-gemini-v1).
                reasoning_text = delta_payload.get("reasoning")
                if reasoning_text:
                    reasoning_text_chunk_count += 1
                    reasoning_character_count += len(str(reasoning_text))
                for detail in delta_payload.get("reasoning_details") or []:
                    if not isinstance(detail, Mapping):
                        continue
                    if detail.get("format"):
                        reasoning_formats.add(str(detail["format"]))
                    detail_text = detail.get("text")
                    if detail_text:
                        reasoning_text_chunk_count += 1
                        reasoning_character_count += len(str(detail_text))
                    if detail.get("signature"):
                        reasoning_signature_count += 1
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason:
            finish_reasons.append(str(finish_reason))
        content = getattr(delta, "content", None) or ""
        if content:
            text_parts.append(content)
    content = "".join(text_parts)
    parsed = extract_json_value(content, require_object=require_object)
    raw_response = {
        "stream": True,
        "content": content,
        "chunk_count": chunk_count,
        "ids": sorted(response_ids),
        "models": sorted(response_models),
        "finish_reasons": finish_reasons,
        "usage": usage_payload,
        "reasoning_signature_count": reasoning_signature_count,
        "reasoning_signature_formats": sorted(reasoning_formats),
        "reasoning_text_chunk_count": reasoning_text_chunk_count,
        "reasoning_character_count": reasoning_character_count,
        "request_provider_profile": provider_profile or "legacy",
        "request_reasoning": extra_body,
        "request_response_format": dict(response_format or {}),
        "request_omitted_sampling_parameters": [
            "temperature",
            "top_p",
            "top_k",
        ],
    }
    if store_stream_chunks:
        raw_response["chunks"] = chunks
    return parsed, raw_response


