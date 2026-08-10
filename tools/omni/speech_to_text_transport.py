#!/usr/bin/env python3
"""Provider-isolated transports for speech-to-text batch tooling.

The batch core owns concurrency and persistence.  This module owns provider
authentication, request shapes, and response normalization, matching the
existing audio Teacher Core + Adapter split without forcing STT through the
chat-completions transport.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tools.omni.gemini_native import first_mapping_value
from tools.omni.openai_compat import (
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    load_env_file,
    normalize_openai_compat_base_url,
)


KNOWN_SPEECH_TO_TEXT_PROFILES = ("openrouter",)
DEFAULT_OPENROUTER_STT_MODEL = "x-ai/grok-stt-1.0"


@dataclass(frozen=True)
class SpeechToTextResult:
    parsed: dict[str, Any]
    raw: dict[str, Any]
    response_headers: dict[str, str]


class SpeechToTextTransport:
    profile: str
    model: str
    transport_name: str
    max_concurrency: int

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str,
        diarize: bool,
        filler_words: bool,
        vad_threshold: float,
    ) -> SpeechToTextResult:
        raise NotImplementedError


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def normalize_stt_response(
    response: Mapping[str, Any],
    *,
    require_speakers: bool,
) -> dict[str, Any]:
    """Normalize xAI/OpenRouter word fields while preserving diarization."""

    words: list[dict[str, Any]] = []
    invalid_words: list[dict[str, Any]] = []
    missing_speaker_count = 0
    for index, raw in enumerate(response.get("words") or []):
        if not isinstance(raw, Mapping):
            invalid_words.append({"index": index, "reason": "not_an_object"})
            continue
        text = str(raw.get("word") or raw.get("text") or "").strip()
        start = _number(raw.get("start"))
        end = _number(raw.get("end"))
        if not text or start is None or end is None or start < 0 or end <= start:
            invalid_words.append(
                {"index": index, "reason": "invalid_text_or_span", "raw": dict(raw)}
            )
            continue
        speaker = raw.get("speaker")
        if speaker is None:
            missing_speaker_count += 1
        words.append(
            {
                "text": text,
                "start_s": round(start, 6),
                "end_s": round(end, 6),
                "speaker": speaker,
                "confidence": _number(raw.get("confidence")),
            }
        )
    if require_speakers and words and missing_speaker_count:
        raise RuntimeError(
            "speaker diarization contract failed: "
            f"{missing_speaker_count}/{len(words)} valid words have no speaker"
        )
    words.sort(
        key=lambda item: (item["start_s"], item["end_s"], item["text"])
    )
    return {
        "text": str(response.get("text") or ""),
        "language": str(response.get("language") or ""),
        "duration_s": _number(response.get("duration")),
        "words": words,
        "usage": dict(response.get("usage") or {}),
        "diagnostics": {
            "word_count": len(words),
            "invalid_word_count": len(invalid_words),
            "missing_speaker_count": missing_speaker_count,
            "speaker_count": len(
                {word["speaker"] for word in words if word["speaker"] is not None}
            ),
        },
        "invalid_words": invalid_words,
    }


class OpenRouterSpeechToTextTransport(SpeechToTextTransport):
    """OpenRouter STT adapter with xAI diarization passthrough."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout_s: float,
        provider_option_slug: str = "x-ai",
    ) -> None:
        if not model:
            raise ValueError("OpenRouter STT model must not be empty")
        if not api_key:
            raise ValueError("OpenRouter STT API key must not be empty")
        if timeout_s <= 0:
            raise ValueError("OpenRouter STT timeout must be positive")
        if not provider_option_slug:
            raise ValueError("OpenRouter provider option slug must not be empty")
        self.profile = "openrouter"
        self.model = model
        self.api_key = api_key
        self.base_url = normalize_openai_compat_base_url(base_url)
        self.timeout_s = float(timeout_s)
        self.provider_option_slug = provider_option_slug
        self.transport_name = "openrouter_audio_transcriptions"
        self.max_concurrency = 10

    @property
    def endpoint(self) -> str:
        base_url = self.base_url or "https://openrouter.ai/api/v1"
        return base_url.rstrip("/") + "/audio/transcriptions"

    def request_payload(
        self,
        *,
        audio_path: Path,
        language: str,
        diarize: bool,
        filler_words: bool,
        vad_threshold: float,
    ) -> dict[str, Any]:
        if not 0.0 <= float(vad_threshold) <= 1.0:
            raise ValueError("STT vad_threshold must be between 0 and 1")
        return {
            "model": self.model,
            "language": str(language or ""),
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
            "provider": {
                "data_collection": "allow",
                "zdr": False,
                "require_parameters": True,
                "options": {
                    self.provider_option_slug: {
                        "diarize": bool(diarize),
                        "filler_words": bool(filler_words),
                        "vad_threshold": float(vad_threshold),
                    }
                },
            },
            "input_audio": {
                "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                "format": audio_path.suffix.lstrip(".").lower() or "wav",
            },
        }

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str = "ja",
        diarize: bool = True,
        filler_words: bool = False,
        vad_threshold: float = 0.5,
    ) -> SpeechToTextResult:
        request = Request(
            self.endpoint,
            data=json.dumps(
                self.request_payload(
                    audio_path=audio_path,
                    language=language,
                    diarize=diarize,
                    filler_words=filler_words,
                    vad_threshold=vad_threshold,
                ),
                separators=(",", ":"),
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/local/jav-trans",
                "X-Title": "jav-trans Grok STT",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8"))
                headers = {
                    "x-generation-id": str(
                        response.headers.get("X-Generation-Id") or ""
                    )
                }
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenRouter STT HTTP {error.code}: {body}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"OpenRouter STT transport error: {error.reason}"
            ) from error
        if not isinstance(raw, Mapping):
            raise RuntimeError("OpenRouter STT response must be a JSON object")
        parsed = normalize_stt_response(raw, require_speakers=bool(diarize))
        return SpeechToTextResult(
            parsed=parsed,
            raw=dict(raw),
            response_headers=headers,
        )


def create_speech_to_text_transport(
    *,
    profile: str,
    env_file: Path,
    model_override: str = "",
    timeout_s: float = 180.0,
    provider_option_slug: str = "x-ai",
    model_env: tuple[str, ...] = ("OMNI_MODEL", "QWEN_OMNI_MODEL"),
    api_key_env: tuple[str, ...] = DEFAULT_API_KEY_ENV_CANDIDATES,
    base_url_env: tuple[str, ...] = DEFAULT_BASE_URL_ENV_CANDIDATES,
) -> SpeechToTextTransport:
    """Load one named Omni profile and construct its STT adapter."""

    if profile not in KNOWN_SPEECH_TO_TEXT_PROFILES:
        raise ValueError(f"unknown speech-to-text profile: {profile}")
    loaded = load_env_file(env_file)
    _, configured_model = first_mapping_value(loaded, model_env)
    _, api_key = first_mapping_value(loaded, api_key_env)
    _, base_url = first_mapping_value(loaded, base_url_env)
    if not api_key:
        raise RuntimeError(f"{profile} STT profile requires an API key")
    model = model_override or configured_model or DEFAULT_OPENROUTER_STT_MODEL
    return OpenRouterSpeechToTextTransport(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        provider_option_slug=provider_option_slug,
    )
