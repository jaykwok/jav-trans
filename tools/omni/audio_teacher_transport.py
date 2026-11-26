#!/usr/bin/env python3
"""Provider-isolated transports for reusable audio Teacher tooling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.omni.openai_compat import (
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    audio_content_mode_for_profile,
    call_omni,
    load_env_file,
    normalize_openai_compat_base_url,
)
from tools.omni.gemini_native import (
    GEMINI_NATIVE_EXECUTION_CONTRACT,
    GEMINI_NATIVE_MODEL,
    GeminiNativeAudioClient,
    first_mapping_value,
    parse_comma_separated_api_keys,
)


KNOWN_AUDIO_TEACHER_PROFILES = ("qwen", "openrouter", "gemini")
OPENROUTER_GEMINI_EXECUTION_CONTRACT = (
    "openrouter_gemini36_reasoning_require_parameters_v1"
)


@dataclass(frozen=True)
class AudioTeacherResult:
    parsed: Any
    raw: dict[str, Any]


class AudioTeacherTransport:
    """Small common surface; provider request/response details stay private."""

    profile: str
    model: str
    transport_name: str
    audio_content_mode: str
    execution_contract: str
    api_key_count: int
    max_concurrency: int

    def quota_status(self) -> dict[str, Any] | None:
        return None

    def call_json(
        self,
        *,
        audio_path: Path,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        enable_thinking: bool,
        thinking_level: str,
        thinking_budget: int,
        response_schema: Mapping[str, Any] | None = None,
        require_object: bool = True,
        store_stream_chunks: bool = False,
        require_provider_parameters: bool = False,
    ) -> AudioTeacherResult:
        raise NotImplementedError


class OpenAICompatibleAudioTeacherTransport(AudioTeacherTransport):
    def __init__(
        self,
        *,
        profile: str,
        model: str,
        api_key: str,
        base_url: str,
        timeout_s: float,
    ) -> None:
        if profile not in {"qwen", "openrouter"}:
            raise ValueError(f"invalid compatible API profile: {profile}")
        self.profile = profile
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = float(timeout_s)
        self.transport_name = "openai_compatible_input_audio"
        self.audio_content_mode = audio_content_mode_for_profile(profile)
        self.execution_contract = (
            OPENROUTER_GEMINI_EXECUTION_CONTRACT
            if profile == "openrouter"
            else "qwen_omni_openai_compatible_v1"
        )
        self.api_key_count = 1
        self.max_concurrency = 16 if profile == "openrouter" else 1

    def call_json(
        self,
        *,
        audio_path: Path,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        enable_thinking: bool,
        thinking_level: str,
        thinking_budget: int,
        response_schema: Mapping[str, Any] | None = None,
        require_object: bool = True,
        store_stream_chunks: bool = False,
        require_provider_parameters: bool = False,
    ) -> AudioTeacherResult:
        del response_schema  # OpenRouter/Qwen have a different schema dialect.
        parsed, raw = call_omni(
            audio_path=audio_path,
            fmt=audio_path.suffix.lstrip(".") or "wav",
            audio_content_mode=self.audio_content_mode,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
            store_stream_chunks=store_stream_chunks,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            provider_profile=self.profile,
            reasoning_effort=thinking_level if self.profile == "openrouter" else "",
            exclude_reasoning=False,
            require_provider_parameters=require_provider_parameters,
            response_format=(
                {"type": "json_object"}
                if self.profile == "openrouter" and require_object
                else None
            ),
            require_object=require_object,
        )
        if require_object and not isinstance(parsed, Mapping):
            raise ValueError("compatible audio Teacher JSON output must be an object")
        return AudioTeacherResult(parsed=parsed, raw=dict(raw))


class GoogleAIStudioAudioTeacherTransport(AudioTeacherTransport):
    PROVIDER_CONCURRENCY_CAP = 4

    def __init__(
        self,
        *,
        model: str,
        api_keys: tuple[str, ...],
        timeout_s: float,
        log,
        quota_state_path: Path | None,
    ) -> None:
        self.profile = "gemini"
        self.model = model
        self.transport_name = "google_ai_interactions_inline_audio"
        self.audio_content_mode = "native_inline_audio"
        self.execution_contract = GEMINI_NATIVE_EXECUTION_CONTRACT
        self.api_key_count = len(api_keys)
        self.max_concurrency = min(
            len(api_keys),
            self.PROVIDER_CONCURRENCY_CAP,
        )
        self.quota_state_path = quota_state_path
        self.client = GeminiNativeAudioClient(
            api_keys=api_keys,
            model=model,
            timeout_s=timeout_s,
            log=log,
            quota_state_path=quota_state_path,
        )
        status = self.client.quota_status()
        key_states = list(status["keys"].values())
        ready_values = sorted(
            str(item.get("rpm_ready_at_utc") or "") for item in key_states
        )
        rpd_ready_values = sorted(
            str(item.get("rpd_ready_at_utc") or "") for item in key_states
        )
        log(
            "gemini_quota_status "
            f"key_count={len(key_states)} "
            f"max_concurrency={self.max_concurrency} "
            f"rpd_remaining_total={sum(int(item['rpd_remaining']) for item in key_states)} "
            f"next_rpm_ready_at_utc={ready_values[0] if ready_values else 'n/a'} "
            f"next_rpd_ready_at_utc={rpd_ready_values[0] if rpd_ready_values else 'n/a'} "
            f"rpd_accounting_mode={status['rpd_accounting_mode']} "
            f"advisory_reset_at_utc={status['rpd_reset_at_utc']}"
        )

    def quota_status(self) -> dict[str, Any]:
        return self.client.quota_status()

    def call_json(
        self,
        *,
        audio_path: Path,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        enable_thinking: bool,
        thinking_level: str,
        thinking_budget: int,
        response_schema: Mapping[str, Any] | None = None,
        require_object: bool = True,
        store_stream_chunks: bool = False,
        require_provider_parameters: bool = False,
    ) -> AudioTeacherResult:
        del thinking_budget, store_stream_chunks, require_provider_parameters
        response = self.client.call_json(
            audio_path=audio_path,
            system_prompt=system_prompt,
            prompt=prompt,
            response_schema=response_schema,
            require_object=require_object,
            thinking_level=thinking_level if enable_thinking else "minimal",
            max_output_tokens=max_tokens,
        )
        return AudioTeacherResult(parsed=response.parsed, raw=response.raw)


def create_audio_teacher_transport(
    *,
    profile: str,
    env_file: Path,
    model_override: str = "",
    timeout_s: float = 240.0,
    log=None,
    model_env: tuple[str, ...] = ("OMNI_MODEL", "QWEN_OMNI_MODEL"),
    api_key_env: tuple[str, ...] = DEFAULT_API_KEY_ENV_CANDIDATES,
    base_url_env: tuple[str, ...] = DEFAULT_BASE_URL_ENV_CANDIDATES,
) -> AudioTeacherTransport:
    """Load exactly one named config and build its isolated transport."""

    if profile not in KNOWN_AUDIO_TEACHER_PROFILES:
        raise ValueError(f"unknown audio Teacher profile: {profile}")
    loaded = load_env_file(env_file)
    logger = log or (lambda _message: None)
    if profile == "gemini":
        _, configured_model = first_mapping_value(
            loaded, ("GEMINI_MODEL", *model_env)
        )
        _, raw_keys = first_mapping_value(
            loaded,
            ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OMNI_API_KEY", *api_key_env),
        )
        if not raw_keys:
            raise RuntimeError(
                "native Gemini profile requires GEMINI_API_KEY, GOOGLE_API_KEY, "
                "or OMNI_API_KEY"
            )
        return GoogleAIStudioAudioTeacherTransport(
            model=model_override or configured_model or GEMINI_NATIVE_MODEL,
            api_keys=parse_comma_separated_api_keys(raw_keys),
            timeout_s=timeout_s,
            log=logger,
            quota_state_path=env_file.with_name(f"{env_file.name}.quota.json"),
        )
    _, configured_model = first_mapping_value(
        loaded, model_env
    )
    _, api_key = first_mapping_value(loaded, api_key_env)
    _, base_url = first_mapping_value(loaded, base_url_env)
    model = model_override or configured_model
    if not model or not api_key:
        raise RuntimeError(f"{profile} profile requires a model and API key")
    return OpenAICompatibleAudioTeacherTransport(
        profile=profile,
        model=model,
        api_key=api_key,
        base_url=normalize_openai_compat_base_url(base_url),
        timeout_s=timeout_s,
    )
