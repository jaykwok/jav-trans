from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MAX_TRANSLATION_WORKERS = 64


def normalize_llm_api_format(value: str | None) -> str:
    """Clamp an LLM API format to the supported set; default 'chat'."""
    normalized = (value or "chat").strip().lower()
    return normalized if normalized in {"chat", "responses"} else "chat"


def normalize_llm_reasoning_effort(value: str | None) -> str:
    """Clamp an LLM reasoning effort to the supported set; default 'xhigh'."""
    normalized = (value or "xhigh").strip().lower()
    return normalized if normalized in {"medium", "xhigh"} else "xhigh"


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_paths: list[str] = Field(min_length=1, max_length=200)
    output_dir: str | None = None
    asr_backend: Literal[
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf",
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    ] = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    subtitle_mode: Literal["zh", "bilingual"] = "zh"
    skip_translation: bool = False
    keep_quality_report: bool = False
    translation_max_workers: int = Field(default=16, ge=1, le=MAX_TRANSLATION_WORKERS)
    target_lang: str | None = Field(default=None, max_length=64)
    translation_glossary: str | None = Field(default=None, max_length=20000)
    llm_api_format: Literal["chat", "responses"] | None = None
    llm_reasoning_effort: Literal["medium", "xhigh"] | None = None
    keep_temp_files: bool = False
    resume_from_job_id: str = Field(default="", max_length=128)
    advanced: dict[str, str] = Field(default_factory=dict, max_length=100)


class JobState(BaseModel):
    id: str
    spec: JobSpec
    created_at: str
    status: Literal[
        "queued",
        "asr",
        "translating",
        "writing",
        "done",
        "failed",
        "cancelled",
    ]
    current_stage: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


class SettingsRead(BaseModel):
    api_key_set: bool
    api_key_preview: str
    base_url: str
    model: str = ""
    proxy_protocol: Literal["http", "https", "socks5"] = "http"
    proxy_host: str = ""
    proxy_port: int | None = None
    translation_glossary: str = ""
    llm_api_format: Literal["chat", "responses"] = "chat"
    llm_reasoning_effort: Literal["medium", "xhigh"] = "xhigh"
    target_lang: str = "简体中文"


class SettingsUpdate(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    proxy_protocol: Literal["http", "https", "socks5"] | None = None
    proxy_host: str | None = Field(default=None, max_length=255)
    proxy_port: int | None = Field(default=None, ge=1, le=65535)
    translation_glossary: str | None = None
    llm_api_format: Literal["chat", "responses"] | None = None
    llm_reasoning_effort: Literal["medium", "xhigh"] | None = None
    target_lang: str | None = None
