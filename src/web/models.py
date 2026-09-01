from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.config import (
    DEFAULT_REASONING_EFFORT,
    normalize_reasoning_effort,
)


MAX_TRANSLATION_WORKERS = 64

normalize_llm_reasoning_effort = normalize_reasoning_effort


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_paths: list[str] = Field(min_length=1, max_length=200)
    output_dir: str | None = None
    subtitle_mode: Literal["zh", "bilingual"] = "zh"
    skip_translation: bool = False
    keep_quality_report: bool = False
    # 4, not 16. Workers do not cost tokens directly, but they set the batch
    # size, and reasoning is a per-request cost that barely scales with the
    # batch - so a wider pool splits the same film into more requests and pays
    # for the thinking preamble again in each. Measured on 1,396 cues: 16
    # workers is 32 requests, 4 workers is 8. Anything from 1 to 6 costs the
    # same as 4 on a film this size (the batch cap binds first), so 4 is the
    # fastest member of the cheapest tier rather than a compromise.
    translation_max_workers: int = Field(default=4, ge=1, le=MAX_TRANSLATION_WORKERS)
    target_lang: str | None = Field(default=None, max_length=64)
    translation_glossary: str | None = Field(default=None, max_length=20000)
    llm_reasoning_effort: Literal["none", "low", "high"] | None = None
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
    llm_reasoning_effort: Literal["none", "low", "high"] = DEFAULT_REASONING_EFFORT
    target_lang: str = "简体中文"
    translation_backend: Literal["openai", "llamacpp"] = "openai"
    llamacpp_server_path: str = ""


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    proxy_protocol: Literal["http", "https", "socks5"] | None = None
    proxy_host: str | None = Field(default=None, max_length=255)
    proxy_port: int | None = Field(default=None, ge=1, le=65535)
    translation_glossary: str | None = Field(default=None, max_length=20000)
    llm_reasoning_effort: Literal["none", "low", "high"] | None = None
    target_lang: str | None = Field(default=None, max_length=64)
    translation_backend: Literal["openai", "llamacpp"] | None = None
    llamacpp_server_path: str | None = Field(default=None, max_length=4096)
