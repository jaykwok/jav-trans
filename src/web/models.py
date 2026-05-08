from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class JobSpec(BaseModel):
    video_paths: list[str] = Field(min_length=1, max_length=200)
    output_dir: str | None = None
    asr_backend: Literal[
        "anime-whisper",
        "qwen3-asr-1.7b",
        "whisper-ja-1.5b",
        "whisper-ja-anime-v0.3",
    ] = "whisper-ja-anime-v0.3"
    subtitle_mode: Literal["zh", "bilingual"] = "zh"
    skip_translation: bool = False
    show_gender: bool = True
    multi_cue_split: bool = True
    asr_recovery: bool = False
    vad_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    translation_batch_size: int = Field(default=200, ge=1, le=500)
    asr_context: str = Field(default="", max_length=4000)
    keep_quality_report: bool = False
    translation_max_workers: int = Field(default=4, ge=1, le=8)
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
    hf_endpoint: str = ""
    translation_glossary: str = ""
    llm_api_format: Literal["chat", "responses"] = "chat"
    llm_reasoning_effort: Literal["medium", "xhigh"] = "xhigh"
    target_lang: str = "简体中文"


class SettingsUpdate(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    hf_endpoint: str | None = None
    translation_glossary: str | None = None
    llm_api_format: Literal["chat", "responses"] | None = None
    llm_reasoning_effort: Literal["medium", "xhigh"] | None = None
    target_lang: str | None = None
