from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class JobSpec(BaseModel):
    video_paths: list[str]
    output_dir: str | None = None
    asr_backend: Literal[
        "anime-whisper",
        "qwen3-asr-1.7b",
        "whisper-ja-1.5b",
        "whisper-ja-anime-v0.3",
    ] = "whisper-ja-anime-v0.3"
    subtitle_mode: Literal["zh", "bilingual"] = "bilingual"
    skip_translation: bool = False
    show_gender: bool = True
    multi_cue_split: bool = True
    asr_recovery: bool = False
    vad_threshold: float = 0.35
    translation_batch_size: int = 100
    asr_context: str = ""
    keep_quality_report: bool = False
    translation_max_workers: int = 8
    target_lang: str | None = None
    translation_glossary: str | None = None
    llm_api_format: Literal["chat", "responses"] | None = None
    llm_reasoning_effort: Literal["low", "medium", "max"] | None = None
    keep_temp_files: bool = False
    resume_from_job_id: str = ""
    advanced: dict[str, str] = Field(default_factory=dict)


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
    llm_reasoning_effort: str = "max"
    target_lang: str = "简体中文"


class SettingsUpdate(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    hf_endpoint: str | None = None
    translation_glossary: str | None = None
    llm_api_format: Literal["chat", "responses"] | None = None
    llm_reasoning_effort: str | None = None
    target_lang: str | None = None
