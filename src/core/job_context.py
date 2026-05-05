from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from core.config import DEFAULT_SETTINGS


def _flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _setting(key: str, default: str = "") -> str:
    return os.getenv(key, DEFAULT_SETTINGS.get(key, default)).strip()


@dataclass
class JobContext:
    asr_backend: str
    asr_context: str
    subtitle_mode: str
    show_gender: bool
    multi_cue_split: bool
    asr_recovery: bool
    vad_threshold: float
    skip_translation: bool
    target_lang: str
    translation_glossary: str
    translation_batch_size: int
    translation_max_workers: int
    translation_cache_path: str
    job_id: str
    job_temp_dir: str
    output_dir: str | None
    keep_quality_report: bool
    keep_temp_files: bool
    run_log_enabled: bool = False
    run_log_dir: str = "./log"
    fail_on_qc_block: bool = True
    advanced: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        job_id: str,
        job_temp_dir: str,
        cache_path: str,
    ) -> "JobContext":
        advanced = {
            str(key): str(value)
            for key, value in dict(getattr(spec, "advanced", {}) or {}).items()
            if key
        }
        return cls(
            asr_backend=str(getattr(spec, "asr_backend", "anime-whisper") or "anime-whisper"),
            asr_context=str(getattr(spec, "asr_context", "") or ""),
            subtitle_mode=str(getattr(spec, "subtitle_mode", "bilingual") or "bilingual"),
            show_gender=bool(getattr(spec, "show_gender", True)),
            multi_cue_split=bool(getattr(spec, "multi_cue_split", True)),
            asr_recovery=bool(getattr(spec, "asr_recovery", False)),
            vad_threshold=float(getattr(spec, "vad_threshold", 0.35)),
            skip_translation=bool(getattr(spec, "skip_translation", False)),
            target_lang=advanced.get("TARGET_LANG", _setting("TARGET_LANG", "简体中文")) or "简体中文",
            translation_glossary=advanced.get("TRANSLATION_GLOSSARY", _setting("TRANSLATION_GLOSSARY")),
            translation_batch_size=int(getattr(spec, "translation_batch_size", 100)),
            translation_max_workers=max(1, int(getattr(spec, "translation_max_workers", 8))),
            translation_cache_path=str(cache_path or ""),
            job_id=str(job_id or ""),
            job_temp_dir=str(job_temp_dir or ""),
            output_dir=str(getattr(spec, "output_dir", "") or "") or None,
            keep_quality_report=bool(getattr(spec, "keep_quality_report", False)),
            keep_temp_files=bool(getattr(spec, "keep_temp_files", False)),
            run_log_enabled=_flag(
                advanced.get("RUN_LOG_ENABLED"),
                bool(getattr(spec, "run_log_enabled", False)),
            ),
            run_log_dir=advanced.get(
                "RUN_LOG_DIR",
                str(getattr(spec, "run_log_dir", "./log") or "./log"),
            ),
            fail_on_qc_block=_flag(
                advanced.get("FAIL_ON_QC_BLOCK"),
                bool(getattr(spec, "fail_on_qc_block", True)),
            ),
            advanced=advanced,
        )
