from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


_DEFAULT_ASR_BACKEND = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame"
_MAX_TRANSLATION_WORKERS = 4

from core.config import DEFAULT_SETTINGS


def _flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _setting(key: str, default: str = "") -> str:
    return os.getenv(key, DEFAULT_SETTINGS.get(key, default)).strip()


def _spec_text(spec: Any, name: str) -> str | None:
    value = getattr(spec, name, None)
    if value is None:
        return None
    return str(value).strip()


def _translation_setting(
    spec: Any,
    advanced: dict[str, str],
    env_key: str,
    spec_name: str,
    default: str = "",
    *,
    allow_empty_spec: bool = False,
) -> str:
    if env_key in advanced:
        return advanced[env_key].strip()
    spec_value = _spec_text(spec, spec_name)
    if spec_value is not None and (spec_value or allow_empty_spec):
        return spec_value
    return _setting(env_key, default)


def _llm_api_format(value: str) -> str:
    normalized = (value or "chat").strip().lower()
    return normalized if normalized in {"chat", "responses"} else "chat"


def _llm_reasoning_effort(value: str) -> str:
    normalized = (value or "xhigh").strip().lower()
    if normalized in {"medium", "xhigh"}:
        return normalized
    return "xhigh"


@dataclass
class JobContext:
    asr_backend: str
    asr_context: str
    subtitle_mode: str
    skip_translation: bool
    target_lang: str
    translation_glossary: str
    translation_max_workers: int
    translation_cache_path: str
    job_id: str
    job_temp_dir: str
    output_dir: str | None
    keep_quality_report: bool
    keep_temp_files: bool
    run_log_enabled: bool = False
    run_log_dir: str = "./tmp/log"
    fail_on_qc_block: bool = True
    llm_api_format: str = "chat"
    llm_reasoning_effort: str = "xhigh"
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
            asr_backend=str(
                getattr(spec, "asr_backend", _DEFAULT_ASR_BACKEND)
                or _DEFAULT_ASR_BACKEND
            ),
            asr_context=str(getattr(spec, "asr_context", "") or ""),
            subtitle_mode=str(getattr(spec, "subtitle_mode", "bilingual") or "bilingual"),
            skip_translation=bool(getattr(spec, "skip_translation", False)),
            target_lang=_translation_setting(
                spec,
                advanced,
                "TARGET_LANG",
                "target_lang",
                "简体中文",
            ) or "简体中文",
            translation_glossary=_translation_setting(
                spec,
                advanced,
                "TRANSLATION_GLOSSARY",
                "translation_glossary",
                allow_empty_spec=True,
            ),
            translation_max_workers=min(
                _MAX_TRANSLATION_WORKERS,
                max(1, int(getattr(spec, "translation_max_workers", 4))),
            ),
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
                str(getattr(spec, "run_log_dir", "./tmp/log") or "./tmp/log"),
            ),
            fail_on_qc_block=_flag(
                advanced.get("FAIL_ON_QC_BLOCK"),
                bool(getattr(spec, "fail_on_qc_block", True)),
            ),
            llm_api_format=_llm_api_format(
                _translation_setting(
                    spec,
                    advanced,
                    "LLM_API_FORMAT",
                    "llm_api_format",
                    "chat",
                )
            ),
            llm_reasoning_effort=_llm_reasoning_effort(
                _translation_setting(
                    spec,
                    advanced,
                    "LLM_REASONING_EFFORT",
                    "llm_reasoning_effort",
                    "xhigh",
                )
            ),
            advanced=advanced,
        )
