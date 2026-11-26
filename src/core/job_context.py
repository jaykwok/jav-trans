from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from core.config import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SETTINGS,
    normalize_reasoning_effort,
)

_MAX_TRANSLATION_WORKERS = 64


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
    env_key: str,
    spec_name: str,
    default: str = "",
    *,
    allow_empty_spec: bool = False,
) -> str:
    spec_value = _spec_text(spec, spec_name)
    if spec_value is not None and (spec_value or allow_empty_spec):
        return spec_value
    return _setting(env_key, default)


_llm_reasoning_effort = normalize_reasoning_effort


@dataclass
class JobContext:
    subtitle_mode: str
    skip_translation: bool
    target_lang: str
    translation_glossary: str
    translation_max_workers: int
    translation_cache_path: str
    job_id: str
    job_temp_dir: str
    output_dir: str | None
    run_log_enabled: bool = True
    run_log_dir: str = "./logs/jobs"
    llm_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    output_target: str = ""
    output_generation: int = 0
    output_run_id: str = ""

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        job_id: str,
        job_temp_dir: str,
        cache_path: str,
    ) -> "JobContext":
        return cls(
            subtitle_mode=str(getattr(spec, "subtitle_mode", "bilingual") or "bilingual"),
            skip_translation=bool(getattr(spec, "skip_translation", False)),
            target_lang=_translation_setting(
                spec,
                "TARGET_LANG",
                "target_lang",
                "简体中文",
            ) or "简体中文",
            translation_glossary=_translation_setting(
                spec,
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
            run_log_enabled=_flag(
                getattr(spec, "run_log_enabled", None),
                _flag(_setting("RUN_LOG_ENABLED", "1"), True),
            ),
            run_log_dir=str(
                getattr(spec, "run_log_dir", None)
                or _setting("RUN_LOG_DIR", "./logs/jobs")
                or "./logs/jobs"
            ),
            llm_reasoning_effort=_llm_reasoning_effort(
                _translation_setting(
                    spec,
                    "LLM_REASONING_EFFORT",
                    "llm_reasoning_effort",
                    DEFAULT_REASONING_EFFORT,
                )
            ),
        )
