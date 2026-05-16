from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class SubtitleOptions:
    max_duration: float = 8.0
    soft_max: float = 6.0
    soft_split_enabled: bool = True
    line_max_chars: int = 25
    show_speaker: bool = False
    show_gender: bool = False

    @classmethod
    def from_env(cls) -> "SubtitleOptions":
        return cls(
            max_duration=float(os.getenv("MAX_SUBTITLE_DURATION", "8.0")),
            soft_max=float(os.getenv("SUBTITLE_SOFT_MAX_S", "6.0")),
            soft_split_enabled=_env_bool("SUBTITLE_SOFT_SPLIT_ENABLED", True),
            line_max_chars=max(0, int(os.getenv("SRT_LINE_MAX_CHARS", "25"))),
            show_speaker=_env_bool("SUBTITLE_SHOW_SPEAKER", False),
            show_gender=_env_bool("SUBTITLE_SHOW_GENDER", False),
        )
