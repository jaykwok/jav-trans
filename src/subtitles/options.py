from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import asdict


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class SubtitleOptions:
    max_duration: float = 6.5
    soft_max: float = 5.5
    soft_split_enabled: bool = True
    min_duration: float = 0.6
    reading_cps: float = 7.0
    reading_base: float = 0.35
    duration_ratio_cap: float = 1.65
    duration_grace: float = 0.9
    gap_padding: float = 0.05
    timeline_mode: str = "alignment"
    bilingual_secondary_weight: float = 0.4
    ascii_char_weight: float = 0.55
    line_max_chars: int = 25
    merge_adjacent: bool = True
    show_speaker: bool = False
    show_gender: bool = False

    @classmethod
    def from_env(cls) -> "SubtitleOptions":
        return cls(
            max_duration=float(os.getenv("MAX_SUBTITLE_DURATION", "6.5")),
            soft_max=float(os.getenv("SUBTITLE_SOFT_MAX_S", "5.5")),
            soft_split_enabled=_env_bool("SUBTITLE_SOFT_SPLIT_ENABLED", True),
            min_duration=float(
                os.getenv(
                    "SUBTITLE_MIN_DURATION",
                    os.getenv("MIN_SUBTITLE_DURATION", "0.6"),
                )
            ),
            reading_cps=max(1.0, float(os.getenv("SUBTITLE_READING_CPS", "7.0"))),
            reading_base=float(os.getenv("SUBTITLE_READING_BASE", "0.35")),
            duration_ratio_cap=max(
                1.0,
                float(os.getenv("SUBTITLE_DURATION_RATIO_CAP", "1.65")),
            ),
            duration_grace=float(os.getenv("SUBTITLE_DURATION_GRACE", "0.9")),
            gap_padding=float(os.getenv("SUBTITLE_GAP_PADDING", "0.05")),
            timeline_mode=os.getenv("SUBTITLE_TIMELINE_MODE", "alignment").strip().lower(),
            bilingual_secondary_weight=float(
                os.getenv("SUBTITLE_BILINGUAL_SECONDARY_WEIGHT", "0.4")
            ),
            ascii_char_weight=float(os.getenv("SUBTITLE_ASCII_CHAR_WEIGHT", "0.55")),
            line_max_chars=max(0, int(os.getenv("SRT_LINE_MAX_CHARS", "25"))),
            merge_adjacent=_env_bool("SUBTITLE_MERGE_ADJACENT", True),
            show_speaker=_env_bool("SUBTITLE_SHOW_SPEAKER", False),
            show_gender=_env_bool("SUBTITLE_SHOW_GENDER", False),
        )

    def signature(self) -> dict:
        return asdict(self)
