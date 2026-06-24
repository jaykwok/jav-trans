from __future__ import annotations

import math
import os
from dataclasses import dataclass
from dataclasses import asdict


FALLBACK_VIDEO_FPS = 30000 / 1001


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class SubtitleOptions:
    max_display_duration_s: float = 7.0
    min_duration: float = 0.6
    reading_cps: float = 7.0
    reading_base: float = 0.35
    duration_ratio_cap: float = 1.65
    duration_grace: float = 0.9
    timeline_mode: str = "alignment"
    bilingual_secondary_weight: float = 0.4
    ascii_char_weight: float = 0.55
    line_max_chars: int = 25
    video_fps: float = FALLBACK_VIDEO_FPS
    timing_polish_enabled: bool = True
    short_gap_collapse_s: float = 0.5
    linger_s: float = 0.45

    @property
    def effective_video_fps(self) -> float:
        try:
            fps = float(self.video_fps)
        except (TypeError, ValueError):
            return FALLBACK_VIDEO_FPS
        return fps if math.isfinite(fps) and fps > 0 else FALLBACK_VIDEO_FPS

    @property
    def frame_duration_s(self) -> float:
        return 1.0 / self.effective_video_fps

    @property
    def frame_gap_s(self) -> float:
        return 2.0 * self.frame_duration_s

    @property
    def frame_min_duration_s(self) -> float:
        return 20.0 * self.frame_duration_s

    def with_video_fps(self, video_fps: float | None) -> "SubtitleOptions":
        values = asdict(self)
        if video_fps is None:
            values["video_fps"] = FALLBACK_VIDEO_FPS
        else:
            try:
                fps = float(video_fps)
            except (TypeError, ValueError):
                fps = FALLBACK_VIDEO_FPS
            values["video_fps"] = fps if math.isfinite(fps) and fps > 0 else FALLBACK_VIDEO_FPS
        return type(self)(**values)

    @classmethod
    def from_env(cls) -> "SubtitleOptions":
        return cls(
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
            timeline_mode=os.getenv("SUBTITLE_TIMELINE_MODE", "alignment").strip().lower(),
            bilingual_secondary_weight=float(
                os.getenv("SUBTITLE_BILINGUAL_SECONDARY_WEIGHT", "0.4")
            ),
            ascii_char_weight=float(os.getenv("SUBTITLE_ASCII_CHAR_WEIGHT", "0.55")),
            line_max_chars=max(0, int(os.getenv("SRT_LINE_MAX_CHARS", "25"))),
            timing_polish_enabled=_env_bool("SUBTITLE_TIMING_POLISH_ENABLED", True),
            short_gap_collapse_s=max(
                0.0,
                float(os.getenv("SUBTITLE_SHORT_GAP_COLLAPSE_S", "0.5")),
            ),
            linger_s=max(0.0, float(os.getenv("SUBTITLE_LINGER_S", "0.45"))),
        )

    def signature(self) -> dict:
        return asdict(self)
