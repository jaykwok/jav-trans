from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import asdict


# Fixed display-time baseline for frame-derived subtitle constraints.
# This is not the source video FPS.
BASE_FPS = 24000 / 1001


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class SubtitleOptions:
    layout_engine: str = "anchor_aware_dp_v2"
    timing_model: str = "acoustic_display_dual_timeline_v1"
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
    timing_polish_enabled: bool = True
    short_gap_collapse_s: float = 0.5
    linger_s: float = 0.45
    weak_cut_snap_short_s: float = 0.25
    weak_cut_snap_normal_s: float = 0.40
    weak_cut_snap_long_s: float = 0.60
    max_display_shift_from_acoustic_end_s: float = 0.20

    @property
    def frame_duration_s(self) -> float:
        """Fixed 24000/1001 baseline frame duration used for display timing."""
        return 1.0 / BASE_FPS

    @property
    def frame_gap_s(self) -> float:
        """Two baseline frames of display gap, independent of source FPS."""
        return 2.0 * self.frame_duration_s

    @property
    def frame_min_duration_s(self) -> float:
        """Twenty baseline frames of minimum reading time, independent of source FPS."""
        return 20.0 * self.frame_duration_s

    @classmethod
    def from_env(cls) -> "SubtitleOptions":
        return cls(
            layout_engine=os.getenv("SUBTITLE_LAYOUT_ENGINE", "anchor_aware_dp_v2").strip(),
            timing_model=os.getenv(
                "SUBTITLE_TIMING_MODEL",
                "acoustic_display_dual_timeline_v1",
            ).strip(),
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
            weak_cut_snap_short_s=max(
                0.0,
                float(os.getenv("SUBTITLE_WEAK_CUT_SNAP_SHORT_S", "0.25")),
            ),
            weak_cut_snap_normal_s=max(
                0.0,
                float(os.getenv("SUBTITLE_WEAK_CUT_SNAP_NORMAL_S", "0.40")),
            ),
            weak_cut_snap_long_s=max(
                0.0,
                float(os.getenv("SUBTITLE_WEAK_CUT_SNAP_LONG_S", "0.60")),
            ),
            max_display_shift_from_acoustic_end_s=max(
                0.0,
                float(os.getenv("SUBTITLE_MAX_DISPLAY_SHIFT_FROM_ACOUSTIC_END_S", "0.20")),
            ),
        )

    def signature(self) -> dict:
        return asdict(self)
