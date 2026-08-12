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


# There is one layout and one timing model, and these two names are what every
# cue, artifact and A/B comparison is stamped with. They are not a switch: no
# code branches on them. While two layouts coexisted that distinction was
# invisible, and setting the old name produced v3 output labelled v2 - output
# that lies about its own provenance is worse than no knob at all. So an
# unknown value is refused rather than accepted and ignored.
LAYOUT_ENGINE = "measured_safe_boundary_dp_v3"
TIMING_MODEL = "measured_lexical_extent_v2"


@dataclass(frozen=True)
class SubtitleOptions:
    layout_engine: str = LAYOUT_ENGINE
    timing_model: str = TIMING_MODEL
    # Japanese source-text target for one translated cue. This and the 7s
    # duration target are deliberately soft: measured character/word timings
    # are authoritative, so an unsplittable cue remains over the target rather
    # than receiving an invented boundary.
    max_source_chars: int = 20
    max_display_duration_s: float = 7.0
    min_duration: float = 0.6
    reading_cps: float = 7.0
    reading_base: float = 0.35
    duration_ratio_cap: float = 1.65
    duration_grace: float = 0.9
    timeline_mode: str = "alignment"
    bilingual_secondary_weight: float = 0.4
    ascii_char_weight: float = 0.55
    line_max_chars: int = 16
    timing_polish_enabled: bool = True
    short_gap_collapse_s: float = 0.5
    linger_s: float = 0.5
    max_display_shift_from_acoustic_end_s: float = 0.5
    # Local ASR transcribes moaning as text and forced alignment cannot refuse
    # it, so whole cues of nothing but vocalisation are dropped here. Only runs
    # are dropped: an isolated one between two lines of dialogue is far more
    # likely to be a real reaction than part of a moaning passage, and on a real
    # film requiring a run of 2 leaves 125 such cues alone while still removing
    # 224 of 1983 cues (11.3%).
    drop_vocalisation_only_cues: bool = True
    vocalisation_min_run: int = 2

    def __post_init__(self) -> None:
        for value, expected, name in (
            (self.layout_engine, LAYOUT_ENGINE, "SUBTITLE_LAYOUT_ENGINE"),
            (self.timing_model, TIMING_MODEL, "SUBTITLE_TIMING_MODEL"),
        ):
            if value != expected:
                raise ValueError(
                    f"{name}={value!r} is not a layout this build can produce; "
                    f"the only value is {expected!r}. It names the output, it "
                    "does not select an implementation, so accepting it would "
                    "stamp cues with a layout that did not make them."
                )

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
            layout_engine=os.getenv("SUBTITLE_LAYOUT_ENGINE", LAYOUT_ENGINE).strip(),
            timing_model=os.getenv("SUBTITLE_TIMING_MODEL", TIMING_MODEL).strip(),
            max_source_chars=max(
                1,
                int(os.getenv("SUBTITLE_MAX_SOURCE_CHARS", "20")),
            ),
            max_display_duration_s=max(
                0.0,
                float(os.getenv("SUBTITLE_MAX_DISPLAY_DURATION_S", "7.0")),
            ),
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
            line_max_chars=max(0, int(os.getenv("SRT_LINE_MAX_CHARS", "16"))),
            timing_polish_enabled=_env_bool("SUBTITLE_TIMING_POLISH_ENABLED", True),
            short_gap_collapse_s=max(
                0.0,
                float(os.getenv("SUBTITLE_SHORT_GAP_COLLAPSE_S", "0.5")),
            ),
            linger_s=max(0.0, float(os.getenv("SUBTITLE_LINGER_S", "0.5"))),
            max_display_shift_from_acoustic_end_s=max(
                0.0,
                float(os.getenv("SUBTITLE_MAX_DISPLAY_SHIFT_FROM_ACOUSTIC_END_S", "0.5")),
            ),
            drop_vocalisation_only_cues=_env_bool(
                "SUBTITLE_DROP_VOCALISATION_ONLY_CUES", True
            ),
            vocalisation_min_run=max(
                1, int(os.getenv("SUBTITLE_VOCALISATION_MIN_RUN", "2"))
            ),
        )

    def signature(self) -> dict:
        return asdict(self)
