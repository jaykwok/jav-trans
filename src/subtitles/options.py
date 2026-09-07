from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict

from core.typed_config import FALL_BACK, env_bool, env_choice, env_float, env_int, env_text


# Fixed display-time baseline for frame-derived subtitle constraints.
# This is not the source video FPS.
BASE_FPS = 24000 / 1001

# `alignment`/`aligned`/`raw` keep the measured cue window; anything else runs
# the reading-time model. An unrecognised value used to fall into the reading
# model silently, which is a different subtitle timeline than the one asked for.
TIMELINE_MODES = ("alignment", "aligned", "raw", "reading")


# There is one layout and one timing model, and these two names are what every
# cue, artifact and A/B comparison is stamped with. They are not a switch: no
# code branches on them. While two layouts coexisted that distinction was
# invisible, and setting the old name produced v3 output labelled v2 - output
# that lies about its own provenance is worse than no knob at all. So an
# unknown value is refused rather than accepted and ignored.
# v3_1: the same DP and the same candidate boundaries, with the measured gap
# graded inside `word_gap` instead of every word gap scoring alike. It moves
# ~1.4% of cuts, so an artifact from before it must not claim to be one from
# after it.
LAYOUT_ENGINE = "measured_safe_boundary_dp_v3_1"
# v3: the display end may linger into silence that is already empty, bounded by
# `linger_s` and `max_display_shift_from_acoustic_end_s` and stopping two frames
# before the next cue. Acoustic edges and every word timing are unchanged, and
# the layout stamp above does not move with it - the cuts land where they did,
# which is the reason these are two fields rather than one.
TIMING_MODEL = "measured_lexical_extent_v3"


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
    # Separate from `line_max_chars` because the two style guides disagree: CHS
    # allows 16 full-width per line, the Japanese guide 13 for horizontal
    # subtitles. One shared number would put one of the two tracks out of spec.
    ja_line_max_chars: int = 13
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
    # The acoustic half of the verdict, from the v2 head's three-class frame
    # output. It only ever ADDS drops: requiring acoustic confirmation before
    # honouring the text rule put 457 cues of plain moaning back on eight films,
    # because text evidence for a run of pure-vocalisation cues is already
    # strong and the acoustics are here to reach what text cannot see - the
    # isolated cue and the onomatopoeia no allow-list spells.
    #
    # Off by default is NOT an option here: a v1 head simply produces no
    # acoustics and every cue falls back to the text rule, so this switch is for
    # turning the addition off deliberately, not for the absence of a head.
    vocalisation_use_acoustics: bool = True
    # Thresholds, not hard-coded in the classifier, because they reach the cache
    # signature through `asdict` - a rerun after retuning one must not serve the
    # cues the previous value produced.
    vocalisation_vocal_speech_max: float = 0.10
    vocalisation_vocal_speech_run_max_s: float = 0.30
    vocalisation_kana_speech_max: float = 0.05
    vocalisation_kana_vocalisation_min: float = 0.60
    vocalisation_vocal_text_speech_min: float = 0.30
    # Take a purely-vocal head or tail off a cue that also holds real speech.
    # This is the one place the filter edits text inside a cue rather than
    # keeping or dropping the whole of it, which is why it has its own switch -
    # but the criterion is not new: a fragment goes only when the same joint
    # verdict applied to that fragment's own re-measured frames says drop, so
    # nothing can be removed here that would have been kept as a cue.
    vocalisation_split_mixed_cues: bool = True

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
        """Read the layout and timing settings, with every field bounded.

        Ranges are declared rather than assumed: a ratio is a ratio, a duration
        is not negative, and a reading speed of zero divides. Out-of-range and
        unparseable values are recorded field by field (see `core.typed_config`)
        instead of raising here, so a mistyped `.env` produces a list of what is
        wrong rather than a traceback from whichever field parsed first.

        The two version stamps stay strict - `__post_init__` refuses them.
        """
        return cls(
            layout_engine=env_text("SUBTITLE_LAYOUT_ENGINE", LAYOUT_ENGINE),
            timing_model=env_text("SUBTITLE_TIMING_MODEL", TIMING_MODEL),
            max_source_chars=env_int("SUBTITLE_MAX_SOURCE_CHARS", 20, minimum=1),
            max_display_duration_s=env_float(
                "SUBTITLE_MAX_DISPLAY_DURATION_S", 7.0, minimum=0.0
            ),
            min_duration=env_float(
                "SUBTITLE_MIN_DURATION",
                env_float("MIN_SUBTITLE_DURATION", 0.6, minimum=0.0),
                minimum=0.0,
            ),
            reading_cps=env_float("SUBTITLE_READING_CPS", 7.0, minimum=1.0, out_of_range=FALL_BACK),
            reading_base=env_float("SUBTITLE_READING_BASE", 0.35, minimum=0.0),
            duration_ratio_cap=env_float(
                "SUBTITLE_DURATION_RATIO_CAP", 1.65, minimum=1.0
            ),
            duration_grace=env_float("SUBTITLE_DURATION_GRACE", 0.9, minimum=0.0),
            timeline_mode=env_choice(
                "SUBTITLE_TIMELINE_MODE", "alignment", choices=TIMELINE_MODES
            ),
            bilingual_secondary_weight=env_float(
                "SUBTITLE_BILINGUAL_SECONDARY_WEIGHT", 0.4, minimum=0.0, maximum=1.0
            ),
            ascii_char_weight=env_float(
                "SUBTITLE_ASCII_CHAR_WEIGHT", 0.55, minimum=0.0, maximum=1.0
            ),
            line_max_chars=env_int("SRT_LINE_MAX_CHARS", 16, minimum=0),
            ja_line_max_chars=env_int("SRT_JA_LINE_MAX_CHARS", 13, minimum=0),
            timing_polish_enabled=env_bool("SUBTITLE_TIMING_POLISH_ENABLED", True),
            short_gap_collapse_s=env_float(
                "SUBTITLE_SHORT_GAP_COLLAPSE_S", 0.5, minimum=0.0
            ),
            linger_s=env_float("SUBTITLE_LINGER_S", 0.5, minimum=0.0),
            max_display_shift_from_acoustic_end_s=env_float(
                "SUBTITLE_MAX_DISPLAY_SHIFT_FROM_ACOUSTIC_END_S", 0.5, minimum=0.0
            ),
            drop_vocalisation_only_cues=env_bool(
                "SUBTITLE_DROP_VOCALISATION_ONLY_CUES", True
            ),
            vocalisation_min_run=env_int(
                "SUBTITLE_VOCALISATION_MIN_RUN", 2, minimum=1
            ),
            vocalisation_use_acoustics=env_bool(
                "SUBTITLE_VOCALISATION_USE_ACOUSTICS", True
            ),
            vocalisation_vocal_speech_max=env_float(
                "SUBTITLE_VOCALISATION_VOCAL_SPEECH_MAX", 0.10, minimum=0.0, maximum=1.0
            ),
            vocalisation_vocal_speech_run_max_s=env_float(
                "SUBTITLE_VOCALISATION_VOCAL_SPEECH_RUN_MAX_S", 0.30, minimum=0.0
            ),
            vocalisation_kana_speech_max=env_float(
                "SUBTITLE_VOCALISATION_KANA_SPEECH_MAX", 0.05, minimum=0.0, maximum=1.0
            ),
            vocalisation_kana_vocalisation_min=env_float(
                "SUBTITLE_VOCALISATION_KANA_VOCALISATION_MIN",
                0.60,
                minimum=0.0,
                maximum=1.0,
            ),
            vocalisation_vocal_text_speech_min=env_float(
                "SUBTITLE_VOCALISATION_VOCAL_TEXT_SPEECH_MIN",
                0.30,
                minimum=0.0,
                maximum=1.0,
            ),
            vocalisation_split_mixed_cues=env_bool(
                "SUBTITLE_VOCALISATION_SPLIT_MIXED_CUES", True
            ),
        )

    def signature(self) -> dict:
        return asdict(self)
