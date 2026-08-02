"""Flag transcripts the audio does not support - and only flag them.

This runs after decoding, which is what makes it the cheap side of the design.
Nothing here removes a cue. Every finding is a mark the subtitle layer and any
later audit can read, filter on, or ignore, and that is a deliberate inversion
of the retired pre-ASR chain: that chain deleted audio on acoustic evidence
before any text existed, and its mistakes were unrecoverable. Here the text is
already in hand, the evidence is far stronger, and a wrong call costs a label
rather than a line.

The features come from `asr.cue_features`, which has been written and tested since the
old design but was never wired into anything. What it could not see is the one
signal the alignment head adds: **text the acoustics do not support aligns
badly**. A runaway decode or an invented phrase still reads as plausible
Japanese, so `unique_ratio` and `chars_per_sec` catch only the degenerate cases;
a low forced-alignment score catches text that is fluent and simply not there.

Thresholds here are measured or disabled. `min_unique_ratio` comes from the
2026-07-31 real-audio pilot (runaway regions 0.107 against 0.475 for real
speech, threshold placed between them at 0.25). `min_alignment_score` has no
measured value yet, so it defaults to None and the check does not run - an
uncalibrated threshold that silently marks a third of the output would be worse
than no check at all. `tools/align/calibrate_alignment_score.py` produces it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

POSTGATE_SCHEMA = "text_alignment_postgate_v1"

FLAG_RUNAWAY = "runaway_repetition"
FLAG_IMPOSSIBLE_RATE = "impossible_speech_rate"
FLAG_REPEATED_UNIT = "repeated_unit"
FLAG_DUPLICATE_NEIGHBOUR = "duplicate_of_neighbour"
FLAG_UNSUPPORTED_BY_AUDIO = "unsupported_by_audio"
FLAG_EMPTY = "empty_text"

ALL_FLAGS = (
    FLAG_EMPTY,
    FLAG_RUNAWAY,
    FLAG_IMPOSSIBLE_RATE,
    FLAG_REPEATED_UNIT,
    FLAG_DUPLICATE_NEIGHBOUR,
    FLAG_UNSUPPORTED_BY_AUDIO,
)


@dataclass(frozen=True)
class PostGateConfig:
    # Below this, the decode has collapsed into a repeating loop. Measured, not
    # chosen: runaway regions scored 0.107 and real speech 0.475 in the pilot.
    min_unique_ratio: float = 0.25
    # Japanese tops out around 8-10 characters/second in continuous speech; the
    # cap is set clear of that so it fires on decode artefacts rather than on
    # fast talkers. It is a sanity bound, not a discriminator.
    max_chars_per_sec: float = 14.0
    # A unit repeated this many times in a row is a loop, not emphasis. Kept
    # generous because this domain genuinely repeats short interjections.
    max_repeat_run: int = 5
    # Minimum characters before the ratio checks mean anything. On two or three
    # characters `unique_ratio` is quantised so coarsely that it flags ordinary
    # short interjections.
    min_chars_for_ratio_checks: int = 6
    # No measured value yet, so disabled. See the module docstring.
    min_alignment_score: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_unique_ratio <= 1.0:
            raise ValueError("min_unique_ratio must be in [0, 1]")
        if self.max_chars_per_sec <= 0.0:
            raise ValueError("max_chars_per_sec must be > 0")
        if self.max_repeat_run < 1:
            raise ValueError("max_repeat_run must be >= 1")
        if self.min_chars_for_ratio_checks < 1:
            raise ValueError("min_chars_for_ratio_checks must be >= 1")


def _observation(candidate: Mapping[str, Any]) -> dict[str, Any]:
    cue = candidate.get("cue_features")
    if isinstance(cue, Mapping):
        observation = cue.get("text_observation")
        if isinstance(observation, Mapping):
            return dict(observation)
    return {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def review(
    candidate: Mapping[str, Any],
    *,
    alignment_score: float | None = None,
    config: PostGateConfig | None = None,
) -> dict[str, Any]:
    """Flags for one decoded cue. `kept` is always True by construction.

    `candidate` is an `asr.cue_features.build_candidate` payload. `alignment_score` is
    the mean per-character forced-alignment score for the same cue, which
    `asr.subtitle_timing.build_aligned_word_timestamps` returns; passing None
    means the cue has no measured timing and the audio-support check is skipped
    rather than assumed to have passed.
    """
    config = config or PostGateConfig()
    observation = _observation(candidate)
    text = str(candidate.get("text") or candidate.get("raw_text") or "").strip()
    chars = int(_number(observation.get("char_count")))
    flags: list[str] = []

    if not text or chars <= 0:
        flags.append(FLAG_EMPTY)
    else:
        ratio_checks_apply = chars >= config.min_chars_for_ratio_checks
        if (
            ratio_checks_apply
            and _number(observation.get("unique_ratio"), 1.0) < config.min_unique_ratio
        ):
            flags.append(FLAG_RUNAWAY)
        if _number(observation.get("chars_per_sec")) > config.max_chars_per_sec:
            flags.append(FLAG_IMPOSSIBLE_RATE)
        if (
            ratio_checks_apply
            and int(_number(observation.get("repeat_run"))) > config.max_repeat_run
        ):
            flags.append(FLAG_REPEATED_UNIT)

    adjacency = candidate.get("adjacency")
    if isinstance(adjacency, Mapping) and text:
        if bool(adjacency.get("prev_text_same")) or bool(
            adjacency.get("next_text_same")
        ):
            flags.append(FLAG_DUPLICATE_NEIGHBOUR)

    threshold = config.min_alignment_score
    if threshold is not None and alignment_score is not None:
        if float(alignment_score) < float(threshold):
            flags.append(FLAG_UNSUPPORTED_BY_AUDIO)

    return {
        "schema": POSTGATE_SCHEMA,
        # Never False. The post-gate marks; the caller decides what to do with
        # a mark. Anything that drops a cue belongs outside this module, where
        # it is visible.
        "kept": True,
        "flags": flags,
        "flagged": bool(flags),
        "alignment_score": (
            round(float(alignment_score), 4) if alignment_score is not None else None
        ),
        "alignment_score_checked": bool(
            threshold is not None and alignment_score is not None
        ),
        "observed": {
            "char_count": chars,
            "unique_ratio": _number(observation.get("unique_ratio")),
            "chars_per_sec": _number(observation.get("chars_per_sec")),
            "repeat_run": int(_number(observation.get("repeat_run"))),
        },
    }


def review_all(
    candidates: list[Mapping[str, Any]],
    *,
    alignment_scores: list[float | None] | None = None,
    config: PostGateConfig | None = None,
) -> list[dict[str, Any]]:
    scores = alignment_scores or [None] * len(candidates)
    if len(scores) != len(candidates):
        raise ValueError("alignment_scores must line up with candidates")
    return [
        review(candidate, alignment_score=score, config=config)
        for candidate, score in zip(candidates, scores)
    ]
