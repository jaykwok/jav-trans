"""`non_speech` says a chunk has no words, not that it has no human voice.

The type track is recovered by stem-matching the teacher's free-form
`omni_flags`, which has 237 surface forms for a handful of concepts. That
recovery had one systematic error: `non_speech` sat at the head of
NON_VOCAL_STEMS, so a chunk flagged only `non_speech` was typed `non_vocal`.
But the teacher used `non_speech` as a synonym for "drop" - a moan, a breath
and a laugh are all non_speech - and 1866 of the 2945 `non_vocal` spans (63.4%)
were typed from exactly that flag and nothing else.

Deleting the stem outright would have been worse than leaving it, because
"speech" is a substring of "non_speech" and the flag would have fallen through
to SPEECH_STEMS. So the fix is ordering: concrete acoustic evidence is matched
first and still wins, and only a flag with no concrete evidence at all reaches
NON_LEXICAL_STEMS, where it contributes nothing and the span becomes `unsure`.
These tests pin both halves - the flag that must stop typing spans, and the
flags that must keep typing them exactly as before.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.datasets.build_typed_span_dataset import (  # noqa: E402
    NON_LEXICAL_STEMS,
    NON_SEMANTIC_VOCAL_STEMS,
    NON_VOCAL_STEMS,
    TYPE_NON_SEMANTIC_VOCAL,
    TYPE_NON_VOCAL,
    TYPE_UNSURE,
    resolve_drop_subtype,
)


@pytest.mark.parametrize(
    "flags",
    [
        ["non_speech"],
        ["no_speech"],
        ["NON_SPEECH"],
        ["non_speech", "no_speech"],
    ],
)
def test_absence_of_words_alone_types_nothing(flags: list[str]) -> None:
    """The regression this file exists for: these used to become non_vocal."""
    assert resolve_drop_subtype(flags) == TYPE_UNSURE


@pytest.mark.parametrize(
    "flags",
    [
        ["non_speech", "music"],
        ["no_speech", "silence"],
        ["mechanical_noise"],
        ["train_noise"],
        ["water_noise"],
        ["environmental_noise"],
        ["object_noise"],
    ],
)
def test_concrete_acoustic_evidence_still_wins(flags: list[str]) -> None:
    """A flag naming an actual source must keep typing the span non_vocal.

    The compounds matter: bare `noise` is no longer evidence, so these survive
    only because the source word in them is matched first.
    """
    assert resolve_drop_subtype(flags) == TYPE_NON_VOCAL


@pytest.mark.parametrize(
    "flags",
    [
        ["noise"],
        ["noise_only"],
        ["short_noise"],
        ["non_speech_noise"],
        ["speechless_noise"],
        ["non_speech", "noise"],
        ["sound_event"],
        ["very_short"],
    ],
)
def test_the_mere_presence_of_sound_types_nothing(flags: list[str]) -> None:
    """`noise` names no source, so it cannot say the source was not a person.

    A 70-window audio audit scored bare `noise` at 75.5% human-produced against
    a concrete-vocal control at 81.2% and concrete non-vocal flags at 36.1%.
    `non_speech_noise` and `speechless_noise` reach 84.6% - these two were
    previously asserted to be non_vocal on the grounds that they contain
    `noise`, which the audit shows is not evidence in the first place.
    """
    assert resolve_drop_subtype(flags) == TYPE_UNSURE


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["movement"], TYPE_NON_VOCAL),
        (["rustling"], TYPE_NON_VOCAL),
        (["paper_rustling"], TYPE_NON_VOCAL),
        (["footsteps"], TYPE_NON_VOCAL),
        (["heavy_rain"], TYPE_NON_VOCAL),
        (["heavy_machinery"], TYPE_NON_VOCAL),
        (["door_slam"], TYPE_NON_VOCAL),
        (["object_handling"], TYPE_NON_VOCAL),
        (["coughing"], TYPE_NON_SEMANTIC_VOCAL),
        (["slurping"], TYPE_NON_SEMANTIC_VOCAL),
        (["exhalation"], TYPE_NON_SEMANTIC_VOCAL),
        (["mouth_sound"], TYPE_NON_SEMANTIC_VOCAL),
    ],
)
def test_sources_the_tables_used_to_miss_are_now_evidence(
    flags: list[str], expected: str
) -> None:
    """55 surface forms matched no table and were silently discarded.

    Many named a source perfectly clearly. Dropping them starved non_vocal,
    which is the opposite error from the `non_speech` one and had to be fixed
    in the same pass or the class would look rarer than it is.
    """
    assert resolve_drop_subtype(flags) == expected


@pytest.mark.parametrize(
    "flags",
    [
        ["breathing", "non_speech"],
        ["moaning", "non_speech"],
        ["non_speech", "laughter"],
        ["kiss_sound", "no_speech"],
    ],
)
def test_a_vocal_cue_beside_non_speech_types_the_span_vocal(
    flags: list[str],
) -> None:
    assert resolve_drop_subtype(flags) == TYPE_NON_SEMANTIC_VOCAL


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["breathing"], TYPE_NON_SEMANTIC_VOCAL),
        (["moaning"], TYPE_NON_SEMANTIC_VOCAL),
        (["non_verbal"], TYPE_NON_SEMANTIC_VOCAL),
        (["crying"], TYPE_NON_SEMANTIC_VOCAL),
        (["music"], TYPE_NON_VOCAL),
        (["silence"], TYPE_NON_VOCAL),
        (["water_noise"], TYPE_NON_VOCAL),
        (["ambience"], TYPE_NON_VOCAL),
        ([], TYPE_UNSURE),
        (["speech_fragment"], TYPE_UNSURE),
    ],
)
def test_unrelated_flags_are_unchanged(flags: list[str], expected: str) -> None:
    assert resolve_drop_subtype(flags) == expected


def test_mixed_vocal_and_non_vocal_still_prefers_vocal() -> None:
    """Unchanged policy: a human cue present means the vocal subtype is safer."""
    assert resolve_drop_subtype(["breathing", "music"]) == TYPE_NON_SEMANTIC_VOCAL


def test_non_lexical_stems_are_not_reachable_by_the_other_tables() -> None:
    """No stem list may claim a bare `non_speech`, or ordering would be moot."""
    for flag in NON_LEXICAL_STEMS:
        assert not any(stem in flag for stem in NON_VOCAL_STEMS)
        assert not any(stem in flag for stem in NON_SEMANTIC_VOCAL_STEMS)


def test_non_speech_no_longer_appears_in_the_non_vocal_table() -> None:
    assert "non_speech" not in NON_VOCAL_STEMS
    assert "no_speech" not in NON_VOCAL_STEMS
