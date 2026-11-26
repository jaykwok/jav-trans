"""CTC targets with the vocalisation stripped out, and the pause reading.

These cover the parts of the 2026-08-11 v2 training chain that decide what the
head is taught, and they are pure text/array logic - no GPU, no encoder, no
checkpoint - so the decisions can be pinned without the machinery around them.

The asymmetry under test throughout: leaving a moan in the target costs a little
supervision, while stripping a *word* teaches the head that speech is silence.
Every case below is written from the side that would do the damage.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.build_vocalisation_stripped_manifest import (  # noqa: E402
    split_parts,
    strip_vocalisation,
)
from tools.align.measure_pause_structure import (  # noqa: E402
    pauses_lost_to_fragmentation,
    runs_of,
)


class TestStripping:
    def test_the_speech_survives_and_the_moaning_does_not(self) -> None:
        kept, keeps, drops = strip_vocalisation(
            "あぅぅ、ごめんなさい。わたしが兄さんに甘えたことで、こんなことに"
        )
        assert kept == "ごめんなさい。わたしが兄さんに甘えたことで、こんなことに"
        assert (keeps, drops) == (3, 1)

    def test_a_part_that_cannot_be_decomposed_is_kept(self) -> None:
        """The classifier's whole design is that anything it cannot account for
        is speech. A missed moan is cheap; a stripped word is not."""
        kept, _, drops = strip_vocalisation("むにゃむにゃ、ふあぁ")
        assert kept == "むにゃむにゃ"
        assert drops == 1

    def test_a_protected_reply_is_not_stripped_even_spelled_like_a_moan(self) -> None:
        """`うん` and `んん` differ by meaning, not by sound class."""
        kept, _, drops = strip_vocalisation("あぁ、うん、はぁ")
        assert kept == "うん"
        assert drops == 2

    def test_kanji_is_never_stripped(self) -> None:
        for text in ("奥を突かれながら", "はぁ、奥", "1000円"):
            kept, _, _ = strip_vocalisation(text)
            assert "奥" in kept or "1000" in kept

    def test_an_all_vocalisation_line_strips_to_nothing(self) -> None:
        """Which is the signal to route the clip to the blank library instead of
        keeping a target that would teach the head to align moaning."""
        kept, keeps, drops = strip_vocalisation("ふあぁ、んっ、ああっ、ああぁぁっ!")
        assert kept == ""
        assert (keeps, drops) == (0, 4)

    def test_split_parts_can_rebuild_the_original(self) -> None:
        text = "あぅ…やっぱりダメですかね！"
        assert "".join(part for part, _ in split_parts(text)) == text

    def test_leading_separators_do_not_survive_a_stripped_head(self) -> None:
        """A removed first part used to leave its punctuation stranded at the
        front, which an unpunctuated vocab then had to encode as nothing."""
        kept, _, _ = strip_vocalisation("んんっ…ごめんなさい、ママ")
        assert kept.startswith("ごめんなさい")


class TestPauseFragmentation:
    """Whether a punctuation frame hides a pause from the chunker.

    Counting runs on each side answers nothing: merging two already-cuttable
    runs lowers the count without losing anything, and on a real film that
    effect is larger than the one being looked for. The question has to be asked
    per silence.
    """

    @staticmethod
    def _mask(pattern: str):
        return [char == "1" for char in pattern]

    def test_runs_are_half_open_frame_spans(self) -> None:
        assert runs_of(self._mask("0110011100")) == [(1, 3), (5, 8)]

    def test_a_silence_split_below_the_floor_is_counted_as_lost(self) -> None:
        # Ten frames of silence with one punctuation frame in the middle: as
        # blank runs it is 5 + 4, both under a 6-frame floor.
        strict = [(0, 5), (6, 10)]
        lenient = [(0, 10)]
        lost, seconds = pauses_lost_to_fragmentation(strict, lenient, 0.1, 0.6)
        assert lost == 1
        assert seconds == pytest.approx(1.0)

    def test_a_silence_still_cuttable_after_the_split_is_not_lost(self) -> None:
        strict = [(0, 7), (8, 12)]
        lenient = [(0, 12)]
        lost, _ = pauses_lost_to_fragmentation(strict, lenient, 0.1, 0.6)
        assert lost == 0

    def test_merging_two_cuttable_runs_is_not_reported_as_a_loss(self) -> None:
        """The confound that makes the naive count metric useless."""
        strict = [(0, 8), (9, 20)]
        lenient = [(0, 20)]
        lost, _ = pauses_lost_to_fragmentation(strict, lenient, 0.1, 0.6)
        assert lost == 0

    def test_an_acoustic_only_head_reports_nothing_by_construction(self) -> None:
        """With no punctuation classes the two readings are the same array, so
        this metric is identically zero - which is what makes it the control."""
        spans = [(0, 8), (12, 30)]
        lost, seconds = pauses_lost_to_fragmentation(spans, spans, 0.1, 0.6)
        assert (lost, seconds) == (0, 0.0)
