"""Punctuation must survive the head, whichever vocabulary the head has.

Two decisions that are each right on their own used to combine into a silent
data loss. `alignment._spans_for_full_text` gives an unpronounced character a
zero-width span on purpose - a comma occupies no audio, and giving it width
would take that width from a character that was spoken. `normalize_word_dicts`
dropped zero-width words, on the assumption that they only ever came from
floating-point drift in proportional timing.

With a punctuated vocabulary the two never met: punctuation is a real class, so
forced alignment gives it a real span. Switching to an acoustic-only head made
every comma and ellipsis zero-width, and they were then dropped - which deletes
them from the subtitle outright, because `transcribe._group_words_to_segments`
rebuilds segment text by joining these tokens. On a real film that removed 100%
of `、` and `…`, and the knock-on was worse than the missing marks: the text
glued into one run, so the vocalisation filter could no longer decompose a cue
and moaning stayed on screen.

The second half covers the other side of the same tensor: a punctuation frame in
the middle of a silence must not end a blank run, or the chunker loses the pause.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from asr.alignment import BLANK_INDEX, blank_runs  # noqa: E402
from asr.local_backend import normalize_word_dicts  # noqa: E402


class TestZeroWidthWords:
    def test_a_zero_width_comma_is_kept(self) -> None:
        """It is zero-width by design, not by accident."""
        words = [
            {"word": "はい", "start": 1.0, "end": 1.4},
            {"word": "、", "start": 1.4, "end": 1.4},
            {"word": "そう", "start": 1.5, "end": 1.9},
        ]
        assert [w["word"] for w in normalize_word_dicts(words)] == ["はい", "、", "そう"]

    def test_a_zero_width_spoken_word_is_still_dropped(self) -> None:
        """That one really is degenerate - it claims a character with no time."""
        words = [
            {"word": "はい", "start": 1.0, "end": 1.4},
            {"word": "そう", "start": 1.5, "end": 1.5},
        ]
        assert [w["word"] for w in normalize_word_dicts(words)] == ["はい"]

    def test_an_inverted_punctuation_span_is_flattened_not_dropped(self) -> None:
        kept = normalize_word_dicts([{"word": "…", "start": 2.0, "end": 1.0}])
        assert [w["word"] for w in kept] == ["…"]
        assert kept[0]["start"] == kept[0]["end"] == 2.0

    def test_the_rebuilt_text_still_reads_as_japanese(self) -> None:
        """What the subtitle layer actually consumes is this join."""
        words = [
            {"word": "来て", "start": 0.0, "end": 0.4},
            {"word": "しまった", "start": 0.4, "end": 0.9},
            {"word": "…", "start": 0.9, "end": 0.9},
            {"word": "ね", "start": 1.0, "end": 1.2},
            {"word": "、", "start": 1.2, "end": 1.2},
        ]
        assert "".join(w["word"] for w in normalize_word_dicts(words)) == "来てしまった…ね、"

    def test_a_positive_span_is_untouched(self) -> None:
        words = [{"word": "、", "start": 1.0, "end": 1.2}]
        assert normalize_word_dicts(words) == [{"start": 1.0, "end": 1.2, "word": "、"}]


class FakeLogProbs:
    """Just enough tensor surface for `blank_runs`."""

    def __init__(self, argmax: list[int]) -> None:
        self._argmax = argmax

    def argmax(self, dim=-1):  # noqa: ARG002
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._argmax


class TestBlankRuns:
    @staticmethod
    def runs(tokens: list[int], **kwargs):
        # upsample=1 so one frame is one ENCODER_FRAME_S and the maths is legible
        return blank_runs(FakeLogProbs(tokens), upsample=1, **kwargs)

    def test_a_punctuation_frame_splits_the_pause_by_default(self) -> None:
        """The behaviour every existing caller still gets."""
        tokens = [BLANK_INDEX] * 5 + [7] + [BLANK_INDEX] * 5
        assert len(self.runs(tokens)) == 2

    def test_declaring_it_silent_keeps_the_pause_whole(self) -> None:
        tokens = [BLANK_INDEX] * 5 + [7] + [BLANK_INDEX] * 5
        assert len(self.runs(tokens, silent_classes=frozenset({7}))) == 1

    def test_a_spoken_class_still_ends_the_run(self) -> None:
        tokens = [BLANK_INDEX] * 5 + [9] + [BLANK_INDEX] * 5
        assert len(self.runs(tokens, silent_classes=frozenset({7}))) == 2

    def test_an_empty_silent_set_changes_nothing(self) -> None:
        """The acoustic-only case: no punctuation classes exist to declare."""
        tokens = [BLANK_INDEX] * 4 + [9] + [BLANK_INDEX] * 4
        assert self.runs(tokens, silent_classes=frozenset()) == self.runs(tokens)

    def test_the_recovered_pause_is_long_enough_to_be_a_cut_point(self) -> None:
        """The point of the change: min_seconds is applied after the join, so a
        silence that was two sub-floor fragments becomes one usable pause."""
        tokens = [BLANK_INDEX] * 5 + [7] + [BLANK_INDEX] * 5
        floor = 8 / 13.0  # eight encoder frames
        assert self.runs(tokens, min_seconds=floor) == []
        assert len(self.runs(tokens, min_seconds=floor, silent_classes=frozenset({7}))) == 1

    def test_punctuation_alone_is_not_a_pause_boundary_artefact(self) -> None:
        """A run made only of punctuation frames is still silence."""
        tokens = [9] + [7] * 6 + [9]
        assert len(self.runs(tokens, silent_classes=frozenset({7}))) == 1
