"""Real timestamps must be real, or must say they are not.

`word_timestamps_real` has been `False` on every branch since the field existed,
so nothing downstream has ever been exercised against a `True`. Two properties
matter more than the happy path:

  * the flag has to track reality. `src/subtitles/writer.py::_word_start_anchor`
    keys off `timestamp_kind` to decide whether a word may pull a subtitle's
    start earlier, so mislabelling synthetic times as measured would let
    proportional guesses move real subtitle boundaries.
  * a text/span mismatch must degrade to synthetic rather than mapping
    timestamps onto the wrong characters, which would be undetectable
    downstream and would look like an alignment quality problem.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr import subtitle_timing  # noqa: E402
from asr.alignment import CharSpan  # noqa: E402


def _spans(text: str, *, step: float = 0.2, origin: float = 0.0) -> list[CharSpan]:
    """Evenly paced characters, the shape a clean alignment produces."""
    return [
        CharSpan(
            char=char,
            index=index,
            start_frame=index,
            end_frame=index + 1,
            start_s=origin + index * step,
            end_s=origin + index * step + step,
            score=-0.2,
        )
        for index, char in enumerate(text)
    ]


class TestFlagHonesty:
    def test_aligned_words_are_marked_measured(self) -> None:
        text = "こんにちは"
        words, mode, meta = subtitle_timing.build_aligned_word_timestamps(
            text, _spans(text), 0.0, 5.0
        )
        assert mode == "ctc_forced_alignment"
        assert meta["word_timestamps_real"] is True
        assert meta["timing_source"] == subtitle_timing.ALIGNED_TIMING_SOURCE
        assert all(
            word["timestamp_kind"] == subtitle_timing.ALIGNED_TIMESTAMP_KIND
            for word in words
        )

    def test_the_writer_accepts_aligned_words_as_anchors(self) -> None:
        """The seam that makes real timestamps matter at all."""
        from subtitles.writer import _word_start_anchor

        text = "こんにちは"
        words, _, _ = subtitle_timing.build_aligned_word_timestamps(
            text, _spans(text, origin=3.0), 0.0, 10.0
        )
        assert _word_start_anchor(words) == pytest.approx(3.0)

        synthetic, _, _ = subtitle_timing.build_boundary_word_timestamps(
            text, 3.0, 10.0
        )
        assert _word_start_anchor(synthetic) is None

    def test_the_proportional_path_still_declares_itself_synthetic(self) -> None:
        words, mode, meta = subtitle_timing.build_boundary_word_timestamps(
            "ABC", 2.0, 5.0
        )
        assert mode == "boundary_proportional"
        assert meta["word_timestamps_real"] is False
        assert all(
            word["timestamp_kind"] == subtitle_timing.SYNTHETIC_TIMESTAMP_KIND
            for word in words
        )


class TestFallback:
    def test_a_span_count_mismatch_falls_back_instead_of_misaligning(self) -> None:
        text = "こんにちは"
        words, mode, meta = subtitle_timing.build_aligned_word_timestamps(
            text, _spans("こんに"), 0.0, 5.0
        )
        assert mode == "boundary_proportional"
        assert meta["word_timestamps_real"] is False
        assert meta["alignment_fallback_reason"] == "char_span_count_mismatch"
        assert words and words[0]["timestamp_kind"] == (
            subtitle_timing.SYNTHETIC_TIMESTAMP_KIND
        )

    def test_no_spans_falls_back(self) -> None:
        _, mode, meta = subtitle_timing.build_aligned_word_timestamps(
            "こんにちは", [], 0.0, 5.0
        )
        assert mode == "boundary_proportional"
        assert meta["word_timestamps_real"] is False

    def test_empty_text_reports_empty_not_measured(self) -> None:
        words, mode, meta = subtitle_timing.build_aligned_word_timestamps(
            "   ", [], 0.0, 5.0
        )
        assert (words, mode) == ([], "empty")
        assert meta["word_timestamps_real"] is False


class TestTokenMapping:
    def test_space_separated_tokens_take_their_own_characters(self) -> None:
        """Whitespace is dropped before alignment, so indices must skip it."""
        text = "AB CD"
        spans = _spans("ABCD", step=1.0)
        words, _, meta = subtitle_timing.build_aligned_word_timestamps(
            text, spans, 0.0, 10.0
        )
        assert [word["word"] for word in words] == ["AB", "CD"]
        assert words[0]["start"] == pytest.approx(0.0)
        assert words[0]["end"] == pytest.approx(2.0)
        assert words[1]["start"] == pytest.approx(2.0)
        assert words[1]["end"] == pytest.approx(4.0)
        assert meta["aligned_characters"] == 4

    def test_japanese_keeps_the_granularity_that_was_measured(self) -> None:
        """One token per segment would discard the per-character alignment.

        `_tokenize`'s `\\S+` rule makes a spaceless line a single token, so
        without re-splitting a 16 s segment would emit one 16 s "word" and the
        measured character times would be thrown away at the last step.
        """
        text = "こんにちは"
        words, _, _ = subtitle_timing.build_aligned_word_timestamps(
            text, _spans(text, step=0.5, origin=10.0), 0.0, 30.0
        )
        assert [word["word"] for word in words] == list(text)
        assert words[0]["start"] == pytest.approx(10.0)
        assert words[0]["end"] == pytest.approx(10.5)
        assert words[-1]["start"] == pytest.approx(12.0)
        assert words[-1]["end"] == pytest.approx(12.5)

    def test_latin_runs_stay_whole_words(self) -> None:
        """Splitting these per letter would report letter times as word times."""
        text = "OK です"
        words, _, _ = subtitle_timing.build_aligned_word_timestamps(
            text, _spans("OKです", step=1.0), 0.0, 10.0
        )
        assert [word["word"] for word in words] == ["OK", "で", "す"]
        assert words[0]["start"] == pytest.approx(0.0)
        assert words[0]["end"] == pytest.approx(2.0)

    def test_alignment_score_is_carried_for_the_post_gate(self) -> None:
        text = "こんにちは"
        words, _, meta = subtitle_timing.build_aligned_word_timestamps(
            text, _spans(text), 0.0, 5.0
        )
        assert words[0]["alignment_score"] == pytest.approx(-0.2)
        assert meta["alignment_score"] == pytest.approx(-0.2)


class TestSeamOutlierTrimming:
    def test_a_dragged_first_character_does_not_set_the_line_start(self) -> None:
        """The measured Phase 1 failure mode, as a test.

        One character pulled a second into the preceding audio must not drag
        the subtitle with it; the other characters agree on where speech began.
        """
        text = "あいうえお"
        spans = _spans(text, step=0.2, origin=5.0)
        dragged = [
            CharSpan(**{**spans[0].__dict__, "start_s": 4.0, "end_s": 4.2}),
            *spans[1:],
        ]
        words, _, meta = subtitle_timing.build_aligned_word_timestamps(
            text, dragged, 0.0, 10.0
        )
        assert min(word["start"] for word in words) == pytest.approx(5.0, abs=1e-6)
        assert meta["boundary_trimmed"] is True

    def test_a_dragged_last_character_does_not_set_the_line_end(self) -> None:
        text = "あいうえお"
        spans = _spans(text, step=0.2, origin=5.0)
        dragged = [
            *spans[:-1],
            CharSpan(**{**spans[-1].__dict__, "start_s": 9.0, "end_s": 9.5}),
        ]
        words, _, meta = subtitle_timing.build_aligned_word_timestamps(
            text, dragged, 0.0, 10.0
        )
        assert max(word["end"] for word in words) == pytest.approx(6.0, abs=1e-6)
        assert meta["boundary_trimmed"] is True

    def test_evenly_paced_characters_are_left_alone(self) -> None:
        """Trimming must not fire on ordinary speech; interior stays untouched."""
        text = "あいうえお"
        words, _, meta = subtitle_timing.build_aligned_word_timestamps(
            text, _spans(text, step=0.2, origin=5.0), 0.0, 10.0
        )
        assert words[0]["start"] == pytest.approx(5.0)
        assert words[-1]["end"] == pytest.approx(6.0)
        assert meta["boundary_trimmed"] is False

    def test_a_natural_pause_is_not_mistaken_for_a_dragged_character(self) -> None:
        """A real breath inside a line is a gap, but not a seam outlier."""
        text = "あいうえお"
        spans = _spans(text, step=0.2, origin=5.0)
        paused = [
            *spans[:2],
            *[
                CharSpan(
                    **{
                        **span.__dict__,
                        "start_s": span.start_s + 0.5,
                        "end_s": span.end_s + 0.5,
                    }
                )
                for span in spans[2:]
            ],
        ]
        words, _, meta = subtitle_timing.build_aligned_word_timestamps(
            text, paused, 0.0, 10.0
        )
        assert words[0]["start"] == pytest.approx(5.0)
        assert meta["boundary_trimmed"] is False
        # The pause itself must survive into the words, since splitting lines on
        # measured pauses is the whole point of having real times.
        assert words[2]["start"] - words[1]["end"] == pytest.approx(0.5)

    def test_short_texts_have_no_gap_distribution_to_judge_against(self) -> None:
        for text in ("あ", "あい"):
            words, _, meta = subtitle_timing.build_aligned_word_timestamps(
                text, _spans(text, step=0.2, origin=5.0), 0.0, 10.0
            )
            assert meta["boundary_trimmed"] is False
            assert words[0]["start"] == pytest.approx(5.0)

    def test_interior_characters_are_never_moved(self) -> None:
        """Trimming clamps the segment edge; it must not smooth the inside."""
        text = "あいうえお"
        spans = _spans(text, step=0.2, origin=5.0)
        dragged = [
            CharSpan(**{**spans[0].__dict__, "start_s": 4.0, "end_s": 4.2}),
            *spans[1:],
        ]
        words, _, _ = subtitle_timing.build_aligned_word_timestamps(
            text, dragged, 0.0, 10.0
        )
        for index, word in enumerate(words[1:], start=1):
            assert word["start"] == pytest.approx(spans[index].start_s)
            assert word["end"] == pytest.approx(spans[index].end_s)


class TestAcousticExtent:
    """Line edges come from the extent, per-character times from the spans.

    The blind listening pass on 2026-07-31 found the predicted start lands
    inside the first sound (48.1% heard as chopped against a 3.3% floor), so the
    first word's start has to be allowed to move out to the acoustic boundary.
    What these tests protect is the "out, and only out, and only so far" part.
    """

    def test_the_outer_edges_move_out_to_the_extent(self) -> None:
        spans = _spans("あいうえお", step=0.2, origin=2.0)
        words, mode, meta = subtitle_timing.build_aligned_word_timestamps(
            "あいうえお", spans, 0.0, 10.0, (1.85, 3.15)
        )
        assert mode == "ctc_forced_alignment"
        assert meta["boundary_edged"] is True
        assert words[0]["start"] == pytest.approx(1.85)
        assert words[-1]["end"] == pytest.approx(3.15)

    def test_interior_characters_are_untouched(self) -> None:
        """The measurement says the body is right; only the seam was inset."""
        spans = _spans("あいうえお", step=0.2, origin=2.0)
        plain, _, _ = subtitle_timing.build_aligned_word_timestamps(
            "あいうえお", spans, 0.0, 10.0
        )
        edged, _, _ = subtitle_timing.build_aligned_word_timestamps(
            "あいうえお", spans, 0.0, 10.0, (1.85, 3.15)
        )
        assert [w["start"] for w in plain[1:]] == [w["start"] for w in edged[1:]]
        assert [w["end"] for w in plain[:-1]] == [w["end"] for w in edged[:-1]]

    def test_an_extent_inside_the_spans_changes_nothing(self) -> None:
        """One-directional by construction.

        `speech_extent` cannot return a narrower extent, but a caller passing
        one must not be able to shrink a line onto times no measurement
        supports - that would be the seam-trim rule silently overridden.
        """
        spans = _spans("あいうえお", step=0.2, origin=2.0)
        plain, _, meta_plain = subtitle_timing.build_aligned_word_timestamps(
            "あいうえお", spans, 0.0, 10.0
        )
        narrowed, _, meta = subtitle_timing.build_aligned_word_timestamps(
            "あいうえお", spans, 0.0, 10.0, (2.4, 2.6)
        )
        assert meta["boundary_edged"] is False
        assert [w["start"] for w in plain] == [w["start"] for w in narrowed]
        assert [w["end"] for w in plain] == [w["end"] for w in narrowed]

    def test_the_widening_is_capped(self) -> None:
        """A runaway extent cannot drag a line across a neighbour.

        The walk already stops at non-blank frames, so this is defence in depth
        for the case where the extent arrives from somewhere else - and for the
        case where `_robust_bounds` has just pulled a dragged seam character in
        and the extent would otherwise haul it straight back out.
        """
        from asr import alignment

        spans = _spans("あいうえお", step=0.2, origin=2.0)
        words, _, _ = subtitle_timing.build_aligned_word_timestamps(
            "あいうえお", spans, 0.0, 60.0, (-40.0, 55.0)
        )
        assert words[0]["start"] == pytest.approx(2.0 - alignment.ONSET_BACKOFF_MAX_S)
        assert words[-1]["end"] == pytest.approx(3.0 + alignment.CODA_EXTEND_MAX_S)

    def test_omitting_the_extent_keeps_the_previous_behaviour(self) -> None:
        spans = _spans("あいうえお", step=0.2, origin=2.0)
        words, mode, meta = subtitle_timing.build_aligned_word_timestamps(
            "あいうえお", spans, 0.0, 10.0
        )
        assert meta["boundary_edged"] is False
        assert words[0]["start"] == pytest.approx(2.0)
        assert words[-1]["end"] == pytest.approx(3.0)
        assert mode == "ctc_forced_alignment"

    def test_a_span_mismatch_still_falls_back_to_synthetic(self) -> None:
        """The extent must not create a path around the mismatch guard."""
        words, mode, meta = subtitle_timing.build_aligned_word_timestamps(
            "あいうえお", _spans("あい"), 0.0, 10.0, (0.0, 9.0)
        )
        assert mode == "boundary_proportional"
        assert meta["word_timestamps_real"] is False
        assert meta["alignment_fallback_reason"] == "char_span_count_mismatch"
        assert all(w["end"] <= 10.0 for w in words)
