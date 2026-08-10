from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.audit_teacher_silence_against_head import (  # noqa: E402
    acoustic,
    evaluate_film,
    head_speech_spans,
    merge,
    overlap_s,
    silent_runs,
)


def _word(start: float, end: float, text: str = "あ", quality: str = "aligned") -> dict:
    return {
        "start": start,
        "end": end,
        "word": text,
        "timestamp_kind": "ctc_forced_alignment",
        "alignment_quality": quality,
    }


def test_punctuation_is_not_evidence_of_speech():
    # CTC assigns a frame to `。` like any other target, but a frame of
    # punctuation must not count as the head having heard something.
    spans = head_speech_spans(
        [{"words": [_word(0.0, 0.1, "。"), _word(1.0, 1.2, "あ")]}]
    )

    assert spans == [(1.0, 1.2)]


def test_synthetic_timestamps_are_ignored():
    spans = head_speech_spans(
        [
            {
                "words": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "word": "あ",
                        "timestamp_kind": "synthetic_proportional",
                    },
                    _word(6.0, 6.2),
                ]
            }
        ]
    )

    assert spans == [(6.0, 6.2)]


def test_low_confidence_words_are_excluded_by_default_and_optional_otherwise():
    segments = [{"words": [_word(0.0, 0.2), _word(1.0, 1.2, quality="degraded")]}]

    assert head_speech_spans(segments) == [(0.0, 0.2)]
    assert head_speech_spans(segments, confident_only=False) == [
        (0.0, 0.2),
        (1.0, 1.2),
    ]


def test_window_clips_spans_for_a_prefix_only_run():
    segments = [{"words": [_word(0.0, 0.2), _word(80.0, 95.0)]}]

    assert head_speech_spans(segments, window_s=90.0) == [(0.0, 0.2), (80.0, 90.0)]


def test_silent_runs_respect_the_boundary_margin_and_minimum():
    runs = silent_runs(
        [(10.0, 11.0)], total_s=20.0, margin_s=0.5, minimum_s=0.8
    )

    # The margin shrinks both gaps; both still clear the minimum.
    assert runs == [(0.0, 9.5), (11.5, 20.0)]

    # A gap under the minimum is not offered at all.
    assert silent_runs([(0.0, 9.6), (10.0, 20.0)], total_s=20.0, margin_s=0.0, minimum_s=0.8) == []


def test_overlap_counts_only_the_intersection():
    assert overlap_s([(0.0, 10.0)], [(2.0, 4.0), (6.0, 7.0)]) == pytest.approx(3.0)
    assert overlap_s([(0.0, 1.0)], [(5.0, 6.0)]) == pytest.approx(0.0)


def test_a_film_the_teacher_mostly_missed_is_rejected():
    # The teacher heard 2 s of a 100 s film; the head found 40 s of speech, and
    # nearly all of it falls inside what the blank rule would claim.
    result = evaluate_film(
        teacher_words=[(0.0, 2.0)],
        head_spans=[(0.0, 2.0), (20.0, 58.0)],
        duration_s=100.0,
        window_s=None,
        merge_gap_s=0.15,
        boundary_ignore_s=0.1,
        minimum_blank_s=0.8,
        max_swallowed_share=0.10,
    )

    assert result["admitted_as_blank_source"] is False
    assert result["rejection_reason"] == "teacher_silence_swallows_head_speech"
    assert result["head_speech_swallowed_share"] == pytest.approx(38.0 / 40.0, rel=1e-3)
    # The share people reach for first stays small, which is exactly why it
    # cannot be the gate.
    assert result["disputed_share_of_blank"] < 0.40


def test_a_film_the_two_readings_agree_on_is_admitted():
    result = evaluate_film(
        teacher_words=[(0.0, 40.0)],
        head_spans=[(0.0, 39.0)],
        duration_s=100.0,
        window_s=None,
        merge_gap_s=0.15,
        boundary_ignore_s=0.1,
        minimum_blank_s=0.8,
        max_swallowed_share=0.10,
    )

    assert result["admitted_as_blank_source"] is True
    assert result["rejection_reason"] == ""
    assert result["proposed_blank_s"] == pytest.approx(59.9, rel=1e-3)


def test_a_head_that_found_nothing_cannot_certify_a_film():
    # No speech on the reference side is not agreement; there is nothing to
    # check the teacher against, so the film must not be admitted by default.
    result = evaluate_film(
        teacher_words=[(0.0, 1.0)],
        head_spans=[],
        duration_s=100.0,
        window_s=None,
        merge_gap_s=0.15,
        boundary_ignore_s=0.1,
        minimum_blank_s=0.8,
        max_swallowed_share=0.10,
    )

    assert result["admitted_as_blank_source"] is False
    assert result["rejection_reason"] == "head_found_no_speech_to_check_against"
    assert result["head_speech_swallowed_share"] is None


def test_merge_joins_within_the_gap_only():
    assert merge([(0.0, 1.0), (1.1, 2.0)], gap_s=0.15) == [(0.0, 2.0)]
    assert merge([(0.0, 1.0), (1.3, 2.0)], gap_s=0.15) == [(0.0, 1.0), (1.3, 2.0)]


def test_acoustic_matches_the_head_vocabulary_rule():
    assert acoustic("あ、A1。") == "あA1"
    assert acoustic("……！？") == ""
