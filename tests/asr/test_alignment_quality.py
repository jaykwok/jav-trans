from __future__ import annotations

from asr.alignment_quality import classify_alignment_quality


def test_classifies_boundary_timing_as_normal():
    result = classify_alignment_quality(
        text="こんにちは",
        duration_s=1.2,
        alignment_mode="boundary_proportional",
        aligned_segment_count=1,
        word_stats={"word_count": 3, "zero_or_negative_count": 0},
    )

    assert result == {
        "alignment_quality": "boundary",
        "alignment_issue_type": "none",
        "alignment_issue_subtype": "none",
        "alignment_quality_reasons": [],
    }
    assert "fallback_subtype" not in result


def test_classifies_nonlexical_boundary_text():
    result = classify_alignment_quality(
        text="…、…",
        duration_s=3.0,
        nonlexical_text=True,
        alignment_mode="boundary_proportional",
        aligned_segment_count=1,
        word_stats={"word_count": 3},
    )

    assert result == {
        "alignment_quality": "nonlexical",
        "alignment_issue_type": "none",
        "alignment_issue_subtype": "nonlexical_text",
        "alignment_quality_reasons": ["nonlexical_text"],
    }


def test_review_for_text_without_timing_words():
    result = classify_alignment_quality(
        text="こんにちは",
        duration_s=1.2,
        alignment_mode="boundary_proportional",
        aligned_segment_count=0,
        word_stats={"word_count": 0},
    )

    assert result["alignment_quality"] == "drop_or_review"
    assert result["alignment_issue_type"] == "none"
    assert result["alignment_issue_subtype"] == "text_without_output_segment"
    assert result["alignment_quality_reasons"] == ["text_without_output_segment"]


def test_unknown_timing_mode_is_partial():
    result = classify_alignment_quality(
        text="こんにちは",
        duration_s=1.2,
        alignment_mode="unknown_mode",
        aligned_segment_count=1,
        word_stats={"word_count": 1},
    )

    assert result["alignment_quality"] == "partial"
    assert result["alignment_issue_type"] == "unknown"
    assert result["alignment_issue_subtype"] == "unknown_alignment_mode"
