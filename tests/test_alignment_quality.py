from __future__ import annotations

from asr.alignment_quality import classify_alignment_quality


def test_classifies_clean_forced_alignment():
    result = classify_alignment_quality(
        text="こんにちは",
        duration_s=1.2,
        align_text_empty=False,
        asr_dropped_uncertain=False,
        alignment_mode="forced_aligner",
        aligned_segment_count=1,
        word_stats={"word_count": 3, "zero_or_negative_count": 0},
    )

    assert result == {
        "alignment_quality": "forced",
        "fallback_type": "none",
        "fallback_subtype": "none",
        "alignment_quality_reasons": [],
    }


def test_classifies_partial_forced_alignment_with_sentinel():
    result = classify_alignment_quality(
        text="こんにちは",
        duration_s=1.2,
        align_text_empty=False,
        asr_dropped_uncertain=False,
        alignment_mode="forced_aligner",
        sentinel_lines=["Alignment 哨兵触发: 时间轴异常"],
        aligned_segment_count=1,
        word_stats={"word_count": 2, "zero_or_negative_count": 2},
        word_failure_reasons=["word_timing_zero_heavy"],
    )

    assert result["alignment_quality"] == "partial"
    assert result["fallback_type"] == "none"
    assert result["fallback_subtype"] == "sentinel_word_timing_zero_heavy"
    assert result["alignment_quality_reasons"] == [
        "alignment_sentinel",
        "word_timing_zero_heavy",
    ]


def test_classifies_vad_and_proportional_fallbacks():
    vad = classify_alignment_quality(
        text="こんにちは",
        duration_s=1.2,
        align_text_empty=False,
        asr_dropped_uncertain=False,
        alignment_mode="aligner_vad_fallback",
        aligned_segment_count=1,
    )
    proportional = classify_alignment_quality(
        text="こんにちは",
        duration_s=1.2,
        align_text_empty=False,
        asr_dropped_uncertain=False,
        alignment_mode="even_fallback",
        fallback_lines=["Alignment 回退: 使用比例时间戳"],
        aligned_segment_count=1,
    )

    assert vad["alignment_quality"] == "vad_coarse"
    assert vad["fallback_type"] == "vad_coarse"
    assert vad["fallback_subtype"] == "vad_coarse"
    assert proportional["alignment_quality"] == "proportional"
    assert proportional["fallback_type"] == "proportional"
    assert proportional["fallback_subtype"] == "proportional"


def test_review_overrides_fallback_quality_for_unalignable_text():
    result = classify_alignment_quality(
        text="~~~♡!!!",
        duration_s=1.2,
        align_text_empty=True,
        asr_dropped_uncertain=False,
        alignment_mode="even_fallback",
        aligned_segment_count=0,
    )

    assert result["alignment_quality"] == "drop_or_review"
    assert result["fallback_type"] == "proportional"
    assert result["fallback_subtype"] == "align_text_empty"
    assert result["alignment_quality_reasons"] == [
        "align_text_empty",
        "text_without_output_segment",
    ]


def test_nonlexical_empty_align_text_is_coarse_not_review():
    result = classify_alignment_quality(
        text="…、…",
        duration_s=3.0,
        align_text_empty=True,
        nonlexical_text=True,
        asr_dropped_uncertain=False,
        alignment_mode="nonlexical",
        fallback_lines=["Alignment VAD 回退语音区间: 1"],
        aligned_segment_count=1,
    )

    assert result == {
        "alignment_quality": "nonlexical",
        "fallback_type": "none",
        "fallback_subtype": "nonlexical_text",
        "alignment_quality_reasons": ["nonlexical_text"],
    }
