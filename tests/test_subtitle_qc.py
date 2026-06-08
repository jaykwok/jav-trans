import pytest
from subtitles.qc import compute_quality_report


def _seg(ja: str, zh: str, start: float = 0.0, end: float = 1.0) -> dict:
    return {"text": ja, "zh": zh, "start": start, "end": end}


def test_empty_zh_triggers_warning():
    segs = [_seg("テスト", "测试")] * 8 + [_seg("テスト", ""), _seg("テスト", "")]
    report = compute_quality_report(segs, 60.0, [], 0, 10)
    assert report["empty_zh_ratio"] == pytest.approx(0.2)
    assert any("empty_zh_ratio" in w for w in report["warnings"])


def test_repetition_ratio_correct():
    segs = [
        _seg("ア", "甲"),
        _seg("イ", "乙"),
        _seg("ウ", "乙"),
        _seg("エ", "乙"),
        _seg("オ", "丙"),
    ]
    report = compute_quality_report(segs, 60.0, [], 0, 5)
    assert report["repetition_ratio"] == pytest.approx(2 / 5)


def test_kana_only_detection():
    segs = [
        _seg("あああ", "啊"),
        _seg("テスト", "测试"),
        _seg("はい", "是的"),
        _seg("漢字", ""),
    ]
    report = compute_quality_report(segs, 60.0, [], 0, 3)
    assert report["kana_only_ratio"] == pytest.approx(4 / 4)


def test_glossary_hit_rate_bilateral():
    segs = [
        _seg("ちんぽが", "肉棒啊"),
        _seg("ちんぽだよ", "这是什么"),  # ja hit but zh miss
        _seg("普通の文", "普通的文字"),
    ]
    pairs = [("ちんぽ", "肉棒")]
    report = compute_quality_report(segs, 60.0, pairs, 0, 3)
    # 2 ja hits, 1 zh hit -> 0.5
    assert report["glossary_hit_rate"] == pytest.approx(0.5)
    assert any("glossary_hit_rate" in w for w in report["warnings"])


def test_glossary_empty_returns_null():
    segs = [_seg("テスト", "测试")]
    report = compute_quality_report(segs, 60.0, [], 0, 1)
    assert report["glossary_hit_rate"] is None
    assert not any("glossary_hit_rate" in w for w in report["warnings"])


def test_alignment_fallback_ratio():
    segs = [_seg("テスト", "测试")] * 10
    report = compute_quality_report(segs, 60.0, [], 3, 10)
    assert report["alignment_fallback_count"] == 3
    assert report["alignment_fallback_total"] == 10
    assert report["alignment_fallback_ratio"] == pytest.approx(0.3)
    assert any("alignment_fallback_ratio" in w for w in report["warnings"])


def test_short_segment_ratio():
    segs = [_seg("ア", "甲", 0.0, 0.5)] * 3 + [_seg("イ", "乙", 0.0, 1.5)] * 7
    report = compute_quality_report(segs, 60.0, [], 0, 10)
    assert report["short_segment_ratio"] == pytest.approx(0.3)
    assert report["short_segment_count"] == 3
    assert report["micro_segment_count"] == 0
    assert report["long_segment_count"] == 0
    assert report["subtitle_duration_p50_s"] == pytest.approx(1.5)
    assert report["subtitle_duration_p90_s"] == pytest.approx(1.5)
    assert report["subtitle_duration_max_s"] == pytest.approx(1.5)
    assert any("short_segment_ratio" in w for w in report["warnings"])


def test_empty_segments_returns_zeros():
    report = compute_quality_report([], 60.0, [], 0, 0)
    assert report["empty_zh_ratio"] == 0.0
    assert report["subtitle_overlap_count"] == 0
    assert report["subtitle_duration_p50_s"] == 0.0
    assert report["short_segment_count"] == 0
    assert report["warnings"] == []


def test_legacy_acoustic_metadata_does_not_create_role_metrics():
    segs = [_seg("ア", "甲"), _seg("イ", "乙")]
    segs[0]["source_note"] = "legacy"
    report = compute_quality_report(segs, 60.0, [], 0, 2)
    assert "male_ratio" not in report
    assert "female_ratio" not in report
    assert "role_none_ratio" not in report


def test_asr_generation_errors_are_reported_and_warned():
    segs = [_seg("テスト", "测试")]
    report = compute_quality_report(
        segs,
        60.0,
        [],
        0,
        1,
        asr_qc={
            "generation_error_count": 2,
            "generation_overflow_count": 1,
            "timeout_count": 1,
            "quarantined_count": 1,
            "empty_text_for_speech_count": 1,
            "review_uncertain_count": 3,
            "review_uncertain_items": [{"chunk_index": 7, "original_text": "怪しい"}],
        },
    )

    assert report["asr_generation_error_count"] == 2
    assert report["asr_generation_overflow_count"] == 1
    assert report["asr_timeout_count"] == 1
    assert report["asr_quarantined_count"] == 1
    assert report["asr_empty_text_for_speech_count"] == 1
    assert report["asr_review_uncertain_count"] == 3
    assert report["asr_review_uncertain_items"][0]["original_text"] == "怪しい"
    assert any("asr_generation_error_count" in warning for warning in report["warnings"])
    assert any("asr_generation_overflow_count" in warning for warning in report["warnings"])


def test_empty_segments_keep_asr_qc_counts():
    report = compute_quality_report(
        [],
        60.0,
        [],
        0,
        0,
        asr_qc={"generation_error_count": 1, "generation_overflow_count": 1},
    )

    assert report["asr_generation_error_count"] == 1
    assert report["asr_generation_overflow_count"] == 1
    assert any("asr_generation_error_count" in warning for warning in report["warnings"])
    assert any("asr_generation_overflow_count" in warning for warning in report["warnings"])


def test_subtitle_overlap_stats_warn_when_present():
    report = compute_quality_report(
        [
            _seg("ア", "甲", 0.0, 1.0),
            _seg("イ", "乙", 0.8, 2.0),
        ],
        60.0,
        [],
        0,
        2,
    )

    assert report["subtitle_overlap_count"] == 1
    assert report["subtitle_overlap_total_s"] == pytest.approx(0.2)
    assert report["subtitle_overlap_max_s"] == pytest.approx(0.2)
    assert report["subtitle_overlap_examples"][0]["overlap_s"] == pytest.approx(0.2)
    assert any("subtitle_overlap_count" in warning for warning in report["warnings"])

