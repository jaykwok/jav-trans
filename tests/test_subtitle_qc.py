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
    assert report["alignment_fallback_ratio"] == pytest.approx(0.3)
    assert any("alignment_fallback_ratio" in w for w in report["warnings"])


def test_short_segment_ratio():
    segs = [_seg("ア", "甲", 0.0, 0.5)] * 3 + [_seg("イ", "乙", 0.0, 1.5)] * 7
    report = compute_quality_report(segs, 60.0, [], 0, 10)
    assert report["short_segment_ratio"] == pytest.approx(0.3)
    assert any("short_segment_ratio" in w for w in report["warnings"])


def test_empty_segments_returns_zeros():
    report = compute_quality_report([], 60.0, [], 0, 0)
    assert report["empty_zh_ratio"] == 0.0
    assert report["f0_filtered_count"] == 0
    assert report["warnings"] == []


def test_gender_ratios_present_when_field_populated():
    segs = [
        {**_seg("ア", "甲"), "gender": "M"},
        {**_seg("イ", "乙"), "gender": "F"},
        {**_seg("ウ", "丙"), "gender": "F"},
        {**_seg("エ", "丁"), "gender": None},
    ]
    report = compute_quality_report(segs, 60.0, [], 0, 4)
    assert report["male_ratio"] == pytest.approx(1 / 4)
    assert report["female_ratio"] == pytest.approx(2 / 4)
    assert report["gender_none_ratio"] == pytest.approx(1 / 4)


def test_gender_ratios_absent_without_field():
    segs = [_seg("ア", "甲"), _seg("イ", "乙")]
    report = compute_quality_report(segs, 60.0, [], 0, 2)
    assert "male_ratio" not in report
    assert "female_ratio" not in report
    assert "gender_none_ratio" not in report


def test_f0_filtered_count_is_reported():
    segs = [_seg("ア", "甲"), _seg("イ", "乙")]
    report = compute_quality_report(segs, 60.0, [], 0, 2, f0_filtered_count=3)
    assert report["f0_filtered_count"] == 3

