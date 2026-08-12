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


def test_alignment_issue_ratio_is_observation_only():
    segs = [_seg("テスト", "测试")] * 10
    report = compute_quality_report(segs, 60.0, [], 3, 10)
    assert report["alignment_issue_count"] == 3
    assert report["alignment_issue_total"] == 10
    assert report["alignment_issue_ratio"] == pytest.approx(0.3)
    assert "alignment_fallback_ratio" not in report
    assert not any("alignment_issue_ratio" in w for w in report["warnings"])


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
        asr_generation={
            "generation_error_count": 2,
            "generation_overflow_count": 1,
            "timeout_count": 1,
            "quarantined_count": 1,
        },
    )

    assert report["asr_generation_error_count"] == 2
    assert report["asr_generation_overflow_count"] == 1
    assert report["asr_timeout_count"] == 1
    assert report["asr_quarantined_count"] == 1
    assert any("asr_generation_error_count" in warning for warning in report["warnings"])
    assert any("asr_generation_overflow_count" in warning for warning in report["warnings"])


def test_empty_segments_keep_asr_generation_counts():
    report = compute_quality_report(
        [],
        60.0,
        [],
        0,
        0,
        asr_generation={"generation_error_count": 1, "generation_overflow_count": 1},
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


def test_subtitle_density_audit_marks_over_4cps_without_warning():
    report = compute_quality_report(
        [
            _seg("あいうえお", "", 0.0, 1.0),
            _seg("はい", "", 1.2, 2.2),
            _seg("ありがとうございます", "", 2.4, 3.4),
        ],
        10.0,
        [],
        0,
        3,
    )

    assert report["subtitle_density_cps_threshold"] == pytest.approx(4.0)
    assert report["subtitle_density_over_4cps_count"] == 2
    assert report["subtitle_density_max_ja_cps"] == pytest.approx(10.0)
    assert report["subtitle_density_p90_ja_cps"] > 4.0
    assert report["subtitle_density_window_10s_max_cue_count"] == 3
    assert report["subtitle_density_window_10s_min_gap_s"] == pytest.approx(0.2)
    assert report["subtitle_density_review_examples"][0]["ja_cps"] == pytest.approx(10.0)
    assert not any("subtitle_density" in warning for warning in report["warnings"])


def test_subtitle_density_below_4cps_not_marked():
    report = compute_quality_report(
        [_seg("あいう", "", 0.0, 1.0), _seg("はい", "", 2.0, 3.0)],
        10.0,
        [],
        0,
        2,
    )

    assert report["subtitle_density_over_4cps_count"] == 0
    assert report["subtitle_density_max_ja_cps"] == pytest.approx(3.0)



def test_spec_compliance_flags_violations():
    segs = [
        # banned punctuation + cps: 22 reading units in 1s
        _seg("ア", "这是一句故意塞满很多字的中文字幕，它明显超速了。", 0.0, 1.0),
        # duration over 7s
        _seg("イ", "太长了", 10.0, 18.0),
        # duration under 5/6s, and gap to the next cue below two frames
        _seg("ウ", "太短", 20.0, 20.5),
        _seg("エ", "紧跟", 20.51, 22.0),
    ]
    report = compute_quality_report(segs, 60.0, [], 0, 4)

    # Rendered output is normalized, so banned marks survive only in raw zh.
    assert report["spec_zh_banned_punct_count"] == 0
    assert report["spec_zh_raw_banned_punct_count"] > 0
    assert report["spec_zh_cps_over_9_count"] >= 1
    assert report["spec_zh_cps_max"] > 9.0
    assert report["spec_duration_over_7s_count"] == 1
    assert report["spec_duration_under_min_count"] == 1
    assert report["spec_gap_under_2frames_count"] == 1
    assert report["spec_cue_count"] == 4
    assert report["spec_duration_under_min_share"] == pytest.approx(0.25)
    assert report["spec_gap_under_2frames_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert any("spec_zh_cps_over_9_count" in w for w in report["warnings"])
    assert any("spec_duration_over_7s_count" in w for w in report["warnings"])
    assert report["spec_review_examples"]


def test_the_two_deviating_timing_rules_warn_on_rate_not_on_count():
    """Layout v3 ends a cue at the last spoken character, so short cues and
    tight gaps are produced by design. A count threshold would fire on every
    film; the rate has to move before it means anything."""
    tight = [
        _seg("ア", "短", 0.0, 0.5),
        _seg("イ", "紧跟", 0.51, 2.0),
        *[_seg("ウ", "正常一句话", 3.0 + index * 3.0, 5.0 + index * 3.0) for index in range(18)],
    ]
    report = compute_quality_report(tight, 120.0, [], 0, len(tight))

    assert report["spec_duration_under_min_count"] == 1
    assert report["spec_gap_under_2frames_count"] == 1
    assert not any("spec_duration_under_min" in w for w in report["warnings"])
    assert not any("spec_gap_under_2frames" in w for w in report["warnings"])


def test_a_rate_above_the_threshold_still_warns(monkeypatch):
    monkeypatch.setenv("QC_MAX_SPEC_DURATION_UNDER_SHARE", "0.10")
    segs = [
        _seg("ア", "短", 0.0, 0.5),
        *[_seg("イ", "正常一句话", 2.0 + index * 3.0, 4.0 + index * 3.0) for index in range(4)],
    ]
    report = compute_quality_report(segs, 60.0, [], 0, len(segs))

    assert report["spec_duration_under_min_share"] == pytest.approx(0.2)
    assert any("spec_duration_under_min_share" in w for w in report["warnings"])


def test_spec_compliance_clean_output_has_no_spec_warnings():
    segs = [
        _seg("ア", "你好 今天天气不错", 0.0, 2.0),
        _seg("イ", "我们出去走走吧", 3.0, 5.0),
    ]
    report = compute_quality_report(segs, 60.0, [], 0, 2)

    assert report["spec_zh_line_over_16_count"] == 0
    assert report["spec_zh_lines_over_2_count"] == 0
    assert report["spec_zh_banned_punct_count"] == 0
    assert report["spec_zh_cps_over_9_count"] == 0
    assert report["spec_duration_over_7s_count"] == 0
    assert report["spec_duration_under_min_count"] == 0
    assert report["spec_gap_under_2frames_count"] == 0
    assert not [w for w in report["warnings"] if w.startswith("spec_")]


def test_the_report_carries_the_chunk_cut_provenance_it_is_given():
    """Neither how the audio was cut nor what the layout claimed about split
    sentences can be read back off the finished subtitles, so the report is the
    only place a run keeps them."""
    segs = [_seg("ア", "一句话", 0.0, 2.0)]
    report = compute_quality_report(
        segs,
        60.0,
        [],
        0,
        1,
        chunk_cuts={
            "schema": "chunk_cut_provenance_v1",
            "policy": "latest_pause_midpoint",
            "source": "alignment_head_blank_runs",
            "chunk_count": 5,
            "cut_count": 4,
            "pause_cut_count": 3,
            "max_chunk_fallback_count": 1,
            "max_chunk_fallback_share": 0.25,
            "cut_pause_width_median_s": 1.4,
            "cut_pause_width_min_s": 0.7,
            "chunk_duration_median_s": 28.1,
            "chunk_duration_min_s": 21.0,
            "chunk_duration_max_s": 30.0,
        },
    )

    assert report["chunk_cut_policy"] == "latest_pause_midpoint"
    assert report["chunk_cut_source"] == "alignment_head_blank_runs"
    assert report["chunk_cut_at_pause_count"] == 3
    assert report["chunk_cut_max_fallback_count"] == 1
    assert report["chunk_cut_max_fallback_share"] == pytest.approx(0.25)
    assert report["chunk_duration_median_s"] == pytest.approx(28.1)


def test_a_high_hard_cut_rate_is_reported_and_not_warned_about():
    """It ranges 0.7%-53% across eight real films, and the high ones are films
    that are mostly continuous vocalisation. Any threshold that passed those
    would be met by everything."""
    segs = [_seg("ア", "一句话", 0.0, 2.0)]
    report = compute_quality_report(
        segs,
        60.0,
        [],
        0,
        1,
        chunk_cuts={
            "cut_count": 146,
            "pause_cut_count": 68,
            "max_chunk_fallback_count": 78,
            "max_chunk_fallback_share": 0.5342,
        },
    )

    assert report["chunk_cut_max_fallback_share"] == pytest.approx(0.5342)
    assert not [w for w in report["warnings"] if w.startswith("chunk_")]


def test_the_report_counts_continuation_claims_and_the_ones_withdrawn():
    segs = [_seg("ア", "一句话", 0.0, 2.0)]
    report = compute_quality_report(
        segs,
        60.0,
        [],
        0,
        1,
        cue_plan={
            "cues_after": 200,
            "layout_diagnostics": {
                "subtitle_layout_break_type": {
                    "sentence_punctuation": 148,
                    "word_gap": 40,
                    "end": 12,
                },
                "layout_word_gap_cut_count": 40,
                "layout_word_gap_cut_under_0p2s": 9,
                "layout_word_gap_median_s": 0.269,
                "continues_from_previous": 50,
                "continues_into_next": 52,
                "vocalisation_cues_dropped": 224,
                "vocalisation_runs_dropped": 60,
                "vocalisation_continuity_flags_cleared": 12,
            },
        },
    )

    assert report["cue_continues_from_previous_count"] == 50
    assert report["cue_continues_into_next_count"] == 52
    assert report["cue_continues_from_previous_share"] == pytest.approx(0.25)
    assert report["cue_plan_cue_count"] == 200
    assert report["vocalisation_continuity_flags_cleared"] == 12
    assert report["vocalisation_runs_dropped"] == 60
    # The break types are the reason the continuation counts are what they are:
    # every cue not ending on a sentence end claims to continue.
    assert report["layout_break_type_counts"]["word_gap"] == 40
    assert report["layout_word_gap_cut_under_0p2s"] == 9
    assert report["layout_word_gap_median_s"] == pytest.approx(0.269)


def test_the_report_shows_both_what_the_postgate_saw_and_what_survived():
    """The two numbers answer different questions.

    A chunk-level count says how much the detector marked; a cue-level count
    says how much of it got past the layout and the vocalisation filter into the
    file a viewer opens. Only the second one is a reason to act.
    """
    segs = [_seg("ア", "一句话", 0.0, 2.0)]
    report = compute_quality_report(
        segs,
        60.0,
        [],
        0,
        1,
        cue_plan={
            "cues_after": 400,
            "layout_diagnostics": {
                "postgate_flagged_cues": 20,
                "postgate_cue_flags": {"repeated_unit": 18, "runaway_repetition": 4},
            },
        },
        postgate={
            "schema": "text_alignment_postgate_v1",
            "reviewed": 250,
            "flagged": 40,
            "flags": {"repeated_unit": 36, "runaway_repetition": 7},
            "alignment_score_checked": 0,
        },
    )

    assert report["postgate_chunks_reviewed"] == 250
    assert report["postgate_chunks_flagged"] == 40
    assert report["postgate_chunks_flagged_share"] == pytest.approx(0.16)
    assert report["postgate_chunk_flag_counts"]["repeated_unit"] == 36
    assert report["postgate_flagged_cue_count"] == 20
    assert report["postgate_flagged_cue_share"] == pytest.approx(0.05)
    assert report["postgate_cue_flag_counts"]["repeated_unit"] == 18
    # The uncalibrated alignment-score check is off, and a report that hid that
    # would read as "the audio supports every cue".
    assert report["postgate_alignment_score_checked"] == 0
    assert not [w for w in report["warnings"] if w.startswith("postgate")]


def test_provenance_absent_from_the_run_stays_absent_from_the_report():
    """A resume that never chunked has nothing to say about cutting, and a null
    would read as 'measured zero'."""
    report = compute_quality_report([_seg("ア", "一句话", 0.0, 2.0)], 60.0, [], 0, 1)

    assert "chunk_cut_policy" not in report
    assert "cue_continues_from_previous_count" not in report


def test_an_empty_subtitle_run_still_reports_how_it_was_cut():
    """The run that produced nothing is exactly the one whose chunking is worth
    reading."""
    report = compute_quality_report(
        [],
        60.0,
        [],
        0,
        0,
        chunk_cuts={"cut_count": 2, "max_chunk_fallback_count": 2},
    )

    assert report["chunk_cut_count"] == 2
    assert report["chunk_cut_max_fallback_count"] == 2
