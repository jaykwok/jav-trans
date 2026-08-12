import re

import pytest

from subtitles.options import BASE_FPS, SubtitleOptions
from subtitles import writer as subtitle

# The one-character filler in these fixtures is kanji (私 / 君 / 僕), not kana.
# These tests are about timing and layout, but `prepare_srt_blocks` also drops
# runs of vocalisation-only cues by default, and two adjacent `あ` / `い` cues
# are exactly such a run - the filter deleted the fixtures out from under 16 of
# these tests. Kanji is lexical by construction and one character wide, so the
# reading-window and duration arithmetic they assert on is unchanged.


def _word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


def _cue_count(content: str) -> int:
    # write_srt / write_bilingual_srt emit a UTF-8 BOM for CJK player compat;
    # strip it so the first cue index ("1") still matches.
    return len(re.findall(r"^\d+$", content.lstrip("﻿"), flags=re.MULTILINE))


def test_bilingual_srt_keeps_the_japanese_when_the_translation_is_empty(tmp_path):
    """No 「未翻译」 line: it read as a translation failure to the viewer, and in
    this mode the Japanese is still worth showing on its own."""
    path = tmp_path / "out.srt"

    written = subtitle.write_bilingual_srt(
        [{"start": 0.0, "end": 1.0, "ja_text": "いい", "zh_text": ""}],
        str(path),
    )

    content = path.read_text(encoding="utf-8")
    assert "「未翻译」" not in content
    assert "いい" in content
    assert len(written) == 1


def test_a_punctuation_only_translation_is_not_a_translation(tmp_path):
    """The CHS style rules clear 。 / ， / 、 by design (no periods or commas,
    no trailing 、). A cue whose whole translation was punctuation therefore has
    nothing to display - which is what nonverbal cues translate to in practice."""
    path = tmp_path / "punct.srt"

    written = subtitle.write_srt(
        [
            {"start": 0.0, "end": 1.0, "ja_text": "あっ…", "zh_text": "。"},
            {"start": 1.0, "end": 2.0, "ja_text": "いや", "zh_text": "不要"},
            {"start": 2.0, "end": 3.0, "ja_text": "ああ", "zh_text": "、、、"},
        ],
        str(path),
    )

    content = path.read_text(encoding="utf-8")
    assert "「未翻译」" not in content
    assert [block["zh_text"] for block in written] == ["不要"]
    # Renumbered, not just skipped: a gap in the sequence breaks some players.
    assert _cue_count(content) == 1
    assert content.lstrip("﻿").startswith("1\n")


def test_a_dropped_cue_is_dropped_from_the_returned_blocks_too(tmp_path):
    """The quality report and the .json sidecar are built from the return value,
    so a cue kept there but missing from the SRT would make them disagree."""
    path = tmp_path / "all-punct.srt"

    written = subtitle.write_srt(
        [{"start": 0.0, "end": 1.0, "ja_text": "あっ", "zh_text": "，"}],
        str(path),
    )

    assert written == []
    assert path.read_text(encoding="utf-8-sig") == ""


def test_bilingual_drops_a_cue_only_when_both_lines_are_empty(tmp_path):
    path = tmp_path / "both.srt"

    written = subtitle.write_bilingual_srt(
        [
            {"start": 0.0, "end": 1.0, "ja_text": "", "zh_text": "。"},
            {"start": 1.0, "end": 2.0, "ja_text": "はい", "zh_text": "。"},
        ],
        str(path),
    )

    assert [block["ja_text"] for block in written] == ["はい"]
    assert _cue_count(path.read_text(encoding="utf-8")) == 1


def test_write_bilingual_srt_does_not_normalize_unprepared_blocks(tmp_path):
    path = tmp_path / "raw.srt"
    blocks = [
        {"start": 0.0, "end": 1.2, "ja_text": "私", "zh_text": "甲"},
        {"start": 1.0, "end": 2.0, "ja_text": "君", "zh_text": "乙"},
    ]

    written = subtitle.write_bilingual_srt(blocks, str(path), options=SubtitleOptions())

    assert written[0]["end"] == pytest.approx(1.2)
    assert "00:00:00,000 --> 00:00:01,199" in path.read_text(encoding="utf-8")


def test_write_srt_returned_blocks_match_min_written_duration(tmp_path):
    path = tmp_path / "min-duration.srt"
    written = subtitle.write_srt(
        [{"start": 1.0, "end": 1.0, "zh_text": "短い"}],
        str(path),
    )

    assert written[0]["end"] == pytest.approx(1.05)
    assert "00:00:01,000 --> 00:00:01,050" in path.read_text(encoding="utf-8")


def test_write_bilingual_srt_returned_blocks_match_min_written_duration(tmp_path):
    path = tmp_path / "min-duration-bilingual.srt"
    written = subtitle.write_bilingual_srt(
        [{"start": 2.0, "end": 2.0, "ja_text": "私", "zh_text": "啊"}],
        str(path),
    )

    assert written[0]["end"] == pytest.approx(2.05)
    assert "00:00:02,000 --> 00:00:02,049" in path.read_text(encoding="utf-8")


def test_wrap_subtitle_line_uses_hiragana_kanji_boundary():
    assert subtitle._wrap_subtitle_line("あいうえ漢字テスト", max_chars=5) == (
        "あいうえ\n漢字テスト"
    )


def test_alignment_window_extends_min_duration_without_overlapping_next():
    blocks = [
        {"start": 0.0, "end": 0.1, "ja_text": "私", "zh_text": "啊"},
        {"start": 0.7, "end": 1.0, "ja_text": "君", "zh_text": "咿"},
    ]

    start, end = subtitle._resolve_subtitle_window(blocks, 1)

    assert start == 0.0
    assert end == pytest.approx(0.7 - SubtitleOptions().frame_gap_s)


def test_alignment_window_extends_micro_cue_to_fixed_frame_floor():
    blocks = [
        {"start": 0.0, "end": 0.1, "ja_text": "私", "zh_text": "啊"},
        {"start": 2.0, "end": 2.5, "ja_text": "君", "zh_text": "咿"},
    ]
    options = SubtitleOptions()

    start, end = subtitle._resolve_subtitle_window(blocks, 1, options=options)

    assert start == 0.0
    assert end == pytest.approx(20.0 / BASE_FPS)


def test_alignment_window_uses_fixed_two_frame_gap():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "私", "zh_text": "啊"},
        {"start": 1.0, "end": 2.0, "ja_text": "君", "zh_text": "咿"},
    ]
    options = SubtitleOptions()

    _start, end = subtitle._resolve_subtitle_window(blocks, 1, options=options)

    assert end == pytest.approx(1.0 - options.frame_gap_s)


def test_prepare_srt_blocks_sorts_and_removes_overlap_with_frame_gap():
    blocks = [
        {"start": 1.0, "end": 2.0, "ja_text": "君", "zh_text": "乙"},
        {"start": 0.0, "end": 1.2, "ja_text": "私", "zh_text": "甲"},
    ]
    options = SubtitleOptions()

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert [block["ja_text"] for block in prepared] == ["私", "君"]
    assert prepared[0]["end"] == pytest.approx(1.0 - options.frame_gap_s)
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]


def test_prepare_srt_blocks_reports_dp_stage_progress():
    events: list[tuple[str, int, int]] = []
    blocks = [
        {
            "start": 0.0,
            "end": 20.0,
            "ja_text": "長い字幕です。" * 20,
            "zh_text": "很长的字幕。" * 20,
        }
    ]

    subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(max_display_duration_s=6.0),
        mode="bilingual",
        on_stage=lambda stage, current, total: events.append(
            (stage, current, total)
        ),
    )

    assert ("layout_measured_safe_dp", 0, 1) in events
    assert ("layout_measured_safe_dp", 1, 1) in events
    assert events[-1] == ("layout_finalize", 1, 1)


def test_prepare_srt_blocks_anchors_start_to_first_timed_word():
    blocks = [
        {
            "start": 10.35,
            "end": 11.2,
            "ja_text": "小那海あやです",
            "zh_text": "我是小那海绫",
            "words": [
                _word("小那海", 10.0, 10.35),
                _word("あや", 10.35, 10.55),
                _word("です", 10.55, 11.2),
            ],
        }
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(),
        mode="bilingual",
    )

    assert prepared[0]["start"] == pytest.approx(10.0)


def test_prepare_srt_blocks_does_not_anchor_to_synthetic_proportional_words():
    blocks = [
        {
            "start": 10.35,
            "end": 11.2,
            "ja_text": "小那海あやです",
            "zh_text": "我是小那海绫",
            "words": [
                {
                    **_word("小那海あやです", 10.0, 11.2),
                    "timestamp_kind": "synthetic_proportional",
                }
            ],
        }
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(),
        mode="bilingual",
    )

    assert prepared[0]["start"] == pytest.approx(10.35)


def test_prepare_srt_blocks_preserves_earliest_word_start_anchor_without_merge():
    blocks = [
        {
            "start": 10.35,
            "end": 10.7,
            "ja_text": "小那海",
            "zh_text": "小那海",
            "words": [_word("小那海", 10.0, 10.35)],
        },
        {
            "start": 10.76,
            "end": 11.2,
            "ja_text": "あやです",
            "zh_text": "绫",
            "words": [_word("あやです", 10.76, 11.2)],
        },
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(),
        mode="bilingual",
    )

    assert len(prepared) == 2
    assert prepared[0]["start"] == pytest.approx(10.0)


def test_prepare_srt_blocks_final_normalize_guards_reading_window_overlap(monkeypatch):
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "私", "zh_text": "甲"},
        {"start": 1.2, "end": 2.0, "ja_text": "君", "zh_text": "乙"},
    ]
    options = SubtitleOptions()
    original_resolve = subtitle._resolve_subtitle_window

    def expand_first_window(blocks, idx, *, options=None):
        if idx == 1:
            return 0.0, 1.25
        return original_resolve(blocks, idx, options=options)

    monkeypatch.setattr(subtitle, "_resolve_subtitle_window", expand_first_window)

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert prepared[0]["end"] == pytest.approx(1.2 - options.frame_gap_s)
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]


def test_timing_polish_collapses_short_gap_to_two_frames():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "私", "zh_text": "甲"},
        {"start": 1.2, "end": 2.0, "ja_text": "君", "zh_text": "乙"},
    ]
    options = SubtitleOptions(
        timing_polish_enabled=True,
        short_gap_collapse_s=0.5,
        linger_s=0.45,
    )

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert prepared[0]["end"] == pytest.approx(1.2 - options.frame_gap_s)
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]


def test_timing_polish_preserves_natural_pause():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "私", "zh_text": "甲"},
        {"start": 1.8, "end": 2.5, "ja_text": "君", "zh_text": "乙"},
    ]
    options = SubtitleOptions(
        timing_polish_enabled=True,
        short_gap_collapse_s=0.5,
        linger_s=0.45,
    )

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    # Linger stops at next_start - short_gap_collapse_s, keeping a visible
    # half-second pause; the 0.5s acoustic-shift cap no longer binds first.
    assert prepared[0]["end"] == pytest.approx(1.3)
    assert prepared[1]["start"] - prepared[0]["end"] == pytest.approx(0.5)


def test_timing_polish_disabled_keeps_existing_alignment_end():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "私", "zh_text": "甲"},
        {"start": 1.2, "end": 2.0, "ja_text": "君", "zh_text": "乙"},
    ]
    options = SubtitleOptions(
        timing_polish_enabled=False,
        short_gap_collapse_s=0.5,
        linger_s=0.45,
    )

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert prepared[0]["end"] == pytest.approx(1.0)


def test_unmeasured_weak_cut_candidate_is_not_used_as_a_timeline():
    blocks = [
        {
            "start": 0.0,
            "end": 9.0,
            "ja_text": "これは長い台詞です。次の台詞です。",
            "zh_text": "这是很长的台词。下一句台词。",
            "weak_cut_candidates": [
                {
                    "kind": "weak",
                    "time_s": 4.2,
                    "frame": 210,
                    "score": 0.2,
                    "prominence": 0.1,
                    "speech_valley": 0.8,
                    "strength": 1.1,
                }
            ],
        }
    ]
    options = SubtitleOptions(
        timing_polish_enabled=False,
        linger_s=0.45,
    )

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert len(prepared) == 1
    assert prepared[0]["start"] == pytest.approx(0.0)
    assert prepared[0]["end"] == pytest.approx(9.0)
    assert prepared[0]["subtitle_layout_split_skipped"] == (
        "measured_word_timestamps_unavailable"
    )
    assert prepared[0]["proportional_fallback_used"] is False


def test_long_display_cue_never_falls_back_to_proportional_text_split():
    blocks = [
        {
            "start": 0.0,
            "end": 9.0,
            "ja_text": "これは長い台詞です。次の台詞です。",
            "zh_text": "这是很长的台词。下一句台词。",
        }
    ]
    options = SubtitleOptions(timing_polish_enabled=False)

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert len(prepared) == 1
    assert prepared[0]["proportional_fallback_used"] is False
    assert prepared[0]["subtitle_layout_split_skipped"] == (
        "measured_word_timestamps_unavailable"
    )


def test_short_cues_are_not_merged():
    blocks = [
        {"start": 0.0, "end": 0.35, "ja_text": "私", "zh_text": "啊"},
        {"start": 0.42, "end": 0.80, "ja_text": "僕", "zh_text": "嗯"},
        {"start": 1.40, "end": 1.80, "ja_text": "いい", "zh_text": "舒服"},
    ]
    options = SubtitleOptions()

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert len(prepared) == 3
    assert [item["ja_text"] for item in prepared] == ["私", "僕", "いい"]


def test_close_short_cues_remain_separate():
    blocks = [
        {"start": 0.0, "end": 0.35, "ja_text": "私", "zh_text": "啊"},
        {"start": 0.42, "end": 0.80, "ja_text": "僕", "zh_text": "嗯"},
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(),
        mode="bilingual",
    )

    assert len(prepared) == 2


def test_short_cues_ignore_acoustic_metadata_without_merge():
    blocks = [
        {"start": 0.0, "end": 0.35, "ja_text": "私", "zh_text": "啊"},
        {"start": 0.42, "end": 0.80, "ja_text": "僕", "zh_text": "嗯"},
    ]
    options = SubtitleOptions()

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert len(prepared) == 2


def test_prepare_srt_blocks_has_same_no_merge_behavior_for_japanese_only():
    blocks = [
        {"start": 0.0, "end": 0.40, "ja_text": "私", "zh_text": "私"},
        {"start": 0.46, "end": 0.90, "ja_text": "僕", "zh_text": "僕"},
    ]

    merged = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(),
        mode="srt",
    )
    unmerged = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(),
        mode="srt",
    )

    assert len(merged) == 2
    assert len(unmerged) == 2


def test_timing_polish_does_not_merge_after_collapsing_gap():
    blocks = [
        {"start": 0.0, "end": 0.40, "ja_text": "私", "zh_text": "私"},
        {"start": 0.90, "end": 1.30, "ja_text": "僕", "zh_text": "僕"},
    ]
    options = SubtitleOptions(
        timing_polish_enabled=True,
        short_gap_collapse_s=0.5,
    )

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="srt")

    assert len(prepared) == 2
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]


def test_timing_polish_keeps_short_cues_separate():
    blocks = [
        {"start": 0.0, "end": 0.40, "ja_text": "私", "zh_text": "私"},
        {"start": 0.90, "end": 1.30, "ja_text": "僕", "zh_text": "僕"},
    ]
    options = SubtitleOptions(
        timing_polish_enabled=True,
        short_gap_collapse_s=0.5,
    )

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="srt")

    assert len(prepared) == 2
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]


def test_prepare_srt_blocks_merges_overlap_when_too_tight():
    blocks = [
        {
            "start": 1.0,
            "end": 1.2,
            "ja_text": "私",
            "zh_text": "甲",
        },
        {
            "start": 1.05,
            "end": 1.4,
            "ja_text": "君",
            "zh_text": "乙",
        },
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(),
        mode="bilingual",
    )

    assert len(prepared) == 2
    assert prepared[0]["end"] <= prepared[1]["start"]


def test_normalize_subtitle_timeline_locks_next_start_when_too_tight():
    blocks = [
        {
            "start": 1.0,
            "end": 1.03,
            "ja_text": "前" * 80,
            "zh_text": "前" * 80,
        },
        {
            "start": 1.02,
            "end": 1.5,
            "ja_text": "次" * 80,
            "zh_text": "下" * 80,
        },
    ]
    options = SubtitleOptions()

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert len(prepared) == 2
    assert prepared[1]["start"] == pytest.approx(1.02)
    assert prepared[0]["end"] <= prepared[1]["start"]


def test_too_close_cues_keep_two_frame_gap_and_report_min_display_violation():
    blocks = [
        {
            "start": 1.0,
            "end": 1.03,
            "ja_text": "前",
            "zh_text": "前",
        },
        {
            "start": 1.2,
            "end": 1.6,
            "ja_text": "次",
            "zh_text": "下",
        },
    ]
    options = SubtitleOptions()

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert prepared[0]["end"] == pytest.approx(prepared[1]["start"] - options.frame_gap_s)
    assert prepared[0]["display_duration"] < options.frame_min_duration_s
    assert prepared[0]["duration_violation"] is True
    assert prepared[0]["gap_violation"] is False
    assert prepared[1]["gap_violation"] is False


def test_write_bilingual_srt_returns_normalized_blocks(tmp_path):
    path = tmp_path / "normalized.srt"
    blocks = [
        {"start": 0.0, "end": 1.2, "ja_text": "私", "zh_text": "甲"},
        {"start": 1.0, "end": 2.0, "ja_text": "君", "zh_text": "乙"},
    ]
    options = SubtitleOptions()

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")
    written = subtitle.write_bilingual_srt(prepared, str(path), options=options)

    assert written[0]["end"] == pytest.approx(1.0 - options.frame_gap_s)
    assert "00:00:00,000 --> 00:00:00,916" in path.read_text(encoding="utf-8")


def test_adjacent_short_blocks_are_not_merged(tmp_path):
    path = tmp_path / "not_merged.srt"
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "いい", "zh_text": "好"},
        {"start": 1.1, "end": 2.0, "ja_text": "もっと", "zh_text": "更多"},
    ]

    prepared = subtitle.prepare_srt_blocks(blocks, mode="bilingual")
    subtitle.write_bilingual_srt(prepared, str(path))

    content = path.read_text(encoding="utf-8")
    assert _cue_count(content) == 2
    assert "いい\n好" in content
    assert "もっと\n更多" in content
    assert "いい もっと" not in content
    assert "好，更多" not in content


def test_adjacent_blocks_stay_separate_after_sentence_punctuation(tmp_path):
    path = tmp_path / "blocked.srt"
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "終わり。", "zh_text": "结束。"},
        {"start": 1.05, "end": 2.0, "ja_text": "次", "zh_text": "下一句"},
    ]

    prepared = subtitle.prepare_srt_blocks(blocks, mode="bilingual")
    subtitle.write_bilingual_srt(prepared, str(path))

    content = path.read_text(encoding="utf-8")
    assert _cue_count(content) == 2



def test_write_srt_does_not_emit_acoustic_prefix(tmp_path):
    path = tmp_path / "plain.srt"

    subtitle.write_srt(
        [{"start": 0.0, "end": 1.0, "zh_text": "过来"}],
        str(path),
    )

    content = path.read_text(encoding="utf-8")
    assert "过来" in content


def test_write_bilingual_srt_does_not_emit_acoustic_prefix(tmp_path):
    path = tmp_path / "plain_bilingual.srt"

    subtitle.write_bilingual_srt(
        [{"start": 0.0, "end": 1.0, "ja_text": "来て", "zh_text": "过来"}],
        str(path),
    )

    content = path.read_text(encoding="utf-8")
    assert "过来" in content


def test_unsplittable_long_block_keeps_its_timeline_instead_of_clamping():
    # Without a measured safe boundary, the 7s target loses to timeline truth.
    blocks = [{
        "start": 0.0,
        "end": 30.0,
        "ja_text": "んっ",
        "zh_text": "嗯",
        "words": [{
            "word": "んっ",
            "start": 0.0,
            "end": 30.0,
            "timestamp_kind": "grok_stt_word",
        }],
    }]

    prepared = subtitle.prepare_srt_blocks(blocks, options=SubtitleOptions())

    assert len(prepared) == 1
    cue = prepared[0]
    assert cue["display_end"] - cue["display_start"] == pytest.approx(30.0)
    assert cue["display_clamped_to_max"] is False
    assert cue["acoustic_end"] == pytest.approx(30.0)
    assert cue["duration_soft_cap_violation"] is True
    assert cue["duration_violation"] is True
    assert cue["proportional_fallback_used"] is False


def test_punctuation_splits_are_exact_when_word_times_are_measured():
    text = "今日は本当にいい天気ですね。散歩に行きましょう。公園でお弁当を食べたいです。"
    words = _aligned_words(text, 0.0, 0.25)
    blocks = [{
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }]

    prepared = subtitle.prepare_srt_blocks(blocks, options=SubtitleOptions())

    assert len(prepared) >= 2
    for cue in prepared:
        assert cue["display_end"] - cue["display_start"] <= 7.0 + 1e-6
        assert cue["display_clamped_to_max"] is False
        assert cue["exact_measured_timeline"] is True
        assert cue["proportional_fallback_used"] is False


def _aligned_words(text: str, start: float, char_s: float) -> list[dict]:
    words = []
    cursor = start
    for char in text:
        words.append(
            {
                "word": char,
                "start": cursor,
                "end": cursor + char_s,
                "timestamp_kind": "ctc_forced_alignment",
            }
        )
        cursor += char_s
    return words


def test_source_char_target_uses_the_only_later_measured_safe_point():
    text = "あ" * 26
    words = _aligned_words(text, 0.0, 0.20)
    for word in words[23:]:
        word["start"] += 0.20
        word["end"] += 0.20
    block = {
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }

    pieces = subtitle.prepare_srt_blocks(
        [block],
        options=SubtitleOptions(drop_vocalisation_only_cues=False),
    )

    assert [len(piece["ja_text"]) for piece in pieces] == [23, 3]
    assert pieces[0]["source_char_violation"] is True
    assert pieces[1]["start"] == pytest.approx(words[23]["start"])
    assert pieces[0]["end"] == pytest.approx(words[22]["end"])


def test_source_char_target_does_not_invent_a_boundary_when_none_is_safe():
    text = "あ" * 25
    words = _aligned_words(text, 0.0, 0.20)
    block = {
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }

    pieces = subtitle.prepare_srt_blocks(
        [block],
        options=SubtitleOptions(drop_vocalisation_only_cues=False),
    )

    assert len(pieces) == 1
    assert pieces[0]["source_char_violation"] is True
    assert pieces[0]["subtitle_layout_split_skipped"] == (
        "measured_safe_boundaries_unavailable"
    )
    assert pieces[0]["proportional_fallback_used"] is False


def test_duration_target_uses_the_same_measured_safe_boundary_dp():
    text = "あ" * 12
    words = _aligned_words(text, 0.0, 0.70)
    for word in words[6:]:
        word["start"] += 0.20
        word["end"] += 0.20
    block = {
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }

    pieces = subtitle.prepare_srt_blocks(
        [block],
        options=SubtitleOptions(drop_vocalisation_only_cues=False),
    )

    assert [piece["ja_text"] for piece in pieces] == ["あ" * 6, "あ" * 6]
    assert all(piece["duration_soft_cap_violation"] is False for piece in pieces)
    assert pieces[0]["end"] == pytest.approx(words[5]["end"])
    assert pieces[1]["start"] == pytest.approx(words[6]["start"])


def test_long_cue_splits_at_measured_word_gap_not_mid_word():
    # The failure this fixes: with no punctuation and two different speaking
    # rates, a character-ratio split lands inside a word. The 0.8s silence
    # between them is the only correct break, and only the measured word
    # timings know where it is.
    first = "あのちょっとだけ"
    second = "まってくださいよおねがい"
    words = _aligned_words(first, 0.0, 0.20)
    gap_start = words[-1]["end"]
    words += _aligned_words(second, gap_start + 0.8, 0.50)
    block = {
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": first + second,
        "zh_text": first + second,
        "words": words,
    }

    prepared = subtitle.prepare_srt_blocks([block], options=SubtitleOptions())

    assert len(prepared) == 2
    assert prepared[0]["ja_text"] == first
    assert prepared[1]["ja_text"] == second
    # Blank says the text boundary is safe; the next cue enters exactly with
    # its first measured word, not halfway through the preceding silence.
    assert prepared[1]["start"] == pytest.approx(gap_start + 0.8)
    assert prepared[0]["subtitle_layout_split_source"] == (
        "measured_safe_boundary_dp"
    )
    assert prepared[0]["text_break_type"] == "strong_gap"


def test_long_grok_cue_splits_at_measured_word_gap_not_mid_word():
    first = "あのちょっとだけ"
    second = "まってくださいよおねがい"
    words = _aligned_words(first, 0.0, 0.20)
    gap_start = words[-1]["end"]
    words += _aligned_words(second, gap_start + 0.8, 0.50)
    for word in words:
        word["timestamp_kind"] = "grok_stt_word"
    block = {
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": first + second,
        "zh_text": first + second,
        "words": words,
    }

    prepared = subtitle.prepare_srt_blocks([block], options=SubtitleOptions())

    assert len(prepared) == 2
    assert prepared[0]["ja_text"] == first
    assert prepared[1]["ja_text"] == second
    assert prepared[1]["start"] == pytest.approx(gap_start + 0.8)
    assert prepared[0]["subtitle_layout_split_source"] == (
        "measured_safe_boundary_dp"
    )


def test_a_long_silence_puts_the_next_cue_at_its_measured_word_start():
    text = "この村の儀式を受けてもらうために必ず儀式をしなければいけない男子は一週間耐えなければいけない"
    words = _aligned_words(text, 0.0, 0.25)
    target_position = text.index("必")
    # Model a long non-speech interval before this phrase. Character ratio would
    # put the boundary far too early; the measured word start is authoritative.
    for index, word in enumerate(words):
        if index >= target_position:
            word["start"] += 5.0
            word["end"] += 5.0
    block = {
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }

    pieces = subtitle.prepare_srt_blocks(
        [block],
        options=SubtitleOptions(drop_vocalisation_only_cues=False),
    )

    following = [
        piece for piece in pieces if str(piece["ja_text"]).startswith("必")
    ]
    assert following, [piece["ja_text"] for piece in pieces]
    assert following[0]["start"] == pytest.approx(words[target_position]["start"])
    assert following[0]["exact_measured_timeline"] is True
    # The pause itself stays empty: the previous cue ends on its own last word.
    previous = pieces[pieces.index(following[0]) - 1]
    assert previous["end"] == pytest.approx(words[target_position - 1]["end"])


def test_incomplete_measured_word_map_never_falls_back_to_proportional_time():
    text = "この文字列は十分に長いので表示時間による分割が必要になります"
    words = _aligned_words(text.replace("文字", ""), 0.0, 0.5)
    block = {
        "start": 0.0,
        "end": 20.0,
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }

    pieces = subtitle._split_long_display_block(
        block,
        options=SubtitleOptions(),
    )

    assert len(pieces) == 1
    assert pieces[0]["ja_text"] == text
    assert pieces[0]["subtitle_layout_split_skipped"] == (
        "measured_word_text_map_incomplete"
    )
    assert "subtitle_layout_split_source" not in pieces[0]


def test_single_measured_token_never_gets_split_at_an_invented_time():
    text = "ひとつの計測済みトークンとして返された長い字幕テキスト"
    block = {
        "start": 0.0,
        "end": 18.0,
        "ja_text": text,
        "zh_text": text,
        "words": [
            {
                "word": text,
                "start": 4.0,
                "end": 17.0,
                "timestamp_kind": "grok_stt_word",
            }
        ],
    }

    pieces = subtitle._split_long_display_block(
        block,
        options=SubtitleOptions(),
    )

    assert len(pieces) == 1
    assert pieces[0]["subtitle_layout_split_skipped"] == (
        "measured_safe_boundaries_unavailable"
    )
    assert "subtitle_layout_split_source" not in pieces[0]


def test_ctc_punctuation_frames_do_not_become_subtitle_onsets():
    first = "前の台詞"
    ellipsis = "..."
    second = "こんな出来損ない"
    words = _aligned_words(first, 0.0, 0.35)
    words += _aligned_words(ellipsis, 2.0, 0.60)
    second_start = 6.5
    words += _aligned_words(second, second_start, 0.35)
    text = first + ellipsis + second
    block = {
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }

    pieces = subtitle.prepare_srt_blocks([block], options=SubtitleOptions())

    assert [piece["ja_text"] for piece in pieces] == [first + ellipsis, second]
    assert pieces[1]["start"] == pytest.approx(second_start)
    assert pieces[0]["end"] < second_start
    assert not any(piece["display_clamped_to_max"] for piece in pieces)
    assert not pieces[1]["ja_text"].startswith(tuple(".,，、。！？!?…；;"))


def test_long_blank_is_a_real_gap_between_independent_cue_edges():
    words = [
        {
            "word": "先",
            "start": 0.0,
            "end": 0.5,
            "timestamp_kind": "ctc_forced_alignment",
        },
        {
            "word": "後",
            "start": 15.0,
            "end": 15.5,
            "timestamp_kind": "ctc_forced_alignment",
        },
    ]
    block = {
        "start": 0.0,
        "end": 15.5,
        "ja_text": "先後",
        "zh_text": "先後",
        "words": words,
    }

    pieces = subtitle.prepare_srt_blocks([block], options=SubtitleOptions())

    assert [piece["ja_text"] for piece in pieces] == ["先", "後"]
    # Both edges remain exactly on their measured lexical words. The 14.5s
    # blank is genuinely subtitle-free; no display linger is added.
    assert pieces[0]["end"] == pytest.approx(words[0]["end"])
    assert pieces[1]["start"] == pytest.approx(15.0)
    assert not any(piece["display_clamped_to_max"] for piece in pieces)


def test_long_cue_with_only_synthetic_word_timings_remains_unsplit():
    # Proportional timings are a restatement of the character ratio the DP
    # already has. Treating them as measured evidence would launder a guess
    # into an acoustic anchor.
    first = "あのちょっとだけ"
    second = "まってくださいよおねがい"
    words = _aligned_words(first, 0.0, 0.20)
    gap_start = words[-1]["end"]
    words += _aligned_words(second, gap_start + 0.8, 0.50)
    for word in words:
        word["timestamp_kind"] = "synthetic_proportional"
    block = {
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": first + second,
        "zh_text": first + second,
        "words": words,
    }

    prepared = subtitle.prepare_srt_blocks([block], options=SubtitleOptions())

    assert len(prepared) == 1
    assert prepared[0]["subtitle_layout_split_skipped"] == (
        "measured_word_timestamps_unavailable"
    )
    assert prepared[0]["proportional_fallback_used"] is False


def test_within_word_spacing_is_not_a_safe_boundary():
    # Continuous speech has small inter-character gaps. Treating those as safe
    # boundaries would put the split back inside a word.
    text = "きょうはいいてんきですね" * 2
    words = _aligned_words(text, 0.0, 0.36)
    for word in words:
        # 60ms of separation everywhere: real, but far below a pause.
        word["end"] = word["start"] + 0.30
    block = {
        "start": 0.0,
        "end": words[-1]["end"],
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }

    timed = subtitle._timed_words(block)
    assert not any(
        subtitle._is_exact_safe_boundary(timed, index)
        for index in range(1, len(timed))
    )
    # And end to end: no measured gap, no split, no invented boundary.
    pieces = subtitle.prepare_srt_blocks(
        [block],
        options=SubtitleOptions(drop_vocalisation_only_cues=False),
    )
    assert len(pieces) == 1
    assert pieces[0]["subtitle_layout_split_skipped"] == (
        "measured_safe_boundaries_unavailable"
    )
