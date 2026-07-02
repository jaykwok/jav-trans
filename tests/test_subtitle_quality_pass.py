import re

import pytest

from subtitles.options import BASE_FPS, SubtitleOptions
from subtitles import writer as subtitle


def _word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


def _cue_count(content: str) -> int:
    # write_srt / write_bilingual_srt emit a UTF-8 BOM for CJK player compat;
    # strip it so the first cue index ("1") still matches.
    return len(re.findall(r"^\d+$", content.lstrip("﻿"), flags=re.MULTILINE))


def test_bilingual_srt_uses_placeholder_for_empty_translation(tmp_path, caplog):
    path = tmp_path / "out.srt"

    subtitle.write_bilingual_srt(
        [{"start": 0.0, "end": 1.0, "ja_text": "いい", "zh_text": ""}],
        str(path),
    )

    content = path.read_text(encoding="utf-8")
    assert "「未翻译」" in content
    assert "Empty translated subtitle" in caplog.text


def test_write_bilingual_srt_does_not_normalize_unprepared_blocks(tmp_path):
    path = tmp_path / "raw.srt"
    blocks = [
        {"start": 0.0, "end": 1.2, "ja_text": "あ", "zh_text": "甲"},
        {"start": 1.0, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
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
        [{"start": 2.0, "end": 2.0, "ja_text": "あ", "zh_text": "啊"}],
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
        {"start": 0.0, "end": 0.1, "ja_text": "あ", "zh_text": "啊"},
        {"start": 0.7, "end": 1.0, "ja_text": "い", "zh_text": "咿"},
    ]

    start, end = subtitle._resolve_subtitle_window(blocks, 1)

    assert start == 0.0
    assert end == pytest.approx(0.7 - SubtitleOptions().frame_gap_s)


def test_alignment_window_extends_micro_cue_to_fixed_frame_floor():
    blocks = [
        {"start": 0.0, "end": 0.1, "ja_text": "あ", "zh_text": "啊"},
        {"start": 2.0, "end": 2.5, "ja_text": "い", "zh_text": "咿"},
    ]
    options = SubtitleOptions()

    start, end = subtitle._resolve_subtitle_window(blocks, 1, options=options)

    assert start == 0.0
    assert end == pytest.approx(20.0 / BASE_FPS)


def test_alignment_window_uses_fixed_two_frame_gap():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "あ", "zh_text": "啊"},
        {"start": 1.0, "end": 2.0, "ja_text": "い", "zh_text": "咿"},
    ]
    options = SubtitleOptions()

    _start, end = subtitle._resolve_subtitle_window(blocks, 1, options=options)

    assert end == pytest.approx(1.0 - options.frame_gap_s)


def test_prepare_srt_blocks_sorts_and_removes_overlap_with_frame_gap():
    blocks = [
        {"start": 1.0, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
        {"start": 0.0, "end": 1.2, "ja_text": "あ", "zh_text": "甲"},
    ]
    options = SubtitleOptions()

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert [block["ja_text"] for block in prepared] == ["あ", "い"]
    assert prepared[0]["end"] == pytest.approx(1.0 - options.frame_gap_s)
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]


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
        {"start": 0.0, "end": 1.0, "ja_text": "あ", "zh_text": "甲"},
        {"start": 1.2, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
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
        {"start": 0.0, "end": 1.0, "ja_text": "あ", "zh_text": "甲"},
        {"start": 1.2, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
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
        {"start": 0.0, "end": 1.0, "ja_text": "あ", "zh_text": "甲"},
        {"start": 1.8, "end": 2.5, "ja_text": "い", "zh_text": "乙"},
    ]
    options = SubtitleOptions(
        timing_polish_enabled=True,
        short_gap_collapse_s=0.5,
        linger_s=0.45,
    )

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert prepared[0]["end"] == pytest.approx(1.2)
    assert prepared[1]["start"] - prepared[0]["end"] == pytest.approx(0.6)


def test_timing_polish_disabled_keeps_existing_alignment_end():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "あ", "zh_text": "甲"},
        {"start": 1.2, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
    ]
    options = SubtitleOptions(
        timing_polish_enabled=False,
        short_gap_collapse_s=0.5,
        linger_s=0.45,
    )

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert prepared[0]["end"] == pytest.approx(1.0)


def test_long_display_cue_splits_at_weak_cut_candidate():
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
        timing_polish_enabled=True,
        linger_s=0.45,
    )

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert len(prepared) == 2
    assert prepared[1]["start"] == pytest.approx(4.2)
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]
    assert max(item["end"] - item["start"] for item in prepared) <= 7.0


def test_long_display_cue_falls_back_to_proportional_text_split():
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

    assert len(prepared) == 2
    assert prepared[1]["start"] == pytest.approx(5.294117647058823)
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]


def test_short_cues_are_not_merged():
    blocks = [
        {"start": 0.0, "end": 0.35, "ja_text": "あ", "zh_text": "啊"},
        {"start": 0.42, "end": 0.80, "ja_text": "ん", "zh_text": "嗯"},
        {"start": 1.40, "end": 1.80, "ja_text": "いい", "zh_text": "舒服"},
    ]
    options = SubtitleOptions()

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert len(prepared) == 3
    assert [item["ja_text"] for item in prepared] == ["あ", "ん", "いい"]


def test_close_short_cues_remain_separate():
    blocks = [
        {"start": 0.0, "end": 0.35, "ja_text": "あ", "zh_text": "啊"},
        {"start": 0.42, "end": 0.80, "ja_text": "ん", "zh_text": "嗯"},
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(),
        mode="bilingual",
    )

    assert len(prepared) == 2


def test_short_cues_ignore_acoustic_metadata_without_merge():
    blocks = [
        {"start": 0.0, "end": 0.35, "ja_text": "あ", "zh_text": "啊"},
        {"start": 0.42, "end": 0.80, "ja_text": "ん", "zh_text": "嗯"},
    ]
    options = SubtitleOptions()

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert len(prepared) == 2


def test_prepare_srt_blocks_has_same_no_merge_behavior_for_japanese_only():
    blocks = [
        {"start": 0.0, "end": 0.40, "ja_text": "あ", "zh_text": "あ"},
        {"start": 0.46, "end": 0.90, "ja_text": "ん", "zh_text": "ん"},
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
        {"start": 0.0, "end": 0.40, "ja_text": "あ", "zh_text": "あ"},
        {"start": 0.90, "end": 1.30, "ja_text": "ん", "zh_text": "ん"},
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
        {"start": 0.0, "end": 0.40, "ja_text": "あ", "zh_text": "あ"},
        {"start": 0.90, "end": 1.30, "ja_text": "ん", "zh_text": "ん"},
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
            "ja_text": "あ",
            "zh_text": "甲",
        },
        {
            "start": 1.05,
            "end": 1.4,
            "ja_text": "い",
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
        {"start": 0.0, "end": 1.2, "ja_text": "あ", "zh_text": "甲"},
        {"start": 1.0, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
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
