import re

import pytest

from subtitles.options import SubtitleOptions
from subtitles import writer as subtitle


def _word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


def _cue_count(content: str) -> int:
    return len(re.findall(r"^\d+$", content, flags=re.MULTILINE))


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

    written = subtitle.write_bilingual_srt(blocks, str(path), options=SubtitleOptions(video_fps=25.0))

    assert written[0]["end"] == pytest.approx(1.2)
    assert "00:00:00,000 --> 00:00:01,199" in path.read_text(encoding="utf-8")


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
    assert end == 0.6


def test_alignment_window_uses_two_frame_gap_from_video_fps():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "あ", "zh_text": "啊"},
        {"start": 1.0, "end": 2.0, "ja_text": "い", "zh_text": "咿"},
    ]
    options = SubtitleOptions(video_fps=25.0)

    _start, end = subtitle._resolve_subtitle_window(blocks, 1, options=options)

    assert end == pytest.approx(0.92)


def test_prepare_srt_blocks_sorts_and_removes_overlap_with_frame_gap():
    blocks = [
        {"start": 1.0, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
        {"start": 0.0, "end": 1.2, "ja_text": "あ", "zh_text": "甲"},
    ]
    options = SubtitleOptions(video_fps=25.0, merge_adjacent=False)

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert [block["ja_text"] for block in prepared] == ["あ", "い"]
    assert prepared[0]["end"] == pytest.approx(0.92)
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]


def test_prepare_srt_blocks_final_normalize_guards_reading_window_overlap(monkeypatch):
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "あ", "zh_text": "甲"},
        {"start": 1.2, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
    ]
    options = SubtitleOptions(video_fps=25.0, merge_adjacent=False)
    original_resolve = subtitle._resolve_subtitle_window

    def expand_first_window(blocks, idx, *, options=None):
        if idx == 1:
            return 0.0, 1.25
        return original_resolve(blocks, idx, options=options)

    monkeypatch.setattr(subtitle, "_resolve_subtitle_window", expand_first_window)

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")

    assert prepared[0]["end"] == pytest.approx(1.12)
    assert prepared[0]["end"] + options.frame_gap_s <= prepared[1]["start"]


def test_prepare_srt_blocks_merges_same_speaker_overlap_when_too_tight():
    blocks = [
        {
            "start": 1.0,
            "end": 1.2,
            "ja_text": "あ",
            "zh_text": "甲",
            "speaker": "S0",
        },
        {
            "start": 1.05,
            "end": 1.4,
            "ja_text": "い",
            "zh_text": "乙",
            "speaker": "S0",
        },
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(video_fps=29.97, merge_adjacent=False),
        mode="bilingual",
    )

    assert len(prepared) == 1
    assert prepared[0]["ja_text"] == "あ い"
    assert prepared[0]["zh_text"] == "甲，乙"


def test_write_bilingual_srt_returns_normalized_blocks(tmp_path):
    path = tmp_path / "normalized.srt"
    blocks = [
        {"start": 0.0, "end": 1.2, "ja_text": "あ", "zh_text": "甲"},
        {"start": 1.0, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
    ]
    options = SubtitleOptions(video_fps=25.0, merge_adjacent=False)

    prepared = subtitle.prepare_srt_blocks(blocks, options=options, mode="bilingual")
    written = subtitle.write_bilingual_srt(prepared, str(path), options=options)

    assert written[0]["end"] == pytest.approx(0.92)
    assert "00:00:00,000 --> 00:00:00,920" in path.read_text(encoding="utf-8")


def test_merge_adjacent_short_blocks(tmp_path):
    path = tmp_path / "merged.srt"
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "いい", "zh_text": "好"},
        {"start": 1.1, "end": 2.0, "ja_text": "もっと", "zh_text": "更多"},
    ]

    prepared = subtitle.prepare_srt_blocks(blocks, mode="bilingual")
    subtitle.write_bilingual_srt(prepared, str(path))

    content = path.read_text(encoding="utf-8")
    assert _cue_count(content) == 1
    assert "いい もっと" in content
    assert "好，更多" in content


def test_merge_adjacent_short_blocks_stops_after_sentence_punctuation(tmp_path):
    path = tmp_path / "blocked.srt"
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "終わり。", "zh_text": "结束。"},
        {"start": 1.05, "end": 2.0, "ja_text": "次", "zh_text": "下一句"},
    ]

    prepared = subtitle.prepare_srt_blocks(blocks, mode="bilingual")
    subtitle.write_bilingual_srt(prepared, str(path))

    content = path.read_text(encoding="utf-8")
    assert _cue_count(content) == 2


def test_soft_split_prefers_translated_sentence_punctuation(monkeypatch, tmp_path):
    path = tmp_path / "soft_zh.srt"
    blocks = [
        {
            "start": 0.0,
            "end": 7.0,
            "ja_text": "これはいい次が来る",
            "zh_text": "这是第一句。然后继续",
            "words": [
                _word("これは", 0.0, 1.5),
                _word("いい", 1.5, 3.0),
                _word("次が", 3.0, 5.0),
                _word("来る", 5.0, 7.0),
            ],
        }
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(soft_split_enabled=True, soft_max=6.0),
        mode="bilingual",
    )
    subtitle.write_bilingual_srt(
        prepared,
        str(path),
        options=SubtitleOptions(soft_split_enabled=True, soft_max=6.0),
    )

    content = path.read_text(encoding="utf-8")
    assert _cue_count(content) == 2
    assert "これはいい" in content
    assert "次が来る" in content
    assert "这是第一句。" in content
    assert "然后继续" in content
    assert "00:00:03,000" in content


def test_soft_split_falls_back_to_japanese_particle_boundary(monkeypatch, tmp_path):
    path = tmp_path / "soft_particle.srt"
    blocks = [
        {
            "start": 0.0,
            "end": 7.2,
            "ja_text": "私もあなたに近づきたい今すぐ",
            "zh_text": "我也想要靠近你现在就继续",
            "words": [
                _word("私も", 0.0, 2.0),
                _word("あなたに", 2.0, 4.0),
                _word("近づきたい", 4.0, 5.6),
                _word("今すぐ", 5.6, 7.2),
            ],
        }
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(soft_split_enabled=True, soft_max=6.0),
        mode="bilingual",
    )
    subtitle.write_bilingual_srt(
        prepared,
        str(path),
        options=SubtitleOptions(soft_split_enabled=True, soft_max=6.0),
    )

    content = path.read_text(encoding="utf-8")
    assert _cue_count(content) == 2
    assert "私もあなたに" in content
    assert "近づきたい今すぐ" in content


def test_soft_split_does_not_change_blocks_without_words(monkeypatch, tmp_path):
    path = tmp_path / "fallback_no_words.srt"
    blocks = [
        {
            "start": 0.0,
            "end": 7.0,
            "ja_text": "これはいい。次が来る",
            "zh_text": "这是第一句。然后继续",
        }
    ]

    prepared = subtitle.prepare_srt_blocks(
        blocks,
        options=SubtitleOptions(soft_split_enabled=True, soft_max=6.0),
        mode="bilingual",
    )
    subtitle.write_bilingual_srt(
        prepared,
        str(path),
        options=SubtitleOptions(soft_split_enabled=True, soft_max=6.0),
    )

    content = path.read_text(encoding="utf-8")
    assert _cue_count(content) == 1
    assert "这是第一句。然后继续" in content


def test_soft_split_recurses_until_hard_duration_is_respected(monkeypatch):
    words = [_word(f"語{index}", index * 2.0, (index + 1) * 2.0) for index in range(9)]
    block = {
        "start": 0.0,
        "end": 18.0,
        "ja_text": "".join(word["word"] for word in words),
        "zh_text": "第一句。后续内容没有更多句末标点但仍然很长",
        "words": words,
        "gender": "M",
    }

    options = SubtitleOptions(soft_split_enabled=True, soft_max=5.5, max_duration=6.5)
    split = subtitle._soft_split_subtitle_blocks([block], options=options)

    assert len(split) >= 3
    assert all(part["end"] - part["start"] <= 6.5 for part in split)


def test_none_gender_over_soft_limit_gets_hard_split(monkeypatch):
    words = [_word(f"語{i}", i * 1.4, (i + 1) * 1.4) for i in range(5)]
    block = {
        "start": 0.0,
        "end": 7.0,
        "ja_text": "".join(w["word"] for w in words),
        "zh_text": "这段没有句末标点无法找到分割点",
        "words": words,
        "gender": None,
    }
    options = SubtitleOptions(soft_split_enabled=True, soft_max=5.5, max_duration=6.5)
    split = subtitle._soft_split_subtitle_blocks([block], options=options)
    assert len(split) == 2


def test_non_none_gender_over_soft_limit_below_hard_limit_no_forced_split(monkeypatch):
    words = [_word(f"語{i}", i * 1.4, (i + 1) * 1.4) for i in range(5)]
    block = {
        "start": 0.0,
        "end": 6.2,
        "ja_text": "".join(w["word"] for w in words),
        "zh_text": "这段没有句末标点无法找到分割点",
        "words": words,
        "gender": "F",
    }
    options = SubtitleOptions(soft_split_enabled=True, soft_max=5.5, max_duration=6.5)
    split = subtitle._soft_split_subtitle_blocks([block], options=options)
    assert len(split) == 1


def test_none_gender_below_soft_limit_not_split(monkeypatch):
    words = [_word(f"語{i}", i * 1.0, (i + 1) * 1.0) for i in range(4)]
    block = {
        "start": 0.0,
        "end": 5.4,
        "ja_text": "".join(w["word"] for w in words),
        "zh_text": "短于软上限",
        "words": words,
        "gender": None,
    }
    options = SubtitleOptions(soft_split_enabled=True, soft_max=5.5, max_duration=6.5)
    split = subtitle._soft_split_subtitle_blocks([block], options=options)
    assert len(split) == 1


def test_gender_same_adjacent_short_blocks_merge():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "いい", "zh_text": "好", "gender": "F"},
        {"start": 1.1, "end": 2.0, "ja_text": "もっと", "zh_text": "更多", "gender": "F"},
    ]

    merged = subtitle._merge_adjacent_short_blocks(blocks)

    assert len(merged) == 1
    assert merged[0]["ja_text"] == "いい もっと"
    assert merged[0]["zh_text"] == "好，更多"


def test_gender_different_long_enough_blocks_do_not_merge():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "来て", "zh_text": "过来", "gender": "M"},
        {"start": 1.1, "end": 2.0, "ja_text": "いや", "zh_text": "不要", "gender": "F"},
    ]

    merged = subtitle._merge_adjacent_short_blocks(blocks)

    assert len(merged) == 2


def test_short_overlapping_tail_merges_across_gender_guard():
    blocks = [
        {
            "start": 176.84,
            "end": 178.493,
            "ja_text": "アルマリスト 室で イラックス ステイマンを受け",
            "zh_text": "在芳香蒸汽室接受放松",
            "gender": "F",
        },
        {
            "start": 178.56,
            "end": 179.16,
            "ja_text": "受けて",
            "zh_text": "接受",
            "gender": "M",
        },
    ]

    merged = subtitle._merge_adjacent_short_blocks(
        blocks,
        options=SubtitleOptions(video_fps=29.97),
    )

    assert len(merged) == 1
    assert merged[0]["ja_text"] == "アルマリスト 室で イラックス ステイマンを受けて"
    assert merged[0]["zh_text"] == "在芳香蒸汽室接受放松，接受"
    assert merged[0]["gender"] is None


def test_short_overlapping_tail_merges_compacted_text_without_duplicate():
    blocks = [
        {
            "start": 0.0,
            "end": 1.0,
            "ja_text": "受け",
            "zh_text": "接受",
            "gender": "F",
        },
        {
            "start": 1.06,
            "end": 1.5,
            "ja_text": "受 けて",
            "zh_text": "接受",
            "gender": "M",
        },
    ]

    merged = subtitle._merge_adjacent_short_blocks(
        blocks,
        options=SubtitleOptions(video_fps=29.97),
    )

    assert len(merged) == 1
    assert merged[0]["ja_text"] == "受けて"


def test_short_tail_without_text_overlap_keeps_gender_guard():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "来て", "zh_text": "过来", "gender": "M"},
        {"start": 1.06, "end": 1.6, "ja_text": "いや", "zh_text": "不要", "gender": "F"},
    ]

    merged = subtitle._merge_adjacent_short_blocks(
        blocks,
        options=SubtitleOptions(video_fps=29.97),
    )

    assert len(merged) == 2


def test_frame_rate_controls_short_tail_merge_gap():
    blocks = [
        {
            "start": 0.0,
            "end": 1.0,
            "ja_text": "受け",
            "zh_text": "接受",
            "gender": "F",
        },
        {
            "start": 1.1,
            "end": 1.9,
            "ja_text": "受けて",
            "zh_text": "接受",
            "gender": "M",
        },
    ]

    assert len(
        subtitle._merge_adjacent_short_blocks(
            blocks,
            options=SubtitleOptions(video_fps=24.0),
        )
    ) == 1
    assert len(
        subtitle._merge_adjacent_short_blocks(
            blocks,
            options=SubtitleOptions(video_fps=30.0),
        )
    ) == 2


def test_gender_different_short_fragment_still_merges():
    blocks = [
        {"start": 0.0, "end": 0.4, "ja_text": "ん", "zh_text": "嗯", "gender": "M"},
        {"start": 0.5, "end": 1.3, "ja_text": "いい", "zh_text": "舒服", "gender": "F"},
    ]

    merged = subtitle._merge_adjacent_short_blocks(blocks)

    assert len(merged) == 1


def test_gender_none_does_not_block_merge():
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "ん", "zh_text": "嗯", "gender": None},
        {"start": 1.1, "end": 2.0, "ja_text": "いい", "zh_text": "舒服", "gender": "F"},
    ]

    merged = subtitle._merge_adjacent_short_blocks(blocks)

    assert len(merged) == 1


def test_write_srt_can_show_gender_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBTITLE_SHOW_GENDER", "1")
    path = tmp_path / "gender_on.srt"

    subtitle.write_srt(
        [{"start": 0.0, "end": 1.0, "zh_text": "过来", "gender": "M"}],
        str(path),
    )

    assert "[M] 过来" in path.read_text(encoding="utf-8")


def test_write_srt_hides_gender_prefix_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBTITLE_SHOW_GENDER", "0")
    path = tmp_path / "gender_off.srt"

    subtitle.write_srt(
        [{"start": 0.0, "end": 1.0, "zh_text": "过来", "gender": "M"}],
        str(path),
    )

    content = path.read_text(encoding="utf-8")
    assert "[M]" not in content
    assert "过来" in content
