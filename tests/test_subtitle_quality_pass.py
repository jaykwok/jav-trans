import re

from subtitles import writer as subtitle
def test_bilingual_srt_uses_placeholder_for_empty_translation(tmp_path, caplog):
    path = tmp_path / "out.srt"

    subtitle.write_bilingual_srt(
        [{"start": 0.0, "end": 1.0, "ja_text": "いい", "zh_text": ""}],
        str(path),
    )

    content = path.read_text(encoding="utf-8")
    assert "「未翻译」" in content
    assert "Empty translated subtitle" in caplog.text


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


def test_merge_adjacent_short_blocks(tmp_path):
    path = tmp_path / "merged.srt"
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "いい", "zh_text": "好"},
        {"start": 1.1, "end": 2.0, "ja_text": "もっと", "zh_text": "更多"},
    ]

    subtitle.write_bilingual_srt(blocks, str(path))

    content = path.read_text(encoding="utf-8")
    assert len(re.findall(r"^\d+$", content, flags=re.MULTILINE)) == 1
    assert "いい もっと" in content
    assert "好，更多" in content


def test_merge_adjacent_short_blocks_stops_after_sentence_punctuation(tmp_path):
    path = tmp_path / "blocked.srt"
    blocks = [
        {"start": 0.0, "end": 1.0, "ja_text": "終わり。", "zh_text": "结束。"},
        {"start": 1.05, "end": 2.0, "ja_text": "次", "zh_text": "下一句"},
    ]

    subtitle.write_bilingual_srt(blocks, str(path))

    content = path.read_text(encoding="utf-8")
    assert len(re.findall(r"^\d+$", content, flags=re.MULTILINE)) == 2


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

