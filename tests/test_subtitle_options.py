from subtitles.options import SubtitleOptions
from subtitles import writer as subtitle


def test_subtitle_options_from_env(monkeypatch):
    monkeypatch.setenv("MAX_SUBTITLE_DURATION", "9.5")
    monkeypatch.setenv("SUBTITLE_SOFT_MAX_S", "7.5")
    monkeypatch.setenv("SUBTITLE_SOFT_SPLIT_ENABLED", "0")
    monkeypatch.setenv("SRT_LINE_MAX_CHARS", "18")
    monkeypatch.setenv("SUBTITLE_TIMELINE_MODE", "reading")
    monkeypatch.setenv("SUBTITLE_READING_CPS", "10")
    monkeypatch.setenv("SUBTITLE_GAP_PADDING", "0.25")
    monkeypatch.setenv("SUBTITLE_MERGE_ADJACENT", "0")
    monkeypatch.setenv("SUBTITLE_SHOW_SPEAKER", "1")
    monkeypatch.setenv("SUBTITLE_SHOW_GENDER", "1")

    options = SubtitleOptions.from_env()

    assert options.max_duration == 9.5
    assert options.soft_max == 7.5
    assert options.soft_split_enabled is False
    assert options.timeline_mode == "reading"
    assert options.reading_cps == 10
    assert options.gap_padding == 0.25
    assert options.merge_adjacent is False
    assert options.line_max_chars == 18
    assert options.show_speaker is True
    assert options.show_gender is True


def test_subtitle_options_defaults_are_conservative(monkeypatch):
    monkeypatch.delenv("MAX_SUBTITLE_DURATION", raising=False)
    monkeypatch.delenv("SUBTITLE_SOFT_MAX_S", raising=False)

    options = SubtitleOptions.from_env()

    assert options.max_duration == 6.5
    assert options.soft_max == 5.5


def test_write_srt_explicit_options_override_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBTITLE_SHOW_SPEAKER", "0")
    monkeypatch.setenv("SUBTITLE_SHOW_GENDER", "0")
    path = tmp_path / "speaker.srt"

    subtitle.write_srt(
        [{"start": 0.0, "end": 1.0, "zh_text": "过来", "speaker": "S0", "gender": "M"}],
        str(path),
        options=SubtitleOptions(show_speaker=True, show_gender=False),
    )

    content = path.read_text(encoding="utf-8")
    assert "[S0] 过来" in content
    assert "[SS0]" not in content
    assert "[M]" not in content


def test_write_srt_timeline_mode_is_per_options(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBTITLE_TIMELINE_MODE", "alignment")
    path = tmp_path / "reading.srt"

    subtitle.write_srt(
        [{"start": 0.0, "end": 10.0, "zh_text": "短句"}],
        str(path),
        options=SubtitleOptions(
            timeline_mode="reading",
            reading_cps=20.0,
            reading_base=0.0,
            min_duration=0.6,
            duration_ratio_cap=1.0,
            duration_grace=0.0,
            max_duration=8.0,
        ),
    )

    content = path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:00,600" in content


def test_write_bilingual_merge_adjacent_is_per_options(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBTITLE_MERGE_ADJACENT", "1")
    path = tmp_path / "no-merge.srt"

    subtitle.write_bilingual_srt(
        [
            {"start": 0.0, "end": 1.0, "ja_text": "あ", "zh_text": "甲"},
            {"start": 1.05, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
        ],
        str(path),
        options=SubtitleOptions(merge_adjacent=False),
    )

    content = path.read_text(encoding="utf-8")
    assert "\n1\n" not in content
    assert content.count("-->") == 2
