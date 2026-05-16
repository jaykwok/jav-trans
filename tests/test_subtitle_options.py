from subtitles.options import SubtitleOptions
from subtitles import writer as subtitle


def test_subtitle_options_from_env(monkeypatch):
    monkeypatch.setenv("MAX_SUBTITLE_DURATION", "9.5")
    monkeypatch.setenv("SUBTITLE_SOFT_MAX_S", "7.5")
    monkeypatch.setenv("SUBTITLE_SOFT_SPLIT_ENABLED", "0")
    monkeypatch.setenv("SRT_LINE_MAX_CHARS", "18")
    monkeypatch.setenv("SUBTITLE_SHOW_SPEAKER", "1")
    monkeypatch.setenv("SUBTITLE_SHOW_GENDER", "1")

    options = SubtitleOptions.from_env()

    assert options.max_duration == 9.5
    assert options.soft_max == 7.5
    assert options.soft_split_enabled is False
    assert options.line_max_chars == 18
    assert options.show_speaker is True
    assert options.show_gender is True


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
