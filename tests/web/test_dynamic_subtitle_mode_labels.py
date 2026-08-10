from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"


def test_subtitle_mode_labels_cover_every_target_language() -> None:
    settings = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")

    for expected in (
        "简体中文字幕（仅译文）",
        "中日双语字幕（简体）",
        "繁體中文字幕（仅译文）",
        "中日双语字幕（繁體）",
        "英文字幕（仅译文）",
        "英日双语字幕",
    ):
        assert expected in settings


def test_target_language_changes_refresh_subtitle_mode_options() -> None:
    settings = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")
    main = (STATIC / "js" / "main.js").read_text(encoding="utf-8")

    assert "$('api-target-lang')?.addEventListener('change', updateSubtitleModeLabels)" in settings
    assert "option.textContent = subtitleModeLabel(option.value, targetLang)" in settings
    assert main.index("applyFormMemory();") < main.index("updateSubtitleModeLabels();")


def test_custom_settings_explain_the_target_language_link() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'id="r-mode"' in html
    assert "选项名称会随“翻译设置 → 目标语言”自动更新。" in html
