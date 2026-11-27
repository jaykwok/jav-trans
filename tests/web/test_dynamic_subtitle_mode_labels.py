from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"


def test_subtitle_mode_labels_derive_dynamically_from_target_language() -> None:
    settings = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")

    assert "if (mode === 'ja_only') return '仅日文（不翻译）';" in settings
    assert "if (mode === 'bilingual') return `双语（日文 + ${language}）`;" in settings
    assert "return language;" in settings
    assert "中日双语" not in settings


def test_target_language_changes_refresh_subtitle_mode_options() -> None:
    settings = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")
    main = (STATIC / "js" / "main.js").read_text(encoding="utf-8")

    assert "$('api-target-lang')?.addEventListener('change', updateSubtitleModeLabels)" in settings
    assert "$('r-mode')?.addEventListener('change', updateSkipTransState)" in settings
    assert "option.textContent = subtitleModeLabel(option.value, targetLang)" in settings
    assert main.index("applyFormMemory();") < main.index("updateSubtitleModeLabels();")
    assert main.index("updateSubtitleModeLabels();") < main.index("updateSkipTransState();")


def test_subtitle_output_selector_merges_skip_translation_and_maps_to_job_spec() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    files_js = (STATIC / "js" / "files.js").read_text(encoding="utf-8")

    assert "📁 视频与字幕" in html
    assert 'id="r-mode"' in html
    assert 'value="ja_only"' in html
    assert 'id="r-skip-translation"' not in html
    assert "subtitle_mode:            outputMode === 'bilingual' ? 'bilingual' : 'zh'" in files_js
    assert "skip_translation:         outputMode === 'ja_only'" in files_js
