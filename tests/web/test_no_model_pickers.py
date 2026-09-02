"""Neither the local translation model nor the ASR model is user-selectable.

Decided 2026-08-05: one local translation model ships (now Hy-MT2-7B Q4) and one ASR
model ships (the galgame Qwen3-ASR), so a picker can only offer a wrong answer.
Every model choice that used to exist here was removed after the model behind it
was retired, and each removal was found late - the GGUF preset, the Transformers
backend's Qwen3 tier list, the 0.6B ASR tier. This file is the standing check so
the next one is found by the suite instead of by a user.

`api-model` is deliberately allowed: that is the *remote* API model name, which
only the user can know. The local backend is the fixed Hy-MT2 + llama.cpp stack;
even a free-form GGUF path would be a model picker and may not appear.

`api-model` itself is a filterable text input (`js/modelCombobox.js`) rather than
a `<select>`, since providers with long model lists need to be searchable - but
the combobox still only accepts values that came back from `/api/models`, so
this stays a "pick from what the vendor offers" control, not free text. Hence
it does not appear in `ALLOWED_SELECT_IDS` below, which only inventories native
`<select>` elements.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"
INDEX = STATIC / "index.html"

# Every `<select>` the settings page may contain. A new entry here is a
# deliberate decision to give the user a choice, not an implementation detail.
ALLOWED_SELECT_IDS = {
    "proxy-protocol",
    "r-mode",
    "translation-backend",
    "api-reasoning-effort",
    "api-target-lang",
}


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_no_unreviewed_dropdown_exists() -> None:
    found = set(re.findall(r'<select[^>]*\bid="([^"]+)"', _html()))
    assert found == ALLOWED_SELECT_IDS, (
        f"unexpected: {sorted(found - ALLOWED_SELECT_IDS)}, "
        f"missing: {sorted(ALLOWED_SELECT_IDS - found)}"
    )


def test_the_backend_selector_offers_only_the_two_that_exist() -> None:
    """`local` (in-process Transformers) was removed on 2026-08-05. An option
    string left behind would post a 422 from `SettingsUpdate`."""
    html = _html()
    block = html[html.index('<select id="translation-backend"') :]
    block = block[: block.index("</select>")]
    assert set(re.findall(r'value="([^"]*)"', block)) == {"openai", "llamacpp"}


def test_no_local_translation_model_can_be_named_in_the_browser() -> None:
    """The local path is fixed to Hy-MT2; no preset or arbitrary GGUF input."""
    sources = [INDEX] + sorted((STATIC / "js").glob("*.js"))
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for token in ("Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3.5", "local-model-preset"):
            assert token not in text, f"{path.name} still names {token}"
        assert "llamacpp-gguf-path" not in text


def test_local_backend_is_described_as_the_fixed_hymt2_stack() -> None:
    html = _html()
    assert "本地翻译（Hy-MT2-7B Q4 · llama.cpp）" in html
    assert "Hy-MT2-7B Q4_K_M" in html
    assert "推荐本地" not in html

    from web.models import SettingsRead, SettingsUpdate

    for removed in ("llamacpp_model_repo", "llamacpp_model_file", "llamacpp_gguf_path"):
        assert removed not in SettingsRead.model_fields
        assert removed not in SettingsUpdate.model_fields


def test_no_asr_model_can_be_chosen_in_the_browser() -> None:
    sources = [INDEX] + sorted((STATIC / "js").glob("*.js"))
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for token in ("asr-backend", "asr_backend", "ASR_BACKEND"):
            assert token not in text, f"{path.name} still exposes {token}"


def test_the_shipped_defaults_are_the_two_models_that_ship() -> None:
    """The other half of "no picker": if nothing can be chosen, the default has
    to be right. Reads the real settings rather than restating them."""
    from core.config import DEFAULT_SETTINGS

    assert DEFAULT_SETTINGS["LLAMACPP_MODEL_REPO"] == "tencent/Hy-MT2-7B-GGUF"
    assert DEFAULT_SETTINGS["LLAMACPP_MODEL_FILE"] == "Hy-MT2-7B-Q4_K_M.gguf"
    assert DEFAULT_SETTINGS["LLAMACPP_PARALLEL"] == "8"
    assert (
        DEFAULT_SETTINGS["ASR_BACKEND"]
        == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    )
