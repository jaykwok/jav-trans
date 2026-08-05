"""Neither the local translation model nor the ASR model is user-selectable.

Decided 2026-08-05: one local translation model ships (Hy-MT2-1.8B) and one ASR
model ships (the galgame Qwen3-ASR), so a picker can only offer a wrong answer.
Every model choice that used to exist here was removed after the model behind it
was retired, and each removal was found late - the GGUF preset, the Transformers
backend's Qwen3 tier list, the 0.6B ASR tier. This file is the standing check so
the next one is found by the suite instead of by a user.

`api-model` is deliberately allowed: that is the *remote* API model name, which
only the user can know.
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
    "api-model",
    "api-reasoning-effort",
    "api-format",
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
    """A GGUF path box stays (an escape hatch for a user who brings their own),
    but no list of model names may reappear."""
    sources = [INDEX] + sorted((STATIC / "js").glob("*.js"))
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for token in ("Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3.5", "local-model-preset"):
            assert token not in text, f"{path.name} still names {token}"


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

    assert DEFAULT_SETTINGS["LLAMACPP_MODEL_REPO"] == "tencent/Hy-MT2-1.8B-GGUF"
    assert DEFAULT_SETTINGS["LLAMACPP_MODEL_FILE"] == "Hy-MT2-1.8B-Q8_0.gguf"
    assert (
        DEFAULT_SETTINGS["ASR_BACKEND"]
        == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    )
