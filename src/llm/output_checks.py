"""Cheap output checks independent of JSON or line-oriented model contracts."""

from __future__ import annotations

import re
import unicodedata

_KANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")


def is_chinese_target(target_lang: str) -> bool:
    return any(token in str(target_lang).lower() for token in ("中文", "简体", "簡體", "繁体", "繁體", "chinese"))


def invalid_line_output(source: str, target: str, target_lang: str) -> str | None:
    if not str(target or "").strip():
        return "empty_translation"
    if not is_chinese_target(target_lang):
        return None
    # Shared Han characters and Latin names can legitimately be unchanged.
    # Kana is positive evidence that a Chinese-target request was not finished.
    if _KANA.search(target):
        fold = lambda text: re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text))
        return "source_echo" if fold(source) == fold(target) else "japanese_remaining"
    return None
