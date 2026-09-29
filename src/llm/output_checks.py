"""Cheap output checks independent of JSON or line-oriented model contracts."""

from __future__ import annotations

import re
import unicodedata

from llm.zh_variant import target_variant

# Kana letters, iteration/voicing marks and halfwidth forms. Middle dots and
# prolonged sound marks alone are not evidence of untranslated Japanese: they
# also occur as separators or expressive punctuation in translated subtitles.
# Keep the raw character ranges shared with GBNF; normalizing only the check
# would let a constrained reply bypass the grammar with halfwidth kana.
_KANA_RANGES = (
    (0x3041, 0x3096), (0x3099, 0x309F),
    (0x30A1, 0x30FA), (0x30FD, 0x30FF), (0x31F0, 0x31FF),
    (0xFF66, 0xFF6F), (0xFF71, 0xFF9F),
)
_KANA_CLASS = "".join(f"\\u{low:04X}-\\u{high:04X}" for low, high in _KANA_RANGES)
_KANA = re.compile(f"[{_KANA_CLASS}]")

# The verdicts a kana-free reply cannot earn.
KANA_REASONS = frozenset({"source_echo", "japanese_remaining"})
# GBNF for llama.cpp: any non-empty reply without a character `_KANA` rejects.
NO_KANA_GRAMMAR = f"root ::= [^{_KANA_CLASS}]+"


def is_chinese_target(target_lang: str) -> bool:
    label = str(target_lang or "").strip().lower()
    return (
        target_variant(label) is not None
        or label == "zh"
        or any(token in label for token in ("中文", "chinese"))
    )


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
