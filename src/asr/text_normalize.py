from __future__ import annotations

import re
import unicodedata


_STRIP_PUNCT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]\s~〜ー-]+")
_SPACE_RE = re.compile(r"[ \t]+")


def normalize_display_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").strip()
    return _SPACE_RE.sub(" ", cleaned).strip()


def strip_text_punctuation(text: str) -> str:
    return _STRIP_PUNCT_RE.sub("", text or "")
