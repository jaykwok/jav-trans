from __future__ import annotations

import re
import unicodedata


_ASR_NOISE_ONLY_RE = re.compile(r"^[\s\"'`“”‘’「」『』（）()\[\]［］【】〈〉《》<>]+$")


def _has_language_or_number_signal(text: str) -> bool:
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in text)


def _is_symbol_only_noise(text: str) -> bool:
    compact = "".join(char for char in text if not char.isspace())
    return bool(compact) and not _has_language_or_number_signal(compact)


def is_asr_noise_text(text) -> bool:
    cleaned = str(text or "").strip().replace("\u200b", "").replace("\ufeff", "")
    if not cleaned:
        return True
    unescaped = cleaned.replace("\\", "").strip()
    if _ASR_NOISE_ONLY_RE.fullmatch(unescaped):
        return True
    if _is_symbol_only_noise(unescaped):
        return True
    return False


def find_asr_noise_segments(segments: list[dict], *, limit: int = 50) -> list[dict]:
    items: list[dict] = []
    for index, segment in enumerate(segments):
        text = segment.get("text", segment.get("ja_text", segment.get("ja", "")))
        if not is_asr_noise_text(text):
            continue
        items.append(
            {
                "position": index,
                "source_chunk_index": segment.get("source_chunk_index"),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": str(text or ""),
            }
        )
        if len(items) >= limit:
            break
    return items
