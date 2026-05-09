from __future__ import annotations

import re
import unicodedata
from typing import Callable


_ASR_NOISE_ONLY_RE = re.compile(r"^[\s\"'`“”‘’「」『』（）()\[\]［］【】〈〉《》<>]+$")
_ASR_JA_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_ASR_LONG_LATIN_RE = re.compile(r"[A-Za-z]{4,}")
_ASR_ASCII_OR_WESTERN_PUNCT_RE = re.compile(r"^[\s\x00-\x7F“”‘’…]+$")


def _has_language_or_number_signal(text: str) -> bool:
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in text)


def _is_symbol_only_noise(text: str) -> bool:
    compact = "".join(char for char in text if not char.isspace())
    return bool(compact) and not _has_language_or_number_signal(compact)


def filter_f0_none_segments(
    segments: list[dict],
    *,
    f0_failed: bool = False,
    enabled: bool | None = None,
    default_enabled: Callable[[], bool] | None = None,
    warn: Callable[[str], None] | None = None,
) -> tuple[list[dict], int]:
    if enabled is None:
        enabled = default_enabled() if default_enabled is not None else False
    if not enabled:
        return segments, 0
    if f0_failed:
        if warn is not None:
            warn("F0 detection failed; skipping non-voice filter to preserve all segments")
        return segments, 0
    filtered = [segment for segment in segments if segment.get("gender") is not None]
    return filtered, len(segments) - len(filtered)


def word_gender_value(word: dict) -> str | None:
    gender = word.get("gender")
    return gender if gender in {"M", "F"} else None


def word_text(word: dict) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def gender_for_words(words: list[dict]) -> str | None:
    genders = [word_gender_value(word) for word in words]
    genders = [gender for gender in genders if gender is not None]
    if not genders:
        return None
    return max(("M", "F"), key=genders.count)


def segment_from_f0_words(segment: dict, words: list[dict]) -> dict | None:
    if not words:
        return None
    text = "".join(word_text(word) for word in words).strip()
    if not text:
        return None
    try:
        start = min(float(word.get("start", segment.get("start", 0.0))) for word in words)
        end = max(float(word.get("end", segment.get("end", start))) for word in words)
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    rebuilt = dict(segment)
    rebuilt.update(
        {
            "start": start,
            "end": end,
            "text": text,
            "words": [dict(word) for word in words],
            "gender": gender_for_words(words),
            "source_chunk_index": words[0].get(
                "source_chunk_index",
                segment.get("source_chunk_index"),
            ),
        }
    )
    return rebuilt


def split_segment_on_f0_gender_turns(segment: dict) -> list[dict]:
    words = [dict(word) for word in segment.get("words") or [] if isinstance(word, dict)]
    if len(words) < 2:
        return [segment]
    try:
        words.sort(
            key=lambda word: (
                float(word.get("start", segment.get("start", 0.0))),
                float(word.get("end", segment.get("end", 0.0))),
            )
        )
    except (TypeError, ValueError):
        return [segment]

    groups: list[list[dict]] = []
    current_words: list[dict] = []
    active_gender: str | None = None
    for word in words:
        gender = word_gender_value(word)
        if (
            current_words
            and gender is not None
            and active_gender is not None
            and gender != active_gender
        ):
            groups.append(current_words)
            current_words = [word]
            active_gender = gender
            continue
        current_words.append(word)
        if gender is not None:
            active_gender = gender
    if current_words:
        groups.append(current_words)

    if len(groups) < 2:
        return [segment]

    rebuilt_segments = [
        rebuilt
        for group in groups
        for rebuilt in [segment_from_f0_words(segment, group)]
        if rebuilt is not None
    ]
    return rebuilt_segments if len(rebuilt_segments) >= 2 else [segment]


def split_segments_on_f0_gender_turns(segments: list[dict]) -> tuple[list[dict], int]:
    split_segments: list[dict] = []
    split_count = 0
    for segment in segments:
        pieces = split_segment_on_f0_gender_turns(segment)
        split_count += max(0, len(pieces) - 1)
        split_segments.extend(pieces)
    return split_segments, split_count


def is_asr_noise_text(text) -> bool:
    cleaned = str(text or "").strip().replace("\u200b", "").replace("\ufeff", "")
    if not cleaned:
        return True
    unescaped = cleaned.replace("\\", "").strip()
    if _ASR_NOISE_ONLY_RE.fullmatch(unescaped):
        return True
    if _is_symbol_only_noise(unescaped):
        return True
    if (
        _ASR_LONG_LATIN_RE.search(unescaped)
        and not _ASR_JA_OR_CJK_RE.search(unescaped)
        and _ASR_ASCII_OR_WESTERN_PUNCT_RE.fullmatch(unescaped)
    ):
        return True
    return False


def filter_asr_noise_segments(segments: list[dict]) -> tuple[list[dict], int]:
    filtered: list[dict] = []
    for segment in segments:
        text = segment.get("text", segment.get("ja_text", segment.get("ja", "")))
        if is_asr_noise_text(text):
            continue
        filtered.append(segment)
    return filtered, len(segments) - len(filtered)
