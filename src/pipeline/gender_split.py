from __future__ import annotations

import os
import re
import unicodedata
from typing import Callable


_ASR_NOISE_ONLY_RE = re.compile(r"^[\s\"'`“”‘’「」『』（）()\[\]［］【】〈〉《》<>]+$")
_ASR_JA_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_ASR_LONG_LATIN_RE = re.compile(r"[A-Za-z]{4,}")
_ASR_ASCII_OR_WESTERN_PUNCT_RE = re.compile(r"^[\s\x00-\x7F“”‘’…]+$")


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


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


def _segment_duration(segment: dict) -> float:
    try:
        return max(0.0, float(segment["end"]) - float(segment["start"]))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _meets_gender_turn_duration_floor(segment: dict) -> bool:
    min_duration = _env_float("SUBTITLE_MIN_DURATION_GENDER_TURN", 0.4)
    return _segment_duration(segment) >= min_duration


def _flush_group(
    groups: list[list[dict]],
    words: list[dict],
    *,
    force: bool = False,
) -> None:
    if not words:
        return
    if force or any(word_gender_value(word) is not None for word in words):
        groups.append(list(words))


def _known_gender_word_count(words: list[dict], gender: str | None) -> int:
    if gender is None:
        return 0
    return sum(1 for word in words if word_gender_value(word) == gender)


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
    pending_nones: list[dict] = []
    active_gender: str | None = None
    none_tolerance = max(1, _env_int("F0_GENDER_NONE_TOLERANCE", 3))

    for word in words:
        gender = word_gender_value(word)
        if gender is None:
            pending_nones.append(word)
            continue

        if active_gender is None:
            if len(pending_nones) >= none_tolerance:
                _flush_group(groups, pending_nones, force=True)
                pending_nones = []
                current_words = [word]
                active_gender = gender
                continue
            if pending_nones:
                current_words.extend(pending_nones)
                pending_nones = []
            current_words.append(word)
            active_gender = gender
            continue

        if len(pending_nones) >= none_tolerance:
            _flush_group(groups, current_words)
            _flush_group(groups, pending_nones, force=True)
            current_words = [word]
            active_gender = gender
            pending_nones = []
            continue

        if gender == active_gender:
            if pending_nones:
                current_words.extend(pending_nones)
                pending_nones = []
            current_words.append(word)
            continue

        if pending_nones:
            if _known_gender_word_count(current_words, active_gender) >= none_tolerance:
                _flush_group(groups, current_words)
                _flush_group(groups, pending_nones, force=True)
            elif none_tolerance >= 3 and len(pending_nones) == none_tolerance - 1:
                current_words.extend(pending_nones)
                current_words.append(word)
                pending_nones = []
                active_gender = gender
                continue
            else:
                current_words.extend(pending_nones)
                _flush_group(groups, current_words)
            pending_nones = []
        else:
            _flush_group(groups, current_words)
        current_words = [word]
        active_gender = gender

    if pending_nones:
        current_words.extend(pending_nones)
    if current_words:
        _flush_group(groups, current_words, force=active_gender is None)

    if len(groups) < 2:
        return [segment]

    rebuilt_segments = [
        rebuilt
        for group in groups
        for rebuilt in [segment_from_f0_words(segment, group)]
        if rebuilt is not None
    ]
    if not all(_meets_gender_turn_duration_floor(item) for item in rebuilt_segments):
        return [segment]
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
