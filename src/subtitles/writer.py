import os
import re
import logging
from pathlib import Path
from typing import Literal

from subtitles.options import SubtitleOptions

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


_COMPACT_SPACE_RE = re.compile(r"\s+")
_WRAP_PUNCTUATION = "，、。！？…"
_SENTENCE_END_PUNCTUATION = "。！？…"
_ZH_SOFT_SPLIT_PUNCTUATION = "。？！"
_JA_PARTICLE_SPLIT_SUFFIXES = (
    "は",
    "が",
    "を",
    "に",
    "で",
    "も",
    "と",
    "へ",
    "の",
    "や",
    "から",
    "まで",
    "より",
)
_JA_BOUNDARY_STRIP_CHARS = " \t\r\n、。！？!?…"
_ADJACENT_SHORT_MERGE_MAX_GAP_FRAMES = 6.0
_ADJACENT_SHORT_MERGE_MAX_DURATION_FRAMES = 120.0
_OVERLAPPING_TAIL_MAX_GAP_FRAMES = 2.5
_OVERLAPPING_TAIL_MAX_DURATION_FRAMES = 20.5


def _count_text_units(text: str, *, ascii_char_weight: float = 0.55) -> float:
    compact = _COMPACT_SPACE_RE.sub("", (text or "").strip())
    if not compact:
        return 0.0

    units = 0.0
    for char in compact:
        if char.isascii() and char.isalnum():
            units += ascii_char_weight
        elif char.isascii():
            units += 0.35
        else:
            units += 1.0
    return units


def _coerce_options(options: SubtitleOptions | None = None) -> SubtitleOptions:
    return options if options is not None else SubtitleOptions.from_env()


def _estimate_reading_duration(
    block: dict,
    *,
    options: SubtitleOptions | None = None,
) -> float:
    options = _coerce_options(options)
    ja_units = _count_text_units(
        block.get("ja_text", ""),
        ascii_char_weight=options.ascii_char_weight,
    )
    zh_units = _count_text_units(
        block.get("zh_text", ""),
        ascii_char_weight=options.ascii_char_weight,
    )
    primary_units = max(ja_units, zh_units)
    secondary_units = min(ja_units, zh_units)
    total_units = primary_units + secondary_units * options.bilingual_secondary_weight

    if total_units <= 0:
        return options.min_duration

    reading_duration = options.reading_base + total_units / options.reading_cps
    reading_duration = max(options.min_duration, reading_duration)
    if options.max_duration > 0:
        reading_duration = min(reading_duration, options.max_duration)
    return reading_duration


def _subtitle_gap_s(options: SubtitleOptions) -> float:
    return max(float(options.frame_gap_s), 0.0)


def _frames_to_seconds(frames: float, options: SubtitleOptions) -> float:
    return max(0.0, float(frames)) * options.frame_duration_s


def _resolve_subtitle_window(
    blocks: list[dict],
    idx: int,
    *,
    options: SubtitleOptions | None = None,
) -> tuple[float, float]:
    options = _coerce_options(options)
    gap_s = _subtitle_gap_s(options)
    block = blocks[idx - 1]
    start = float(block["start"])
    raw_end = max(start, float(block["end"]))
    if options.timeline_mode in {"alignment", "aligned", "raw"}:
        end = raw_end
        if idx < len(blocks):
            next_start = max(start + 0.05, float(blocks[idx]["start"]))
            end = min(end, max(start + 0.05, next_start - gap_s))
        if options.max_duration > 0:
            end = min(end, start + options.max_duration)
        end = max(end, start + options.min_duration)
        if idx < len(blocks):
            next_start = float(blocks[idx]["start"])
            end = min(end, max(start + 0.05, next_start - gap_s))
        if end <= start:
            end = start + 0.05
        return start, end

    target_duration = _estimate_reading_duration(block, options=options)
    trim_cap_duration = max(
        target_duration * options.duration_ratio_cap,
        target_duration + options.duration_grace,
        options.min_duration,
    )
    if options.max_duration > 0:
        trim_cap_duration = min(trim_cap_duration, options.max_duration)

    end = raw_end

    next_limit = None
    if idx < len(blocks):
        next_limit = max(start + 0.05, float(blocks[idx]["start"]) - gap_s)
        end = min(end, next_limit)

    if options.max_duration > 0:
        end = min(end, start + options.max_duration)

    end = min(end, start + trim_cap_duration)

    min_end = start + target_duration
    if next_limit is not None:
        min_end = min(min_end, next_limit)

    end = max(end, min_end)

    if next_limit is not None:
        end = min(end, next_limit)
    if end <= start:
        end = start + 0.05

    return start, end


def _wrap_subtitle_line(text: str, max_chars: int = 25) -> str:
    normalized = str(text or "")
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized

    lines: list[str] = []
    remaining = normalized
    while len(remaining) > max_chars:
        split_at = -1
        for index in range(min(max_chars, len(remaining) - 1), 0, -1):
            if remaining[index] in _WRAP_PUNCTUATION:
                split_at = index + 1
                break
        if split_at <= 0:
            prefix = remaining[: min(max_chars, len(remaining))]
            matches = list(re.finditer(r"(?<=[ぁ-ゟ])(?=[一-鿿])", prefix))
            if matches:
                split_at = matches[-1].start()
        if split_at <= 0:
            split_at = max_chars

        lines.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        lines.append(remaining)
    return "\n".join(line for line in lines if line)


def _wrap_subtitle_text(
    text: str,
    *,
    options: SubtitleOptions | None = None,
) -> str:
    options = _coerce_options(options)
    lines = str(text or "").replace("\\n", "\n").split("\n")
    return "\n".join(
        wrapped
        for line in lines
        for wrapped in [_wrap_subtitle_line(line.strip(), options.line_max_chars)]
        if wrapped.strip()
    )


def _word_text(word: dict) -> str:
    return str(word.get("word") or word.get("text") or "")


def _timed_words(block: dict) -> list[dict]:
    words: list[dict] = []
    for item in block.get("words") or []:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            continue
        word = dict(item)
        word["start"] = start
        word["end"] = end
        words.append(word)
    return sorted(words, key=lambda word: (word["start"], word["end"]))


def _subtitle_block_window(block: dict, words: list[dict]) -> tuple[float, float]:
    fallback_start = float(words[0]["start"]) if words else 0.0
    try:
        start = float(block.get("start", fallback_start))
    except (TypeError, ValueError):
        start = fallback_start

    fallback_end = float(words[-1]["end"]) if words else start
    try:
        end = float(block.get("end", fallback_end))
    except (TypeError, ValueError):
        end = fallback_end

    return start, max(start, end)


def _copy_sorted_blocks(blocks: list[dict]) -> list[dict]:
    sortable: list[tuple[float, float, int, dict]] = []
    for index, block in enumerate(blocks):
        copied = dict(block)
        start, end = _subtitle_block_window(copied, _timed_words(copied))
        copied["start"] = start
        copied["end"] = end
        sortable.append((start, end, index, copied))
    sortable.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in sortable]


def _block_text_units(block: dict) -> float:
    return _count_text_units(
        str(block.get("ja_text") or block.get("text") or "")
        + str(block.get("zh_text") or block.get("zh") or ""),
    )


def _text_overlap_len(left: str, right: str, *, min_overlap: int = 2) -> int:
    left_compact = _COMPACT_SPACE_RE.sub("", left or "")
    right_compact = _COMPACT_SPACE_RE.sub("", right or "")
    max_overlap = min(len(left_compact), len(right_compact))
    for size in range(max_overlap, min_overlap - 1, -1):
        if left_compact[-size:] == right_compact[:size]:
            return size
    return 0


def _merge_text_with_overlap(left: str, right: str, *, separator: str) -> str:
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    return separator.join((left, right))


def _looks_like_short_overlapping_tail(
    current_ja: str,
    next_ja: str,
    *,
    gap: float,
    next_duration: float,
    options: SubtitleOptions,
) -> bool:
    current_compact = _COMPACT_SPACE_RE.sub("", current_ja or "").strip()
    next_compact = _COMPACT_SPACE_RE.sub("", next_ja or "").strip()
    if not current_compact or not next_compact:
        return False
    if current_compact.endswith(tuple(_SENTENCE_END_PUNCTUATION)):
        return False
    if gap > _frames_to_seconds(_OVERLAPPING_TAIL_MAX_GAP_FRAMES, options):
        return False
    if next_duration > _frames_to_seconds(_OVERLAPPING_TAIL_MAX_DURATION_FRAMES, options):
        return False
    return _text_overlap_len(current_compact, next_compact) > 0


def _can_merge_overlapping_blocks(
    left: dict,
    right: dict,
    *,
    options: SubtitleOptions,
) -> bool:
    start = min(float(left["start"]), float(right["start"]))
    end = max(float(left["end"]), float(right["end"]))
    max_duration = options.max_duration if options.max_duration > 0 else end - start
    max_chars = options.line_max_chars if options.line_max_chars > 0 else 25
    return (
        end - start <= max_duration
        and _block_text_units(left) + _block_text_units(right) <= max_chars * 2.4
    )


def _merge_overlapping_blocks(left: dict, right: dict) -> dict:
    merged = dict(left)
    merged["start"] = min(float(left["start"]), float(right["start"]))
    merged["end"] = max(float(left["end"]), float(right["end"]))
    for key in ("ja_text", "zh_text", "text", "zh"):
        parts = [
            str(block.get(key) or "").strip()
            for block in (left, right)
            if str(block.get(key) or "").strip()
        ]
        if not parts:
            continue
        separator = "，" if key in {"zh_text", "zh"} else " "
        merged[key] = separator.join(parts)
    merged["words"] = sorted(
        list(left.get("words") or []) + list(right.get("words") or []),
        key=lambda word: (
            float(word.get("start", 0.0)) if isinstance(word, dict) else 0.0,
            float(word.get("end", 0.0)) if isinstance(word, dict) else 0.0,
        ),
    )
    source_ids = []
    for block in (left, right):
        for source_id in block.get("source_segment_ids") or []:
            if source_id not in source_ids:
                source_ids.append(source_id)
    if source_ids:
        merged["source_segment_ids"] = source_ids
    return merged


def _can_dense_merge_blocks(
    left: dict,
    right: dict,
    *,
    options: SubtitleOptions,
) -> bool:
    if not options.dense_cue_merge_enabled:
        return False

    left_start = float(left.get("start", 0.0))
    left_end = max(left_start, float(left.get("end", left_start)))
    right_start = float(right.get("start", left_end))
    right_end = max(right_start, float(right.get("end", right_start)))
    left_duration = left_end - left_start
    right_duration = right_end - right_start
    combined_duration = right_end - left_start
    gap = right_start - left_end

    if gap < -_subtitle_gap_s(options):
        return False
    if gap > _frames_to_seconds(options.dense_cue_merge_max_gap_frames, options):
        return False
    max_single = _frames_to_seconds(options.dense_cue_merge_max_single_frames, options)
    if max(left_duration, right_duration) > max_single:
        return False
    if combined_duration > _frames_to_seconds(
        options.dense_cue_merge_max_combined_frames,
        options,
    ):
        return False

    combined_units = _block_text_units(left) + _block_text_units(right)
    if combined_units > options.dense_cue_merge_max_text_units:
        return False

    left_ja = str(left.get("ja_text") or left.get("text") or "").strip()
    left_zh = str(left.get("zh_text") or left.get("zh") or "").strip()
    if left_ja.endswith(tuple(_SENTENCE_END_PUNCTUATION)):
        return False
    if left_zh.endswith(tuple(_ZH_SOFT_SPLIT_PUNCTUATION)):
        return False

    return True


def _merge_dense_short_cues(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    if not options.merge_adjacent or not options.dense_cue_merge_enabled or len(blocks) < 2:
        return list(blocks)

    merged: list[dict] = []
    index = 0
    while index < len(blocks):
        current = dict(blocks[index])
        while index + 1 < len(blocks):
            nxt = blocks[index + 1]
            if not _can_dense_merge_blocks(current, nxt, options=options):
                break
            current = _merge_overlapping_blocks(current, nxt)
            current["dense_cue_merge_count"] = (
                int(current.get("dense_cue_merge_count") or 0) + 1
            )
            index += 1
        merged.append(current)
        index += 1
    return merged


def _normalize_subtitle_timeline(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    normalized = _copy_sorted_blocks(blocks)
    if len(normalized) < 2:
        return normalized

    gap_s = _subtitle_gap_s(options)
    min_abs_s = min(0.20, max(0.05, options.frame_min_duration_s))
    index = 0
    while index + 1 < len(normalized):
        current = normalized[index]
        nxt = normalized[index + 1]
        current_start = float(current["start"])
        current_end = max(current_start, float(current["end"]))
        next_start = float(nxt["start"])
        next_end = max(next_start, float(nxt["end"]))
        current["end"] = current_end
        nxt["end"] = next_end

        if current_end + gap_s <= next_start:
            index += 1
            continue

        limit_end = max(current_start, next_start - gap_s)
        if limit_end - current_start >= min_abs_s:
            current["end"] = limit_end
            index += 1
            continue

        if _can_merge_overlapping_blocks(current, nxt, options=options):
            normalized[index] = _merge_overlapping_blocks(current, nxt)
            del normalized[index + 1]
            if index:
                index -= 1
            continue

        if limit_end > current_start:
            current["end"] = max(current_start + 0.001, limit_end)
        else:
            current["end"] = current_start + 0.05
            shifted_next_start = current["end"] + gap_s
            nxt["start"] = max(next_start, shifted_next_start)
            nxt["end"] = max(next_end, float(nxt["start"]) + 0.05)
        index += 1

    return normalized


def _polish_subtitle_timeline(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    polished = _copy_sorted_blocks(blocks)
    if not options.timing_polish_enabled or not polished:
        return polished

    gap_s = _subtitle_gap_s(options)
    short_gap_s = max(gap_s, float(options.short_gap_collapse_s))
    linger_s = max(0.0, float(options.linger_s))

    for index, block in enumerate(polished):
        start = float(block["start"])
        end = max(start + 0.05, float(block["end"]))
        if options.max_duration > 0:
            end = min(end, start + options.max_duration)

        if index + 1 < len(polished):
            next_start = float(polished[index + 1]["start"])
            current_gap = max(0.0, next_start - end)
            max_end = max(start + 0.05, next_start - gap_s)
            if current_gap < short_gap_s:
                target_end = max_end
            else:
                preserve_pause_end = max(start + 0.05, next_start - short_gap_s)
                target_end = min(end + linger_s, preserve_pause_end)
            end = min(max(end, target_end), max_end)
        elif linger_s > 0:
            end += linger_s

        if options.max_duration > 0:
            end = min(end, start + options.max_duration)
        block["end"] = max(start + 0.05, end)

    return polished


def _prepare_subtitle_blocks(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
    merge_adjacent: bool,
) -> list[dict]:
    options = _coerce_options(options)
    prepared = _copy_sorted_blocks(blocks)
    if merge_adjacent:
        prepared = _merge_adjacent_short_blocks(prepared, options=options)
        prepared = _copy_sorted_blocks(prepared)
    prepared = _soft_split_subtitle_blocks(prepared, options=options)
    prepared = _merge_dense_short_cues(prepared, options=options)
    prepared = _normalize_subtitle_timeline(prepared, options=options)
    for idx in range(1, len(prepared) + 1):
        start, end = _resolve_subtitle_window(prepared, idx, options=options)
        prepared[idx - 1]["start"] = start
        prepared[idx - 1]["end"] = end
    prepared = _polish_subtitle_timeline(prepared, options=options)
    # Reading-duration expansion can push a cue back into the next cue window.
    # Keep this final pass as the hard no-overlap, frame-gap guard for the cue plan.
    prepared = _normalize_subtitle_timeline(prepared, options=options)
    if merge_adjacent:
        # Timing polish intentionally collapses very short display gaps to the
        # frame gap. A final bounded merge pass prevents those polished micro
        # cues from surviving only because the first merge pass ran on raw
        # alignment gaps.
        prepared = _merge_adjacent_short_blocks(prepared, options=options)
        prepared = _normalize_subtitle_timeline(prepared, options=options)
    return prepared


def prepare_srt_blocks(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
    mode: Literal["srt", "bilingual"] = "srt",
) -> list[dict]:
    """Return the stable cue plan to translate and write as SRT."""
    options = _coerce_options(options)
    del mode
    return _prepare_subtitle_blocks(
        blocks,
        options=options,
        merge_adjacent=options.merge_adjacent,
    )


def _split_candidate_timing(
    words: list[dict],
    split_index: int,
    block_start: float,
    block_end: float,
    *,
    options: SubtitleOptions | None = None,
) -> tuple[float, float, float, float] | None:
    options = _coerce_options(options)
    if split_index <= 0 or split_index >= len(words):
        return None

    left_end = max(block_start, float(words[split_index - 1]["end"]))
    right_start = max(left_end, float(words[split_index]["start"]))
    if left_end <= block_start or right_start >= block_end:
        return None

    left_duration = left_end - block_start
    right_duration = block_end - right_start
    if left_duration < options.min_duration or right_duration < options.min_duration:
        return None

    return left_end, right_start, left_duration, right_duration


def _best_split_candidate(
    candidates: list[tuple[int, int | None]],
    words: list[dict],
    block_start: float,
    block_end: float,
    *,
    options: SubtitleOptions | None = None,
) -> tuple[int, int | None] | None:
    options = _coerce_options(options)
    if not candidates:
        return None

    duration = block_end - block_start
    target_offset = min(max(options.min_duration, duration / 2), options.soft_max)
    target_time = block_start + target_offset
    scored: list[tuple[float, int, int | None]] = []
    seen: set[tuple[int, int | None]] = set()

    for split_index, zh_split_at in candidates:
        key = (split_index, zh_split_at)
        if key in seen:
            continue
        seen.add(key)
        timing = _split_candidate_timing(
            words,
            split_index,
            block_start,
            block_end,
            options=options,
        )
        if timing is None:
            continue
        left_end, _right_start, left_duration, right_duration = timing
        score = abs(left_end - target_time)
        if options.max_duration > 0 and left_duration > options.max_duration:
            score += 1000 + left_duration
        if options.max_duration > 0 and right_duration > options.max_duration:
            score += 50 + right_duration
        scored.append((score, split_index, zh_split_at))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0])
    _score, split_index, zh_split_at = scored[0]
    return split_index, zh_split_at


def _zh_sentence_split_candidates(block: dict, words: list[dict]) -> list[tuple[int, int]]:
    zh_text = str(block.get("zh_text") or block.get("zh") or "").strip()
    if len(zh_text) < 2:
        return []

    candidates: list[tuple[int, int]] = []
    for index, char in enumerate(zh_text[:-1]):
        if char not in _ZH_SOFT_SPLIT_PUNCTUATION:
            continue
        split_at = index + 1
        if not zh_text[:split_at].strip() or not zh_text[split_at:].strip():
            continue
        ratio = split_at / max(1, len(zh_text))
        split_index = int(ratio * len(words) + 0.5)
        split_index = max(1, min(len(words) - 1, split_index))
        candidates.append((split_index, split_at))
    return candidates


def _ja_particle_split_candidates(words: list[dict]) -> list[tuple[int, None]]:
    candidates: list[tuple[int, None]] = []
    for split_index in range(1, len(words)):
        token = _word_text(words[split_index - 1]).strip(_JA_BOUNDARY_STRIP_CHARS)
        if token.endswith(_JA_PARTICLE_SPLIT_SUFFIXES):
            candidates.append((split_index, None))
    return candidates


def _hard_word_split_candidate(
    words: list[dict],
    block_start: float,
    block_end: float,
    *,
    threshold: float | None = None,
    options: SubtitleOptions | None = None,
) -> tuple[int, None] | None:
    options = _coerce_options(options)
    effective = options.max_duration if threshold is None else threshold
    if effective <= 0 or block_end - block_start <= effective:
        return None

    target_time = block_start + min(effective, (block_end - block_start) / 2)
    scored: list[tuple[float, int]] = []
    for split_index in range(1, len(words)):
        timing = _split_candidate_timing(
            words,
            split_index,
            block_start,
            block_end,
            options=options,
        )
        if timing is None:
            continue
        left_end, _right_start, left_duration, right_duration = timing
        score = abs(left_end - target_time)
        if left_duration > effective:
            score += 1000 + left_duration
        if right_duration < options.min_duration:
            score += 500 + right_duration
        scored.append((score, split_index))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0])
    return scored[0][1], None


def _split_text_by_ratio(text: str, ratio: float) -> tuple[str, str]:
    normalized = str(text or "").strip()
    if len(normalized) < 2:
        return normalized, ""

    split_at = int(len(normalized) * ratio + 0.5)
    split_at = max(1, min(len(normalized) - 1, split_at))
    preferred_chars = "，、；;：:,.!?！？ "
    candidates = [
        index + 1
        for index, char in enumerate(normalized[:-1])
        if char in preferred_chars
        and normalized[: index + 1].strip()
        and normalized[index + 1 :].strip()
    ]
    if candidates:
        split_at = min(candidates, key=lambda index: abs(index - split_at))

    return normalized[:split_at].strip(), normalized[split_at:].strip()


def _split_block_at_word_boundary(
    block: dict,
    words: list[dict],
    split_index: int,
    zh_split_at: int | None,
    *,
    options: SubtitleOptions | None = None,
) -> tuple[dict, dict] | None:
    block_start, block_end = _subtitle_block_window(block, words)
    timing = _split_candidate_timing(
        words,
        split_index,
        block_start,
        block_end,
        options=options,
    )
    if timing is None:
        return None

    left_end, right_start, _left_duration, _right_duration = timing
    left_words = words[:split_index]
    right_words = words[split_index:]
    left_ja_from_words = "".join(_word_text(word) for word in left_words).strip()
    right_ja_from_words = "".join(_word_text(word) for word in right_words).strip()
    if not left_ja_from_words or not right_ja_from_words:
        return None

    ja_text = str(block.get("ja_text") or block.get("text") or "").strip()
    zh_text = str(block.get("zh_text") or block.get("zh") or "").strip()
    if zh_text and ja_text and zh_text == ja_text:
        left_zh, right_zh = left_ja_from_words, right_ja_from_words
    elif zh_split_at is not None:
        left_zh = zh_text[:zh_split_at].strip()
        right_zh = zh_text[zh_split_at:].strip()
    else:
        boundary_ratio = (left_end - block_start) / max(0.001, block_end - block_start)
        left_zh, right_zh = _split_text_by_ratio(zh_text, boundary_ratio)

    if zh_text and (not left_zh or not right_zh):
        return None

    left = dict(block)
    right = dict(block)
    left.update(
        {
            "start": block_start,
            "end": left_end,
            "ja_text": left_ja_from_words,
            "zh_text": left_zh,
            "words": left_words,
        }
    )
    right.update(
        {
            "start": right_start,
            "end": block_end,
            "ja_text": right_ja_from_words,
            "zh_text": right_zh,
            "words": right_words,
        }
    )
    if "text" in block:
        left["text"] = left_ja_from_words
        right["text"] = right_ja_from_words
    return left, right


def _soft_split_subtitle_block(
    block: dict,
    *,
    depth: int = 0,
    options: SubtitleOptions | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    if not options.soft_split_enabled:
        return [block]

    words = _timed_words(block)
    if len(words) < 2 or depth > len(words):
        return [block]

    block_start, block_end = _subtitle_block_window(block, words)
    duration = block_end - block_start
    exceeds_soft_limit = options.soft_max > 0 and duration > options.soft_max
    exceeds_hard_limit = options.max_duration > 0 and duration > options.max_duration
    if not exceeds_soft_limit and not exceeds_hard_limit:
        return [block]

    candidate = _best_split_candidate(
        _zh_sentence_split_candidates(block, words),
        words,
        block_start,
        block_end,
        options=options,
    )
    if candidate is None:
        candidate = _best_split_candidate(
            _ja_particle_split_candidates(words),
            words,
            block_start,
            block_end,
            options=options,
        )
    if candidate is None and exceeds_hard_limit:
        candidate = _hard_word_split_candidate(
            words,
            block_start,
            block_end,
            options=options,
        )
    if candidate is None and exceeds_soft_limit:
        candidate = _hard_word_split_candidate(
            words,
            block_start,
            block_end,
            threshold=options.soft_max,
            options=options,
        )
    if candidate is None:
        return [block]

    split_result = _split_block_at_word_boundary(
        block,
        words,
        candidate[0],
        candidate[1],
        options=options,
    )
    if split_result is None:
        return [block]

    left, right = split_result
    return (
        _soft_split_subtitle_block(left, depth=depth + 1, options=options)
        + _soft_split_subtitle_block(right, depth=depth + 1, options=options)
    )


def _soft_split_subtitle_blocks(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    split_blocks: list[dict] = []
    for block in blocks:
        split_blocks.extend(_soft_split_subtitle_block(dict(block), options=options))
    return split_blocks


def write_srt(
    blocks: list[dict],
    path: str,
    *,
    options: SubtitleOptions | None = None,
):
    """
    blocks: [{start, end, zh_text}]
    zh_text may contain \\n to preserve manual line breaks within one subtitle block.
    """
    options = _coerce_options(options)
    blocks = [dict(block) for block in blocks]
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        for idx, block in enumerate(blocks, 1):
            start = float(block.get("start", 0.0))
            end = max(start + 0.05, float(block.get("end", start)))

            start_str = format_timestamp(start)
            end_str   = format_timestamp(end)
            wrapped = _wrap_subtitle_text(block.get("zh_text", ""), options=options)
            f.write(f"{idx}\n{start_str} --> {end_str}\n{wrapped}\n\n")
    return blocks


def _merge_adjacent_short_blocks(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    if not options.merge_adjacent or len(blocks) < 2:
        return list(blocks)

    merged: list[dict] = []
    index = 0
    max_chars = options.line_max_chars if options.line_max_chars > 0 else 25
    while index < len(blocks):
        current = dict(blocks[index])
        while index + 1 < len(blocks):
            nxt = blocks[index + 1]
            current_end = float(current.get("end", current.get("start", 0.0)))
            next_start = float(nxt.get("start", 0.0))
            gap = next_start - current_end
            ja_text = str(current.get("ja_text", "")).strip()
            next_ja_text = str(nxt.get("ja_text", "")).strip()
            current_zh_text = str(current.get("zh_text", "")).strip()
            next_zh_text = str(nxt.get("zh_text", "")).strip()
            combined_ja = _merge_text_with_overlap(
                ja_text,
                next_ja_text,
                separator=" ",
            )
            combined_zh = _merge_text_with_overlap(
                current_zh_text,
                next_zh_text,
                separator="，",
            )
            combined_chars = len(_COMPACT_SPACE_RE.sub("", combined_ja + combined_zh))
            combined_duration = float(nxt.get("end", next_start)) - float(
                current.get("start", 0.0)
            )
            next_duration = float(nxt.get("end", next_start)) - next_start
            continuation_tail = _looks_like_short_overlapping_tail(
                ja_text,
                next_ja_text,
                gap=gap,
                next_duration=next_duration,
                options=options,
            )

            if (
                gap <= _frames_to_seconds(_ADJACENT_SHORT_MERGE_MAX_GAP_FRAMES, options)
                and combined_chars <= max_chars * 2
                and combined_duration
                <= _frames_to_seconds(
                    _ADJACENT_SHORT_MERGE_MAX_DURATION_FRAMES,
                    options,
                )
                and not ja_text.endswith(tuple(_SENTENCE_END_PUNCTUATION))
            ):
                current["end"] = nxt.get("end", current.get("end"))
                current["ja_text"] = combined_ja
                current["zh_text"] = combined_zh
                if "text" in current or "text" in nxt:
                    current["text"] = current["ja_text"]
                current["words"] = list(current.get("words") or []) + list(
                    nxt.get("words") or []
                )
                source_ids = []
                for block in (current, nxt):
                    for source_id in block.get("source_segment_ids") or []:
                        if source_id not in source_ids:
                            source_ids.append(source_id)
                if source_ids:
                    current["source_segment_ids"] = source_ids
                index += 1
                continue
            break
        merged.append(current)
        index += 1
    return merged


def write_bilingual_srt(
    blocks: list[dict],
    path: str,
    *,
    options: SubtitleOptions | None = None,
):
    """blocks: [{start, end, ja_text, zh_text}] — Japanese line above Chinese."""
    options = _coerce_options(options)
    blocks = [dict(block) for block in blocks]
    with open(path, "w", encoding="utf-8") as f:
        for idx, block in enumerate(blocks, 1):
            start = float(block.get("start", 0.0))
            end = max(start + 0.05, float(block.get("end", start)))

            start_str = format_timestamp(start)
            end_str   = format_timestamp(end)
            ja_line = _wrap_subtitle_text(block.get("ja_text", ""), options=options)
            zh_text = str(block.get("zh_text", "")).strip()
            if not zh_text:
                logger.warning("Empty translated subtitle at index %s; using placeholder", idx)
                zh_text = "「未翻译」"
            zh_line = _wrap_subtitle_text(zh_text, options=options)
            content = "\n".join(
                line for line in (ja_line + "\n" + zh_line).split("\n") if line.strip()
            )
            f.write(f"{idx}\n{start_str} --> {end_str}\n{content}\n\n")
    return blocks
