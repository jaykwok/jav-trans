import os
import re
import logging
import math
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
_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？!?…；;，、,]\s*")

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
    return reading_duration


def _subtitle_gap_s(options: SubtitleOptions) -> float:
    return max(float(options.frame_gap_s), 0.0)


def _subtitle_min_duration_s(options: SubtitleOptions) -> float:
    return max(float(options.min_duration), float(options.frame_min_duration_s))


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
    min_duration_s = _subtitle_min_duration_s(options)
    block = blocks[idx - 1]
    start = float(block["start"])
    raw_end = max(start, float(block["end"]))
    if options.timeline_mode in {"alignment", "aligned", "raw"}:
        end = raw_end
        if idx < len(blocks):
            next_start = max(start + 0.05, float(blocks[idx]["start"]))
            end = min(end, max(start + 0.05, next_start - gap_s))
        end = max(end, start + min_duration_s)
        if idx < len(blocks):
            next_start = float(blocks[idx]["start"])
            end = min(end, max(start + 0.05, next_start - gap_s))
        if end <= start:
            end = start + 0.05
        return start, end

    target_duration = max(_estimate_reading_duration(block, options=options), min_duration_s)
    trim_cap_duration = max(
        target_duration * options.duration_ratio_cap,
        target_duration + options.duration_grace,
        min_duration_s,
    )

    end = raw_end

    next_limit = None
    if idx < len(blocks):
        next_limit = max(start + 0.05, float(blocks[idx]["start"]) - gap_s)
        end = min(end, next_limit)

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


def _subtitle_max_display_duration_s(options: SubtitleOptions) -> float:
    return max(0.0, float(options.max_display_duration_s))


def _text_for_timing(block: dict) -> str:
    return str(
        block.get("ja_text")
        or block.get("text")
        or block.get("zh_text")
        or ""
    )


def _candidate_text_boundaries(text: str) -> list[int]:
    stripped = str(text or "")
    if len(stripped) <= 1:
        return []
    positions = {
        match.end()
        for match in _SENTENCE_BOUNDARY_RE.finditer(stripped)
        if 0 < match.end() < len(stripped)
    }
    positions.update(
        index
        for index, char in enumerate(stripped)
        if 0 < index < len(stripped) and char.isspace()
    )
    return sorted(positions)


def _text_unit_prefix_ratios(text: str, positions: list[int]) -> dict[int, float]:
    total = _count_text_units(text)
    if total <= 0.0:
        return {position: 0.0 for position in positions}
    return {
        position: max(0.0, min(1.0, _count_text_units(text[:position]) / total))
        for position in positions
    }


def _fallback_text_position(text: str, ratio: float) -> int:
    stripped = str(text or "")
    if len(stripped) <= 1:
        return len(stripped)
    return max(1, min(len(stripped) - 1, int(round(len(stripped) * ratio))))


def _choose_text_split_positions(text: str, split_count: int) -> list[int]:
    if split_count <= 0:
        return []
    stripped = str(text or "")
    if len(stripped) <= 1:
        return []
    candidate_positions = _candidate_text_boundaries(stripped)
    ratios_by_position = _text_unit_prefix_ratios(stripped, candidate_positions)
    selected: list[int] = []
    for index in range(1, split_count + 1):
        target_ratio = index / float(split_count + 1)
        available = [position for position in candidate_positions if position not in selected]
        if available:
            position = min(
                available,
                key=lambda item: (
                    abs(ratios_by_position.get(item, 0.0) - target_ratio),
                    item,
                ),
            )
        else:
            position = _fallback_text_position(stripped, target_ratio)
            while position in selected and position + 1 < len(stripped):
                position += 1
            while position in selected and position > 1:
                position -= 1
        if 0 < position < len(stripped):
            selected.append(position)
    return sorted(set(selected))


def _split_text_by_positions(text: str, positions: list[int]) -> list[str]:
    raw = str(text or "")
    if not positions:
        return [raw]
    pieces: list[str] = []
    cursor = 0
    for position in positions:
        pieces.append(raw[cursor:position].strip())
        cursor = position
    pieces.append(raw[cursor:].strip())
    return pieces


def _split_text_by_ratios(text: str, ratios: list[float]) -> list[str]:
    raw = str(text or "")
    if not ratios:
        return [raw]
    positions = []
    for ratio in ratios:
        position = _fallback_text_position(raw, ratio)
        if 0 < position < len(raw):
            positions.append(position)
    return _split_text_by_positions(raw, sorted(set(positions)))


def _weak_cut_times(block: dict, *, start: float, end: float) -> list[float]:
    times: list[float] = []
    for candidate in block.get("weak_cut_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        try:
            time_s = float(candidate["time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < time_s < end:
            times.append(time_s)
    return sorted(set(times))


def _choose_display_split_times(
    block: dict,
    *,
    text_ratios: list[float],
    options: SubtitleOptions,
) -> list[float]:
    start = float(block["start"])
    end = max(start, float(block["end"]))
    if not text_ratios:
        return []
    duration = end - start
    max_display_s = _subtitle_max_display_duration_s(options)
    if duration <= 0.0 or max_display_s <= 0.0:
        return []
    min_duration_s = _subtitle_min_duration_s(options)
    weak_times = _weak_cut_times(block, start=start, end=end)
    split_times: list[float] = []
    previous = start
    for index, ratio in enumerate(text_ratios):
        remaining = len(text_ratios) - index
        target = start + duration * max(0.0, min(1.0, ratio))
        lower = max(previous + min_duration_s, end - max_display_s * (remaining + 1))
        upper = min(end - min_duration_s * remaining, previous + max_display_s)
        if upper <= lower:
            chosen = max(previous + 0.05, min(target, end - 0.05 * remaining))
        else:
            available = [
                time_s
                for time_s in weak_times
                if lower <= time_s <= upper and time_s not in split_times
            ]
            chosen = min(available, key=lambda item: abs(item - target)) if available else min(max(target, lower), upper)
        split_times.append(chosen)
        previous = chosen
    return sorted(time for time in split_times if start < time < end)


def _filter_candidates_for_window(
    candidates: list[dict],
    *,
    start: float,
    end: float,
) -> list[dict]:
    filtered: list[dict] = []
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        try:
            time_s = float(candidate["time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < time_s < end:
            filtered.append(dict(candidate))
    return filtered


def _split_long_display_block(
    block: dict,
    *,
    options: SubtitleOptions,
) -> list[dict]:
    start = float(block.get("start", 0.0))
    end = max(start, float(block.get("end", start)))
    max_display_s = _subtitle_max_display_duration_s(options)
    if max_display_s <= 0.0 or end - start <= max_display_s:
        return [dict(block)]
    timing_text = _text_for_timing(block)
    if not timing_text.strip():
        return [dict(block)]
    split_count = max(1, int(math.ceil((end - start) / max_display_s)) - 1)
    timing_positions = _choose_text_split_positions(timing_text, split_count)
    if len(timing_positions) != split_count:
        return [dict(block)]
    ratios = [
        _count_text_units(timing_text[:position]) / max(_count_text_units(timing_text), 1e-6)
        for position in timing_positions
    ]
    split_times = _choose_display_split_times(block, text_ratios=ratios, options=options)
    if len(split_times) != split_count:
        return [dict(block)]
    weak_times = _weak_cut_times(block, start=start, end=end)
    split_source = (
        "weak_cut_candidates"
        if any(
            any(abs(split_time - weak_time) <= 1e-3 for weak_time in weak_times)
            for split_time in split_times
        )
        else "proportional_text"
    )

    boundaries = [start, *split_times, end]
    text_fields = {}
    for key in ("ja_text", "zh_text", "text"):
        value = block.get(key)
        if value is None:
            continue
        if str(value) == timing_text:
            text_fields[key] = _split_text_by_positions(str(value), timing_positions)
        else:
            text_fields[key] = _split_text_by_ratios(str(value), ratios)

    split_blocks: list[dict] = []
    for index in range(len(boundaries) - 1):
        piece_start = boundaries[index]
        piece_end = boundaries[index + 1]
        item = dict(block)
        item["start"] = piece_start
        item["end"] = max(piece_start + 0.05, piece_end)
        for key, pieces in text_fields.items():
            item[key] = pieces[index] if index < len(pieces) else ""
        item["words"] = [
            word
            for word in _timed_words(block)
            if piece_start <= float(word.get("start", piece_start)) < piece_end
        ]
        item["primary_cut_candidates"] = _filter_candidates_for_window(
            list(block.get("primary_cut_candidates") or []),
            start=piece_start,
            end=piece_end,
        )
        item["weak_cut_candidates"] = _filter_candidates_for_window(
            list(block.get("weak_cut_candidates") or []),
            start=piece_start,
            end=piece_end,
        )
        item["subtitle_layout_split"] = "max_display_duration"
        item["subtitle_layout_split_source"] = split_source
        split_blocks.append(item)
    return split_blocks


def _split_long_display_blocks(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    split: list[dict] = []
    for block in blocks:
        split.extend(_split_long_display_block(block, options=options))
    return split




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


def _word_start_anchor(words: list[dict]) -> float | None:
    if not words:
        return None
    timed = [
        word
        for word in words
        if float(word.get("end", word.get("start", 0.0))) > float(word.get("start", 0.0))
    ]
    if not timed:
        return None
    return min(float(word["start"]) for word in timed)


def _subtitle_block_window(block: dict, words: list[dict]) -> tuple[float, float]:
    fallback_start = float(words[0]["start"]) if words else 0.0
    try:
        start = float(block.get("start", fallback_start))
    except (TypeError, ValueError):
        start = fallback_start
    word_anchor = _word_start_anchor(words)
    if word_anchor is not None:
        start = min(start, word_anchor)

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

        if limit_end > current_start:
            current["end"] = max(current_start + 0.001, limit_end)
        else:
            current["end"] = current_start
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

        block["end"] = max(start + 0.05, end)

    return polished


def _prepare_subtitle_blocks(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    prepared = _copy_sorted_blocks(blocks)
    prepared = _normalize_subtitle_timeline(prepared, options=options)
    for idx in range(1, len(prepared) + 1):
        start, end = _resolve_subtitle_window(prepared, idx, options=options)
        prepared[idx - 1]["start"] = start
        prepared[idx - 1]["end"] = end
    prepared = _copy_sorted_blocks(_split_long_display_blocks(prepared, options=options))
    prepared = _polish_subtitle_timeline(prepared, options=options)
    prepared = _copy_sorted_blocks(_split_long_display_blocks(prepared, options=options))
    # Reading-duration expansion can push a cue back into the next cue window.
    # Keep this final pass as the hard no-overlap, frame-gap guard for the cue plan.
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
    )



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
            block["start"] = start
            block["end"] = end

            start_str = format_timestamp(start)
            end_str   = format_timestamp(end)
            wrapped = _wrap_subtitle_text(block.get("zh_text", ""), options=options)
            f.write(f"{idx}\n{start_str} --> {end_str}\n{wrapped}\n\n")
    return blocks


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
            block["start"] = start
            block["end"] = end

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
