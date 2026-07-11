import os
import re
import logging
import math
from pathlib import Path
from typing import Callable, Literal

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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)

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


def _weak_cut_snap_window_s(duration_s: float, options: SubtitleOptions) -> float:
    duration = max(0.0, float(duration_s))
    if duration <= 3.0:
        return max(0.0, float(options.weak_cut_snap_short_s))
    if duration <= _subtitle_max_display_duration_s(options):
        return max(0.0, float(options.weak_cut_snap_normal_s))
    return max(0.0, float(options.weak_cut_snap_long_s))


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


def _anchor_times(block: dict, *, start: float, end: float) -> list[dict]:
    anchors: list[dict] = []
    for key, anchor_type in (
        ("primary_cut_candidates", "primary_cut"),
        ("weak_cut_candidates", "weak_cut"),
    ):
        for candidate in block.get(key) or []:
            if not isinstance(candidate, dict):
                continue
            try:
                time_s = float(candidate["time_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if not start < time_s < end:
                continue
            anchors.append(
                {
                    "time_s": time_s,
                    "anchor_type": anchor_type,
                    "score": _safe_float(candidate.get("score"), 0.0),
                    "prominence": _safe_float(candidate.get("prominence"), 0.0),
                    "speech_valley": _safe_float(candidate.get("speech_valley"), 0.0),
                    "strength": _safe_float(candidate.get("strength"), 0.0),
                }
            )
    anchors.sort(
        key=lambda item: (
            float(item["time_s"]),
            0 if item["anchor_type"] == "primary_cut" else 1,
            -float(item.get("strength") or 0.0),
        )
    )
    return anchors


def _text_break_score(text: str, position: int) -> float:
    if position <= 0 or position >= len(text):
        return 0.0
    previous = text[position - 1]
    if previous in "。！？!?…":
        return 0.0
    if previous in "；;，、,":
        return 0.15
    if previous.isspace():
        return 0.35
    return 0.75


def _candidate_text_positions_for_dp(text: str, *, split_count: int) -> list[int]:
    raw = str(text or "")
    if len(raw) <= 1:
        return []
    positions = set(_candidate_text_boundaries(raw))
    for index in range(1, max(1, split_count) + 1):
        position = _fallback_text_position(raw, index / float(split_count + 1))
        if 0 < position < len(raw):
            positions.add(position)
    return sorted(position for position in positions if 0 < position < len(raw))


def _choose_anchor_for_target(
    anchors: list[dict],
    *,
    target: float,
    snap_window_s: float,
) -> tuple[float, str, float, float]:
    available = [
        anchor
        for anchor in anchors
        if abs(float(anchor["time_s"]) - target) <= snap_window_s
    ]
    if not available:
        return target, "proportional_text", 0.0, 0.0
    selected = min(
        available,
        key=lambda anchor: (
            abs(float(anchor["time_s"]) - target),
            0 if anchor["anchor_type"] == "primary_cut" else 1,
            -float(anchor.get("strength") or 0.0),
        ),
    )
    distance = abs(float(selected["time_s"]) - target)
    strength = float(selected.get("strength") or selected.get("score") or 0.0)
    return float(selected["time_s"]), str(selected["anchor_type"]), strength, distance


def _long_display_dp_plan(
    block: dict,
    *,
    options: SubtitleOptions,
) -> dict | None:
    start = float(block.get("start", 0.0))
    end = max(start, float(block.get("end", start)))
    duration = end - start
    max_display_s = _subtitle_max_display_duration_s(options)
    if max_display_s <= 0.0 or duration <= max_display_s:
        return None
    text = _text_for_timing(block)
    if not text.strip():
        return None
    min_duration_s = _subtitle_min_duration_s(options)
    split_count = max(1, int(math.ceil(duration / max_display_s)) - 1)
    positions = _candidate_text_positions_for_dp(text, split_count=split_count)
    if not positions:
        return None
    ratios_by_position = _text_unit_prefix_ratios(text, positions)
    anchors = _anchor_times(block, start=start, end=end)
    snap_window_s = _weak_cut_snap_window_s(duration, options)
    nodes: list[dict] = [
        {
            "position": 0,
            "time_s": start,
            "ratio": 0.0,
            "source": "start",
            "anchor_strength": 0.0,
            "snap_distance_s": 0.0,
        }
    ]
    for position in positions:
        ratio = ratios_by_position.get(position, 0.0)
        target = start + duration * ratio
        time_s, source, strength, distance = _choose_anchor_for_target(
            anchors,
            target=target,
            snap_window_s=snap_window_s,
        )
        nodes.append(
            {
                "position": position,
                "time_s": max(start, min(end, time_s)),
                "ratio": ratio,
                "source": source,
                "anchor_strength": strength,
                "snap_distance_s": distance,
            }
        )
    for anchor in anchors:
        anchor_ratio = (float(anchor["time_s"]) - start) / max(duration, 1e-6)
        if positions:
            position = min(
                positions,
                key=lambda item: (
                    abs(ratios_by_position.get(item, 0.0) - anchor_ratio),
                    _text_break_score(text, item),
                    item,
                ),
            )
        else:
            position = _fallback_text_position(text, anchor_ratio)
        if not 0 < position < len(text):
            continue
        target = start + duration * ratios_by_position.get(position, anchor_ratio)
        nodes.append(
            {
                "position": position,
                "time_s": float(anchor["time_s"]),
                "ratio": ratios_by_position.get(position, anchor_ratio),
                "source": str(anchor["anchor_type"]),
                "anchor_strength": float(anchor.get("strength") or anchor.get("score") or 0.0),
                "snap_distance_s": abs(float(anchor["time_s"]) - target),
            }
        )
    nodes.append(
        {
            "position": len(text),
            "time_s": end,
            "ratio": 1.0,
            "source": "end",
            "anchor_strength": 0.0,
            "snap_distance_s": 0.0,
        }
    )
    nodes = sorted(nodes, key=lambda item: (int(item["position"]), float(item["time_s"])))

    best: dict[int, tuple[float, int | None]] = {0: (0.0, None)}
    for j in range(1, len(nodes)):
        best_cost = float("inf")
        best_prev: int | None = None
        for i in range(0, j):
            if i not in best:
                continue
            piece_start = float(nodes[i]["time_s"])
            piece_end = float(nodes[j]["time_s"])
            if piece_end <= piece_start:
                continue
            piece_duration = piece_end - piece_start
            if piece_duration > max_display_s + 1e-6:
                continue
            piece_text = text[int(nodes[i]["position"]) : int(nodes[j]["position"])].strip()
            if not piece_text:
                continue
            duration_penalty = 0.0
            if piece_duration < min_duration_s:
                duration_penalty += (min_duration_s - piece_duration) * 6.0
            text_penalty = 0.0 if j == len(nodes) - 1 else _text_break_score(
                text,
                int(nodes[j]["position"]),
            )
            line_penalty = max(0.0, len(piece_text) - max(1, options.line_max_chars)) / max(
                1.0,
                float(options.line_max_chars),
            )
            anchor_bonus = 0.0
            source = str(nodes[j]["source"])
            if source == "primary_cut":
                anchor_bonus = 1.20 + min(0.30, float(nodes[j]["anchor_strength"]) * 0.05)
            elif source == "weak_cut":
                anchor_bonus = 0.95 + min(0.25, float(nodes[j]["anchor_strength"]) * 0.05)
            snap_penalty = min(1.0, float(nodes[j]["snap_distance_s"]) / max(snap_window_s, 1e-6)) * 0.15
            transition_cost = (
                1.0
                + duration_penalty
                + text_penalty
                + line_penalty
                + snap_penalty
                - anchor_bonus
            )
            cost = best[i][0] + transition_cost
            if cost < best_cost:
                best_cost = cost
                best_prev = i
        if best_prev is not None:
            best[j] = (best_cost, best_prev)
    last = len(nodes) - 1
    if last not in best:
        return None
    path: list[int] = []
    cursor: int | None = last
    while cursor is not None:
        path.append(cursor)
        cursor = best[cursor][1]
    path.reverse()
    if len(path) < 3:
        return None
    return {
        "nodes": [nodes[index] for index in path],
        "score": best[last][0],
    }


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
    plan = _long_display_dp_plan(block, options=options)
    if plan is None:
        return [dict(block)]
    nodes = list(plan["nodes"])
    timing_positions = [int(node["position"]) for node in nodes[1:-1]]
    split_times = [float(node["time_s"]) for node in nodes[1:-1]]
    if not timing_positions or len(timing_positions) != len(split_times):
        return [dict(block)]
    ratios = [
        _count_text_units(timing_text[:position]) / max(_count_text_units(timing_text), 1e-6)
        for position in timing_positions
    ]
    split_sources = [str(node["source"]) for node in nodes[1:-1]]
    split_source = "acoustic_anchor_dp" if any(
        source in {"primary_cut", "weak_cut"} for source in split_sources
    ) else "proportional_text_dp"

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
        item["acoustic_start"] = piece_start
        item["acoustic_end"] = max(piece_start + 0.05, piece_end)
        item["acoustic_duration"] = max(0.0, item["acoustic_end"] - item["acoustic_start"])
        item["display_start"] = item["start"]
        item["display_end"] = item["end"]
        item["display_duration"] = max(0.0, item["display_end"] - item["display_start"])
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
        item["layout_engine"] = options.layout_engine
        item["layout_version"] = "subtitle_layout_v2"
        item["timing_model"] = options.timing_model
        boundary_node = nodes[index + 1] if index + 1 < len(nodes) else None
        boundary_source = str(boundary_node.get("source") or "") if boundary_node else ""
        item["anchor_used"] = boundary_source in {"primary_cut", "weak_cut"}
        item["anchor_type"] = boundary_source if item["anchor_used"] else ""
        item["anchor_score"] = 0.0 if boundary_node is None else float(boundary_node.get("anchor_strength") or 0.0)
        item["snap_distance_s"] = 0.0 if boundary_node is None else float(boundary_node.get("snap_distance_s") or 0.0)
        item["snap_reason"] = "anchor_aware_dp_v2"
        item["layout_score"] = float(plan.get("score") or 0.0)
        item["text_break_type"] = "dp_text_boundary"
        item["proportional_fallback_used"] = not item["anchor_used"]
        split_blocks.append(item)
    return split_blocks


def _split_long_display_blocks(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    split: list[dict] = []
    total = len(blocks)
    interval = max(1, total // 100)
    if progress is not None:
        progress(0, total)
    for index, block in enumerate(blocks, start=1):
        split.extend(_split_long_display_block(block, options=options))
        if progress is not None and (index >= total or index % interval == 0):
            progress(index, total)
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
        if str(word.get("timestamp_kind") or "") != "synthetic_proportional"
        if float(word.get("end", word.get("start", 0.0))) > float(word.get("start", 0.0))
    ]
    if not timed:
        return None
    return min(float(word["start"]) for word in timed)


def _subtitle_block_window(block: dict, words: list[dict]) -> tuple[float, float]:
    fallback_start = float(words[0]["start"]) if words else 0.0
    start = _safe_float(block.get("display_start", block.get("start")), fallback_start)
    word_anchor = _word_start_anchor(words)
    if word_anchor is not None:
        start = min(start, word_anchor)

    fallback_end = float(words[-1]["end"]) if words else start
    end = _safe_float(block.get("display_end", block.get("end")), fallback_end)

    return start, max(start, end)


def _ensure_timeline_fields(block: dict) -> dict:
    start = _safe_float(block.get("start"), 0.0)
    end = max(start, _safe_float(block.get("end"), start))
    acoustic_start = _safe_float(block.get("acoustic_start"), start)
    acoustic_end = max(acoustic_start, _safe_float(block.get("acoustic_end"), end))
    display_start = _safe_float(block.get("display_start"), start)
    display_end = max(display_start, _safe_float(block.get("display_end"), end))
    block["acoustic_start"] = acoustic_start
    block["acoustic_end"] = acoustic_end
    block["acoustic_duration"] = max(0.0, acoustic_end - acoustic_start)
    block["display_start"] = display_start
    block["display_end"] = display_end
    block["display_duration"] = max(0.0, display_end - display_start)
    block["start"] = display_start
    block["end"] = display_end
    return block


def _copy_sorted_blocks(blocks: list[dict]) -> list[dict]:
    sortable: list[tuple[float, float, int, dict]] = []
    for index, block in enumerate(blocks):
        copied = _ensure_timeline_fields(dict(block))
        start, end = _subtitle_block_window(copied, _timed_words(copied))
        copied["start"] = start
        copied["end"] = end
        copied["display_start"] = start
        copied["display_end"] = end
        copied["display_duration"] = max(0.0, end - start)
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
    min_display_s = _subtitle_min_duration_s(options)
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
            current["display_start"] = current_start
            current["display_end"] = current_end
            current["display_duration"] = max(0.0, current_end - current_start)
            index += 1
            continue

        limit_end = max(current_start, next_start - gap_s)
        current["end"] = limit_end
        current["display_start"] = current_start
        current["display_end"] = limit_end
        current["display_duration"] = max(0.0, limit_end - current_start)
        if limit_end - current_start < min_display_s:
            current["duration_violation"] = True
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

        acoustic_end = _safe_float(block.get("acoustic_end"), end)
        max_end_from_acoustic = acoustic_end + max(
            0.0,
            float(options.max_display_shift_from_acoustic_end_s),
        )
        end = min(end, max_end_from_acoustic)
        block["end"] = max(start + 0.05, end)
        block["display_start"] = start
        block["display_end"] = block["end"]
        block["display_duration"] = max(0.0, block["display_end"] - block["display_start"])

    return polished


def _finalize_layout_fields(
    blocks: list[dict],
    *,
    options: SubtitleOptions,
) -> list[dict]:
    finalized: list[dict] = []
    min_duration_s = _subtitle_min_duration_s(options)
    max_display_s = _subtitle_max_display_duration_s(options)
    for block in blocks:
        item = _ensure_timeline_fields(dict(block))
        display_start = _safe_float(item.get("start"), 0.0)
        display_end = max(display_start, _safe_float(item.get("end"), display_start))
        acoustic_start = _safe_float(item.get("acoustic_start"), display_start)
        acoustic_end = max(acoustic_start, _safe_float(item.get("acoustic_end"), display_end))
        item["start"] = display_start
        item["end"] = display_end
        item["display_start"] = display_start
        item["display_end"] = display_end
        item["display_duration"] = max(0.0, display_end - display_start)
        item["acoustic_start"] = acoustic_start
        item["acoustic_end"] = acoustic_end
        item["acoustic_duration"] = max(0.0, acoustic_end - acoustic_start)
        item["display_shift_start_s"] = display_start - acoustic_start
        item["display_shift_end_s"] = display_end - acoustic_end
        item["display_extension_total_s"] = max(0.0, acoustic_start - display_start) + max(
            0.0,
            display_end - acoustic_end,
        )
        item.setdefault("layout_engine", options.layout_engine)
        item.setdefault("layout_version", "subtitle_layout_v2")
        item.setdefault("timing_model", options.timing_model)
        item["duration_violation"] = bool(
            item["display_duration"] < min_duration_s
            or (max_display_s > 0.0 and item["display_duration"] > max_display_s)
        )
        item["gap_violation"] = False
        item["proportional_fallback_used"] = bool(item.get("proportional_fallback_used", False))
        finalized.append(item)
    for current, nxt in zip(finalized, finalized[1:]):
        gap = _safe_float(nxt.get("display_start"), _safe_float(nxt.get("start"))) - _safe_float(
            current.get("display_end"),
            _safe_float(current.get("end")),
        )
        if gap < _subtitle_gap_s(options) - 1e-9:
            current["gap_violation"] = True
            nxt["gap_violation"] = True
    return finalized


def _prepare_subtitle_blocks(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
    on_stage: Callable[[str, int, int], None] | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    def stage(name: str, current: int, total: int) -> None:
        if on_stage is not None:
            on_stage(name, current, total)

    stage("timeline_normalize", 0, 1)
    prepared = _copy_sorted_blocks(blocks)
    prepared = _normalize_subtitle_timeline(prepared, options=options)
    for idx in range(1, len(prepared) + 1):
        start, end = _resolve_subtitle_window(prepared, idx, options=options)
        prepared[idx - 1]["start"] = start
        prepared[idx - 1]["end"] = end
        prepared[idx - 1]["display_start"] = start
        prepared[idx - 1]["display_end"] = end
        prepared[idx - 1]["display_duration"] = max(0.0, end - start)
    stage("timeline_normalize", 1, 1)
    prepared = _copy_sorted_blocks(
        _split_long_display_blocks(
            prepared,
            options=options,
            progress=lambda current, total: stage("layout_dp_pass1", current, total),
        )
    )
    stage("timeline_polish", 0, 1)
    prepared = _polish_subtitle_timeline(prepared, options=options)
    stage("timeline_polish", 1, 1)
    prepared = _copy_sorted_blocks(
        _split_long_display_blocks(
            prepared,
            options=options,
            progress=lambda current, total: stage("layout_dp_pass2", current, total),
        )
    )
    # Reading-duration expansion can push a cue back into the next cue window.
    # Keep this final pass as the hard no-overlap, frame-gap guard for the cue plan.
    prepared = _normalize_subtitle_timeline(prepared, options=options)
    prepared = _finalize_layout_fields(prepared, options=options)
    stage("layout_finalize", 1, 1)
    return prepared


def prepare_srt_blocks(
    blocks: list[dict],
    *,
    options: SubtitleOptions | None = None,
    mode: Literal["srt", "bilingual"] = "srt",
    on_stage: Callable[[str, int, int], None] | None = None,
) -> list[dict]:
    """Return the stable cue plan to translate and write as SRT."""
    options = _coerce_options(options)
    del mode
    return _prepare_subtitle_blocks(
        blocks,
        options=options,
        on_stage=on_stage,
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
    with path_obj.open("w", encoding="utf-8-sig") as f:
        for idx, block in enumerate(blocks, 1):
            start = _safe_float(block.get("display_start", block.get("start")), 0.0)
            end = max(start + 0.05, _safe_float(block.get("display_end", block.get("end")), start))
            block["start"] = start
            block["end"] = end
            block["display_start"] = start
            block["display_end"] = end
            block["display_duration"] = max(0.0, end - start)

            start_str = format_timestamp(start)
            end_str   = format_timestamp(end)
            zh_text = str(block.get("zh_text", "")).strip()
            if not zh_text:
                logger.warning("Empty translated subtitle at index %s; using placeholder", idx)
                zh_text = "「未翻译」"
            wrapped = _wrap_subtitle_text(zh_text, options=options)
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
    with open(path, "w", encoding="utf-8-sig") as f:
        for idx, block in enumerate(blocks, 1):
            start = _safe_float(block.get("display_start", block.get("start")), 0.0)
            end = max(start + 0.05, _safe_float(block.get("display_end", block.get("end")), start))
            block["start"] = start
            block["end"] = end
            block["display_start"] = start
            block["display_end"] = end
            block["display_duration"] = max(0.0, end - start)

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
