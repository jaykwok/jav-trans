import re
import logging
import math
from pathlib import Path
from typing import Callable, Literal

from subtitles.options import SubtitleOptions
from subtitles.vocalisation import drop_vocalisation_runs
from subtitles.zh_style import normalize_zh_subtitle_text, wrap_zh_subtitle_text

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


_COMPACT_SPACE_RE = re.compile(r"\s+")
_WRAP_PUNCTUATION = "，、。！？…"


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


def _render_zh_subtitle_text(text: str, *, options: SubtitleOptions) -> str:
    """Netflix CHS TTSG presentation pass: normalize punctuation, wrap ≤2 lines."""
    normalized = normalize_zh_subtitle_text(text)
    return wrap_zh_subtitle_text(
        normalized,
        line_max_units=float(options.line_max_chars),
        ascii_alnum_weight=options.ascii_char_weight,
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


# A silence between two measured words has to be at least this long before it is
# offered as a cue boundary. Below it the "gap" is the ordinary space between
# syllables of continuous speech, and cutting there splits a word.
WORD_GAP_MIN_S = 0.12

# A gap this long is treated as a full pause and scored like a strong cut. Sized
# to the chunker's own pause floor (ASR_CHUNK_MIN_PAUSE_S default 0.6s) so the
# two readings of silence agree on what counts as a pause.
WORD_GAP_STRONG_S = 0.60


MEASURED_WORD_TIMESTAMP_KINDS = frozenset(
    {
        "ctc_forced_alignment",
        "grok_stt_word",
    }
)


def _has_measured_word_timestamp(word: dict) -> bool:
    return str(word.get("timestamp_kind") or "") in MEASURED_WORD_TIMESTAMP_KINDS


_SENTENCE_END_CHARS = frozenset("。！？!?…")
_CLAUSE_END_CHARS = frozenset("、，,；;")
_EXACT_CLOSING_CHARS = frozenset("。！？!?…、，,；;）」』】〉》〕］｝”’")
_EXACT_OPENING_CHARS = frozenset("（「『【〈《〔［｛“‘")


def _compact_source_length(text: str) -> int:
    return len(_COMPACT_SPACE_RE.sub("", str(text or "")))


def _exact_layout_words(block: dict, text: str) -> tuple[list[dict], str]:
    """Return a complete measured text/time map or a reason it is unusable.

    The layout planner is allowed to leave a cue too long; it is never allowed
    to repair a partial map with character-ratio timing. Requiring the measured
    tokens to reconstruct the source text exactly makes that distinction
    mechanical and auditable.
    """
    words = [
        word
        for word in _timed_words(block)
        if _has_measured_word_timestamp(word) and str(word.get("word") or "")
    ]
    if not words:
        return [], "measured_word_timestamps_unavailable"
    if "".join(str(word["word"]) for word in words) != text:
        return [], "measured_word_text_map_incomplete"
    return words, "complete"


def _exact_boundary_kind(words: list[dict], index: int) -> tuple[str, float]:
    left = words[index - 1]
    right = words[index]
    left_text = str(left.get("word") or "")
    gap = max(0.0, float(right["start"]) - float(left["end"]))
    if left_text[-1] in _SENTENCE_END_CHARS:
        return "sentence_punctuation", gap
    if gap >= WORD_GAP_STRONG_S:
        return "strong_gap", gap
    if left_text[-1] in _CLAUSE_END_CHARS:
        return "clause_punctuation", gap
    if gap >= WORD_GAP_MIN_S:
        return "word_gap", gap
    return "measured_character", gap


def _is_exact_safe_boundary(words: list[dict], index: int) -> bool:
    if not 0 < index < len(words):
        return False
    left_text = str(words[index - 1].get("word") or "")
    right_text = str(words[index].get("word") or "")
    if not left_text or not right_text:
        return False
    if right_text[0] in _EXACT_CLOSING_CHARS or left_text[-1] in _EXACT_OPENING_CHARS:
        return False
    return _exact_boundary_kind(words, index)[0] != "measured_character"


def _exact_lexical_extent(
    words: list[dict],
    start_index: int,
    end_index: int,
) -> tuple[float, float] | None:
    # Acoustic-only vocabularies intentionally return zero-width punctuation.
    # Such tokens remain in the text slice but cannot define a spoken edge.
    lexical = [
        word
        for word in words[start_index:end_index]
        if float(word["end"]) > float(word["start"])
    ]
    if not lexical:
        return None
    return float(lexical[0]["start"]), float(lexical[-1]["end"])


_BOUNDARY_BASE_PENALTY = {
    "sentence_punctuation": 0.0,
    "strong_gap": 0.05,
    "clause_punctuation": 0.10,
    "word_gap": 0.20,
    "end": 0.0,
}


def _boundary_penalty(kind: str, gap_s: float) -> float:
    """What a cut at this boundary costs the DP.

    The kinds are ranked by what justifies the cut: written sentence end, a
    pause long enough to be a pause, written clause end, and last a gap that is
    merely wider than the space between syllables.

    `word_gap` is the only kind whose entire evidence is the silence, and its
    0.12s floor admits boundaries barely distinguishable from continuous speech.
    So inside that kind the measured gap is read as a strength rather than
    discarded: 0.12s costs 0.36, 0.60s costs 0.20, continuously in between.
    Without this the DP scores every word gap alike and the choice falls to the
    length terms, i.e. to whichever gap fills the character cap best.

    Measured on eight films, 7,257 internal cuts (`agents/temp/
    20260812_180000_cut-tracking/evaluate_layout_boundary_policy.py`): cuts
    sitting in a marginal 0.12-0.20s gap fall 432 -> 330 (-23.6%), while cue
    count moves 9,008 -> 9,012, characters p50/p90 stay 16/20 with the same 51
    over the cap, and duration p50 stays 2.615s. The same grading applied to
    every kind was also measured and rejected: it reaches only 353 and pays for
    it by moving 134 cuts off written commas onto acoustic gaps, which trades
    syntax for silence.
    """
    base = _BOUNDARY_BASE_PENALTY[kind]
    if kind != "word_gap":
        return base
    shortfall = max(0.0, WORD_GAP_STRONG_S - float(gap_s)) / WORD_GAP_STRONG_S
    return base + 0.20 * shortfall


def _exact_safe_dp_plan(
    block: dict,
    *,
    options: SubtitleOptions,
) -> dict | None:
    """Jointly optimize source length and lexical duration on exact boundaries.

    Both caps are best-effort. The only hard constraints are a complete measured
    text map and safe candidate boundaries: punctuation, or a measured gap of at
    least 120ms. A direct start-to-end edge always exists, so an unsplittable
    segment is retained intact and its overflow remains visible in diagnostics.
    """
    text = _text_for_timing(block)
    if not text.strip():
        return None
    words, map_status = _exact_layout_words(block, text)
    if not words:
        return {"pieces": [], "reason": map_status, "score": 0.0}

    word_count = len(words)
    candidates = [0, word_count]
    candidates.extend(
        index
        for index in range(1, word_count)
        if _is_exact_safe_boundary(words, index)
    )
    candidates = sorted(set(candidates))

    char_prefix = [0]
    text_prefix = [0]
    for word in words:
        token = str(word["word"])
        char_prefix.append(char_prefix[-1] + _compact_source_length(token))
        text_prefix.append(text_prefix[-1] + len(token))

    char_cap = max(1, int(options.max_source_chars))
    duration_cap_s = _subtitle_max_display_duration_s(options)
    best: dict[int, tuple[float, int | None]] = {0: (0.0, None)}
    for end_index in candidates[1:]:
        best_cost = float("inf")
        best_start: int | None = None
        for start_index in candidates:
            if start_index >= end_index or start_index not in best:
                continue
            extent = _exact_lexical_extent(words, start_index, end_index)
            if extent is None:
                continue
            length = char_prefix[end_index] - char_prefix[start_index]
            duration_s = max(0.0, extent[1] - extent[0])
            char_overflow = max(0, length - char_cap)
            duration_overflow = (
                max(0.0, duration_s - duration_cap_s)
                if duration_cap_s > 0.0
                else 0.0
            )
            underfill = max(
                0.0,
                (char_cap * 0.35 - length) / max(1.0, float(char_cap)),
            )
            if end_index == word_count:
                kind, gap = "end", 0.0
            else:
                kind, gap = _exact_boundary_kind(words, end_index)
            cost = (
                best[start_index][0]
                + 1.0
                + _boundary_penalty(kind, gap)
                + underfill * 0.25
                + float(char_overflow * char_overflow) * 25.0
                + float(duration_overflow * duration_overflow) * 25.0
            )
            if cost < best_cost:
                best_cost = cost
                best_start = start_index
        if best_start is not None:
            best[end_index] = (best_cost, best_start)

    if word_count not in best:
        return {"pieces": [], "reason": "no_complete_safe_path", "score": 0.0}
    path = [word_count]
    cursor = word_count
    while cursor:
        previous = best[cursor][1]
        if previous is None:
            return {"pieces": [], "reason": "broken_safe_path", "score": 0.0}
        path.append(previous)
        cursor = previous
    path.reverse()

    pieces: list[dict] = []
    for start_index, end_index in zip(path, path[1:]):
        extent = _exact_lexical_extent(words, start_index, end_index)
        if extent is None:
            return {"pieces": [], "reason": "punctuation_only_piece", "score": 0.0}
        end_kind, end_gap = (
            ("end", 0.0)
            if end_index == word_count
            else _exact_boundary_kind(words, end_index)
        )
        pieces.append(
            {
                "word_start": start_index,
                "word_end": end_index,
                "text_start": text_prefix[start_index],
                "text_end": text_prefix[end_index],
                "start": extent[0],
                "end": extent[1],
                "end_boundary_kind": end_kind,
                # The measured silence the cut sits in. The DP scores boundaries
                # by kind alone, so this is the graded evidence behind a coarse
                # label: 0.12s and 0.55s are both `word_gap` and are not equally
                # likely to be the end of a thought.
                "end_boundary_gap_s": round(float(end_gap), 3),
                "source_char_count": (
                    char_prefix[end_index] - char_prefix[start_index]
                ),
            }
        )
    if "".join(
        text[int(piece["text_start"]):int(piece["text_end"])] for piece in pieces
    ) != text:
        return {"pieces": [], "reason": "text_not_preserved", "score": 0.0}
    return {
        "pieces": pieces,
        "reason": "split" if len(pieces) > 1 else "kept_as_one",
        "score": best[word_count][0],
        "internal_safe_boundary_count": max(0, len(candidates) - 2),
        "words": words,
        "text": text,
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
    """Split one source segment only at exact, measured, safe boundaries."""
    timing_text = _text_for_timing(block)
    if not timing_text.strip():
        return [dict(block)]
    plan = _exact_safe_dp_plan(block, options=options)
    if plan is None:
        return [dict(block)]
    pieces = list(plan.get("pieces") or [])
    if not pieces:
        item = dict(block)
        item["subtitle_layout_split_skipped"] = str(
            plan.get("reason") or "measured_safe_boundaries_unavailable"
        )
        item["proportional_fallback_used"] = False
        item["exact_measured_timeline"] = False
        return [item]

    words = list(plan["words"])
    timing_total_units = max(_count_text_units(timing_text), 1e-6)
    timing_positions = [int(piece["text_end"]) for piece in pieces[:-1]]
    ratios = [
        _count_text_units(timing_text[:position]) / timing_total_units
        for position in timing_positions
    ]
    text_fields: dict[str, list[str]] = {}
    for key in ("ja_text", "zh_text", "text"):
        value = block.get(key)
        if value is None:
            continue
        if str(value) == timing_text:
            text_fields[key] = _split_text_by_positions(
                str(value),
                timing_positions,
            )
        else:
            # This only distributes an already translated secondary text field.
            # Cue timestamps still come exclusively from measured source words.
            text_fields[key] = _split_text_by_ratios(str(value), ratios)

    split_blocks: list[dict] = []
    for index, piece in enumerate(pieces):
        piece_start = float(piece["start"])
        piece_end = float(piece["end"])
        word_start = int(piece["word_start"])
        word_end = int(piece["word_end"])
        piece_words = [dict(word) for word in words[word_start:word_end]]
        item = dict(block)
        item["start"] = piece_start
        item["end"] = piece_end
        item["acoustic_start"] = piece_start
        item["acoustic_end"] = piece_end
        item["acoustic_duration"] = max(0.0, piece_end - piece_start)
        item["display_start"] = piece_start
        item["display_end"] = piece_end
        item["display_duration"] = max(0.0, piece_end - piece_start)
        for key, values in text_fields.items():
            item[key] = values[index] if index < len(values) else ""
        item["words"] = piece_words
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

        previous_boundary_kind = (
            str(pieces[index - 1]["end_boundary_kind"])
            if index > 0
            else "start"
        )
        end_boundary_kind = str(piece["end_boundary_kind"])
        item["continues_from_previous"] = (
            bool(block.get("continues_from_previous"))
            if index == 0
            else previous_boundary_kind != "sentence_punctuation"
        )
        item["continues_into_next"] = (
            bool(block.get("continues_into_next"))
            if index == len(pieces) - 1
            else end_boundary_kind != "sentence_punctuation"
        )
        if len(pieces) > 1:
            item["subtitle_layout_split"] = "source_char_or_duration_soft_cap"
            item["subtitle_layout_split_source"] = "measured_safe_boundary_dp"
            item.pop("subtitle_layout_split_skipped", None)
        elif (
            int(plan.get("internal_safe_boundary_count") or 0) == 0
            and (
                int(piece["source_char_count"]) > int(options.max_source_chars)
                or (
                    _subtitle_max_display_duration_s(options) > 0.0
                    and piece_end - piece_start
                    > _subtitle_max_display_duration_s(options) + 1e-9
                )
            )
        ):
            item["subtitle_layout_split_skipped"] = (
                "measured_safe_boundaries_unavailable"
            )
        item["layout_engine"] = options.layout_engine
        item["layout_version"] = "subtitle_layout_v3"
        item["timing_model"] = options.timing_model
        item["exact_measured_timeline"] = True
        item["layout_timeline_locked"] = True
        item["source_char_count"] = int(piece["source_char_count"])
        item["source_char_violation"] = (
            int(piece["source_char_count"]) > int(options.max_source_chars)
        )
        item["duration_soft_cap_violation"] = (
            _subtitle_max_display_duration_s(options) > 0.0
            and piece_end - piece_start
            > _subtitle_max_display_duration_s(options) + 1e-9
        )
        item["anchor_used"] = end_boundary_kind != "end"
        item["anchor_type"] = end_boundary_kind if item["anchor_used"] else ""
        item["anchor_score"] = 0.0
        item["snap_distance_s"] = 0.0
        item["snap_reason"] = options.layout_engine
        item["layout_score"] = float(plan.get("score") or 0.0)
        item["text_break_type"] = end_boundary_kind
        item["text_break_gap_s"] = float(piece.get("end_boundary_gap_s") or 0.0)
        item["proportional_fallback_used"] = False
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

        if bool(current.get("layout_timeline_locked")):
            current["display_start"] = current_start
            current["display_end"] = current_end
            current["display_duration"] = max(0.0, current_end - current_start)
            index += 1
            continue

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
        if bool(block.get("layout_timeline_locked")):
            continue
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


def _linger_locked_display_ends(
    blocks: list[dict],
    *,
    options: SubtitleOptions,
    diagnostics: dict | None = None,
) -> list[dict]:
    """Let a locked cue stay on screen through the silence that follows it.

    v3 ends a cue at its last spoken character, and `layout_timeline_locked`
    then exempts it from `_polish_subtitle_timeline`, so the out-point extension
    every subtitle spec asks for stopped running. Measured on eight films that
    costs 487 of 7,016 cues a display shorter than the 20-frame floor and puts
    the source reading rate above 7 chars/s on 40.6% of them.

    The silence is already there - free space after a cue is a median 0.57-2.13s
    per film, and only 10.4% of cues have none - so this buys reading time
    without inventing a single timestamp:

      - starts never move, and neither does any word timing;
      - `acoustic_end` stays exactly where the last character was spoken, so
        `display_shift_end_s` reports precisely what was added, and re-running
        this pass cannot compound (the acoustic cap is absolute, not relative);
      - the extension stops two frames before the next cue appears, and a cue
        whose neighbour is already that close is left alone rather than
        shortened - truncating measured speech is what v3 exists to prevent;
      - nothing is pushed past the display soft cap it was laid out under.

    Runs after the vocalisation filter, so the ceiling is the neighbour that
    actually survives: a dropped moaning run leaves real silence, and holding
    the previous line over 0.5s of it is what a viewer wants.
    """
    linger_s = max(0.0, float(options.linger_s))
    acoustic_cap_s = max(0.0, float(options.max_display_shift_from_acoustic_end_s))
    gap_s = _subtitle_gap_s(options)
    max_display_s = _subtitle_max_display_duration_s(options)
    extended = 0
    added_s = 0.0
    if linger_s <= 0.0 or acoustic_cap_s <= 0.0:
        return blocks

    for index, block in enumerate(blocks):
        if not bool(block.get("layout_timeline_locked")):
            continue
        start = _safe_float(block.get("display_start"), _safe_float(block.get("start")))
        end = max(start, _safe_float(block.get("display_end"), _safe_float(block.get("end"))))
        acoustic_end = max(start, _safe_float(block.get("acoustic_end"), end))
        if index + 1 < len(blocks):
            next_start = _safe_float(
                blocks[index + 1].get("display_start"),
                _safe_float(blocks[index + 1].get("start")),
            )
            ceiling = next_start - gap_s
        else:
            ceiling = end + linger_s
        limits = [end + linger_s, ceiling, acoustic_end + acoustic_cap_s]
        if max_display_s > 0.0:
            limits.append(start + max_display_s)
        new_end = min(limits)
        if new_end <= end + 1e-9:
            continue
        extended += 1
        added_s += new_end - end
        block["end"] = new_end
        block["display_end"] = new_end
        block["display_duration"] = max(0.0, new_end - start)

    if diagnostics is not None:
        diagnostics["display_linger_applied_count"] = extended
        diagnostics["display_linger_total_s"] = round(added_s, 3)
    return blocks


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
        # The source-character and duration targets are soft. If measured safe
        # boundaries cannot satisfy them, retain the exact lexical extent and
        # expose the violation; never clamp the display window to an invented
        # time. `display_clamped_to_max` remains in the schema so old/new runs
        # can be compared mechanically, but v3 always writes false.
        item["display_clamped_to_max"] = False
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
        item.setdefault("layout_version", "subtitle_layout_v3")
        item.setdefault("timing_model", options.timing_model)
        item["duration_soft_cap_violation"] = bool(
            max_display_s > 0.0 and item["display_duration"] > max_display_s
        )
        item["source_char_count"] = int(
            item.get("source_char_count", _compact_source_length(_text_for_timing(item)))
        )
        item["source_char_violation"] = bool(
            item["source_char_count"] > max(1, int(options.max_source_chars))
        )
        item["duration_violation"] = bool(
            item["display_duration"] < min_duration_s
            or item["duration_soft_cap_violation"]
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
    diagnostics: dict | None = None,
) -> list[dict]:
    options = _coerce_options(options)
    def stage(name: str, current: int, total: int) -> None:
        if on_stage is not None:
            on_stage(name, current, total)

    stage("timeline_normalize", 0, 1)
    prepared = _copy_sorted_blocks(blocks)
    # Reading-duration display policy, kept for inputs without measured word
    # timing. Exact pieces created below overwrite these provisional windows
    # with their lexical edges and are then exempt from later polish/normalize.
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
            progress=lambda current, total: stage(
                "layout_measured_safe_dp", current, total
            ),
        )
    )
    stage("timeline_polish", 0, 1)
    prepared = _polish_subtitle_timeline(prepared, options=options)
    stage("timeline_polish", 1, 1)
    # Only now are these the cues a viewer will see. Filtering earlier looks
    # cheaper but measures the wrong thing: before the DP runs, a "block" is a
    # whole ASR segment, and a segment mixing dialogue with moaning is not pure
    # vocalisation - on a real film that placement dropped nothing at all while
    # 398 of the 1983 finished cues qualified. Translation happens after this
    # function returns, so a cue dropped here still never reaches the translator.
    if options.drop_vocalisation_only_cues:
        prepared, vocalisation_diagnostics = drop_vocalisation_runs(
            prepared, min_run=options.vocalisation_min_run
        )
        if diagnostics is not None:
            diagnostics.update(vocalisation_diagnostics)
    if options.timing_polish_enabled:
        # After the filter, not before: the silence a dropped run leaves behind
        # is real silence, and the cue before it should be allowed to use it.
        prepared = _linger_locked_display_ends(
            prepared,
            options=options,
            diagnostics=diagnostics,
        )
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
    diagnostics: dict | None = None,
) -> list[dict]:
    """Return the stable cue plan to translate and write as SRT.

    `diagnostics`, when given, is filled with counts for anything this stage
    removed. Cues dropped here never reach the caller, so without it the only
    evidence would be a smaller list.
    """
    options = _coerce_options(options)
    del mode
    return _prepare_subtitle_blocks(
        blocks,
        options=options,
        on_stage=on_stage,
        diagnostics=diagnostics,
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
    kept: list[dict] = []
    dropped = 0
    with path_obj.open("w", encoding="utf-8-sig") as f:
        for block in blocks:
            start = _safe_float(block.get("display_start", block.get("start")), 0.0)
            measured_exact = bool(block.get("exact_measured_timeline"))
            minimum_end = start if measured_exact else start + 0.05
            end = max(
                minimum_end,
                _safe_float(block.get("display_end", block.get("end")), start),
            )
            block["start"] = start
            block["end"] = end
            block["display_start"] = start
            block["display_end"] = end
            block["display_duration"] = max(0.0, end - start)

            zh_text = str(block.get("zh_text", "")).strip()
            wrapped = _render_zh_subtitle_text(zh_text, options=options)
            if not wrapped:
                # Nothing displayable: either the translation was empty, or it
                # was punctuation only and the CHS style rules (no periods or
                # commas, no trailing 、) legitimately cleared it. A placeholder
                # line here read as a translation failure to the viewer; the cue
                # is dropped instead, and the returned list is what the quality
                # report and the sidecar see, so all three agree.
                dropped += 1
                continue
            kept.append(block)
            start_str = format_timestamp(start)
            end_str   = format_timestamp(end)
            f.write(f"{len(kept)}\n{start_str} --> {end_str}\n{wrapped}\n\n")
    if dropped:
        logger.warning(
            "Dropped %s of %s cues with no displayable text", dropped, len(blocks)
        )
    return kept


def write_bilingual_srt(
    blocks: list[dict],
    path: str,
    *,
    options: SubtitleOptions | None = None,
):
    """blocks: [{start, end, ja_text, zh_text}] — Japanese line above Chinese."""
    options = _coerce_options(options)
    blocks = [dict(block) for block in blocks]
    kept: list[dict] = []
    dropped = 0
    with open(path, "w", encoding="utf-8-sig") as f:
        for block in blocks:
            start = _safe_float(block.get("display_start", block.get("start")), 0.0)
            measured_exact = bool(block.get("exact_measured_timeline"))
            minimum_end = start if measured_exact else start + 0.05
            end = max(
                minimum_end,
                _safe_float(block.get("display_end", block.get("end")), start),
            )
            block["start"] = start
            block["end"] = end
            block["display_start"] = start
            block["display_end"] = end
            block["display_duration"] = max(0.0, end - start)

            ja_line = _wrap_subtitle_text(block.get("ja_text", ""), options=options)
            zh_line = _render_zh_subtitle_text(
                str(block.get("zh_text", "")).strip(),
                options=options,
            )
            # No placeholder line: in this mode the Japanese is still worth
            # showing on its own, and a cue is only dropped when neither side
            # has anything to display.
            content = "\n".join(
                line for line in (ja_line + "\n" + zh_line).split("\n") if line.strip()
            )
            if not content.strip():
                dropped += 1
                continue
            kept.append(block)
            start_str = format_timestamp(start)
            end_str   = format_timestamp(end)
            f.write(f"{len(kept)}\n{start_str} --> {end_str}\n{content}\n\n")
    if dropped:
        logger.warning(
            "Dropped %s of %s cues with no displayable text", dropped, len(blocks)
        )
    return kept
