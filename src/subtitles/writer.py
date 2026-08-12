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
_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？!?…；;，、,]\s*")
_CLOSING_PUNCTUATION = frozenset(
    ".,，、。！？!?…；;:：)]}）］｝」』】〉》〕〗〙〛’”"
)
_OPENING_PUNCTUATION = frozenset("([{（［｛「『【〈《〔〖〘〚‘“")


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
    anchors.extend(_word_gap_anchors(block, start=start, end=end))
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


def _is_valid_subtitle_text_boundary(text: str, position: int) -> bool:
    """Keep closing punctuation with the text on its left.

    CTC gives punctuation a frame just like lexical characters, but punctuation
    is not a spoken onset. A long blank before an ellipsis must therefore not
    make the following cue enter on the first dot; the safe boundary is after
    the complete ellipsis, at the next lexical word.
    """
    if not 0 < position < len(text):
        return False
    return (
        text[position] not in _CLOSING_PUNCTUATION
        and text[position - 1] not in _OPENING_PUNCTUATION
    )


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


def _word_gap_anchors(block: dict, *, start: float, end: float) -> list[dict]:
    """Cut candidates read off measured word timings.

    The acoustic candidate channel (`primary_cut_candidates`/`weak_cut_candidates`)
    was fed by the pre-ASR chain that was retired on 2026-07-31; nothing has
    populated it since, so the "anchor-aware" DP has been running with an empty
    anchor list and splitting long cues purely on character ratio - which lands
    mid-word, because character count is not time.

    With the alignment head on, every character has a measured extent, so the
    silences between words are known directly. Only real gaps become anchors and
    only aligned words are trusted: proportional timings are a restatement of the
    character ratio the DP already has, and offering them as "acoustic" evidence
    would launder a guess into an anchor.
    """
    words = [
        word
        for word in _timed_words(block)
        if _has_measured_word_timestamp(word)
    ]
    if len(words) < 2:
        return []

    anchors: list[dict] = []
    for earlier, later in zip(words, words[1:]):
        gap_start = float(earlier["end"])
        gap_end = float(later["start"])
        gap = gap_end - gap_start
        if gap < WORD_GAP_MIN_S:
            continue
        # Blank is evidence that this text boundary is safe, but a subtitle cut
        # has two different jobs: the previous cue's out-point and the next
        # cue's in-point. The chunker can cut audio at the middle of the blank;
        # a shared subtitle boundary cannot, because that would display the next
        # text for half the silence before its first measured word. Anchor the
        # subtitle boundary at that next word instead. Timeline polish remains
        # free to extend the previous cue into the pause independently.
        time_s = gap_end
        if not start < time_s < end:
            continue
        strong = gap >= WORD_GAP_STRONG_S
        anchors.append(
            {
                "time_s": time_s,
                "anchor_type": "primary_cut" if strong else "weak_cut",
                "score": round(gap, 4),
                "prominence": 0.0,
                "speech_valley": 0.0,
                # Longer silence is a better place to cut; the DP reads this as a
                # tie-break and as a small bonus, both monotone in gap length.
                "strength": round(min(gap / WORD_GAP_STRONG_S, 1.0), 4),
                "anchor_source": "word_gap",
                "gap_start_s": gap_start,
                "gap_midpoint_s": gap_start + gap / 2.0,
                # The word this silence ends at. Lets the DP pair the anchor with
                # the exact character the word starts at instead of guessing the
                # position back out of a ratio.
                "next_word_start_s": gap_end,
            }
        )
    return anchors


def _measured_word_text_map(
    block: dict,
    text: str,
) -> tuple[dict[int, float], dict[int, float], bool, int]:
    """Character offsets in `text` where a measured word starts, by time.

    The DP picks a text position and a time independently and pays a penalty for
    how far apart they are. Feeding it the positions that correspond to real word
    starts is what lets the two agree: without them the only candidate positions
    are punctuation, spaces, and evenly-spaced ratio guesses, and in Japanese —
    no spaces, sparse punctuation in ASR output — that means the ratio guesses.
    """
    words = [
        word
        for word in _timed_words(block)
        if _has_measured_word_timestamp(word)
    ]
    if not words:
        return {}, {}, False, 0

    positions: dict[int, float] = {}
    previous_word_ends: dict[int, float] = {}
    cursor = 0
    for index, word in enumerate(words):
        token = str(word.get("word") or "")
        if not token:
            return {}, {}, False, len(words)
        found = text.find(token, cursor)
        if found < 0 or text[cursor:found].strip():
            # Measured timings may never be silently attached to a different
            # piece of text. The aligned production path is expected to be a
            # complete map; returning it as incomplete keeps the caller from
            # laundering this mismatch through proportional timing.
            return {}, {}, False, len(words)
        cursor = found + len(token)
        if index == 0:
            continue
        if 0 < found < len(text):
            positions[found] = float(word["start"])
            previous_word_ends[found] = float(words[index - 1]["end"])
    if text[cursor:].strip():
        return {}, {}, False, len(words)
    previous_word_ends[len(text)] = float(words[-1]["end"])
    return positions, previous_word_ends, True, len(words)


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
    boundary_penalty = {
        "sentence_punctuation": 0.0,
        "strong_gap": 0.05,
        "clause_punctuation": 0.10,
        "word_gap": 0.20,
        "end": 0.0,
    }
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
            kind = (
                "end"
                if end_index == word_count
                else _exact_boundary_kind(words, end_index)[0]
            )
            cost = (
                best[start_index][0]
                + 1.0
                + boundary_penalty[kind]
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
        end_kind = (
            "end"
            if end_index == word_count
            else _exact_boundary_kind(words, end_index)[0]
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


def _candidate_text_positions_for_dp(
    text: str,
    *,
    split_count: int,
    word_positions: dict[int, float] | None = None,
) -> list[int]:
    raw = str(text or "")
    if len(raw) <= 1:
        return []
    # A measured text/time map is authoritative. Mixing its word positions with
    # punctuation and ratio-only positions lets the DP choose the right piece
    # of text at a time invented from the whole block's character ratio. That
    # is catastrophic when speech density is uneven: a word measured at 20.8s
    # can be rendered from 15.7s even though forced alignment got it right.
    # Fall back to text-only positions only when no measured mapping exists.
    if word_positions:
        return sorted(
            position
            for position in word_positions
            if _is_valid_subtitle_text_boundary(raw, position)
        )
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
    # Character offsets that correspond to a measured word start, so the DP can
    # break where a word begins rather than where the character count says.
    (
        word_positions,
        previous_word_ends,
        measured_map_complete,
        measured_word_count,
    ) = _measured_word_text_map(block, text)
    if measured_word_count and not measured_map_complete:
        # A successful CTC/Grok word-timestamp path must map the complete source
        # text. Do not turn a data-integrity error into plausible-looking
        # proportional subtitles; leave the block intact for the duration guard
        # and expose the problem in tests/logged artifacts instead.
        return None
    if measured_word_count and not word_positions:
        # A complete measured map with no internal word boundary (for example,
        # one provider token covering the whole cue) cannot be split without
        # inventing a time inside that token. Keep it intact instead.
        return None
    positions = _candidate_text_positions_for_dp(
        text,
        split_count=split_count,
        word_positions=word_positions,
    )
    if not positions:
        return None
    ratios_by_position = _text_unit_prefix_ratios(text, positions)
    anchors = _anchor_times(block, start=start, end=end)
    # Reverse map: a word-gap anchor knows which word follows it, and that word's
    # character offset is the position it should pair with exactly.
    position_by_word_start = {
        round(word_start, 4): position for position, word_start in word_positions.items()
    }
    snap_window_s = _weak_cut_snap_window_s(duration, options)
    nodes: list[dict] = [
        {
            "position": 0,
            "time_s": start,
            "ratio": 0.0,
            "source": "start",
            "previous_word_end_s": start,
            "anchor_strength": 0.0,
            "snap_distance_s": 0.0,
        }
    ]
    for position in positions:
        ratio = ratios_by_position.get(position, 0.0)
        target = start + duration * ratio
        measured_word_start = word_positions.get(position)
        if measured_word_start is not None:
            # Text position and time come from the same measured word. A
            # separate word-gap node may still mark this as a preferable text
            # break, but both use the same next-word onset. A non-silence
            # fallback must never detach the two again.
            time_s = float(measured_word_start)
            source = "measured_word_start"
            strength = 0.0
            distance = abs(time_s - target)
        else:
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
                "previous_word_end_s": previous_word_ends.get(position, time_s),
                "anchor_strength": strength,
                "snap_distance_s": distance,
            }
        )
    for anchor in anchors:
        anchor_ratio = (float(anchor["time_s"]) - start) / max(duration, 1e-6)
        exact_position = position_by_word_start.get(
            round(float(anchor.get("next_word_start_s") or -1.0), 4)
        )
        if exact_position is not None:
            # A word-gap anchor: text position and time are the same measurement,
            # so pair them directly and let the snap distance be zero.
            position = exact_position
        elif positions:
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
        if not _is_valid_subtitle_text_boundary(text, position):
            continue
        target = start + duration * ratios_by_position.get(position, anchor_ratio)
        nodes.append(
            {
                "position": position,
                "time_s": float(anchor["time_s"]),
                "ratio": ratios_by_position.get(position, anchor_ratio),
                "source": str(anchor["anchor_type"]),
                "previous_word_end_s": previous_word_ends.get(
                    position,
                    float(anchor["time_s"]),
                ),
                "anchor_source": str(anchor.get("anchor_source") or ""),
                "anchor_strength": float(anchor.get("strength") or anchor.get("score") or 0.0),
                "snap_distance_s": 0.0
                if exact_position is not None
                else abs(float(anchor["time_s"]) - target),
            }
        )
    nodes.append(
        {
            "position": len(text),
            "time_s": end,
            "ratio": 1.0,
            "source": "end",
            "previous_word_end_s": previous_word_ends.get(len(text), end),
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
            # A long CTC blank may sit between the last word in this piece and
            # the first word in the next. It is not display content. Enforce the
            # 7s ceiling against measured speech extent; the previous cue can
            # end after its last word while the next still enters at its own
            # onset, leaving the blank genuinely subtitle-free.
            piece_content_end = max(
                piece_start,
                float(nodes[j].get("previous_word_end_s", piece_end)),
            )
            piece_duration = piece_content_end - piece_start
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


def _split_long_display_block_legacy(
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
        item = dict(block)
        _, _, measured_map_complete, measured_word_count = _measured_word_text_map(
            block,
            timing_text,
        )
        if measured_word_count and not measured_map_complete:
            item["subtitle_layout_split_skipped"] = (
                "measured_word_text_map_incomplete"
            )
        elif measured_word_count:
            item["subtitle_layout_split_skipped"] = (
                "measured_word_boundaries_unavailable"
            )
        return [item]
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
    split_anchor_sources = [str(node.get("anchor_source") or "") for node in nodes[1:-1]]
    has_word_gap = any(source == "word_gap" for source in split_anchor_sources)
    has_measured_word = any(source == "measured_word_start" for source in split_sources)
    if has_word_gap and has_measured_word:
        split_source = "word_gap_and_measured_word_dp"
    elif has_word_gap:
        # Split at a measured silence between two words, which is the only
        # source here that cannot land mid-word.
        split_source = "word_gap_dp"
    elif has_measured_word:
        split_source = "measured_word_dp"
    elif any(source in {"primary_cut", "weak_cut"} for source in split_sources):
        split_source = "acoustic_anchor_dp"
    else:
        split_source = "proportional_text_dp"

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
        piece_words = [
            word
            for word in _timed_words(block)
            if piece_start <= float(word.get("start", piece_start)) < piece_end
        ]
        measured_piece_words = [
            word for word in piece_words if _has_measured_word_timestamp(word)
        ]
        if measured_piece_words:
            item["acoustic_start"] = float(measured_piece_words[0]["start"])
            item["acoustic_end"] = max(
                item["acoustic_start"] + 0.05,
                float(measured_piece_words[-1]["end"]),
            )
        else:
            item["acoustic_start"] = piece_start
            item["acoustic_end"] = max(piece_start + 0.05, piece_end)
        item["acoustic_duration"] = max(
            0.0,
            item["acoustic_end"] - item["acoustic_start"],
        )
        item["display_start"] = item["start"]
        item["display_end"] = item["end"]
        item["display_duration"] = max(0.0, item["display_end"] - item["display_start"])
        for key, pieces in text_fields.items():
            item[key] = pieces[index] if index < len(pieces) else ""
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
        item["subtitle_layout_split"] = "max_display_duration"
        item["subtitle_layout_split_source"] = split_source
        # Which side of this piece is the middle of a sentence. The translator
        # sees cues one at a time and will otherwise close each fragment off as
        # a complete sentence; these two flags are what tells it not to.
        #
        # Written relative to the parent rather than to this pass, because
        # `_split_long_display_blocks` runs twice and a piece can be split
        # again: the first piece inherits whatever the parent said about its
        # left edge, the last inherits its right edge, and every internal edge
        # is a continuation by construction.
        item["continues_from_previous"] = (
            bool(block.get("continues_from_previous")) if index == 0 else True
        )
        item["continues_into_next"] = (
            bool(block.get("continues_into_next"))
            if index == len(boundaries) - 2
            else True
        )
        item["layout_engine"] = options.layout_engine
        item["layout_version"] = "subtitle_layout_v2"
        item["timing_model"] = options.timing_model
        boundary_node = nodes[index + 1] if index + 1 < len(nodes) else None
        boundary_source = str(boundary_node.get("source") or "") if boundary_node else ""
        boundary_anchor_source = (
            str(boundary_node.get("anchor_source") or "") if boundary_node else ""
        )
        item["anchor_used"] = boundary_source in {
            "primary_cut",
            "weak_cut",
            "measured_word_start",
        }
        item["anchor_type"] = boundary_source if item["anchor_used"] else ""
        item["anchor_score"] = 0.0 if boundary_node is None else float(boundary_node.get("anchor_strength") or 0.0)
        item["snap_distance_s"] = 0.0 if boundary_node is None else float(boundary_node.get("snap_distance_s") or 0.0)
        item["snap_reason"] = "anchor_aware_dp_v2"
        item["layout_score"] = float(plan.get("score") or 0.0)
        item["text_break_type"] = (
            "word_gap_boundary"
            if boundary_anchor_source == "word_gap"
            else "measured_word_boundary"
            if boundary_source == "measured_word_start"
            else "dp_text_boundary"
        )
        item["proportional_fallback_used"] = not item["anchor_used"]
        split_blocks.append(item)
    return split_blocks


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
        item["snap_reason"] = "measured_safe_boundary_dp_v3"
        item["layout_score"] = float(plan.get("score") or 0.0)
        item["text_break_type"] = end_boundary_kind
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
    # Preserve the legacy display policy for inputs without measured word
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
            progress=lambda current, total: stage("layout_dp_pass1", current, total),
        )
    )
    stage("timeline_polish", 0, 1)
    prepared = _polish_subtitle_timeline(prepared, options=options)
    stage("timeline_polish", 1, 1)
    # v3 jointly plans source length and lexical duration in one pass. Running
    # it twice can only repeat work; more importantly, a later pass must not
    # reinterpret an exact piece as permission to introduce a new boundary.
    stage("layout_dp_pass2", 1, 1)
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
