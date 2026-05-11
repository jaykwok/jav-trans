import os
import re
import logging

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


MAX_SUBTITLE_DURATION = float(os.getenv("MAX_SUBTITLE_DURATION", "8.0"))
SUBTITLE_SOFT_MAX_S = float(os.getenv("SUBTITLE_SOFT_MAX_S", "6.0"))
SUBTITLE_SOFT_SPLIT_ENABLED = os.getenv(
    "SUBTITLE_SOFT_SPLIT_ENABLED", "1"
).strip().lower() in {"1", "true", "yes", "on"}
MIN_SUBTITLE_DURATION = float(os.getenv("MIN_SUBTITLE_DURATION", "0.6"))
SUBTITLE_READING_CPS = max(1.0, float(os.getenv("SUBTITLE_READING_CPS", "7.0")))
SUBTITLE_READING_BASE = float(os.getenv("SUBTITLE_READING_BASE", "0.35"))
SUBTITLE_DURATION_RATIO_CAP = max(1.0, float(os.getenv("SUBTITLE_DURATION_RATIO_CAP", "1.65")))
SUBTITLE_DURATION_GRACE = float(os.getenv("SUBTITLE_DURATION_GRACE", "0.9"))
SUBTITLE_GAP_PADDING = float(os.getenv("SUBTITLE_GAP_PADDING", "0.05"))
SUBTITLE_MIN_DURATION = float(os.getenv("SUBTITLE_MIN_DURATION", "0.6"))
SUBTITLE_TIMELINE_MODE = os.getenv("SUBTITLE_TIMELINE_MODE", "alignment").strip().lower()
SUBTITLE_BILINGUAL_SECONDARY_WEIGHT = float(
    os.getenv("SUBTITLE_BILINGUAL_SECONDARY_WEIGHT", "0.4")
)
SUBTITLE_ASCII_CHAR_WEIGHT = float(os.getenv("SUBTITLE_ASCII_CHAR_WEIGHT", "0.55"))
SRT_LINE_MAX_CHARS = max(0, int(os.getenv("SRT_LINE_MAX_CHARS", "25")))
_MIN_DUR_FOR_GENDER_GUARD = 0.5
SUBTITLE_MERGE_ADJACENT = os.getenv("SUBTITLE_MERGE_ADJACENT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
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


def _count_text_units(text: str) -> float:
    compact = _COMPACT_SPACE_RE.sub("", (text or "").strip())
    if not compact:
        return 0.0

    units = 0.0
    for char in compact:
        if char.isascii() and char.isalnum():
            units += SUBTITLE_ASCII_CHAR_WEIGHT
        elif char.isascii():
            units += 0.35
        else:
            units += 1.0
    return units


def _estimate_reading_duration(block: dict) -> float:
    ja_units = _count_text_units(block.get("ja_text", ""))
    zh_units = _count_text_units(block.get("zh_text", ""))
    primary_units = max(ja_units, zh_units)
    secondary_units = min(ja_units, zh_units)
    total_units = primary_units + secondary_units * SUBTITLE_BILINGUAL_SECONDARY_WEIGHT

    if total_units <= 0:
        return MIN_SUBTITLE_DURATION

    reading_duration = SUBTITLE_READING_BASE + total_units / SUBTITLE_READING_CPS
    reading_duration = max(MIN_SUBTITLE_DURATION, reading_duration)
    if MAX_SUBTITLE_DURATION > 0:
        reading_duration = min(reading_duration, MAX_SUBTITLE_DURATION)
    return reading_duration


def _resolve_subtitle_window(blocks: list[dict], idx: int) -> tuple[float, float]:
    block = blocks[idx - 1]
    start = float(block["start"])
    raw_end = max(start, float(block["end"]))
    if SUBTITLE_TIMELINE_MODE in {"alignment", "aligned", "raw"}:
        end = raw_end
        if idx < len(blocks):
            next_start = max(start + 0.05, float(blocks[idx]["start"]))
            end = min(end, max(start + 0.05, next_start - SUBTITLE_GAP_PADDING))
        if MAX_SUBTITLE_DURATION > 0:
            end = min(end, start + MAX_SUBTITLE_DURATION)
        end = max(end, start + SUBTITLE_MIN_DURATION)
        if idx < len(blocks):
            next_start = float(blocks[idx]["start"])
            end = min(end, max(start + 0.05, next_start - SUBTITLE_GAP_PADDING))
        if end <= start:
            end = start + 0.05
        return start, end

    target_duration = _estimate_reading_duration(block)
    trim_cap_duration = max(
        target_duration * SUBTITLE_DURATION_RATIO_CAP,
        target_duration + SUBTITLE_DURATION_GRACE,
        MIN_SUBTITLE_DURATION,
    )
    if MAX_SUBTITLE_DURATION > 0:
        trim_cap_duration = min(trim_cap_duration, MAX_SUBTITLE_DURATION)

    end = raw_end

    next_limit = None
    if idx < len(blocks):
        next_limit = max(start + 0.05, float(blocks[idx]["start"]) - SUBTITLE_GAP_PADDING)
        end = min(end, next_limit)

    if MAX_SUBTITLE_DURATION > 0:
        end = min(end, start + MAX_SUBTITLE_DURATION)

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


def _wrap_subtitle_text(text: str) -> str:
    lines = str(text or "").replace("\\n", "\n").split("\n")
    return "\n".join(
        wrapped
        for line in lines
        for wrapped in [_wrap_subtitle_line(line.strip(), SRT_LINE_MAX_CHARS)]
        if wrapped.strip()
    )


def _subtitle_prefix(block: dict, *, show_gender_override: bool | None = None) -> str:
    show_speaker = os.getenv("SUBTITLE_SHOW_SPEAKER", "0") == "1"
    show_gender = (
        os.getenv("SUBTITLE_SHOW_GENDER", "0") == "1"
        if show_gender_override is None
        else show_gender_override
    )
    speaker = block.get("speaker")
    gender = block.get("gender")
    if show_speaker and speaker is not None:
        return f"[S{speaker}] "
    if show_gender and gender in ("M", "F"):
        return f"[{gender}] "
    return ""


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


def _split_candidate_timing(
    words: list[dict],
    split_index: int,
    block_start: float,
    block_end: float,
) -> tuple[float, float, float, float] | None:
    if split_index <= 0 or split_index >= len(words):
        return None

    left_end = max(block_start, float(words[split_index - 1]["end"]))
    right_start = max(left_end, float(words[split_index]["start"]))
    if left_end <= block_start or right_start >= block_end:
        return None

    left_duration = left_end - block_start
    right_duration = block_end - right_start
    if left_duration < SUBTITLE_MIN_DURATION or right_duration < SUBTITLE_MIN_DURATION:
        return None

    return left_end, right_start, left_duration, right_duration


def _best_split_candidate(
    candidates: list[tuple[int, int | None]],
    words: list[dict],
    block_start: float,
    block_end: float,
) -> tuple[int, int | None] | None:
    if not candidates:
        return None

    duration = block_end - block_start
    target_offset = min(max(SUBTITLE_MIN_DURATION, duration / 2), SUBTITLE_SOFT_MAX_S)
    target_time = block_start + target_offset
    scored: list[tuple[float, int, int | None]] = []
    seen: set[tuple[int, int | None]] = set()

    for split_index, zh_split_at in candidates:
        key = (split_index, zh_split_at)
        if key in seen:
            continue
        seen.add(key)
        timing = _split_candidate_timing(words, split_index, block_start, block_end)
        if timing is None:
            continue
        left_end, _right_start, left_duration, right_duration = timing
        score = abs(left_end - target_time)
        if MAX_SUBTITLE_DURATION > 0 and left_duration > MAX_SUBTITLE_DURATION:
            score += 1000 + left_duration
        if MAX_SUBTITLE_DURATION > 0 and right_duration > MAX_SUBTITLE_DURATION:
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
) -> tuple[int, None] | None:
    if MAX_SUBTITLE_DURATION <= 0 or block_end - block_start <= MAX_SUBTITLE_DURATION:
        return None

    target_time = block_start + min(MAX_SUBTITLE_DURATION, (block_end - block_start) / 2)
    scored: list[tuple[float, int]] = []
    for split_index in range(1, len(words)):
        timing = _split_candidate_timing(words, split_index, block_start, block_end)
        if timing is None:
            continue
        left_end, _right_start, left_duration, right_duration = timing
        score = abs(left_end - target_time)
        if left_duration > MAX_SUBTITLE_DURATION:
            score += 1000 + left_duration
        if right_duration < SUBTITLE_MIN_DURATION:
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
) -> tuple[dict, dict] | None:
    block_start, block_end = _subtitle_block_window(block, words)
    timing = _split_candidate_timing(words, split_index, block_start, block_end)
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


def _soft_split_subtitle_block(block: dict, *, depth: int = 0) -> list[dict]:
    if not SUBTITLE_SOFT_SPLIT_ENABLED:
        return [block]

    words = _timed_words(block)
    if len(words) < 2 or depth > len(words):
        return [block]

    block_start, block_end = _subtitle_block_window(block, words)
    duration = block_end - block_start
    exceeds_soft_limit = SUBTITLE_SOFT_MAX_S > 0 and duration > SUBTITLE_SOFT_MAX_S
    exceeds_hard_limit = MAX_SUBTITLE_DURATION > 0 and duration > MAX_SUBTITLE_DURATION
    if not exceeds_soft_limit and not exceeds_hard_limit:
        return [block]

    candidate = _best_split_candidate(
        _zh_sentence_split_candidates(block, words),
        words,
        block_start,
        block_end,
    )
    if candidate is None:
        candidate = _best_split_candidate(
            _ja_particle_split_candidates(words),
            words,
            block_start,
            block_end,
        )
    if candidate is None and exceeds_hard_limit:
        candidate = _hard_word_split_candidate(words, block_start, block_end)
    if candidate is None:
        return [block]

    split_result = _split_block_at_word_boundary(block, words, candidate[0], candidate[1])
    if split_result is None:
        return [block]

    left, right = split_result
    return (
        _soft_split_subtitle_block(left, depth=depth + 1)
        + _soft_split_subtitle_block(right, depth=depth + 1)
    )


def _soft_split_subtitle_blocks(blocks: list[dict]) -> list[dict]:
    split_blocks: list[dict] = []
    for block in blocks:
        split_blocks.extend(_soft_split_subtitle_block(dict(block)))
    return split_blocks


def write_srt(blocks: list[dict], path: str, *, show_gender: bool | None = None):
    """
    blocks: [{start, end, zh_text}]
    zh_text may contain \\n to separate multiple speakers within one subtitle block.
    """
    blocks = _soft_split_subtitle_blocks(blocks)
    with open(path, "w", encoding="utf-8") as f:
        for idx, block in enumerate(blocks, 1):
            start, end = _resolve_subtitle_window(blocks, idx)

            start_str = format_timestamp(start)
            end_str   = format_timestamp(end)
            wrapped = _wrap_subtitle_text(
                _subtitle_prefix(block, show_gender_override=show_gender) + block.get("zh_text", "")
            )
            f.write(f"{idx}\n{start_str} --> {end_str}\n{wrapped}\n\n")


def _merge_adjacent_short_blocks(blocks: list[dict]) -> list[dict]:
    if not SUBTITLE_MERGE_ADJACENT or len(blocks) < 2:
        return list(blocks)

    merged: list[dict] = []
    index = 0
    max_chars = SRT_LINE_MAX_CHARS if SRT_LINE_MAX_CHARS > 0 else 25
    while index < len(blocks):
        current = dict(blocks[index])
        while index + 1 < len(blocks):
            nxt = blocks[index + 1]
            current_end = float(current.get("end", current.get("start", 0.0)))
            next_start = float(nxt.get("start", 0.0))
            gap = next_start - current_end
            ja_text = str(current.get("ja_text", "")).strip()
            combined_ja = " ".join(
                part
                for part in (ja_text, str(nxt.get("ja_text", "")).strip())
                if part
            )
            combined_zh = "，".join(
                part
                for part in (
                    str(current.get("zh_text", "")).strip(),
                    str(nxt.get("zh_text", "")).strip(),
                )
                if part
            )
            combined_chars = len(_COMPACT_SPACE_RE.sub("", combined_ja + combined_zh))
            combined_duration = float(nxt.get("end", next_start)) - float(
                current.get("start", 0.0)
            )

            # Speaker-aware merge guard: never merge across speaker boundaries
            if (
                current.get("speaker") is not None
                and nxt.get("speaker") is not None
                and current.get("speaker") != nxt.get("speaker")
            ):
                break

            # Gender-aware merge guard (F0 ground truth; duration floor prevents short noise fragments)
            if (
                current.get("gender") is not None
                and nxt.get("gender") is not None
                and current.get("gender") != nxt.get("gender")
                and (current.get("end", 0) - current.get("start", 0)) >= _MIN_DUR_FOR_GENDER_GUARD
                and (nxt.get("end", 0) - nxt.get("start", 0)) >= _MIN_DUR_FOR_GENDER_GUARD
            ):
                break

            if (
                gap < 0.2
                and combined_chars <= max_chars * 2
                and combined_duration <= 4.0
                and not ja_text.endswith(tuple(_SENTENCE_END_PUNCTUATION))
            ):
                current["end"] = nxt.get("end", current.get("end"))
                current["ja_text"] = combined_ja
                current["zh_text"] = combined_zh
                current["words"] = list(current.get("words") or []) + list(
                    nxt.get("words") or []
                )
                index += 1
                continue
            break
        merged.append(current)
        index += 1
    return merged


def write_bilingual_srt(blocks: list[dict], path: str, *, show_gender: bool | None = None):
    """blocks: [{start, end, ja_text, zh_text}] — Japanese line above Chinese."""
    blocks = _merge_adjacent_short_blocks(blocks)
    blocks = _soft_split_subtitle_blocks(blocks)
    with open(path, "w", encoding="utf-8") as f:
        for idx, block in enumerate(blocks, 1):
            start, end = _resolve_subtitle_window(blocks, idx)

            start_str = format_timestamp(start)
            end_str   = format_timestamp(end)
            prefix = _subtitle_prefix(block, show_gender_override=show_gender)
            ja_line = _wrap_subtitle_text(prefix + block.get("ja_text", ""))
            zh_text = str(block.get("zh_text", "")).strip()
            if not zh_text:
                logger.warning("Empty translated subtitle at index %s; using placeholder", idx)
                zh_text = "「未翻译」"
            zh_line = _wrap_subtitle_text(prefix + zh_text)
            content = "\n".join(
                line for line in (ja_line + "\n" + zh_line).split("\n") if line.strip()
            )
            f.write(f"{idx}\n{start_str} --> {end_str}\n{content}\n\n")

