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


def write_srt(blocks: list[dict], path: str, *, show_gender: bool | None = None):
    """
    blocks: [{start, end, zh_text}]
    zh_text may contain \\n to separate multiple speakers within one subtitle block.
    """
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
                index += 1
                continue
            break
        merged.append(current)
        index += 1
    return merged


def write_bilingual_srt(blocks: list[dict], path: str, *, show_gender: bool | None = None):
    """blocks: [{start, end, ja_text, zh_text}] — Japanese line above Chinese."""
    blocks = _merge_adjacent_short_blocks(blocks)
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

