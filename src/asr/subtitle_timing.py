from __future__ import annotations

import re
import statistics

from asr import alignment
from asr.alignment import normalize_text

SYNTHETIC_TIMESTAMP_KIND = "synthetic_proportional"
ALIGNED_TIMESTAMP_KIND = "ctc_forced_alignment"

SYNTHETIC_TIMING_SOURCE = "synthetic_boundary_proportional"
ALIGNED_TIMING_SOURCE = "ctc_forced_alignment_v1"

# A leading or trailing character further than this many median inter-character
# gaps from its neighbour is treated as dragged rather than spoken there.
#
# This is not a general robustness knob; it targets one measured failure mode.
# Phase 1 found that alignment bodies are near-frame-exact (median character
# shift 1.9 ms) while isolated characters at a segment seam get pulled into
# neighbouring audio - 18% of evaluated cores had at least one character off by
# more than 400 ms. A subtitle whose start came straight from `spans[0]` would
# inherit that error at a rate of roughly one line in five.
BOUNDARY_OUTLIER_GAP_FACTOR = 4.0


def _clean_text(text: str) -> str:
    cleaned = (text or "").replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"[ \t]+", " ", cleaned)


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\S+|.", text) if token.strip()]


def _build_tokens_over_window(tokens: list[str], start: float, end: float) -> list[dict]:
    if not tokens:
        return []

    clipped_end = max(start, end)
    total_chars = sum(max(1, len(token.strip())) for token in tokens)
    total_duration = max(0.0, clipped_end - start)
    if total_chars <= 0 or total_duration <= 0:
        return []

    cursor = start
    words: list[dict] = []
    for idx, token in enumerate(tokens):
        weight = max(1, len(token.strip()))
        token_end = (
            clipped_end
            if idx == len(tokens) - 1
            else min(clipped_end, cursor + total_duration * (weight / total_chars))
        )
        words.append(
            {
                "start": cursor,
                "end": max(cursor, token_end),
                "word": token,
                "timestamp_kind": SYNTHETIC_TIMESTAMP_KIND,
            }
        )
        cursor = token_end
    return words


def build_boundary_word_timestamps(
    text: str,
    start: float,
    end: float,
) -> tuple[list[dict], str, dict]:
    cleaned = _clean_text(text)
    tokens = _tokenize(cleaned)
    if not tokens:
        return [], "empty", {
            "timing_source": SYNTHETIC_TIMING_SOURCE,
            "word_timestamps_real": False,
        }
    return (
        _build_tokens_over_window(tokens, start, end),
        "boundary_proportional",
        {
            "timing_source": SYNTHETIC_TIMING_SOURCE,
            "word_timestamps_real": False,
        },
    )


def _alignment_tokens(tokens: list[str]) -> list[str]:
    """Re-split whitespace tokens down to the granularity that was measured.

    Japanese has no spaces, so `_tokenize`'s `\\S+` rule turns an entire line
    into a single token - and a single token spanning a segment discards exactly
    the per-character timing the alignment was built to produce. Latin runs stay
    grouped, because there a whitespace token really is a word and splitting it
    per letter would report letter times as word times.
    """
    out: list[str] = []
    for token in tokens:
        buffer = ""
        for char in token:
            if char.isascii() and char.isalnum():
                buffer += char
                continue
            if buffer:
                out.append(buffer)
                buffer = ""
            out.append(char)
        if buffer:
            out.append(buffer)
    return out


def _robust_bounds(
    starts: list[float], ends: list[float]
) -> tuple[float, float, bool]:
    """Token extent, with a dragged first or last character pulled back in.

    Returns `(start, end, trimmed)`. Interior characters are never touched: the
    measurement says they are right, and moving them would trade a real
    timestamp for a smoothed one.
    """
    if len(starts) < 3:
        # Two characters give no gap distribution to call anything an outlier
        # against, so there is nothing to decide and the raw extent stands.
        return starts[0], ends[-1], False

    gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
    median_gap = statistics.median(gaps)
    if median_gap <= 0.0:
        return starts[0], ends[-1], False

    limit = BOUNDARY_OUTLIER_GAP_FACTOR * median_gap
    trimmed = False

    start = starts[0]
    if starts[1] - starts[0] > limit:
        start = starts[1] - median_gap
        trimmed = True

    end = ends[-1]
    if ends[-1] - ends[-2] > limit:
        end = ends[-2] + median_gap
        trimmed = True

    return start, max(start, end), trimmed


def build_aligned_word_timestamps(
    text: str,
    char_spans: list,
    window_start: float,
    window_end: float,
    acoustic_extent: tuple[float, float] | None = None,
) -> tuple[list[dict], str, dict]:
    """Word timestamps measured from audio, not spread across the window.

    `char_spans` are `asr.alignment.CharSpan` values in absolute seconds, one
    per character of `normalize_text(text)`. The caller produces them by forced
    alignment; this function only regroups them onto the tokens the subtitle
    layer works in and applies the seam-outlier rule.

    Falls back to the proportional path whenever the spans cannot be trusted to
    correspond to the text - a silent mismatch here would attach real-looking
    timestamps to the wrong characters, which is worse than admitting the times
    are synthetic.
    """
    cleaned = _clean_text(text)
    tokens = _tokenize(cleaned)
    if not tokens:
        return [], "empty", {
            "timing_source": SYNTHETIC_TIMING_SOURCE,
            "word_timestamps_real": False,
        }

    normalized_total = sum(len(normalize_text(token)) for token in tokens)
    if not char_spans or normalized_total != len(char_spans):
        # Normalisation changed the character count (NFKC can expand a glyph),
        # or the aligner was given different text. Either way the index mapping
        # below would be off by an unknown amount.
        words, _, _ = build_boundary_word_timestamps(text, window_start, window_end)
        return (
            words,
            "boundary_proportional",
            {
                "timing_source": SYNTHETIC_TIMING_SOURCE,
                "word_timestamps_real": False,
                "alignment_fallback_reason": "char_span_count_mismatch",
            },
        )

    # The seam outlier is a property of the segment's edge, not of any one
    # token, so the trusted extent is decided once over every character and each
    # token is then clamped into it. Deciding per token would be a no-op here:
    # most tokens are a single character and have no gap distribution to judge.
    segment_start, segment_end, trimmed = _robust_bounds(
        [span.start_s for span in char_spans], [span.end_s for span in char_spans]
    )

    # The edge characters' spans mark where the head is confident, which sits
    # inside where the sound actually is (see `alignment.speech_extent`). The
    # extent may therefore only push an edge OUTWARD, and by at most the walk's
    # own cap measured from the trusted bound - so a seam outlier that
    # `_robust_bounds` just pulled in cannot be dragged back out past it.
    edged = False
    if acoustic_extent is not None:
        extent_start, extent_end = acoustic_extent
        widened_start = max(
            segment_start - alignment.ONSET_BACKOFF_MAX_S,
            min(segment_start, float(extent_start)),
        )
        widened_end = min(
            segment_end + alignment.CODA_EXTEND_MAX_S,
            max(segment_end, float(extent_end)),
        )
        edged = widened_start < segment_start or widened_end > segment_end
        segment_start, segment_end = widened_start, max(widened_start, widened_end)

    words: list[dict] = []
    cursor = 0
    for token in _alignment_tokens(tokens):
        width = len(normalize_text(token))
        if width <= 0:
            continue
        spans = char_spans[cursor : cursor + width]
        cursor += width
        if not spans:
            continue
        start = min(max(spans[0].start_s, segment_start), segment_end)
        end = min(max(spans[-1].end_s, start), segment_end)
        words.append(
            {
                "start": start,
                "end": end,
                "word": token,
                "timestamp_kind": ALIGNED_TIMESTAMP_KIND,
                "alignment_score": round(
                    sum(span.score for span in spans) / len(spans), 4
                ),
            }
        )

    if words and edged:
        # Only the outer edges. The clamp above already pulls tokens INTO the
        # trusted extent, but it cannot push the first and last one out to it,
        # and those two are the only ones the inset affects.
        words[0]["start"] = min(words[0]["start"], segment_start)
        words[-1]["end"] = max(words[-1]["end"], segment_end)

    return (
        words,
        "ctc_forced_alignment",
        {
            "timing_source": ALIGNED_TIMING_SOURCE,
            "word_timestamps_real": True,
            "aligned_characters": len(char_spans),
            "boundary_trimmed": bool(trimmed),
            "boundary_edged": bool(words and edged),
            # Mean over tokens of the mean per-character score. The post-gate
            # reads this: text the acoustics do not support aligns badly.
            "alignment_score": round(
                sum(word["alignment_score"] for word in words) / len(words), 4
            )
            if words
            else None,
        },
    )
