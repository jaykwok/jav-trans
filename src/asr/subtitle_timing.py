from __future__ import annotations

import re


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
                "timestamp_kind": "synthetic_proportional",
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
            "timing_source": "synthetic_boundary_proportional",
            "word_timestamps_real": False,
        }
    return (
        _build_tokens_over_window(tokens, start, end),
        "boundary_proportional",
        {
            "timing_source": "synthetic_boundary_proportional",
            "word_timestamps_real": False,
        },
    )
