from __future__ import annotations


def build_asr_manifest(segments: list[dict]) -> dict:
    total = len(segments)
    empty_count = sum(
        1 for segment in segments if not str(segment.get("text", "") or "").strip()
    )
    empty_ratio = empty_count / max(1, total)
    return {
        "segment_count": total,
        "empty_text_count": empty_count,
        "empty_text_ratio": round(empty_ratio, 6),
    }
