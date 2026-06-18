from __future__ import annotations

from typing import Any, Literal


AlignmentQuality = Literal[
    "boundary",
    "partial",
    "nonlexical",
    "drop_or_review",
]
FallbackType = Literal["none", "unknown"]


_NORMAL_MODES = {
    "",
    "empty",
    "boundary_proportional",
    "nonlexical",
}


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fallback_subtype(
    *,
    fallback_type: FallbackType,
    nonlexical_text: bool,
    stripped_text: str,
    duration_s: float,
    aligned_segment_count: int,
    align_error: str,
) -> str:
    if nonlexical_text:
        return "nonlexical_text"
    if not stripped_text and duration_s >= 1.0:
        return "asr_empty_text"
    if stripped_text and aligned_segment_count <= 0:
        return "text_without_output_segment"
    if fallback_type == "unknown":
        return "unknown_fallback"
    if align_error:
        return "subtitle_timing_error"
    return "none"


def infer_alignment_fallback_type(*, alignment_mode: str) -> FallbackType:
    mode = (alignment_mode or "").strip()
    if mode in {"boundary_proportional", "nonlexical"}:
        return "none"
    if mode and mode not in _NORMAL_MODES:
        return "unknown"
    return "none"


def classify_alignment_quality(
    *,
    text: str,
    duration_s: float,
    nonlexical_text: bool = False,
    alignment_mode: str = "",
    align_error: str = "",
    aligned_segment_count: int = 0,
    word_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify subtitle timing output.

    Boundary chunk timing is the runtime source of truth. The label is
    diagnostic metadata; CueQC is responsible for keep/drop routing.
    """

    stripped_text = (text or "").strip()
    fallback_type = infer_alignment_fallback_type(alignment_mode=alignment_mode)
    word_count = _safe_int((word_stats or {}).get("word_count"))
    subtype = _fallback_subtype(
        fallback_type=fallback_type,
        nonlexical_text=nonlexical_text,
        stripped_text=stripped_text,
        duration_s=duration_s,
        aligned_segment_count=aligned_segment_count,
        align_error=align_error,
    )

    reasons: list[str] = []
    if not stripped_text and duration_s >= 1.0:
        reasons.append("empty_text_for_chunk")
    if nonlexical_text and not reasons:
        return {
            "alignment_quality": "nonlexical",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": ["nonlexical_text"],
        }
    if stripped_text and aligned_segment_count <= 0:
        reasons.append("text_without_output_segment")

    if reasons:
        return {
            "alignment_quality": "drop_or_review",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": _unique(reasons),
        }

    if fallback_type == "unknown":
        return {
            "alignment_quality": "partial",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": ["fallback_type_unknown"],
        }

    if align_error:
        return {
            "alignment_quality": "partial",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": ["subtitle_timing_error"],
        }

    mode = (alignment_mode or "").strip()
    if mode == "boundary_proportional" and (word_count > 0 or aligned_segment_count > 0):
        return {
            "alignment_quality": "boundary",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": [],
        }
    if not stripped_text or mode == "empty":
        return {
            "alignment_quality": "drop_or_review",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": ["empty_or_unaligned_text"],
        }
    return {
        "alignment_quality": "partial",
        "fallback_type": fallback_type,
        "fallback_subtype": subtype,
        "alignment_quality_reasons": ["alignment_mode_missing_or_unconfirmed"],
    }
