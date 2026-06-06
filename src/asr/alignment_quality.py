from __future__ import annotations

from typing import Any, Literal


AlignmentQuality = Literal[
    "forced",
    "partial",
    "nonlexical",
    "vad_coarse",
    "proportional",
    "drop_or_review",
]
FallbackType = Literal["none", "vad_coarse", "proportional", "unknown"]


_NORMAL_MODES = {"", "empty", "forced_aligner", "nonlexical", "align_text_empty"}


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
    align_text_empty: bool,
    nonlexical_text: bool,
    asr_review_uncertain: bool,
    asr_qc_severity: str,
    stripped_text: str,
    duration_s: float,
    aligned_segment_count: int,
    align_error: str,
    sentinel_lines: list[str] | tuple[str, ...] | None,
    word_failure_reasons: list[str],
    zero_heavy: bool,
) -> str:
    severity = (asr_qc_severity or "").strip()
    if asr_review_uncertain:
        return "asr_review_uncertain"
    if severity == "reject":
        return "asr_qc_reject"
    if nonlexical_text and align_text_empty:
        return "nonlexical_text"
    if not stripped_text and duration_s >= 1.0:
        return "asr_empty_text"
    if stripped_text and align_text_empty:
        return "align_text_empty"
    if stripped_text and aligned_segment_count <= 0:
        return "text_without_output_segment"
    if fallback_type == "vad_coarse":
        if align_error:
            return "vad_coarse_after_align_error"
        if sentinel_lines:
            return "vad_coarse_after_sentinel"
        return "vad_coarse"
    if fallback_type == "proportional":
        if align_error:
            return "proportional_after_align_error"
        if sentinel_lines:
            return "proportional_after_sentinel"
        return "proportional"
    if fallback_type == "unknown":
        return "unknown_fallback"
    if align_error:
        return "forced_align_error"
    if sentinel_lines:
        if word_failure_reasons:
            return f"sentinel_{word_failure_reasons[0]}"
        if zero_heavy:
            return "sentinel_word_timing_zero_heavy"
        return "forced_sentinel"
    if word_failure_reasons:
        return word_failure_reasons[0]
    if zero_heavy:
        return "word_timing_zero_heavy"
    return "none"


def infer_alignment_fallback_type(*, alignment_mode: str) -> FallbackType:
    mode = (alignment_mode or "").strip()
    if mode == "aligner_vad_fallback":
        return "vad_coarse"
    if mode == "even_fallback":
        return "proportional"
    if mode in {"nonlexical", "align_text_empty"}:
        return "none"
    if mode and mode not in _NORMAL_MODES:
        return "unknown"
    return "none"


def classify_alignment_quality(
    *,
    text: str,
    duration_s: float,
    align_text_empty: bool,
    asr_review_uncertain: bool,
    nonlexical_text: bool = False,
    asr_qc_severity: str = "",
    alignment_mode: str = "",
    align_error: str = "",
    sentinel_lines: list[str] | tuple[str, ...] | None = None,
    aligned_segment_count: int = 0,
    word_stats: dict[str, Any] | None = None,
    word_failure_reasons: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Classify alignment output without pretending coarse fallback is precise.

    The label is diagnostic metadata. It should not delete ASR text by itself.
    """

    stripped_text = (text or "").strip()
    fallback_type = infer_alignment_fallback_type(alignment_mode=alignment_mode)
    stats = word_stats or {}
    word_count = _safe_int(stats.get("word_count"))
    zero_count = _safe_int(stats.get("zero_or_negative_count"))
    zero_heavy = word_count >= 2 and zero_count / max(1, word_count) >= 0.55
    subtype = _fallback_subtype(
        fallback_type=fallback_type,
        align_text_empty=align_text_empty,
        nonlexical_text=nonlexical_text,
        asr_review_uncertain=asr_review_uncertain,
        asr_qc_severity=asr_qc_severity,
        stripped_text=stripped_text,
        duration_s=duration_s,
        aligned_segment_count=aligned_segment_count,
        align_error=align_error,
        sentinel_lines=sentinel_lines,
        word_failure_reasons=_unique(list(word_failure_reasons or [])),
        zero_heavy=zero_heavy,
    )

    reasons: list[str] = []
    if asr_review_uncertain:
        reasons.append("asr_review_uncertain")
    if not stripped_text and duration_s >= 1.0:
        reasons.append("empty_text_for_chunk")
    if nonlexical_text and align_text_empty and not reasons:
        return {
            "alignment_quality": "nonlexical",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": ["nonlexical_text"],
        }
    if stripped_text and align_text_empty and not nonlexical_text:
        reasons.append("align_text_empty")
    if stripped_text and aligned_segment_count <= 0 and not asr_review_uncertain:
        reasons.append("text_without_output_segment")
    if (asr_qc_severity or "").strip() == "reject":
        reasons.append("asr_qc_reject")

    if reasons:
        return {
            "alignment_quality": "drop_or_review",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": _unique(reasons),
        }

    if fallback_type == "vad_coarse":
        return {
            "alignment_quality": "vad_coarse",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": ["fallback_type_vad_coarse"],
        }
    if fallback_type == "proportional":
        return {
            "alignment_quality": "proportional",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": ["fallback_type_proportional"],
        }
    if fallback_type == "unknown":
        return {
            "alignment_quality": "partial",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": ["fallback_type_unknown"],
        }

    partial_reasons: list[str] = []
    if align_error:
        partial_reasons.append("alignment_error")
    if sentinel_lines:
        partial_reasons.append("alignment_sentinel")
    if zero_heavy:
        partial_reasons.append("word_timing_zero_heavy")
    partial_reasons.extend(word_failure_reasons or [])
    if partial_reasons:
        return {
            "alignment_quality": "partial",
            "fallback_type": fallback_type,
            "fallback_subtype": subtype,
            "alignment_quality_reasons": _unique(partial_reasons),
        }

    mode = (alignment_mode or "").strip()
    if mode == "forced_aligner" and (word_count > 0 or aligned_segment_count > 0):
        return {
            "alignment_quality": "forced",
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
