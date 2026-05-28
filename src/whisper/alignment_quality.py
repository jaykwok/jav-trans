from __future__ import annotations

from typing import Any, Literal


AlignmentQuality = Literal[
    "forced",
    "partial",
    "vad_coarse",
    "proportional",
    "drop_or_review",
]
FallbackType = Literal["none", "vad_coarse", "proportional", "unknown"]


_NORMAL_MODES = {"", "empty", "forced_aligner"}
_VAD_FALLBACK_MARKERS = (
    "aligner_vad_fallback",
    "vad fallback",
    "vad 回退",
    "vad 约束",
    "silero_vad",
)
_PROPORTIONAL_FALLBACK_MARKERS = (
    "even_fallback",
    "proportional",
    "等比分配",
    "比例时间戳",
)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def infer_alignment_fallback_type(
    *,
    alignment_mode: str,
    fallback_lines: list[str] | tuple[str, ...] | None = None,
) -> FallbackType:
    mode = (alignment_mode or "").strip()
    lines = "\n".join(str(line) for line in (fallback_lines or []))
    haystack = f"{mode}\n{lines}".lower()

    if any(marker in haystack for marker in _VAD_FALLBACK_MARKERS):
        return "vad_coarse"
    if any(marker in haystack for marker in _PROPORTIONAL_FALLBACK_MARKERS):
        return "proportional"
    if mode and mode not in _NORMAL_MODES:
        return "unknown"
    if lines.strip():
        return "unknown"
    return "none"


def classify_alignment_quality(
    *,
    text: str,
    duration_s: float,
    align_text_empty: bool,
    asr_dropped_uncertain: bool,
    asr_qc_severity: str = "",
    alignment_mode: str = "",
    align_error: str = "",
    fallback_lines: list[str] | tuple[str, ...] | None = None,
    sentinel_lines: list[str] | tuple[str, ...] | None = None,
    aligned_segment_count: int = 0,
    word_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify alignment output without pretending coarse fallback is precise.

    The label is diagnostic metadata. It should not delete ASR text by itself.
    """

    stripped_text = (text or "").strip()
    fallback_type = infer_alignment_fallback_type(
        alignment_mode=alignment_mode,
        fallback_lines=fallback_lines,
    )
    stats = word_stats or {}
    word_count = _safe_int(stats.get("word_count"))
    zero_count = _safe_int(stats.get("zero_or_negative_count"))
    zero_heavy = word_count >= 2 and zero_count / max(1, word_count) >= 0.55

    reasons: list[str] = []
    if asr_dropped_uncertain:
        reasons.append("asr_dropped_uncertain")
    if not stripped_text and duration_s >= 1.0:
        reasons.append("empty_text_for_chunk")
    if stripped_text and align_text_empty:
        reasons.append("align_text_empty")
    if stripped_text and aligned_segment_count <= 0 and not asr_dropped_uncertain:
        reasons.append("text_without_output_segment")
    if (asr_qc_severity or "").strip() == "reject":
        reasons.append("asr_qc_reject")

    if reasons:
        return {
            "alignment_quality": "drop_or_review",
            "fallback_type": fallback_type,
            "alignment_quality_reasons": _unique(reasons),
        }

    if fallback_type == "vad_coarse":
        return {
            "alignment_quality": "vad_coarse",
            "fallback_type": fallback_type,
            "alignment_quality_reasons": ["fallback_type_vad_coarse"],
        }
    if fallback_type == "proportional":
        return {
            "alignment_quality": "proportional",
            "fallback_type": fallback_type,
            "alignment_quality_reasons": ["fallback_type_proportional"],
        }
    if fallback_type == "unknown":
        return {
            "alignment_quality": "partial",
            "fallback_type": fallback_type,
            "alignment_quality_reasons": ["fallback_type_unknown"],
        }

    partial_reasons: list[str] = []
    if align_error:
        partial_reasons.append("alignment_error")
    if sentinel_lines:
        partial_reasons.append("alignment_sentinel")
    if zero_heavy:
        partial_reasons.append("word_timing_zero_heavy")
    if partial_reasons:
        return {
            "alignment_quality": "partial",
            "fallback_type": fallback_type,
            "alignment_quality_reasons": _unique(partial_reasons),
        }

    mode = (alignment_mode or "").strip()
    if mode == "forced_aligner" and (word_count > 0 or aligned_segment_count > 0):
        return {
            "alignment_quality": "forced",
            "fallback_type": fallback_type,
            "alignment_quality_reasons": [],
        }
    if not stripped_text or mode == "empty":
        return {
            "alignment_quality": "drop_or_review",
            "fallback_type": fallback_type,
            "alignment_quality_reasons": ["empty_or_unaligned_text"],
        }
    return {
        "alignment_quality": "partial",
        "fallback_type": fallback_type,
        "alignment_quality_reasons": ["alignment_mode_missing_or_unconfirmed"],
    }
