"""Pre-translation subtitle cue planning and frame-class track integration."""

from __future__ import annotations

import json
from collections import Counter

from pipeline import stage_log as stage_log_module


def _build_japanese_srt_blocks(segments: list[dict]) -> list[dict]:
    return [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "ja_text": str(seg.get("text", "")),
            "zh_text": str(seg.get("text", "")),
            "words": list(seg.get("words") or []),
            "source_segment_ids": list(seg.get("source_segment_ids") or [idx]),
            "acoustic_start": seg.get("acoustic_start", seg.get("start")),
            "acoustic_end": seg.get("acoustic_end", seg.get("end")),
            "acoustic_duration": seg.get("acoustic_duration"),
            "chunk_acoustic_start": seg.get("chunk_acoustic_start"),
            "chunk_acoustic_end": seg.get("chunk_acoustic_end"),
            "chunk_acoustic_duration": seg.get("chunk_acoustic_duration"),
            "primary_cut_candidates": list(seg.get("primary_cut_candidates") or []),
            "weak_cut_candidates": list(seg.get("weak_cut_candidates") or []),
            # The post-gate marks a chunk and `_annotate_segments_with_postgate`
            # brings the mark down to the segment, but this key list is where it
            # used to stop: a verdict nobody downstream can read is a detector
            # running for nothing. Pieces inherit it through `dict(block)`.
            "postgate_flags": list(seg.get("postgate_flags") or []),
        }
        for idx, seg in enumerate(segments)
    ]


def _frame_class_reader(asr_details: dict | None):
    """A ``(start_s, end_s) -> shares`` callable, or None without a v2 head.

    Built here because this is where the ASR stage and the subtitle stage meet.
    Neither imports the other, and neither should: the arithmetic lives in
    ``asr.alignment``, the decision lives in ``subtitles.vocalisation``, and the
    integration belongs to the pipeline layer.
    """
    if not isinstance(asr_details, dict):
        return None
    from asr.alignment import ENCODER_FRAME_S, span_class_shares
    from asr.pipeline import decode_frame_class_track

    decoded = decode_frame_class_track(asr_details.get("frame_class_track"))
    if decoded is None:
        return None
    posteriors, frame_s = decoded
    # `span_class_shares` takes an upsample factor and derives the frame width
    # from the encoder rate; the track already knows its own width, so the
    # factor is recovered rather than assumed - a head trained at a different
    # upsample would otherwise silently index the wrong frames.
    upsample = max(1, int(round(ENCODER_FRAME_S / frame_s))) if frame_s > 0 else 1

    def shares(start_s: float, end_s: float) -> dict | None:
        measured = span_class_shares(
            posteriors,
            start_s,
            end_s,
            upsample=upsample,
        )
        return measured if measured.get("frames") else None

    return shares


def _prepare_translation_cues(
    segments: list[dict],
    *,
    subtitle_options,
    on_stage=None,
    asr_details: dict | None = None,
    subtitle_module=None,
) -> tuple[list[dict], dict]:
    if subtitle_module is None:
        from subtitles import writer as subtitle_module
    source_blocks = _build_japanese_srt_blocks(segments)
    prepare_diagnostics: dict = {}
    cues = subtitle_module.prepare_srt_blocks(
        source_blocks,
        options=subtitle_options,
        on_stage=on_stage,
        diagnostics=prepare_diagnostics,
        acoustic_classes=_frame_class_reader(asr_details),
    )
    normalized: list[dict] = []
    for cue_id, cue in enumerate(cues):
        item = dict(cue)
        ja_text = str(item.get("ja_text") or item.get("text") or "").strip()
        item["cue_id"] = cue_id
        item["text"] = ja_text
        item["ja_text"] = ja_text
        item["zh_text"] = str(item.get("zh_text") or ja_text)
        item["words"] = list(item.get("words") or [])
        # Normalized here rather than left absent so every cue answers the
        # question, and the prompt layer can serialize a fact instead of an
        # inference from a missing key.
        item["continues_from_previous"] = bool(item.get("continues_from_previous"))
        item["continues_into_next"] = bool(item.get("continues_into_next"))
        if "source_segment_ids" not in item:
            item["source_segment_ids"] = [cue_id]
        normalized.append(item)
    split_skipped = Counter(
        str(cue.get("subtitle_layout_split_skipped") or "").strip()
        for cue in normalized
        if str(cue.get("subtitle_layout_split_skipped") or "").strip()
    )
    split_sources = Counter(
        str(cue.get("subtitle_layout_split_source") or "").strip()
        for cue in normalized
        if str(cue.get("subtitle_layout_split_source") or "").strip()
    )
    # Which evidence ended each cue, and how much silence was actually there.
    # This is the layout's cut-point decision, and the finished SRT keeps no
    # trace of it: a cue ending at 。 and one ending in a 0.13s gap look the
    # same afterwards, and only the second is a guess.
    break_types = Counter(
        str(cue.get("text_break_type") or "").strip()
        for cue in normalized
        if str(cue.get("text_break_type") or "").strip()
    )
    word_gap_widths = sorted(
        float(cue.get("text_break_gap_s") or 0.0)
        for cue in normalized
        if str(cue.get("text_break_type") or "") == "word_gap"
    )
    # Counted here rather than at the ASR stage because the question worth
    # answering is not how many chunks were flagged but how many flagged cues
    # survived the layout and the vocalisation filter into the finished file.
    postgate_flags = Counter(
        flag
        for cue in normalized
        for flag in (cue.get("postgate_flags") or [])
        if str(flag).strip()
    )
    postgate_flagged_cues = sum(
        1 for cue in normalized if (cue.get("postgate_flags") or [])
    )
    return normalized, {
        "schema": "subtitle_cue_summary_v2",
        "cues_before": len(cues),
        "cues_after": len(normalized),
        "counts": {"keep": len(normalized)},
        "layout_diagnostics": {
            **prepare_diagnostics,
            "subtitle_layout_split_skipped": dict(sorted(split_skipped.items())),
            "subtitle_layout_split_source": dict(sorted(split_sources.items())),
            "subtitle_layout_break_type": dict(sorted(break_types.items())),
            "layout_word_gap_cut_count": len(word_gap_widths),
            "layout_word_gap_cut_under_0p2s": sum(
                1 for width in word_gap_widths if width < 0.2
            ),
            "layout_word_gap_median_s": (
                round(word_gap_widths[len(word_gap_widths) // 2], 3)
                if word_gap_widths
                else None
            ),
            "display_clamped_to_max": sum(
                bool(cue.get("display_clamped_to_max")) for cue in normalized
            ),
            "proportional_fallback_used": sum(
                bool(cue.get("proportional_fallback_used")) for cue in normalized
            ),
            "duration_violation": sum(
                bool(cue.get("duration_violation")) for cue in normalized
            ),
            "source_char_violation": sum(
                bool(cue.get("source_char_violation")) for cue in normalized
            ),
            "duration_soft_cap_violation": sum(
                bool(cue.get("duration_soft_cap_violation")) for cue in normalized
            ),
            "exact_measured_timeline": sum(
                bool(cue.get("exact_measured_timeline")) for cue in normalized
            ),
            # What the translator is told about sentences the layout split.
            "continues_from_previous": sum(
                bool(cue.get("continues_from_previous")) for cue in normalized
            ),
            "continues_into_next": sum(
                bool(cue.get("continues_into_next")) for cue in normalized
            ),
            "postgate_flagged_cues": postgate_flagged_cues,
            "postgate_cue_flags": dict(sorted(postgate_flags.items())),
        },
    }


def _subtitle_cue_plan_summary(
    *,
    segments_before: int,
    mode: str,
    cue_summary: dict,
) -> dict:
    return {
        "schema": str(cue_summary.get("schema") or "subtitle_cue_summary_v2"),
        "segments_before": segments_before,
        "cues_before": int(cue_summary.get("cues_before", 0)),
        "cues_after": int(cue_summary.get("cues_after", 0)),
        "layout_diagnostics": dict(cue_summary.get("layout_diagnostics") or {}),
        "mode": mode,
        "stage": "pre_translation",
    }


def _log_subtitle_cue_diagnostics(logger, cue_summary: dict) -> None:
    diagnostics = dict(cue_summary.get("layout_diagnostics") or {})
    stage_log_module._log_stage(
        logger,
        "subtitle_cue_plan_diagnostics "
        + json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
    )
