"""Sparse frame supervision compiled from accepted Grok word timestamps.

The teacher is deliberately not expanded into a dense pseudo-alignment.  Word
islands are positive, only long gaps far from an island are negative, and
everything uncertain remains ignored.  This makes the auxiliary objective an
occupancy/boundary hint while canonical full-clip CTC remains the text loss.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from asr.alignment import ENCODER_FRAME_S, is_acoustic_char


IGNORE_LABEL = -1
BLANK_LABEL = 0
SPEECH_LABEL = 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _lexical(value: str) -> bool:
    return any(is_acoustic_char(character) for character in str(value))


def merge_intervals(
    intervals: Iterable[tuple[float, float]], *, maximum_gap_s: float
) -> list[tuple[float, float]]:
    ordered = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    )
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start - merged[-1][1] <= maximum_gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def load_accepted_frame_teachers(
    results_path: Path, accepted_manifest_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Join provider words to the strict accepted-source manifest.

    The results file also contains rejected transcripts.  The accepted
    manifest is therefore mandatory and acts as the quality gate rather than
    allowing every provider response to become supervision accidentally.
    """

    accepted_rows = _read_jsonl(accepted_manifest_path)
    accepted_ids = {str(row["source_id"]) for row in accepted_rows}
    if not accepted_ids:
        raise ValueError("accepted teacher manifest contains no source IDs")
    results: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(results_path):
        source_id = str(row.get("source_id") or "")
        if source_id in accepted_ids:
            if source_id in results:
                raise ValueError(f"duplicate teacher result for {source_id}")
            results[source_id] = row
    missing = sorted(accepted_ids - set(results))
    if missing:
        raise ValueError(
            f"accepted teacher sources missing from results: {len(missing)}; "
            f"first={missing[0]}"
        )

    manifest_sha = {
        str(row["source_id"]): str(row.get("teacher_audio_sha256") or "")
        for row in accepted_rows
    }
    teachers: dict[str, dict[str, Any]] = {}
    lexical_units = 0
    for source_id in sorted(accepted_ids):
        result = results[source_id]
        result_sha = str(result.get("audio_sha256") or "")
        expected_sha = manifest_sha[source_id]
        if expected_sha and result_sha != expected_sha:
            raise ValueError(f"teacher audio hash mismatch for {source_id}")
        duration_s = float(result.get("source_duration_s") or 0.0)
        intervals: list[tuple[float, float]] = []
        for word in (result.get("response") or {}).get("words") or []:
            if not _lexical(str(word.get("text") or "")):
                continue
            start = max(0.0, min(duration_s, float(word["start_s"])))
            end = max(0.0, min(duration_s, float(word["end_s"])))
            if end > start:
                intervals.append((start, end))
        if not intervals:
            raise ValueError(f"accepted teacher source has no lexical units: {source_id}")
        lexical_units += len(intervals)
        teachers[source_id] = {
            "source_id": source_id,
            "duration_s": duration_s,
            "lexical_intervals": intervals,
        }
    return teachers, {
        "accepted_sources": len(accepted_ids),
        "teacher_sources": len(teachers),
        "lexical_units": lexical_units,
    }


def compile_sparse_frame_targets(
    teacher: Mapping[str, Any],
    *,
    output_frames: int,
    upsample: int,
    positive_merge_gap_s: float = 0.15,
    positive_minimum_s: float | None = None,
    boundary_ignore_s: float = 0.10,
    negative_minimum_s: float = 0.50,
    start_offset_s: float = 0.0,
) -> np.ndarray:
    """Return ``speech=1``, ``blank=0``, and ``ignore=-1`` output labels.

    ``start_offset_s`` is where these frames begin inside the clip the teacher
    timed. Word timestamps are absolute to the source clip, while frame 0 is
    whatever the cached row starts at - so a row cropped to a sub-span needs the
    offset or every label lands early by exactly that much. Nothing raises when
    it is wrong: the loss simply teaches the head that speech happens somewhere
    other than where it does.
    """

    if output_frames < 1:
        raise ValueError("output_frames must be positive")
    if upsample < 1:
        raise ValueError("upsample must be positive")
    if min(positive_merge_gap_s, boundary_ignore_s, negative_minimum_s) < 0.0:
        raise ValueError("frame teacher durations must be non-negative")
    if start_offset_s < 0.0:
        raise ValueError("start_offset_s must be non-negative")
    frame_s = ENCODER_FRAME_S / float(upsample)
    minimum_s = (
        2.0 * frame_s if positive_minimum_s is None else positive_minimum_s
    )
    if minimum_s < 0.0:
        raise ValueError("positive_minimum_s must be non-negative")
    # Everything below works in row time: 0.0 is this row's first frame.
    duration_s = min(
        float(teacher["duration_s"]) - start_offset_s, output_frames * frame_s
    )
    if duration_s <= 0.0:
        raise ValueError("start_offset_s is past the end of the teacher clip")
    centers = (np.arange(output_frames, dtype=np.float64) + 0.5) * frame_s
    labels = np.full(output_frames, IGNORE_LABEL, dtype=np.int8)

    islands = merge_intervals(
        (
            (start - start_offset_s, end - start_offset_s)
            for start, end in (teacher.get("lexical_intervals") or ())
        ),
        maximum_gap_s=positive_merge_gap_s,
    )
    islands = [
        (max(0.0, start), min(duration_s, end))
        for start, end in islands
        if min(duration_s, end) - max(0.0, start) >= minimum_s
    ]
    if not islands:
        return labels

    protected = merge_intervals(
        (
            (max(0.0, start - boundary_ignore_s), min(duration_s, end + boundary_ignore_s))
            for start, end in islands
        ),
        maximum_gap_s=0.0,
    )
    cursor = 0.0
    safe_gaps: list[tuple[float, float]] = []
    for start, end in protected:
        if start - cursor >= negative_minimum_s:
            safe_gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration_s - cursor >= negative_minimum_s:
        safe_gaps.append((cursor, duration_s))

    for start, end in safe_gaps:
        labels[(centers >= start) & (centers < end)] = BLANK_LABEL
    for start, end in islands:
        labels[(centers >= start) & (centers < end)] = SPEECH_LABEL
    return labels


def balanced_sparse_frame_loss(log_probs, labels, torch):
    """Balanced blank-vs-speech NLL over labelled frames only."""

    if labels.shape != log_probs.shape[:2]:
        raise ValueError(
            f"frame label shape {tuple(labels.shape)} != logits {tuple(log_probs.shape[:2])}"
        )
    blank_log_probability = log_probs[..., 0]
    speech_log_probability = torch.logsumexp(log_probs[..., 1:], dim=-1)
    blank_mask = labels == BLANK_LABEL
    speech_mask = labels == SPEECH_LABEL
    parts = []
    if bool(blank_mask.any()):
        parts.append(-blank_log_probability[blank_mask].mean())
    if bool(speech_mask.any()):
        parts.append(-speech_log_probability[speech_mask].mean())
    if not parts:
        return log_probs.sum() * 0.0, {"blank_frames": 0, "speech_frames": 0}
    return torch.stack(parts).mean(), {
        "blank_frames": int(blank_mask.sum().item()),
        "speech_frames": int(speech_mask.sum().item()),
    }


def summarize_sparse_frame_probabilities(
    blank_probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, float | int]:
    """Describe held-out teacher agreement without choosing a train loss."""

    probabilities = np.asarray(blank_probabilities, dtype=np.float64)
    target = np.asarray(labels)
    if probabilities.shape != target.shape:
        raise ValueError(
            f"probability shape {probabilities.shape} != labels {target.shape}"
        )
    blank = probabilities[target == BLANK_LABEL]
    speech = probabilities[target == SPEECH_LABEL]
    if not len(blank) or not len(speech):
        return {"blank_frames": len(blank), "speech_frames": len(speech)}
    epsilon = np.finfo(np.float64).eps
    blank_nll = -np.log(np.clip(blank, epsilon, 1.0)).mean()
    speech_nll = -np.log(np.clip(1.0 - speech, epsilon, 1.0)).mean()
    blank_accuracy = float(np.mean(blank >= 0.5))
    speech_accuracy = float(np.mean(speech < 0.5))
    return {
        "blank_frames": len(blank),
        "speech_frames": len(speech),
        "mean_blank_probability_on_blank": round(float(np.mean(blank)), 6),
        "mean_blank_probability_on_speech": round(float(np.mean(speech)), 6),
        "blank_probability_margin": round(
            float(np.mean(blank) - np.mean(speech)), 6
        ),
        "blank_accuracy_at_0_5": round(blank_accuracy, 6),
        "speech_accuracy_at_0_5": round(speech_accuracy, 6),
        "balanced_accuracy_at_0_5": round(
            (blank_accuracy + speech_accuracy) / 2.0, 6
        ),
        "balanced_nll": round(float((blank_nll + speech_nll) / 2.0), 6),
    }
