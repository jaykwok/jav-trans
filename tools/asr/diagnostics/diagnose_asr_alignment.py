#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.alignment_quality import classify_alignment_quality  # noqa: E402
from asr.cueqc import text_features as cueqc_text_features  # noqa: E402
from asr.prealign import prepare_text_for_alignment, strip_alignment_punctuation  # noqa: E402
from asr.transcribe import _alignment_failure_reasons  # noqa: E402


_CHUNK_LOG_RE = re.compile(r"^chunk\s+(\d+):\s*(.*)$")
_WORD_COUNT_RE = re.compile(r"Alignment\s+词数:\s*(\d+)")
_ALIGNMENT_MODE_RE = re.compile(r"Alignment\s+模式:\s*(\S+)")
_CANDIDATE_BUCKET_ORDER = (
    "nonlexical_text",
    "align_text_empty",
    "empty_text_for_chunk",
    "repeat_repair_suggested",
    "text_without_output_segment",
    "partial_alignment",
    "vad_coarse_alignment",
    "proportional_alignment",
    "unknown_alignment_fallback",
    "abnormal_char_density",
    "diagnostic_warning",
)
_NONLEXICAL_TEXT_RE = re.compile(r"^[.。…、,!?！？\s]+$")


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def discover_aligned_jsons(workflow_root: Path | None, explicit: list[Path]) -> list[Path]:
    paths: list[Path] = []
    paths.extend(explicit)
    if workflow_root is not None:
        archive = workflow_root / "archived"
        if archive.exists():
            paths.extend(sorted(archive.glob("*/*.aligned_segments.json")))
        paths.extend(sorted(workflow_root.glob("*.aligned_segments.json")))
    existing = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def sibling_path(aligned_path: Path, suffix: str) -> Path | None:
    stem = aligned_path.name.removesuffix(".aligned_segments.json")
    candidate = aligned_path.with_name(f"{stem}.{suffix}")
    return candidate if candidate.exists() else None


def find_quality_json(aligned_path: Path, workflow_root: Path | None) -> Path | None:
    stem = aligned_path.name.removesuffix(".aligned_segments.json")
    candidates = [
        aligned_path.with_name(f"{stem}.quality_report.json"),
        aligned_path.parent.parent.parent / "quality_reports" / f"{stem}.quality_report.json",
    ]
    if workflow_root is not None:
        candidates.append(workflow_root / "quality_reports" / f"{stem}.quality_report.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_case_label(aligned_path: Path, workflow_root: Path | None) -> str:
    if workflow_root is not None:
        try:
            return workflow_root.name
        except Exception:
            pass
    parts = aligned_path.parts
    for part in reversed(parts):
        if part.startswith("full-workflow-"):
            return part.removeprefix("full-workflow-")
    return aligned_path.parent.name


def parse_chunk_logs(asr_log: Iterable[str]) -> dict[int, dict[str, Any]]:
    logs: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "lines": [],
            "alignment_mode": "",
            "alignment_word_count": None,
            "align_error": "",
            "sentinel_lines": [],
        }
    )
    for entry in asr_log:
        match = _CHUNK_LOG_RE.match(str(entry))
        if not match:
            continue
        chunk_index = int(match.group(1)) - 1
        line = match.group(2)
        item = logs[chunk_index]
        item["lines"].append(line)
        if mode_match := _ALIGNMENT_MODE_RE.search(line):
            item["alignment_mode"] = mode_match.group(1)
        if count_match := _WORD_COUNT_RE.search(line):
            item["alignment_word_count"] = int(count_match.group(1))
        if "Alignment 异常:" in line:
            item["align_error"] = line.split("Alignment 异常:", 1)[1].strip()
        if "哨兵" in line:
            item["sentinel_lines"].append(line)
    return {index: dict(value) for index, value in logs.items()}


def words_by_chunk(segments: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_chunk: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        for word in segment.get("words") or []:
            if not isinstance(word, dict):
                continue
            try:
                chunk_index = int(word.get("source_chunk_index", segment.get("source_chunk_index")))
            except (TypeError, ValueError):
                continue
            by_chunk[chunk_index].append(word)
    return {index: list(words) for index, words in by_chunk.items()}


def segment_counts_by_chunk(segments: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            counts[int(segment.get("source_chunk_index"))] += 1
        except (TypeError, ValueError):
            continue
    return counts


def word_timing_stats(words: list[dict[str, Any]]) -> dict[str, Any]:
    if not words:
        return {
            "word_count": 0,
            "zero_or_negative_count": 0,
            "tiny_span_count": 0,
            "min_word_duration_s": None,
            "max_word_duration_s": None,
        }
    durations: list[float] = []
    zero_or_negative = 0
    tiny = 0
    for word in words:
        try:
            start = float(word.get("start", 0.0))
            end = float(word.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        duration = end - start
        durations.append(duration)
        if duration <= 0.0:
            zero_or_negative += 1
        if duration <= 0.08:
            tiny += 1
    if not durations:
        return {
            "word_count": len(words),
            "zero_or_negative_count": 0,
            "tiny_span_count": 0,
            "min_word_duration_s": None,
            "max_word_duration_s": None,
        }
    return {
        "word_count": len(words),
        "zero_or_negative_count": zero_or_negative,
        "tiny_span_count": tiny,
        "min_word_duration_s": round(min(durations), 4),
        "max_word_duration_s": round(max(durations), 4),
    }


def chunk_failure_reasons(
    *,
    text: str,
    duration_s: float,
    compact_chars: int,
    prealign_empty: bool,
    nonlexical_text: bool,
    repetition_repair: dict[str, Any],
    alignment_mode: str,
    align_error: str,
    sentinel_lines: list[str],
    aligned_segment_count: int,
    word_stats: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not text.strip() and duration_s >= 1.0:
        reasons.append("empty_text_for_chunk")
    if text.strip() and prealign_empty and not nonlexical_text:
        reasons.append("align_text_empty")
    if text.strip() and prealign_empty and nonlexical_text:
        reasons.append("nonlexical_text")
    if alignment_mode and alignment_mode not in {
        "forced_aligner",
        "empty",
        "nonlexical",
        "align_text_empty",
        "aligner_vad_fallback",
        "even_fallback",
    }:
        reasons.append(f"alignment_mode_{alignment_mode}")
    if align_error:
        reasons.append("alignment_error")
    if alignment_mode in {"aligner_vad_fallback", "even_fallback"}:
        reasons.append("alignment_fallback")
    if sentinel_lines:
        reasons.append("alignment_sentinel")
    if text.strip() and aligned_segment_count <= 0:
        reasons.append("text_without_output_segment")
    if repetition_repair.get("changed") and (
        compact_chars >= 6 or aligned_segment_count <= 0
    ):
        reasons.append("repeat_repair_suggested")
    if compact_chars >= 30 and duration_s > 0 and compact_chars / duration_s > 14.0:
        reasons.append("abnormal_char_density")
    if int(word_stats.get("word_count") or 0) >= 2:
        zero_ratio = int(word_stats.get("zero_or_negative_count") or 0) / max(
            1,
            int(word_stats.get("word_count") or 0),
        )
        if zero_ratio >= 0.55:
            reasons.append("word_timing_zero_heavy")
    return list(dict.fromkeys(reasons))


def diagnostic_text_metrics(
    *,
    analysis_text: str,
    raw_text: str,
    duration_s: float,
) -> dict[str, Any]:
    features = cueqc_text_features(raw_text or analysis_text, analysis_text, duration_s=max(duration_s, 1e-6))
    repeat_profile = features.get("repeat_profile") if isinstance(features.get("repeat_profile"), dict) else {}
    char_count = int(features.get("char_count") or 0)
    has_stable = bool(features.get("has_stable_vocabulary"))
    repeat_run = int(repeat_profile.get("run") or 0)
    repeat_ratio = float(repeat_profile.get("ratio") or 0.0)
    if char_count <= 0:
        density_level = "empty_or_punctuation"
    elif duration_s >= 6.0 and char_count <= 5 and not has_stable:
        density_level = "long_sparse_text"
    elif char_count <= 2 and bool(features.get("kana_only")):
        density_level = "short_vocalization_candidate"
    elif repeat_run >= 4 and repeat_ratio >= 0.4 and not has_stable:
        density_level = "repeated_vocalization_candidate"
    elif bool(features.get("kana_only")) and char_count <= 5 and not has_stable:
        density_level = "short_kana_dialogue_candidate"
    else:
        density_level = "normal_dialogue_candidate"
    repetition_repair = {
        "changed": repeat_run >= 4 and repeat_ratio >= 0.35 and not has_stable,
        "unit": repeat_profile.get("unit", ""),
        "run": repeat_run,
        "ratio": repeat_ratio,
        "suggested_text": "",
    }
    return {
        "text_features": features,
        "max_repeat": repeat_profile,
        "repetition_repair": repetition_repair,
        "text_density": {
            "level": density_level,
            "duration_s": round(duration_s, 3),
            "char_count": char_count,
        },
        "vocalization_repetition": {
            "preserve_candidate": density_level == "repeated_vocalization_candidate",
        },
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return round(numerator / denominator, 6)


def is_failure_candidate(row: dict[str, Any]) -> bool:
    return bool(row.get("failure_reasons")) or str(row.get("alignment_quality") or "") != "forced"


def failure_candidate_bucket(row: dict[str, Any]) -> str:
    reasons = set(str(reason) for reason in (row.get("failure_reasons") or []))
    quality = str(row.get("alignment_quality") or "")
    fallback_type = str(row.get("fallback_type") or "")

    bucket_flags = {
        "nonlexical_text": bool(row.get("nonlexical_text")) or "nonlexical_text" in reasons,
        "align_text_empty": bool(row.get("align_text_empty")) and bool(str(row.get("analysis_text") or "").strip()),
        "empty_text_for_chunk": "empty_text_for_chunk" in reasons,
        "text_without_output_segment": "text_without_output_segment" in reasons,
        "repeat_repair_suggested": "repeat_repair_suggested" in reasons,
        "partial_alignment": quality == "partial",
        "vad_coarse_alignment": quality == "vad_coarse" or fallback_type == "vad_coarse",
        "proportional_alignment": quality == "proportional" or fallback_type == "proportional",
        "unknown_alignment_fallback": fallback_type == "unknown",
        "abnormal_char_density": "abnormal_char_density" in reasons,
        "diagnostic_warning": bool(reasons),
    }
    for bucket in _CANDIDATE_BUCKET_ORDER:
        if bucket_flags.get(bucket):
            return bucket
    return ""


def diagnose_case(
    *,
    aligned_path: Path,
    workflow_root: Path | None,
    case_label: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aligned = read_json(aligned_path)
    if not isinstance(aligned, dict):
        raise ValueError(f"Expected object JSON: {aligned_path}")

    video_stem = aligned_path.name.removesuffix(".aligned_segments.json")
    label = case_label or infer_case_label(aligned_path, workflow_root)
    transcript_path = sibling_path(aligned_path, "transcript.json")
    quality_path = find_quality_json(aligned_path, workflow_root)
    source_audio_path = str(aligned.get("audio_path") or "")
    asr_details = aligned.get("asr_details") if isinstance(aligned.get("asr_details"), dict) else {}
    chunks = list(asr_details.get("transcript_chunks") or [])
    if not chunks and transcript_path is not None:
        transcript = read_json(transcript_path)
        if isinstance(transcript, dict):
            chunks = list(transcript.get("chunks") or [])

    segments = [item for item in aligned.get("segments") or [] if isinstance(item, dict)]
    logs = parse_chunk_logs(aligned.get("asr_log") or [])
    output_counts = segment_counts_by_chunk(segments)
    words = words_by_chunk(segments)

    rows: list[dict[str, Any]] = []
    reason_counts: Counter = Counter()
    alignment_modes: Counter = Counter()
    alignment_quality_counts: Counter = Counter()
    fallback_type_counts: Counter = Counter()
    fallback_subtype_counts: Counter = Counter()
    failure_bucket_counts: Counter = Counter()
    prealign_flag_counts: Counter = Counter()
    text_density_counts: Counter = Counter()
    fallback_chunks = 0
    align_text_empty = 0
    nonempty_chunks = 0

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        try:
            chunk_index = int(chunk.get("index", position))
        except (TypeError, ValueError):
            chunk_index = position
        start = float(chunk.get("start", 0.0) or 0.0)
        end = float(chunk.get("end", start) or start)
        duration = float(chunk.get("duration", max(0.0, end - start)) or 0.0)
        if "alignment_fallback_start_s" in chunk and "alignment_fallback_end_s" in chunk:
            fallback_start_local = float(chunk.get("alignment_fallback_start_s") or 0.0)
            fallback_end_local = float(
                chunk.get("alignment_fallback_end_s") or fallback_start_local
            )
            fallback_end_local = max(fallback_start_local, fallback_end_local)
            fallback_duration = float(
                chunk.get(
                    "alignment_fallback_duration_s",
                    max(0.0, fallback_end_local - fallback_start_local),
                )
                or 0.0
            )
            fallback_abs_start = float(
                chunk.get("alignment_fallback_abs_start_s", start + fallback_start_local)
                or start
            )
            fallback_abs_end = float(
                chunk.get("alignment_fallback_abs_end_s", start + fallback_end_local)
                or fallback_abs_start
            )
            fallback_abs_end = max(fallback_abs_start, fallback_abs_end)
            fallback_source = str(chunk.get("alignment_fallback_source") or "chunk")
        else:
            fallback_abs_start = start
            fallback_abs_end = end
            fallback_duration = duration
            fallback_source = "chunk"
        text = str(chunk.get("text") or "")
        raw_text = str(chunk.get("raw_text") or text)
        analysis_text = text or raw_text
        prealign = prepare_text_for_alignment(analysis_text)
        nonlexical_text = bool(analysis_text.strip() and _NONLEXICAL_TEXT_RE.fullmatch(analysis_text.strip()))
        compact_chars = len(strip_alignment_punctuation(prealign.display_text))
        chars_per_sec = compact_chars / duration if duration > 0 else 0.0
        fallback_cps = compact_chars / fallback_duration if fallback_duration > 0 else 0.0
        chunk_extra_to_fallback_s = max(0.0, duration - fallback_duration)
        fallback_to_chunk_ratio = _safe_ratio(fallback_duration, duration)
        chunk_to_fallback_ratio = _safe_ratio(duration, fallback_duration)
        chunk_log = logs.get(chunk_index, {})
        alignment_mode = str(chunk_log.get("alignment_mode") or "")
        if alignment_mode:
            alignment_modes[alignment_mode] += 1
        sentinel_lines = list(chunk_log.get("sentinel_lines") or [])
        aligned_segment_count = int(output_counts.get(chunk_index, 0))
        stats = word_timing_stats(words.get(chunk_index, []))
        word_failure_reasons = _alignment_failure_reasons(
            words.get(chunk_index, []),
            scene_duration_sec=duration,
        )
        text_metrics = diagnostic_text_metrics(
            analysis_text=analysis_text,
            raw_text=raw_text,
            duration_s=duration,
        )
        repetition_repair = (
            text_metrics.get("repetition_repair")
            if isinstance(text_metrics.get("repetition_repair"), dict)
            else {}
        )
        text_density = (
            text_metrics.get("text_density")
            if isinstance(text_metrics.get("text_density"), dict)
            else {}
        )
        repeat_profile = (
            text_metrics.get("max_repeat")
            if isinstance(text_metrics.get("max_repeat"), dict)
            else {}
        )
        vocalization_repetition = (
            text_metrics.get("vocalization_repetition")
            if isinstance(text_metrics.get("vocalization_repetition"), dict)
            else {}
        )
        text_density_level = str(text_density.get("level") or "")
        if text_density_level:
            text_density_counts[text_density_level] += 1
        repeat_run = int(repeat_profile.get("run") or 0)
        repeat_ratio = float(repeat_profile.get("ratio") or 0.0)
        preserve_nonlexical_repeat = bool(vocalization_repetition.get("preserve_candidate"))
        chunk_repetition_risk = bool(
            chunk_extra_to_fallback_s >= 0.75
            and repeat_run >= 4
            and repeat_ratio >= 0.35
            and (
                fallback_cps >= 4.0
                or text_density_level == "repeated_vocalization_candidate"
            )
        )
        prealign_flag_counts.update(prealign.flags)
        reasons = chunk_failure_reasons(
            text=analysis_text,
            duration_s=duration,
            compact_chars=compact_chars,
            prealign_empty=prealign.empty_after_cleaning,
            nonlexical_text=nonlexical_text,
            repetition_repair=repetition_repair,
            alignment_mode=alignment_mode,
            align_error=str(chunk_log.get("align_error") or ""),
            sentinel_lines=sentinel_lines,
            aligned_segment_count=aligned_segment_count,
            word_stats=stats,
        )
        quality = classify_alignment_quality(
            text=analysis_text,
            duration_s=duration,
            align_text_empty=prealign.empty_after_cleaning,
            nonlexical_text=nonlexical_text,
            alignment_mode=alignment_mode,
            align_error=str(chunk_log.get("align_error") or ""),
            sentinel_lines=sentinel_lines,
            aligned_segment_count=aligned_segment_count,
            word_stats=stats,
            word_failure_reasons=word_failure_reasons,
        )
        reason_counts.update(reasons)
        alignment_quality_counts[str(quality["alignment_quality"])] += 1
        fallback_type_counts[str(quality["fallback_type"])] += 1
        fallback_subtype_counts[str(quality["fallback_subtype"])] += 1
        if prealign.empty_after_cleaning and analysis_text.strip():
            align_text_empty += 1
        if analysis_text.strip():
            nonempty_chunks += 1
        if quality["fallback_type"] != "none":
            fallback_chunks += 1

        row = {
            "case_label": label,
            "video": video_stem,
            "aligned_path": project_rel(aligned_path),
            "source_audio_path": project_rel(source_audio_path),
            "chunk_index": chunk_index,
            "log_chunk_number": chunk_index + 1,
            "position": position,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration_s": round(duration, 3),
            "fallback_window_start": round(fallback_abs_start, 3),
            "fallback_window_end": round(fallback_abs_end, 3),
            "fallback_duration_s": round(fallback_duration, 3),
            "fallback_window_source": fallback_source,
            "chunk_duration_s": round(duration, 3),
            "core_duration_s": round(fallback_duration, 3),
            "chunk_extra_to_fallback_s": round(chunk_extra_to_fallback_s, 3),
            "fallback_to_chunk_duration_ratio": fallback_to_chunk_ratio,
            "chunk_to_fallback_duration_ratio": chunk_to_fallback_ratio,
            "text": text,
            "raw_text": raw_text,
            "analysis_text": analysis_text,
            "review_original_text": "",
            "display_text": prealign.display_text,
            "align_text": prealign.align_text,
            "prealign_flags": prealign.flags,
            "prealign_display_len": prealign.display_len,
            "prealign_align_len": prealign.align_len,
            "prealign_removed_ratio": round(
                max(0, prealign.display_len - prealign.align_len)
                / max(1, prealign.display_len),
                6,
            ),
            "align_text_empty": prealign.empty_after_cleaning,
            "nonlexical_text": nonlexical_text,
            "compact_chars": compact_chars,
            "chars_per_sec": round(chars_per_sec, 3),
            "chunk_cps": round(chars_per_sec, 3),
            "core_cps": round(fallback_cps, 3),
            "fallback_cps": round(fallback_cps, 3),
            "chunk_text_leak_risk": bool(
                compact_chars >= 12
                and fallback_cps > 4.0
                and chunk_extra_to_fallback_s >= 0.75
                and (
                    fallback_to_chunk_ratio < 0.72
                    or fallback_cps >= max(7.0, chars_per_sec * 1.35)
                )
            ),
            "chunk_repetition_risk": chunk_repetition_risk,
            "repeat_profile": repeat_profile,
            "vocalization_repetition": vocalization_repetition,
            "preserve_nonlexical_repetition": preserve_nonlexical_repeat,
            "text_density": text_density,
            "text_density_level": text_density_level,
            "repetition_repair": repetition_repair,
            "signal_quality_verdict": "",
            "signal_quality_reason": "",
            "avg_logprob": chunk.get("avg_logprob"),
            "compression_ratio": chunk.get("compression_ratio"),
            "no_speech_prob": chunk.get("no_speech_prob"),
            "repetition_suggested_text": str(
                repetition_repair.get("suggested_text") or ""
            )
            if repetition_repair.get("changed")
            else "",
            "alignment_mode": alignment_mode,
            "alignment_quality": quality["alignment_quality"],
            "fallback_type": quality["fallback_type"],
            "fallback_subtype": quality["fallback_subtype"],
            "alignment_quality_reasons": quality["alignment_quality_reasons"],
            "alignment_word_count": chunk_log.get("alignment_word_count"),
            "align_error": chunk_log.get("align_error") or "",
            "sentinel_lines": sentinel_lines,
            "aligned_segment_count": aligned_segment_count,
            "word_timing": stats,
            "word_timing_failure_reasons": word_failure_reasons,
            "failure_reasons": reasons,
        }
        row["failure_candidate"] = is_failure_candidate(row)
        row["failure_bucket"] = failure_candidate_bucket(row) if row["failure_candidate"] else ""
        if row["failure_bucket"]:
            failure_bucket_counts[str(row["failure_bucket"])] += 1
        rows.append(row)

    quality = read_json(quality_path) if quality_path is not None else {}
    quality = quality if isinstance(quality, dict) else {}
    summary = {
        "case_label": label,
        "video": video_stem,
        "aligned_path": project_rel(aligned_path),
        "transcript_path": project_rel(transcript_path),
        "quality_path": project_rel(quality_path),
        "chunk_count": len(chunks),
        "output_segment_count": len(segments),
        "nonempty_chunk_count": nonempty_chunks,
        "align_text_empty_count": align_text_empty,
        "fallback_chunk_count": fallback_chunks,
        "fallback_chunk_ratio": round(fallback_chunks / max(1, len(chunks)), 6),
        "reason_counts": dict(reason_counts.most_common()),
        "alignment_mode_counts": dict(alignment_modes.most_common()),
        "alignment_quality_counts": dict(alignment_quality_counts.most_common()),
        "fallback_type_counts": dict(fallback_type_counts.most_common()),
        "fallback_subtype_counts": dict(fallback_subtype_counts.most_common()),
        "failure_bucket_counts": dict(failure_bucket_counts.most_common()),
        "prealign_flag_counts": dict(prealign_flag_counts.most_common()),
        "text_density_counts": dict(text_density_counts.most_common()),
        "repeat_repair_suggested_count": sum(
            1
            for row in rows
            if "repeat_repair_suggested" in set(row.get("failure_reasons") or [])
        ),
        "quality_alignment_fallback_ratio": quality.get("alignment_fallback_ratio"),
        "asr_details_fallback_count": asr_details.get("fallback_count"),
    }
    return rows, summary


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ASR / Alignment Diagnostics",
        "",
        "## Summary",
        "",
        f"- cases: {summary['case_count']}",
        f"- chunks: {summary['chunk_count']}",
        f"- output segments: {summary['output_segment_count']}",
        f"- fallback chunks: {summary['fallback_chunk_count']} ({summary['fallback_chunk_ratio']:.1%})",
        f"- align-text-empty chunks: {summary['align_text_empty_count']}",
        f"- failure candidates: {summary['failure_candidate_count']}",
        f"- chunk text leak risk: {summary.get('chunk_text_leak_risk_count', 0)}",
        f"- chunk repetition risk: {summary.get('chunk_repetition_risk_count', 0)}",
        "",
        "## Alignment Quality",
        "",
    ]
    if summary.get("alignment_quality_counts"):
        for quality, count in summary["alignment_quality_counts"].items():
            lines.append(f"- `{quality}`: {count}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Fallback Type",
            "",
        ]
    )
    if summary.get("fallback_type_counts"):
        for fallback_type, count in summary["fallback_type_counts"].items():
            lines.append(f"- `{fallback_type}`: {count}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Fallback Subtype",
            "",
        ]
    )
    if summary.get("fallback_subtype_counts"):
        for subtype, count in summary["fallback_subtype_counts"].items():
            lines.append(f"- `{subtype}`: {count}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Top Reasons",
            "",
        ]
    )
    if summary["reason_counts"]:
        for reason, count in list(summary["reason_counts"].items())[:20]:
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None")
    lines.extend(["", "## Failure Buckets", ""])
    if summary.get("failure_bucket_counts"):
        for bucket, count in summary["failure_bucket_counts"].items():
            lines.append(f"- `{bucket}`: {count}")
    else:
        lines.append("- None")
    lines.extend(["", "## Per Video", ""])
    for item in summary["cases"]:
        lines.append(
            "- {label}/{video}: chunks={chunks}, fallback={fallback} ({ratio:.1%}), "
            "align_empty={align_empty}, segments={segments}, "
            "quality={quality}".format(
                label=item.get("case_label") or "case",
                video=item["video"],
                chunks=item["chunk_count"],
                fallback=item["fallback_chunk_count"],
                ratio=item["fallback_chunk_ratio"],
                align_empty=item["align_text_empty_count"],
                segments=item["output_segment_count"],
                quality=item.get("alignment_quality_counts", {}),
            )
        )
    lines.append("")
    return "\n".join(lines)


def summarize(rows: list[dict[str, Any]], case_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter = Counter()
    mode_counts: Counter = Counter()
    quality_counts: Counter = Counter()
    fallback_type_counts: Counter = Counter()
    fallback_subtype_counts: Counter = Counter()
    failure_bucket_counts: Counter = Counter()
    prealign_flag_counts: Counter = Counter()
    text_density_counts: Counter = Counter()
    repeat_repair_suggested_count = 0
    chunk_repetition_risk_count = 0
    for row in rows:
        reason_counts.update(row.get("failure_reasons") or [])
        mode = str(row.get("alignment_mode") or "")
        if mode:
            mode_counts[mode] += 1
        quality = str(row.get("alignment_quality") or "")
        if quality:
            quality_counts[quality] += 1
        fallback_type = str(row.get("fallback_type") or "")
        if fallback_type:
            fallback_type_counts[fallback_type] += 1
        fallback_subtype = str(row.get("fallback_subtype") or "")
        if fallback_subtype:
            fallback_subtype_counts[fallback_subtype] += 1
        failure_bucket = str(row.get("failure_bucket") or "")
        if failure_bucket:
            failure_bucket_counts[failure_bucket] += 1
        prealign_flag_counts.update(row.get("prealign_flags") or [])
        text_density_level = str(row.get("text_density_level") or "")
        if text_density_level and text_density_level != "unknown":
            text_density_counts[text_density_level] += 1
        if "repeat_repair_suggested" in set(row.get("failure_reasons") or []):
            repeat_repair_suggested_count += 1
        if bool(row.get("chunk_repetition_risk")):
            chunk_repetition_risk_count += 1

    chunk_count = len(rows)
    fallback_count = sum(1 for row in rows if row.get("fallback_type") not in {"", "none", None})
    failure_candidate_count = sum(1 for row in rows if row.get("failure_candidate"))
    return {
        "case_count": len(case_summaries),
        "chunk_count": chunk_count,
        "output_segment_count": sum(int(item.get("output_segment_count") or 0) for item in case_summaries),
        "nonempty_chunk_count": sum(1 for row in rows if str(row.get("analysis_text") or "").strip()),
        "align_text_empty_count": sum(1 for row in rows if row.get("align_text_empty") and str(row.get("analysis_text") or "").strip()),
        "fallback_chunk_count": fallback_count,
        "fallback_chunk_ratio": fallback_count / max(1, chunk_count),
        "failure_candidate_count": failure_candidate_count,
        "reason_counts": dict(reason_counts.most_common()),
        "alignment_mode_counts": dict(mode_counts.most_common()),
        "alignment_quality_counts": dict(quality_counts.most_common()),
        "fallback_type_counts": dict(fallback_type_counts.most_common()),
        "fallback_subtype_counts": dict(fallback_subtype_counts.most_common()),
        "failure_bucket_counts": dict(failure_bucket_counts.most_common()),
        "prealign_flag_counts": dict(prealign_flag_counts.most_common()),
        "text_density_counts": dict(text_density_counts.most_common()),
        "repeat_repair_suggested_count": repeat_repair_suggested_count,
        "chunk_text_leak_risk_count": sum(1 for row in rows if row.get("chunk_text_leak_risk")),
        "chunk_repetition_risk_count": chunk_repetition_risk_count,
        "cases": case_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose offline ASR text, pre-alignment cleaning, and forced-alignment fallback artifacts.",
    )
    parser.add_argument(
        "--workflow-root",
        default="",
        help="Full workflow output root containing archived/*/*.aligned_segments.json.",
    )
    parser.add_argument(
        "--aligned-json",
        action="append",
        default=[],
        help="Explicit aligned_segments JSON path; can be passed multiple times.",
    )
    parser.add_argument(
        "--case-label",
        action="append",
        default=[],
        help="Optional label for each --aligned-json path, in the same order.",
    )
    parser.add_argument(
        "--output-dir",
        default="agents/temp/speech-boundary-ja/asr-alignment-diagnostics",
        help="Directory for summary.json, diagnostics.jsonl, failure_candidates.jsonl, summary.md.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workflow_root = project_path(args.workflow_root) if args.workflow_root else None
    explicit = [project_path(path) for path in args.aligned_json]
    aligned_paths = discover_aligned_jsons(workflow_root, explicit)
    if not aligned_paths:
        raise SystemExit("No aligned_segments JSON files found.")
    if args.case_label and len(args.case_label) not in {1, len(aligned_paths)}:
        raise SystemExit(
            "--case-label count must be 1 or match discovered aligned JSON count."
        )
    case_labels = (
        args.case_label * len(aligned_paths)
        if len(args.case_label) == 1 and len(aligned_paths) > 1
        else args.case_label
    )

    all_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for idx, aligned_path in enumerate(aligned_paths):
        rows, case_summary = diagnose_case(
            aligned_path=aligned_path,
            workflow_root=workflow_root,
            case_label=case_labels[idx] if case_labels else None,
        )
        all_rows.extend(rows)
        case_summaries.append(case_summary)

    summary = summarize(all_rows, case_summaries)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "diagnostics.jsonl"
    candidates_path = output_dir / "failure_candidates.jsonl"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"

    candidates = [row for row in all_rows if row.get("failure_candidate")]
    append_jsonl(diagnostics_path, all_rows)
    append_jsonl(candidates_path, candidates)
    write_json(summary_path, summary)
    markdown_path.write_text(build_markdown(summary), encoding="utf-8")

    print(
        "diagnosed cases={cases} chunks={chunks} candidates={candidates} output={output}".format(
            cases=summary["case_count"],
            chunks=summary["chunk_count"],
            candidates=summary["failure_candidate_count"],
            output=project_rel(output_dir),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
