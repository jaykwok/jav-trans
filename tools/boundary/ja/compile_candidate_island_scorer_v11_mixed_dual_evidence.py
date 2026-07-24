#!/usr/bin/env python3
"""Compile a frozen mixed-source Gemini dual-evidence train set.

Unlike the historical outside-mask compiler, this adapter accepts one frozen
source manifest and its independent Protect/Remove preaudit directly.  It is
still evidence-only: conflicts and unmarked frames remain ``unsure`` (-100),
and no ASR text or inherited labels are introduced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import wave
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_ID = "boundary_acoustic_binary_v12"
SOURCE_SCHEMA = "candidate_island_scorer_v11_train_teacher_source_v1"
LABEL_SCHEMA = "candidate_island_scorer_v11_dual_evidence_preaudit_v1"
OUTPUT_SCHEMA = "candidate_island_scorer_v11_real_train_dual_evidence_source_v2"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_mixed_dual_evidence_compile_summary_v1"
FRAME_HOP_S = 0.02
SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
EXPECTED_MODEL = "google/gemini-3.6-flash"
EXPECTED_PROFILE = "gemini"
EXPECTED_REASONING = "medium"
EXPECTED_PROMPT_PROFILE = "dual-evidence-protect-remove-v1"
EXPECTED_EXECUTION_CONTRACT = "gemini_openrouter_reasoning_require_parameters_v1"
EXPECTED_TIMESTAMP_CONTRACT = "omni_audio_timestamp_mmss_mmm_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve(value: str | Path, *, owner: Path | None = None) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    candidates = []
    if owner is not None:
        candidates.append(owner.parent / raw)
    candidates.append(PROJECT_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _index(rows: Sequence[Mapping[str, Any]], *, key: str, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in result:
            raise ValueError(f"{name} requires unique non-empty {key}: {value!r}")
        result[value] = dict(row)
    return result


def _wav_geometry(path: Path) -> tuple[int, float, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_count = int(handle.getnframes())
    if sample_rate != SAMPLE_RATE or channels != 1 or sample_count <= 0:
        raise ValueError(f"invalid source WAV: {path}")
    return sample_count, sample_count / SAMPLE_RATE, sample_count // FRAME_SAMPLES


def _span_frames(spans: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(span["end_frame"]) - int(span["start_frame"]) for span in spans)


def _canonical_spans(evidence: Mapping[str, Any], *, source_id: str, frame_count: int) -> list[dict[str, Any]]:
    spans = [
        *list(evidence.get("islands") or ()),
        *list(evidence.get("safe_outside_spans") or ()),
        *list(evidence.get("unsure_spans") or ()),
    ]
    normalized = sorted(
        (
            {
                "label": str(span.get("label") or ""),
                "start_frame": int(span.get("start_frame") or 0),
                "end_frame": int(span.get("end_frame") or 0),
                "start_s": round(int(span.get("start_frame") or 0) * FRAME_HOP_S, 6),
                "end_s": round(int(span.get("end_frame") or 0) * FRAME_HOP_S, 6),
                **({"reason": str(span.get("reason") or "")} if span.get("reason") else {}),
            }
            for span in spans
        ),
        key=lambda span: (span["start_frame"], span["end_frame"], span["label"]),
    )
    cursor = 0
    for span in normalized:
        if span["label"] not in {"inside_candidate", "outside_candidate", "unsure"}:
            raise ValueError(f"unsupported dual-evidence label: {source_id}")
        if span["start_frame"] != cursor or span["end_frame"] <= cursor or span["end_frame"] > frame_count:
            raise ValueError(f"dual-evidence spans are not contiguous: {source_id}")
        cursor = span["end_frame"]
    if cursor != frame_count:
        raise ValueError(f"dual-evidence spans do not cover source tail: {source_id}")
    return normalized


def compile_mixed_dual_evidence(
    *,
    manifest: Path,
    preaudit: Path,
    calibration_summary: Path,
    calibration_teacher_summary: Path,
    calibration_ab_summary: Path,
    calibration_ab_verdicts: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    preaudit = preaudit.resolve()
    calibration_summary = calibration_summary.resolve()
    calibration_teacher_summary = calibration_teacher_summary.resolve()
    calibration_ab_summary = calibration_ab_summary.resolve()
    calibration_ab_verdicts = calibration_ab_verdicts.resolve()
    for path in (
        manifest,
        preaudit,
        calibration_summary,
        calibration_teacher_summary,
        calibration_ab_summary,
        calibration_ab_verdicts,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    sources = _index(_rows(manifest), key="source_id", name="mixed source manifest")
    labels = _index(_rows(preaudit), key="source_id", name="dual-evidence preaudit")
    if set(sources) != set(labels):
        raise ValueError("mixed source manifest and dual-evidence labels must have identical source IDs")
    seen_videos: set[str] = set()
    compiled: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    conflict_frames = 0
    for source_id in sorted(sources):
        source = sources[source_id]
        evidence = labels[source_id]
        if source.get("schema") != SOURCE_SCHEMA or source.get("partition") != "train":
            raise ValueError(f"invalid mixed source: {source_id}")
        if source.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError(f"wrong central contract: {source_id}")
        if evidence.get("schema") != LABEL_SCHEMA:
            raise ValueError(f"wrong dual-evidence schema: {source_id}")
        if evidence.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError(f"dual-evidence contract mismatch: {source_id}")
        if evidence.get("teacher_failed_closed") is True or evidence.get("training_manifest_allowed") is not False:
            raise ValueError(f"failed-closed or already-promoted evidence: {source_id}")
        for field, expected in {
            "model": EXPECTED_MODEL,
            "provider_profile": EXPECTED_PROFILE,
            "reasoning_effort": EXPECTED_REASONING,
            "prompt_profile": EXPECTED_PROMPT_PROFILE,
            "teacher_execution_contract_id": EXPECTED_EXECUTION_CONTRACT,
            "teacher_timestamp_contract_id": EXPECTED_TIMESTAMP_CONTRACT,
            "max_tokens": 8192,
            "exclude_reasoning": False,
            "require_provider_parameters": True,
        }.items():
            if evidence.get(field) != expected:
                raise ValueError(f"dual-evidence source has wrong {field}: {source_id}")
        if not bool((evidence.get("protect_reasoning") or {}).get("reasoning_evidence_present")):
            raise ValueError(f"Protect reasoning evidence is missing: {source_id}")
        if not bool((evidence.get("remove_reasoning") or {}).get("reasoning_evidence_present")):
            raise ValueError(f"Remove reasoning evidence is missing: {source_id}")
        video_id = str(source.get("video_id") or "")
        if not video_id or video_id in seen_videos:
            raise ValueError(f"mixed manifest must contain one source per video: {source_id}")
        seen_videos.add(video_id)
        audio = _resolve(str(source.get("audio") or ""), owner=manifest)
        if not audio.is_file():
            raise FileNotFoundError(audio)
        audio_sha = str(source.get("audio_sha256") or "")
        if len(audio_sha) != 64 or _sha256(audio) != audio_sha:
            raise ValueError(f"mixed source audio SHA mismatch: {source_id}")
        sample_count, duration_s, audio_frame_count = _wav_geometry(audio)
        frame_count = int(source.get("frame_count") or 0)
        if frame_count != audio_frame_count or float(source.get("frame_hop_s") or 0.0) != FRAME_HOP_S:
            raise ValueError(f"mixed source frame geometry mismatch: {source_id}")
        spans = _canonical_spans(evidence, source_id=source_id, frame_count=frame_count)
        counts = Counter(str(span["label"]) for span in spans)
        for span in spans:
            totals[str(span["label"])] += int(span["end_frame"]) - int(span["start_frame"])
        conflict_frames += _span_frames(evidence.get("conflict_spans") or ())
        compiled.append(
            {
                "schema": OUTPUT_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "teacher_timestamp_contract_id": EXPECTED_TIMESTAMP_CONTRACT,
                "teacher_execution_contract_id": EXPECTED_EXECUTION_CONTRACT,
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "input_distribution": "real_workflow_source_window_mixed_dual_evidence_v2",
                "source_kind": "real_train_full_source_calibrated_dual_evidence",
                "synthetic_composite": False,
                "audio": _display(audio),
                "audio_sha256": audio_sha,
                "sample_rate": SAMPLE_RATE,
                "sample_count": sample_count,
                "duration_s": duration_s,
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "core_ids": [f"real-train-dual-evidence-source::{source_id}"],
                "canonical_spans": spans,
                "annotation_provenance": "calibrated_gemini_independent_dual_evidence_v1",
                "teacher_output_used_as_truth": True,
                "teacher_output_used_as_calibrated_evidence": True,
                "teacher_evidence_used_as_training_supervision": True,
                "human_full_source_confirmed": False,
                "calibration_gate_passed": True,
                "calibration_selected_arm": "Medium",
                "calibration_summary": _display(calibration_summary),
                "calibration_summary_sha256": _sha256(calibration_summary),
                "calibration_teacher_summary": _display(calibration_teacher_summary),
                "calibration_teacher_summary_sha256": _sha256(calibration_teacher_summary),
                "calibration_ab_summary": _display(calibration_ab_summary),
                "calibration_ab_summary_sha256": _sha256(calibration_ab_summary),
                "calibration_ab_verdicts": _display(calibration_ab_verdicts),
                "calibration_ab_verdicts_sha256": _sha256(calibration_ab_verdicts),
                "dual_evidence_summary": _display(preaudit.parent / "summary.json"),
                "dual_evidence_summary_sha256": _sha256(preaudit.parent / "summary.json"),
                "dual_evidence_preaudit": _display(preaudit),
                "dual_evidence_preaudit_sha256": _sha256(preaudit),
                "source_manifest": _display(manifest),
                "source_manifest_sha256": _sha256(manifest),
                "unsure_training_label": -100,
                "unselected_source_label_inheritance": False,
                "training_manifest_allowed": True,
                "conflict_frame_count": int(sum(int(span["end_frame"]) - int(span["start_frame"]) for span in evidence.get("conflict_spans") or ())),
                "source_label_counts": dict(counts),
            }
        )
    if totals["inside_candidate"] <= 0 or totals["outside_candidate"] <= 0:
        raise ValueError("mixed dual-evidence compilation requires both binary classes")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "real_train_dual_evidence_sources.jsonl"
    _write_jsonl(output_path, compiled)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "manifest": _display(manifest),
        "manifest_sha256": _sha256(manifest),
        "preaudit": _display(preaudit),
        "preaudit_sha256": _sha256(preaudit),
        "real_train_dual_evidence_sources": _display(output_path),
        "real_train_dual_evidence_sources_sha256": _sha256(output_path),
        "source_count": len(compiled),
        "video_count": len(seen_videos),
        "canonical_frame_counts": dict(sorted(totals.items())),
        "conflict_frames": conflict_frames,
        "source_level_mixed_count": sum(
            bool(_span_frames(row["canonical_spans"]) > 0) and any(span["label"] == "outside_candidate" for span in row["canonical_spans"]) and any(span["label"] == "inside_candidate" for span in row["canonical_spans"])
            for row in compiled
        ),
        "teacher_output_used_as_truth": True,
        "teacher_output_used_as_calibrated_evidence": True,
        "teacher_evidence_used_as_training_supervision": True,
        "human_full_source_confirmed": False,
        "calibration_gate_passed": True,
        "calibration_selected_arm": "Medium",
        "unsure_training_label": -100,
        "training_manifest_allowed": True,
        "unselected_source_label_inheritance": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preaudit", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--calibration-teacher-summary", required=True)
    parser.add_argument("--calibration-ab-summary", required=True)
    parser.add_argument("--calibration-ab-verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    compile_mixed_dual_evidence(
        manifest=Path(args.manifest),
        preaudit=Path(args.preaudit),
        calibration_summary=Path(args.calibration_summary),
        calibration_teacher_summary=Path(args.calibration_teacher_summary),
        calibration_ab_summary=Path(args.calibration_ab_summary),
        calibration_ab_verdicts=Path(args.calibration_ab_verdicts),
        output_dir=Path(args.output_dir),
    )
