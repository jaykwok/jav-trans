#!/usr/bin/env python3
"""Select Gemini outside complements on train-only real workflow sources.

The output is diagnostic input for the 1.7B ASR probe.  It is never accepted
as training truth directly: Gemini inside and unsure spans are blocked, and
every remaining outside span still requires empty, error-free ASR evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


SOURCE_SCHEMA = "candidate_island_scorer_v11_train_teacher_source_v1"
TEACHER_SCHEMA = "candidate_island_scorer_v11_omni_preaudit_v2"
SCHEMA = "candidate_island_scorer_v11_real_outside_asr_selection_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_real_outside_asr_selection_summary_v1"
FRAME_HOP_S = 0.02


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"{name} has missing or duplicate source_id: {source_id!r}")
        result[source_id] = row
    return result


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [
        (int(start), int(end))
        for start, end in zip(edges[0::2], edges[1::2], strict=True)
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    sources_path = Path(args.train_teacher_sources).resolve()
    teacher_path = Path(args.gemini_preaudit).resolve()
    for path in (sources_path, teacher_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sources = _index(_rows(sources_path), name="train teacher sources")
    teachers = _index(_rows(teacher_path), name="Gemini preaudit")
    if set(sources) != set(teachers):
        missing = sorted(set(sources) - set(teachers))
        foreign = sorted(set(teachers) - set(sources))
        raise ValueError(
            f"Gemini preaudit source scope mismatch: missing={missing[:3]}, foreign={foreign[:3]}"
        )
    output_rows: list[dict[str, Any]] = []
    outside_frames = blocked_inside_frames = blocked_unsure_frames = 0
    span_count = 0
    full_inside_source_count = 0
    for source_id in sorted(sources):
        source = sources[source_id]
        teacher = teachers[source_id]
        if source.get("schema") != SOURCE_SCHEMA or teacher.get("schema") != TEACHER_SCHEMA:
            raise ValueError(f"wrong real-source/Gemini schema: {source_id}")
        if source.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"wrong central Boundary contract: {source_id}")
        if source.get("partition") != "train" or teacher.get("partition") != "train":
            raise ValueError(f"real outside selection is not train-only: {source_id}")
        if str(source.get("audio_sha256") or "") != str(teacher.get("audio_sha256") or ""):
            raise ValueError(f"Gemini preaudit audio SHA mismatch: {source_id}")
        frame_count = int(source.get("frame_count") or 0)
        if frame_count <= 0 or int(teacher.get("frame_count") or 0) != frame_count:
            raise ValueError(f"Gemini preaudit frame geometry mismatch: {source_id}")
        blocked = np.zeros(frame_count, dtype=bool)
        inside_mask = np.zeros(frame_count, dtype=bool)
        unsure_mask = np.zeros(frame_count, dtype=bool)
        for raw, target in (
            *((span, inside_mask) for span in teacher.get("islands") or ()),
            *((span, unsure_mask) for span in teacher.get("unsure_spans") or ()),
        ):
            start = max(0, min(frame_count, int(raw["start_frame"])))
            end = max(0, min(frame_count, int(raw["end_frame"])))
            if start >= end:
                raise ValueError(f"Gemini preaudit contains empty span: {source_id}")
            target[start:end] = True
        blocked |= inside_mask | unsure_mask
        outside = ~blocked
        spans = [
            {
                "label": "asr_probe_candidate",
                "start_frame": start,
                "end_frame": end,
                "start_s": start * FRAME_HOP_S,
                "end_s": end * FRAME_HOP_S,
                "selection_role": "gemini_outside_complement_pending_asr",
            }
            for start, end in _runs(outside)
        ]
        blocked_inside_frames += int(inside_mask.sum())
        blocked_unsure_frames += int(unsure_mask.sum())
        outside_frames += int(outside.sum())
        span_count += len(spans)
        if not spans:
            full_inside_source_count += 1
            continue
        output_rows.append(
            {
                "schema": SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": str(source.get("video_id") or ""),
                "partition": "train",
                "audio": str(source["audio"]),
                "audio_sha256": str(source["audio_sha256"]),
                "duration_s": float(source["duration_s"]),
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "prediction_spans": spans,
                "gemini_model": str(teacher.get("model") or ""),
                "gemini_prompt_version": str(teacher.get("prompt_version") or ""),
                "diagnostic_only": True,
                "training_manifest_allowed": False,
            }
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = output_dir / "real_outside_asr_selection.jsonl"
    _write_jsonl(selection, output_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "train_teacher_sources": str(sources_path),
        "train_teacher_sources_sha256": _sha256(sources_path),
        "gemini_preaudit": str(teacher_path),
        "gemini_preaudit_sha256": _sha256(teacher_path),
        "source_count": len(sources),
        "selected_source_count": len(output_rows),
        "full_inside_source_count": full_inside_source_count,
        "outside_span_count": span_count,
        "outside_frame_count": outside_frames,
        "outside_duration_s": outside_frames * FRAME_HOP_S,
        "blocked_inside_frame_count": blocked_inside_frames,
        "blocked_unsure_frame_count": blocked_unsure_frames,
        "real_outside_asr_selection": str(selection),
        "real_outside_asr_selection_sha256": _sha256(selection),
        "gemini_output_used_as_training_truth": False,
        "asr_review_required": True,
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-teacher-sources", required=True)
    parser.add_argument("--gemini-preaudit", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
