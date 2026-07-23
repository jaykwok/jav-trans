#!/usr/bin/env python3
"""Build a label-free Scorer v11 source manifest for raw feature pre-extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    CANDIDATE_ISLAND_SCORER_V11_RAW_PREEXTRACT_SOURCE_SCHEMA,
)


AUDIT_SUMMARY_SCHEMA = "candidate_island_scorer_v11_train_teacher_review_summary_v1"
AUDIT_ITEM_SCHEMA = "candidate_island_scorer_v11_train_teacher_review_item_v1"
TEACHER_SOURCE_SCHEMA = "candidate_island_scorer_v11_train_teacher_source_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_raw_preextract_manifest_summary_v1"
FRAME_HOP_S = 0.02


def _resolve(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index(rows: Iterable[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in indexed:
            raise ValueError(f"{name} requires unique non-empty source_id: {source_id!r}")
        indexed[source_id] = row
    return indexed


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
    )


def prepare_raw_preextract_manifest(
    *, audit_summary_path: Path, output_dir: Path
) -> dict[str, Any]:
    audit_summary_path = audit_summary_path.resolve()
    if not audit_summary_path.is_file():
        raise FileNotFoundError(audit_summary_path)
    summary = json.loads(audit_summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != AUDIT_SUMMARY_SCHEMA:
        raise ValueError("wrong Scorer v11 train review summary schema")
    if summary.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("wrong central Boundary contract")
    if summary.get("training_manifest_allowed") is not False:
        raise ValueError("raw pre-extraction requires a non-training audit summary")
    if summary.get("human_full_source_confirmation_required") is not True:
        raise ValueError("raw pre-extraction requires the human full-source gate")
    if str(summary.get("manual_gate_status") or "") != "pending":
        raise ValueError("raw pre-extraction is only valid while the manual gate is pending")

    audit_manifest_path = _resolve(str(summary.get("audit_manifest") or ""))
    source_manifest_path = _resolve(str(summary.get("source_manifest") or ""))
    for path, expected_sha, name in (
        (
            audit_manifest_path,
            str(summary.get("audit_manifest_sha256") or ""),
            "audit manifest",
        ),
        (
            source_manifest_path,
            str(summary.get("source_manifest_sha256") or ""),
            "teacher source manifest",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if not expected_sha or _sha256(path) != expected_sha:
            raise ValueError(f"{name} SHA256 mismatch")

    selected_ids = [str(value) for value in summary.get("selected_source_ids") or ()]
    if (
        not selected_ids
        or len(selected_ids) != len(set(selected_ids))
        or int(summary.get("source_count") or 0) != len(selected_ids)
    ):
        raise ValueError("invalid frozen selected source identities")
    audit_rows = _index(_read_jsonl(audit_manifest_path), name="audit manifest")
    teacher_rows = _index(_read_jsonl(source_manifest_path), name="teacher manifest")
    if set(audit_rows) != set(selected_ids):
        raise ValueError("audit manifest does not exactly match selected source identities")
    if not set(selected_ids).issubset(teacher_rows):
        raise ValueError("teacher manifest is missing selected source identities")

    output_rows: list[dict[str, Any]] = []
    video_ids: set[str] = set()
    total_frames = 0
    total_audio_bytes = 0
    for source_id in selected_ids:
        audit = audit_rows[source_id]
        teacher = teacher_rows[source_id]
        if audit.get("schema") != AUDIT_ITEM_SCHEMA:
            raise ValueError(f"wrong audit item schema: {source_id}")
        if teacher.get("schema") != TEACHER_SOURCE_SCHEMA:
            raise ValueError(f"wrong teacher source schema: {source_id}")
        for row_name, row in (("audit", audit), ("teacher", teacher)):
            if row.get("boundary_serialization_contract_id") != (
                ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            ):
                raise ValueError(f"{row_name} Boundary contract mismatch: {source_id}")
            if str(row.get("partition") or "") != "train":
                raise ValueError(f"{row_name} partition is not train: {source_id}")

        video_id = str(teacher.get("video_id") or "")
        if not video_id or video_id != str(audit.get("video_id") or ""):
            raise ValueError(f"video identity mismatch: {source_id}")
        if video_id in video_ids:
            raise ValueError(f"multiple selected sources share a video: {video_id}")
        video_ids.add(video_id)

        frame_count = int(teacher.get("frame_count") or 0)
        frame_hop_s = float(teacher.get("frame_hop_s") or 0.0)
        duration_s = float(teacher.get("duration_s") or 0.0)
        sample_rate = int(teacher.get("sample_rate") or 0)
        sample_count = int(teacher.get("sample_count") or 0)
        if frame_count <= 0 or not math.isclose(frame_hop_s, FRAME_HOP_S):
            raise ValueError(f"invalid teacher frame geometry: {source_id}")
        if sample_rate != 16000 or sample_count <= 0 or duration_s <= 0.0:
            raise ValueError(f"invalid teacher audio geometry: {source_id}")
        if (
            int(audit.get("frame_count") or 0) != frame_count
            or not math.isclose(float(audit.get("frame_hop_s") or 0.0), frame_hop_s)
            or not math.isclose(float(audit.get("duration_s") or 0.0), duration_s)
        ):
            raise ValueError(f"audit/teacher geometry mismatch: {source_id}")

        source_audio = _resolve(str(teacher.get("audio") or ""))
        audit_audio = _resolve(
            str(audit.get("audio") or ""), base=audit_manifest_path.parent
        )
        for audio_path, declared_sha, name in (
            (source_audio, str(teacher.get("audio_sha256") or ""), "teacher audio"),
            (audit_audio, str(audit.get("audio_sha256") or ""), "audit audio"),
        ):
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            if not declared_sha or _sha256(audio_path) != declared_sha:
                raise ValueError(f"{name} SHA256 mismatch: {source_id}")
        if teacher.get("audio_sha256") != audit.get("audio_sha256"):
            raise ValueError(f"audit/teacher audio identity mismatch: {source_id}")

        output_rows.append(
            {
                "schema": CANDIDATE_ISLAND_SCORER_V11_RAW_PREEXTRACT_SOURCE_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "input_distribution": "real_workflow_source_window",
                "audio": _display(source_audio),
                "audio_sha256": str(teacher["audio_sha256"]),
                "audio_sample_count": sample_count,
                "sample_rate": sample_rate,
                "frame_count": frame_count,
                "frame_hop_s": frame_hop_s,
                "duration_s": duration_s,
                "labels_available": False,
                "human_manual_verdicts_required": True,
                "human_gate_status": "pending",
                "feature_extraction_allowed": True,
                "training_manifest_allowed": False,
                "teacher_output_used_as_truth": False,
                "audit_item_sha256": _json_sha256(audit),
                "teacher_source_row_sha256": _json_sha256(teacher),
            }
        )
        total_frames += frame_count
        total_audio_bytes += source_audio.stat().st_size

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "raw_preextract_sources.jsonl"
    summary_path = output_dir / "summary.json"
    _write_jsonl(manifest_path, output_rows)
    result = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "audit_summary": _display(audit_summary_path),
        "audit_summary_sha256": _sha256(audit_summary_path),
        "audit_manifest": _display(audit_manifest_path),
        "audit_manifest_sha256": _sha256(audit_manifest_path),
        "teacher_source_manifest": _display(source_manifest_path),
        "teacher_source_manifest_sha256": _sha256(source_manifest_path),
        "raw_preextract_sources": _display(manifest_path),
        "raw_preextract_sources_sha256": _sha256(manifest_path),
        "source_count": len(output_rows),
        "video_count": len(video_ids),
        "frame_count": total_frames,
        "audio_bytes": total_audio_bytes,
        "source_schema": CANDIDATE_ISLAND_SCORER_V11_RAW_PREEXTRACT_SOURCE_SCHEMA,
        "feature_extraction_allowed": True,
        "labels_available": False,
        "manual_gate_status": "pending",
        "training_manifest_allowed": False,
    }
    _atomic_text(
        summary_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return prepare_raw_preextract_manifest(
        audit_summary_path=Path(args.audit_summary), output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
