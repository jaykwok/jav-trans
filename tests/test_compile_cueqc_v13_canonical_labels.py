from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from tools.asr.cueqc.compile_cueqc_v13_canonical_labels import compile_labels
from tools.asr.cueqc.label_runtime_v12_cueqc_v13_with_omni import (
    PROMPT_VERSION as TEACHER_PROMPT_VERSION,
    SCHEMA as TEACHER_SCHEMA,
)
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        "utf-8",
    )


def _write_runtime(path: Path, rows: list[dict]) -> None:
    enriched = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "schema": "runtime_v12_provisional_subisland_v2",
                "inner_execution_status": "deferred_until_cueqc_keep",
                "training_manifest_allowed": True,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "sample_id": str(item.get("sample_id") or "s"),
                "source_audio_sha256": "c" * 64,
                "source_audio_size": 123,
                "source_core_ids": [f"core-{item['subisland_id']}"],
                "semantic_split_weights_sha256": "a" * 64,
                "inner_edge_refiner_weights_sha256": "b" * 64,
                "pre_asr_candidate": {
                    "schema": "pre_asr_cueqc_features_v10",
                    "boundary_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                    "start": float(item["start_s"]),
                    "end": float(item["end_s"]),
                },
            }
        )
        enriched.append(item)
    _write(path, enriched)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        "schema": "runtime_v12_provisional_export_summary_v3",
        "training_manifest_allowed": True,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "output_sha256": digest,
        "subisland_count": len(enriched),
        "semantic_split_weights_sha256": "a" * 64,
        "inner_edge_refiner_weights_sha256": "b" * 64,
    }
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _teacher_row(row: dict, *, label: str) -> dict:
    return {
        "schema": TEACHER_SCHEMA,
        "prompt_version": TEACHER_PROMPT_VERSION,
        "model": "omni-model",
        "sample_id": str(row["sample_id"]),
        "subisland_id": str(row["subisland_id"]),
        "source_id": str(row["source_id"]),
        "source_partition": str(row["source_partition"]),
        "audio": str(row["audio"]),
        "source_audio_sha256": "c" * 64,
        "source_audio_size": 123,
        "semantic_split_weights_sha256": "a" * 64,
        "inner_edge_refiner_weights_sha256": "b" * 64,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "start_s": float(row["start_s"]),
        "end_s": float(row["end_s"]),
        "duration_s": float(row["duration_s"]),
        "label": label,
    }


def test_canonical_labels_keep_unsure_and_resolve_conflicts_before_training(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    manual = tmp_path / "manual.jsonl"
    output = tmp_path / "canonical.jsonl"
    chunks = [
        {
            "sample_id": "s",
            "source_id": "source-s",
            "subisland_id": item_id,
            "source_partition": "train",
            "audio": "s.wav",
            "start_s": index,
            "end_s": index + 1,
            "duration_s": 1.0,
        }
        for index, item_id in enumerate(("a", "b", "c"))
    ]
    _write_runtime(runtime, chunks)
    _write(
        teacher,
        [
            _teacher_row(chunks[0], label="keep"),
            _teacher_row(chunks[1], label="drop"),
            _teacher_row(chunks[1], label="keep"),
            _teacher_row(chunks[2], label="unsure"),
        ],
    )
    _write(manual, [{"subisland_id": "c", "label": "drop"}])

    summary = compile_labels(
        runtime_chunks=runtime,
        teacher_labels=teacher,
        manual_overrides=manual,
        output=output,
    )
    rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]

    assert [row["teacher_label"] for row in rows] == ["keep", "unsure", "unsure"]
    assert [row["label"] for row in rows] == ["keep", "unsure", "drop"]
    assert [row["training_label"] for row in rows] == [1, -100, 0]
    assert rows[1]["label_source"] == "duplicate_request_conflict_to_unsure"
    assert rows[2]["manual_override_applied"] is True
    assert summary["canonical_label_counts"] == {"drop": 1, "keep": 1, "unsure": 1}
    assert summary["teacher_unsure_ignored"] == 1


def test_canonical_labels_reject_stale_or_incomplete_exact_projection(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    exact = tmp_path / "exact.jsonl"
    output = tmp_path / "canonical.jsonl"
    chunks = [
            {
                "sample_id": "s",
                "source_id": "source-s",
                "subisland_id": item_id,
                "source_partition": "test",
                "audio": "s.wav",
                "start_s": index,
                "end_s": index + 1,
                "duration_s": 1.0,
            }
            for index, item_id in enumerate(("a", "b"))
        ]
    _write_runtime(runtime, chunks)
    _write(teacher, [_teacher_row(row, label="keep") for row in chunks])
    _write(exact, [{"subisland_id": "a", "label": "keep"}])

    with pytest.raises(ValueError, match="exact labels are incomplete; missing 1 chunks"):
        compile_labels(
            runtime_chunks=runtime,
            teacher_labels=teacher,
            exact_labels=exact,
            output=output,
        )


def test_canonical_labels_reject_legacy_pre_asr_teacher_schema(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    output = tmp_path / "canonical.jsonl"
    chunk = {
        "sample_id": "s",
        "source_id": "source-s",
        "subisland_id": "a",
        "source_partition": "train",
        "audio": "s.wav",
        "start_s": 0.0,
        "end_s": 1.0,
        "duration_s": 1.0,
    }
    _write_runtime(runtime, [chunk])
    stale = _teacher_row(chunk, label="keep")
    stale["schema"] = "pre_asr_omni_label_v1"
    _write(teacher, [stale])

    with pytest.raises(ValueError, match="current teacher schema"):
        compile_labels(
            runtime_chunks=runtime,
            teacher_labels=teacher,
            output=output,
        )
