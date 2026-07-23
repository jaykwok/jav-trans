#!/usr/bin/env python3
"""Compile canonical CueQC v13 labels while excluding unsure from training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.asr.cueqc.label_runtime_v12_cueqc_v13_with_omni import (
    PROMPT_VERSION as TEACHER_PROMPT_VERSION,
    SCHEMA as TEACHER_SCHEMA,
)


VALID_LABELS = {"drop", "keep", "unsure"}
TRAINING_LABELS = {"drop": 0, "keep": 1, "unsure": -100}
RUNTIME_SCHEMA = "runtime_v12_provisional_subisland_v2"
CANONICAL_SCHEMA = "cueqc_v13_canonical_label_v2"
SUMMARY_SCHEMA = "cueqc_v13_canonical_label_summary_v2"
BOUNDARY_CONTRACT_ID = ACOUSTIC_BINARY_V12_CONTRACT.contract_id


def _rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
        return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_summary_path(runtime_chunks: Path) -> Path:
    return runtime_chunks.with_suffix(".summary.json")


def _validate_current_runtime(
    rows: list[dict[str, Any]], *, runtime_chunks: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, set[str]]]:
    if not rows:
        raise ValueError("Runtime v12 chunk manifest is empty")
    summary_path = _runtime_summary_path(runtime_chunks)
    if not summary_path.is_file():
        raise ValueError(
            "current CueQC canonical compilation requires the Runtime export summary"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Runtime export summary: {summary_path}") from exc
    if not isinstance(summary, dict) or summary.get("schema") != "runtime_v12_provisional_export_summary_v3":
        raise ValueError("CueQC canonical compilation requires the current Runtime export summary schema")
    if summary.get("training_manifest_allowed") is not True:
        raise ValueError("Runtime export summary is not approved for training")
    if summary.get("boundary_serialization_contract_id") != BOUNDARY_CONTRACT_ID:
        raise ValueError("Runtime export summary uses a stale Boundary contract")
    expected_runtime_sha = str(summary.get("output_sha256") or "").lower()
    if expected_runtime_sha != _sha256(runtime_chunks):
        raise ValueError("Runtime export summary SHA does not match the chunk manifest")
    runtime: dict[str, dict[str, Any]] = {}
    source_partitions: dict[str, set[str]] = {}
    source_audio: dict[str, str] = {}
    core_owner: dict[str, str] = {}
    upstream = {
        "semantic_split_weights_sha256": set(),
        "inner_edge_refiner_weights_sha256": set(),
    }
    for row in rows:
        if row.get("schema") != RUNTIME_SCHEMA:
            raise ValueError("CueQC canonical compilation accepts only Runtime v12 provisional v2 rows")
        if row.get("inner_execution_status") != "deferred_until_cueqc_keep":
            raise ValueError("CueQC canonical compilation requires Inner to be deferred until keep")
        if row.get("training_manifest_allowed") is not True:
            raise ValueError("CueQC canonical compilation requires an approved Runtime manifest")
        if str(row.get("boundary_serialization_contract_id") or "") != BOUNDARY_CONTRACT_ID:
            raise ValueError("CueQC canonical compilation requires the current Boundary contract")
        item_id = str(row.get("subisland_id") or "").strip()
        if not item_id or item_id in runtime:
            raise ValueError(f"duplicate or missing Runtime subisland_id: {item_id!r}")
        sample_id = str(row.get("sample_id") or "").strip()
        source_id = str(row.get("source_id") or "").strip()
        partition = str(row.get("source_partition") or "").strip()
        if not sample_id or not source_id or partition not in {"train", "val", "test"}:
            raise ValueError(f"Runtime row {item_id!r} has incomplete frozen identity")
        try:
            start = float(row["start_s"])
            end = float(row["end_s"])
            duration = float(row["duration_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Runtime row {item_id!r} has invalid coordinates") from exc
        if not all(math.isfinite(value) for value in (start, end, duration)) or end <= start:
            raise ValueError(f"Runtime row {item_id!r} has invalid coordinates")
        if not math.isclose(duration, end - start, abs_tol=1e-5):
            raise ValueError(f"Runtime row {item_id!r} duration mismatch")
        candidate = row.get("pre_asr_candidate")
        if not isinstance(candidate, dict) or candidate.get("schema") != "pre_asr_cueqc_features_v10":
            raise ValueError(f"Runtime row {item_id!r} has stale Pre-ASR candidate provenance")
        if candidate.get("boundary_contract_id") != BOUNDARY_CONTRACT_ID:
            raise ValueError(f"Runtime row {item_id!r} has stale candidate Boundary contract")
        for key in upstream:
            value = str(row.get(key) or "").lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"Runtime row {item_id!r} is missing exact {key}")
            upstream[key].add(value)
        source_partitions.setdefault(source_id, set()).add(partition)
        audio = str(row.get("audio") or "").strip()
        audio_sha = str(row.get("source_audio_sha256") or "").lower()
        if not audio or len(audio_sha) != 64:
            raise ValueError(f"Runtime row {item_id!r} is missing source audio binding")
        previous_audio = source_audio.setdefault(sample_id, f"{audio}\0{audio_sha}")
        if previous_audio != f"{audio}\0{audio_sha}":
            raise ValueError(f"Runtime source audio binding is inconsistent: {sample_id}")
        core_ids = row.get("source_core_ids")
        if not isinstance(core_ids, list) or len(core_ids) != len(set(str(value) for value in core_ids)):
            raise ValueError(f"Runtime row {item_id!r} has invalid source_core_ids")
        for core_id in core_ids:
            core = str(core_id).strip()
            if not core:
                raise ValueError(f"Runtime row {item_id!r} has an empty source_core_id")
            previous = core_owner.get(core)
            if previous is not None and previous != item_id:
                raise ValueError(f"Runtime core {core!r} is reused by multiple subislands")
            core_owner[core] = item_id
        runtime[item_id] = row
    leaked = [source_id for source_id, values in source_partitions.items() if len(values) != 1]
    if leaked:
        raise ValueError(f"CueQC source identity crosses partitions: {sorted(leaked)[:3]}")
    for key, values in upstream.items():
        expected = str(summary.get(key) or "").lower()
        if len(values) != 1 or expected not in values:
            raise ValueError(f"Runtime summary has inconsistent {key}")
    if int(summary.get("subisland_count") or -1) != len(rows):
        raise ValueError("Runtime summary subisland_count does not match the manifest")
    return summary, runtime, source_partitions


def _label(row: dict[str, Any]) -> str:
    value = str(row.get("label") or row.get("verdict") or "").strip().lower()
    return value if value in VALID_LABELS else ""


def _unique_by_id(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("subisland_id") or "").strip()
        if not item_id:
            raise ValueError(f"{name} row is missing subisland_id")
        if item_id in result:
            raise ValueError(f"duplicate {name} subisland_id: {item_id}")
        result[item_id] = row
    return result


def compile_labels(
    *,
    runtime_chunks: Path,
    teacher_labels: Path,
    output: Path,
    manual_overrides: Path | None = None,
    exact_labels: Path | None = None,
) -> dict[str, Any]:
    runtime_rows = _rows(runtime_chunks)
    runtime_summary, runtime, source_partitions = _validate_current_runtime(
        runtime_rows, runtime_chunks=runtime_chunks
    )
    observed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    teacher_models: set[str] = set()
    for row in _rows(teacher_labels):
        if row.get("schema") != TEACHER_SCHEMA:
            raise ValueError(
                f"CueQC canonical compilation requires current teacher schema "
                f"{TEACHER_SCHEMA}"
            )
        if str(row.get("prompt_version") or "") != TEACHER_PROMPT_VERSION:
            raise ValueError(
                "CueQC canonical compilation requires the current v13 teacher prompt"
            )
        model = str(row.get("model") or "").strip()
        if not model:
            raise ValueError("CueQC teacher row is missing model identity")
        teacher_models.add(model)
        item_id = str(row.get("subisland_id") or "").strip()
        label = _label(row)
        if not item_id or not label:
            raise ValueError("teacher row must contain subisland_id and keep/drop/unsure")
        if item_id not in runtime:
            raise ValueError(f"teacher label has no runtime chunk: {item_id}")
        runtime_row = runtime[item_id]
        for key in ("sample_id", "source_id", "source_partition", "audio"):
            if str(row.get(key) or "") != str(runtime_row.get(key) or ""):
                raise ValueError(f"teacher/runtime {key} mismatch for {item_id}")
        for key in ("start_s", "end_s", "duration_s"):
            try:
                teacher_value = float(row[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"teacher row has invalid {key} for {item_id}"
                ) from exc
            if not math.isfinite(teacher_value) or not math.isclose(
                teacher_value, float(runtime_row[key]), abs_tol=1e-6
            ):
                raise ValueError(f"teacher/runtime {key} mismatch for {item_id}")
        for key in (
            "source_audio_sha256",
            "source_audio_size",
            "semantic_split_weights_sha256",
            "inner_edge_refiner_weights_sha256",
            "boundary_serialization_contract_id",
        ):
            if str(row.get(key) or "") != str(runtime_row.get(key) or ""):
                raise ValueError(f"teacher/runtime {key} mismatch for {item_id}")
        observed[item_id].append(row)
    if len(teacher_models) != 1:
        raise ValueError("CueQC teacher labels mix Omni model identities")
    missing = sorted(set(runtime) - set(observed))
    if missing:
        raise ValueError(f"teacher labels are incomplete; missing {len(missing)} chunks")

    manual = _unique_by_id(_rows(manual_overrides), name="manual override")
    unknown_manual = sorted(set(manual) - set(runtime))
    if unknown_manual:
        raise ValueError(f"manual overrides have no runtime chunk: {unknown_manual[:3]}")
    exact = _unique_by_id(_rows(exact_labels), name="exact label")
    if exact_labels is not None:
        unknown_exact = sorted(set(exact) - set(runtime))
        if unknown_exact:
            raise ValueError(f"exact labels have no runtime chunk: {unknown_exact[:3]}")
        missing_exact = sorted(set(runtime) - set(exact))
        if missing_exact:
            raise ValueError(
                f"exact labels are incomplete; missing {len(missing_exact)} chunks"
            )

    result: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    conflicts = 0
    manual_count = 0
    for chunk in runtime_rows:
        item_id = str(chunk["subisland_id"])
        teacher_rows = observed[item_id]
        teacher_values = sorted({_label(row) for row in teacher_rows})
        if len(teacher_rows) > 1 and len(teacher_values) == 1:
            raise ValueError(
                f"duplicate teacher requests with the same label for Runtime chunk: {item_id}"
            )
        conflict = len(teacher_values) > 1
        teacher_label = "unsure" if conflict else teacher_values[0]
        source = "duplicate_request_conflict_to_unsure" if conflict else str(
            teacher_rows[-1].get("label_source") or "omni_teacher"
        )
        if conflict:
            conflicts += 1

        override = manual.get(item_id)
        override_label = _label(override or {})
        if override is not None and not override_label:
            raise ValueError(f"manual override has invalid label: {item_id}")
        canonical = override_label or teacher_label
        if override_label:
            source = "manual_override"
            manual_count += 1
        counts[canonical] += 1
        exact_row = exact.get(item_id)
        result.append(
            {
                "schema": CANONICAL_SCHEMA,
                "sample_id": str(chunk["sample_id"]),
                "source_id": str(chunk["source_id"]),
                "subisland_id": item_id,
                "query_id": item_id,
                "source_partition": str(chunk["source_partition"]),
                "audio": str(chunk["audio"]),
                "start_s": float(chunk["start_s"]),
                "end_s": float(chunk["end_s"]),
                "duration_s": float(chunk["duration_s"]),
                "teacher_label": teacher_label,
                "label": canonical,
                "training_label": TRAINING_LABELS[canonical],
                "training_label_included": canonical in {"drop", "keep"},
                "training_ignore_reason": "teacher_unsure" if canonical == "unsure" else "",
                "label_source": source,
                "teacher_labels_observed": teacher_values,
                "teacher_response_count": len(teacher_rows),
                "teacher_conflict": conflict,
                "manual_override_applied": bool(override_label),
                "exact_core_label": _label(exact_row or {}),
                "training_manifest_allowed": True,
                "runtime_schema": RUNTIME_SCHEMA,
                "inner_execution_status": str(chunk["inner_execution_status"]),
                "boundary_serialization_contract_id": BOUNDARY_CONTRACT_ID,
                "source_core_ids": list(chunk.get("source_core_ids") or []),
                "source_audio_sha256": str(chunk.get("source_audio_sha256") or ""),
                "source_audio_size": int(chunk.get("source_audio_size") or 0),
                "semantic_split_weights_sha256": str(
                    chunk.get("semantic_split_weights_sha256") or ""
                ),
                "inner_edge_refiner_weights_sha256": str(
                    chunk.get("inner_edge_refiner_weights_sha256") or ""
                ),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        output,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result),
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "runtime_chunk_count": len(runtime_rows),
        "canonical_label_counts": dict(sorted(counts.items())),
        "training_label_count": counts["drop"] + counts["keep"],
        "teacher_unsure_ignored": counts["unsure"],
        "duplicate_request_conflict_count": conflicts,
        "manual_override_count": manual_count,
        "exact_label_count": len(exact),
        "source_count": len(source_partitions),
        "partition_counts": dict(
            sorted(
                Counter(next(iter(values)) for values in source_partitions.values()).items()
            )
        ),
        "training_manifest_allowed": True,
        "runtime_schema": RUNTIME_SCHEMA,
        "runtime_summary": str(_runtime_summary_path(runtime_chunks)),
        "runtime_summary_sha256": _sha256(_runtime_summary_path(runtime_chunks)),
        "runtime_chunks": str(runtime_chunks),
        "runtime_chunks_sha256": _sha256(runtime_chunks),
        "teacher_labels": str(teacher_labels),
        "teacher_labels_sha256": _sha256(teacher_labels),
        "teacher_schema": TEACHER_SCHEMA,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "teacher_model": next(iter(teacher_models)),
        "semantic_split_weights_sha256": str(runtime_summary["semantic_split_weights_sha256"]),
        "inner_edge_refiner_weights_sha256": str(runtime_summary["inner_edge_refiner_weights_sha256"]),
        "boundary_serialization_contract_id": BOUNDARY_CONTRACT_ID,
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    _write_text_atomic(
        output.with_suffix(".summary.json"),
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-chunks", required=True)
    parser.add_argument("--teacher-labels", required=True)
    parser.add_argument("--manual-overrides", default="")
    parser.add_argument("--exact-labels", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(compile_labels(
        runtime_chunks=Path(args.runtime_chunks),
        teacher_labels=Path(args.teacher_labels),
        manual_overrides=Path(args.manual_overrides) if args.manual_overrides else None,
        exact_labels=Path(args.exact_labels) if args.exact_labels else None,
        output=Path(args.output),
    ), ensure_ascii=False))
