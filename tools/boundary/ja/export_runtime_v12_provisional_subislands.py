#!/usr/bin/env python3
"""Export actual Runtime v12 provisional sub-islands for a fixed source pool."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.pipeline import (  # noqa: E402
    _boundary_config,
    _build_processing_spans,
    _pre_asr_candidates_for_spans,
)
from audio.chunk_packer import PackedChunk  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.split_model import SEMANTIC_SPLIT_V4_RUNTIME_ADAPTER  # noqa: E402


RUNTIME_SCHEMA = "runtime_v12_provisional_subisland_v2"
RUNTIME_SUMMARY_SCHEMA = "runtime_v12_provisional_export_summary_v3"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Runtime export JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Runtime export row must be an object at {path}:{line_number}")
            rows.append(row)
        return rows


def _write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace the JSONL in one rename so an interrupted source cannot corrupt it."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_audio_path(*, value: str, items_path: Path) -> Path:
    audio = Path(value)
    if audio.is_absolute():
        resolved = audio
    else:
        manifest_relative = items_path.parent / audio
        project_relative = PROJECT_ROOT / audio
        resolved = manifest_relative if manifest_relative.exists() else project_relative
    if not resolved.exists():
        raise FileNotFoundError(f"source audio not found: {resolved}")
    return resolved.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_sha(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"Runtime export row is missing exact {field}")
    return normalized


def _validate_exported_rows(
    rows: list[dict[str, Any]],
    *,
    manifest_sources: dict[str, tuple[Path, dict[str, Any], str]],
    split_sha256: str,
    inner_sha256: str,
) -> tuple[set[str], dict[str, str], set[str]]:
    """Validate resume rows and distinguish complete from interrupted sources."""

    seen_subislands: set[str] = set()
    core_owners: dict[str, str] = {}
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != RUNTIME_SCHEMA:
            raise ValueError("resume requires current Runtime v12 provisional subisland v2 rows")
        if row.get("inner_execution_status") != "deferred_until_cueqc_keep":
            raise ValueError("resume rows must defer Inner until CueQC keep")
        if row.get("training_manifest_allowed") is not True:
            raise ValueError("resume rows are not training-approved")
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id not in manifest_sources:
            raise ValueError(f"resume row references unknown source sample_id: {sample_id!r}")
        _audio, source, audio_sha = manifest_sources[sample_id]
        if str(row.get("source_id") or "").strip() != str(source.get("source_id") or "").strip():
            raise ValueError(f"resume row source_id mismatch for {sample_id!r}")
        if str(row.get("source_partition") or "").strip() != str(source.get("source_partition") or "").strip():
            raise ValueError(f"resume row source_partition mismatch for {sample_id!r}")
        if str(row.get("audio") or "").strip() != str(_audio):
            raise ValueError(f"resume row audio mismatch for {sample_id!r}")
        if str(row.get("source_audio_sha256") or "").lower() != audio_sha:
            raise ValueError(f"resume row source audio SHA mismatch for {sample_id!r}")
        if int(row.get("source_audio_size") or -1) != _audio.stat().st_size:
            raise ValueError(f"resume row source audio size mismatch for {sample_id!r}")
        subisland_id = str(row.get("subisland_id") or "").strip()
        if not subisland_id or subisland_id in seen_subislands:
            raise ValueError(f"duplicate or missing resumed subisland_id: {subisland_id!r}")
        seen_subislands.add(subisland_id)
        try:
            source_index = int(row["source_subisland_index"])
            source_count = int(row["source_subisland_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("resume rows require source_subisland_index/count") from exc
        if source_count <= 0 or source_index < 0 or source_index >= source_count:
            raise ValueError(f"invalid source subisland ordinal for {subisland_id!r}")
        expected_id = f"{sample_id}__v12s{source_index:02d}"
        if subisland_id != expected_id:
            raise ValueError(f"resume subisland_id does not match source ordinal: {subisland_id!r}")
        try:
            start = float(row["start_s"])
            end = float(row["end_s"])
            duration = float(row["duration_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"resume row {subisland_id!r} has invalid coordinates") from exc
        if not all(math.isfinite(value) for value in (start, end, duration)) or end <= start:
            raise ValueError(f"resume row {subisland_id!r} has invalid coordinates")
        if not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(f"resume row {subisland_id!r} duration mismatch")
        if any(float(value) < 0.0 for value in (start, end)):
            raise ValueError(f"resume row {subisland_id!r} has negative coordinates")
        core_ids = row.get("source_core_ids")
        if not isinstance(core_ids, list) or any(not str(value).strip() for value in core_ids):
            raise ValueError(f"resume row {subisland_id!r} has invalid source_core_ids")
        if len(core_ids) != len(set(str(value) for value in core_ids)):
            raise ValueError(f"resume row {subisland_id!r} repeats source_core_ids")
        expected_core_ids = source_core_ids_for_span(source, start_s=start, end_s=end)
        if [str(value) for value in core_ids] != expected_core_ids:
            raise ValueError(f"resume row {subisland_id!r} source_core_ids do not match coordinates")
        for core_id in expected_core_ids:
            previous = core_owners.get(core_id)
            if previous is not None and previous != subisland_id:
                raise ValueError(f"core {core_id!r} is reused by resumed subislands")
            core_owners[core_id] = subisland_id
        candidate = row.get("pre_asr_candidate")
        if not isinstance(candidate, dict) or candidate.get("schema") != "pre_asr_cueqc_features_v10":
            raise ValueError(f"resume row {subisland_id!r} lacks current Pre-ASR candidate")
        if not ACOUSTIC_BINARY_V12_CONTRACT.matches(candidate.get("boundary_contract_id")):
            raise ValueError(f"resume row {subisland_id!r} has stale candidate contract")
        if not ACOUSTIC_BINARY_V12_CONTRACT.matches(row.get("boundary_serialization_contract_id")):
            raise ValueError(f"resume row {subisland_id!r} has stale Boundary contract")
        if _exact_sha(row.get("semantic_split_weights_sha256"), field="semantic_split_weights_sha256") != split_sha256:
            raise ValueError("resumed CueQC runtime rows use a stale Split checkpoint")
        if _exact_sha(row.get("inner_edge_refiner_weights_sha256"), field="inner_edge_refiner_weights_sha256") != inner_sha256:
            raise ValueError("resumed CueQC runtime rows use a stale Inner checkpoint")
        if "inner_edge_prediction" in row:
            raise ValueError("resume rows must not contain pre-CueQC Inner predictions")
        by_source.setdefault(sample_id, []).append(row)
    complete: set[str] = set()
    partial: set[str] = set()
    for sample_id, source_rows in by_source.items():
        counts = {int(row["source_subisland_count"]) for row in source_rows}
        indexes = {int(row["source_subisland_index"]) for row in source_rows}
        if len(counts) != 1:
            raise ValueError(f"resumed source {sample_id!r} has inconsistent subisland counts")
        count = next(iter(counts))
        if indexes == set(range(count)) and len(source_rows) == count:
            complete.add(sample_id)
        else:
            partial.add(sample_id)
    return complete, core_owners, partial


def source_core_ids_for_span(
    source_row: dict[str, Any], *, start_s: float, end_s: float
) -> list[str]:
    core_ids: list[str] = []
    for core_number, core in enumerate(source_row.get("core_spans") or [], start=1):
        core_id = str(core.get("core_id") or "").strip()
        if not core_id:
            raise ValueError(f"source core {core_number} is missing core_id")
        core_start = float(core.get("start_s"))
        core_end = float(core.get("end_s"))
        if core_end <= core_start:
            raise ValueError(f"source core {core_id!r} has invalid coordinates")
        if core_end > start_s and core_start < end_s:
            core_ids.append(core_id)
    return core_ids


def validate_binary_split_chunk(chunk: PackedChunk, *, sample_id: str) -> None:
    if not ACOUSTIC_BINARY_V12_CONTRACT.matches(chunk.boundary_contract_id):
        raise ValueError(f"{sample_id}: current Boundary contract is required")
    weak = list(chunk.weak_cut_candidates or [])
    if any(str(row.get("label") or "") not in {"cut", "continue"} for row in weak):
        raise ValueError(f"{sample_id}: Split v4 candidate emitted a non-binary label")
    if any(float(row.get("p_unsure") or 0.0) != 0.0 for row in weak):
        raise ValueError(f"{sample_id}: Split v4 candidate emitted p_unsure")
    if chunk.boundary_decision_source == SEMANTIC_SPLIT_V4_RUNTIME_ADAPTER:
        return
    if (
        chunk.boundary_source == "outer_edge_refiner_v3"
        and chunk.boundary_decision_source == "outer_edge_refiner_v3"
        and not chunk.semantic_event_ids
        and not chunk.primary_cut_candidates
    ):
        return
    raise ValueError(f"{sample_id}: expected binary Split v4 runtime adapter")


def run(args: argparse.Namespace) -> None:
    apply_vram_safety_cap(0.95)
    manifest_sources: dict[str, tuple[Path, dict[str, Any], str]] = {}
    source_partitions: dict[str, set[str]] = {}
    core_sources: dict[str, str] = {}
    items_path = Path(args.audit_items).resolve()
    for row in _rows(items_path):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("source manifest row is missing sample_id")
        if sample_id in manifest_sources:
            raise ValueError(f"duplicate source sample_id: {sample_id}")
        try:
            audio = resolve_audio_path(value=str(row["audio"]), items_path=items_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{sample_id}: {exc}") from exc
        source_id = str(row.get("source_id") or "").strip()
        partition = str(row.get("source_partition") or "").strip()
        if not source_id or partition not in {"train", "val", "test"}:
            raise ValueError(
                f"{sample_id}: source manifest requires frozen source_id and "
                "source_partition=train|val|test"
            )
        source_partitions.setdefault(source_id, set()).add(partition)
        for core in row.get("core_spans") or []:
            core_id = str(core.get("core_id") or "").strip()
            if not core_id:
                raise ValueError(f"{sample_id}: source core is missing core_id")
            previous_source = core_sources.get(core_id)
            if previous_source is not None:
                raise ValueError(
                    f"core {core_id!r} is reused by sources "
                    f"{previous_source!r} and {source_id!r}"
                )
            core_sources[core_id] = source_id
        audio_sha256 = _sha256(audio)
        manifest_sources[sample_id] = (audio, row, audio_sha256)
    if any(len(values) != 1 for values in source_partitions.values()):
        raise ValueError("CueQC source identity crosses frozen partitions")
    if not manifest_sources:
        raise ValueError("source manifest is empty")
    sources: dict[str, tuple[Path, dict[str, Any], str]] = dict(manifest_sources)
    if args.max_sources > 0:
        sources = dict(list(sorted(sources.items()))[: args.max_sources])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = _rows(output) if args.resume and output.exists() else []
    if not args.resume:
        _write_rows_atomic(output, [])
    boundary_config = _boundary_config()
    split_checkpoint = Path(boundary_config["semantic_split_model_path"])
    outer_checkpoint = Path(boundary_config["outer_edge_refiner_model_path"])
    inner_checkpoint = Path(boundary_config["inner_edge_refiner_model_path"])
    split_sha256 = _sha256(split_checkpoint)
    inner_sha256 = _sha256(inner_checkpoint)
    if args.expected_split_sha256 and split_sha256 != args.expected_split_sha256:
        raise ValueError(
            "active Split checkpoint SHA mismatch: "
            f"expected {args.expected_split_sha256}, got {split_sha256}"
        )
    completed_sources: set[str] = set()
    exported_core_owners: dict[str, str] = {}
    partial_sources: set[str] = set()
    recovered_partial_sources: set[str] = set()
    if exported:
        completed_sources, exported_core_owners, partial_sources = _validate_exported_rows(
            exported,
            manifest_sources=manifest_sources,
            split_sha256=split_sha256,
            inner_sha256=inner_sha256,
        )
        if partial_sources:
            # A crash between rows is recoverable: remove only the incomplete
            # source groups, then regenerate them.  A complete source is never
            # silently skipped based on a single row.
            exported = [
                row for row in exported if str(row.get("sample_id") or "") not in partial_sources
            ]
            recovered_partial_sources = set(partial_sources)
            _write_rows_atomic(output, exported)
            exported_core_owners = {
                core_id: owner
                for core_id, owner in exported_core_owners.items()
                if owner in {str(row.get("subisland_id") or "") for row in exported}
            }
            completed_sources -= partial_sources
    for source_index, (sample_id, source_item) in enumerate(
        sorted(sources.items()), start=1
    ):
        audio, source_row, audio_sha256 = source_item
        if sample_id in completed_sources:
            print(
                f"runtime_v12_provisional={source_index}/{len(sources)} "
                f"sample={sample_id} status=resume_skip",
                flush=True,
            )
            continue
        print(
            f"runtime_v12_provisional={source_index}/{len(sources)} sample={sample_id}",
            flush=True,
        )
        spans = _build_processing_spans(str(audio))
        if not all(isinstance(span, PackedChunk) for span in spans):
            raise ValueError(f"{sample_id}: Runtime v12 did not return PackedChunk rows")
        chunks = [span for span in spans if isinstance(span, PackedChunk)]
        for chunk in chunks:
            validate_binary_split_chunk(chunk, sample_id=sample_id)
        candidates = _pre_asr_candidates_for_spans(str(audio), chunks)
        if len(candidates) != len(chunks):
            raise ValueError(f"{sample_id}: candidate/chunk count mismatch")
        source_rows: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            subisland_id = f"{sample_id}__v12s{index:02d}"
            source_core_ids = source_core_ids_for_span(
                source_row,
                start_s=float(chunk.start),
                end_s=float(chunk.end),
            )
            for core_id in source_core_ids:
                previous = exported_core_owners.get(core_id)
                if previous is not None and previous != subisland_id:
                    raise ValueError(
                        f"core {core_id!r} is reused by provisional subislands "
                        f"{previous!r} and {subisland_id!r}"
                    )
                exported_core_owners[core_id] = subisland_id
            source_rows.append(
                {
                    "schema": RUNTIME_SCHEMA,
                    "sample_id": sample_id,
                    "source_id": str(source_row["source_id"]),
                    "source_partition": str(source_row["source_partition"]),
                    "subisland_id": subisland_id,
                    "source_core_ids": source_core_ids,
                    "audio": str(audio),
                    "start_s": float(chunk.start),
                    "end_s": float(chunk.end),
                    "duration_s": float(chunk.duration),
                    "source_audio_sha256": audio_sha256,
                    "source_audio_size": int(audio.stat().st_size),
                    "source_subisland_index": index,
                    "source_subisland_count": len(chunks),
                    "left_event_id": (
                        chunk.semantic_event_ids[0]
                        if chunk.semantic_event_ids
                        else None
                    ),
                    "right_event_id": (
                        chunk.semantic_event_ids[-1]
                        if chunk.semantic_event_ids
                        else None
                    ),
                    "semantic_event_ids": list(chunk.semantic_event_ids or []),
                    "semantic_event_probabilities": list(
                        chunk.semantic_event_probabilities or []
                    ),
                    "inner_execution_status": "deferred_until_cueqc_keep",
                    "boundary_source": chunk.boundary_source,
                    "boundary_decision_source": chunk.boundary_decision_source,
                    "primary_cut_candidates": list(
                        chunk.primary_cut_candidates or []
                    ),
                    "weak_cut_candidates": list(chunk.weak_cut_candidates or []),
                    "pre_asr_candidate": candidates[index],
                    "training_manifest_allowed": True,
                    "semantic_split_weights_sha256": split_sha256,
                    "inner_edge_refiner_weights_sha256": inner_sha256,
                    "boundary_serialization_contract_id": (
                        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                    ),
                }
            )
        exported.extend(source_rows)
        _write_rows_atomic(output, exported)
        completed_sources.add(sample_id)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    # Re-validate the final file, including all source ordinals, before writing
    # the summary.  This catches a duplicate or partial row introduced by an
    # external edit after the last source was processed.
    completed_sources, exported_core_owners, partial_sources = _validate_exported_rows(
        exported,
        manifest_sources=manifest_sources,
        split_sha256=split_sha256,
        inner_sha256=inner_sha256,
    )
    if partial_sources or completed_sources != {str(row.get("sample_id") or "") for row in exported}:
        raise ValueError("Runtime export finished with incomplete source groups")
    actual_source_ids = sorted({str(row["sample_id"]) for row in exported})
    actual_partition_counts = {
        partition: sum(str(row.get("source_partition")) == partition for row in exported)
        for partition in ("train", "val", "test")
    }
    core_use_counts: dict[str, int] = {}
    for row in exported:
        for core_id in row.get("source_core_ids") or []:
            core_use_counts[str(core_id)] = core_use_counts.get(str(core_id), 0) + 1
    summary = {
        "schema": RUNTIME_SUMMARY_SCHEMA,
        "source_count": len(actual_source_ids),
        "source_manifest_count": len(manifest_sources),
        "subisland_count": len(exported),
        "source_ids": actual_source_ids,
        "source_partition_counts": {
            partition: sum(
                str(manifest_sources[sample_id][1].get("source_partition")) == partition
                for sample_id in actual_source_ids
            )
            for partition in ("train", "val", "test")
        },
        "subisland_partition_counts": actual_partition_counts,
        "partition_counts": dict(actual_partition_counts),
        "unique_core_count": len(core_use_counts),
        "max_core_use": max(core_use_counts.values(), default=0),
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "training_manifest_allowed": True,
        "split_runtime_adapter": SEMANTIC_SPLIT_V4_RUNTIME_ADAPTER,
        "split_checkpoint": str(split_checkpoint),
        "split_checkpoint_sha256": split_sha256,
        "outer_checkpoint": str(outer_checkpoint),
        "inner_checkpoint": str(inner_checkpoint),
        "outer_checkpoint_sha256": _sha256(outer_checkpoint),
        "inner_checkpoint_sha256": _sha256(inner_checkpoint),
        "audit_items": str(items_path),
        "audit_items_sha256": _sha256(items_path),
        "output_sha256": _sha256(output),
        "recovered_partial_sources": sorted(recovered_partial_sources),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-items", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-split-sha256", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
