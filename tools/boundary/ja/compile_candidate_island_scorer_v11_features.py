#!/usr/bin/env python3
"""Verify raw 1.7B features and compile Scorer v11 overlap-save rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.candidate_training import candidate_boundary_heatmap_targets  # noqa: E402
from boundary.ja.candidate_windows import plan_candidate_context_windows  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT,
    CANDIDATE_ISLAND_SCORER_V11_FEATURE_CACHE_GATE_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
    CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM,
    CANDIDATE_ISLAND_SCORER_V11_TRAINING_ROW_SCHEMA,
)
from tools.boundary.ja.compile_candidate_island_scorer_v11_canonical import (  # noqa: E402
    LABELS,
    PARTITIONS,
    canonical_frame_labels,
)


SUMMARY_SCHEMA = "candidate_island_scorer_v11_feature_compile_summary_v1"
SIGNED_FEATURE_MANIFEST_SCHEMA = (
    "candidate_island_scorer_v11_signed_feature_manifest_row_v1"
)
FEATURE_CONFIG = {
    "feature_extractor_schema": CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
    "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
    "raw_ptm_dim": CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM,
    "mfcc_dim": CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
    "frame_hop_s": 0.02,
    "context_window_frames": CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT[
        "context_window_frames"
    ],
    "context_overlap_frames": CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT[
        "context_overlap_frames"
    ],
    "window_ownership": CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT[
        "window_ownership"
    ],
    "row_relative_position_features": False,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _unique(rows: Sequence[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"invalid or duplicate {name} source_id: {source_id!r}")
        result[source_id] = row
    return result


def _validate_raw_feature(
    row: dict[str, Any], *, canonical: dict[str, Any]
) -> tuple[Path, str]:
    source_id = str(canonical["source_id"])
    if row.get("schema") != CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA:
        raise ValueError(f"wrong Scorer v11 raw cache row schema: {source_id}")
    if row.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError(f"wrong central boundary contract: {source_id}")
    if row.get("feature_extractor_schema") != (
        CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA
    ):
        raise ValueError(f"wrong Scorer v11 feature extractor schema: {source_id}")
    if row.get("ptm_repo_id") != QWEN_ASR_17B_REPO_ID:
        raise ValueError(f"Scorer v11 features must come from the current 1.7B PTM: {source_id}")
    if row.get("partition") != canonical.get("partition"):
        raise ValueError(f"raw feature partition mismatch: {source_id}")
    if int(row.get("frame_count") or 0) != int(canonical["frame_count"]):
        raise ValueError(f"raw feature frame count mismatch: {source_id}")
    if int(row.get("ptm_dim") or 0) != CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM:
        raise ValueError(
            f"Scorer v11 rejects projected/truncated PTM features: {source_id}"
        )
    if int(row.get("mfcc_dim") or 0) != CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM:
        raise ValueError(f"raw feature MFCC dimension mismatch: {source_id}")
    if not np.isclose(
        float(row.get("frame_hop_s") or 0.0),
        float(canonical["frame_hop_s"]),
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(f"raw feature frame hop mismatch: {source_id}")
    path = _resolve(str(row.get("feature_path") or ""))
    if not path.exists():
        raise FileNotFoundError(path)
    expected_sha = str(row.get("feature_sha256") or "")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"raw feature SHA256 mismatch: {source_id}")
    with np.load(path, allow_pickle=False) as payload:
        if not {"ptm", "mfcc", "frame_hop_s"}.issubset(payload.files):
            raise ValueError(f"raw feature payload is incomplete: {source_id}")
        ptm = np.asarray(payload["ptm"])
        mfcc = np.asarray(payload["mfcc"])
        frame_hop_s = float(np.asarray(payload["frame_hop_s"]).reshape(-1)[0])
        expected_frames = int(canonical["frame_count"])
        if ptm.shape != (expected_frames, CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM):
            raise ValueError(f"raw PTM payload shape mismatch: {source_id} {ptm.shape}")
        if mfcc.shape != (expected_frames, CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM):
            raise ValueError(f"raw MFCC payload shape mismatch: {source_id} {mfcc.shape}")
        if not np.isclose(
            frame_hop_s,
            float(canonical["frame_hop_s"]),
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError(f"raw feature payload frame hop mismatch: {source_id}")
    return path, actual_sha


def compile_features(
    *, canonical_sources: Path, raw_feature_manifest: Path, output_dir: Path
) -> dict[str, Any]:
    canonical_sources = canonical_sources.resolve()
    raw_feature_manifest = raw_feature_manifest.resolve()
    for path in (canonical_sources, raw_feature_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    canonical_rows = _unique(_read_jsonl(canonical_sources), name="canonical")
    raw_rows = _unique(_read_jsonl(raw_feature_manifest), name="raw feature")
    if set(canonical_rows) != set(raw_rows):
        missing = sorted(set(canonical_rows) - set(raw_rows))
        extra = sorted(set(raw_rows) - set(canonical_rows))
        raise ValueError(
            f"canonical/raw feature identities differ: missing={missing[:8]}, extra={extra[:8]}"
        )

    canonical_sha = _sha256(canonical_sources)
    raw_manifest_sha = _sha256(raw_feature_manifest)
    feature_config_sha = _json_sha256(FEATURE_CONFIG)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_dir = output_dir / "source_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    signed_features: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    partition_sources: Counter[str] = Counter()
    partition_windows: Counter[str] = Counter()
    partition_supervised_windows: Counter[str] = Counter()
    partition_ignored_only_windows: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    seen_core: dict[str, tuple[str, str]] = {}

    for source_id in sorted(canonical_rows):
        canonical = canonical_rows[source_id]
        if canonical.get("schema") != CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA:
            raise ValueError(f"wrong Scorer v11 canonical source schema: {source_id}")
        if canonical.get("canonical_label_schema") != (
            CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA
        ):
            raise ValueError(f"wrong Scorer v11 canonical label schema: {source_id}")
        if not bool(canonical.get("training_manifest_allowed")):
            raise ValueError(f"canonical source is not approved for training: {source_id}")
        partition = str(canonical.get("partition") or "")
        if partition not in PARTITIONS:
            raise ValueError(f"invalid source partition: {source_id}")
        synthetic = bool(canonical.get("synthetic_composite"))
        if synthetic and partition != "train":
            raise ValueError(f"synthetic Scorer v11 source leaked into held-out: {source_id}")
        if partition in {"val", "test"} and canonical.get("input_distribution") != (
            "real_workflow_source_windows"
        ):
            raise ValueError(f"held-out source is not real workflow data: {source_id}")
        core_ids = [str(value) for value in canonical.get("core_ids") or ()]
        if not core_ids:
            raise ValueError(f"Scorer v11 source has no frozen core identity: {source_id}")
        for core_id in core_ids:
            current = (source_id, partition)
            previous = seen_core.setdefault(core_id, current)
            if previous != current:
                raise ValueError(f"Scorer v11 core is reused or crosses partitions: {core_id}")

        raw_path, raw_sha = _validate_raw_feature(raw_rows[source_id], canonical=canonical)
        labels = canonical_frame_labels(canonical)
        label_counts["outside_candidate"] += int(np.sum(labels == LABELS["outside_candidate"]))
        label_counts["inside_candidate"] += int(np.sum(labels == LABELS["inside_candidate"]))
        label_counts["unsure"] += int(np.sum(labels == LABELS["unsure"]))
        training_labels = np.where(labels == LABELS["unsure"], -100, labels).astype(np.int64)
        heatmaps = candidate_boundary_heatmap_targets(training_labels)
        label_path = label_dir / f"{source_id}.npz"
        np.savez_compressed(
            label_path,
            canonical_labels=labels,
            training_labels=training_labels,
            start_heatmap=heatmaps.start,
            end_heatmap=heatmaps.end,
            boundary_valid=heatmaps.valid,
            frame_hop_s=np.asarray([float(canonical["frame_hop_s"])], dtype=np.float32),
        )
        label_sha = _sha256(label_path)
        signed_features.append(
            {
                "schema": SIGNED_FEATURE_MANIFEST_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "partition": partition,
                "core_ids": core_ids,
                "synthetic_composite": synthetic,
                "input_distribution": canonical["input_distribution"],
                "feature_extractor_schema": CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
                "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "feature_path": _display(raw_path),
                "feature_sha256": raw_sha,
                "label_path": _display(label_path),
                "label_sha256": label_sha,
                "frame_count": int(canonical["frame_count"]),
                "frame_hop_s": float(canonical["frame_hop_s"]),
                "ptm_dim": CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM,
                "mfcc_dim": CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
                "canonical_sources_sha256": canonical_sha,
                "raw_feature_manifest_sha256": raw_manifest_sha,
                "feature_config_sha256": feature_config_sha,
            }
        )
        windows = plan_candidate_context_windows(int(canonical["frame_count"]))
        for index, window in enumerate(windows):
            definite_owner_frame_count = int(
                np.count_nonzero(
                    training_labels[window.owner_start_frame : window.owner_end_frame]
                    != -100
                )
            )
            training_rows.append(
                {
                    "schema": CANDIDATE_ISLAND_SCORER_V11_TRAINING_ROW_SCHEMA,
                    "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                    "row_id": f"{source_id}::window{index:04d}",
                    "source_id": source_id,
                    "video_id": str(canonical.get("video_id") or ""),
                    "core_ids": core_ids,
                    "partition": partition,
                    "synthetic_composite": synthetic,
                    "input_distribution": canonical["input_distribution"],
                    "feature_path": _display(raw_path),
                    "feature_sha256": raw_sha,
                    "label_path": _display(label_path),
                    "label_sha256": label_sha,
                    "source_frame_count": int(canonical["frame_count"]),
                    "frame_hop_s": float(canonical["frame_hop_s"]),
                    "window_start_frame": window.start_frame,
                    "window_end_frame": window.end_frame,
                    "owner_start_frame": window.owner_start_frame,
                    "owner_end_frame": window.owner_end_frame,
                    "owner_local_start": window.owner_local_start,
                    "owner_local_end": window.owner_local_end,
                    "definite_owner_frame_count": definite_owner_frame_count,
                    "context_window_frames": FEATURE_CONFIG["context_window_frames"],
                    "context_overlap_frames": FEATURE_CONFIG["context_overlap_frames"],
                    "window_ownership": FEATURE_CONFIG["window_ownership"],
                    "canonical_label_schema": CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA,
                    "canonical_sources_sha256": canonical_sha,
                    "raw_feature_manifest_sha256": raw_manifest_sha,
                    "feature_config_sha256": feature_config_sha,
                }
            )
            partition_windows[partition] += 1
            if definite_owner_frame_count > 0:
                partition_supervised_windows[partition] += 1
            else:
                partition_ignored_only_windows[partition] += 1
        partition_sources[partition] += 1

    if set(partition_sources) != PARTITIONS:
        raise ValueError(f"feature compile requires train/val/test: {dict(partition_sources)}")
    signed_manifest = output_dir / "signed_feature_manifest.jsonl"
    _write_jsonl(signed_manifest, signed_features)
    signed_manifest_sha = _sha256(signed_manifest)
    for row in training_rows:
        row["signed_feature_manifest_sha256"] = signed_manifest_sha
    dataset_manifest = output_dir / "training_windows.jsonl"
    _write_jsonl(dataset_manifest, training_rows)
    dataset_manifest_sha = _sha256(dataset_manifest)
    gate = {
        "schema": CANDIDATE_ISLAND_SCORER_V11_FEATURE_CACHE_GATE_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_sources": _display(canonical_sources),
        "canonical_sources_sha256": canonical_sha,
        "raw_feature_manifest": _display(raw_feature_manifest),
        "raw_feature_manifest_sha256": raw_manifest_sha,
        "signed_feature_manifest": _display(signed_manifest),
        "signed_feature_manifest_sha256": signed_manifest_sha,
        "dataset_manifest": _display(dataset_manifest),
        "dataset_manifest_sha256": dataset_manifest_sha,
        "feature_config": FEATURE_CONFIG,
        "feature_config_sha256": feature_config_sha,
        "partition_source_counts": dict(sorted(partition_sources.items())),
        "partition_window_counts": dict(sorted(partition_windows.items())),
        "partition_supervised_window_counts": dict(
            sorted(partition_supervised_windows.items())
        ),
        "partition_ignored_only_window_counts": {
            partition: int(partition_ignored_only_windows.get(partition, 0))
            for partition in sorted(PARTITIONS)
        },
        "canonical_frame_counts": dict(sorted(label_counts.items())),
        "unsure_excluded_from_normalization_loss_metrics_gate": True,
        "owner_frames_are_unique": True,
        "training_manifest_allowed": True,
    }
    gate_path = output_dir / "feature_cache_gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_sources_sha256": canonical_sha,
        "signed_feature_manifest": _display(signed_manifest),
        "signed_feature_manifest_sha256": signed_manifest_sha,
        "dataset_manifest": _display(dataset_manifest),
        "dataset_manifest_sha256": dataset_manifest_sha,
        "feature_cache_gate": _display(gate_path),
        "feature_cache_gate_sha256": _sha256(gate_path),
        "feature_config_sha256": feature_config_sha,
        "source_count": len(canonical_rows),
        "window_count": len(training_rows),
        "partition_source_counts": dict(sorted(partition_sources.items())),
        "partition_window_counts": dict(sorted(partition_windows.items())),
        "partition_supervised_window_counts": dict(
            sorted(partition_supervised_windows.items())
        ),
        "partition_ignored_only_window_counts": {
            partition: int(partition_ignored_only_windows.get(partition, 0))
            for partition in sorted(PARTITIONS)
        },
        "canonical_frame_counts": dict(sorted(label_counts.items())),
        "training_manifest_allowed": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--raw-feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return compile_features(
        canonical_sources=Path(args.canonical_sources),
        raw_feature_manifest=Path(args.raw_feature_manifest),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
