#!/usr/bin/env python3
"""Build label-only diagnostic rows for rescoring a Scorer v10 checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT,
    SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
    SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
)
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (  # noqa: E402
    CANONICAL_LABELS,
    CANONICAL_LABEL_SCHEMA,
    _safe_id,
    _validate_sources,
    _write_jsonl,
    canonical_frame_labels,
)


SUMMARY_SCHEMA = "speech_scorer_v10_canonical_rescore_manifest_summary_v1"
DIAGNOSTIC_ROW_SCHEMA = "speech_scorer_v10_binary_diagnostic_row_v1"
FRAME_HOP_S = 0.02


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    *,
    canonical_sources: Path,
    audio_manifest: Path,
    feature_manifest: Path,
    feature_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    summary = json.loads(feature_summary.read_text(encoding="utf-8-sig"))
    if int(summary.get("errors") or 0) or int(summary.get("skipped") or 0):
        raise ValueError("Scorer diagnostic reuse requires a complete feature cache")
    if str(summary.get("ptm") or "") != QWEN_ASR_17B_REPO_ID:
        raise ValueError("Scorer diagnostic reuse requires the 1.7B PTM")
    original_audio_manifest = Path(str(summary.get("source_manifest") or ""))
    if not original_audio_manifest.is_file():
        raise ValueError("Scorer feature summary source manifest is missing")
    if _sha256(original_audio_manifest) != _sha256(audio_manifest):
        raise ValueError("Scorer diagnostic feature reuse requires identical audio manifests")

    sources = _rows(canonical_sources)
    _validate_sources(sources)
    source_by_id = {str(row["source_id"]): row for row in sources}
    audio_rows = json.loads(audio_manifest.read_text(encoding="utf-8-sig"))
    audio_by_id = {str(row["audio_id"]): row for row in audio_rows}
    if set(audio_by_id) != set(source_by_id):
        raise ValueError("Scorer diagnostic audio identities do not match canonical")
    for source_id, source in source_by_id.items():
        audio_row = audio_by_id[source_id]
        if (
            str(audio_row.get("audio") or "") != str(source["audio"])
            or str(audio_row.get("partition") or "") != str(source["partition"])
            or str(audio_row.get("row_role") or "") != str(source["row_role"])
        ):
            raise ValueError(f"Scorer diagnostic audio metadata changed: {source_id}")

    features: dict[str, dict[str, Any]] = {}
    for row in _rows(feature_manifest):
        source_id = str(row.get("audio_id") or "")
        if not source_id or source_id in features:
            raise ValueError("Scorer feature manifest requires unique audio_id values")
        if str(row.get("ptm") or "") != QWEN_ASR_17B_REPO_ID:
            raise ValueError("Scorer cached feature row has the wrong PTM")
        if int(row.get("ptm_dim") or 0) != SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM:
            raise ValueError("Scorer cached feature row must retain raw PTM2048")
        if int(row.get("mfcc_dim") or 0) != SPEECH_ISLAND_SCORER_V10_MFCC_DIM:
            raise ValueError("Scorer cached feature row must contain MFCC40")
        if abs(float(row.get("frame_hop_s") or 0.0) - FRAME_HOP_S) > 1e-9:
            raise ValueError("Scorer cached feature row requires a 20ms frame hop")
        feature_path = Path(str(row.get("feature_path") or ""))
        if not feature_path.is_file():
            raise ValueError(f"Scorer cached feature file is missing: {feature_path}")
        features[source_id] = row
    if set(features) != set(source_by_id):
        raise ValueError("Scorer feature identities do not match corrected canonical")

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for source in sources:
        source_id = str(source["source_id"])
        feature = features[source_id]
        frame_count = int(feature["frame_count"])
        labels = canonical_frame_labels(
            source, frame_count=frame_count, frame_hop_s=FRAME_HOP_S
        )
        weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0).astype(
            np.float32
        )
        label_path = labels_dir / f"{_safe_id(source_id)}.labels.npz"
        np.savez_compressed(
            label_path,
            canonical_labels=labels,
            frame_weights=weights,
            canonical_label_schema=np.asarray([CANONICAL_LABEL_SCHEMA]),
            boundary_serialization_contract_id=np.asarray(
                [ACOUSTIC_BINARY_V12_CONTRACT.contract_id]
            ),
            diagnostic_only=np.asarray([True]),
        )
        label_counts.update(
            background=int(np.sum(labels == CANONICAL_LABELS["background"])),
            speech=int(np.sum(labels == CANONICAL_LABELS["speech"])),
            unsure=int(np.sum(labels == CANONICAL_LABELS["unsure"])),
        )
        rows.append(
            {
                "schema": DIAGNOSTIC_ROW_SCHEMA,
                "diagnostic_only": True,
                "training_manifest_allowed": False,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "audio": source["audio"],
                "row_role": source["row_role"],
                "partition": source["partition"],
                "core_ids": list(source.get("core_ids") or ()),
                "background_id": str(source.get("background_id") or ""),
                "background_source_ids": list(
                    source.get("background_source_ids") or ()
                ),
                "input_distribution": SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT[
                    "input_distribution"
                ],
                "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
                "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "feature_path": str(feature["feature_path"]),
                "feature_cache_key": str(feature.get("cache_key") or ""),
                "label_path": str(label_path),
                "frame_count": frame_count,
            }
        )
    diagnostic_manifest = output_dir / "diagnostic_manifest.jsonl"
    _write_jsonl(diagnostic_manifest, rows)
    result = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "canonical_sources": str(canonical_sources),
        "canonical_sources_sha256": _sha256(canonical_sources),
        "audio_manifest": str(audio_manifest),
        "audio_manifest_sha256": _sha256(audio_manifest),
        "original_feature_audio_manifest": str(original_audio_manifest),
        "audio_manifest_byte_identical": True,
        "feature_manifest": str(feature_manifest),
        "feature_manifest_sha256": _sha256(feature_manifest),
        "feature_summary": str(feature_summary),
        "feature_summary_sha256": _sha256(feature_summary),
        "row_count": len(rows),
        "canonical_frame_counts": dict(label_counts),
        "diagnostic_manifest": str(diagnostic_manifest),
        "feature_values_reused": True,
        "checkpoint_rescore_allowed": True,
        "training_manifest_allowed": False,
        "promotion_ready": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--audio-manifest", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--feature-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_manifest(
                canonical_sources=Path(args.canonical_sources),
                audio_manifest=Path(args.audio_manifest),
                feature_manifest=Path(args.feature_manifest),
                feature_summary=Path(args.feature_summary),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
