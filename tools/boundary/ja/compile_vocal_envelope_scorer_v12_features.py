#!/usr/bin/env python3
"""Compile Scorer v12 canonical sources and raw features into owned windows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.ja.candidate_windows import (  # noqa: E402
    CANDIDATE_CONTEXT_OVERLAP_FRAMES,
    CANDIDATE_CONTEXT_WINDOW_FRAMES,
    CANDIDATE_WINDOW_OWNERSHIP,
    plan_candidate_context_windows,
)
from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_FEATURE_MANIFEST_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    VOCAL_ENVELOPE_SCORER_V12_TRAINING_ROW_SCHEMA,
)

CONTRACT_ID = "boundary_acoustic_binary_v12"
SIGNED_SCHEMA = "vocal_envelope_scorer_v12_signed_feature_manifest_row_v1"
GATE_SCHEMA = "vocal_envelope_scorer_v12_feature_cache_gate_v1"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_feature_compile_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _binding_sha(
    *,
    source_id: str,
    audio_sha256: str,
    frame_count: int,
    ptm_sha256: str,
    mfcc_sha256: str,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "source_id": source_id,
                "audio_sha256": audio_sha256,
                "frame_count": frame_count,
                "frame_hop_s": 0.02,
                "ptm_sha256": ptm_sha256,
                "mfcc_sha256": mfcc_sha256,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "source"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{cleaned[:155]}-{digest}"


def _resolve(value: str, owner: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [owner.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[0])


def _labels(source: Mapping[str, Any]) -> np.ndarray:
    count = int(source.get("frame_count") or 0)
    result = np.full(count, VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX, dtype=np.int64)
    cursor = 0
    for span in list(source.get("canonical_spans") or ()):
        start, end = int(span["start_frame"]), int(span["end_frame"])
        if start != cursor or not (start < end <= count):
            raise ValueError(f"v12 canonical spans are not contiguous: {source.get('source_id')}")
        label = str(span.get("label") or "")
        value = 1 if label == "vocal_candidate" else 0 if label == "non_vocal_candidate" else VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX if label == "unsure" else None
        if value is None:
            raise ValueError(f"unsupported v12 canonical label: {label}")
        result[start:end] = value
        cursor = end
    if cursor != count:
        raise ValueError(f"v12 canonical misses source tail: {source.get('source_id')}")
    return result


def compile_features(*, canonical: Path, raw_manifest: Path, output_dir: Path, require_final_distribution: bool = False) -> dict[str, Any]:
    canonical = canonical.resolve()
    raw_manifest = raw_manifest.resolve()
    source_rows = _rows(canonical)
    feature_rows = _rows(raw_manifest)
    sources = {str(row["source_id"]): row for row in source_rows}
    features = {str(row["source_id"]): row for row in feature_rows}
    if len(sources) != len(source_rows) or len(features) != len(feature_rows) or set(sources) != set(features):
        raise ValueError("v12 canonical/raw feature IDs must be unique and identical")
    canonical_sha = _sha256(canonical)
    raw_sha = _sha256(raw_manifest)
    output_dir = output_dir.resolve()
    label_dir = output_dir / "source_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    signed: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    partition_sources = Counter()
    source_kinds = Counter()
    frame_counts = Counter()
    heldout_strata = Counter()
    seen_core: set[str] = set()
    partition_source_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for source_id in sorted(sources):
        source = sources[source_id]
        feature = features[source_id]
        if source.get("schema") != VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA:
            raise ValueError(f"wrong v12 canonical schema: {source_id}")
        if feature.get("schema") != VOCAL_ENVELOPE_SCORER_V12_FEATURE_MANIFEST_SCHEMA:
            raise ValueError(f"wrong v12 feature manifest schema: {source_id}")
        if source.get("boundary_serialization_contract_id") != CONTRACT_ID or feature.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError(f"wrong central contract: {source_id}")
        if source.get("canonical_label_schema") != VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA:
            raise ValueError(f"wrong v12 canonical label schema: {source_id}")
        if source.get("training_manifest_allowed") is not True:
            raise ValueError(
                f"v12 canonical source is review-only and cannot train: {source_id}"
            )
        if feature.get("canonical_sources_sha256") != canonical_sha:
            raise ValueError(f"v12 raw feature canonical SHA mismatch: {source_id}")
        if int(feature.get("ptm_dim") or 0) != 2048 or int(feature.get("mfcc_dim") or 0) != 40:
            raise ValueError(f"v12 feature dimensions mismatch: {source_id}")
        if feature.get("ptm_repo_id") != QWEN_ASR_17B_REPO_ID:
            raise ValueError(f"v12 feature PTM must be exactly the 1.7B model: {source_id}")
        if feature.get("feature_extractor_schema") != VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA:
            raise ValueError(f"v12 feature extractor schema mismatch: {source_id}")
        if float(feature.get("frame_hop_s") or 0.0) != 0.02:
            raise ValueError(f"v12 feature frame hop mismatch: {source_id}")
        if int(feature.get("frame_count") or 0) != int(source.get("frame_count") or 0) or feature.get("audio_sha256") != source.get("audio_sha256"):
            raise ValueError(f"v12 source/audio/frame binding mismatch: {source_id}")
        audio_path = _resolve(str(source.get("audio") or ""), canonical)
        if _sha256(audio_path) != str(source.get("audio_sha256") or ""):
            raise ValueError(f"v12 canonical audio content changed: {source_id}")
        feature_path = _resolve(str(feature.get("feature_path") or ""), raw_manifest)
        if _sha256(feature_path) != str(feature.get("feature_sha256") or ""):
            raise ValueError(f"v12 feature SHA mismatch: {source_id}")
        labels = _labels(source)
        with np.load(feature_path, allow_pickle=False) as payload:
            ptm = np.asarray(payload["ptm"])
            mfcc = np.asarray(payload["mfcc"])
            ptm_shape = tuple(ptm.shape)
            mfcc_shape = tuple(mfcc.shape)
        if ptm_shape != (len(labels), 2048) or mfcc_shape != (len(labels), 40):
            raise ValueError(f"v12 feature array geometry mismatch: {source_id}")
        if ptm.dtype != np.float32 or mfcc.dtype != np.float32:
            raise ValueError(f"v12 feature arrays must be float32: {source_id}")
        if not np.all(np.isfinite(ptm)) or not np.all(np.isfinite(mfcc)):
            raise ValueError(f"v12 feature arrays contain non-finite values: {source_id}")
        ptm_sha = _array_sha(ptm)
        mfcc_sha = _array_sha(mfcc)
        if ptm_sha != str(feature.get("ptm_sha256") or "") or mfcc_sha != str(feature.get("mfcc_sha256") or ""):
            raise ValueError(f"v12 feature array SHA mismatch: {source_id}")
        expected_binding = _binding_sha(
            source_id=source_id,
            audio_sha256=str(source["audio_sha256"]),
            frame_count=len(labels),
            ptm_sha256=ptm_sha,
            mfcc_sha256=mfcc_sha,
        )
        if expected_binding != str(feature.get("source_audio_frame_binding_sha256") or ""):
            raise ValueError(f"v12 source/audio/frame binding SHA mismatch: {source_id}")
        partition = str(source.get("partition") or "")
        synthetic = bool(source.get("synthetic_composite", False))
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"invalid v12 partition: {source_id}")
        if synthetic and partition != "train":
            raise ValueError("v12 synthetic component sources are train-only")
        if partition in {"val", "test"} and str(source.get("source_kind") or "").startswith("synthetic"):
            raise ValueError("v12 held-out sources must be real full-source audio")
        cores = [str(value) for value in list(source.get("core_ids") or ())]
        if len(cores) != 1 or cores[0] in seen_core:
            raise ValueError(f"v12 core identity is reused: {source_id}")
        seen_core.add(cores[0])
        label_path = label_dir / f"{_safe_id(source_id)}.npz"
        np.savez(label_path, labels=labels)
        label_sha = _sha256(label_path)
        signed_row = {
            "schema": SIGNED_SCHEMA, "boundary_serialization_contract_id": CONTRACT_ID,
            "canonical_label_schema": VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
            "canonical_sources_sha256": canonical_sha, "raw_feature_manifest_sha256": raw_sha,
            "source_id": source_id, "partition": partition, "core_ids": cores,
            "source_kind": str(source.get("source_kind") or ""), "synthetic_composite": synthetic,
            "audio": source["audio"], "audio_sha256": source["audio_sha256"],
            "feature_path": str(feature_path), "feature_sha256": feature["feature_sha256"],
            "ptm_sha256": ptm_sha, "mfcc_sha256": mfcc_sha,
            "source_audio_frame_binding_sha256": feature["source_audio_frame_binding_sha256"],
            "label_path": str(label_path), "label_sha256": label_sha,
            "frame_count": len(labels), "frame_hop_s": 0.02,
            "ptm_dim": 2048, "mfcc_dim": 40, "ptm_repo_id": feature["ptm_repo_id"],
        }
        signed.append(signed_row)
        for window_index, window in enumerate(plan_candidate_context_windows(len(labels))):
            windows.append({
                "schema": VOCAL_ENVELOPE_SCORER_V12_TRAINING_ROW_SCHEMA,
                **{key: signed_row[key] for key in ("boundary_serialization_contract_id", "canonical_label_schema", "canonical_sources_sha256", "raw_feature_manifest_sha256", "source_id", "partition", "core_ids", "source_kind", "synthetic_composite", "feature_path", "feature_sha256", "label_path", "label_sha256", "frame_hop_s")},
                "row_id": f"{source_id}::window{window_index:04d}",
                "source_frame_count": len(labels),
                "window_start_frame": window.start_frame, "window_end_frame": window.end_frame,
                "owner_start_frame": window.owner_start_frame, "owner_end_frame": window.owner_end_frame,
                "owner_local_start": window.owner_local_start, "owner_local_end": window.owner_local_end,
                "context_window_frames": CANDIDATE_CONTEXT_WINDOW_FRAMES,
                "context_overlap_frames": CANDIDATE_CONTEXT_OVERLAP_FRAMES,
                "window_ownership": CANDIDATE_WINDOW_OWNERSHIP,
            })
        definite = labels[labels != VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX]
        vocal_count, non_vocal_count = int(np.sum(definite == 1)), int(np.sum(definite == 0))
        frame_counts["vocal_candidate"] += vocal_count
        frame_counts["non_vocal_candidate"] += non_vocal_count
        frame_counts["unsure"] += int(np.sum(labels == VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX))
        partition_sources[partition] += 1
        source_kinds["synthetic" if synthetic else "real"] += 1
        if partition in {"val", "test"}:
            if bool(np.all(labels == 1)):
                stratum = "all_vocal"
            elif bool(np.all(labels == 0)):
                stratum = "all_nonvocal"
            elif vocal_count and non_vocal_count:
                stratum = "mixed"
            else:
                stratum = "incomplete_control"
            heldout_strata[f"{partition}:{stratum}"] += 1
            partition_source_labels[partition][stratum] += 1
    if any(partition_sources[name] <= 0 for name in ("train", "val", "test")):
        raise ValueError("v12 training set requires non-empty train/val/test")
    if frame_counts["vocal_candidate"] <= 0 or frame_counts["non_vocal_candidate"] <= 0:
        raise ValueError("v12 training set requires both definite classes")
    train_real = sum(1 for row in signed if row["partition"] == "train" and not row["synthetic_composite"])
    train_synthetic = sum(1 for row in signed if row["partition"] == "train" and row["synthetic_composite"])
    ratio = None if train_real == 0 or train_synthetic == 0 else train_real / train_synthetic
    ratio_ok = ratio is not None and 0.8 <= ratio <= 1.25
    strata_complete = all(partition_source_labels[p][s] > 0 for p in ("val", "test") for s in ("mixed", "all_vocal", "all_nonvocal"))
    if require_final_distribution and (not ratio_ok or not strata_complete):
        raise ValueError("v12 final distribution requires ~1:1 real/synthetic train and all held-out strata")
    signed_path = output_dir / "signed_feature_manifest.jsonl"
    windows_path = output_dir / "training_windows.jsonl"
    for path, rows in ((signed_path, signed), (windows_path, windows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    signed_sha = _sha256(signed_path)
    # Bind every row to the final signed manifest after it has been written.
    rewritten = [{**row, "signed_feature_manifest_sha256": signed_sha} for row in windows]
    with windows_path.open("w", encoding="utf-8") as handle:
        for row in rewritten:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    gate = {
        "schema": GATE_SCHEMA, "boundary_serialization_contract_id": CONTRACT_ID,
        "status": "approved_for_training", "training_allowed": True,
        "canonical_sources": str(canonical), "canonical_sources_sha256": canonical_sha,
        "raw_feature_manifest": str(raw_manifest), "raw_feature_manifest_sha256": raw_sha,
        "signed_feature_manifest": str(signed_path), "signed_feature_manifest_sha256": signed_sha,
        "training_windows": str(windows_path), "training_windows_sha256": _sha256(windows_path),
        "source_count": len(signed), "window_count": len(rewritten),
        "partition_counts": dict(partition_sources), "frame_counts": dict(frame_counts),
        "train_real_source_count": train_real, "train_synthetic_source_count": train_synthetic,
        "train_real_to_synthetic_ratio": ratio if ratio is not None else "n/a",
        "train_ratio_gate": ratio_ok if ratio is not None else "n/a",
        "heldout_strata": {f"{p}:{s}": partition_source_labels[p][s] if partition_source_labels[p][s] else "n/a" for p in ("val", "test") for s in ("mixed", "all_vocal", "all_nonvocal")},
        "heldout_strata_complete": strata_complete,
        "final_distribution_required": bool(require_final_distribution),
    }
    (output_dir / "feature_cache_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"schema": SUMMARY_SCHEMA, **gate}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-final-distribution", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(compile_features(canonical=Path(args.canonical), raw_manifest=Path(args.raw_manifest), output_dir=Path(args.output_dir), require_final_distribution=args.require_final_distribution), ensure_ascii=False))
