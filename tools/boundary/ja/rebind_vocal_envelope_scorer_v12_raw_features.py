#!/usr/bin/env python3
"""Strictly rebind existing raw PTM2048/MFCC40 files to Scorer v12 sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_FEATURE_MANIFEST_SCHEMA,
)

CONTRACT_ID = "boundary_acoustic_binary_v12"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_raw_feature_rebind_summary_v1"


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


def _resolve(value: str, owner: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [owner.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[0])


def rebind(*, canonical: Path, raw_manifest: Path, output_dir: Path) -> dict[str, Any]:
    canonical = canonical.resolve()
    raw_manifest = raw_manifest.resolve()
    source_rows = _rows(canonical)
    sources = {str(row["source_id"]): row for row in source_rows}
    if len(sources) != len(source_rows) or any(not source_id for source_id in sources):
        raise ValueError("v12 canonical source IDs must be unique and non-empty")
    raw_rows: dict[str, dict[str, Any]] = {}
    for row in _rows(raw_manifest):
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in raw_rows:
            raise ValueError(f"legacy raw manifest has duplicate source_id: {source_id!r}")
        raw_rows[source_id] = row
    if not set(sources).issubset(set(raw_rows)):
        missing = sorted(set(sources) - set(raw_rows))
        raise ValueError(f"raw feature manifest is missing canonical sources: {missing[:5]}")
    ignored_extra = sorted(set(raw_rows) - set(sources))
    canonical_sha = _sha256(canonical)
    rebound: list[dict[str, Any]] = []
    for source_id in sorted(sources):
        source = sources[source_id]
        raw = raw_rows[source_id]
        if source.get("schema") != VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA:
            raise ValueError(f"wrong v12 canonical schema: {source_id}")
        if source.get("boundary_serialization_contract_id") != CONTRACT_ID or raw.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError(f"central contract mismatch: {source_id}")
        repo = str(raw.get("ptm_repo_id") or "")
        if repo != QWEN_ASR_17B_REPO_ID:
            raise ValueError(f"Scorer v12 only accepts the 1.7B raw PTM: {source_id}")
        if raw.get("feature_extractor_schema") != VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA:
            raise ValueError(f"raw feature extractor schema mismatch: {source_id}")
        if int(raw.get("ptm_dim") or 0) != 2048 or int(raw.get("mfcc_dim") or 0) != 40:
            raise ValueError(f"Scorer v12 rejects projected/legacy feature dimensions: {source_id}")
        if int(raw.get("frame_count") or 0) != int(source.get("frame_count") or 0) or float(raw.get("frame_hop_s") or 0.0) != 0.02:
            raise ValueError(f"raw feature frame geometry mismatch: {source_id}")
        if str(raw.get("audio_sha256") or "") != str(source.get("audio_sha256") or ""):
            raise ValueError(f"raw feature audio SHA mismatch: {source_id}")
        audio = _resolve(str(source.get("audio") or ""), canonical)
        if _sha256(audio) != str(source.get("audio_sha256") or ""):
            raise ValueError(f"v12 canonical audio content changed: {source_id}")
        feature = _resolve(str(raw.get("feature_path") or ""), raw_manifest)
        if str(raw.get("feature_sha256") or "") != _sha256(feature):
            raise ValueError(f"raw feature file SHA mismatch: {source_id}")
        with np.load(feature, allow_pickle=False) as payload:
            ptm = np.asarray(payload["ptm"])
            mfcc = np.asarray(payload["mfcc"])
        expected_shape = (int(source["frame_count"]), 2048)
        if ptm.shape != expected_shape or mfcc.shape != (expected_shape[0], 40):
            raise ValueError(f"raw feature array geometry mismatch: {source_id}")
        if ptm.dtype != np.float32 or mfcc.dtype != np.float32:
            raise ValueError(f"raw feature arrays must be float32: {source_id}")
        if not np.all(np.isfinite(ptm)) or not np.all(np.isfinite(mfcc)):
            raise ValueError(f"raw feature arrays contain non-finite values: {source_id}")
        ptm_sha = _array_sha(ptm)
        mfcc_sha = _array_sha(mfcc)
        if raw.get("ptm_sha256") and str(raw["ptm_sha256"]) != ptm_sha:
            raise ValueError(f"raw PTM array SHA mismatch: {source_id}")
        if raw.get("mfcc_sha256") and str(raw["mfcc_sha256"]) != mfcc_sha:
            raise ValueError(f"raw MFCC array SHA mismatch: {source_id}")
        frame_binding = hashlib.sha256(
            json.dumps({
                "source_id": source_id,
                "audio_sha256": source["audio_sha256"],
                "frame_count": source["frame_count"],
                "frame_hop_s": 0.02,
                "ptm_sha256": ptm_sha,
                "mfcc_sha256": mfcc_sha,
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if raw.get("source_audio_frame_binding_sha256") and str(raw["source_audio_frame_binding_sha256"]) != frame_binding:
            raise ValueError(f"raw source/audio/frame binding SHA mismatch: {source_id}")
        rebound.append({
            "schema": VOCAL_ENVELOPE_SCORER_V12_FEATURE_MANIFEST_SCHEMA,
            "boundary_serialization_contract_id": CONTRACT_ID,
            "source_id": source_id,
            "partition": source["partition"],
            "core_ids": source["core_ids"],
            "audio": source["audio"],
            "audio_sha256": source["audio_sha256"],
            "frame_count": source["frame_count"],
            "frame_hop_s": 0.02,
            "ptm_dim": 2048, "mfcc_dim": 40,
            "ptm_repo_id": repo,
            "feature_extractor_schema": VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA,
            "feature_path": str(feature),
            "feature_sha256": _sha256(feature),
            "ptm_sha256": ptm_sha,
            "mfcc_sha256": mfcc_sha,
            "source_audio_frame_binding_sha256": frame_binding,
            "canonical_sources_sha256": canonical_sha,
            "raw_feature_reuse": "exact_source_audio_frame_sha_rebind_v1",
            "legacy_label_reuse": False,
        })
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "raw_feature_manifest.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rebound:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "canonical": str(canonical), "canonical_sha256": canonical_sha,
        "legacy_raw_manifest": str(raw_manifest), "legacy_raw_manifest_sha256": _sha256(raw_manifest),
        "raw_feature_manifest": str(output), "raw_feature_manifest_sha256": _sha256(output),
        "source_count": len(rebound), "ptm_dim": 2048, "mfcc_dim": 40,
        "ignored_legacy_source_count": len(ignored_extra),
        "raw_feature_reuse": "exact_source_audio_frame_sha_rebind_v1",
        "legacy_label_reuse": False, "training_manifest_allowed": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(rebind(canonical=Path(args.canonical), raw_manifest=Path(args.raw_manifest), output_dir=Path(args.output_dir)), ensure_ascii=False))
