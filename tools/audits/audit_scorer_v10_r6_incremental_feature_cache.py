#!/usr/bin/env python3
"""Merge signed r5 features with fully re-extracted r6 train-only rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.features import FeatureConfig, cache_key_for_audio  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
    SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
)
from pipeline.memory_safety import runtime_memory_snapshot  # noqa: E402
from tools.boundary.ja.build_speech_island_scorer_v10_sparse_train_layout import (  # noqa: E402
    SUMMARY_SCHEMA as R6_SUMMARY_SCHEMA,
)


SUMMARY_SCHEMA = "speech_scorer_v10_r6_incremental_feature_cache_audit_summary_v1"
CANONICAL_SOURCE_SCHEMA = "speech_scorer_v10_canonical_source_v1"
CANONICAL_LABEL_SCHEMA = "speech_scorer_canonical_frames_v1"
FRAME_HOP_S = 0.02


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


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


def _index(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        identity = str(row.get(key) or "")
        if not identity or identity in result:
            raise ValueError(f"incremental feature audit requires unique {key}")
        result[identity] = row
    return result


def _frame_count(sample_count: int, sample_rate: int) -> int:
    return int(math.ceil((sample_count / sample_rate / FRAME_HOP_S) - 1e-9))


def _validate_memory(snapshot: Mapping[str, Any]) -> None:
    used = float(snapshot.get("physical_ram_used_mb") or 0.0)
    budget = float(snapshot.get("physical_ram_budget_mb") or 0.0)
    if budget > 0.0 and used > budget:
        raise MemoryError("incremental feature audit exceeded the 0.95 RAM budget")


def _feature_config(base_gate: Mapping[str, Any], changed: Mapping[str, Any]) -> tuple[dict[str, Any], str, FeatureConfig]:
    payload = dict(base_gate.get("feature_config") or {})
    digest = _canonical_digest(payload)
    if digest != str(base_gate.get("feature_config_sha256") or ""):
        raise ValueError("base feature gate configuration SHA256 mismatch")
    expected = {
        "schema": SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "config": dict(changed.get("config") or {}),
        "feature_window_s": float(changed.get("feature_window_s") or 0.0),
        "feature_overlap_s": float(changed.get("feature_overlap_s") or 0.0),
        "ptm_window_batch_size": int(changed.get("ptm_window_batch_size") or 0),
        "compressed": bool(changed.get("compressed")),
    }
    if expected != payload:
        raise ValueError("changed extraction configuration differs from the signed base")
    config = FeatureConfig(**expected["config"])
    return payload, digest, config


def _sign_changed_row(
    *,
    source: Mapping[str, Any],
    feature: Mapping[str, Any],
    config: FeatureConfig,
    feature_config_sha256: str,
) -> dict[str, Any]:
    source_id = str(source["source_id"])
    audio_path = _resolve(str(source["audio"]))
    if _resolve(str(feature.get("audio_path") or "")) != audio_path:
        raise ValueError(f"changed raw feature audio path mismatch: {source_id}")
    if cache_key_for_audio(audio_path=audio_path, config=config) != str(
        feature.get("cache_key") or ""
    ):
        raise ValueError(f"changed raw feature cache key mismatch: {source_id}")
    info = sf.info(str(audio_path))
    sample_rate = int(info.samplerate)
    sample_count = int(info.frames)
    if sample_rate != int(source["sample_rate"]) or sample_count != int(source["sample_count"]):
        raise ValueError(f"changed audio sample identity mismatch: {source_id}")
    expected_frames = _frame_count(sample_count, sample_rate)
    feature_path = _resolve(str(feature.get("feature_path") or ""))
    if not feature_path.is_file():
        raise FileNotFoundError(f"changed feature file is missing: {source_id}")
    with np.load(feature_path, allow_pickle=False) as payload:
        if set(payload.files) != {"ptm", "mfcc", "duration_s", "sample_rate"}:
            raise ValueError(f"changed feature keys mismatch: {source_id}")
        ptm = np.asarray(payload["ptm"])
        mfcc = np.asarray(payload["mfcc"])
        duration = np.asarray(payload["duration_s"])
        rate = np.asarray(payload["sample_rate"])
        if ptm.dtype != np.float32 or ptm.shape != (
            expected_frames,
            SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
        ):
            raise ValueError(f"changed PTM shape/dtype mismatch: {source_id}")
        if mfcc.dtype != np.float32 or mfcc.shape != (
            expected_frames,
            SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
        ):
            raise ValueError(f"changed MFCC shape/dtype mismatch: {source_id}")
        if not np.isfinite(ptm).all() or not np.isfinite(mfcc).all():
            raise ValueError(f"changed feature contains non-finite values: {source_id}")
        if duration.dtype != np.float32 or duration.shape != (1,):
            raise ValueError(f"changed duration metadata mismatch: {source_id}")
        if rate.dtype != np.int32 or rate.shape != (1,) or int(rate[0]) != sample_rate:
            raise ValueError(f"changed sample-rate metadata mismatch: {source_id}")
    required_manifest = {
        "frame_count": expected_frames,
        "ptm_dim": SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
        "mfcc_dim": SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
        "ptm": QWEN_ASR_17B_REPO_ID,
        "feature_window_count": 1,
    }
    for key, expected_value in required_manifest.items():
        if feature.get(key) != expected_value:
            raise ValueError(f"changed feature manifest {key} mismatch: {source_id}")
    for key, expected_value in (
        ("frame_hop_s", FRAME_HOP_S),
        ("feature_window_s", 30.0),
        ("feature_overlap_s", 5.0),
        ("feature_coverage_ratio", 1.0),
    ):
        if abs(float(feature.get(key) or 0.0) - expected_value) > 1e-9:
            raise ValueError(f"changed feature manifest {key} mismatch: {source_id}")

    audio_signature = {
        "source_id": source_id,
        "path": _display(audio_path),
        "sha256": _sha256(audio_path),
        "size_bytes": audio_path.stat().st_size,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
    }
    feature_signature = {
        "source_id": source_id,
        "path": _display(feature_path),
        "sha256": _sha256(feature_path),
        "size_bytes": feature_path.stat().st_size,
        "frame_count": expected_frames,
        "ptm_dim": SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
        "mfcc_dim": SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
    }
    binding = {
        "audio": audio_signature,
        "feature": feature_signature,
        "feature_config_sha256": feature_config_sha256,
        "cache_key": str(feature["cache_key"]),
    }
    return {
        "schema": SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "feature_extractor_schema": SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
        "feature_config_sha256": feature_config_sha256,
        "source_id": source_id,
        "audio_path": audio_signature["path"],
        "audio_sha256": audio_signature["sha256"],
        "audio_size_bytes": audio_signature["size_bytes"],
        "audio_sample_rate": sample_rate,
        "audio_sample_count": sample_count,
        "cache_key": str(feature["cache_key"]),
        "feature_path": feature_signature["path"],
        "feature_sha256": feature_signature["sha256"],
        "feature_size_bytes": feature_signature["size_bytes"],
        "frame_count": expected_frames,
        "frame_hop_s": FRAME_HOP_S,
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "ptm_dim": SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
        "mfcc_dim": SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
        "feature_window_s": 30.0,
        "feature_overlap_s": 5.0,
        "feature_window_count": int(feature["feature_window_count"]),
        "feature_coverage_ratio": 1.0,
        "cache_binding_sha256": _canonical_digest(binding),
    }


def audit(
    *,
    r6_summary_path: Path,
    base_feature_gate_path: Path,
    changed_feature_summary_path: Path,
    changed_feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    memory_before = runtime_memory_snapshot(require_shared_vram=False)
    _validate_memory(memory_before)
    r6 = _json(r6_summary_path)
    if r6.get("schema") != R6_SUMMARY_SCHEMA:
        raise ValueError("incremental feature audit requires sparse-layout r6")
    if r6.get("boundary_serialization_contract_id") != ACOUSTIC_BINARY_V12_CONTRACT.contract_id:
        raise ValueError("r6 uses another Boundary contract")
    if r6.get("audio_bytes_changed") is not True or r6.get("changed_partition") != "train":
        raise ValueError("r6 is not a train-only audio reconstruction")
    if r6.get("heldout_audio_identity_changed") is not False:
        raise ValueError("r6 changed held-out audio identity")
    canonical_path = _resolve(str(r6.get("canonical_sources") or ""))
    audio_manifest_path = _resolve(str(r6.get("audio_manifest") or ""))
    labels_path = _resolve(str(r6.get("feature_cache_labels") or ""))
    for path, field in (
        (canonical_path, "canonical_sources_sha256"),
        (audio_manifest_path, "audio_manifest_sha256"),
        (labels_path, "feature_cache_labels_sha256"),
    ):
        if _sha256(path) != str(r6.get(field) or ""):
            raise ValueError(f"r6 {field} mismatch")
    sources = _index(_rows(canonical_path), "source_id")
    if any(row.get("schema") != CANONICAL_SOURCE_SCHEMA for row in sources.values()):
        raise ValueError("r6 canonical source schema changed")
    changed_ids = {str(value) for value in r6.get("selected_source_ids") or ()}
    if len(changed_ids) != int(r6.get("selected_source_count") or -1):
        raise ValueError("r6 changed source identities are invalid")
    if any(sources[source_id].get("partition") != "train" for source_id in changed_ids):
        raise ValueError("r6 changed source escaped the train partition")

    base_gate = _json(base_feature_gate_path)
    if base_gate.get("schema") != SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA:
        raise ValueError("base feature cache gate schema changed")
    if base_gate.get("boundary_serialization_contract_id") != ACOUSTIC_BINARY_V12_CONTRACT.contract_id:
        raise ValueError("base feature gate uses another contract")
    base_manifest_path = _resolve(str(base_gate.get("signed_feature_manifest") or ""))
    if _sha256(base_manifest_path) != str(base_gate.get("signed_feature_manifest_sha256") or ""):
        raise ValueError("base signed feature manifest SHA256 mismatch")
    base_rows = _index(_rows(base_manifest_path), "source_id")
    if set(base_rows) != set(sources):
        raise ValueError("base/r6 source identities differ")

    changed_summary = _json(changed_feature_summary_path)
    if (
        int(changed_summary.get("records") or 0) != len(changed_ids)
        or int(changed_summary.get("examples") or 0) != len(changed_ids)
        or int(changed_summary.get("cached") or 0) != len(changed_ids)
        or int(changed_summary.get("errors") or 0)
        or int(changed_summary.get("skipped") or 0)
    ):
        raise ValueError("changed feature extraction is incomplete")
    configured_manifest = _resolve(str(changed_summary.get("feature_manifest") or ""))
    if configured_manifest != changed_feature_manifest_path.resolve():
        raise ValueError("changed feature summary references another manifest")
    feature_config_payload, feature_config_sha256, config = _feature_config(
        base_gate, changed_summary
    )
    changed_raw = _index(_rows(changed_feature_manifest_path), "audio_id")
    if set(changed_raw) != changed_ids:
        raise ValueError("changed feature/source identities differ")

    signed_by_id = dict(base_rows)
    for source_id in sorted(changed_ids):
        signed_by_id[source_id] = _sign_changed_row(
            source=sources[source_id],
            feature=changed_raw[source_id],
            config=config,
            feature_config_sha256=feature_config_sha256,
        )
    for source_id, row in signed_by_id.items():
        source = sources[source_id]
        if row.get("schema") != SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA:
            raise ValueError("incremental signed cache contains a legacy row")
        if row.get("boundary_serialization_contract_id") != ACOUSTIC_BINARY_V12_CONTRACT.contract_id:
            raise ValueError("incremental signed cache mixes Boundary contracts")
        if row.get("feature_config_sha256") != feature_config_sha256:
            raise ValueError("incremental signed cache mixes feature configurations")
        if _resolve(str(row.get("audio_path") or "")) != _resolve(str(source["audio"])):
            raise ValueError(f"incremental signed audio path mismatch: {source_id}")
        if int(row.get("audio_sample_count") or 0) != int(source["sample_count"]):
            raise ValueError(f"incremental signed sample count mismatch: {source_id}")
        if int(row.get("audio_sample_rate") or 0) != int(source["sample_rate"]):
            raise ValueError(f"incremental signed sample rate mismatch: {source_id}")

    signed_rows = [signed_by_id[source_id] for source_id in sorted(signed_by_id)]
    output_dir.mkdir(parents=True, exist_ok=True)
    signed_path = output_dir / "signed_feature_manifest.jsonl"
    _write_rows(signed_path, signed_rows)
    audio_signatures = [
        {
            "source_id": row["source_id"],
            "path": row["audio_path"],
            "sha256": row["audio_sha256"],
            "size_bytes": row["audio_size_bytes"],
            "sample_rate": row["audio_sample_rate"],
            "sample_count": row["audio_sample_count"],
        }
        for row in signed_rows
    ]
    feature_signatures = [
        {
            "source_id": row["source_id"],
            "path": row["feature_path"],
            "sha256": row["feature_sha256"],
            "size_bytes": row["feature_size_bytes"],
            "frame_count": row["frame_count"],
            "ptm_dim": row["ptm_dim"],
            "mfcc_dim": row["mfcc_dim"],
        }
        for row in signed_rows
    ]
    gate = {
        "schema": SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "feature_extractor_schema": SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
        "r6_summary": _display(r6_summary_path),
        "r6_summary_sha256": _sha256(r6_summary_path),
        "canonical_sources": _display(canonical_path),
        "canonical_sources_sha256": _sha256(canonical_path),
        "audio_manifest": _display(audio_manifest_path),
        "audio_manifest_sha256": _sha256(audio_manifest_path),
        "feature_cache_labels": _display(labels_path),
        "feature_cache_labels_sha256": _sha256(labels_path),
        "base_feature_gate": _display(base_feature_gate_path),
        "base_feature_gate_sha256": _sha256(base_feature_gate_path),
        "base_signed_feature_manifest": _display(base_manifest_path),
        "base_signed_feature_manifest_sha256": _sha256(base_manifest_path),
        "changed_feature_summary": _display(changed_feature_summary_path),
        "changed_feature_summary_sha256": _sha256(changed_feature_summary_path),
        "changed_feature_manifest": _display(changed_feature_manifest_path),
        "changed_feature_manifest_sha256": _sha256(changed_feature_manifest_path),
        "signed_feature_manifest": _display(signed_path),
        "signed_feature_manifest_sha256": _sha256(signed_path),
        "feature_config": feature_config_payload,
        "feature_config_sha256": feature_config_sha256,
        "audio_content_signature": _aggregate_digest(audio_signatures),
        "feature_content_signature": _aggregate_digest(feature_signatures),
        "cache_binding_signature": _aggregate_digest(signed_rows),
        "source_count": len(signed_rows),
        "changed_source_count": len(changed_ids),
        "reused_signed_source_count": len(signed_rows) - len(changed_ids),
        "changed_audio_content_hash_count": len(changed_ids),
        "changed_feature_content_hash_count": len(changed_ids),
        "changed_array_shape_dtype_finite_check_count": len(changed_ids),
        "total_audio_bytes": sum(int(row["audio_size_bytes"]) for row in signed_rows),
        "total_feature_bytes": sum(int(row["feature_size_bytes"]) for row in signed_rows),
        "cache_reuse_basis": "prior_signed_gate_plus_changed_extraction_full_hash_v1",
        "feature_cache_reuse_allowed": True,
        "training_manifest_allowed": True,
        "checkpoint_promotion_authorized": False,
    }
    gate_path = output_dir / "feature_cache_gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    memory_after = runtime_memory_snapshot(require_shared_vram=False)
    _validate_memory(memory_after)
    summary = {
        **gate,
        "schema": SUMMARY_SCHEMA,
        "feature_cache_gate": _display(gate_path),
        "feature_cache_gate_sha256": _sha256(gate_path),
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6-summary", required=True)
    parser.add_argument("--base-feature-gate", required=True)
    parser.add_argument("--changed-feature-summary", required=True)
    parser.add_argument("--changed-feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            audit(
                r6_summary_path=Path(args.r6_summary),
                base_feature_gate_path=Path(args.base_feature_gate),
                changed_feature_summary_path=Path(args.changed_feature_summary),
                changed_feature_manifest_path=Path(args.changed_feature_manifest),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
