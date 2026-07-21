#!/usr/bin/env python3
"""Bind the Scorer v10 corrected-r5 canonical set to an existing raw cache."""
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


R5_SUMMARY_SCHEMA = "speech_scorer_v10_corrected_canonical_r5_summary_v1"
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
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
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


def _frame_count(sample_count: int, sample_rate: int) -> int:
    duration_s = float(sample_count) / float(sample_rate)
    return int(math.ceil((duration_s / FRAME_HOP_S) - 1e-9))


def _validate_memory(snapshot: Mapping[str, Any]) -> None:
    used = float(snapshot.get("physical_ram_used_mb") or 0.0)
    budget = float(snapshot.get("physical_ram_budget_mb") or 0.0)
    if budget > 0.0 and used > budget:
        raise MemoryError(
            "Scorer v10 cache audit exceeded the 0.95 physical RAM budget: "
            f"used_mb={used:.1f} budget_mb={budget:.1f}"
        )


def _require_unique(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        identity = str(row.get(key) or "")
        if not identity or identity in indexed:
            raise ValueError(f"Scorer v10 cache requires unique non-empty {key}")
        indexed[identity] = row
    return indexed


def _validate_feature_summary(summary: Mapping[str, Any], source_count: int) -> FeatureConfig:
    if int(summary.get("records") or 0) != source_count:
        raise ValueError("raw feature summary record count does not match corrected r5")
    if int(summary.get("examples") or 0) != source_count:
        raise ValueError("raw feature summary example count does not match corrected r5")
    if int(summary.get("cached") or 0) != source_count:
        raise ValueError("raw feature cache is incomplete")
    if int(summary.get("errors") or 0) or int(summary.get("skipped") or 0):
        raise ValueError("raw feature cache contains errors or skipped sources")
    if summary.get("compressed") is not False:
        raise ValueError("Scorer v10 audited cache must use the uncompressed raw cache")
    if int(summary.get("ptm_window_batch_size") or 0) != 1:
        raise ValueError("Scorer v10 cache must use singleton PTM window forwards")
    if abs(float(summary.get("feature_window_s") or 0.0) - 30.0) > 1e-9:
        raise ValueError("Scorer v10 cache must preserve the complete 30-second window")
    if abs(float(summary.get("feature_overlap_s") or 0.0) - 5.0) > 1e-9:
        raise ValueError("Scorer v10 cache must use the audited 5-second overlap")

    config_payload = dict(summary.get("config") or {})
    config = FeatureConfig(**config_payload)
    if config.ptm != QWEN_ASR_17B_REPO_ID:
        raise ValueError("Scorer v10 cache must use the 1.7B PTM")
    if config.feature_dim is not None:
        raise ValueError("Scorer v10 cache must retain raw PTM2048")
    if config.device != "cuda" or config.dtype != "bfloat16":
        raise ValueError("Scorer v10 cache requires the audited CUDA/bfloat16 extraction")
    if config.download is not False:
        raise ValueError("Scorer v10 cache must be bound to the local 1.7B model")
    if config.attention != "sdpa" or config.language != "Japanese":
        raise ValueError("Scorer v10 cache extractor configuration changed")
    if config.n_mfcc != SPEECH_ISLAND_SCORER_V10_MFCC_DIM or config.n_fft != 400:
        raise ValueError("Scorer v10 cache requires MFCC40/n_fft=400")
    if abs(config.frame_hop_s - FRAME_HOP_S) > 1e-9:
        raise ValueError("Scorer v10 cache requires a 20 ms frame hop")
    if abs(config.window_s - 30.0) > 1e-9 or abs(config.overlap_s - 5.0) > 1e-9:
        raise ValueError("Scorer v10 FeatureConfig window contract changed")
    return config


def audit_feature_cache(
    *,
    r5_summary_path: Path,
    feature_summary_path: Path,
    feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    memory_before = runtime_memory_snapshot(require_shared_vram=False)
    _validate_memory(memory_before)

    r5 = _json(r5_summary_path)
    if r5.get("schema") != R5_SUMMARY_SCHEMA:
        raise ValueError("Scorer v10 cache audit requires corrected-r5 summary")
    if r5.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("Scorer v10 corrected-r5 uses another boundary contract")
    required_r5_flags = {
        "audio_bytes_changed": False,
        "source_identity_changed": False,
        "partition_identity_changed": False,
        "replacement_manual_review_complete": True,
        "replacement_resolution_pass": True,
        "feature_cache_labels_ready": True,
        "training_manifest_ready": False,
        "checkpoint_promotion_authorized": False,
    }
    for key, expected in required_r5_flags.items():
        if r5.get(key) is not expected:
            raise ValueError(f"Scorer v10 corrected-r5 flag {key} is not {expected}")

    canonical_sources_path = _resolve(str(r5.get("canonical_sources") or ""))
    audio_manifest_path = _resolve(str(r5.get("audio_manifest") or ""))
    feature_labels_path = _resolve(str(r5.get("feature_cache_labels") or ""))
    for path, field in (
        (canonical_sources_path, "canonical_sources_sha256"),
        (audio_manifest_path, "audio_manifest_sha256"),
        (feature_labels_path, "feature_cache_labels_sha256"),
    ):
        if _sha256(path) != str(r5.get(field) or ""):
            raise ValueError(f"Scorer v10 corrected-r5 {field} mismatch")

    sources = _rows(canonical_sources_path)
    source_map = _require_unique(sources, "source_id")
    for source in sources:
        if source.get("schema") != CANONICAL_SOURCE_SCHEMA:
            raise ValueError("Scorer v10 canonical source schema changed")
        if source.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError("Scorer v10 canonical source uses another contract")
        if source.get("canonical_label_schema") != CANONICAL_LABEL_SCHEMA:
            raise ValueError("Scorer v10 canonical label schema changed")

    current_audio_manifest = _require_unique(_json(audio_manifest_path), "audio_id")
    if set(current_audio_manifest) != set(source_map):
        raise ValueError("corrected-r5 audio manifest identity mismatch")

    feature_summary = _json(feature_summary_path)
    feature_config = _validate_feature_summary(feature_summary, len(source_map))
    configured_manifest = _resolve(str(feature_summary.get("feature_manifest") or ""))
    if configured_manifest != feature_manifest_path.resolve():
        raise ValueError("feature summary is bound to another feature manifest")
    source_audio_manifest_path = _resolve(str(feature_summary.get("source_manifest") or ""))
    source_labels_path = _resolve(str(feature_summary.get("labels") or ""))
    source_audio_manifest = _require_unique(_json(source_audio_manifest_path), "audio_id")
    if set(source_audio_manifest) != set(source_map):
        raise ValueError("source feature-cache audio identity differs from corrected-r5")

    feature_rows = _rows(feature_manifest_path)
    feature_map = _require_unique(feature_rows, "audio_id")
    if set(feature_map) != set(source_map):
        raise ValueError("raw feature manifest identity differs from corrected-r5")

    feature_config_payload = {
        "schema": SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "config": dict(feature_summary.get("config") or {}),
        "feature_window_s": float(feature_summary["feature_window_s"]),
        "feature_overlap_s": float(feature_summary["feature_overlap_s"]),
        "ptm_window_batch_size": int(feature_summary["ptm_window_batch_size"]),
        "compressed": bool(feature_summary["compressed"]),
    }
    feature_config_sha256 = _canonical_digest(feature_config_payload)

    signed_rows: list[dict[str, Any]] = []
    audio_signature_rows: list[dict[str, Any]] = []
    feature_signature_rows: list[dict[str, Any]] = []
    total_audio_bytes = 0
    total_feature_bytes = 0
    for source_id in sorted(source_map):
        source = source_map[source_id]
        current_audio = current_audio_manifest[source_id]
        source_audio = source_audio_manifest[source_id]
        feature = feature_map[source_id]

        if str(current_audio.get("audio") or "") != str(source.get("audio") or ""):
            raise ValueError(f"corrected-r5 audio path mismatch: {source_id}")
        if str(current_audio.get("partition") or "") != str(source.get("partition") or ""):
            raise ValueError(f"corrected-r5 partition mismatch: {source_id}")
        if str(source_audio.get("audio") or "") != str(source.get("audio") or ""):
            raise ValueError(f"feature source audio path changed: {source_id}")
        if str(source_audio.get("partition") or "") != str(source.get("partition") or ""):
            raise ValueError(f"feature source partition changed: {source_id}")

        audio_path = _resolve(str(source.get("audio") or ""))
        if _resolve(str(feature.get("audio_path") or "")) != audio_path:
            raise ValueError(f"raw feature audio path changed: {source_id}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Scorer v10 audio is missing: {audio_path}")
        if cache_key_for_audio(audio_path=audio_path, config=feature_config) != str(
            feature.get("cache_key") or ""
        ):
            raise ValueError(f"raw feature extraction cache key no longer matches: {source_id}")

        audio_info = sf.info(str(audio_path))
        sample_rate = int(audio_info.samplerate)
        sample_count = int(audio_info.frames)
        if sample_rate != int(source.get("sample_rate") or 0):
            raise ValueError(f"audio sample rate changed: {source_id}")
        if sample_count != int(source.get("sample_count") or 0):
            raise ValueError(f"audio sample count changed: {source_id}")
        expected_frames = _frame_count(sample_count, sample_rate)
        audio_size = audio_path.stat().st_size
        audio_sha256 = _sha256(audio_path)
        total_audio_bytes += audio_size

        feature_path = _resolve(str(feature.get("feature_path") or ""))
        if not feature_path.is_file():
            raise FileNotFoundError(f"Scorer v10 feature file is missing: {feature_path}")
        feature_size = feature_path.stat().st_size
        feature_sha256 = _sha256(feature_path)
        total_feature_bytes += feature_size
        with np.load(feature_path, allow_pickle=False) as payload:
            if set(payload.files) != {"ptm", "mfcc", "duration_s", "sample_rate"}:
                raise ValueError(f"raw feature keys changed: {source_id}")
            ptm = np.asarray(payload["ptm"])
            mfcc = np.asarray(payload["mfcc"])
            if ptm.dtype != np.float32 or mfcc.dtype != np.float32:
                raise ValueError(f"raw feature dtype changed: {source_id}")
            if ptm.shape != (expected_frames, SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM):
                raise ValueError(f"raw PTM shape mismatch: {source_id}")
            if mfcc.shape != (expected_frames, SPEECH_ISLAND_SCORER_V10_MFCC_DIM):
                raise ValueError(f"MFCC shape mismatch: {source_id}")
            if not np.isfinite(ptm).all() or not np.isfinite(mfcc).all():
                raise ValueError(f"raw feature contains non-finite values: {source_id}")
            cached_duration = np.asarray(payload["duration_s"])
            cached_rate = np.asarray(payload["sample_rate"])
            if cached_duration.dtype != np.float32 or cached_duration.shape != (1,):
                raise ValueError(f"raw feature duration dtype/shape mismatch: {source_id}")
            if cached_rate.dtype != np.int32 or cached_rate.shape != (1,):
                raise ValueError(f"raw feature sample rate dtype/shape mismatch: {source_id}")
            cached_duration_s = float(cached_duration[0])
            cached_sample_rate = int(cached_rate[0])
            expected_duration_s = sample_count / sample_rate
            duration_tolerance = max(
                1e-7, float(abs(np.spacing(np.float32(expected_duration_s))))
            )
            if abs(cached_duration_s - expected_duration_s) > duration_tolerance:
                raise ValueError(f"raw feature duration metadata mismatch: {source_id}")
            if cached_sample_rate != sample_rate:
                raise ValueError(f"raw feature sample rate metadata mismatch: {source_id}")

        if int(feature.get("frame_count") or 0) != expected_frames:
            raise ValueError(f"feature manifest frame count mismatch: {source_id}")
        if int(feature.get("ptm_dim") or 0) != SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM:
            raise ValueError(f"feature manifest PTM dimension mismatch: {source_id}")
        if int(feature.get("mfcc_dim") or 0) != SPEECH_ISLAND_SCORER_V10_MFCC_DIM:
            raise ValueError(f"feature manifest MFCC dimension mismatch: {source_id}")
        if str(feature.get("ptm") or "") != QWEN_ASR_17B_REPO_ID:
            raise ValueError(f"feature manifest PTM repository mismatch: {source_id}")
        if abs(float(feature.get("frame_hop_s") or 0.0) - FRAME_HOP_S) > 1e-9:
            raise ValueError(f"feature manifest frame hop mismatch: {source_id}")
        if abs(float(feature.get("feature_window_s") or 0.0) - 30.0) > 1e-9:
            raise ValueError(f"feature manifest window mismatch: {source_id}")
        if abs(float(feature.get("feature_overlap_s") or 0.0) - 5.0) > 1e-9:
            raise ValueError(f"feature manifest overlap mismatch: {source_id}")
        if abs(float(feature.get("feature_coverage_ratio") or 0.0) - 1.0) > 1e-9:
            raise ValueError(f"feature manifest coverage mismatch: {source_id}")
        if abs(float(feature.get("duration_s") or 0.0) - sample_count / sample_rate) > 1e-9:
            raise ValueError(f"feature manifest duration mismatch: {source_id}")

        audio_signature_row = {
            "source_id": source_id,
            "path": _display(audio_path),
            "sha256": audio_sha256,
            "size_bytes": audio_size,
            "sample_rate": sample_rate,
            "sample_count": sample_count,
        }
        feature_signature_row = {
            "source_id": source_id,
            "path": _display(feature_path),
            "sha256": feature_sha256,
            "size_bytes": feature_size,
            "frame_count": expected_frames,
            "ptm_dim": SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
            "mfcc_dim": SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
        }
        audio_signature_rows.append(audio_signature_row)
        feature_signature_rows.append(feature_signature_row)
        binding_payload = {
            "audio": audio_signature_row,
            "feature": feature_signature_row,
            "feature_config_sha256": feature_config_sha256,
            "cache_key": str(feature["cache_key"]),
        }
        signed_rows.append(
            {
                "schema": SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "feature_extractor_schema": (
                    SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA
                ),
                "feature_config_sha256": feature_config_sha256,
                "source_id": source_id,
                "audio_path": _display(audio_path),
                "audio_sha256": audio_sha256,
                "audio_size_bytes": audio_size,
                "audio_sample_rate": sample_rate,
                "audio_sample_count": sample_count,
                "cache_key": str(feature["cache_key"]),
                "feature_path": _display(feature_path),
                "feature_sha256": feature_sha256,
                "feature_size_bytes": feature_size,
                "frame_count": expected_frames,
                "frame_hop_s": FRAME_HOP_S,
                "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "ptm_dim": SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
                "mfcc_dim": SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
                "feature_window_s": 30.0,
                "feature_overlap_s": 5.0,
                "feature_window_count": int(feature["feature_window_count"]),
                "feature_coverage_ratio": 1.0,
                "cache_binding_sha256": _canonical_digest(binding_payload),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    signed_manifest_path = output_dir / "signed_feature_manifest.jsonl"
    _write_rows(signed_manifest_path, signed_rows)
    signed_manifest_sha256 = _sha256(signed_manifest_path)
    audio_content_signature = _aggregate_digest(audio_signature_rows)
    feature_content_signature = _aggregate_digest(feature_signature_rows)
    cache_binding_signature = _aggregate_digest(signed_rows)

    gate = {
        "schema": SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "feature_extractor_schema": (
            SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA
        ),
        "r5_summary": _display(r5_summary_path),
        "r5_summary_sha256": _sha256(r5_summary_path),
        "canonical_sources": _display(canonical_sources_path),
        "canonical_sources_sha256": _sha256(canonical_sources_path),
        "audio_manifest": _display(audio_manifest_path),
        "audio_manifest_sha256": _sha256(audio_manifest_path),
        "feature_cache_labels": _display(feature_labels_path),
        "feature_cache_labels_sha256": _sha256(feature_labels_path),
        "source_feature_summary": _display(feature_summary_path),
        "source_feature_summary_sha256": _sha256(feature_summary_path),
        "source_feature_manifest": _display(feature_manifest_path),
        "source_feature_manifest_sha256": _sha256(feature_manifest_path),
        "source_feature_audio_manifest": _display(source_audio_manifest_path),
        "source_feature_audio_manifest_sha256": _sha256(source_audio_manifest_path),
        "source_feature_labels": _display(source_labels_path),
        "source_feature_labels_sha256": _sha256(source_labels_path),
        "signed_feature_manifest": _display(signed_manifest_path),
        "signed_feature_manifest_sha256": signed_manifest_sha256,
        "feature_config": feature_config_payload,
        "feature_config_sha256": feature_config_sha256,
        "audio_content_signature": audio_content_signature,
        "feature_content_signature": feature_content_signature,
        "cache_binding_signature": cache_binding_signature,
        "source_count": len(signed_rows),
        "total_audio_bytes": total_audio_bytes,
        "total_feature_bytes": total_feature_bytes,
        "extraction_cache_key_replay_count": len(signed_rows),
        "audio_content_hash_count": len(signed_rows),
        "feature_content_hash_count": len(signed_rows),
        "array_shape_dtype_finite_check_count": len(signed_rows),
        "extraction_time_content_sha256_available": False,
        "cache_reuse_basis": (
            "extraction_stat_config_key_replay_plus_current_audio_and_feature_sha256_v1"
        ),
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
        "schema": "speech_scorer_v10_r5_feature_cache_audit_summary_v1",
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
    parser.add_argument("--r5-summary", required=True)
    parser.add_argument("--feature-summary", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = audit_feature_cache(
        r5_summary_path=Path(args.r5_summary),
        feature_summary_path=Path(args.feature_summary),
        feature_manifest_path=Path(args.feature_manifest),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False))
