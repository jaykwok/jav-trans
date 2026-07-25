#!/usr/bin/env python3
"""Extract singleton full-source PTM2048/MFCC40 features for Scorer v12."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.features import FeatureConfig, build_ptm_feature_extractor, extract_feature_bundle  # noqa: E402
from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_FEATURE_MANIFEST_SCHEMA,
)
from pipeline.memory_safety import reset_shared_vram_baseline, runtime_memory_snapshot  # noqa: E402

CONTRACT_ID = "boundary_acoustic_binary_v12"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_raw_feature_extract_summary_v1"
PROGRESS_SCHEMA = "vocal_envelope_scorer_v12_raw_feature_extract_progress_v1"


def _memory_snapshot(torch, device: str) -> dict[str, Any]:
    snapshot = runtime_memory_snapshot(require_shared_vram=str(device).startswith("cuda"))
    if str(device).startswith("cuda"):
        snapshot.update(
            cuda_allocated_mb=round(torch.cuda.memory_allocated() / 2**20, 3),
            cuda_reserved_mb=round(torch.cuda.memory_reserved() / 2**20, 3),
            cuda_max_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20, 3),
            cuda_max_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20, 3),
        )
    else:
        snapshot.update(shared_vram_mb=0.0, shared_vram_monitor="not_applicable_cpu_stage")
    if float(snapshot.get("physical_ram_used_mb") or 0.0) > float(snapshot.get("physical_ram_budget_mb") or float("inf")):
        raise MemoryError("Scorer v12 extraction exceeded the 95% physical RAM budget")
    return snapshot


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


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "source"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{cleaned[:155]}-{digest}"


def _binding_sha(*, source_id: str, audio_sha256: str, frame_count: int, ptm_sha256: str, mfcc_sha256: str) -> str:
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


def _resolve(value: str, owner: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [owner.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[0])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _validate_resume_row(
    row: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    canonical_sha: str,
    feature_dir: Path,
) -> None:
    source_id = str(source["source_id"])
    expected = {
        "schema": VOCAL_ENVELOPE_SCORER_V12_FEATURE_MANIFEST_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "canonical_sources_sha256": canonical_sha,
        "source_id": source_id,
        "partition": source["partition"],
        "audio_sha256": source["audio_sha256"],
        "frame_count": source["frame_count"],
        "frame_hop_s": 0.02,
        "ptm_dim": 2048,
        "mfcc_dim": 40,
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "feature_extractor_schema": VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA,
        "ptm_extraction": "singleton_full_source_forward_v1",
        "ptm_batch_size": 1,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"v12 resume {field} mismatch: {source_id}")
    if list(row.get("core_ids") or ()) != list(source.get("core_ids") or ()):
        raise ValueError(f"v12 resume core mismatch: {source_id}")
    feature_path = _resolve(
        str(row.get("feature_path") or ""),
        feature_dir.parent / "raw_feature_manifest.jsonl",
    )
    if feature_path.parent != feature_dir.resolve():
        raise ValueError(f"v12 resume feature path escaped output: {source_id}")
    if _sha256(feature_path) != str(row.get("feature_sha256") or ""):
        raise ValueError(f"v12 resume feature file SHA mismatch: {source_id}")
    with np.load(feature_path, allow_pickle=False) as payload:
        ptm = np.asarray(payload["ptm"])
        mfcc = np.asarray(payload["mfcc"])
    if ptm.shape != (int(source["frame_count"]), 2048) or mfcc.shape != (int(source["frame_count"]), 40):
        raise ValueError(f"v12 resume feature geometry mismatch: {source_id}")
    if ptm.dtype != np.float32 or mfcc.dtype != np.float32:
        raise ValueError(f"v12 resume feature dtype mismatch: {source_id}")
    ptm_sha, mfcc_sha = _array_sha(ptm), _array_sha(mfcc)
    if row.get("ptm_sha256") != ptm_sha or row.get("mfcc_sha256") != mfcc_sha:
        raise ValueError(f"v12 resume feature array SHA mismatch: {source_id}")
    binding = _binding_sha(
        source_id=source_id,
        audio_sha256=str(source["audio_sha256"]),
        frame_count=int(source["frame_count"]),
        ptm_sha256=ptm_sha,
        mfcc_sha256=mfcc_sha,
    )
    if row.get("source_audio_frame_binding_sha256") != binding:
        raise ValueError(f"v12 resume source/audio/frame binding mismatch: {source_id}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if str(args.device).lower() != "cuda":
        raise ValueError("Scorer v12 formal raw feature extraction is CUDA-only")
    if str(args.dtype).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("Scorer v12 formal raw feature extraction requires bfloat16")
    if not torch.cuda.is_available():
        raise RuntimeError("Scorer v12 raw feature extraction requires CUDA")
    apply_vram_safety_cap(0.95)
    canonical = Path(args.canonical).resolve()
    all_rows = _rows(canonical)
    canonical_by_id = {str(row.get("source_id") or ""): row for row in all_rows}
    if len(canonical_by_id) != len(all_rows) or any(not source_id for source_id in canonical_by_id):
        raise ValueError("v12 canonical source IDs must be unique and non-empty")
    rows = list(all_rows)
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError("v12 canonical is empty")
    output = Path(args.output_dir).resolve()
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "raw_feature_manifest.jsonl"
    progress_path = output / "progress.json"
    canonical_sha = _sha256(canonical)
    selected_ids = {str(row["source_id"]) for row in rows}
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        if not args.resume:
            raise FileExistsError(f"v12 raw feature manifest exists; pass --resume: {manifest_path}")
        for saved in _rows(manifest_path):
            source_id = str(saved.get("source_id") or "")
            if not source_id or source_id in existing or source_id not in selected_ids:
                raise ValueError(f"invalid v12 resume source identity: {source_id!r}")
            _validate_resume_row(
                saved,
                source=canonical_by_id[source_id],
                canonical_sha=canonical_sha,
                feature_dir=feature_dir,
            )
            existing[source_id] = saved
    config = FeatureConfig(
        ptm=QWEN_ASR_17B_REPO_ID,
        frame_hop_s=0.02,
        n_mfcc=40,
        feature_dim=2048,
        device=args.device,
        dtype=args.dtype,
        model_path=args.model_path,
        download=not args.no_download,
        attention=args.attention,
        language="Japanese",
    )
    extractor = None
    started = time.perf_counter()
    shared_peak = 0
    try:
        extractor = build_ptm_feature_extractor(config)
        first_parameter = next(extractor.model.parameters())
        if first_parameter.device.type != "cuda":
            raise RuntimeError(f"Scorer v12 PTM loaded on {first_parameter.device}; CPU fallback is disabled")
        del first_parameter
        warmup = extractor.extract(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        if warmup.ndim != 2 or warmup.shape[1] != 2048:
            raise ValueError(f"Scorer v12 PTM warmup returned {warmup.shape}")
        del warmup
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        reset_shared_vram_baseline(required=True)
        torch.cuda.reset_peak_memory_stats()
        pending = [row for row in rows if str(row.get("source_id")) not in existing]
        _write_json(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "completed": len(existing), "total": len(rows), "pending": len(pending), "memory": _memory_snapshot(torch, args.device)})
        for row in pending:
            source_id = str(row.get("source_id") or "")
            if row.get("schema") != VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA or row.get("boundary_serialization_contract_id") != CONTRACT_ID:
                raise ValueError(f"invalid v12 canonical source: {source_id}")
            audio = _resolve(str(row.get("audio") or ""), canonical)
            if _sha256(audio) != str(row.get("audio_sha256") or ""):
                raise ValueError(f"v12 source audio SHA mismatch: {source_id}")
            print(f"v12_feature={len(existing)+1}/{len(rows)} source={source_id}", flush=True)
            bundle = extract_feature_bundle(audio_path=audio, config=config, ptm_extractor=extractor)
            ptm = np.asarray(bundle["ptm"], dtype=np.float32)
            mfcc = np.asarray(bundle["mfcc"], dtype=np.float32)
            frame_count = int(row.get("frame_count") or 0)
            if ptm.shape != (frame_count, 2048) or mfcc.shape != (frame_count, 40):
                raise ValueError(f"v12 extracted frame geometry mismatch: {source_id}; ptm={ptm.shape} mfcc={mfcc.shape} expected={frame_count}")
            if not np.all(np.isfinite(ptm)) or not np.all(np.isfinite(mfcc)):
                raise ValueError(f"v12 feature extraction produced non-finite values: {source_id}")
            feature_path = feature_dir / f"{_safe_id(source_id)}.npz"
            np.savez(feature_path, ptm=ptm, mfcc=mfcc, frame_hop_s=np.asarray([0.02], dtype=np.float32))
            feature_sha = _sha256(feature_path)
            ptm_sha, mfcc_sha = _array_sha(ptm), _array_sha(mfcc)
            binding_sha = _binding_sha(source_id=source_id, audio_sha256=str(row["audio_sha256"]), frame_count=frame_count, ptm_sha256=ptm_sha, mfcc_sha256=mfcc_sha)
            result = {
                "schema": VOCAL_ENVELOPE_SCORER_V12_FEATURE_MANIFEST_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "canonical_sources_sha256": canonical_sha,
                "source_id": source_id, "partition": row["partition"], "core_ids": row["core_ids"],
                "audio": str(audio), "audio_sha256": row["audio_sha256"],
                "frame_count": frame_count, "frame_hop_s": 0.02,
                "ptm_dim": 2048, "mfcc_dim": 40, "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "feature_extractor_schema": VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA,
                "ptm_extraction": "singleton_full_source_forward_v1",
                "ptm_batch_size": 1,
                "feature_path": str(feature_path), "feature_sha256": feature_sha,
                "ptm_sha256": ptm_sha, "mfcc_sha256": mfcc_sha,
                "source_audio_frame_binding_sha256": binding_sha,
                "feature_dtype": "float32", "legacy_label_reuse": False,
            }
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            existing[source_id] = result
            del bundle, ptm, mfcc
            memory = _memory_snapshot(torch, args.device)
            shared_peak = max(shared_peak, float(memory.get("shared_vram_mb") or 0.0))
            if shared_peak > 0.0:
                raise RuntimeError("Scorer v12 raw extraction shared VRAM spill is a soft OOM")
            elapsed = time.perf_counter() - started
            rate = len(existing) / max(elapsed, 1e-9)
            eta = (len(rows) - len(existing)) / max(rate, 1e-9)
            _write_json(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "completed": len(existing), "total": len(rows), "pending": len(rows) - len(existing), "last_source_id": source_id, "elapsed_s": round(elapsed, 3), "eta_s": round(eta, 3), "memory": memory})
    finally:
        if extractor is not None:
            extractor.close()
        extractor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    post_gpu = _memory_snapshot(torch, args.device)
    summary = {
        "schema": SUMMARY_SCHEMA, "boundary_serialization_contract_id": CONTRACT_ID,
        "canonical": str(canonical), "canonical_sha256": canonical_sha,
        "raw_feature_manifest": str(manifest_path), "raw_feature_manifest_sha256": _sha256(manifest_path),
        "source_count": len(existing), "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "ptm_dim": 2048, "mfcc_dim": 40, "singleton_full_source": True,
        "shared_vram_peak_mb": shared_peak, "post_cleanup_memory": post_gpu,
        "training_manifest_allowed": len(existing) == len(rows) and int(args.limit) == 0,
    }
    _write_json(output / "summary.json", summary)
    _write_json(progress_path, {"schema": PROGRESS_SCHEMA, "status": "completed", "completed": len(existing), "total": len(rows), "pending": 0, "elapsed_s": round(time.perf_counter() - started, 3), "post_cleanup_memory": post_gpu, "summary": str(output / "summary.json")})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
