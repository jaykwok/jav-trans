#!/usr/bin/env python3
"""Extract singleton full-source raw PTM2048+MFCC40 for Scorer v11."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.features import (  # noqa: E402
    FeatureConfig,
    build_ptm_feature_extractor,
    extract_mfcc,
    resize_feature_frames,
)
from boundary.ja.model import (  # noqa: E402
    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
    CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM,
)
from pipeline.memory_safety import (  # noqa: E402
    physical_ram_snapshot,
    reset_shared_vram_baseline,
    runtime_memory_snapshot,
)


SUMMARY_SCHEMA = "candidate_island_scorer_v11_raw_feature_extract_summary_v1"
FRAME_HOP_S = 0.02
FRAME_SAMPLES = 320


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
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


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not cleaned:
        cleaned = "source"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{cleaned[:155]}-{digest}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def align_raw_features(
    *, ptm: np.ndarray, mfcc: np.ndarray, expected_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    raw_ptm = np.asarray(ptm)
    raw_mfcc = np.asarray(mfcc)
    if raw_ptm.ndim != 2 or raw_ptm.shape[1] != CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM:
        raise ValueError(f"Scorer v11 extractor requires raw PTM2048, got {raw_ptm.shape}")
    if raw_mfcc.ndim != 2 or raw_mfcc.shape[1] != CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM:
        raise ValueError(f"Scorer v11 extractor requires MFCC40, got {raw_mfcc.shape}")
    if expected_frames <= 0 or raw_mfcc.shape[0] < expected_frames:
        raise ValueError(
            "Scorer v11 MFCC sequence does not cover canonical frames: "
            f"expected={expected_frames}, actual={raw_mfcc.shape[0]}"
        )
    aligned_ptm = resize_feature_frames(raw_ptm, expected_frames)
    aligned_mfcc = np.ascontiguousarray(raw_mfcc[:expected_frames], dtype=np.float32)
    if not np.all(np.isfinite(aligned_ptm)) or not np.all(np.isfinite(aligned_mfcc)):
        raise ValueError("Scorer v11 feature extraction produced non-finite values")
    return (
        np.ascontiguousarray(aligned_ptm, dtype=np.float32),
        aligned_mfcc,
    )


def _memory_snapshot(torch, *, stage: str) -> dict[str, Any]:
    snapshot = runtime_memory_snapshot(require_shared_vram=True)
    snapshot.update(
        stage=stage,
        cuda_allocated_mb=round(torch.cuda.memory_allocated() / (1024 * 1024), 3),
        cuda_reserved_mb=round(torch.cuda.memory_reserved() / (1024 * 1024), 3),
        cuda_peak_allocated_mb=round(
            torch.cuda.max_memory_allocated() / (1024 * 1024), 3
        ),
        cuda_peak_reserved_mb=round(
            torch.cuda.max_memory_reserved() / (1024 * 1024), 3
        ),
    )
    if float(snapshot.get("physical_ram_used_mb") or 0.0) > float(
        snapshot.get("physical_ram_budget_mb") or 0.0
    ):
        raise MemoryError(
            "Scorer v11 raw feature extraction exceeded the 0.95 physical RAM budget"
        )
    if float(snapshot.get("shared_vram_mb") or 0.0) > 0.0:
        raise MemoryError(
            "Scorer v11 raw feature extraction hit shared VRAM soft OOM: "
            f"shared_vram_mb={snapshot.get('shared_vram_mb')}"
        )
    return snapshot


def _validate_resume_row(
    row: dict[str, Any],
    *,
    canonical_sha: str,
    canonical: dict[str, Any],
    feature_dir: Path,
) -> None:
    source_id = str(canonical["source_id"])
    if row.get("schema") != CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA:
        raise ValueError(f"resume manifest has wrong schema: {source_id}")
    if row.get("source_id") != source_id:
        raise ValueError(f"resume source identity mismatch: {source_id}")
    if row.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError(f"resume central Boundary contract mismatch: {source_id}")
    if row.get("canonical_sources_sha256") != canonical_sha:
        raise ValueError(f"resume manifest is bound to another canonical: {source_id}")
    if row.get("partition") != canonical.get("partition"):
        raise ValueError(f"resume partition mismatch: {source_id}")
    if int(row.get("frame_count") or 0) != int(canonical["frame_count"]):
        raise ValueError(f"resume frame count mismatch: {source_id}")
    if row.get("feature_extractor_schema") != (
        CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA
    ):
        raise ValueError(f"resume feature extractor mismatch: {source_id}")
    if row.get("ptm_repo_id") != QWEN_ASR_17B_REPO_ID:
        raise ValueError(f"resume PTM identity mismatch: {source_id}")
    if int(row.get("ptm_dim") or 0) != CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM:
        raise ValueError(f"resume PTM width mismatch: {source_id}")
    if int(row.get("mfcc_dim") or 0) != CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM:
        raise ValueError(f"resume MFCC width mismatch: {source_id}")
    if not np.isclose(float(row.get("frame_hop_s") or 0.0), FRAME_HOP_S):
        raise ValueError(f"resume frame hop mismatch: {source_id}")
    if row.get("ptm_extraction") != "singleton_full_source_forward_v1":
        raise ValueError(f"resume PTM extraction mode mismatch: {source_id}")
    if int(row.get("ptm_batch_size") or 0) != 1:
        raise ValueError(f"resume PTM batch mismatch: {source_id}")
    if row.get("audio_sha256") != canonical.get("audio_sha256"):
        raise ValueError(f"resume audio binding mismatch: {source_id}")
    feature_path = _resolve(str(row.get("feature_path") or ""))
    if feature_path.parent != feature_dir.resolve():
        raise ValueError(f"resume feature path escaped its output directory: {source_id}")
    if not feature_path.exists() or _sha256(feature_path) != row.get("feature_sha256"):
        raise ValueError(f"resume feature content mismatch: {source_id}")
    expected_frames = int(canonical["frame_count"])
    with np.load(feature_path, allow_pickle=False) as payload:
        if payload["ptm"].shape != (
            expected_frames,
            CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM,
        ) or payload["mfcc"].shape != (
            expected_frames,
            CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
        ):
            raise ValueError(f"resume feature shape mismatch: {source_id}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if str(args.device).lower() != "cuda":
        raise ValueError("Scorer v11 formal raw feature extraction is CUDA-only")
    if str(args.dtype).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("Scorer v11 formal raw feature extraction requires bfloat16")
    if not torch.cuda.is_available():
        raise RuntimeError("Scorer v11 raw feature extraction requested CUDA but it is unavailable")
    apply_vram_safety_cap(0.95)
    canonical_path = _resolve(args.canonical_sources)
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)
    canonical_sha = _sha256(canonical_path)
    canonical_rows = _read_jsonl(canonical_path)
    canonical_by_id: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        source_id = str(row.get("source_id") or "")
        if (
            row.get("schema") != CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA
            or not source_id
            or source_id in canonical_by_id
        ):
            raise ValueError(f"invalid Scorer v11 canonical source: {source_id!r}")
        if row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"wrong central Boundary contract: {source_id}")
        canonical_by_id[source_id] = row

    selected = canonical_rows[: int(args.limit)] if args.limit is not None else canonical_rows
    selected_ids = {str(row["source_id"]) for row in selected}
    output_dir = _resolve(args.output_dir)
    feature_dir = output_dir / "features"
    manifest_path = output_dir / "raw_feature_manifest.jsonl"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"raw feature manifest already exists; pass --resume: {manifest_path}"
            )
        for row in _read_jsonl(manifest_path):
            source_id = str(row.get("source_id") or "")
            if not source_id or source_id in existing or source_id not in canonical_by_id:
                raise ValueError(f"invalid resume source identity: {source_id!r}")
            if source_id not in selected_ids:
                raise ValueError(
                    "resume manifest contains a source outside the selected --limit; "
                    f"do not shrink an existing extraction scope: {source_id}"
                )
            _validate_resume_row(
                row,
                canonical_sha=canonical_sha,
                canonical=canonical_by_id[source_id],
                feature_dir=feature_dir,
            )
            existing[source_id] = row

    config = FeatureConfig(
        ptm=QWEN_ASR_17B_REPO_ID,
        frame_hop_s=FRAME_HOP_S,
        n_mfcc=CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
        n_fft=400,
        feature_dim=None,
        device="cuda",
        dtype="bfloat16",
        model_path=str(args.model_path),
        download=False,
        attention=str(args.attention),
        language="Japanese",
    )
    started = time.monotonic()
    memory_snapshots: list[dict[str, Any]] = [
        {**physical_ram_snapshot(0.95), "stage": "preload"}
    ]
    extractor = None
    completed = len(existing)
    try:
        extractor = build_ptm_feature_extractor(config)
        first_parameter = next(extractor.model.parameters())
        if first_parameter.device.type != "cuda":
            raise RuntimeError(
                f"Scorer v11 PTM loaded on {first_parameter.device}; CPU fallback is disabled"
            )
        del first_parameter
        warmup = extractor.extract(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        if warmup.ndim != 2 or warmup.shape[1] != CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM:
            raise ValueError(f"Scorer v11 PTM warmup returned {warmup.shape}")
        del warmup
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = reset_shared_vram_baseline(required=True)
        memory_snapshots.append({**baseline, "stage": "execution_baseline"})
        torch.cuda.reset_peak_memory_stats()
        memory_snapshots.append(_memory_snapshot(torch, stage="post_load"))

        for index, canonical in enumerate(selected):
            source_id = str(canonical["source_id"])
            if source_id in existing:
                continue
            audio_path = _resolve(str(canonical.get("audio") or ""))
            if not audio_path.exists() or _sha256(audio_path) != canonical.get("audio_sha256"):
                raise ValueError(f"Scorer v11 canonical audio changed: {source_id}")
            audio, sample_rate = load_audio_16k_mono(str(audio_path))
            expected_frames = int(canonical["frame_count"])
            if sample_rate != 16000 or len(audio) != expected_frames * FRAME_SAMPLES:
                raise ValueError(f"Scorer v11 audio/frame geometry mismatch: {source_id}")
            mfcc = extract_mfcc(audio, sample_rate=sample_rate, config=config)
            ptm = extractor.extract(audio, sample_rate=sample_rate)
            aligned_ptm, aligned_mfcc = align_raw_features(
                ptm=ptm, mfcc=mfcc, expected_frames=expected_frames
            )
            feature_path = feature_dir / f"{_safe_id(source_id)}.npz"
            np.savez(
                feature_path,
                ptm=aligned_ptm,
                mfcc=aligned_mfcc,
                frame_hop_s=np.asarray([FRAME_HOP_S], dtype=np.float32),
            )
            feature_sha = _sha256(feature_path)
            row = {
                "schema": CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": source_id,
                "partition": str(canonical["partition"]),
                "feature_extractor_schema": (
                    CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA
                ),
                "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "feature_path": _display(feature_path),
                "feature_sha256": feature_sha,
                "frame_count": expected_frames,
                "frame_hop_s": FRAME_HOP_S,
                "ptm_dim": CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM,
                "mfcc_dim": CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
                "feature_dtype": "float32",
                "ptm_extraction": "singleton_full_source_forward_v1",
                "ptm_batch_size": 1,
                "audio": _display(audio_path),
                "audio_sha256": str(canonical["audio_sha256"]),
                "canonical_sources_sha256": canonical_sha,
            }
            _append_jsonl(manifest_path, row)
            existing[source_id] = row
            completed += 1
            del audio, mfcc, ptm, aligned_ptm, aligned_mfcc
            if completed % int(args.memory_log_every) == 0 or completed == len(selected):
                snapshot = _memory_snapshot(
                    torch, stage=f"source_{completed:04d}_complete"
                )
                memory_snapshots.append(snapshot)
                print(
                    json.dumps(
                        {
                            "source": source_id,
                            "completed": completed,
                            "selected": len(selected),
                            "elapsed_s": time.monotonic() - started,
                            "memory": snapshot,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if completed % int(args.summary_every) == 0:
                progress = {
                    "schema": SUMMARY_SCHEMA,
                    "canonical_sources_sha256": canonical_sha,
                    "selected_source_count": len(selected),
                    "completed_source_count": completed,
                    "complete": completed == len(selected),
                    "raw_feature_manifest": _display(manifest_path),
                    "elapsed_s": time.monotonic() - started,
                    "memory_snapshots": memory_snapshots,
                    "training_manifest_allowed": False,
                }
                summary_path.write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        if extractor is not None:
            extractor.close()
        extractor = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
        memory_snapshots.append(_memory_snapshot(torch, stage="post_release"))

    manifest_rows = _read_jsonl(manifest_path) if manifest_path.exists() else []
    complete = len(manifest_rows) == len(selected)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_sources": _display(canonical_path),
        "canonical_sources_sha256": canonical_sha,
        "selected_source_count": len(selected),
        "completed_source_count": len(manifest_rows),
        "complete": complete,
        "raw_feature_manifest": _display(manifest_path),
        "raw_feature_manifest_sha256": _sha256(manifest_path) if manifest_path.exists() else "",
        "feature_extractor_schema": CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "ptm_dim": CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM,
        "mfcc_dim": CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
        "ptm_extraction": "singleton_full_source_forward_v1",
        "ptm_batch_size": 1,
        "vram_safety_ratio": 0.95,
        "shared_vram_counted_as_available": False,
        "shared_vram_spill_policy": "soft_oom_abort",
        "elapsed_s": time.monotonic() - started,
        "memory_snapshots": memory_snapshots,
        "training_manifest_allowed": complete and args.limit is None,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--memory-log-every", type=_positive_int, default=25)
    parser.add_argument("--summary-every", type=_positive_int, default=10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
