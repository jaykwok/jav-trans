#!/usr/bin/env python3
"""Recompute semantic-split candidate features for already-labeled real windows.

Candidates, labels, and runtime metadata stay untouched; only frame/scalar
feature arrays are rebuilt (multi-scale context + PTM projection need the
full-dim PTM frames, which the exported window npz files do not keep).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.features import (  # noqa: E402
    FeatureConfig,
    build_ptm_feature_extractor,
    extract_mfcc,
)
from boundary.ja.model import (  # noqa: E402
    load_speech_island_scorer_checkpoint,
    score_speech_island_probabilities,
)
from boundary.sequence_features import (  # noqa: E402
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
    load_ptm_projection,
    parse_extra_context_scales,
)
from tools.boundary.ja.build_feature_cache import (  # noqa: E402
    _combine_workflow_window_features,
    _extract_ptm_window_features,
    _workflow_window_starts,
)

BACKUP_SUFFIX = ".pre_refresh.npz"


def expected_bins(extra_context_scales: list[Mapping[str, Any]]) -> int:
    return 20 + sum(
        int(scale["left_bins"]) + int(scale["right_bins"])
        for scale in extra_context_scales
    )


def rebuild_npz_arrays(
    original: Mapping[str, np.ndarray],
    frame_rows: list[np.ndarray],
    scalar_rows: list[np.ndarray],
) -> dict[str, np.ndarray]:
    count = int(original["frame_features"].shape[0])
    if len(frame_rows) != count or len(scalar_rows) != count:
        raise ValueError(
            f"feature row count mismatch: npz has {count}, "
            f"rebuilt {len(frame_rows)} frames / {len(scalar_rows)} scalars"
        )
    arrays = {key: np.asarray(original[key]) for key in original}
    if count:
        arrays["frame_features"] = np.stack(frame_rows).astype(np.float32)
        arrays["scalar_features"] = np.stack(scalar_rows).astype(np.float32)
    return arrays


def _score_windowed(
    scorer,
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
    window_frames: int = 1000,
    overlap_frames: int = 200,
) -> np.ndarray:
    total = min(ptm.shape[0], mfcc.shape[0])
    stride = window_frames - overlap_frames
    score_sum = np.zeros(total, dtype=np.float64)
    score_count = np.zeros(total, dtype=np.float64)
    for start in range(0, total, stride):
        end = min(total, start + window_frames)
        scores = score_speech_island_probabilities(
            scorer,
            ptm=ptm[start:end],
            mfcc=mfcc[start:end],
        )
        score_sum[start:end] += scores
        score_count[start:end] += 1.0
    return np.asarray(score_sum / np.maximum(score_count, 1.0), dtype=np.float32)


def _extract_full_features(
    *,
    audio_path: str,
    extractor: Any,
    config: FeatureConfig,
    ptm_window_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    audio, sample_rate = load_audio_16k_mono(audio_path)
    window_samples = max(1, int(round(config.window_s * sample_rate)))
    windows: list[dict[str, Any]] = []
    for window_index, start_sample in enumerate(
        _workflow_window_starts(
            sample_count=len(audio),
            sample_rate=sample_rate,
            window_s=config.window_s,
            overlap_s=config.overlap_s,
        )
    ):
        end_sample = min(len(audio), start_sample + window_samples)
        if start_sample >= end_sample:
            continue
        chunk = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
        windows.append(
            {
                "window_index": window_index,
                "start_sample": int(start_sample),
                "audio": chunk,
                "mfcc": extract_mfcc(chunk, sample_rate=sample_rate, config=config),
            }
        )
    ptm_features, _ = _extract_ptm_window_features(
        ptm_extractor=extractor,
        window_audios=[window["audio"] for window in windows],
        sample_rate=sample_rate,
        ptm_window_batch_size=ptm_window_batch_size,
    )
    bundle = _combine_workflow_window_features(
        windows=windows,
        ptm_features=ptm_features,
        duration_s=len(audio) / sample_rate,
        sample_rate=sample_rate,
        config=config,
    )
    ptm = np.asarray(bundle["ptm"], dtype=np.float32)
    mfcc = np.asarray(bundle["mfcc"], dtype=np.float32)
    return ptm, mfcc, len(audio) / sample_rate


def _load_meta_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open("r", encoding="utf-8")
        if line.strip()
    ]


def run(args: argparse.Namespace) -> None:
    apply_vram_safety_cap()
    scales = parse_extra_context_scales(args.extra_context_scales)
    projection = load_ptm_projection(args.ptm_projection)
    target_bins = expected_bins(scales)
    output_ptm_dim = (
        int(projection["components"].shape[0]) if projection else args.ptm_dim
    )
    target_feature_dim = output_ptm_dim + args.n_mfcc
    scorer = load_speech_island_scorer_checkpoint(
        args.scorer_checkpoint, device=args.device
    )
    scorer_repo = str(scorer.metadata.get("ptm_repo_id") or "")
    if scorer_repo != args.ptm_repo_id:
        raise ValueError(
            f"scorer ptm_repo_id={scorer_repo!r} does not match {args.ptm_repo_id!r}"
        )
    config = FeatureConfig(
        ptm=args.ptm_repo_id,
        frame_hop_s=args.frame_hop_s,
        window_s=args.window_s,
        overlap_s=args.overlap_s,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        feature_dim=None,
        device=args.device,
        dtype=args.dtype,
        model_path=args.model_path,
        download=False,
        attention=args.attention,
        language=args.language,
    )
    extractor = build_ptm_feature_extractor(config)
    summary: dict[str, Any] = {
        "schema": "refresh_joint_split_features_v1",
        "ptm_repo_id": args.ptm_repo_id,
        "extra_context_scales": args.extra_context_scales,
        "ptm_projection": args.ptm_projection,
        "target_bins": target_bins,
        "datasets": [],
    }
    try:
        for dataset_dir in args.dataset:
            dataset = Path(dataset_dir)
            rows = _load_meta_rows(dataset / "source_windows.jsonl")
            refreshed = skipped = empty = 0
            started = time.perf_counter()
            for index, row in enumerate(rows):
                npz_path = Path(str(row["semantic_split_features"]))
                meta_path = Path(str(row["semantic_split_metadata"]))
                if not npz_path.exists() or not meta_path.exists():
                    raise FileNotFoundError(f"missing split artifacts: {npz_path}")
                with np.load(npz_path) as handle:
                    original = {key: np.asarray(handle[key]) for key in handle.files}
                frame_shape = original["frame_features"].shape
                if frame_shape[0] == 0:
                    original["frame_features"] = np.zeros(
                        (0, target_bins, target_feature_dim), dtype=np.float32
                    )
                    np.savez_compressed(npz_path, **original)
                    empty += 1
                    continue
                if (
                    int(frame_shape[1]) == target_bins
                    and int(frame_shape[2]) == target_feature_dim
                ):
                    skipped += 1
                    continue
                meta_rows = _load_meta_rows(meta_path)
                if len(meta_rows) != int(frame_shape[0]):
                    raise ValueError(
                        f"{npz_path}: metadata rows {len(meta_rows)} != "
                        f"npz rows {frame_shape[0]}"
                    )
                ptm, mfcc, duration_s = _extract_full_features(
                    audio_path=str(row["audio_wav"]),
                    extractor=extractor,
                    config=config,
                    ptm_window_batch_size=args.ptm_window_batch_size,
                )
                speech = _score_windowed(
                    scorer,
                    ptm=ptm,
                    mfcc=mfcc,
                    window_frames=max(1, int(round(args.window_s / args.frame_hop_s))),
                    overlap_frames=max(
                        0, int(round(args.overlap_s / args.frame_hop_s))
                    ),
                )
                provider = FrameSequenceFeatureProvider(
                    duration_s=min(ptm.shape[0], mfcc.shape[0]) * args.frame_hop_s,
                    frame_hop_s=args.frame_hop_s,
                    ptm=ptm,
                    mfcc=mfcc,
                    config=FrameSequenceFeatureConfig(max_ptm_dims=args.ptm_dim),
                )
                frame_rows: list[np.ndarray] = []
                scalar_rows: list[np.ndarray] = []
                for meta in meta_rows:
                    frames, scalars = provider.features_for_split_candidate(
                        core_start_s=float(meta["core_start"]),
                        core_end_s=float(meta["core_end"]),
                        candidate=meta,
                        speech_probabilities=speech,
                        left_context_s=1.6,
                        right_context_s=1.6,
                        gap_context_s=0.3,
                        left_bins=8,
                        gap_bins=4,
                        right_bins=8,
                        ptm_dim=args.ptm_dim,
                        extra_context_scales=scales,
                        ptm_projection_mean=(
                            projection["mean"] if projection else None
                        ),
                        ptm_projection_components=(
                            projection["components"] if projection else None
                        ),
                    )
                    frame_rows.append(frames)
                    scalar_rows.append(scalars)
                arrays = rebuild_npz_arrays(original, frame_rows, scalar_rows)
                backup = npz_path.with_name(npz_path.stem + BACKUP_SUFFIX)
                if not backup.exists():
                    shutil.copy2(npz_path, backup)
                np.savez_compressed(npz_path, **arrays)
                refreshed += 1
                if args.log_every and (index + 1) % args.log_every == 0:
                    rate = (index + 1) / max(time.perf_counter() - started, 1e-6)
                    print(
                        f"{dataset.name}: {index + 1}/{len(rows)} windows "
                        f"(refreshed={refreshed} skipped={skipped} "
                        f"rate={rate:.2f}/s)",
                        flush=True,
                    )
            summary["datasets"].append(
                {
                    "dataset": str(dataset),
                    "windows": len(rows),
                    "refreshed": refreshed,
                    "skipped_already_current": skipped,
                    "empty_windows": empty,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                }
            )
            print(json.dumps(summary["datasets"][-1], ensure_ascii=False), flush=True)
    finally:
        extractor.close()
    if args.summary_output:
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild semantic-split frame/scalar features in place for labeled "
            "real-window datasets (multi-scale context + PTM projection)."
        )
    )
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--ptm-repo-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ptm-projection", default="")
    parser.add_argument("--extra-context-scales", default="3.2:4,6.4:4")
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument(
        "--window-s",
        type=float,
        default=20.0,
        help="Extraction/scoring window; matches the runtime backend window_s.",
    )
    parser.add_argument(
        "--overlap-s",
        type=float,
        default=4.0,
        help="Window overlap; matches the runtime backend overlap_s.",
    )
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--ptm-window-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--summary-output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
