#!/usr/bin/env python3
"""Train the learned BoundaryProposalScorer on exact hardmix boundaries.

Dense per-frame supervision: frames within ``--boundary-radius-frames`` of any
``semantic_split_boundaries`` truth time are positive. The model only needs
high recall after the split peak decode (quantile floors + NMS); accept/reject
belongs to the Semantic Split verifier. Gate the result with
``build_runtime_semantic_split_dataset.py --proposal-checkpoint`` coverage,
not frame F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (SRC_ROOT, PROJECT_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from asr.backends.qwen import qwen_asr_repo_tag  # noqa: E402
from boundary.backbones import SpeechIslandSequenceClassifier  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.dataset import read_jsonl  # noqa: E402
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.ja.model import checkpoint_sha256  # noqa: E402
from boundary.ja.proposal import build_boundary_proposal_checkpoint  # noqa: E402
from tools.boundary.ja.build_runtime_semantic_split_dataset import (  # noqa: E402
    semantic_split_truth_boundaries,
)
from tools.boundary.ja.label_semantic_source_candidates_with_omni import (  # noqa: E402
    learned_frame_embeddings,
    load_task_aware_projection,
)


def boundary_target_frames(
    *,
    boundary_times_s: list[float],
    frame_count: int,
    frame_hop_s: float,
    radius_frames: int,
) -> np.ndarray:
    targets = np.zeros(frame_count, dtype=np.float32)
    for time_s in boundary_times_s:
        center = int(round(float(time_s) / frame_hop_s))
        lower = max(0, center - radius_frames)
        upper = min(frame_count, center + radius_frames + 1)
        if upper > lower:
            targets[lower:upper] = 1.0
    return targets


def _training_arrays(
    row,
    records,
    *,
    ptm_dim: int,
    radius_frames: int,
    projection_mean: np.ndarray | None = None,
    projection_components: np.ndarray | None = None,
):
    record = records[int(row["label_index"])]
    ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
    total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
    ptm_values = np.asarray(ptm[:total], dtype=np.float32)
    if projection_mean is not None and projection_components is not None:
        ptm_values = learned_frame_embeddings(
            ptm=ptm_values,
            projection_mean=projection_mean,
            projection_components=projection_components,
        )
    else:
        ptm_values = ptm_values[:, :ptm_dim]
    if int(ptm_values.shape[1]) != int(ptm_dim):
        raise ValueError(
            f"proposal PTM feature dim {ptm_values.shape[1]} != requested {ptm_dim}"
        )
    features = np.concatenate((ptm_values, mfcc[:total]), axis=1).astype(np.float32)
    boundaries = [
        float(item["time_s"])
        for item in semantic_split_truth_boundaries(
            dict(record.boundary_metadata or {})
        )
        if 0.0 < float(item["time_s"]) < record.duration_s
    ]
    targets = boundary_target_frames(
        boundary_times_s=boundaries,
        frame_count=total,
        frame_hop_s=record.frame_hop_s,
        radius_frames=radius_frames,
    )
    partition = str((record.boundary_metadata or {}).get("source_partition") or "train")
    return features, targets, partition


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    apply_vram_safety_cap()
    records = read_jsonl(Path(args.labels))
    with Path(args.feature_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if records[int(row["label_index"])].boundary_metadata]
    if not rows:
        raise ValueError("no manifest rows with boundary metadata")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for row in rows:
        record = records[int(row["label_index"])]
        partition = str(
            (record.boundary_metadata or {}).get("source_partition") or "train"
        )
        (val_rows if partition == "val" else train_rows).append(row)
    if not val_rows:
        raise ValueError("no val-partition rows for proposal training")
    ptm_dim = int(args.ptm_dim)
    projection_mean: np.ndarray | None = None
    projection_components: np.ndarray | None = None
    projection_digest = ""
    projection_file_sha256 = ""
    projection_path = ""
    if args.ptm_projection:
        projection_path = str(Path(args.ptm_projection))
        projection_mean, projection_components, projection_digest = (
            load_task_aware_projection(Path(args.ptm_projection))
        )
        projection_file_sha256 = checkpoint_sha256(Path(args.ptm_projection))
        if int(projection_components.shape[0]) != ptm_dim:
            raise ValueError(
                "--ptm-dim must match the learned projection output dimension"
            )
    mfcc_dim = int(rows[0]["mfcc_dim"])
    input_dim = ptm_dim + mfcc_dim

    # Weighted normalization over a train subsample keeps startup fast.
    sample_rows = train_rows[:: max(1, len(train_rows) // 512)]
    feature_sum = np.zeros(input_dim, dtype=np.float64)
    square_sum = np.zeros(input_dim, dtype=np.float64)
    frame_total = 0
    for row in sample_rows:
        features, _targets, _partition = _training_arrays(
            row,
            records,
            ptm_dim=ptm_dim,
            radius_frames=args.boundary_radius_frames,
            projection_mean=projection_mean,
            projection_components=projection_components,
        )
        feature_sum += features.sum(axis=0)
        square_sum += np.square(features).sum(axis=0)
        frame_total += features.shape[0]
    mean = feature_sum / max(frame_total, 1)
    std = np.sqrt(np.maximum(square_sum / max(frame_total, 1) - np.square(mean), 1e-6))
    normalization = {
        "feature_mean": mean.astype(np.float32).tolist(),
        "feature_std": std.astype(np.float32).tolist(),
    }
    mean32 = mean.astype(np.float32)
    std32 = std.astype(np.float32)

    model_config = {
        "ptm_dim": ptm_dim,
        "mfcc_dim": mfcc_dim,
        "input_dim": input_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "state_size": 32,
        "num_heads": 4,
        "n_groups": 2,
        "chunk_size": 8,
        "conv_kernel": 4,
        "bidirectional": True,
        "model_arch": "v1-boundary-proposal",
        "output_dim": 1,
    }
    device = torch.device(args.device)
    model = SpeechIslandSequenceClassifier(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_dim=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    for step in range(1, args.max_steps + 1):
        row = train_rows[int(rng.integers(0, len(train_rows)))]
        features, targets, _partition = _training_arrays(
            row,
            records,
            ptm_dim=ptm_dim,
            radius_frames=args.boundary_radius_frames,
            projection_mean=projection_mean,
            projection_components=projection_components,
        )
        if features.shape[0] > args.max_train_frames:
            start = int(rng.integers(0, features.shape[0] - args.max_train_frames + 1))
            features = features[start : start + args.max_train_frames]
            targets = targets[start : start + args.max_train_frames]
        features = (features - mean32) / std32
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        target_tensor = (
            torch.from_numpy(targets).unsqueeze(0).unsqueeze(-1).to(device)
        )
        logits = model(feature_tensor)
        raw = F.binary_cross_entropy_with_logits(logits, target_tensor, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = torch.where(target_tensor > 0.5, probabilities, 1.0 - probabilities)
        weights = torch.where(
            target_tensor > 0.5,
            torch.full_like(target_tensor, args.positive_weight),
            torch.ones_like(target_tensor),
        )
        loss = (raw * torch.pow(1.0 - pt, args.focal_gamma) * weights).sum()
        loss = loss / weights.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and step % args.log_every == 0:
            print(
                f"proposal_train={step}/{args.max_steps} "
                f"loss={losses[-1]:.6f} avg={np.mean(losses[-args.log_every:]):.6f}",
                flush=True,
            )

    # Light frame-level validation; the real gate is decoded proposal coverage.
    model.eval()
    tp = fp = fn = 0
    with torch.inference_mode():
        for row in val_rows[: args.max_eval_windows]:
            features, targets, _partition = _training_arrays(
                row,
                records,
                ptm_dim=ptm_dim,
                radius_frames=args.boundary_radius_frames,
                projection_mean=projection_mean,
                projection_components=projection_components,
            )
            features = (features - mean32) / std32
            logits = model(torch.from_numpy(features).unsqueeze(0).to(device))
            predicted = (
                torch.sigmoid(logits)[0, :, 0].cpu().numpy() >= args.eval_threshold
            )
            truth = targets >= 0.5
            tp += int((predicted & truth).sum())
            fp += int((predicted & ~truth).sum())
            fn += int((~predicted & truth).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ptm_repo_id = str(rows[0].get("ptm") or "")
    checkpoint_path = output_dir / (
        f"boundary_proposal_scorer_v1.{qwen_asr_repo_tag(ptm_repo_id)}.pt"
    )
    torch.save(
        build_boundary_proposal_checkpoint(
            model=model,
            model_config=model_config,
            normalization=normalization,
            metadata={
                "ptm_repo_id": ptm_repo_id,
                "labels": str(Path(args.labels)),
                "feature_manifest": str(Path(args.feature_manifest)),
                "trained_steps": args.max_steps,
                "boundary_radius_frames": args.boundary_radius_frames,
                "positive_weight": args.positive_weight,
                "focal_gamma": args.focal_gamma,
                "ptm_projection_path": projection_path,
                "ptm_projection_digest": projection_digest,
                "ptm_projection_file_sha256": projection_file_sha256,
                "ptm_projection_contract": (
                    "task_aware_linear_2048_to_128"
                    if projection_path
                    else "legacy_front_slice"
                ),
            },
        ),
        checkpoint_path,
    )
    metrics = {
        "schema": "boundary_proposal_scorer_train_v1",
        "loss": float(np.mean(losses)),
        "frame_precision": precision,
        "frame_recall": recall,
        "eval_threshold": args.eval_threshold,
        "train_windows": len(train_rows),
        "val_windows": min(len(val_rows), args.max_eval_windows),
        "checkpoint": str(checkpoint_path),
        "ptm_projection_path": projection_path,
        "ptm_projection_digest": projection_digest,
        "ptm_projection_file_sha256": projection_file_sha256,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the dense BoundaryProposalScorer on hardmix truth."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument(
        "--ptm-projection",
        default="",
        help=(
            "Task-aware learned Linear(2048->128) deploy projection. When set, "
            "raw cached PTM is projected before proposal training; PCA/front-128 "
            "features are not used."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--boundary-radius-frames", type=int, default=2)
    parser.add_argument("--positive-weight", type=float, default=30.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-train-frames", type=int, default=1024)
    parser.add_argument("--max-eval-windows", type=int, default=256)
    parser.add_argument("--eval-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()
    if args.boundary_radius_frames < 0:
        parser.error("--boundary-radius-frames must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
