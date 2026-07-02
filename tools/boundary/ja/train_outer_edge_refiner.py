#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import qwen_asr_repo_tag  # noqa: E402
from boundary.outer_refiner import (  # noqa: E402
    OUTER_EDGE_FEATURE_SCHEMA,
    OuterEdgeRefinerNetwork,
    build_outer_edge_refiner_checkpoint,
)

SCALAR_NAMES = (
    "island_duration_s",
    "island_position_ratio",
    "speech_mean",
    "speech_min",
    "speech_max",
    "speech_active_ratio",
)


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    data = np.load(args.dataset)
    frames = data["frame_features"].astype(np.float32)
    scalars = data["scalar_features"].astype(np.float32)
    deltas = data["target_deltas"].astype(np.float32)
    confidence = data["confidence_targets"].astype(np.float32)
    partitions = data["partitions"].astype(str)
    train_mask = partitions == "train"
    val_mask = partitions == "val"
    if not val_mask.any():
        rng = np.random.default_rng(args.seed)
        val_mask[rng.permutation(deltas.shape[0])[: max(1, deltas.shape[0] // 10)]] = True
        train_mask = ~val_mask
    frame_mean = frames[train_mask].mean(axis=(0, 1))
    frame_std = frames[train_mask].std(axis=(0, 1))
    scalar_mean = scalars[train_mask].mean(axis=0)
    scalar_std = scalars[train_mask].std(axis=0)
    frames = (frames - frame_mean) / np.maximum(frame_std, 1e-6)
    scalars = (scalars - scalar_mean) / np.maximum(scalar_std, 1e-6)
    model_config = {
        "frame_dim": int(frames.shape[2]),
        "scalar_dim": int(scalars.shape[1]),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "state_size": 32,
        "num_heads": 4,
        "head_dim": 64,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 8,
        "bidirectional": True,
    }
    device = torch.device(args.device)
    model = OuterEdgeRefinerNetwork(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    train_indexes = np.flatnonzero(train_mask)
    losses: list[float] = []
    for step in range(args.max_steps):
        indexes = rng.choice(train_indexes, size=args.batch_size, replace=True)
        output = model(
            torch.from_numpy(frames[indexes]).to(device),
            torch.from_numpy(scalars[indexes]).to(device),
        )
        target_delta = torch.from_numpy(deltas[indexes]).to(device)
        target_confidence = torch.from_numpy(confidence[indexes]).to(device)
        delta_loss = F.smooth_l1_loss(
            output[:, :2] * target_confidence,
            target_delta * target_confidence,
            beta=0.05,
        )
        confidence_loss = F.binary_cross_entropy_with_logits(
            output[:, 2:],
            target_confidence,
        )
        loss = delta_loss + args.confidence_loss_weight * confidence_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step + 1) % args.log_every == 0:
            print(
                f"outer_edge_train={step + 1}/{args.max_steps} "
                f"loss={losses[-1]:.6f} avg_loss={np.mean(losses):.6f}",
                flush=True,
            )
    val_indexes = np.flatnonzero(val_mask)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, val_indexes.size, args.batch_size):
            indexes = val_indexes[start : start + args.batch_size]
            outputs.append(
                model(
                    torch.from_numpy(frames[indexes]).to(device),
                    torch.from_numpy(scalars[indexes]).to(device),
                ).cpu().numpy()
            )
    output = np.concatenate(outputs)
    target = deltas[val_indexes]
    conf = confidence[val_indexes] > 0.5
    error = np.abs(output[:, :2] - target)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / (
        f"outer_edge_refiner_v1.{qwen_asr_repo_tag(args.ptm_repo_id)}.pt"
    )
    torch.save(
        build_outer_edge_refiner_checkpoint(
            model=model,
            model_config=model_config,
            feature_config={
                "schema": OUTER_EDGE_FEATURE_SCHEMA,
                "ptm_dim": args.ptm_dim,
                "mfcc_dim": int(frames.shape[2]) - args.ptm_dim,
                "context_s": 0.6,
                "bins": [4, 4, 4, 4],
                "scalar_names": list(SCALAR_NAMES),
            },
            normalization={
                "frame_mean": frame_mean.tolist(),
                "frame_std": frame_std.tolist(),
                "scalar_mean": scalar_mean.tolist(),
                "scalar_std": scalar_std.tolist(),
            },
            metadata={
                "ptm_repo_id": args.ptm_repo_id,
                "dataset": str(Path(args.dataset)),
                "trained_steps": args.max_steps,
                "max_delta_s": 0.5,
                "outer_edges_only": True,
            },
        ),
        checkpoint_path,
    )
    metrics = {
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "loss": float(np.mean(losses)),
        "start_mae_s": float(error[:, 0][conf[:, 0]].mean()),
        "end_mae_s": float(error[:, 1][conf[:, 1]].mean()),
        "raw_start_mae_s": float(np.abs(target[:, 0][conf[:, 0]]).mean()),
        "raw_end_mae_s": float(np.abs(target[:, 1][conf[:, 1]]).mean()),
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train speech-core Outer Edge Refiner.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ptm-repo-id",
        default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    parser.add_argument("--ptm-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
