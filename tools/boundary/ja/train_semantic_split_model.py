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
from boundary.sequence_features import SPLIT_CANDIDATE_SCALAR_NAMES  # noqa: E402
from boundary.split_model import (  # noqa: E402
    SEMANTIC_SPLIT_FEATURE_SCHEMA,
    SEMANTIC_SPLIT_LABELS,
    SemanticSplitVerifierNetwork,
    build_semantic_split_checkpoint,
)


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    bundle = np.load(args.dataset)
    frames = bundle["frame_features"].astype(np.float32)
    scalars = bundle["scalar_features"].astype(np.float32)
    labels = bundle["labels"].astype(np.int64)
    partitions = bundle["partitions"].astype(str)
    train_mask = partitions == "train"
    val_mask = partitions == "val"
    if not val_mask.any():
        rng = np.random.default_rng(args.seed)
        indexes = rng.permutation(labels.size)
        val_mask[indexes[: max(1, labels.size // 10)]] = True
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
        "output_dim": 3,
    }
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = SemanticSplitVerifierNetwork(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_indexes = np.flatnonzero(train_mask)
    class_weights = torch.tensor(
        [args.cut_weight, args.continue_weight, args.unsure_weight],
        dtype=torch.float32,
        device=device,
    )
    losses: list[float] = []
    for step in range(args.max_steps):
        batch_indexes = rng.choice(train_indexes, size=args.batch_size, replace=True)
        frame_tensor = torch.from_numpy(frames[batch_indexes]).to(device)
        scalar_tensor = torch.from_numpy(scalars[batch_indexes]).to(device)
        label_tensor = torch.from_numpy(labels[batch_indexes]).to(device)
        logits = model(frame_tensor, scalar_tensor)
        loss = F.cross_entropy(logits, label_tensor, weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step + 1) % args.log_every == 0:
            print(
                f"semantic_split_train={step + 1}/{args.max_steps} "
                f"loss={losses[-1]:.6f} avg_loss={np.mean(losses):.6f}",
                flush=True,
            )
    val_indexes = np.flatnonzero(val_mask)
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, val_indexes.size, args.batch_size):
            index = val_indexes[start : start + args.batch_size]
            logits = model(
                torch.from_numpy(frames[index]).to(device),
                torch.from_numpy(scalars[index]).to(device),
            )
            predictions.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probabilities = np.concatenate(predictions)
    predicted = probabilities.argmax(axis=1)
    truth = labels[val_indexes]
    confusion = np.zeros((3, 3), dtype=np.int64)
    for expected, actual in zip(truth, predicted):
        confusion[int(expected), int(actual)] += 1
    cut_tp = int(confusion[0, 0])
    cut_precision = cut_tp / max(1, int(confusion[:, 0].sum()))
    continue_recall = int(confusion[1, 1]) / max(1, int(confusion[1].sum()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / (
        f"semantic_split_model_v1.{qwen_asr_repo_tag(args.ptm_repo_id)}.pt"
    )
    normalization = {
        "frame_mean": frame_mean.tolist(),
        "frame_std": frame_std.tolist(),
        "scalar_mean": scalar_mean.tolist(),
        "scalar_std": scalar_std.tolist(),
    }
    torch.save(
        build_semantic_split_checkpoint(
            model=model,
            model_config=model_config,
            feature_config={
                "schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
                "ptm_dim": args.ptm_dim,
                "mfcc_dim": int(frames.shape[2]) - args.ptm_dim,
                "left_context_s": 1.6,
                "right_context_s": 1.6,
                "gap_context_s": 0.3,
                "left_bins": 8,
                "gap_bins": 4,
                "right_bins": 8,
                "scalar_names": list(SPLIT_CANDIDATE_SCALAR_NAMES),
            },
            normalization=normalization,
            metadata={
                "ptm_repo_id": args.ptm_repo_id,
                "dataset": str(Path(args.dataset)),
                "trained_steps": args.max_steps,
                "class_weights": class_weights.detach().cpu().tolist(),
            },
        ),
        checkpoint_path,
    )
    metrics = {
        "labels": list(SEMANTIC_SPLIT_LABELS),
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "loss": float(np.mean(losses)),
        "accuracy": float((predicted == truth).mean()),
        "cut_precision": cut_precision,
        "continue_recall": continue_recall,
        "confusion_expected_rows_predicted_columns": confusion.tolist(),
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train candidate-level Semantic Split Verifier.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ptm-repo-id",
        default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--cut-weight", type=float, default=1.0)
    parser.add_argument("--continue-weight", type=float, default=1.5)
    parser.add_argument("--unsure-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
