#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID, qwen_asr_repo_tag  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    SPEECH_ISLAND_MEMBERSHIP_LABELS,
    SPEECH_ISLAND_SCORER_LABELS,
)
from boundary.outer_refiner_v2 import (  # noqa: E402
    FullIslandOuterEdgeNetwork,
    build_outer_edge_refiner_v2_checkpoint,
    decode_outer_edge_probabilities,
)


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(row["feature_path"])
    return (
        data["features"].astype(np.float32),
        data["labels"].astype(np.int64),
        data["weights"].astype(np.float32),
    )


def _normalization(rows: list[dict]) -> dict[str, list[float]]:
    first, _labels, _weights = _load(rows[0])
    feature_sum = np.zeros(first.shape[1], dtype=np.float64)
    square_sum = np.zeros(first.shape[1], dtype=np.float64)
    weight_sum = 0.0
    for row in rows:
        features, _labels, weights = _load(row)
        weight = weights.reshape(-1, 1).astype(np.float64)
        feature_sum += (features * weight).sum(axis=0)
        square_sum += (np.square(features) * weight).sum(axis=0)
        weight_sum += float(weight.sum())
    mean = feature_sum / max(weight_sum, 1e-6)
    variance = square_sum / max(weight_sum, 1e-6) - np.square(mean)
    return {
        "feature_mean": mean.astype(np.float32).tolist(),
        "feature_std": np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32).tolist(),
    }


def _normalized(features: np.ndarray, normalization: dict) -> np.ndarray:
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    return np.ascontiguousarray((features - mean) / np.maximum(std, 1e-6))


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    if args.ptm_repo_id != QWEN_ASR_17B_REPO_ID:
        raise ValueError("Outer Edge Refiner v2 training is 1.7B-only")
    apply_vram_safety_cap(0.95)
    rows = _rows(Path(args.dataset_manifest))
    train_rows = [row for row in rows if row.get("partition") == "train"]
    val_rows = [row for row in rows if row.get("partition") != "train"]
    if not val_rows:
        rng = np.random.default_rng(args.seed)
        val_indexes = set(
            int(index)
            for index in rng.permutation(len(train_rows))[: max(1, len(train_rows) // 10)]
        )
        val_rows = [row for index, row in enumerate(train_rows) if index in val_indexes]
        train_rows = [row for index, row in enumerate(train_rows) if index not in val_indexes]
    normalization = _normalization(train_rows)
    first, _labels, _weights = _load(train_rows[0])
    model_config = {
        "input_dim": int(first.shape[1]),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "state_size": 32,
        "num_heads": 4,
        "head_dim": 64,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 8,
        "bidirectional": True,
        "output_dim": len(SPEECH_ISLAND_SCORER_LABELS),
    }
    device = torch.device(args.device)
    model = FullIslandOuterEdgeNetwork(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    class_weights = torch.tensor(
        [args.discardable_weight, args.semantic_target_weight, args.unsure_weight],
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(args.seed)
    losses: list[float] = []
    started = time.monotonic()
    for step in range(args.max_steps):
        row = train_rows[int(rng.integers(0, len(train_rows)))]
        features, labels, weights = _load(row)
        logits = model(
            torch.from_numpy(_normalized(features, normalization)).unsqueeze(0).to(device)
        )[0]
        label_tensor = torch.from_numpy(labels).to(device)
        effective = torch.from_numpy(weights).to(device) * class_weights[label_tensor]
        raw_loss = F.cross_entropy(logits, label_tensor, reduction="none")
        probabilities = torch.softmax(logits, dim=-1)
        pt = probabilities.gather(1, label_tensor.unsqueeze(1)).squeeze(1)
        loss = (
            raw_loss * torch.pow(1.0 - pt, args.focal_gamma) * effective
        ).sum() / effective.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step + 1) % args.log_every == 0:
            print(
                f"outer_v2_train={step + 1}/{args.max_steps} "
                f"loss={losses[-1]:.6f} elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
    target_index = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    truth_covered = predicted_covered = edge_total = known_clipped = 0
    with torch.inference_mode():
        for row in val_rows:
            features, labels, _weights = _load(row)
            logits = model(
                torch.from_numpy(_normalized(features, normalization)).unsqueeze(0).to(device)
            )[0]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            truth_target = np.flatnonzero(labels == target_index)
            if truth_target.size == 0:
                continue
            edge_total += 1
            truth_covered += 1
            prediction = decode_outer_edge_probabilities(
                probabilities,
                raw_start_s=0.0,
                raw_end_s=float(len(labels)),
                frame_hop_s=1.0,
            )
            if prediction.start_action == "refined" or prediction.end_action == "refined":
                predicted_covered += 1
            if prediction.start_s > int(truth_target[0]) or prediction.end_s < int(truth_target[-1]) + 1:
                known_clipped += 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / (
        f"outer_edge_refiner_v2.{qwen_asr_repo_tag(args.ptm_repo_id)}.pt"
    )
    torch.save(
        build_outer_edge_refiner_v2_checkpoint(
            model=model,
            model_config=model_config,
            feature_config={
                "semantic_projected_ptm_dim": args.semantic_projected_ptm_dim,
                "mfcc_dim": int(first.shape[1])
                - args.semantic_projected_ptm_dim
                - len(SPEECH_ISLAND_SCORER_LABELS)
                - len(SPEECH_ISLAND_MEMBERSHIP_LABELS)
                - 1,
                "semantic_probability_dim": len(SPEECH_ISLAND_SCORER_LABELS),
                "source_membership_probability_dim": len(
                    SPEECH_ISLAND_MEMBERSHIP_LABELS
                ),
                "relative_position_dim": 1,
                "frame_hop_s": args.frame_hop_s,
                "context": "full_scorer_island",
            },
            normalization=normalization,
            metadata={
                "ptm_repo_id": args.ptm_repo_id,
                "dataset_manifest": str(Path(args.dataset_manifest)),
                "semantic_scorer_sha256": args.semantic_scorer_sha256,
                "trained_steps": args.max_steps,
            },
        ),
        checkpoint_path,
    )
    metrics = {
        "schema": "outer_edge_refiner_v2_train_metrics_v1",
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "loss": float(np.mean(losses)),
        "safe_zone_coverage": predicted_covered / max(edge_total, 1),
        "known_tail_clipping_count": known_clipped,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1.7B full-island Outer v2.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ptm-repo-id", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--semantic-scorer-sha256", required=True)
    parser.add_argument("--semantic-projected-ptm-dim", type=int, default=128)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--discardable-weight", type=float, default=1.0)
    parser.add_argument("--semantic-target-weight", type=float, default=3.0)
    parser.add_argument("--unsure-weight", type=float, default=1.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
