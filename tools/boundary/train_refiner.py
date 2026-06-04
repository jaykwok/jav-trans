#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from boundary.backbones import TRANSFORMERS_MAMBA2_BACKBONE, BoundarySequenceClassifier
from boundary.refiner import (
    DEFAULT_REFINER_FEATURES,
    RefinerInput,
    build_learned_refiner_checkpoint,
    load_boundary_refiner,
    refiner_input_to_features,
)


@dataclass(frozen=True)
class TrainRefinerConfig:
    max_steps: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 13
    device: str = "auto"
    val_ratio: float = 0.10
    hidden_size: int = 64
    num_layers: int = 1
    state_size: int = 16
    num_heads: int = 4
    n_groups: int = 2
    chunk_size: int = 8
    bidirectional: bool = True
    threshold: float = 0.5
    allow_single_class: bool = False
    log_interval_steps: int = 20


def train_refiner(
    *,
    dataset_paths: Sequence[Path],
    output_dir: Path,
    config: TrainRefinerConfig = TrainRefinerConfig(),
) -> dict[str, Any]:
    _validate_config(config)
    rows = _load_dataset_rows(dataset_paths)
    if not rows:
        raise ValueError("boundary refiner dataset is empty")
    feature_names = _feature_names(rows)
    features = torch.tensor([_row_features(row, feature_names) for row in rows], dtype=torch.float32)
    labels = torch.tensor([float(row.get("label", row.get("merge_target", 0))) for row in rows], dtype=torch.float32)
    class_counts = {
        "merge_positive": int(labels.sum().item()),
        "split_negative": int(labels.numel() - labels.sum().item()),
    }
    if not config.allow_single_class and (class_counts["merge_positive"] == 0 or class_counts["split_negative"] == 0):
        raise ValueError(
            "boundary refiner training requires both merge and split labels; "
            f"got {class_counts}. Use --allow-single-class only for loader smoke."
        )

    train_idx, val_idx = _split_indexes(len(rows), val_ratio=config.val_ratio, seed=config.seed)
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    features = (features - feature_mean) / feature_std

    device = _resolve_device(config.device)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model = BoundarySequenceClassifier(
        input_dim=len(feature_names),
        backbone=TRANSFORMERS_MAMBA2_BACKBONE,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        state_size=config.state_size,
        num_heads=config.num_heads,
        n_groups=config.n_groups,
        chunk_size=config.chunk_size,
        bidirectional=config.bidirectional,
    ).to(device)

    train_dataset = TensorDataset(features[train_idx].unsqueeze(1), labels[train_idx].unsqueeze(1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )
    positives = max(1.0, float(train_labels.sum().item()))
    negatives = max(1.0, float(train_labels.numel() - train_labels.sum().item()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    step = 0
    last_loss = math.nan
    while step < config.max_steps:
        for batch_features, batch_labels in train_loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            if config.log_interval_steps > 0 and step % config.log_interval_steps == 0:
                print(f"step={step} loss={last_loss:.6f}", flush=True)
            if step >= config.max_steps:
                break

    metrics = {
        "schema": "boundary_refiner_train_metrics_v1",
        "dataset_paths": [str(path) for path in dataset_paths],
        "output_dir": str(output_dir),
        "config": asdict(config),
        "feature_names": list(feature_names),
        "class_counts": class_counts,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "last_train_loss": last_loss,
        "train": _evaluate(model, features[train_idx], labels[train_idx], device=device, threshold=config.threshold),
        "val": _evaluate(model, features[val_idx], labels[val_idx], device=device, threshold=config.threshold)
        if val_idx
        else {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    checkpoint = build_learned_refiner_checkpoint(
        model=model,
        feature_names=feature_names,
        feature_mean=tuple(float(value) for value in feature_mean.tolist()),
        feature_std=tuple(float(value) for value in feature_std.tolist()),
        metadata={
            "trainer": "tools/boundary/train_refiner.py",
            "dataset_paths": [str(path) for path in dataset_paths],
            "class_counts": class_counts,
            "train_schema": "boundary_refiner_supervised_gap_v1",
        },
    )
    checkpoint_path = output_dir / "boundary_refiner.pt"
    torch.save(checkpoint, checkpoint_path)
    metrics["checkpoint"] = str(checkpoint_path)

    refiner = load_boundary_refiner(
        enabled=True,
        model_path=str(checkpoint_path),
        backbone=TRANSFORMERS_MAMBA2_BACKBONE,
        merge_threshold=config.threshold,
    )
    smoke_input = _row_refiner_input(rows[0])
    smoke_decision = refiner.decide_gap(smoke_input)
    metrics["loader_smoke"] = {
        "signature": refiner.signature(),
        "decision": {
            "merge": smoke_decision.merge,
            "score": smoke_decision.score,
            "reason": smoke_decision.reason,
        },
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")
    print(f"class_counts={json.dumps(class_counts, ensure_ascii=False, sort_keys=True)}")
    return metrics


def _load_dataset_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _feature_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    first = rows[0].get("feature_names") or list(DEFAULT_REFINER_FEATURES)
    names = tuple(str(name) for name in first)
    if not names:
        raise ValueError("dataset feature_names must not be empty")
    for index, row in enumerate(rows):
        row_names = tuple(str(name) for name in (row.get("feature_names") or names))
        if row_names != names:
            raise ValueError(f"feature_names mismatch at row {index}: {row_names} != {names}")
    return names


def _row_features(row: Mapping[str, Any], feature_names: tuple[str, ...]) -> list[float]:
    raw = row.get("features")
    if isinstance(raw, list) and len(raw) == len(feature_names):
        return [float(value) for value in raw]
    return refiner_input_to_features(_row_refiner_input(row), feature_names)


def _row_refiner_input(row: Mapping[str, Any]) -> RefinerInput:
    payload = row.get("refiner_input")
    if not isinstance(payload, Mapping):
        raise ValueError("dataset row missing refiner_input")
    return RefinerInput(
        gap_s=float(payload["gap_s"]),
        left_start=float(payload["left_start"]),
        left_end=float(payload["left_end"]),
        right_start=float(payload["right_start"]),
        right_end=float(payload["right_end"]),
        current_core_s=float(payload["current_core_s"]),
        proposed_core_s=float(payload["proposed_core_s"]),
        gap_merge_s=float(payload["gap_merge_s"]),
        left_score=_optional_float(payload.get("left_score")),
        right_score=_optional_float(payload.get("right_score")),
        valley_score_min=_optional_float(payload.get("valley_score_min")),
        cut_score_max=_optional_float(payload.get("cut_score_max")),
        gap_boundary_score=_optional_float(payload.get("gap_boundary_score")),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _split_indexes(count: int, *, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indexes = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indexes)
    val_count = int(round(count * val_ratio)) if count > 1 else 0
    val_count = min(max(0, val_count), max(0, count - 1))
    return indexes[val_count:], indexes[:val_count]


def _evaluate(
    model: BoundarySequenceClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    if features.numel() == 0:
        return {}
    model.eval()
    with torch.inference_mode():
        logits = model(features.unsqueeze(1).to(device)).reshape(-1).detach().cpu()
        probs = torch.sigmoid(logits)
    preds = probs >= threshold
    truth = labels.reshape(-1) >= 0.5
    tp = int(torch.logical_and(preds, truth).sum().item())
    tn = int(torch.logical_and(~preds, ~truth).sum().item())
    fp = int(torch.logical_and(preds, ~truth).sum().item())
    fn = int(torch.logical_and(~preds, truth).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "count": int(labels.numel()),
        "accuracy": (tp + tn) / max(1, int(labels.numel())),
        "merge_precision": precision,
        "merge_recall": recall,
        "merge_f1": (2 * precision * recall / max(1e-9, precision + recall)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "prob_mean": float(probs.mean().item()),
        "prob_min": float(probs.min().item()),
        "prob_max": float(probs.max().item()),
    }


def _resolve_device(requested: str) -> torch.device:
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA training but torch.cuda.is_available() is false")
    return torch.device(normalized)


def _validate_config(config: TrainRefinerConfig) -> None:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= config.val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.state_size <= 0:
        raise ValueError("state_size must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0.0 <= config.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if config.log_interval_steps < 0:
        raise ValueError("log_interval_steps must be non-negative")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a transformers.Mamba2Model Boundary Refiner from supervised gap samples."
    )
    parser.add_argument("--dataset", action="append", required=True, help="Gap dataset JSONL. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-groups", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow-single-class",
        action="store_true",
        help="Only for loader smoke; formal training should keep both merge and split labels.",
    )
    parser.add_argument("--log-interval-steps", type=int, default=20)
    args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_refiner(
        dataset_paths=[Path(path) for path in args.dataset],
        output_dir=Path(args.output_dir),
        config=TrainRefinerConfig(
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            val_ratio=args.val_ratio,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            state_size=args.state_size,
            num_heads=args.num_heads,
            n_groups=args.n_groups,
            chunk_size=args.chunk_size,
            bidirectional=not args.unidirectional,
            threshold=args.threshold,
            allow_single_class=args.allow_single_class,
            log_interval_steps=args.log_interval_steps,
        ),
    )


if __name__ == "__main__":
    main()
