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
from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    feature_extraction_hash,
    validate_sequence_features,
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
    features, labels, mask = _rows_to_padded_tensors(rows, feature_names)
    feature_metadata = _feature_metadata(rows, feature_names)
    class_counts = {
        "merge_positive": int(labels[mask].sum().item()),
        "split_negative": int(mask.sum().item() - labels[mask].sum().item()),
    }
    if not config.allow_single_class and (class_counts["merge_positive"] == 0 or class_counts["split_negative"] == 0):
        raise ValueError(
            "boundary refiner training requires both merge and split labels; "
            f"got {class_counts}. Use --allow-single-class only for loader smoke."
        )

    train_idx, val_idx = _split_indexes(len(rows), val_ratio=config.val_ratio, seed=config.seed)
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    train_mask = mask[train_idx]
    active_train_features = train_features[train_mask]
    feature_mean = active_train_features.mean(dim=0)
    feature_std = active_train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
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

    train_dataset = TensorDataset(features[train_idx], labels[train_idx], mask[train_idx])
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )
    positives = max(1.0, float(train_labels[train_mask].sum().item()))
    negatives = max(1.0, float(train_mask.sum().item() - train_labels[train_mask].sum().item()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    step = 0
    last_loss = math.nan
    while step < config.max_steps:
        for batch_features, batch_labels, batch_mask in train_loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            batch_mask = batch_mask.to(device)
            logits = model(batch_features.to(device), attention_mask=batch_mask.long())
            loss = criterion(logits[batch_mask], batch_labels.to(device)[batch_mask])
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
        "feature_metadata": feature_metadata,
        "class_counts": class_counts,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "train_items": int(mask[train_idx].sum().item()),
        "val_items": int(mask[val_idx].sum().item()) if val_idx else 0,
        "last_train_loss": last_loss,
        "train": _evaluate(
            model,
            features[train_idx],
            labels[train_idx],
            mask[train_idx],
            device=device,
            threshold=config.threshold,
        ),
        "val": _evaluate(
            model,
            features[val_idx],
            labels[val_idx],
            mask[val_idx],
            device=device,
            threshold=config.threshold,
        )
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
            "train_schema": _train_schema(rows),
            "runtime_adapter": _runtime_adapter_name(feature_names),
            **feature_metadata,
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
    metrics["loader_smoke"] = {
        "signature": refiner.signature(),
        "decision": None,
        "decision_skipped_reason": "",
    }
    if _supports_refiner_input_decision(feature_names):
        smoke_input = _row_smoke_input(rows[0], feature_names)
        smoke_decision = refiner.decide_gap(smoke_input)
        metrics["loader_smoke"]["decision"] = {
            "merge": smoke_decision.merge,
            "score": smoke_decision.score,
            "reason": smoke_decision.reason,
        }
    else:
        metrics["loader_smoke"]["decision_skipped_reason"] = (
            "feature_names require frame/window runtime adapter"
        )

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


def _train_schema(rows: Sequence[Mapping[str, Any]]) -> str:
    schemas = {str(row.get("schema") or "") for row in rows}
    return schemas.pop() if len(schemas) == 1 else "mixed_boundary_refiner_dataset"


def _feature_metadata(
    rows: Sequence[Mapping[str, Any]],
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    schema_values = {str(row.get("feature_schema") or "") for row in rows if row.get("feature_schema")}
    hash_values = {str(row.get("feature_schema_hash") or "") for row in rows if row.get("feature_schema_hash")}
    signature_values = [
        dict(row["feature_signature"])
        for row in rows
        if isinstance(row.get("feature_signature"), Mapping)
    ]
    if schema_values and schema_values != {FRAME_SEQUENCE_FEATURE_SCHEMA}:
        raise ValueError(f"unsupported sequence feature schema: {sorted(schema_values)}")
    if len(hash_values) > 1:
        raise ValueError("mixed feature_schema_hash values are not allowed")
    if signature_values:
        expected_signature = signature_values[0]
        for index, signature in enumerate(signature_values):
            if signature != expected_signature:
                raise ValueError(f"feature_signature mismatch at row {index}")
        feature_schema_hash = hash_values.pop() if hash_values else _hash_from_signature(expected_signature)
        return {
            "feature_schema": str(expected_signature.get("feature_schema") or FRAME_SEQUENCE_FEATURE_SCHEMA),
            "feature_schema_hash": feature_schema_hash,
            "feature_signature": expected_signature,
            "feature_dim": len(feature_names),
        }
    if schema_values or hash_values:
        raise ValueError("sequence feature rows require feature_signature metadata")
    return {
        "feature_dim": len(feature_names),
    }


def _hash_from_signature(signature: Mapping[str, Any]) -> str:
    feature_config = signature.get("feature_config")
    feature_names = signature.get("feature_names")
    if not isinstance(feature_config, Mapping) or not isinstance(feature_names, list):
        raise ValueError("feature_signature must contain feature_config and feature_names")
    from boundary.sequence_features import FrameSequenceFeatureConfig

    config = FrameSequenceFeatureConfig(
        left_context_s=float(feature_config["left_context_s"]),
        right_context_s=float(feature_config["right_context_s"]),
        max_ptm_dims=int(feature_config["max_ptm_dims"]),
        include_mfcc=bool(feature_config["include_mfcc"]),
    )
    return feature_extraction_hash(
        config=config,
        feature_names=[str(name) for name in feature_names],
    )


def _supports_refiner_input_decision(feature_names: Sequence[str]) -> bool:
    return set(str(name) for name in feature_names).issubset(set(DEFAULT_REFINER_FEATURES))


def _runtime_adapter_name(feature_names: Sequence[str]) -> str:
    return "refiner_input_v1" if _supports_refiner_input_decision(feature_names) else "frame_sequence_v1"


def _rows_to_padded_tensors(
    rows: Sequence[Mapping[str, Any]],
    feature_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [_row_sequence(row, feature_names) for row in rows]
    label_sequences = [_row_label_sequence(row, expected_len=len(sequence)) for row, sequence in zip(rows, sequences)]
    max_len = max(len(sequence) for sequence in sequences)
    feature_dim = len(feature_names)
    features = torch.zeros((len(rows), max_len, feature_dim), dtype=torch.float32)
    labels = torch.zeros((len(rows), max_len), dtype=torch.float32)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for row_index, (sequence, label_sequence) in enumerate(zip(sequences, label_sequences)):
        length = len(sequence)
        features[row_index, :length] = torch.tensor(sequence, dtype=torch.float32)
        labels[row_index, :length] = torch.tensor(label_sequence, dtype=torch.float32)
        mask[row_index, :length] = True
    return features, labels, mask


def _row_sequence(row: Mapping[str, Any], feature_names: tuple[str, ...]) -> list[list[float]]:
    raw_sequence = row.get("sequence_features")
    if isinstance(raw_sequence, list) and raw_sequence:
        return validate_sequence_features(
            raw_sequence,
            feature_names=feature_names,
            expected_feature_names=row.get("feature_names") or feature_names,
        ).astype(float).tolist()
    features = _row_features(row, feature_names)
    validate_sequence_features([features], feature_names=feature_names)
    return [features]


def _row_label_sequence(row: Mapping[str, Any], *, expected_len: int) -> list[float]:
    raw = row.get("sequence_labels")
    if isinstance(raw, list) and raw:
        if len(raw) != expected_len:
            raise ValueError("sequence_labels length must match sequence_features length")
        labels = [float(value) for value in raw]
        if any(not math.isfinite(value) for value in labels):
            raise ValueError("sequence_labels must not contain NaN or inf")
        return labels
    if expected_len != 1:
        raise ValueError("sequence row requires sequence_labels")
    return [float(row.get("label", row.get("merge_target", 0)))]


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


def _row_smoke_input(row: Mapping[str, Any], feature_names: tuple[str, ...]) -> RefinerInput:
    if isinstance(row.get("refiner_input"), Mapping):
        return _row_refiner_input(row)
    sequence = _row_sequence(row, feature_names)
    values = {name: float(value) for name, value in zip(feature_names, sequence[0])}
    gap_s = values.get("gap_s", 0.0)
    left_duration_s = max(0.0, values.get("left_duration_s", 1.0))
    right_duration_s = max(0.0, values.get("right_duration_s", 1.0))
    left_start = 0.0
    left_end = left_duration_s
    right_start = left_end + gap_s
    right_end = right_start + right_duration_s
    return RefinerInput(
        gap_s=gap_s,
        left_start=left_start,
        left_end=left_end,
        right_start=right_start,
        right_end=right_end,
        current_core_s=values.get("current_core_s", left_duration_s),
        proposed_core_s=values.get("proposed_core_s", right_end - left_start),
        gap_merge_s=max(1e-6, values.get("gap_merge_s", 1.5)),
        left_score=values.get("left_score"),
        right_score=values.get("right_score"),
        valley_score_min=values.get("valley_score_min"),
        cut_score_max=values.get("cut_score_max"),
        gap_boundary_score=values.get("gap_boundary_score"),
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
    mask: torch.Tensor,
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    if features.numel() == 0 or not bool(mask.any().item()):
        return {}
    model.eval()
    with torch.inference_mode():
        mask_device = mask.to(device)
        logits = model(features.to(device), attention_mask=mask_device.long()).detach().cpu()
        logits = logits[mask]
        probs = torch.sigmoid(logits)
    preds = probs >= threshold
    truth = labels[mask] >= 0.5
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
