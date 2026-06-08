#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from boundary.backbones import TRANSFORMERS_MAMBA2_BACKBONE, BoundarySequenceClassifier
from boundary.refiner import (
    CONTEXT_OUTPUT_DIM,
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
    target_domain_speedup: float = 1.5
    target_chunk_s: float = 3.0
    max_core_chunk_s: float = 5.0
    max_padded_chunk_s: float = 6.5
    min_chunk_s: float = 0.4
    context_max_padding_s: float = 1.5
    context_loss_weight: float = 0.25


@dataclass(frozen=True)
class LoadedRefinerDataset:
    features: torch.Tensor
    labels: torch.Tensor
    context_targets: torch.Tensor
    mask: torch.Tensor
    feature_names: tuple[str, ...]
    feature_metadata: dict[str, Any]
    train_schema: str
    class_counts: dict[str, int]
    first_row: dict[str, Any]


@dataclass(frozen=True)
class DatasetScan:
    row_count: int
    max_len: int
    feature_names: tuple[str, ...]
    feature_metadata: dict[str, Any]
    train_schema: str
    class_counts: dict[str, int]
    first_row: dict[str, Any]


def train_refiner(
    *,
    dataset_paths: Sequence[Path],
    output_dir: Path,
    config: TrainRefinerConfig = TrainRefinerConfig(),
) -> dict[str, Any]:
    _validate_config(config)
    loaded = _load_dataset_tensors(dataset_paths)
    features = loaded.features
    labels = loaded.labels
    context_targets = loaded.context_targets
    mask = loaded.mask
    feature_names = loaded.feature_names
    feature_metadata = loaded.feature_metadata
    class_counts = loaded.class_counts
    if not config.allow_single_class and (class_counts["merge_positive"] == 0 or class_counts["split_negative"] == 0):
        raise ValueError(
            "boundary refiner training requires both merge and split labels; "
            f"got {class_counts}. Use --allow-single-class only for loader smoke."
        )

    train_idx, val_idx = _split_indexes(features.shape[0], val_ratio=config.val_ratio, seed=config.seed)
    train_mask = mask[train_idx]
    train_labels = labels[train_idx]
    train_row_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_row_mask[train_idx] = True
    active_train_mask = train_row_mask[:, None] & mask
    active_train_features = features[active_train_mask]
    feature_mean = active_train_features.mean(dim=0)
    feature_std = active_train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    del active_train_features
    features.sub_(feature_mean).div_(feature_std)

    device = _resolve_device(config.device)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model = BoundarySequenceClassifier(
        input_dim=len(feature_names),
        backbone=TRANSFORMERS_MAMBA2_BACKBONE,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_dim=CONTEXT_OUTPUT_DIM,
        state_size=config.state_size,
        num_heads=config.num_heads,
        n_groups=config.n_groups,
        chunk_size=config.chunk_size,
        bidirectional=config.bidirectional,
    ).to(device)

    train_dataset = Subset(TensorDataset(features, labels, context_targets, mask), train_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )
    positives = max(1.0, float(train_labels[train_mask].sum().item()))
    negatives = max(1.0, float(train_mask.sum().item() - train_labels[train_mask].sum().item()))
    merge_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    context_criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    step = 0
    last_loss = math.nan
    while step < config.max_steps:
        for batch_features, batch_labels, batch_context_targets, batch_mask in train_loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            batch_mask = batch_mask.to(device)
            logits = model(batch_features.to(device), attention_mask=batch_mask.long())
            merge_loss = merge_criterion(logits[..., 0][batch_mask], batch_labels.to(device)[batch_mask])
            context_pred = torch.sigmoid(logits[..., 1:3]) * float(config.context_max_padding_s)
            context_loss = context_criterion(
                context_pred[batch_mask],
                batch_context_targets.to(device)[batch_mask],
            )
            loss = merge_loss + float(config.context_loss_weight) * context_loss
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
            features,
            labels,
            mask,
            indexes=train_idx,
            device=device,
            threshold=config.threshold,
            batch_size=config.batch_size,
        ),
        "val": _evaluate(
            model,
            features,
            labels,
            mask,
            indexes=val_idx,
            device=device,
            threshold=config.threshold,
            batch_size=config.batch_size,
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
            "train_schema": loaded.train_schema,
            "runtime_adapter": _runtime_adapter_name(feature_names),
            "timing_policy": {
                "target_domain_speedup": config.target_domain_speedup,
                "target_chunk_s": config.target_chunk_s,
                "max_core_chunk_s": config.max_core_chunk_s,
                "max_padded_chunk_s": config.max_padded_chunk_s,
                "min_chunk_s": config.min_chunk_s,
                "context_max_padding_s": config.context_max_padding_s,
                "context_loss_weight": config.context_loss_weight,
                "source": "anime_nsfw_sfw_galgame_duration_distribution",
            },
            "context_max_padding_s": config.context_max_padding_s,
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
        smoke_input = _row_smoke_input(loaded.first_row, feature_names)
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


def _load_dataset_tensors(paths: Sequence[Path]) -> LoadedRefinerDataset:
    scan = _scan_dataset(paths)
    features = torch.zeros((scan.row_count, scan.max_len, len(scan.feature_names)), dtype=torch.float32)
    labels = torch.zeros((scan.row_count, scan.max_len), dtype=torch.float32)
    context_targets = torch.zeros((scan.row_count, scan.max_len, 2), dtype=torch.float32)
    mask = torch.zeros((scan.row_count, scan.max_len), dtype=torch.bool)

    row_index = 0
    for row in _iter_dataset_rows(paths):
        sequence = _row_sequence_array(row, scan.feature_names)
        label_sequence = _row_label_sequence(row, expected_len=int(sequence.shape[0]))
        context_sequence = _row_context_target_sequence(row, expected_len=int(sequence.shape[0]))
        length = int(sequence.shape[0])
        features[row_index, :length].copy_(torch.from_numpy(sequence))
        labels[row_index, :length].copy_(torch.tensor(label_sequence, dtype=torch.float32))
        context_targets[row_index, :length].copy_(torch.tensor(context_sequence, dtype=torch.float32))
        mask[row_index, :length] = True
        row_index += 1
    if row_index != scan.row_count:
        raise ValueError(f"dataset changed while loading: expected {scan.row_count}, got {row_index}")
    return LoadedRefinerDataset(
        features=features,
        labels=labels,
        context_targets=context_targets,
        mask=mask,
        feature_names=scan.feature_names,
        feature_metadata=scan.feature_metadata,
        train_schema=scan.train_schema,
        class_counts=scan.class_counts,
        first_row=scan.first_row,
    )


def _scan_dataset(paths: Sequence[Path]) -> DatasetScan:
    row_count = 0
    max_len = 0
    feature_names: tuple[str, ...] | None = None
    first_row: dict[str, Any] | None = None
    class_counts = {"merge_positive": 0, "split_negative": 0}
    train_schemas: set[str] = set()
    feature_schema_values: set[str] = set()
    feature_hash_values: set[str] = set()
    expected_signature: dict[str, Any] | None = None

    for row in _iter_dataset_rows(paths):
        if first_row is None:
            first_row = dict(row)
        row_names = _row_feature_names(row, feature_names)
        if feature_names is None:
            feature_names = row_names
        elif row_names != feature_names:
            raise ValueError(f"feature_names mismatch at row {row_count}: {row_names} != {feature_names}")
        sequence_len = _row_sequence_length(row, feature_names)
        label_sequence = _row_label_sequence(row, expected_len=sequence_len)
        _row_context_target_sequence(row, expected_len=sequence_len)
        row_positive = sum(1 for value in label_sequence if value >= 0.5)
        class_counts["merge_positive"] += row_positive
        class_counts["split_negative"] += len(label_sequence) - row_positive
        max_len = max(max_len, sequence_len)
        train_schemas.add(str(row.get("schema") or ""))
        if row.get("feature_schema"):
            feature_schema_values.add(str(row["feature_schema"]))
        if row.get("feature_schema_hash"):
            feature_hash_values.add(str(row["feature_schema_hash"]))
        signature = row.get("feature_signature")
        if isinstance(signature, Mapping):
            signature_dict = dict(signature)
            if expected_signature is None:
                expected_signature = signature_dict
            elif signature_dict != expected_signature:
                raise ValueError(f"feature_signature mismatch at row {row_count}")
        row_count += 1

    if row_count == 0 or feature_names is None or first_row is None:
        raise ValueError("boundary refiner dataset is empty")
    if max_len <= 0:
        raise ValueError("boundary refiner dataset has no sequence items")
    return DatasetScan(
        row_count=row_count,
        max_len=max_len,
        feature_names=feature_names,
        feature_metadata=_feature_metadata_from_scan(
            feature_names=feature_names,
            schema_values=feature_schema_values,
            hash_values=feature_hash_values,
            expected_signature=expected_signature,
        ),
        train_schema=_train_schema_from_values(train_schemas),
        class_counts=class_counts,
        first_row=first_row,
    )


def _iter_dataset_rows(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _row_feature_names(row: Mapping[str, Any], fallback: tuple[str, ...] | None = None) -> tuple[str, ...]:
    raw = row.get("feature_names")
    if raw:
        names = tuple(str(name) for name in raw)
    elif fallback is not None:
        names = fallback
    else:
        names = tuple(DEFAULT_REFINER_FEATURES)
    if not names:
        raise ValueError("dataset feature_names must not be empty")
    return names


def _train_schema(rows: Sequence[Mapping[str, Any]]) -> str:
    schemas = {str(row.get("schema") or "") for row in rows}
    return _train_schema_from_values(schemas)


def _train_schema_from_values(schemas: set[str]) -> str:
    return schemas.pop() if len(schemas) == 1 else "mixed_boundary_refiner_dataset"


def _feature_metadata_from_scan(
    *,
    feature_names: tuple[str, ...],
    schema_values: set[str],
    hash_values: set[str],
    expected_signature: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if schema_values and schema_values != {FRAME_SEQUENCE_FEATURE_SCHEMA}:
        raise ValueError(f"unsupported sequence feature schema: {sorted(schema_values)}")
    if len(hash_values) > 1:
        raise ValueError("mixed feature_schema_hash values are not allowed")
    if expected_signature is not None:
        feature_schema_hash = next(iter(hash_values)) if hash_values else _hash_from_signature(expected_signature)
        return {
            "feature_schema": str(expected_signature.get("feature_schema") or FRAME_SEQUENCE_FEATURE_SCHEMA),
            "feature_schema_hash": feature_schema_hash,
            "feature_signature": dict(expected_signature),
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
        target_chunk_s=float(feature_config["target_chunk_s"]),
    )
    return feature_extraction_hash(
        config=config,
        feature_names=[str(name) for name in feature_names],
    )


def _supports_refiner_input_decision(feature_names: Sequence[str]) -> bool:
    return set(str(name) for name in feature_names).issubset(set(DEFAULT_REFINER_FEATURES))


def _runtime_adapter_name(feature_names: Sequence[str]) -> str:
    return "refiner_input_v1" if _supports_refiner_input_decision(feature_names) else "frame_sequence_v1"


def _row_sequence_length(row: Mapping[str, Any], feature_names: tuple[str, ...]) -> int:
    raw_sequence = row.get("sequence_features")
    if isinstance(raw_sequence, list) and raw_sequence:
        first = raw_sequence[0]
        if not isinstance(first, list) or len(first) != len(feature_names):
            validate_sequence_features(
                raw_sequence,
                feature_names=feature_names,
                expected_feature_names=row.get("feature_names") or feature_names,
            )
        return len(raw_sequence)
    return 1


def _row_sequence_array(row: Mapping[str, Any], feature_names: tuple[str, ...]) -> np.ndarray:
    raw_sequence = row.get("sequence_features")
    if isinstance(raw_sequence, list) and raw_sequence:
        return validate_sequence_features(
            raw_sequence,
            feature_names=feature_names,
            expected_feature_names=row.get("feature_names") or feature_names,
        ).astype(np.float32, copy=False)
    features = _row_features(row, feature_names)
    return validate_sequence_features([features], feature_names=feature_names).astype(np.float32, copy=False)


def _row_sequence(row: Mapping[str, Any], feature_names: tuple[str, ...]) -> list[list[float]]:
    return _row_sequence_array(row, feature_names).astype(float).tolist()


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


def _row_context_target_sequence(row: Mapping[str, Any], *, expected_len: int) -> list[list[float]]:
    raw = row.get("sequence_context_targets")
    if isinstance(raw, list) and raw:
        if len(raw) != expected_len:
            raise ValueError("sequence_context_targets length must match sequence_features length")
        targets: list[list[float]] = []
        for item in raw:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("sequence_context_targets items must be [left_s, right_s]")
            left, right = float(item[0]), float(item[1])
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("sequence_context_targets must not contain NaN or inf")
            targets.append([max(0.0, left), max(0.0, right)])
        return targets
    raise ValueError("boundary refiner v2 rows require sequence_context_targets")


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
    indexes: Sequence[int] | None = None,
    device: torch.device,
    threshold: float,
    batch_size: int = 256,
) -> dict[str, Any]:
    if features.numel() == 0:
        return {}
    if indexes is None:
        indexes = list(range(features.shape[0]))
    if not indexes:
        return {}
    model.eval()
    tp = tn = fp = fn = count = 0
    prob_sum = 0.0
    prob_min = math.inf
    prob_max = -math.inf
    with torch.inference_mode():
        for start in range(0, len(indexes), max(1, batch_size)):
            batch_indexes = list(indexes[start : start + max(1, batch_size)])
            batch_mask = mask[batch_indexes]
            if not bool(batch_mask.any().item()):
                continue
            logits = model(
                features[batch_indexes].to(device),
                attention_mask=batch_mask.to(device).long(),
            ).detach().cpu()
            if logits.ndim == 3:
                logits = logits[..., 0]
            logits = logits[batch_mask]
            probs = torch.sigmoid(logits)
            truth = labels[batch_indexes][batch_mask] >= 0.5
            preds = probs >= threshold
            tp += int(torch.logical_and(preds, truth).sum().item())
            tn += int(torch.logical_and(~preds, ~truth).sum().item())
            fp += int(torch.logical_and(preds, ~truth).sum().item())
            fn += int(torch.logical_and(~preds, truth).sum().item())
            batch_count = int(truth.numel())
            count += batch_count
            prob_sum += float(probs.sum().item())
            prob_min = min(prob_min, float(probs.min().item()))
            prob_max = max(prob_max, float(probs.max().item()))
    if count == 0:
        return {}
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "count": count,
        "accuracy": (tp + tn) / max(1, count),
        "merge_precision": precision,
        "merge_recall": recall,
        "merge_f1": (2 * precision * recall / max(1e-9, precision + recall)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "prob_mean": prob_sum / count,
        "prob_min": prob_min,
        "prob_max": prob_max,
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
    if config.target_domain_speedup <= 0.0:
        raise ValueError("target_domain_speedup must be positive")
    for name in (
        "target_chunk_s",
        "max_core_chunk_s",
        "max_padded_chunk_s",
        "min_chunk_s",
        "context_max_padding_s",
    ):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if config.context_loss_weight < 0.0:
        raise ValueError("context_loss_weight must be non-negative")


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
    parser.add_argument("--target-domain-speedup", type=float, default=1.5)
    parser.add_argument("--target-chunk-s", type=float, default=3.0)
    parser.add_argument("--max-core-chunk-s", type=float, default=5.0)
    parser.add_argument("--max-padded-chunk-s", type=float, default=6.5)
    parser.add_argument("--min-chunk-s", type=float, default=0.4)
    parser.add_argument("--context-max-padding-s", type=float, default=1.5)
    parser.add_argument("--context-loss-weight", type=float, default=0.25)
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
            target_domain_speedup=args.target_domain_speedup,
            target_chunk_s=args.target_chunk_s,
            max_core_chunk_s=args.max_core_chunk_s,
            max_padded_chunk_s=args.max_padded_chunk_s,
            min_chunk_s=args.min_chunk_s,
            context_max_padding_s=args.context_max_padding_s,
            context_loss_weight=args.context_loss_weight,
        ),
    )


if __name__ == "__main__":
    main()
