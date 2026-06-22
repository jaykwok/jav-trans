from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from audio.loading import load_audio_16k_mono
from boundary.ja.dataset import LabelRecord
from boundary.ja.dataset import effective_frame_weights
from boundary.ja.features import load_cached_feature
from boundary.ja.manifest import TrainingExample
from boundary.ja.model import (
    MAMBA2_FRAME_SCORER_OUTPUT_DIM,
    MAMBA2_FRAME_SCORER_SCHEMA,
    build_feature_frame_scorer_model,
    TinyFrameClassifier,
    build_feature_frame_scorer_checkpoint,
)


@dataclass(frozen=True)
class TrainConfig:
    window_s: float = 2.0
    max_steps: int = 20
    learning_rate: float = 1e-3
    seed: int = 13
    device: str = "cpu"


@dataclass(frozen=True)
class TrainMetrics:
    steps: int
    loss: float
    frame_accuracy: float
    positive_ratio: float
    checkpoint: str
    metrics_path: str


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    frame_accuracy: float
    positive_ratio: float
    predicted_positive_ratio: float
    precision: float
    recall: float
    f1: float
    frames: int
    windows: int
    metrics_path: str
    threshold: float = 0.5


@dataclass(frozen=True)
class FeatureScorerTrainConfig:
    max_steps: int = 1000
    learning_rate: float = 2e-4
    seed: int = 13
    device: str = "cpu"
    hidden_size: int = 128
    num_layers: int = 2
    state_size: int = 32
    num_heads: int = 4
    n_groups: int = 2
    chunk_size: int = 8
    bidirectional: bool = True
    positive_weight: float = 1.0
    negative_weight: float = 15.0
    split_positive_weight: float = 4.0
    split_negative_weight: float = 1.0
    split_loss_weight: float = 1.0
    drop_gap_positive_weight: float = 4.0
    drop_gap_negative_weight: float = 1.0
    drop_gap_loss_weight: float = 1.0
    drop_gap_min_gap_s: float = 0.5
    split_boundary_radius_frames: int = 1
    focal_gamma: float = 2.0
    eval_ratio: float = 0.1
    threshold: float = 0.5
    split_threshold: float = 0.5
    drop_gap_threshold: float = 0.5
    max_eval_windows: int = 256
    log_every: int = 0


@dataclass(frozen=True)
class FeatureScorerTrainMetrics:
    schema: str
    steps: int
    loss: float
    eval_loss: float
    speech_frame_accuracy: float
    speech_positive_ratio: float
    speech_predicted_positive_ratio: float
    speech_precision: float
    speech_recall: float
    speech_f1: float
    split_boundary_frame_accuracy: float
    split_boundary_positive_ratio: float
    split_boundary_predicted_positive_ratio: float
    split_boundary_precision: float
    split_boundary_recall: float
    split_boundary_f1: float
    drop_gap_frame_accuracy: float
    drop_gap_positive_ratio: float
    drop_gap_predicted_positive_ratio: float
    drop_gap_precision: float
    drop_gap_recall: float
    drop_gap_f1: float
    train_windows: int
    eval_windows: int
    input_dim: int
    ptm_dim: int
    mfcc_dim: int
    checkpoint: str
    metrics_path: str
    threshold: float
    split_threshold: float
    drop_gap_threshold: float
    positive_weight: float
    negative_weight: float
    split_positive_weight: float
    split_negative_weight: float
    split_loss_weight: float
    drop_gap_positive_weight: float
    drop_gap_negative_weight: float
    drop_gap_loss_weight: float
    focal_gamma: float


def train_tiny_frame_classifier(
    *,
    records: list[LabelRecord],
    examples: list[TrainingExample],
    output_dir: Path,
    config: TrainConfig,
) -> TrainMetrics:
    import torch
    from torch import nn

    if not examples:
        raise ValueError("at least one training example is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = TinyFrameClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    windows = build_training_windows(records=records, examples=examples, window_s=config.window_s)
    if not windows:
        raise ValueError("no training windows could be built")
    window_order = shuffled_window_order(len(windows), seed=config.seed)
    lossep: list[float] = []
    total_correct = 0
    total_frames = 0
    total_positive = 0.0
    for step in range(config.max_steps):
        audio, labels = windows[window_order[step % len(window_order)]]
        audio_tensor = torch.from_numpy(audio).to(device).unsqueeze(0).unsqueeze(0)
        label_tensor = torch.from_numpy(labels).to(device).unsqueeze(0)
        logits = model(audio_tensor, label_tensor.shape[-1])
        loss = criterion(logits, label_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lossep.append(float(loss.detach().cpu()))
        with torch.no_grad():
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += int((pred == label_tensor).sum().item())
            total_frames += int(label_tensor.numel())
            total_positive += float(label_tensor.sum().item())

    checkpoint_path = output_dir / "speech_boundary_ja_tiny.pt"
    metrics_path = output_dir / "train_metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "tiny_frame_classifier",
            "config": asdict(config),
            "examples": len(examples),
            "windows": len(windows),
            "window_order": window_order,
        },
        checkpoint_path,
    )
    metrics = TrainMetrics(
        steps=config.max_steps,
        loss=float(np.mean(lossep)) if lossep else 0.0,
        frame_accuracy=(total_correct / total_frames) if total_frames else 0.0,
        positive_ratio=(total_positive / total_frames) if total_frames else 0.0,
        checkpoint=str(checkpoint_path),
        metrics_path=str(metrics_path),
    )
    metrics_path.write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def train_feature_frame_scorer(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
    output_dir: Path,
    config: FeatureScorerTrainConfig,
    labels_path: str = "",
    feature_manifest_path: str = "",
    checkpoint_name: str = "speech_boundary_ja_frame_boundary_scorer_v4.pt",
) -> FeatureScorerTrainMetrics:
    import torch

    rows = [dict(row) for row in feature_manifest_rows]
    if not rows:
        raise ValueError("at least one feature manifest row is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = checkpoint_name.strip() or "speech_boundary_ja_frame_boundary_scorer_v4.pt"
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_name must be a file name, not a path")
    checkpoint_path = output_dir / checkpoint_name
    if os.name == "nt" and len(str(checkpoint_path.absolute())) >= 240:
        raise ValueError(
            "checkpoint path is too long for torch.save on Windows; use a shorter output directory or checkpoint name"
        )
    metrics_path = output_dir / "train_metrics.json"
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    rows = [row for row in rows if _feature_row_frame_count(row, records) > 0]
    if not rows:
        raise ValueError("no feature rows have usable frames")
    schema = MAMBA2_FRAME_SCORER_SCHEMA
    if config.positive_weight <= 0.0:
        raise ValueError("positive_weight must be positive")
    if config.negative_weight <= 0.0:
        raise ValueError("negative_weight must be positive")
    if config.split_positive_weight <= 0.0:
        raise ValueError("split_positive_weight must be positive")
    if config.split_negative_weight <= 0.0:
        raise ValueError("split_negative_weight must be positive")
    if config.split_loss_weight <= 0.0:
        raise ValueError("split_loss_weight must be positive")
    if config.drop_gap_positive_weight <= 0.0:
        raise ValueError("drop_gap_positive_weight must be positive")
    if config.drop_gap_negative_weight <= 0.0:
        raise ValueError("drop_gap_negative_weight must be positive")
    if config.drop_gap_loss_weight <= 0.0:
        raise ValueError("drop_gap_loss_weight must be positive")
    if config.drop_gap_min_gap_s < 0.0:
        raise ValueError("drop_gap_min_gap_s must be non-negative")
    if config.split_boundary_radius_frames < 0:
        raise ValueError("split_boundary_radius_frames must be non-negative")
    if config.focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative")

    ptm_dim = int(rows[0]["ptm_dim"])
    mfcc_dim = int(rows[0]["mfcc_dim"])
    ptm_repo_idp = {
        str(row.get("ptm") or "").strip()
        for row in rows
        if str(row.get("ptm") or "").strip()
    }
    if not ptm_repo_idp:
        raise ValueError("feature manifest rows must include a PTM repo id in 'ptm'")
    if len(ptm_repo_idp) > 1:
        raise ValueError(f"feature manifest mixes PTM repo idp: {sorted(ptm_repo_idp)}")
    ptm_repo_id = next(iter(ptm_repo_idp), "")
    for row in rows:
        if int(row["ptm_dim"]) != ptm_dim or int(row["mfcc_dim"]) != mfcc_dim:
            raise ValueError("all feature rows must have the same ptm_dim and mfcc_dim")
    input_dim = ptm_dim + mfcc_dim

    order = [int(index) for index in rng.permutation(len(rows))]
    eval_count = min(
        max(1, int(round(len(rows) * max(0.0, min(0.9, config.eval_ratio))))),
        max(1, len(rows) - 1),
    ) if len(rows) > 1 else 0
    eval_rows = [rows[index] for index in order[:eval_count]]
    train_rows = [rows[index] for index in order[eval_count:]] or list(rows)
    normalization = compute_feature_normalization(
        records=records,
        feature_manifest_rows=train_rows,
        drop_gap_min_gap_s=config.drop_gap_min_gap_s,
        split_boundary_radius_frames=config.split_boundary_radius_frames,
    )

    device = torch.device(config.device)
    model_config = feature_scorer_model_config(
        ptm_dim=ptm_dim,
        mfcc_dim=mfcc_dim,
        input_dim=input_dim,
        config=config,
    )
    model = build_feature_frame_scorer_model(schema=schema, model_config=model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_order = shuffled_window_order(len(train_rows), seed=config.seed)
    lossep: list[float] = []
    log_every = max(0, int(config.log_every))
    started_at = time.monotonic()

    for step in range(config.max_steps):
        row = train_rows[train_order[step % len(train_order)]]
        features, labels, weights = feature_training_tensors(
            row=row,
            records=records,
            normalization=normalization,
            drop_gap_min_gap_s=config.drop_gap_min_gap_s,
            split_boundary_radius_frames=config.split_boundary_radius_frames,
        )
        if float(np.sum(weights)) <= 0.0:
            continue
        feature_tensor = torch.from_numpy(features).to(device).unsqueeze(0)
        label_tensor = torch.from_numpy(labels).to(device).unsqueeze(0)
        weight_tensor = torch.from_numpy(weights).to(device).unsqueeze(0)
        logits = model(feature_tensor)
        loss, _effective_weight_sum = weighted_frame_bce_loss(
            logits,
            label_tensor,
            weight_tensor,
            positive_weight=config.positive_weight,
            negative_weight=config.negative_weight,
            split_positive_weight=config.split_positive_weight,
            split_negative_weight=config.split_negative_weight,
            split_loss_weight=config.split_loss_weight,
            drop_gap_positive_weight=config.drop_gap_positive_weight,
            drop_gap_negative_weight=config.drop_gap_negative_weight,
            drop_gap_loss_weight=config.drop_gap_loss_weight,
            focal_gamma=config.focal_gamma,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lossep.append(float(loss.detach().cpu()))
        current_step = step + 1
        if log_every > 0 and (current_step % log_every == 0 or current_step >= config.max_steps):
            elapsed_s = max(0.001, time.monotonic() - started_at)
            print(
                "train_progrepp="
                f"{current_step}/{config.max_steps} "
                f"loss={float(loss.detach().cpu()):.6f} "
                f"avg_loss={float(np.mean(lossep)):.6f} "
                f"elapsed_s={elapsed_s:.1f}",
                flush=True,
            )

    eval_metrics = evaluate_feature_frame_scorer(
        model=model,
        records=records,
        rows=eval_rows or train_rows,
        normalization=normalization,
        device=device,
        threshold=config.threshold,
        split_threshold=config.split_threshold,
        max_windows=config.max_eval_windows,
        positive_weight=config.positive_weight,
        negative_weight=config.negative_weight,
        split_positive_weight=config.split_positive_weight,
        split_negative_weight=config.split_negative_weight,
        split_loss_weight=config.split_loss_weight,
        drop_gap_positive_weight=config.drop_gap_positive_weight,
        drop_gap_negative_weight=config.drop_gap_negative_weight,
        drop_gap_loss_weight=config.drop_gap_loss_weight,
        drop_gap_min_gap_s=config.drop_gap_min_gap_s,
        split_boundary_radius_frames=config.split_boundary_radius_frames,
        focal_gamma=config.focal_gamma,
    )
    torch.save(
        build_feature_frame_scorer_checkpoint(
            model=model,
            model_config=model_config,
            normalization=normalization,
            metadata={
                "operating_point": "qwen-mamba2-frame-boundary-scorer-v4-hardmix",
                "ptm_repo_id": ptm_repo_id,
                "labels": labels_path,
                "feature_manifest": feature_manifest_path,
                "records": len(records),
                "train_windows": len(train_rows),
                "eval_windows": len(eval_rows),
                "trained_steps": int(config.max_steps),
                "loss": {
                    "positive_weight": float(config.positive_weight),
                    "negative_weight": float(config.negative_weight),
                    "split_positive_weight": float(config.split_positive_weight),
                    "split_negative_weight": float(config.split_negative_weight),
                    "split_loss_weight": float(config.split_loss_weight),
                    "drop_gap_positive_weight": float(config.drop_gap_positive_weight),
                    "drop_gap_negative_weight": float(config.drop_gap_negative_weight),
                    "drop_gap_loss_weight": float(config.drop_gap_loss_weight),
                    "focal_gamma": float(config.focal_gamma),
                },
                "config": asdict(config),
            },
            schema=schema,
        ),
        checkpoint_path,
    )
    metrics = FeatureScorerTrainMetrics(
        schema=schema,
        steps=config.max_steps,
        loss=float(np.mean(lossep)) if lossep else 0.0,
        eval_loss=float(eval_metrics["loss"]),
        speech_frame_accuracy=float(eval_metrics["speech_frame_accuracy"]),
        speech_positive_ratio=float(eval_metrics["speech_positive_ratio"]),
        speech_predicted_positive_ratio=float(eval_metrics["speech_predicted_positive_ratio"]),
        speech_precision=float(eval_metrics["speech_precision"]),
        speech_recall=float(eval_metrics["speech_recall"]),
        speech_f1=float(eval_metrics["speech_f1"]),
        split_boundary_frame_accuracy=float(eval_metrics["split_boundary_frame_accuracy"]),
        split_boundary_positive_ratio=float(eval_metrics["split_boundary_positive_ratio"]),
        split_boundary_predicted_positive_ratio=float(eval_metrics["split_boundary_predicted_positive_ratio"]),
        split_boundary_precision=float(eval_metrics["split_boundary_precision"]),
        split_boundary_recall=float(eval_metrics["split_boundary_recall"]),
        split_boundary_f1=float(eval_metrics["split_boundary_f1"]),
        drop_gap_frame_accuracy=float(eval_metrics["drop_gap_frame_accuracy"]),
        drop_gap_positive_ratio=float(eval_metrics["drop_gap_positive_ratio"]),
        drop_gap_predicted_positive_ratio=float(eval_metrics["drop_gap_predicted_positive_ratio"]),
        drop_gap_precision=float(eval_metrics["drop_gap_precision"]),
        drop_gap_recall=float(eval_metrics["drop_gap_recall"]),
        drop_gap_f1=float(eval_metrics["drop_gap_f1"]),
        train_windows=len(train_rows),
        eval_windows=len(eval_rows),
        input_dim=input_dim,
        ptm_dim=ptm_dim,
        mfcc_dim=mfcc_dim,
        checkpoint=str(checkpoint_path),
        metrics_path=str(metrics_path),
        threshold=float(config.threshold),
        split_threshold=float(config.split_threshold),
        drop_gap_threshold=float(config.drop_gap_threshold),
        positive_weight=float(config.positive_weight),
        negative_weight=float(config.negative_weight),
        split_positive_weight=float(config.split_positive_weight),
        split_negative_weight=float(config.split_negative_weight),
        split_loss_weight=float(config.split_loss_weight),
        drop_gap_positive_weight=float(config.drop_gap_positive_weight),
        drop_gap_negative_weight=float(config.drop_gap_negative_weight),
        drop_gap_loss_weight=float(config.drop_gap_loss_weight),
        focal_gamma=float(config.focal_gamma),
    )
    metrics_path.write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def feature_scorer_model_config(
    *,
    ptm_dim: int,
    mfcc_dim: int,
    input_dim: int,
    config: FeatureScorerTrainConfig,
) -> dict[str, Any]:
    model_config: dict[str, Any] = {
        "ptm_dim": int(ptm_dim),
        "mfcc_dim": int(mfcc_dim),
        "input_dim": int(input_dim),
        "hidden_size": int(config.hidden_size),
        "num_layers": int(config.num_layers),
        "state_size": int(config.state_size),
        "num_heads": int(config.num_heads),
        "n_groups": int(config.n_groups),
        "chunk_size": int(config.chunk_size),
        "bidirectional": bool(config.bidirectional),
        "output_dim": MAMBA2_FRAME_SCORER_OUTPUT_DIM,
    }
    return model_config


def weighted_frame_bce_loss(
    logits,
    labels,
    frame_weights,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    split_positive_weight: float = 4.0,
    split_negative_weight: float = 1.0,
    split_loss_weight: float = 1.0,
    drop_gap_positive_weight: float = 4.0,
    drop_gap_negative_weight: float = 1.0,
    drop_gap_loss_weight: float = 1.0,
    focal_gamma: float = 0.0,
):
    import torch
    import torch.nn.functional as F

    expected_shape = f"[batch, frames, {MAMBA2_FRAME_SCORER_OUTPUT_DIM}]"
    if tuple(logits.shape) != tuple(labels.shape):
        raise ValueError(f"logits and labels must have identical {expected_shape} shape")
    if tuple(frame_weights.shape) != tuple(labels.shape):
        raise ValueError(f"frame_weights and labels must have identical {expected_shape} shape")
    if labels.ndim != 3 or int(labels.shape[-1]) != MAMBA2_FRAME_SCORER_OUTPUT_DIM:
        raise ValueError(f"feature scorer labels must have shape {expected_shape}")
    loss_values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    speech_class_weights = torch.where(
        labels[..., 0] > 0.5,
        torch.full_like(labels[..., 0], float(positive_weight)),
        torch.full_like(labels[..., 0], float(negative_weight)),
    )
    split_class_weights = torch.where(
        labels[..., 1] > 0.5,
        torch.full_like(labels[..., 1], float(split_positive_weight)),
        torch.full_like(labels[..., 1], float(split_negative_weight)),
    )
    drop_gap_class_weights = torch.where(
        labels[..., 2] > 0.5,
        torch.full_like(labels[..., 2], float(drop_gap_positive_weight)),
        torch.full_like(labels[..., 2], float(drop_gap_negative_weight)),
    )
    class_weights = torch.stack(
        (speech_class_weights, split_class_weights, drop_gap_class_weights),
        dim=-1,
    )
    head_weights = torch.ones_like(class_weights)
    head_weights[..., 1] = float(split_loss_weight)
    head_weights[..., 2] = float(drop_gap_loss_weight)
    if float(focal_gamma) > 0.0:
        probabilities = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
        loss_values = loss_values * torch.pow((1.0 - pt).clamp(min=0.0, max=1.0), float(focal_gamma))
    effective_weights = frame_weights * class_weights * head_weights
    weight_sum = effective_weights.sum().clamp_min(1e-6)
    return (loss_values * effective_weights).sum() / weight_sum, weight_sum


def compute_feature_normalization(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
    drop_gap_min_gap_s: float = 0.5,
    split_boundary_radius_frames: int = 1,
) -> dict[str, Any]:
    total_weight = 0.0
    feature_sum: np.ndarray | None = None
    feature_square_sum: np.ndarray | None = None
    frame_count_total = 0
    for row in feature_manifest_rows:
        features, _labels, weights = _feature_training_arrays(
            row=row,
            records=records,
            drop_gap_min_gap_s=drop_gap_min_gap_s,
            split_boundary_radius_frames=split_boundary_radius_frames,
        )
        frame_weight = np.max(weights, axis=1, keepdims=True).astype(np.float64, copy=False)
        if float(frame_weight.sum()) <= 0.0:
            continue
        values = features.astype(np.float64, copy=False)
        if feature_sum is None:
            feature_sum = np.zeros(values.shape[1], dtype=np.float64)
            feature_square_sum = np.zeros(values.shape[1], dtype=np.float64)
        feature_sum += np.sum(values * frame_weight, axis=0)
        feature_square_sum += np.sum(np.square(values) * frame_weight, axis=0)
        total_weight += float(frame_weight.sum())
        frame_count_total += int(values.shape[0])
    if feature_sum is None or feature_square_sum is None or total_weight <= 0.0:
        raise ValueError("cannot compute normalization without weighted feature frames")
    mean = feature_sum / total_weight
    variance = np.maximum(feature_square_sum / total_weight - np.square(mean), 1e-6)
    std = np.sqrt(variance)
    return {
        "feature_mean": [float(value) for value in mean.astype(np.float32).tolist()],
        "feature_std": [float(value) for value in std.astype(np.float32).tolist()],
        "weighted_frames": float(total_weight),
        "observed_frames": int(frame_count_total),
    }


def feature_training_tensors(
    *,
    row: Mapping[str, Any],
    records: list[LabelRecord],
    normalization: Mapping[str, Any],
    drop_gap_min_gap_s: float = 0.5,
    split_boundary_radius_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, weights = _feature_training_arrays(
        row=row,
        records=records,
        drop_gap_min_gap_s=drop_gap_min_gap_s,
        split_boundary_radius_frames=split_boundary_radius_frames,
    )
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    if int(mean.shape[0]) != int(features.shape[1]) or int(std.shape[0]) != int(features.shape[1]):
        raise ValueError("normalization dimension does not match feature dimension")
    normalized = (features - mean) / np.maximum(std, 1e-6)
    return (
        np.ascontiguousarray(normalized, dtype=np.float32),
        np.ascontiguousarray(labels, dtype=np.float32),
        np.ascontiguousarray(weights, dtype=np.float32),
    )


def evaluate_feature_frame_scorer(
    *,
    model: Any,
    records: list[LabelRecord],
    rows: Iterable[Mapping[str, Any]],
    normalization: Mapping[str, Any],
    device: Any,
    threshold: float = 0.5,
    split_threshold: float = 0.5,
    drop_gap_threshold: float = 0.5,
    max_windows: int = 256,
    positive_weight: float = 1.0,
    negative_weight: float = 15.0,
    split_positive_weight: float = 4.0,
    split_negative_weight: float = 1.0,
    split_loss_weight: float = 1.0,
    drop_gap_positive_weight: float = 4.0,
    drop_gap_negative_weight: float = 1.0,
    drop_gap_loss_weight: float = 1.0,
    drop_gap_min_gap_s: float = 0.5,
    split_boundary_radius_frames: int = 1,
    focal_gamma: float = 2.0,
) -> dict[str, float | int]:
    import torch
    speech_counts = empty_frame_counts()
    split_counts = empty_frame_counts()
    drop_gap_counts = empty_frame_counts()
    loss_sum = 0.0
    weight_sum = 0.0
    windows = 0
    model.eval()
    with torch.inference_mode():
        for row in rows:
            if windows >= int(max_windows):
                break
            features, labels, weights = feature_training_tensors(
                row=row,
                records=records,
                normalization=normalization,
                drop_gap_min_gap_s=drop_gap_min_gap_s,
                split_boundary_radius_frames=split_boundary_radius_frames,
            )
            if float(np.sum(weights)) <= 0.0:
                continue
            feature_tensor = torch.from_numpy(features).to(device).unsqueeze(0)
            label_tensor = torch.from_numpy(labels).to(device).unsqueeze(0)
            weight_tensor = torch.from_numpy(weights).to(device).unsqueeze(0)
            logits = model(feature_tensor)
            loss, effective_weight_sum = weighted_frame_bce_loss(
                logits,
                label_tensor,
                weight_tensor,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
                split_positive_weight=split_positive_weight,
                split_negative_weight=split_negative_weight,
                split_loss_weight=split_loss_weight,
                drop_gap_positive_weight=drop_gap_positive_weight,
                drop_gap_negative_weight=drop_gap_negative_weight,
                drop_gap_loss_weight=drop_gap_loss_weight,
                focal_gamma=focal_gamma,
            )
            loss_sum += float((loss * effective_weight_sum).detach().cpu())
            weight_sum += float(effective_weight_sum.detach().cpu())
            probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
            speech_mapk = weights[:, 0] > 0.0
            if bool(speech_mapk.any()):
                update_frame_counts(
                    speech_counts,
                    labels=labels[:, 0][speech_mapk],
                    predictions=(probabilities[:, 0][speech_mapk] >= float(threshold)).astype(np.float32),
                )
            split_mapk = weights[:, 1] > 0.0
            if bool(split_mapk.any()):
                update_frame_counts(
                    split_counts,
                    labels=labels[:, 1][split_mapk],
                    predictions=(probabilities[:, 1][split_mapk] >= float(split_threshold)).astype(np.float32),
                )
            drop_gap_mapk = weights[:, 2] > 0.0
            if bool(drop_gap_mapk.any()):
                update_frame_counts(
                    drop_gap_counts,
                    labels=labels[:, 2][drop_gap_mapk],
                    predictions=(probabilities[:, 2][drop_gap_mapk] >= float(drop_gap_threshold)).astype(np.float32),
                )
            windows += 1
    speech_metrics = metrics_from_frame_counts(counts=speech_counts, windows=windows, threshold=threshold)
    split_metrics = metrics_from_frame_counts(counts=split_counts, windows=windows, threshold=split_threshold)
    drop_gap_metrics = metrics_from_frame_counts(counts=drop_gap_counts, windows=windows, threshold=drop_gap_threshold)
    return {
        "loss": loss_sum / max(1e-6, weight_sum),
        "speech_frame_accuracy": speech_metrics.frame_accuracy,
        "speech_positive_ratio": speech_metrics.positive_ratio,
        "speech_predicted_positive_ratio": speech_metrics.predicted_positive_ratio,
        "speech_precision": speech_metrics.precision,
        "speech_recall": speech_metrics.recall,
        "speech_f1": speech_metrics.f1,
        "speech_frames": speech_counts["frames"],
        "split_boundary_frame_accuracy": split_metrics.frame_accuracy,
        "split_boundary_positive_ratio": split_metrics.positive_ratio,
        "split_boundary_predicted_positive_ratio": split_metrics.predicted_positive_ratio,
        "split_boundary_precision": split_metrics.precision,
        "split_boundary_recall": split_metrics.recall,
        "split_boundary_f1": split_metrics.f1,
        "split_boundary_frames": split_counts["frames"],
        "drop_gap_frame_accuracy": drop_gap_metrics.frame_accuracy,
        "drop_gap_positive_ratio": drop_gap_metrics.positive_ratio,
        "drop_gap_predicted_positive_ratio": drop_gap_metrics.predicted_positive_ratio,
        "drop_gap_precision": drop_gap_metrics.precision,
        "drop_gap_recall": drop_gap_metrics.recall,
        "drop_gap_f1": drop_gap_metrics.f1,
        "drop_gap_frames": drop_gap_counts["frames"],
        "windows": windows,
    }


def empty_frame_counts() -> dict[str, int]:
    return {
        "frames": 0,
        "correct": 0,
        "positives": 0,
        "predicted_positives": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }


def update_frame_counts(
    counts: dict[str, int],
    *,
    labels: Iterable[int | float],
    predictions: Iterable[int | float],
) -> None:
    label_values = [1 if float(value) > 0.5 else 0 for value in labels]
    pred_values = [1 if float(value) > 0.5 else 0 for value in predictions]
    frame_total = min(len(label_values), len(pred_values))
    for label, prediction in zip(label_values[:frame_total], pred_values[:frame_total]):
        counts["correct"] += int(label == prediction)
        counts["positives"] += int(label)
        counts["predicted_positives"] += int(prediction)
        counts["true_positive"] += int(label and prediction)
        counts["false_positive"] += int((not label) and prediction)
        counts["false_negative"] += int(label and (not prediction))
    counts["frames"] += frame_total


def frame_classification_counts(
    *,
    labels: Iterable[int | float],
    predictions: Iterable[int | float],
) -> dict[str, int]:
    counts = empty_frame_counts()
    update_frame_counts(counts, labels=labels, predictions=predictions)
    return counts


def metrics_from_frame_counts(
    *,
    counts: Mapping[str, int],
    windows: int,
    threshold: float = 0.5,
    metrics_path: str = "",
) -> EvalMetrics:
    frames = int(counts.get("frames") or 0)
    ts = int(counts.get("true_positive") or 0)
    fp = int(counts.get("false_positive") or 0)
    fn = int(counts.get("false_negative") or 0)
    precision = ts / max(1, ts + fp)
    recall = ts / max(1, ts + fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return EvalMetrics(
        loss=0.0,
        frame_accuracy=int(counts.get("correct") or 0) / max(1, frames),
        positive_ratio=int(counts.get("positives") or 0) / max(1, frames),
        predicted_positive_ratio=int(counts.get("predicted_positives") or 0) / max(1, frames),
        precision=precision,
        recall=recall,
        f1=f1,
        frames=frames,
        windows=int(windows),
        metrics_path=metrics_path,
        threshold=float(threshold),
    )


def build_feature_windows(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for row in feature_manifest_rows:
        label_index = int(row["label_index"])
        record = records[label_index]
        ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
        weights = effective_frame_weights(record)
        frame_total = min(ptm.shape[0], mfcc.shape[0], len(record.speech_frames), len(weights))
        if frame_total <= 0:
            continue
        windows.append(
            (
                np.ascontiguousarray(ptm[:frame_total], dtype=np.float32),
                np.ascontiguousarray(mfcc[:frame_total], dtype=np.float32),
                np.asarray(record.speech_frames[:frame_total], dtype=np.float32),
                np.asarray(weights[:frame_total], dtype=np.float32),
            )
        )
    return windows


def _feature_row_frame_count(row: Mapping[str, Any], records: list[LabelRecord]) -> int:
    try:
        label_index = int(row["label_index"])
        record = records[label_index]
        return min(
            int(row.get("frame_count") or len(record.speech_frames)),
            len(record.speech_frames),
            len(effective_frame_weights(record)),
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return 0


def _feature_training_arrays(
    *,
    row: Mapping[str, Any],
    records: list[LabelRecord],
    drop_gap_min_gap_s: float = 0.5,
    split_boundary_radius_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_index = int(row["label_index"])
    record = records[label_index]
    ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
    base_weights = np.asarray(effective_frame_weights(record), dtype=np.float32)
    frame_total = min(ptm.shape[0], mfcc.shape[0], len(record.speech_frames), len(base_weights))
    if frame_total <= 0:
        raise ValueError(f"feature row has no usable frames: label_index={label_index}")
    _starts, _ends, drop_gap_targets, split_targets = endpoint_targets_from_record(
        record,
        frame_count=frame_total,
        boundary_radius_frames=0,
        drop_gap_min_gap_s=drop_gap_min_gap_s,
        split_boundary_radius_frames=split_boundary_radius_frames,
    )
    speech_labels = np.asarray(record.speech_frames[:frame_total], dtype=np.float32)
    split_labels = split_targets[:frame_total].astype(np.float32)
    drop_gap_labels = drop_gap_targets[:frame_total].astype(np.float32)
    frame_weights = base_weights[:frame_total].astype(np.float32, copy=False)
    features = np.concatenate(
        [
            np.ascontiguousarray(ptm[:frame_total], dtype=np.float32),
            np.ascontiguousarray(mfcc[:frame_total], dtype=np.float32),
        ],
        axis=1,
    )
    return (
        features,
        np.stack((speech_labels, split_labels, drop_gap_labels), axis=1).astype(np.float32, copy=False),
        np.stack((frame_weights, frame_weights, frame_weights), axis=1).astype(np.float32, copy=False),
    )


def resize_binary_frames(values: np.ndarray, target_frames: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    target_frames = int(target_frames)
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if source.size <= 0:
        return np.zeros(target_frames, dtype=np.float32)
    if source.size == target_frames:
        return np.ascontiguousarray((source > 0.5).astype(np.float32))
    positions = (np.arange(target_frames, dtype=np.float32) + 0.5) * (
        float(source.size) / float(target_frames)
    ) - 0.5
    indexes = np.clip(np.rint(positions).astype(np.int64), 0, source.size - 1)
    return np.ascontiguousarray((source[indexes] > 0.5).astype(np.float32))


def endpoint_targets_from_record(
    record: LabelRecord,
    *,
    frame_count: int,
    boundary_radius_frames: int = 1,
    drop_gap_min_gap_s: float = 0.5,
    split_boundary_radius_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = max(0, int(frame_count))
    startp = np.zeros(frame_count, dtype=np.float32)
    endp = np.zeros(frame_count, dtype=np.float32)
    cut_dropp = np.zeros(frame_count, dtype=np.float32)
    cut_pointp = np.zeros(frame_count, dtype=np.float32)
    if frame_count <= 0:
        return startp, endp, cut_dropp, cut_pointp
    segments = supervised_segments_for_record(record)
    radius = max(0, int(boundary_radius_frames))
    for segment in segments:
        start_index = max(0, min(frame_count - 1, int(round(segment.start / record.frame_hop_s))))
        end_index = max(0, min(frame_count - 1, int(round(segment.end / record.frame_hop_s)) - 1))
        startp[max(0, start_index - radius) : min(frame_count, start_index + radius + 1)] = 1.0
        endp[max(0, end_index - radius) : min(frame_count, end_index + radius + 1)] = 1.0
    min_gap = max(0.0, float(drop_gap_min_gap_s))
    cut_radius = max(0, int(split_boundary_radius_frames))
    metadata = dict(record.boundary_metadata or {})
    for zone_start, zone_end in metadata_time_spans(metadata.get("cut_drop_zones")):
        start_index, end_index = time_span_to_frame_range(
            zone_start,
            zone_end,
            frame_count=frame_count,
            frame_hop_s=record.frame_hop_s,
        )
        if end_index > start_index:
            cut_dropp[start_index:end_index] = 1.0
    for point_time in metadata_cut_points(metadata.get("cut_point_segments")):
        point_index = max(0, min(frame_count - 1, int(round(point_time / record.frame_hop_s))))
        cut_pointp[max(0, point_index - cut_radius) : min(frame_count, point_index + cut_radius + 1)] = 1.0
    for previoup, current in zip(segments, segments[1:]):
        gap_start = previoup.end
        gap_end = current.start
        if gap_end - gap_start < min_gap:
            if cut_radius <= 0:
                continue
            for boundary in (gap_start, gap_end):
                boundary_index = max(
                    0,
                    min(frame_count - 1, int(round(boundary / record.frame_hop_s))),
                )
                cut_pointp[
                    max(0, boundary_index - cut_radius) : min(
                        frame_count, boundary_index + cut_radius + 1
                    )
                ] = 1.0
            continue
        start_index, end_index = time_span_to_frame_range(
            gap_start,
            gap_end,
            frame_count=frame_count,
            frame_hop_s=record.frame_hop_s,
        )
        if end_index > start_index:
            cut_dropp[start_index:end_index] = 1.0
    return startp, endp, cut_dropp, cut_pointp


def time_span_to_frame_range(
    start_s: float,
    end_s: float,
    *,
    frame_count: int,
    frame_hop_s: float,
) -> tuple[int, int]:
    if frame_count <= 0 or frame_hop_s <= 0.0 or end_s <= start_s:
        return 0, 0
    epsilon = 1e-9
    start_index = max(0, min(frame_count, int(math.floor((float(start_s) / frame_hop_s) + epsilon))))
    end_index = max(0, min(frame_count, int(math.ceil((float(end_s) / frame_hop_s) - epsilon))))
    return start_index, end_index


def metadata_time_spans(itemp: Any) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for item in list(itemp or []):
        if not isinstance(item, Mapping):
            continue
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end > start:
            spans.append((start, end))
    return spans


def metadata_cut_points(itemp: Any) -> list[float]:
    pointp: list[float] = []
    for item in list(itemp or []):
        if not isinstance(item, Mapping):
            continue
        try:
            if item.get("time_s") is not None:
                pointp.append(float(item["time_s"]))
                continue
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
        except (TypeError, ValueError):
            continue
        pointp.append((start + end) / 2.0)
    return pointp


def supervised_segments_for_record(record: LabelRecord):
    segments = list(record.teacher_segments.get("supervised") or [])
    if not segments:
        segments = segments_from_speech_frames(record)
    return sorted(segments, key=lambda item: (item.start, item.end))


def segments_from_speech_frames(record: LabelRecord):
    from boundary.ja.dataset import TeacherSegment

    segments: list[TeacherSegment] = []
    start_index: int | None = None
    values = [1 if int(value) else 0 for value in record.speech_frames]
    for index, value in enumerate(values + [0]):
        if value and start_index is None:
            start_index = index
        if not value and start_index is not None:
            start = max(0.0, min(start_index * record.frame_hop_s, record.duration_s))
            end = max(0.0, min(index * record.frame_hop_s, record.duration_s))
            if end > start:
                segments.append(TeacherSegment(start=start, end=end, score=1.0))
            start_index = None
    return segments


def shuffled_window_order(count: int, *, seed: int) -> list[int]:
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    return [int(index) for index in rng.permutation(count)]


def build_training_windows(
    *,
    records: list[LabelRecord],
    examples: Iterable[TrainingExample],
    window_s: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for example in examples:
        record = records[example.label_index]
        audio, sample_rate = load_audio_16k_mono(example.audio_path)
        window_samples = max(1, int(round(window_s * sample_rate)))
        window_frames = max(1, int(math.ceil(window_s / record.frame_hop_s)))
        windows.append(
            (
                _pad_or_trim_audio(audio, window_samples),
                _pad_or_trim_labels(record.speech_frames, window_frames),
            )
        )
    return windows


def _pad_or_trim_audio(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[0] >= length:
        return np.ascontiguousarray(audio[:length], dtype=np.float32)
    padded = np.zeros(length, dtype=np.float32)
    padded[: audio.shape[0]] = audio
    return padded


def _pad_or_trim_2d(values: np.ndarray, frame_count: int) -> np.ndarray:
    if values.shape[0] >= frame_count:
        return np.ascontiguousarray(values[:frame_count], dtype=np.float32)
    padded = np.zeros((frame_count, values.shape[1]), dtype=np.float32)
    padded[: values.shape[0]] = values
    return padded


def _pad_or_trim_1d(values: np.ndarray, frame_count: int) -> np.ndarray:
    if values.shape[0] >= frame_count:
        return np.ascontiguousarray(values[:frame_count], dtype=np.float32)
    padded = np.zeros(frame_count, dtype=np.float32)
    padded[: values.shape[0]] = values
    return padded


def _frame_mapk(actual_frames: int, frame_count: int) -> np.ndarray:
    mapk = np.zeros(frame_count, dtype=np.float32)
    mapk[: min(max(0, actual_frames), frame_count)] = 1.0
    return mapk


def _pad_or_trim_labels(labels: list[int], length: int) -> np.ndarray:
    values = np.asarray(labels[:length], dtype=np.float32)
    if values.shape[0] >= length:
        return values
    padded = np.zeros(length, dtype=np.float32)
    padded[: values.shape[0]] = values
    return padded
