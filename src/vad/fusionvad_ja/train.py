from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from audio.loading import load_audio_16k_mono
from vad.fusionvad_ja.dataset import LabelRecord
from vad.fusionvad_ja.dataset import effective_frame_weights
from vad.fusionvad_ja.features import load_cached_feature
from vad.fusionvad_ja.manifest import TrainingExample
from vad.fusionvad_ja.model import (
    AdditionFusionBiLSTM,
    AdditionFusionEndpointBiLSTM,
    AdditionFusionImitationBiLSTM,
    count_trainable_parameters,
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
class FeatureTrainConfig:
    max_steps: int = 20
    learning_rate: float = 1e-3
    seed: int = 13
    device: str = "cpu"
    fusion_dim: int = 256
    hidden_dim: int = 192
    layers: int = 2
    dropout: float = 0.1
    max_trainable_parameters: int = 2_000_000
    log_interval_steps: int = 0
    batch_size: int = 1
    init_checkpoint: str | None = None
    positive_loss_weight: float = 1.0
    boundary_loss_weight: float = 0.0
    gap_loss_weight: float = 0.0


@dataclass(frozen=True)
class EndpointRefinerTrainConfig:
    max_steps: int = 20
    learning_rate: float = 1e-3
    seed: int = 13
    device: str = "cpu"
    fusion_dim: int = 256
    hidden_dim: int = 192
    layers: int = 2
    dropout: float = 0.1
    max_trainable_parameters: int = 2_000_000
    log_interval_steps: int = 0
    batch_size: int = 1
    positive_loss_weight: float = 1.0
    speech_loss_weight: float = 0.5
    boundary_loss_weight: float = 0.0
    start_loss_weight: float = 2.0
    end_loss_weight: float = 1.5
    internal_gap_loss_weight: float = 1.0
    cut_drop_loss_weight: float = 1.0
    cut_point_loss_weight: float = 1.0
    start_positive_loss_weight: float = 1.0
    end_positive_loss_weight: float = 1.0
    cut_drop_positive_loss_weight: float = 1.0
    cut_point_positive_loss_weight: float = 1.0
    boundary_radius_frames: int = 1
    cut_min_gap_s: float = 0.5
    cut_boundary_radius_frames: int = 0
    save_interval_steps: int = 0
    init_checkpoint: str | None = None


@dataclass(frozen=True)
class ImitationTrainConfig:
    max_steps: int = 20
    learning_rate: float = 1e-3
    seed: int = 13
    device: str = "cpu"
    fusion_dim: int = 256
    hidden_dim: int = 192
    layers: int = 2
    dropout: float = 0.1
    max_trainable_parameters: int = 2_000_000
    log_interval_steps: int = 0
    batch_size: int = 1
    split_loss_weight: float = 1.0
    drop_gap_loss_weight: float = 1.0
    split_positive_loss_weight: float = 1.0
    drop_gap_positive_loss_weight: float = 1.0
    save_interval_steps: int = 0
    window_frames: int = 128
    positive_window_ratio: float = 0.9
    positive_jitter_frames: int = 16
    balanced_frame_loss: bool = True


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
    losses: list[float] = []
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
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += int((pred == label_tensor).sum().item())
            total_frames += int(label_tensor.numel())
            total_positive += float(label_tensor.sum().item())

    checkpoint_path = output_dir / "fusionvad_ja_tiny.pt"
    metrics_path = output_dir / "train_metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "examples": len(examples),
            "windows": len(windows),
            "window_order": window_order,
        },
        checkpoint_path,
    )
    metrics = TrainMetrics(
        steps=config.max_steps,
        loss=float(np.mean(losses)) if losses else 0.0,
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


def evaluate_addition_fusion_classifier(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: list[Mapping[str, Any]],
    checkpoint_path: Path,
    output_dir: Path,
    device: str = "cpu",
    threshold: float = 0.5,
) -> EvalMetrics:
    import torch
    from torch import nn

    output_dir.mkdir(parents=True, exist_ok=True)
    windows = build_feature_windows(records=records, feature_manifest_rows=feature_manifest_rows)
    if not windows:
        raise ValueError("no feature windows could be built")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    whisper_dim = int(checkpoint["whisper_dim"])
    mfcc_dim = int(checkpoint["mfcc_dim"])
    config = dict(checkpoint.get("config") or {})
    model = AdditionFusionBiLSTM(
        whisper_dim=whisper_dim,
        mfcc_dim=mfcc_dim,
        fusion_dim=int(config.get("fusion_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 192)),
        layers=int(config.get("layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    losses: list[float] = []
    total_correct = 0
    total_frames = 0
    total_positive = 0.0
    total_predicted_positive = 0.0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    with torch.inference_mode():
        for whisper, mfcc, labels, weights in windows:
            whisper_tensor = torch.from_numpy(whisper).to(device).unsqueeze(0)
            mfcc_tensor = torch.from_numpy(mfcc).to(device).unsqueeze(0)
            label_tensor = torch.from_numpy(labels).to(device).unsqueeze(0)
            weight_tensor = torch.from_numpy(weights).to(device).unsqueeze(0)
            logits = model(whisper_tensor, mfcc_tensor)
            loss = (criterion(logits, label_tensor) * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)
            pred = (torch.sigmoid(logits) >= threshold).float()
            losses.append(float(loss.detach().cpu()))
            counts = frame_classification_counts(
                labels=label_tensor.detach().cpu().numpy().reshape(-1),
                predictions=pred.detach().cpu().numpy().reshape(-1),
                weights=weight_tensor.detach().cpu().numpy().reshape(-1),
            )
            total_correct += counts["correct"]
            total_frames += counts["frames"]
            total_positive += counts["positives"]
            total_predicted_positive += counts["predicted_positives"]
            true_positive += counts["true_positive"]
            false_positive += counts["false_positive"]
            false_negative += counts["false_negative"]

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision or recall) else 0.0
    metrics_path = output_dir / "eval_metrics.json"
    metrics = EvalMetrics(
        loss=float(np.mean(losses)) if losses else 0.0,
        frame_accuracy=(total_correct / total_frames) if total_frames else 0.0,
        positive_ratio=(total_positive / total_frames) if total_frames else 0.0,
        predicted_positive_ratio=(total_predicted_positive / total_frames) if total_frames else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        frames=total_frames,
        windows=len(windows),
        metrics_path=str(metrics_path),
        threshold=float(threshold),
    )
    metrics_path.write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def frame_classification_counts(
    *,
    labels: np.ndarray | Iterable[float | int | bool],
    predictions: np.ndarray | Iterable[float | int | bool],
    weights: np.ndarray | Iterable[float | int | bool] | None = None,
) -> dict[str, int]:
    label_values = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels).astype(bool)
    prediction_values = np.asarray(
        list(predictions) if not isinstance(predictions, np.ndarray) else predictions
    ).astype(bool)
    frame_total = min(label_values.size, prediction_values.size)
    if weights is not None:
        weight_values = np.asarray(list(weights) if not isinstance(weights, np.ndarray) else weights, dtype=np.float32)
        frame_total = min(frame_total, weight_values.size)
        active = weight_values.reshape(-1)[:frame_total] > 0.0
    else:
        active = np.ones(frame_total, dtype=bool)
    label_values = label_values.reshape(-1)[:frame_total]
    prediction_values = prediction_values.reshape(-1)[:frame_total]
    label_values = label_values[active]
    prediction_values = prediction_values[active]
    frame_total = int(label_values.size)
    return {
        "frames": frame_total,
        "correct": int(np.equal(label_values, prediction_values).sum()),
        "positives": int(label_values.sum()),
        "predicted_positives": int(prediction_values.sum()),
        "true_positive": int(np.logical_and(prediction_values, label_values).sum()),
        "false_positive": int(np.logical_and(prediction_values, np.logical_not(label_values)).sum()),
        "false_negative": int(np.logical_and(np.logical_not(prediction_values), label_values).sum()),
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


def merge_frame_counts(target: dict[str, int], update: Mapping[str, int]) -> None:
    for key in empty_frame_counts():
        target[key] = int(target.get(key, 0)) + int(update.get(key, 0))


def metrics_from_frame_counts(
    *,
    counts: Mapping[str, int],
    windows: int,
    metrics_path: Path | None = None,
    loss: float = 0.0,
    threshold: float = 0.5,
) -> EvalMetrics:
    total_frames = int(counts.get("frames", 0))
    true_positive = int(counts.get("true_positive", 0))
    false_positive = int(counts.get("false_positive", 0))
    false_negative = int(counts.get("false_negative", 0))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision or recall) else 0.0
    return EvalMetrics(
        loss=float(loss),
        frame_accuracy=(int(counts.get("correct", 0)) / total_frames) if total_frames else 0.0,
        positive_ratio=(int(counts.get("positives", 0)) / total_frames) if total_frames else 0.0,
        predicted_positive_ratio=(
            int(counts.get("predicted_positives", 0)) / total_frames
        )
        if total_frames
        else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        frames=total_frames,
        windows=int(windows),
        metrics_path=str(metrics_path or ""),
        threshold=float(threshold),
    )


def train_addition_fusion_classifier(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: list[Mapping[str, Any]],
    output_dir: Path,
    config: FeatureTrainConfig,
) -> TrainMetrics:
    import torch
    from torch import nn

    if not feature_manifest_rows:
        raise ValueError("at least one cached feature row is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    windows = build_feature_windows(records=records, feature_manifest_rows=feature_manifest_rows)
    if not windows:
        raise ValueError("no feature windows could be built")
    whisper_dim = int(windows[0][0].shape[-1])
    mfcc_dim = int(windows[0][1].shape[-1])
    init_checkpoint: Mapping[str, Any] | None = None
    init_checkpoint_path: Path | None = None
    model_config = {
        "fusion_dim": config.fusion_dim,
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "dropout": config.dropout,
    }
    if config.init_checkpoint:
        init_checkpoint_path = Path(config.init_checkpoint)
        init_checkpoint = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
        init_whisper_dim = int(init_checkpoint.get("whisper_dim", -1))
        init_mfcc_dim = int(init_checkpoint.get("mfcc_dim", -1))
        if init_whisper_dim != whisper_dim or init_mfcc_dim != mfcc_dim:
            raise ValueError(
                "init checkpoint feature dimensions do not match current cache: "
                f"checkpoint=({init_whisper_dim}, {init_mfcc_dim}) current=({whisper_dim}, {mfcc_dim})"
            )
        checkpoint_config = dict(init_checkpoint.get("config") or {})
        for key in ("fusion_dim", "hidden_dim", "layers"):
            checkpoint_value = int(checkpoint_config.get(key, model_config[key]))
            if checkpoint_value != int(model_config[key]):
                raise ValueError(
                    f"init checkpoint {key}={checkpoint_value} does not match training config {model_config[key]}"
                )
        if "dropout" in checkpoint_config:
            model_config["dropout"] = float(checkpoint_config["dropout"])

    model = AdditionFusionBiLSTM(
        whisper_dim=whisper_dim,
        mfcc_dim=mfcc_dim,
        fusion_dim=int(model_config["fusion_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        dropout=float(model_config["dropout"]),
    )
    if init_checkpoint is not None:
        model.load_state_dict(init_checkpoint["model_state_dict"])
    trainable_parameters = count_trainable_parameters(model)
    if trainable_parameters > config.max_trainable_parameters:
        raise ValueError(
            f"trainable parameters {trainable_parameters} exceed limit {config.max_trainable_parameters}"
        )
    if config.positive_loss_weight <= 0.0:
        raise ValueError("positive_loss_weight must be positive")
    if config.boundary_loss_weight < 0.0:
        raise ValueError("boundary_loss_weight must be non-negative")
    if config.gap_loss_weight < 0.0:
        raise ValueError("gap_loss_weight must be non-negative")
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    positive_loss_weight = torch.tensor(float(config.positive_loss_weight), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=positive_loss_weight)
    window_order = shuffled_window_order(len(windows), seed=config.seed)
    losses: list[float] = []
    total_correct = 0.0
    total_frames = 0.0
    total_positive = 0.0
    batch_size = max(1, int(config.batch_size))
    for step in range(config.max_steps):
        batch_indexes = [
            window_order[(step * batch_size + offset) % len(window_order)]
            for offset in range(batch_size)
        ]
        batch_windows = [windows[index] for index in batch_indexes]
        max_frames = max(item[2].shape[0] for item in batch_windows)
        whisper_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_2d(item[0], max_frames) for item in batch_windows])
        ).to(device)
        mfcc_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_2d(item[1], max_frames) for item in batch_windows])
        ).to(device)
        label_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[2], max_frames) for item in batch_windows])
        ).to(device)
        weight_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[3], max_frames) for item in batch_windows])
        ).to(device)
        mask_tensor = torch.from_numpy(
            np.stack([_frame_mask(item[2].shape[0], max_frames) for item in batch_windows])
        ).to(device) * weight_tensor
        logits = model(whisper_tensor, mfcc_tensor)
        frame_loss = (criterion(logits, label_tensor) * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)
        probability = torch.sigmoid(logits)
        boundary_loss = boundary_transition_loss(
            probability,
            label_tensor,
            mask_tensor,
        )
        gap_loss = internal_gap_probability_loss(
            probability,
            label_tensor,
            mask_tensor,
        )
        loss = (
            frame_loss
            + float(config.boundary_loss_weight) * boundary_loss
            + float(config.gap_loss_weight) * gap_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += float(((pred == label_tensor).float() * mask_tensor).sum().item())
            total_frames += float(mask_tensor.sum().item())
            total_positive += float((label_tensor * mask_tensor).sum().item())
        if config.log_interval_steps > 0 and (
            (step + 1) == 1
            or (step + 1) % config.log_interval_steps == 0
            or (step + 1) == config.max_steps
        ):
            running_loss = float(np.mean(losses)) if losses else 0.0
            running_accuracy = (total_correct / total_frames) if total_frames else 0.0
            print(
                f"train_step={step + 1}/{config.max_steps} "
                f"loss={running_loss:.4f} frame_accuracy={running_accuracy:.4f}",
                flush=True,
            )

    checkpoint_path = output_dir / "fusionvad_ja_addition_bilstm.pt"
    metrics_path = output_dir / "train_metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "features": len(feature_manifest_rows),
            "windows": len(windows),
            "window_order": window_order,
            "trainable_parameters": trainable_parameters,
            "whisper_dim": whisper_dim,
            "mfcc_dim": mfcc_dim,
            "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path else None,
        },
        checkpoint_path,
    )
    metrics = TrainMetrics(
        steps=config.max_steps,
        loss=float(np.mean(losses)) if losses else 0.0,
        frame_accuracy=(total_correct / total_frames) if total_frames else 0.0,
        positive_ratio=(total_positive / total_frames) if total_frames else 0.0,
        checkpoint=str(checkpoint_path),
        metrics_path=str(metrics_path),
    )
    metrics_payload = asdict(metrics)
    metrics_payload["trainable_parameters"] = trainable_parameters
    metrics_payload["whisper_dim"] = whisper_dim
    metrics_payload["mfcc_dim"] = mfcc_dim
    metrics_payload["init_checkpoint"] = str(init_checkpoint_path) if init_checkpoint_path else None
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def train_endpoint_refiner_classifier(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: list[Mapping[str, Any]],
    output_dir: Path,
    config: EndpointRefinerTrainConfig,
) -> TrainMetrics:
    import torch
    from torch import nn

    if not feature_manifest_rows:
        raise ValueError("at least one cached feature row is required")
    if config.positive_loss_weight <= 0.0:
        raise ValueError("positive_loss_weight must be positive")
    if config.speech_loss_weight < 0.0:
        raise ValueError("speech_loss_weight must be non-negative")
    if config.boundary_loss_weight < 0.0:
        raise ValueError("boundary_loss_weight must be non-negative")
    if config.start_loss_weight < 0.0:
        raise ValueError("start_loss_weight must be non-negative")
    if config.end_loss_weight < 0.0:
        raise ValueError("end_loss_weight must be non-negative")
    if config.internal_gap_loss_weight < 0.0:
        raise ValueError("internal_gap_loss_weight must be non-negative")
    if config.cut_drop_loss_weight < 0.0:
        raise ValueError("cut_drop_loss_weight must be non-negative")
    if config.cut_point_loss_weight < 0.0:
        raise ValueError("cut_point_loss_weight must be non-negative")
    if config.start_positive_loss_weight <= 0.0:
        raise ValueError("start_positive_loss_weight must be positive")
    if config.end_positive_loss_weight <= 0.0:
        raise ValueError("end_positive_loss_weight must be positive")
    if config.cut_drop_positive_loss_weight <= 0.0:
        raise ValueError("cut_drop_positive_loss_weight must be positive")
    if config.cut_point_positive_loss_weight <= 0.0:
        raise ValueError("cut_point_positive_loss_weight must be positive")
    if config.boundary_radius_frames < 0:
        raise ValueError("boundary_radius_frames must be non-negative")
    if config.cut_min_gap_s < 0.0:
        raise ValueError("cut_min_gap_s must be non-negative")
    if config.cut_boundary_radius_frames < 0:
        raise ValueError("cut_boundary_radius_frames must be non-negative")
    if config.save_interval_steps < 0:
        raise ValueError("save_interval_steps must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    windows = build_lazy_endpoint_feature_windows(
        records=records,
        feature_manifest_rows=feature_manifest_rows,
        boundary_radius_frames=config.boundary_radius_frames,
        cut_min_gap_s=config.cut_min_gap_s,
        cut_boundary_radius_frames=config.cut_boundary_radius_frames,
    )
    if not windows:
        raise ValueError("no endpoint feature windows could be built")
    whisper_dim = int(windows[0].whisper_dim)
    mfcc_dim = int(windows[0].mfcc_dim)
    init_checkpoint: Mapping[str, Any] | None = None
    init_checkpoint_path: Path | None = None
    model_config = {
        "fusion_dim": config.fusion_dim,
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "dropout": config.dropout,
    }
    if config.init_checkpoint:
        init_checkpoint_path = Path(config.init_checkpoint)
        init_checkpoint = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
        model_type = str(init_checkpoint.get("model_type") or "")
        if model_type and model_type != "addition_endpoint_bilstm":
            raise ValueError(f"init endpoint checkpoint model_type={model_type!r} is not supported")
        init_whisper_dim = int(init_checkpoint.get("whisper_dim", -1))
        init_mfcc_dim = int(init_checkpoint.get("mfcc_dim", -1))
        if init_whisper_dim != whisper_dim or init_mfcc_dim != mfcc_dim:
            raise ValueError(
                "init endpoint checkpoint feature dimensions do not match current cache: "
                f"checkpoint=({init_whisper_dim}, {init_mfcc_dim}) current=({whisper_dim}, {mfcc_dim})"
            )
        checkpoint_config = dict(init_checkpoint.get("config") or {})
        for key in ("fusion_dim", "hidden_dim", "layers"):
            checkpoint_value = int(checkpoint_config.get(key, model_config[key]))
            if checkpoint_value != int(model_config[key]):
                raise ValueError(
                    f"init endpoint checkpoint {key}={checkpoint_value} does not match "
                    f"training config {model_config[key]}"
                )
        if "dropout" in checkpoint_config:
            model_config["dropout"] = float(checkpoint_config["dropout"])

    model = AdditionFusionEndpointBiLSTM(
        whisper_dim=whisper_dim,
        mfcc_dim=mfcc_dim,
        fusion_dim=int(model_config["fusion_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        dropout=float(model_config["dropout"]),
    )
    if init_checkpoint is not None:
        model.load_state_dict(init_checkpoint["model_state_dict"])
    trainable_parameters = count_trainable_parameters(model)
    if trainable_parameters > config.max_trainable_parameters:
        raise ValueError(
            f"trainable parameters {trainable_parameters} exceed limit {config.max_trainable_parameters}"
        )

    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    speech_pos_weight = torch.tensor(float(config.positive_loss_weight), dtype=torch.float32, device=device)
    start_pos_weight = torch.tensor(float(config.start_positive_loss_weight), dtype=torch.float32, device=device)
    end_pos_weight = torch.tensor(float(config.end_positive_loss_weight), dtype=torch.float32, device=device)
    cut_drop_pos_weight = torch.tensor(
        float(config.cut_drop_positive_loss_weight), dtype=torch.float32, device=device
    )
    cut_point_pos_weight = torch.tensor(
        float(config.cut_point_positive_loss_weight), dtype=torch.float32, device=device
    )
    speech_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=speech_pos_weight)
    start_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=start_pos_weight)
    end_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=end_pos_weight)
    cut_drop_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=cut_drop_pos_weight)
    cut_point_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=cut_point_pos_weight)
    window_order = shuffled_window_order(len(windows), seed=config.seed)
    losses: list[float] = []
    component_losses: dict[str, list[float]] = {
        "speech": [],
        "start": [],
        "end": [],
        "internal_gap": [],
        "cut_drop": [],
        "cut_point": [],
    }
    total_correct = 0.0
    total_frames = 0.0
    total_positive = 0.0
    batch_size = max(1, int(config.batch_size))

    for step in range(config.max_steps):
        batch_indexes = [
            window_order[(step * batch_size + offset) % len(window_order)]
            for offset in range(batch_size)
        ]
        batch_windows = [load_endpoint_feature_window(windows[index]) for index in batch_indexes]
        max_frames = max(item[2].shape[0] for item in batch_windows)
        whisper_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_2d(item[0], max_frames) for item in batch_windows])
        ).to(device)
        mfcc_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_2d(item[1], max_frames) for item in batch_windows])
        ).to(device)
        speech_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[2], max_frames) for item in batch_windows])
        ).to(device)
        weight_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[3], max_frames) for item in batch_windows])
        ).to(device)
        start_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[4], max_frames) for item in batch_windows])
        ).to(device)
        end_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[5], max_frames) for item in batch_windows])
        ).to(device)
        cut_drop_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[6], max_frames) for item in batch_windows])
        ).to(device)
        cut_point_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[7], max_frames) for item in batch_windows])
        ).to(device)
        mask_tensor = torch.from_numpy(
            np.stack([_frame_mask(item[2].shape[0], max_frames) for item in batch_windows])
        ).to(device) * weight_tensor
        logits = model(whisper_tensor, mfcc_tensor)
        speech_loss = (
            speech_criterion(logits["speech"], speech_tensor) * mask_tensor
        ).sum() / mask_tensor.sum().clamp_min(1.0)
        start_loss = (
            start_criterion(logits["start"], start_tensor) * mask_tensor
        ).sum() / mask_tensor.sum().clamp_min(1.0)
        end_loss = (
            end_criterion(logits["end"], end_tensor) * mask_tensor
        ).sum() / mask_tensor.sum().clamp_min(1.0)
        cut_drop_loss = (
            cut_drop_criterion(logits["cut_drop"], cut_drop_tensor) * mask_tensor
        ).sum() / mask_tensor.sum().clamp_min(1.0)
        cut_point_loss = (
            cut_point_criterion(logits["cut_point"], cut_point_tensor) * mask_tensor
        ).sum() / mask_tensor.sum().clamp_min(1.0)
        if float(config.internal_gap_loss_weight) > 0.0:
            gap_loss = internal_gap_probability_loss(
                torch.sigmoid(logits["speech"]),
                speech_tensor,
                mask_tensor,
            )
        else:
            gap_loss = speech_loss.new_tensor(0.0)
        loss = (
            float(config.speech_loss_weight) * speech_loss
            + float(config.boundary_loss_weight) * (start_loss + end_loss)
            + float(config.start_loss_weight) * start_loss
            + float(config.end_loss_weight) * end_loss
            + float(config.internal_gap_loss_weight) * gap_loss
            + float(config.cut_drop_loss_weight) * cut_drop_loss
            + float(config.cut_point_loss_weight) * cut_point_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        component_losses["speech"].append(float(speech_loss.detach().cpu()))
        component_losses["start"].append(float(start_loss.detach().cpu()))
        component_losses["end"].append(float(end_loss.detach().cpu()))
        component_losses["internal_gap"].append(float(gap_loss.detach().cpu()))
        component_losses["cut_drop"].append(float(cut_drop_loss.detach().cpu()))
        component_losses["cut_point"].append(float(cut_point_loss.detach().cpu()))
        with torch.no_grad():
            pred = (torch.sigmoid(logits["speech"]) >= 0.5).float()
            total_correct += float(((pred == speech_tensor).float() * mask_tensor).sum().item())
            total_frames += float(mask_tensor.sum().item())
            total_positive += float((speech_tensor * mask_tensor).sum().item())
        if config.log_interval_steps > 0 and (
            (step + 1) == 1
            or (step + 1) % config.log_interval_steps == 0
            or (step + 1) == config.max_steps
        ):
            running_loss = float(np.mean(losses)) if losses else 0.0
            running_accuracy = (total_correct / total_frames) if total_frames else 0.0
            print(
                f"train_step={step + 1}/{config.max_steps} "
                f"loss={running_loss:.4f} frame_accuracy={running_accuracy:.4f}",
                flush=True,
            )
        if config.save_interval_steps > 0 and (step + 1) % config.save_interval_steps == 0:
            _save_endpoint_refiner_checkpoint(
                output_dir / f"checkpoint-step-{step + 1}.pt",
                model=model,
                config=config,
                feature_count=len(feature_manifest_rows),
                window_count=len(windows),
                window_order=window_order,
                trainable_parameters=trainable_parameters,
                whisper_dim=whisper_dim,
                mfcc_dim=mfcc_dim,
                init_checkpoint_path=init_checkpoint_path,
            )

    checkpoint_path = output_dir / "fusionvad_ja_endpoint_refiner.pt"
    metrics_path = output_dir / "train_metrics.json"
    _save_endpoint_refiner_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        feature_count=len(feature_manifest_rows),
        window_count=len(windows),
        window_order=window_order,
        trainable_parameters=trainable_parameters,
        whisper_dim=whisper_dim,
        mfcc_dim=mfcc_dim,
        init_checkpoint_path=init_checkpoint_path,
    )
    metrics = TrainMetrics(
        steps=config.max_steps,
        loss=float(np.mean(losses)) if losses else 0.0,
        frame_accuracy=(total_correct / total_frames) if total_frames else 0.0,
        positive_ratio=(total_positive / total_frames) if total_frames else 0.0,
        checkpoint=str(checkpoint_path),
        metrics_path=str(metrics_path),
    )
    metrics_payload = asdict(metrics)
    metrics_payload["trainable_parameters"] = trainable_parameters
    metrics_payload["whisper_dim"] = whisper_dim
    metrics_payload["mfcc_dim"] = mfcc_dim
    metrics_payload["model_type"] = "addition_endpoint_bilstm"
    metrics_payload["init_checkpoint"] = str(init_checkpoint_path) if init_checkpoint_path else None
    metrics_payload["mean_component_losses"] = {
        name: float(np.mean(values)) if values else 0.0 for name, values in component_losses.items()
    }
    metrics_payload["boundary_first"] = {
        "speech_loss_weight": float(config.speech_loss_weight),
        "start_loss_weight": float(config.start_loss_weight),
        "end_loss_weight": float(config.end_loss_weight),
        "internal_gap_loss_weight": float(config.internal_gap_loss_weight),
        "cut_drop_loss_weight": float(config.cut_drop_loss_weight),
        "cut_point_loss_weight": float(config.cut_point_loss_weight),
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _save_endpoint_refiner_checkpoint(
    checkpoint_path: Path,
    *,
    model,
    config: EndpointRefinerTrainConfig,
    feature_count: int,
    window_count: int,
    window_order: list[int],
    trainable_parameters: int,
    whisper_dim: int,
    mfcc_dim: int,
    init_checkpoint_path: Path | None = None,
) -> None:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "addition_endpoint_bilstm",
            "config": asdict(config),
            "features": feature_count,
            "windows": window_count,
            "window_order": window_order,
            "trainable_parameters": trainable_parameters,
            "whisper_dim": whisper_dim,
            "mfcc_dim": mfcc_dim,
            "outputs": ["speech", "start", "end", "cut_drop", "cut_point"],
            "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path else None,
        },
        checkpoint_path,
    )


def train_imitation_classifier(
    *,
    feature_manifest_rows: list[Mapping[str, Any]],
    imitation_rows: list[Mapping[str, Any]],
    output_dir: Path,
    config: ImitationTrainConfig,
) -> TrainMetrics:
    import torch
    from torch import nn

    if not feature_manifest_rows:
        raise ValueError("at least one cached feature row is required")
    if not imitation_rows:
        raise ValueError("at least one imitation target row is required")
    if config.split_loss_weight < 0.0:
        raise ValueError("split_loss_weight must be non-negative")
    if config.drop_gap_loss_weight < 0.0:
        raise ValueError("drop_gap_loss_weight must be non-negative")
    if config.split_positive_loss_weight <= 0.0:
        raise ValueError("split_positive_loss_weight must be positive")
    if config.drop_gap_positive_loss_weight <= 0.0:
        raise ValueError("drop_gap_positive_loss_weight must be positive")
    if config.save_interval_steps < 0:
        raise ValueError("save_interval_steps must be non-negative")
    if config.window_frames < 0:
        raise ValueError("window_frames must be non-negative")
    if not 0.0 <= config.positive_window_ratio <= 1.0:
        raise ValueError("positive_window_ratio must be in [0, 1]")
    if config.positive_jitter_frames < 0:
        raise ValueError("positive_jitter_frames must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    windows = build_lazy_imitation_feature_windows(
        feature_manifest_rows=feature_manifest_rows,
        imitation_rows=imitation_rows,
    )
    if not windows:
        raise ValueError("no imitation feature windows could be built")
    whisper_dim = int(windows[0].whisper_dim)
    mfcc_dim = int(windows[0].mfcc_dim)
    model = AdditionFusionImitationBiLSTM(
        whisper_dim=whisper_dim,
        mfcc_dim=mfcc_dim,
        fusion_dim=config.fusion_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
    )
    trainable_parameters = count_trainable_parameters(model)
    if trainable_parameters > config.max_trainable_parameters:
        raise ValueError(
            f"trainable parameters {trainable_parameters} exceed limit {config.max_trainable_parameters}"
        )

    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    split_pos_weight = torch.tensor(float(config.split_positive_loss_weight), dtype=torch.float32, device=device)
    drop_gap_pos_weight = torch.tensor(float(config.drop_gap_positive_loss_weight), dtype=torch.float32, device=device)
    split_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=split_pos_weight)
    drop_gap_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=drop_gap_pos_weight)
    window_order = shuffled_window_order(len(windows), seed=config.seed)
    split_positive_window_indexes = (
        [index for index, window in enumerate(windows) if np.count_nonzero(window.split_frames > 0.5) > 0]
        if config.split_loss_weight > 0.0
        else []
    )
    drop_gap_positive_window_indexes = (
        [index for index, window in enumerate(windows) if np.count_nonzero(window.drop_gap_frames > 0.5) > 0]
        if config.drop_gap_loss_weight > 0.0
        else []
    )
    batch_size = max(1, int(config.batch_size))
    losses: list[float] = []
    component_losses: dict[str, list[float]] = {"split": [], "drop_gap": []}
    aggregate_counts: dict[str, dict[str, int]] = {
        "split": empty_frame_counts(),
        "drop_gap": empty_frame_counts(),
    }

    for step in range(config.max_steps):
        batch_windows = [
            load_imitation_feature_window(
                windows[index],
                start_frame=start_frame,
                frame_count=sample_frame_count,
            )
            for index, start_frame, sample_frame_count in (
                sample_imitation_window(
                    windows=windows,
                    window_order=window_order,
                    step=step,
                    offset=offset,
                    batch_size=batch_size,
                    rng=rng,
                    window_frames=config.window_frames,
                    positive_window_ratio=config.positive_window_ratio,
                    positive_jitter_frames=config.positive_jitter_frames,
                    split_positive_window_indexes=split_positive_window_indexes,
                    drop_gap_positive_window_indexes=drop_gap_positive_window_indexes,
                )
                for offset in range(batch_size)
            )
        ]
        max_frames = max(item[2].shape[0] for item in batch_windows)
        whisper_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_2d(item[0], max_frames) for item in batch_windows])
        ).to(device)
        mfcc_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_2d(item[1], max_frames) for item in batch_windows])
        ).to(device)
        split_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[2], max_frames) for item in batch_windows])
        ).to(device)
        drop_gap_tensor = torch.from_numpy(
            np.stack([_pad_or_trim_1d(item[3], max_frames) for item in batch_windows])
        ).to(device)
        mask_tensor = torch.from_numpy(
            np.stack([_frame_mask(item[2].shape[0], max_frames) for item in batch_windows])
        ).to(device)
        logits = model(whisper_tensor, mfcc_tensor)
        if config.balanced_frame_loss:
            split_loss = balanced_binary_frame_loss(
                logits["split"],
                split_tensor,
                mask_tensor,
                positive_loss_weight=config.split_positive_loss_weight,
            )
            drop_gap_loss = balanced_binary_frame_loss(
                logits["drop_gap"],
                drop_gap_tensor,
                mask_tensor,
                positive_loss_weight=config.drop_gap_positive_loss_weight,
            )
        else:
            split_loss = (
                split_criterion(logits["split"], split_tensor) * mask_tensor
            ).sum() / mask_tensor.sum().clamp_min(1.0)
            drop_gap_loss = (
                drop_gap_criterion(logits["drop_gap"], drop_gap_tensor) * mask_tensor
            ).sum() / mask_tensor.sum().clamp_min(1.0)
        loss = (
            float(config.split_loss_weight) * split_loss
            + float(config.drop_gap_loss_weight) * drop_gap_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        component_losses["split"].append(float(split_loss.detach().cpu()))
        component_losses["drop_gap"].append(float(drop_gap_loss.detach().cpu()))
        with torch.no_grad():
            for name, target_tensor in (("split", split_tensor), ("drop_gap", drop_gap_tensor)):
                pred = (torch.sigmoid(logits[name]) >= 0.5).float()
                counts = frame_classification_counts(
                    labels=target_tensor.detach().cpu().numpy().reshape(-1),
                    predictions=pred.detach().cpu().numpy().reshape(-1),
                    weights=mask_tensor.detach().cpu().numpy().reshape(-1),
                )
                merge_frame_counts(aggregate_counts[name], counts)
        if config.log_interval_steps > 0 and (
            (step + 1) == 1
            or (step + 1) % config.log_interval_steps == 0
            or (step + 1) == config.max_steps
        ):
            running_loss = float(np.mean(losses)) if losses else 0.0
            split_metrics = metrics_from_frame_counts(
                counts=aggregate_counts["split"],
                windows=len(windows),
            )
            drop_metrics = metrics_from_frame_counts(
                counts=aggregate_counts["drop_gap"],
                windows=len(windows),
            )
            print(
                f"train_step={step + 1}/{config.max_steps} loss={running_loss:.4f} "
                f"split_f1={split_metrics.f1:.4f} drop_gap_f1={drop_metrics.f1:.4f}",
                flush=True,
            )
        if config.save_interval_steps > 0 and (step + 1) % config.save_interval_steps == 0:
            _save_imitation_checkpoint(
                output_dir / f"checkpoint-step-{step + 1}.pt",
                model=model,
                config=config,
                feature_count=len(feature_manifest_rows),
                window_count=len(windows),
                window_order=window_order,
                trainable_parameters=trainable_parameters,
                whisper_dim=whisper_dim,
                mfcc_dim=mfcc_dim,
            )

    checkpoint_path = output_dir / "fusionvad_ja_imitation_head.pt"
    metrics_path = output_dir / "train_metrics.json"
    _save_imitation_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        feature_count=len(feature_manifest_rows),
        window_count=len(windows),
        window_order=window_order,
        trainable_parameters=trainable_parameters,
        whisper_dim=whisper_dim,
        mfcc_dim=mfcc_dim,
    )
    split_metrics = metrics_from_frame_counts(counts=aggregate_counts["split"], windows=len(windows))
    drop_gap_metrics = metrics_from_frame_counts(counts=aggregate_counts["drop_gap"], windows=len(windows))
    total_counts = {
        key: int(aggregate_counts["split"].get(key, 0)) + int(aggregate_counts["drop_gap"].get(key, 0))
        for key in empty_frame_counts()
    }
    total_metrics = metrics_from_frame_counts(counts=total_counts, windows=len(windows))
    metrics = TrainMetrics(
        steps=config.max_steps,
        loss=float(np.mean(losses)) if losses else 0.0,
        frame_accuracy=total_metrics.frame_accuracy,
        positive_ratio=total_metrics.positive_ratio,
        checkpoint=str(checkpoint_path),
        metrics_path=str(metrics_path),
    )
    metrics_payload = asdict(metrics)
    metrics_payload["trainable_parameters"] = trainable_parameters
    metrics_payload["whisper_dim"] = whisper_dim
    metrics_payload["mfcc_dim"] = mfcc_dim
    metrics_payload["model_type"] = "addition_imitation_bilstm"
    metrics_payload["sampling"] = {
        "window_frames": int(config.window_frames),
        "positive_window_ratio": float(config.positive_window_ratio),
        "positive_jitter_frames": int(config.positive_jitter_frames),
        "balanced_frame_loss": bool(config.balanced_frame_loss),
        "split_positive_windows": len(split_positive_window_indexes),
        "drop_gap_positive_windows": len(drop_gap_positive_window_indexes),
    }
    metrics_payload["mean_component_losses"] = {
        name: float(np.mean(values)) if values else 0.0 for name, values in component_losses.items()
    }
    metrics_payload["output_metrics"] = {
        "split": asdict(split_metrics),
        "drop_gap": asdict(drop_gap_metrics),
        "combined": asdict(total_metrics),
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _save_imitation_checkpoint(
    checkpoint_path: Path,
    *,
    model,
    config: ImitationTrainConfig,
    feature_count: int,
    window_count: int,
    window_order: list[int],
    trainable_parameters: int,
    whisper_dim: int,
    mfcc_dim: int,
) -> None:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "addition_imitation_bilstm",
            "config": asdict(config),
            "features": feature_count,
            "windows": window_count,
            "window_order": window_order,
            "trainable_parameters": trainable_parameters,
            "whisper_dim": whisper_dim,
            "mfcc_dim": mfcc_dim,
            "outputs": ["split", "drop_gap"],
        },
        checkpoint_path,
    )


def balanced_binary_frame_loss(logits, targets, mask, *, positive_loss_weight: float):
    import torch.nn.functional as F

    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * mask
    positive_mask = (targets > 0.5).float() * mask
    negative_mask = (targets <= 0.5).float() * mask
    positive_count = positive_mask.sum()
    negative_count = negative_mask.sum()
    positive_loss = (raw * positive_mask).sum() / positive_count.clamp_min(1.0)
    negative_loss = (raw * negative_mask).sum() / negative_count.clamp_min(1.0)
    if positive_count.item() <= 0:
        return negative_loss
    if negative_count.item() <= 0:
        return float(positive_loss_weight) * positive_loss
    return float(positive_loss_weight) * positive_loss + negative_loss


def boundary_transition_loss(probability, labels, mask):
    """Penalize delayed probability transitions at speech/non-speech boundaries."""
    import torch

    if probability.shape[-1] < 2:
        return probability.new_tensor(0.0)
    active = (mask[..., 1:] > 0.0) & (mask[..., :-1] > 0.0)
    transitions = (labels[..., 1:] - labels[..., :-1]).abs() * active.float()
    if torch.count_nonzero(transitions).item() <= 0:
        return probability.new_tensor(0.0)
    probability_delta = probability[..., 1:] - probability[..., :-1]
    return ((probability_delta - (labels[..., 1:] - labels[..., :-1])).abs() * transitions).sum() / transitions.sum().clamp_min(1.0)


def internal_gap_probability_loss(probability, labels, mask):
    """Suppress high speech probability inside non-speech gaps between speech islands."""
    import torch

    gap_mask = labels.new_zeros(labels.shape)
    for batch_index in range(labels.shape[0]):
        active_indices = torch.nonzero(mask[batch_index] > 0.0, as_tuple=False).reshape(-1)
        if active_indices.numel() <= 0:
            continue
        active_labels = labels[batch_index, active_indices]
        speech_indices = torch.nonzero(active_labels > 0.5, as_tuple=False).reshape(-1)
        if speech_indices.numel() < 2:
            continue
        start = int(active_indices[int(speech_indices[0])].item())
        end = int(active_indices[int(speech_indices[-1])].item()) + 1
        between = (labels[batch_index, start:end] <= 0.5) & (mask[batch_index, start:end] > 0.0)
        gap_mask[batch_index, start:end] = between.float()
    if torch.count_nonzero(gap_mask).item() <= 0:
        return probability.new_tensor(0.0)
    return (probability * gap_mask).sum() / gap_mask.sum().clamp_min(1.0)


def build_feature_windows(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for row in feature_manifest_rows:
        label_index = int(row["label_index"])
        record = records[label_index]
        whisper, mfcc = load_cached_feature(Path(str(row["feature_path"])))
        weights = effective_frame_weights(record)
        frame_count = min(whisper.shape[0], mfcc.shape[0], len(record.speech_frames), len(weights))
        if frame_count <= 0:
            continue
        windows.append(
            (
                np.ascontiguousarray(whisper[:frame_count], dtype=np.float32),
                np.ascontiguousarray(mfcc[:frame_count], dtype=np.float32),
                np.asarray(record.speech_frames[:frame_count], dtype=np.float32),
                np.asarray(weights[:frame_count], dtype=np.float32),
            )
        )
    return windows


def build_endpoint_feature_windows(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
    boundary_radius_frames: int = 1,
    cut_min_gap_s: float = 0.5,
    cut_boundary_radius_frames: int = 0,
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
]:
    windows: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    for row in feature_manifest_rows:
        label_index = int(row["label_index"])
        record = records[label_index]
        whisper, mfcc = load_cached_feature(Path(str(row["feature_path"])))
        weights = effective_frame_weights(record)
        frame_count = min(whisper.shape[0], mfcc.shape[0], len(record.speech_frames), len(weights))
        if frame_count <= 0:
            continue
        speech = np.asarray(record.speech_frames[:frame_count], dtype=np.float32)
        start_targets, end_targets, cut_drop_targets, cut_point_targets = endpoint_targets_from_record(
            record,
            frame_count=frame_count,
            boundary_radius_frames=boundary_radius_frames,
            cut_min_gap_s=cut_min_gap_s,
            cut_boundary_radius_frames=cut_boundary_radius_frames,
        )
        windows.append(
            (
                np.ascontiguousarray(whisper[:frame_count], dtype=np.float32),
                np.ascontiguousarray(mfcc[:frame_count], dtype=np.float32),
                speech,
                np.asarray(weights[:frame_count], dtype=np.float32),
                start_targets,
                end_targets,
                cut_drop_targets,
                cut_point_targets,
            )
        )
    return windows


@dataclass(frozen=True)
class LazyEndpointFeatureWindow:
    feature_path: str
    label_index: int
    frame_count: int
    whisper_dim: int
    mfcc_dim: int
    speech: np.ndarray
    weights: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    cut_drops: np.ndarray
    cut_points: np.ndarray


def build_lazy_endpoint_feature_windows(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
    boundary_radius_frames: int = 1,
    cut_min_gap_s: float = 0.5,
    cut_boundary_radius_frames: int = 0,
) -> list[LazyEndpointFeatureWindow]:
    windows: list[LazyEndpointFeatureWindow] = []
    for row in feature_manifest_rows:
        label_index = int(row["label_index"])
        record = records[label_index]
        feature_path = str(row["feature_path"])
        whisper_dim = int(row.get("whisper_dim") or 0)
        mfcc_dim = int(row.get("mfcc_dim") or 0)
        manifest_frame_count = int(row.get("frame_count") or 0)
        if whisper_dim <= 0 or mfcc_dim <= 0 or manifest_frame_count <= 0:
            whisper, mfcc = load_cached_feature(Path(feature_path))
            whisper_dim = int(whisper.shape[-1])
            mfcc_dim = int(mfcc.shape[-1])
            manifest_frame_count = int(min(whisper.shape[0], mfcc.shape[0]))
        weights = effective_frame_weights(record)
        frame_count = min(manifest_frame_count, len(record.speech_frames), len(weights))
        if frame_count <= 0:
            continue
        speech = np.asarray(record.speech_frames[:frame_count], dtype=np.float32)
        start_targets, end_targets, cut_drop_targets, cut_point_targets = endpoint_targets_from_record(
            record,
            frame_count=frame_count,
            boundary_radius_frames=boundary_radius_frames,
            cut_min_gap_s=cut_min_gap_s,
            cut_boundary_radius_frames=cut_boundary_radius_frames,
        )
        windows.append(
            LazyEndpointFeatureWindow(
                feature_path=feature_path,
                label_index=label_index,
                frame_count=frame_count,
                whisper_dim=whisper_dim,
                mfcc_dim=mfcc_dim,
                speech=speech,
                weights=np.asarray(weights[:frame_count], dtype=np.float32),
                starts=start_targets,
                ends=end_targets,
                cut_drops=cut_drop_targets,
                cut_points=cut_point_targets,
            )
        )
    return windows


def load_endpoint_feature_window(
    window: LazyEndpointFeatureWindow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    whisper, mfcc = load_cached_feature(Path(window.feature_path))
    frame_count = min(window.frame_count, whisper.shape[0], mfcc.shape[0])
    if frame_count <= 0:
        raise ValueError(f"empty endpoint feature window: {window.feature_path}")
    return (
        np.ascontiguousarray(whisper[:frame_count], dtype=np.float32),
        np.ascontiguousarray(mfcc[:frame_count], dtype=np.float32),
        np.asarray(window.speech[:frame_count], dtype=np.float32),
        np.asarray(window.weights[:frame_count], dtype=np.float32),
        np.asarray(window.starts[:frame_count], dtype=np.float32),
        np.asarray(window.ends[:frame_count], dtype=np.float32),
        np.asarray(window.cut_drops[:frame_count], dtype=np.float32),
        np.asarray(window.cut_points[:frame_count], dtype=np.float32),
    )


@dataclass(frozen=True)
class LazyImitationFeatureWindow:
    feature_path: str
    audio_id: str
    frame_count: int
    whisper_dim: int
    mfcc_dim: int
    split_frames: np.ndarray
    drop_gap_frames: np.ndarray


def build_lazy_imitation_feature_windows(
    *,
    feature_manifest_rows: Iterable[Mapping[str, Any]],
    imitation_rows: Iterable[Mapping[str, Any]],
) -> list[LazyImitationFeatureWindow]:
    targets_by_audio = {str(row.get("audio_id") or ""): row for row in imitation_rows}
    windows: list[LazyImitationFeatureWindow] = []
    for row in feature_manifest_rows:
        audio_id = str(row.get("audio_id") or "")
        target = targets_by_audio.get(audio_id)
        if not target:
            continue
        feature_path = str(row["feature_path"])
        whisper_dim = int(row.get("whisper_dim") or 0)
        mfcc_dim = int(row.get("mfcc_dim") or 0)
        manifest_frame_count = int(row.get("frame_count") or 0)
        if whisper_dim <= 0 or mfcc_dim <= 0 or manifest_frame_count <= 0:
            whisper, mfcc = load_cached_feature(Path(feature_path))
            whisper_dim = int(whisper.shape[-1])
            mfcc_dim = int(mfcc.shape[-1])
            manifest_frame_count = int(min(whisper.shape[0], mfcc.shape[0]))
        split_frames = np.asarray(target.get("split_frames") or [], dtype=np.float32)
        drop_gap_frames = np.asarray(target.get("drop_gap_frames") or [], dtype=np.float32)
        frame_count = manifest_frame_count
        if frame_count <= 0:
            continue
        split_frames = resize_binary_frames(split_frames, frame_count)
        drop_gap_frames = resize_binary_frames(drop_gap_frames, frame_count)
        windows.append(
            LazyImitationFeatureWindow(
                feature_path=feature_path,
                audio_id=audio_id,
                frame_count=frame_count,
                whisper_dim=whisper_dim,
                mfcc_dim=mfcc_dim,
                split_frames=split_frames[:frame_count],
                drop_gap_frames=drop_gap_frames[:frame_count],
            )
        )
    return windows


def resize_binary_frames(values: np.ndarray, target_frames: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    target_frames = int(target_frames)
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if source.size <= 0:
        return np.zeros(target_frames, dtype=np.float32)
    if source.size == target_frames:
        return np.ascontiguousarray((source > 0.5).astype(np.float32))
    positions = (np.arange(target_frames, dtype=np.float32) + 0.5) * (float(source.size) / float(target_frames)) - 0.5
    indexes = np.clip(np.rint(positions).astype(np.int64), 0, source.size - 1)
    return np.ascontiguousarray((source[indexes] > 0.5).astype(np.float32))


def sample_imitation_window(
    *,
    windows: Sequence[LazyImitationFeatureWindow],
    window_order: list[int],
    step: int,
    offset: int,
    batch_size: int,
    rng: np.random.Generator,
    window_frames: int,
    positive_window_ratio: float,
    positive_jitter_frames: int,
    split_positive_window_indexes: Sequence[int],
    drop_gap_positive_window_indexes: Sequence[int],
) -> tuple[int, int, int | None]:
    if not windows:
        raise ValueError("at least one imitation window is required")
    use_positive = (
        window_frames > 0
        and bool(split_positive_window_indexes or drop_gap_positive_window_indexes)
        and float(rng.random()) < positive_window_ratio
    )
    if use_positive:
        prefer_split = bool(rng.random() < 0.5)
        if prefer_split and split_positive_window_indexes:
            index = int(rng.choice(split_positive_window_indexes))
            target_frames = windows[index].split_frames
        elif drop_gap_positive_window_indexes:
            index = int(rng.choice(drop_gap_positive_window_indexes))
            target_frames = windows[index].drop_gap_frames
        elif split_positive_window_indexes:
            index = int(rng.choice(split_positive_window_indexes))
            target_frames = windows[index].split_frames
        else:
            index = int(window_order[(step * batch_size + offset) % len(window_order)])
            target_frames = np.zeros(windows[index].frame_count, dtype=np.float32)
        positive_indices = np.flatnonzero(target_frames[: windows[index].frame_count] > 0.5)
        if positive_indices.size > 0:
            center = int(rng.choice(positive_indices))
            if positive_jitter_frames > 0:
                jitter = int(rng.integers(-positive_jitter_frames, positive_jitter_frames + 1))
                center += jitter
            return index, crop_start_for_center(
                center_frame=center,
                total_frames=windows[index].frame_count,
                window_frames=window_frames,
            ), window_frames

    index = int(window_order[(step * batch_size + offset) % len(window_order)])
    if window_frames <= 0 or windows[index].frame_count <= window_frames:
        return index, 0, None
    max_start = max(0, windows[index].frame_count - window_frames)
    return index, int(rng.integers(0, max_start + 1)), window_frames


def crop_start_for_center(*, center_frame: int, total_frames: int, window_frames: int) -> int:
    if window_frames <= 0 or total_frames <= window_frames:
        return 0
    start = int(center_frame) - int(window_frames) // 2
    return max(0, min(start, int(total_frames) - int(window_frames)))


def load_imitation_feature_window(
    window: LazyImitationFeatureWindow,
    *,
    start_frame: int = 0,
    frame_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    whisper, mfcc = load_cached_feature(Path(window.feature_path))
    available_frames = min(window.frame_count, whisper.shape[0], mfcc.shape[0])
    if available_frames <= 0:
        raise ValueError(f"empty imitation feature window: {window.feature_path}")
    start = max(0, min(int(start_frame), available_frames - 1))
    if frame_count is None or int(frame_count) <= 0:
        end = available_frames
    else:
        end = min(available_frames, start + int(frame_count))
    if end <= start:
        raise ValueError(f"empty imitation feature crop: {window.feature_path}")
    return (
        np.ascontiguousarray(whisper[start:end], dtype=np.float32),
        np.ascontiguousarray(mfcc[start:end], dtype=np.float32),
        np.asarray(window.split_frames[start:end], dtype=np.float32),
        np.asarray(window.drop_gap_frames[start:end], dtype=np.float32),
    )


def endpoint_targets_from_record(
    record: LabelRecord,
    *,
    frame_count: int,
    boundary_radius_frames: int = 1,
    cut_min_gap_s: float = 0.5,
    cut_boundary_radius_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = max(0, int(frame_count))
    starts = np.zeros(frame_count, dtype=np.float32)
    ends = np.zeros(frame_count, dtype=np.float32)
    cut_drops = np.zeros(frame_count, dtype=np.float32)
    cut_points = np.zeros(frame_count, dtype=np.float32)
    if frame_count <= 0:
        return starts, ends, cut_drops, cut_points
    segments = supervised_segments_for_record(record)
    radius = max(0, int(boundary_radius_frames))
    for segment in segments:
        start_index = max(0, min(frame_count - 1, int(round(segment.start / record.frame_hop_s))))
        end_index = max(0, min(frame_count - 1, int(round(segment.end / record.frame_hop_s)) - 1))
        starts[max(0, start_index - radius) : min(frame_count, start_index + radius + 1)] = 1.0
        ends[max(0, end_index - radius) : min(frame_count, end_index + radius + 1)] = 1.0
    min_gap = max(0.0, float(cut_min_gap_s))
    cut_radius = max(0, int(cut_boundary_radius_frames))
    metadata = dict(record.boundary_metadata or {})
    for zone_start, zone_end in metadata_time_spans(metadata.get("cut_drop_zones")):
        start_index, end_index = time_span_to_frame_range(
            zone_start,
            zone_end,
            frame_count=frame_count,
            frame_hop_s=record.frame_hop_s,
        )
        if end_index > start_index:
            cut_drops[start_index:end_index] = 1.0
    for point_time in metadata_cut_points(metadata.get("cut_point_segments")):
        point_index = max(0, min(frame_count - 1, int(round(point_time / record.frame_hop_s))))
        cut_points[max(0, point_index - cut_radius) : min(frame_count, point_index + cut_radius + 1)] = 1.0
    for previous, current in zip(segments, segments[1:]):
        gap_start = previous.end
        gap_end = current.start
        if gap_end - gap_start < min_gap:
            if cut_radius <= 0:
                continue
            for boundary in (gap_start, gap_end):
                boundary_index = max(
                    0,
                    min(frame_count - 1, int(round(boundary / record.frame_hop_s))),
                )
                cut_points[
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
            cut_drops[start_index:end_index] = 1.0
    return starts, ends, cut_drops, cut_points


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


def metadata_time_spans(items: Any) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for item in list(items or []):
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


def metadata_cut_points(items: Any) -> list[float]:
    points: list[float] = []
    for item in list(items or []):
        if not isinstance(item, Mapping):
            continue
        try:
            if item.get("time_s") is not None:
                points.append(float(item["time_s"]))
                continue
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
        except (TypeError, ValueError):
            continue
        points.append((start + end) / 2.0)
    return points


def supervised_segments_for_record(record: LabelRecord) -> list[TeacherSegment]:
    segments = list(record.teacher_segments.get("supervised") or [])
    if not segments:
        segments = segments_from_speech_frames(record)
    return sorted(segments, key=lambda item: (item.start, item.end))


def segments_from_speech_frames(record: LabelRecord) -> list[TeacherSegment]:
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


class TinyFrameClassifier:
    def __new__(cls):
        import torch
        from torch import nn

        class _TinyFrameClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=31, stride=4, padding=15),
                    nn.ReLU(),
                    nn.Conv1d(8, 16, kernel_size=15, stride=4, padding=7),
                    nn.ReLU(),
                    nn.Conv1d(16, 1, kernel_size=5, padding=2),
                )

            def forward(self, audio, target_frames: int):
                import torch.nn.functional as F

                logits = self.net(audio).squeeze(1)
                logits = F.interpolate(
                    logits.unsqueeze(1),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
                return logits

        return _TinyFrameClassifier()


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


def _frame_mask(actual_frames: int, frame_count: int) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=np.float32)
    mask[: min(max(0, actual_frames), frame_count)] = 1.0
    return mask


def _pad_or_trim_labels(labels: list[int], length: int) -> np.ndarray:
    values = np.asarray(labels[:length], dtype=np.float32)
    if values.shape[0] >= length:
        return values
    padded = np.zeros(length, dtype=np.float32)
    padded[: values.shape[0]] = values
    return padded
