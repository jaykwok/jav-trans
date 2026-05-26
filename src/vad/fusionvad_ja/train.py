from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from audio.loading import load_audio_16k_mono
from vad.fusionvad_ja.dataset import LabelRecord
from vad.fusionvad_ja.dataset import effective_frame_weights
from vad.fusionvad_ja.features import load_cached_feature
from vad.fusionvad_ja.manifest import TrainingExample
from vad.fusionvad_ja.model import AdditionFusionBiLSTM, count_trainable_parameters


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
        loss = (criterion(logits, label_tensor) * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)
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
