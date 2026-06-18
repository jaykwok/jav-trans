from __future__ import annotations

import json
import math
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
    FeatureFrameScorer,
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
    dropout: float = 0.05
    eval_ratio: float = 0.1
    threshold: float = 0.5
    max_eval_windows: int = 256


@dataclass(frozen=True)
class FeatureScorerTrainMetrics:
    steps: int
    loss: float
    eval_loss: float
    frame_accuracy: float
    positive_ratio: float
    predicted_positive_ratio: float
    precision: float
    recall: float
    f1: float
    train_windows: int
    eval_windows: int
    input_dim: int
    ptm_dim: int
    mfcc_dim: int
    checkpoint: str
    metrics_path: str
    threshold: float


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


def train_feature_frame_scorer(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
    output_dir: Path,
    config: FeatureScorerTrainConfig,
    labels_path: str = "",
    feature_manifest_path: str = "",
) -> FeatureScorerTrainMetrics:
    import torch
    from torch import nn

    rows = [dict(row) for row in feature_manifest_rows]
    if not rows:
        raise ValueError("at least one feature manifest row is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    rows = [row for row in rows if _feature_row_frame_count(row, records) > 0]
    if not rows:
        raise ValueError("no feature rows have usable frames")

    ptm_dim = int(rows[0]["ptm_dim"])
    mfcc_dim = int(rows[0]["mfcc_dim"])
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
    normalization = compute_feature_normalization(records=records, feature_manifest_rows=train_rows)

    device = torch.device(config.device)
    model = FeatureFrameScorer(
        input_dim=input_dim,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    train_order = shuffled_window_order(len(train_rows), seed=config.seed)
    losses: list[float] = []

    for step in range(config.max_steps):
        row = train_rows[train_order[step % len(train_order)]]
        features, labels, weights = feature_training_tensors(
            row=row,
            records=records,
            normalization=normalization,
        )
        if float(np.sum(weights)) <= 0.0:
            continue
        feature_tensor = torch.from_numpy(features).to(device).unsqueeze(0)
        label_tensor = torch.from_numpy(labels).to(device).unsqueeze(0)
        weight_tensor = torch.from_numpy(weights).to(device).unsqueeze(0)
        logits = model(feature_tensor)
        loss_values = criterion(logits, label_tensor)
        loss = (loss_values * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    eval_metrics = evaluate_feature_frame_scorer(
        model=model,
        records=records,
        rows=eval_rows or train_rows,
        normalization=normalization,
        device=device,
        threshold=config.threshold,
        max_windows=config.max_eval_windows,
    )
    checkpoint_path = output_dir / "speech_boundary_ja_feature_scorer.pt"
    metrics_path = output_dir / "train_metrics.json"
    model_config = {
        "ptm_dim": ptm_dim,
        "mfcc_dim": mfcc_dim,
        "input_dim": input_dim,
        "hidden_size": int(config.hidden_size),
        "dropout": float(config.dropout),
    }
    torch.save(
        build_feature_frame_scorer_checkpoint(
            model=model,
            model_config=model_config,
            normalization=normalization,
            metadata={
                "operating_point": "qwen-feature-scorer-hard-negative-v1",
                "labels": labels_path,
                "feature_manifest": feature_manifest_path,
                "records": len(records),
                "train_windows": len(train_rows),
                "eval_windows": len(eval_rows),
                "trained_steps": int(config.max_steps),
                "config": asdict(config),
            },
        ),
        checkpoint_path,
    )
    metrics = FeatureScorerTrainMetrics(
        steps=config.max_steps,
        loss=float(np.mean(losses)) if losses else 0.0,
        eval_loss=float(eval_metrics["loss"]),
        frame_accuracy=float(eval_metrics["frame_accuracy"]),
        positive_ratio=float(eval_metrics["positive_ratio"]),
        predicted_positive_ratio=float(eval_metrics["predicted_positive_ratio"]),
        precision=float(eval_metrics["precision"]),
        recall=float(eval_metrics["recall"]),
        f1=float(eval_metrics["f1"]),
        train_windows=len(train_rows),
        eval_windows=len(eval_rows),
        input_dim=input_dim,
        ptm_dim=ptm_dim,
        mfcc_dim=mfcc_dim,
        checkpoint=str(checkpoint_path),
        metrics_path=str(metrics_path),
        threshold=float(config.threshold),
    )
    metrics_path.write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def compute_feature_normalization(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    total_weight = 0.0
    feature_sum: np.ndarray | None = None
    feature_square_sum: np.ndarray | None = None
    frame_count_total = 0
    for row in feature_manifest_rows:
        features, _labels, weights = _feature_training_arrays(row=row, records=records)
        frame_weight = weights.reshape(-1, 1).astype(np.float64, copy=False)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, weights = _feature_training_arrays(row=row, records=records)
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
    max_windows: int = 256,
) -> dict[str, float | int]:
    import torch
    from torch import nn

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    counts = {
        "frames": 0,
        "correct": 0,
        "positives": 0,
        "predicted_positives": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
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
            )
            if float(np.sum(weights)) <= 0.0:
                continue
            feature_tensor = torch.from_numpy(features).to(device).unsqueeze(0)
            label_tensor = torch.from_numpy(labels).to(device).unsqueeze(0)
            weight_tensor = torch.from_numpy(weights).to(device).unsqueeze(0)
            logits = model(feature_tensor)
            loss_values = criterion(logits, label_tensor)
            loss_sum += float((loss_values * weight_tensor).sum().detach().cpu())
            weight_sum += float(weight_tensor.sum().detach().cpu())
            probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
            mask = weights > 0.0
            pred_values = (probabilities[mask] >= float(threshold)).astype(np.int32)
            label_values = (labels[mask] > 0.5).astype(np.int32)
            counts["frames"] += int(label_values.size)
            counts["correct"] += int((pred_values == label_values).sum())
            counts["positives"] += int(label_values.sum())
            counts["predicted_positives"] += int(pred_values.sum())
            counts["true_positive"] += int(np.logical_and(label_values == 1, pred_values == 1).sum())
            counts["false_positive"] += int(np.logical_and(label_values == 0, pred_values == 1).sum())
            counts["false_negative"] += int(np.logical_and(label_values == 1, pred_values == 0).sum())
            windows += 1
    metrics = metrics_from_frame_counts(counts=counts, windows=windows, threshold=threshold)
    return {
        "loss": loss_sum / max(1e-6, weight_sum),
        "frame_accuracy": metrics.frame_accuracy,
        "positive_ratio": metrics.positive_ratio,
        "predicted_positive_ratio": metrics.predicted_positive_ratio,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "frames": counts["frames"],
        "windows": windows,
    }


def frame_classification_counts(
    *,
    labels: Iterable[int | float],
    predictions: Iterable[int | float],
) -> dict[str, int]:
    label_values = [1 if float(value) > 0.5 else 0 for value in labels]
    pred_values = [1 if float(value) > 0.5 else 0 for value in predictions]
    frame_total = min(len(label_values), len(pred_values))
    counts = {
        "frames": frame_total,
        "correct": 0,
        "positives": 0,
        "predicted_positives": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    for label, prediction in zip(label_values[:frame_total], pred_values[:frame_total]):
        counts["correct"] += int(label == prediction)
        counts["positives"] += int(label)
        counts["predicted_positives"] += int(prediction)
        counts["true_positive"] += int(label and prediction)
        counts["false_positive"] += int((not label) and prediction)
        counts["false_negative"] += int(label and (not prediction))
    return counts


def metrics_from_frame_counts(
    *,
    counts: Mapping[str, int],
    windows: int,
    threshold: float = 0.5,
    metrics_path: str = "",
) -> EvalMetrics:
    frames = int(counts.get("frames") or 0)
    tp = int(counts.get("true_positive") or 0)
    fp = int(counts.get("false_positive") or 0)
    fn = int(counts.get("false_negative") or 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_index = int(row["label_index"])
    record = records[label_index]
    ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
    weights = effective_frame_weights(record)
    frame_total = min(ptm.shape[0], mfcc.shape[0], len(record.speech_frames), len(weights))
    if frame_total <= 0:
        raise ValueError(f"feature row has no usable frames: label_index={label_index}")
    features = np.concatenate(
        [
            np.ascontiguousarray(ptm[:frame_total], dtype=np.float32),
            np.ascontiguousarray(mfcc[:frame_total], dtype=np.float32),
        ],
        axis=1,
    )
    return (
        features,
        np.asarray(record.speech_frames[:frame_total], dtype=np.float32),
        np.asarray(weights[:frame_total], dtype=np.float32),
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
