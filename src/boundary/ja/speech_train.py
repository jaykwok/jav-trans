from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from boundary.ja.dataset import LabelRecord, effective_frame_weights
from boundary.ja.features import load_cached_feature
from boundary.ja.model import (
    SPEECH_ISLAND_SCORER_MODEL_ARCH,
    SPEECH_ISLAND_SCORER_OUTPUT_DIM,
    SPEECH_ISLAND_SCORER_SCHEMA,
    build_speech_island_scorer_checkpoint,
    build_speech_island_scorer_model,
)


@dataclass(frozen=True)
class SpeechIslandTrainConfig:
    max_steps: int = 3000
    learning_rate: float = 2e-4
    seed: int = 13
    device: str = "cuda"
    hidden_size: int = 128
    num_layers: int = 2
    state_size: int = 32
    num_heads: int = 4
    n_groups: int = 2
    conv_kernel: int = 4
    chunk_size: int = 8
    bidirectional: bool = True
    positive_weight: float = 1.0
    negative_weight: float = 15.0
    focal_gamma: float = 2.0
    eval_ratio: float = 0.1
    threshold: float = 0.5
    max_train_frames: int = 1024
    max_eval_frames: int = 1024
    max_eval_windows: int = 512
    log_every: int = 100
    ptm_dim: int = 128


@dataclass(frozen=True)
class SpeechIslandTrainMetrics:
    schema: str
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
    checkpoint: str
    metrics_path: str


def train_speech_island_scorer(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
    output_dir: Path,
    config: SpeechIslandTrainConfig,
    labels_path: str,
    feature_manifest_path: str,
    normalization_checkpoint: str = "",
    checkpoint_name: str = "speech_island_scorer_v8.pt",
) -> SpeechIslandTrainMetrics:
    import torch
    import torch.nn.functional as F

    rows = [dict(row) for row in feature_manifest_rows]
    if not rows:
        raise ValueError("at least one feature manifest row is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    order = [int(index) for index in rng.permutation(len(rows))]
    eval_count = max(1, int(round(len(rows) * config.eval_ratio)))
    eval_rows = [rows[index] for index in order[:eval_count]]
    train_rows = [rows[index] for index in order[eval_count:]]
    ptm_dim = int(config.ptm_dim)
    mfcc_dim = int(rows[0]["mfcc_dim"])
    input_dim = ptm_dim + mfcc_dim
    normalization = _normalization(
        records=records,
        rows=train_rows,
        input_dim=input_dim,
        checkpoint_path=normalization_checkpoint,
    )
    model_config = {
        "ptm_dim": ptm_dim,
        "mfcc_dim": mfcc_dim,
        "input_dim": input_dim,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "state_size": config.state_size,
        "num_heads": config.num_heads,
        "n_groups": config.n_groups,
        "conv_kernel": config.conv_kernel,
        "chunk_size": config.chunk_size,
        "bidirectional": config.bidirectional,
        "model_arch": SPEECH_ISLAND_SCORER_MODEL_ARCH,
        "output_dim": SPEECH_ISLAND_SCORER_OUTPUT_DIM,
    }
    device = torch.device(config.device)
    model = build_speech_island_scorer_model(
        schema=SPEECH_ISLAND_SCORER_SCHEMA,
        model_config=model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    started = time.monotonic()
    for step in range(config.max_steps):
        row = train_rows[step % len(train_rows)]
        features, labels, weights = _training_arrays(
            row, records, ptm_dim=ptm_dim
        )
        features, labels, weights = _crop(
            features,
            labels,
            weights,
            max_frames=config.max_train_frames,
            rng=rng,
            random=True,
        )
        features = _normalize(features, normalization)
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        label_tensor = torch.from_numpy(labels).unsqueeze(0).unsqueeze(-1).to(device)
        weight_tensor = torch.from_numpy(weights).unsqueeze(0).unsqueeze(-1).to(device)
        logits = model(feature_tensor)
        raw_loss = F.binary_cross_entropy_with_logits(logits, label_tensor, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = torch.where(label_tensor > 0.5, probabilities, 1.0 - probabilities)
        class_weights = torch.where(
            label_tensor > 0.5,
            torch.full_like(label_tensor, config.positive_weight),
            torch.full_like(label_tensor, config.negative_weight),
        )
        effective = weight_tensor * class_weights
        loss = (raw_loss * torch.pow(1.0 - pt, config.focal_gamma) * effective).sum()
        loss = loss / effective.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if config.log_every and (step + 1) % config.log_every == 0:
            print(
                f"speech_island_train={step + 1}/{config.max_steps} "
                f"loss={losses[-1]:.6f} avg_loss={np.mean(losses):.6f} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
    evaluation = _evaluate(
        model=model,
        records=records,
        rows=eval_rows[: config.max_eval_windows],
        normalization=normalization,
        config=config,
        device=device,
    )
    ptm_repo_id = str(rows[0].get("ptm") or "")
    checkpoint_path = output_dir / checkpoint_name
    torch.save(
        build_speech_island_scorer_checkpoint(
            model=model,
            model_config=model_config,
            normalization=normalization,
            metadata={
                "operating_point": "speech-island-high-recall-v8",
                "ptm_repo_id": ptm_repo_id,
                "labels": labels_path,
                "feature_manifest": feature_manifest_path,
                "trained_steps": config.max_steps,
                "config": asdict(config),
            },
        ),
        checkpoint_path,
    )
    metrics_path = output_dir / "train_metrics.json"
    metrics = SpeechIslandTrainMetrics(
        schema=SPEECH_ISLAND_SCORER_SCHEMA,
        steps=config.max_steps,
        loss=float(np.mean(losses)),
        eval_loss=float(evaluation["loss"]),
        frame_accuracy=float(evaluation["accuracy"]),
        positive_ratio=float(evaluation["positive_ratio"]),
        predicted_positive_ratio=float(evaluation["predicted_positive_ratio"]),
        precision=float(evaluation["precision"]),
        recall=float(evaluation["recall"]),
        f1=float(evaluation["f1"]),
        train_windows=len(train_rows),
        eval_windows=min(len(eval_rows), config.max_eval_windows),
        checkpoint=str(checkpoint_path),
        metrics_path=str(metrics_path),
    )
    metrics_path.write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _training_arrays(
    row: Mapping[str, Any],
    records: list[LabelRecord],
    *,
    ptm_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = records[int(row["label_index"])]
    ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
    weights = np.asarray(effective_frame_weights(record), dtype=np.float32)
    total = min(len(record.speech_frames), len(weights), ptm.shape[0], mfcc.shape[0])
    if ptm.shape[1] < ptm_dim:
        raise ValueError(f"training feature ptm_dim={ptm.shape[1]} is below {ptm_dim}")
    features = np.concatenate(
        (ptm[:total, :ptm_dim], mfcc[:total]), axis=1
    ).astype(np.float32)
    labels = np.asarray(record.speech_frames[:total], dtype=np.float32)
    return features, labels, weights[:total]


def _normalization(
    *,
    records: list[LabelRecord],
    rows: list[dict[str, Any]],
    input_dim: int,
    checkpoint_path: str,
) -> dict[str, Any]:
    if checkpoint_path:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        normalization = dict(payload["normalization"])
        if len(normalization["feature_mean"]) != input_dim:
            raise ValueError("normalization checkpoint input dim mismatch")
        return normalization
    feature_sum = np.zeros(input_dim, dtype=np.float64)
    square_sum = np.zeros(input_dim, dtype=np.float64)
    weight_sum = 0.0
    for row in rows:
        features, _labels, weights = _training_arrays(
            row, records, ptm_dim=input_dim - int(row["mfcc_dim"])
        )
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


def _normalize(features: np.ndarray, normalization: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    return np.ascontiguousarray((features - mean) / np.maximum(std, 1e-6))


def _crop(features, labels, weights, *, max_frames: int, rng, random: bool):
    if max_frames <= 0 or features.shape[0] <= max_frames:
        return features, labels, weights
    limit = features.shape[0] - max_frames
    start = int(rng.integers(0, limit + 1)) if random else limit // 2
    end = start + max_frames
    return features[start:end], labels[start:end], weights[start:end]


def _evaluate(*, model, records, rows, normalization, config, device) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    total_loss = 0.0
    total_weight = 0
    tp = fp = fn = correct = positives = predicted = frames = 0
    with torch.inference_mode():
        for row in rows:
            features, labels, weights = _training_arrays(
                row, records, ptm_dim=config.ptm_dim
            )
            features, labels, weights = _crop(
                features,
                labels,
                weights,
                max_frames=config.max_eval_frames,
                rng=None,
                random=False,
            )
            logits = model(
                torch.from_numpy(_normalize(features, normalization)).unsqueeze(0).to(device)
            )[0, :, 0]
            label_tensor = torch.from_numpy(labels).to(device)
            loss = F.binary_cross_entropy_with_logits(logits, label_tensor, reduction="none")
            total_loss += float((loss * torch.from_numpy(weights).to(device)).sum().cpu())
            total_weight += int(weights.sum())
            pred = (torch.sigmoid(logits).cpu().numpy() >= config.threshold).astype(np.int8)
            truth = (labels >= 0.5).astype(np.int8)
            active = weights > 0.0
            pred = pred[active]
            truth = truth[active]
            tp += int(((pred == 1) & (truth == 1)).sum())
            fp += int(((pred == 1) & (truth == 0)).sum())
            fn += int(((pred == 0) & (truth == 1)).sum())
            correct += int((pred == truth).sum())
            positives += int(truth.sum())
            predicted += int(pred.sum())
            frames += int(truth.size)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "loss": total_loss / max(1, total_weight),
        "accuracy": correct / max(1, frames),
        "positive_ratio": positives / max(1, frames),
        "predicted_positive_ratio": predicted / max(1, frames),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-9, precision + recall),
    }
