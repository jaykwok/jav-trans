from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_edge_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_edge_row(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(row["source_feature_path"]) as source:
        ptm = source["ptm"].astype(np.float32)
        mfcc = source["mfcc"].astype(np.float32)
    with np.load(row["feature_path"]) as labels:
        targets = labels["labels"].astype(np.int64)
        weights = labels["weights"].astype(np.float32)
    total = min(ptm.shape[0], mfcc.shape[0], targets.shape[0], weights.shape[0])
    position = (
        np.arange(total, dtype=np.float32) / max(1, total - 1)
    ).reshape(-1, 1)
    features = np.concatenate((ptm[:total], mfcc[:total], position), axis=1)
    return features, targets[:total], weights[:total]


def edge_normalization(rows: list[dict]) -> dict[str, list[float]]:
    first, _labels, _weights = load_edge_row(rows[0])
    feature_sum = np.zeros(first.shape[1], dtype=np.float64)
    square_sum = np.zeros(first.shape[1], dtype=np.float64)
    weight_sum = 0.0
    for row in rows:
        features, _labels, weights = load_edge_row(row)
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


def normalize_edge_features(features: np.ndarray, normalization: dict) -> np.ndarray:
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    return np.ascontiguousarray((features - mean) / np.maximum(std, 1e-6))
