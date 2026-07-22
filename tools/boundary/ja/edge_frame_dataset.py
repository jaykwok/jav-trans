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
    lengths = {
        "ptm": int(ptm.shape[0]),
        "mfcc": int(mfcc.shape[0]),
        "labels": int(targets.shape[0]),
        "weights": int(weights.shape[0]),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"edge frame dataset row has mismatched frame counts: {lengths}"
        )
    total = lengths["ptm"]
    if total <= 0:
        raise ValueError("edge frame dataset row is empty")
    position = (
        np.arange(total, dtype=np.float32) / max(1, total - 1)
    ).reshape(-1, 1)
    features = np.concatenate((ptm[:total], mfcc[:total], position), axis=1)
    return features, targets[:total], weights[:total]


def normalize_edge_features(features: np.ndarray, normalization: dict) -> np.ndarray:
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    return np.ascontiguousarray((features - mean) / np.maximum(std, 1e-6))
