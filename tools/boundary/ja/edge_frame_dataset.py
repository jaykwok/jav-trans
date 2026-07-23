from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_edge_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows: list[dict] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid edge dataset JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"edge dataset row must be an object at {path}:{line_number}")
            rows.append(row)
        return rows


def load_edge_row(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(row["source_feature_path"], allow_pickle=False) as source:
        if "ptm" not in source.files or "mfcc" not in source.files:
            raise ValueError("edge source feature payload requires ptm/mfcc arrays")
        ptm = np.asarray(source["ptm"], dtype=np.float32)
        mfcc = np.asarray(source["mfcc"], dtype=np.float32)
    with np.load(row["feature_path"], allow_pickle=False) as labels:
        if "labels" not in labels.files or "weights" not in labels.files:
            raise ValueError("edge label payload requires labels/weights arrays")
        targets = np.asarray(labels["labels"], dtype=np.int64)
        weights = np.asarray(labels["weights"], dtype=np.float32)
    if ptm.ndim != 2 or mfcc.ndim != 2:
        raise ValueError("edge PTM/MFCC features must be 2D")
    if targets.ndim != 1 or weights.ndim != 1:
        raise ValueError("edge labels/weights must be 1D")
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
    if not np.isfinite(ptm).all() or not np.isfinite(mfcc).all():
        raise ValueError("edge frame features contain non-finite values")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("edge frame weights must be finite and non-negative")
    if np.any(~np.isin(targets, (0, 1, 2, -100))):
        raise ValueError("edge frame labels contain an unsupported class")
    position = (
        np.arange(total, dtype=np.float32) / max(1, total - 1)
    ).reshape(-1, 1)
    features = np.concatenate((ptm[:total], mfcc[:total], position), axis=1)
    return features, targets[:total], weights[:total]


def normalize_edge_features(features: np.ndarray, normalization: dict) -> np.ndarray:
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or mean.ndim != 1 or std.ndim != 1:
        raise ValueError("edge normalization requires 2D features and 1D statistics")
    if values.shape[1] != mean.size or mean.shape != std.shape:
        raise ValueError("edge normalization feature width mismatch")
    if not np.isfinite(values).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("edge normalization contains non-finite values")
    if np.any(std <= 0.0):
        raise ValueError("edge normalization standard deviations must be positive")
    return np.ascontiguousarray((values - mean) / np.maximum(std, 1e-6))
