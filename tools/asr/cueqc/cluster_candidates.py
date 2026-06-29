#!/usr/bin/env python3
"""Cluster CueQC candidates with Torque Clustering (TORC).

The previous HDBSCAN / FINCH / UMAP / PCA backends were removed and the whole
clustering path now routes through :mod:`tools.asr.cueqc.torque`, a single-file
port of Jie Yang's parameter-free Torque Clustering algorithm. TORC decides the
number of clusters autonomously from the distance matrix, so this module no
longer exposes ``--method``, ``--reducer``, ``--min-clusters`` /
``--max-clusters`` or any backend-specific tuning flags — only the feature
space (Pre-ASR numeric / audit-oriented Pre-ASR summaries / structured / dense
/ fused embeddings) and the distance metric remain.

``pre_asr_coldstart`` is an audit-only feature space for first-pass Pre-ASR
CueQC labeling. It fuses structural, text-morphology and compact PTM views
before TORC. The output may seed training labels after human review; runtime
inference must not depend on these clusters.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.cueqc import normalize_feature_matrix
from tools.asr.cueqc.torque import torque_clustering, torque_merge_layer_preview


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec)) or 1.0


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _cosine_distance(left: list[float], right: list[float]) -> float:
    return 1.0 - (_dot(left, right) / (_norm(left) * _norm(right)))


def _distance(left: list[float], right: list[float], metric: str) -> float:
    if metric == "cosine":
        return _cosine_distance(left, right)
    if metric == "euclidean":
        return _euclidean_distance(left, right)
    raise ValueError(f"unsupported metric: {metric}")


def _l2_normalize_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in matrix:
        norm = math.sqrt(sum(value * value for value in row)) or 1.0
        normalized.append([float(value) / norm for value in row])
    return normalized


def _zscore_matrix(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    width = len(matrix[0])
    means: list[float] = []
    stds: list[float] = []
    for col in range(width):
        values = [row[col] for row in matrix]
        mean = sum(values) / max(1, len(values))
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        means.append(mean)
        stds.append(math.sqrt(variance) or 1.0)
    return [
        [(row[col] - means[col]) / stds[col] for col in range(width)]
        for row in matrix
    ]


def _numeric_vector(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        for key in ("vector", "values", "embedding"):
            if key in value:
                return _numeric_vector(value[key])
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            out.extend(_numeric_vector(item))
            continue
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            out.append(parsed)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _pre_asr_feature_names(rows: list[Mapping[str, Any]]) -> list[str]:
    for row in rows:
        features = row.get("features")
        names = row.get("feature_names")
        if not isinstance(features, Mapping):
            continue
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray)):
            ordered = [str(name) for name in names if str(name)]
            if ordered:
                return ordered
    names_seen: set[str] = set()
    for row in rows:
        features = row.get("features")
        if not isinstance(features, Mapping):
            continue
        for name, value in features.items():
            if name in names_seen:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                names_seen.add(str(name))
    return sorted(names_seen)


def _pre_asr_numeric_matrix(rows: list[Mapping[str, Any]]) -> tuple[list[list[float]], dict[str, Any]]:
    names = _pre_asr_feature_names(rows)
    if not names:
        return [], {
            "available": False,
            "source": "features",
            "feature_names": [],
            "rows_with_features": 0,
            "total_dim": 0,
        }
    matrix: list[list[float]] = []
    rows_with_features = 0
    rows_with_ptm_pooling = 0
    ptm_pooling_dim = 0
    for row in rows:
        features = row.get("features")
        if isinstance(features, Mapping):
            rows_with_features += 1
        else:
            features = {}
        pooled = _numeric_vector(row.get("pre_asr_ptm_pooled_features"))
        pooled_by_name: dict[str, float] = {}
        if pooled:
            missing_names = [name for name in names if name not in features]
            if len(missing_names) == len(pooled):
                pooled_by_name = {
                    name: float(value)
                    for name, value in zip(missing_names, pooled)
                }
                rows_with_ptm_pooling += 1
                ptm_pooling_dim = max(ptm_pooling_dim, len(pooled))
        matrix.append([
            _safe_float(
                features.get(name)
                if name in features
                else pooled_by_name.get(name)
            )
            for name in names
        ])
    return matrix, {
        "available": True,
        "source": (
            "features+pre_asr_ptm_pooled_features"
            if rows_with_ptm_pooling
            else "features"
        ),
        "feature_names": names,
        "rows_with_features": rows_with_features,
        "rows_with_ptm_pooling": rows_with_ptm_pooling,
        "ptm_pooling_dim": ptm_pooling_dim,
        "total_dim": len(names),
    }


_AUDIT_EXCLUDED_EXACT_NAMES = {
    "raw_start_s",
    "raw_end_s",
    "acoustic_start_s",
    "acoustic_end_s",
}
_AUDIT_EXCLUDED_PREFIXES = ("ptm_bin",)
_AUDIT_TEXT_FEATURE_NAMES = [
    "audit_text_empty",
    "audit_text_char_count",
    "audit_text_raw_char_count",
    "audit_text_unique_chars",
    "audit_text_unique_ratio",
    "audit_text_kana_ratio",
    "audit_text_kanji_ratio",
    "audit_text_chars_per_sec",
    "audit_text_repeat_run",
    "audit_text_repeat_unit_len",
    "audit_text_repeat_ratio",
    "audit_text_kana_only",
    "audit_text_has_kanji",
    "audit_text_has_latin_or_digit",
    "audit_text_has_stable_vocabulary",
]


def _pre_asr_audit_feature_names(names: Sequence[str]) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    excluded: list[str] = []
    for name in names:
        if name in _AUDIT_EXCLUDED_EXACT_NAMES or any(
            name.startswith(prefix) for prefix in _AUDIT_EXCLUDED_PREFIXES
        ):
            excluded.append(name)
            continue
        selected.append(name)
    return selected, excluded


def _audit_text_values(row: Mapping[str, Any]) -> list[float]:
    text_features = row.get("text_features")
    if not isinstance(text_features, Mapping):
        text_features = {}
    repeat = text_features.get("repeat_profile")
    if not isinstance(repeat, Mapping):
        repeat = {}
    char_count = _safe_float(text_features.get("char_count"))
    return [
        1.0 if char_count <= 0.0 else 0.0,
        char_count,
        _safe_float(text_features.get("raw_char_count")),
        _safe_float(text_features.get("unique_chars")),
        _safe_float(text_features.get("unique_ratio")),
        _safe_float(text_features.get("kana_ratio")),
        _safe_float(text_features.get("kanji_ratio")),
        _safe_float(text_features.get("chars_per_sec")),
        _safe_float(repeat.get("run")),
        _safe_float(repeat.get("unit_len")),
        _safe_float(repeat.get("ratio")),
        1.0 if bool(text_features.get("kana_only")) else 0.0,
        1.0 if bool(text_features.get("has_kanji")) else 0.0,
        1.0 if bool(text_features.get("has_latin_or_digit")) else 0.0,
        1.0 if bool(text_features.get("has_stable_vocabulary")) else 0.0,
    ]


def _pre_asr_audit_numeric_matrix(rows: list[Mapping[str, Any]]) -> tuple[list[list[float]], dict[str, Any]]:
    names = _pre_asr_feature_names(rows)
    selected_names, excluded_names = _pre_asr_audit_feature_names(names)
    if not selected_names and not rows:
        return [], {
            "available": False,
            "source": "features_without_absolute_time_or_ptm_bins+audit_text_morphology",
            "feature_names": [],
            "excluded_feature_names": excluded_names,
            "excluded_feature_count": len(excluded_names),
            "total_dim": 0,
        }
    matrix: list[list[float]] = []
    rows_with_features = 0
    rows_with_text_features = 0
    for row in rows:
        features = row.get("features")
        if isinstance(features, Mapping):
            rows_with_features += 1
        else:
            features = {}
        if isinstance(row.get("text_features"), Mapping):
            rows_with_text_features += 1
        matrix.append(
            [_safe_float(features.get(name)) for name in selected_names]
            + _audit_text_values(row)
        )
    feature_names = list(selected_names) + list(_AUDIT_TEXT_FEATURE_NAMES)
    return matrix, {
        "available": True,
        "source": "features_without_absolute_time_or_ptm_bins+audit_text_morphology",
        "feature_names": feature_names,
        "selected_pre_asr_feature_names": selected_names,
        "audit_text_feature_names": list(_AUDIT_TEXT_FEATURE_NAMES),
        "excluded_feature_names": excluded_names,
        "excluded_feature_count": len(excluded_names),
        "excluded_exact_names": sorted(_AUDIT_EXCLUDED_EXACT_NAMES),
        "excluded_prefixes": list(_AUDIT_EXCLUDED_PREFIXES),
        "rows_with_features": rows_with_features,
        "rows_with_text_features": rows_with_text_features,
        "total_dim": len(feature_names),
    }


def _pre_asr_audit_structural_matrix(rows: list[Mapping[str, Any]]) -> tuple[list[list[float]], dict[str, Any]]:
    names = _pre_asr_feature_names(rows)
    selected_names, excluded_names = _pre_asr_audit_feature_names(names)
    if not selected_names:
        return [], {
            "available": False,
            "source": "features_without_absolute_time_or_ptm_bins",
            "feature_names": [],
            "excluded_feature_names": excluded_names,
            "excluded_feature_count": len(excluded_names),
            "total_dim": 0,
        }
    matrix: list[list[float]] = []
    rows_with_features = 0
    for row in rows:
        features = row.get("features")
        if isinstance(features, Mapping):
            rows_with_features += 1
        else:
            features = {}
        matrix.append([_safe_float(features.get(name)) for name in selected_names])
    return matrix, {
        "available": True,
        "source": "features_without_absolute_time_or_ptm_bins",
        "feature_names": selected_names,
        "excluded_feature_names": excluded_names,
        "excluded_feature_count": len(excluded_names),
        "rows_with_features": rows_with_features,
        "total_dim": len(selected_names),
    }


def _audit_text_matrix(rows: list[Mapping[str, Any]]) -> tuple[list[list[float]], dict[str, Any]]:
    if not rows:
        return [], {
            "available": False,
            "source": "audit_text_morphology",
            "feature_names": list(_AUDIT_TEXT_FEATURE_NAMES),
            "total_dim": len(_AUDIT_TEXT_FEATURE_NAMES),
        }
    rows_with_text_features = sum(1 for row in rows if isinstance(row.get("text_features"), Mapping))
    return [_audit_text_values(row) for row in rows], {
        "available": True,
        "source": "audit_text_morphology",
        "feature_names": list(_AUDIT_TEXT_FEATURE_NAMES),
        "rows_with_text_features": rows_with_text_features,
        "total_dim": len(_AUDIT_TEXT_FEATURE_NAMES),
    }


def _embedding_vectors(row: Mapping[str, Any]) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    for key, name in (
        ("text_embedding", "text"),
        ("audio_embedding", "audio"),
        ("semantic_embedding", "semantic"),
        ("acoustic_embedding", "audio"),
        ("embedding", "generic"),
    ):
        vector = _numeric_vector(row.get(key))
        if vector and name not in vectors:
            vectors[name] = vector
    embeddings = row.get("embeddings")
    if isinstance(embeddings, Mapping):
        for name, payload in embeddings.items():
            vector = _numeric_vector(payload)
            if vector:
                vectors[str(name)] = vector
    return vectors


def _dense_embedding_matrix(rows: list[Mapping[str, Any]]) -> tuple[list[list[float]], dict[str, Any]]:
    row_vectors = [_embedding_vectors(row) for row in rows]
    names = sorted({name for vectors in row_vectors for name in vectors})
    if not names:
        return [], {
            "available": False,
            "sources": [],
            "dimensions": {},
            "rows_with_any_embedding": 0,
        }
    dimensions = {
        name: max((len(vectors.get(name, [])) for vectors in row_vectors), default=0)
        for name in names
    }
    matrix: list[list[float]] = []
    rows_with_any = 0
    source_counts: Counter[str] = Counter()
    for vectors in row_vectors:
        row_values: list[float] = []
        if vectors:
            rows_with_any += 1
        for name in names:
            dim = dimensions[name]
            values = list(vectors.get(name, []))[:dim]
            if values:
                source_counts[name] += 1
            row_values.extend(values + [0.0] * max(0, dim - len(values)))
        matrix.append(row_values)
    return matrix, {
        "available": True,
        "sources": names,
        "dimensions": dimensions,
        "rows_with_any_embedding": rows_with_any,
        "source_counts": dict(source_counts),
        "total_dim": len(matrix[0]) if matrix else 0,
    }


def _feature_matrix(
    rows: list[dict[str, Any]],
    *,
    feature_space: str,
) -> tuple[list[list[float]], dict[str, Any]]:
    pre_asr, pre_asr_meta = _pre_asr_numeric_matrix(rows)
    pre_asr_audit, pre_asr_audit_meta = _pre_asr_audit_numeric_matrix(rows)
    structured: list[list[float]] | None = None
    dense, dense_meta = _dense_embedding_matrix(rows)
    resolved = feature_space
    if feature_space == "auto":
        resolved = "pre_asr" if pre_asr else "fused" if dense else "structured"
    if resolved == "pre_asr":
        if not pre_asr:
            structured = normalize_feature_matrix(rows)
            return structured, {
                "requested": feature_space,
                "resolved": "structured",
                "fallback_reason": "no_pre_asr_numeric_features",
                "structured_dim": len(structured[0]) if structured else 0,
                "pre_asr": pre_asr_meta,
                "pre_asr_audit": pre_asr_audit_meta,
                "dense": dense_meta,
            }
        pre_asr_norm = _l2_normalize_matrix(_zscore_matrix(pre_asr))
        return pre_asr_norm, {
            "requested": feature_space,
            "resolved": resolved,
            "pre_asr": pre_asr_meta,
            "pre_asr_audit": pre_asr_audit_meta,
            "dense": dense_meta,
            "normalization": "zscore_then_l2",
        }
    if resolved == "pre_asr_audit":
        if not pre_asr_audit:
            structured = normalize_feature_matrix(rows)
            return structured, {
                "requested": feature_space,
                "resolved": "structured",
                "fallback_reason": "no_pre_asr_audit_features",
                "structured_dim": len(structured[0]) if structured else 0,
                "pre_asr": pre_asr_meta,
                "pre_asr_audit": pre_asr_audit_meta,
                "dense": dense_meta,
            }
        audit_norm = _l2_normalize_matrix(_zscore_matrix(pre_asr_audit))
        return audit_norm, {
            "requested": feature_space,
            "resolved": resolved,
            "pre_asr": pre_asr_meta,
            "pre_asr_audit": pre_asr_audit_meta,
            "dense": dense_meta,
            "normalization": "zscore_then_l2",
            "audit_note": (
                "excludes absolute timeline coordinates and high-dimensional PTM "
                "bins; includes audit-only text morphology for human cluster review"
            ),
        }
    structured = normalize_feature_matrix(rows)
    if resolved == "structured":
        return structured, {
            "requested": feature_space,
            "resolved": resolved,
            "structured_dim": len(structured[0]) if structured else 0,
            "pre_asr": pre_asr_meta,
            "pre_asr_audit": pre_asr_audit_meta,
            "dense": dense_meta,
        }
    if resolved == "dense":
        if not dense:
            return structured, {
                "requested": feature_space,
                "resolved": "structured",
                "fallback_reason": "no_dense_embeddings",
                "structured_dim": len(structured[0]) if structured else 0,
                "pre_asr": pre_asr_meta,
                "pre_asr_audit": pre_asr_audit_meta,
                "dense": dense_meta,
            }
        dense_norm = _l2_normalize_matrix(_zscore_matrix(dense))
        return dense_norm, {
            "requested": feature_space,
            "resolved": resolved,
            "structured_dim": len(structured[0]) if structured else 0,
            "pre_asr": pre_asr_meta,
            "pre_asr_audit": pre_asr_audit_meta,
            "dense": dense_meta,
        }
    if resolved != "fused":
        raise ValueError(f"unsupported feature space: {feature_space}")
    if not dense:
        return structured, {
            "requested": feature_space,
            "resolved": "structured",
            "fallback_reason": "no_dense_embeddings",
            "structured_dim": len(structured[0]) if structured else 0,
            "pre_asr": pre_asr_meta,
            "pre_asr_audit": pre_asr_audit_meta,
            "dense": dense_meta,
        }
    dense_norm = _l2_normalize_matrix(_zscore_matrix(dense))
    structured_norm = _l2_normalize_matrix(structured)
    fused = [
        list(structured_row) + list(dense_row)
        for structured_row, dense_row in zip(structured_norm, dense_norm)
    ]
    return fused, {
        "requested": feature_space,
        "resolved": "fused",
        "structured_dim": len(structured[0]) if structured else 0,
        "pre_asr": pre_asr_meta,
        "pre_asr_audit": pre_asr_audit_meta,
        "dense": dense_meta,
        "fused_dim": len(fused[0]) if fused else 0,
        "block_scaling": "l2_per_block",
    }


def _distance_matrix(matrix: Sequence[Sequence[float]], *, metric: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("distance matrix input must be 2-D")
    n = values.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if metric == "euclidean":
        return squareform(pdist(values, metric="euclidean"))
    if metric == "cosine":
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        safe = np.divide(values, np.where(norms == 0.0, 1.0, norms))
        distances = 1.0 - np.clip(safe @ safe.T, -1.0, 1.0)
        np.fill_diagonal(distances, 0.0)
        return distances
    raise ValueError(f"unsupported metric: {metric}")


def _ptm_compact_matrix(
    rows: list[Mapping[str, Any]],
    *,
    components: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Build compact PTM features for audit-only second-stage clustering."""
    vectors = [_numeric_vector(row.get("pre_asr_ptm_pooled_features")) for row in rows]
    max_dim = max((len(vector) for vector in vectors), default=0)
    rows_with_ptm = sum(1 for vector in vectors if vector)
    if max_dim <= 0 or rows_with_ptm < 2:
        return [], {
            "available": False,
            "source": "pre_asr_ptm_pooled_features",
            "rows_with_ptm": rows_with_ptm,
            "raw_dim": max_dim,
            "components": 0,
        }

    values = np.zeros((len(rows), max_dim), dtype=np.float64)
    for row_index, vector in enumerate(vectors):
        if not vector:
            continue
        width = min(max_dim, len(vector))
        values[row_index, :width] = np.asarray(vector[:width], dtype=np.float64)
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    stds[stds == 0.0] = 1.0
    z_values = (values - means) / stds
    component_count = max(1, min(int(components), z_values.shape[0] - 1, z_values.shape[1]))
    if component_count <= 0:
        return [], {
            "available": False,
            "source": "pre_asr_ptm_pooled_features",
            "rows_with_ptm": rows_with_ptm,
            "raw_dim": max_dim,
            "components": 0,
        }
    centered = z_values - z_values.mean(axis=0, keepdims=True)
    try:
        left, singular, _right = np.linalg.svd(centered, full_matrices=False)
        compact = left[:, :component_count] * singular[:component_count]
        method = "zscore_svd"
    except np.linalg.LinAlgError:
        compact = centered[:, :component_count]
        method = "zscore_truncated"
    matrix = _l2_normalize_matrix(compact.tolist())
    return matrix, {
        "available": True,
        "source": "pre_asr_ptm_pooled_features",
        "rows_with_ptm": rows_with_ptm,
        "raw_dim": max_dim,
        "components": component_count,
        "method": method,
        "normalization": "zscore_then_svd_then_l2",
    }


def _normalized_distance_view(matrix: list[list[float]], *, metric: str) -> np.ndarray:
    distances = _distance_matrix(matrix, metric=metric)
    if distances.size == 0:
        return distances
    finite = distances[np.isfinite(distances) & (distances > 0.0)]
    if finite.size == 0:
        return np.zeros_like(distances, dtype=np.float64)
    scale = float(np.percentile(finite, 90))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = float(finite.max()) or 1.0
    normalized = np.clip(distances / scale, 0.0, 1.5)
    np.fill_diagonal(normalized, 0.0)
    return normalized


def _matrix_has_signal(matrix: list[list[float]]) -> bool:
    if len(matrix) < 2 or not matrix or not matrix[0]:
        return False
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        return False
    return bool(np.any(np.nanstd(values, axis=0) > 1e-8))


def _knn_adjacency(distances: np.ndarray, *, k: int) -> np.ndarray:
    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=np.int16)
    effective_k = max(1, min(int(k), n - 1))
    adjacency = np.zeros((n, n), dtype=np.int16)
    for row_index in range(n):
        order = np.argsort(distances[row_index], kind="mergesort")
        neighbors = [int(index) for index in order if int(index) != row_index][:effective_k]
        adjacency[row_index, neighbors] = 1
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def _pre_asr_coldstart_inputs(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ptm_components: int,
    graph_k: int,
) -> tuple[list[list[float]], np.ndarray, dict[str, Any]]:
    """Build an audit-only multi-view distance matrix for cold-start labeling."""
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float64), {
            "requested": "pre_asr_coldstart",
            "resolved": "pre_asr_coldstart",
            "source": "multi_view_graph_distance",
            "audit_only": True,
            "runtime_contract": "cold_start_labeling_only_not_runtime_inference",
            "view_count": 0,
        }
    view_specs: list[tuple[str, list[list[float]], dict[str, Any], float]] = []
    structural, structural_meta = _pre_asr_audit_structural_matrix(rows)
    if _matrix_has_signal(structural):
        view_specs.append(("structure", _l2_normalize_matrix(_zscore_matrix(structural)), structural_meta, 1.0))
    text_matrix, text_meta = _audit_text_matrix(rows)
    if _matrix_has_signal(text_matrix):
        view_specs.append(("text_morphology", _l2_normalize_matrix(_zscore_matrix(text_matrix)), text_meta, 0.85))
    ptm_matrix, ptm_meta = _ptm_compact_matrix(rows, components=ptm_components)
    if _matrix_has_signal(ptm_matrix):
        view_specs.append(("compact_ptm", ptm_matrix, ptm_meta, 1.15))

    if not view_specs:
        structured = normalize_feature_matrix(rows)
        return structured, _distance_matrix(structured, metric=metric), {
            "requested": "pre_asr_coldstart",
            "resolved": "structured",
            "fallback_reason": "no_coldstart_views",
            "structured_dim": len(structured[0]) if structured else 0,
        }

    n = len(rows)
    fused = np.zeros((n, n), dtype=np.float64)
    support = np.zeros((n, n), dtype=np.int16)
    total_weight = 0.0
    view_meta: dict[str, Any] = {}
    representative_blocks: list[list[list[float]]] = []
    for name, matrix, meta, weight in view_specs:
        distances = _normalized_distance_view(matrix, metric=metric)
        adjacency = _knn_adjacency(distances, k=graph_k)
        fused += float(weight) * distances
        support += adjacency
        total_weight += float(weight)
        representative_blocks.append(matrix)
        view_meta[name] = {
            **meta,
            "weight": weight,
            "graph_k": max(1, min(int(graph_k), max(1, n - 1))) if n > 1 else 0,
            "knn_edge_count": int(np.sum(adjacency) // 2),
        }

    fused = fused / max(total_weight, 1e-9)
    if len(view_specs) > 1:
        strong_support = support >= min(2, len(view_specs))
        weak_support = (support > 0) & ~strong_support
        fused = np.where(strong_support, fused * 0.72, np.where(weak_support, fused * 0.92, np.minimum(1.5, fused + 0.18)))
    np.fill_diagonal(fused, 0.0)
    representative_matrix = [
        [value for block in representative_blocks for value in block[row_index]]
        for row_index in range(n)
    ]
    return representative_matrix, fused, {
        "requested": "pre_asr_coldstart",
        "resolved": "pre_asr_coldstart",
        "source": "multi_view_graph_distance",
        "audit_only": True,
        "runtime_contract": "cold_start_labeling_only_not_runtime_inference",
        "views": view_meta,
        "view_count": len(view_specs),
        "ptm_components_requested": ptm_components,
        "support_rule": "average_normalized_distances_then_reduce_pairs_connected_in_multiple_views",
        "normalization": "zscore_then_l2_per_view_distance_percentile_scale",
        "representative_dim": len(representative_matrix[0]) if representative_matrix else 0,
    }


def _suggest_small_tail_merges(
    labels: list[str],
    distances: np.ndarray,
    *,
    min_cluster_size: int,
) -> dict[str, Any]:
    if min_cluster_size <= 1 or len(labels) <= 1:
        return {"enabled": False, "reason": "disabled", "suggestions": []}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[str(label)].append(index)
    large_labels = [label for label, indexes in grouped.items() if len(indexes) >= min_cluster_size]
    small_labels = [label for label, indexes in grouped.items() if len(indexes) < min_cluster_size]
    if not small_labels or not large_labels:
        return {
            "enabled": True,
            "min_cluster_size": min_cluster_size,
            "suggestion_count": 0,
            "reason": "no_small_or_no_large_cluster",
            "suggestions": [],
        }

    suggestions: list[dict[str, Any]] = []
    for small_label in small_labels:
        small_indexes = grouped[small_label]
        candidates: list[tuple[str, float]] = []
        for large_label in large_labels:
            large_indexes = grouped[large_label]
            distance = float(np.mean(distances[np.ix_(small_indexes, large_indexes)]))
            candidates.append((large_label, distance))
        candidates.sort(key=lambda item: (item[1], item[0]))
        if not candidates:
            continue
        best_target, best_distance = candidates[0]
        suggestions.append({
            "schema": "cueqc_tail_merge_suggestion_v1",
            "from_cluster_id": small_label,
            "target_cluster_id": best_target,
            "sample_count": len(small_indexes),
            "mean_distance": round(best_distance, 6),
            "candidate_targets": [
                {"cluster_id": label, "mean_distance": round(distance, 6)}
                for label, distance in candidates[:3]
            ],
            "action": "suggest_merge_review",
            "requires_human_confirmation": True,
        })

    return {
        "enabled": True,
        "min_cluster_size": min_cluster_size,
        "small_cluster_count": len(small_labels),
        "large_cluster_count": len(large_labels),
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
        "label_policy": "suggestions_do_not_change_cluster_id_or_training_labels_without_human_confirmation",
    }


def _relabel_by_size(labels: Sequence[int]) -> list[str]:
    """Map integer TORC labels to ``cluster_NN`` / ``noise_00`` strings.

    Noise points (label < 0) collapse into a single ``noise_00`` bucket, matching
    the audit HTML's expectation of one review bucket.
    """
    counts = Counter(int(label) for label in labels if int(label) >= 0)
    order = {
        label: rank
        for rank, (label, _count) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
    }
    out: list[str] = []
    for label in labels:
        raw = int(label)
        out.append(f"cluster_{order[raw]:02d}" if raw >= 0 else "noise_00")
    return out


def _relabel_string_labels_by_size(labels: Sequence[str]) -> list[str]:
    counts = Counter(str(label) for label in labels)
    order = {
        label: rank
        for rank, (label, _count) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
    }
    return [f"cluster_{order[str(label)]:02d}" for label in labels]


def _duration_value(row: Mapping[str, Any]) -> float:
    duration = _safe_float(row.get("duration_s"))
    if duration > 0.0:
        return duration
    start = _safe_float(row.get("start"))
    end = _safe_float(row.get("end"))
    return max(0.0, end - start)


def _audit_risk_score(row: Mapping[str, Any]) -> float:
    text_features = row.get("text_features")
    if not isinstance(text_features, Mapping):
        text_features = {}
    repeat = text_features.get("repeat_profile")
    if not isinstance(repeat, Mapping):
        repeat = {}
    duration = _duration_value(row)
    char_count = _safe_float(text_features.get("char_count"))
    no_text = 1.0 if char_count <= 0.0 else 0.0
    repeated = min(1.0, _safe_float(repeat.get("ratio")))
    below_min = 1.0 if bool(row.get("below_subtitle_min_duration")) else 0.0
    low_confidence = max(0.0, 1.0 - _safe_float(row.get("cluster_confidence"), 1.0))
    short = 1.0 if duration < 0.83 else 0.0
    long = min(1.0, duration / 7.0)
    return round(
        0.26 * no_text
        + 0.22 * repeated
        + 0.18 * below_min
        + 0.14 * short
        + 0.12 * long
        + 0.08 * low_confidence,
        6,
    )


def _sample_stub(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "video_id": row.get("video_id", ""),
        "chunk_index": row.get("chunk_index"),
        "start": row.get("start"),
        "end": row.get("end"),
        "duration_s": row.get("duration_s"),
        "text_preview": row.get("text_preview") or row.get("text") or "",
        "audit_sampling_roles": row.get("audit_sampling_roles", []),
        "audit_sampling_score": row.get("audit_sampling_score"),
        "cluster_centroid_distance": row.get("cluster_centroid_distance"),
    }


def _percentile(values: Sequence[float], percentile: float, default: float = 0.0) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return default
    return float(np.percentile(np.asarray(finite, dtype=np.float64), percentile))


def _std(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return 0.0
    return float(np.std(np.asarray(finite, dtype=np.float64)))


def _feature_value(row: Mapping[str, Any], names: Sequence[str]) -> float:
    features = row.get("features")
    if not isinstance(features, Mapping):
        features = {}
    for name in names:
        value = row.get(name)
        if value is None:
            value = features.get(name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return 0.0


def _text_bucket(row: Mapping[str, Any]) -> str:
    text_features = row.get("text_features")
    if not isinstance(text_features, Mapping):
        text_features = {}
    char_count = _safe_float(text_features.get("char_count"))
    if char_count <= 0.0:
        return "empty"
    if char_count <= 2.0:
        return "short_text"
    repeat = text_features.get("repeat_profile")
    if isinstance(repeat, Mapping) and _safe_float(repeat.get("ratio")) >= 0.65:
        return "repeated"
    return "text_present"


def _normalized_entropy(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy / math.log(len(counts)))


def _homogeneity_review_action(score: float, count: int, outlier_ratio: float) -> str:
    if score >= 0.72 and outlier_ratio <= 0.22:
        return "broadcast_candidate"
    if count >= 80 and score < 0.62:
        return "drill_down"
    if score < 0.58:
        return "mixed_review"
    return "review"


def _cluster_homogeneity(
    rows: list[dict[str, Any]],
    labels: list[str],
    matrix: list[list[float]],
    distances: np.ndarray,
    *,
    feature_summary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not rows or not labels:
        return {}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[str(label)].append(index)
    out: dict[str, dict[str, Any]] = {}
    view_count = int(feature_summary.get("view_count") or len(feature_summary.get("views") or {}) or 1)
    for label, indexes in grouped.items():
        pair_distances: list[float] = []
        if len(indexes) > 1 and distances.size:
            sub = distances[np.ix_(indexes, indexes)]
            upper = sub[np.triu_indices_from(sub, k=1)]
            pair_distances = [float(value) for value in upper if math.isfinite(float(value))]
        centroid_distances = [
            _safe_float(rows[index].get("cluster_centroid_distance"))
            for index in indexes
            if math.isfinite(_safe_float(rows[index].get("cluster_centroid_distance")))
        ]
        p50 = _percentile(pair_distances, 50)
        p90 = _percentile(pair_distances, 90)
        centroid_p75 = _percentile(centroid_distances, 75)
        centroid_p25 = _percentile(centroid_distances, 25)
        iqr = max(0.0, centroid_p75 - centroid_p25)
        outlier_cutoff = centroid_p75 + 1.5 * iqr
        outlier_count = sum(1 for value in centroid_distances if value > outlier_cutoff and value > 0.0)
        outlier_ratio = outlier_count / max(1, len(indexes))
        durations = [_duration_value(rows[index]) for index in indexes]
        refiner_values = [
            _feature_value(rows[index], ("refiner_confidence_mean", "refiner_confidence", "refiner_start_confidence"))
            for index in indexes
        ]
        speech_values = [
            _feature_value(rows[index], ("scorer_speech_mean", "speech_prob_mean", "speech_mean"))
            for index in indexes
        ]
        text_entropy = _normalized_entropy([_text_bucket(rows[index]) for index in indexes])
        duration_mean = sum(durations) / max(1, len(durations))
        duration_cv = _std(durations) / max(0.001, duration_mean)
        refiner_std = _std(refiner_values)
        speech_std = _std(speech_values)
        distance_consensus = max(0.0, min(1.0, 1.0 - p90 / 1.5))
        compactness = max(0.0, min(1.0, 1.0 - p50 / 1.2))
        dispersion_score = max(0.0, min(1.0, 1.0 - min(1.0, duration_cv)))
        score = (
            0.34 * distance_consensus
            + 0.22 * compactness
            + 0.14 * (1.0 - min(1.0, text_entropy))
            + 0.12 * (1.0 - min(1.0, outlier_ratio))
            + 0.10 * (1.0 - min(1.0, refiner_std))
            + 0.08 * (1.0 - min(1.0, speech_std))
        )
        score = max(0.0, min(1.0, score))
        out[label] = {
            "schema": "cueqc_cluster_homogeneity_v1",
            "score": round(score, 4),
            "review_action": _homogeneity_review_action(score, len(indexes), outlier_ratio),
            "multi_view_distance_consensus": round(distance_consensus, 4),
            "view_count": view_count,
            "intra_distance_p50": round(p50, 6),
            "intra_distance_p90": round(p90, 6),
            "cluster_centroid_distance_p90": round(_percentile(centroid_distances, 90), 6),
            "outlier_ratio": round(outlier_ratio, 4),
            "text_morphology_entropy": round(text_entropy, 4),
            "duration_std_s": round(_std(durations), 4),
            "duration_cv": round(duration_cv, 4),
            "refiner_confidence_std": round(refiner_std, 4),
            "speech_mean_std": round(speech_std, 4),
        }
    return out


def _annotate_audit_sampling(
    rows: list[dict[str, Any]],
    matrix: list[list[float]],
    labels: list[str],
    *,
    per_cluster: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows or not matrix:
        return rows, {"enabled": False, "reason": "empty_rows_or_matrix"}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    annotated = [dict(row) for row in rows]
    cluster_plan: dict[str, Any] = {}
    for label, indexes in sorted(grouped.items(), key=lambda item: item[0]):
        centroid = [
            sum(matrix[index][col] for index in indexes) / max(1, len(indexes))
            for col in range(len(matrix[0]))
        ]
        centroid_distances = {
            index: _euclidean_distance(matrix[index], centroid)
            for index in indexes
        }
        medoids = sorted(indexes, key=lambda index: (centroid_distances[index], _duration_value(rows[index])))[:per_cluster]
        outliers = sorted(indexes, key=lambda index: (-centroid_distances[index], _duration_value(rows[index])))[:per_cluster]
        shortest = sorted(indexes, key=lambda index: (_duration_value(rows[index]), index))[:per_cluster]
        longest = sorted(indexes, key=lambda index: (-_duration_value(rows[index]), index))[:per_cluster]
        risk_scores = {index: _audit_risk_score(rows[index]) for index in indexes}
        high_risk = sorted(indexes, key=lambda index: (-risk_scores[index], -_duration_value(rows[index]), index))[:per_cluster]
        role_indexes = {
            "representative": medoids,
            "outlier": outliers,
            "shortest": shortest,
            "longest": longest,
            "high_risk": high_risk,
        }
        for rank, index in enumerate(sorted(indexes, key=lambda item: centroid_distances[item])):
            annotated[index]["cluster_centroid_distance"] = round(centroid_distances[index], 6)
            annotated[index]["cluster_outlier_rank"] = rank
            annotated[index]["cluster_outlier_score"] = round(centroid_distances[index], 6)
            annotated[index]["audit_risk_score"] = risk_scores[index]
            annotated[index]["audit_sampling_roles"] = []
            annotated[index]["audit_sampling_score"] = risk_scores[index]
        for role, role_members in role_indexes.items():
            for role_rank, index in enumerate(role_members):
                roles = list(annotated[index].get("audit_sampling_roles") or [])
                if role not in roles:
                    roles.append(role)
                annotated[index]["audit_sampling_roles"] = roles
                annotated[index][f"{role}_rank"] = role_rank
                role_bonus = {
                    "representative": 0.96,
                    "outlier": 0.92,
                    "longest": 0.86,
                    "shortest": 0.82,
                    "high_risk": 0.90,
                }[role]
                annotated[index]["audit_sampling_score"] = round(
                    max(_safe_float(annotated[index].get("audit_sampling_score")), role_bonus),
                    6,
                )
        cluster_plan[label] = {
            "representatives": [_sample_stub(annotated[index]) for index in medoids],
            "outliers": [_sample_stub(annotated[index]) for index in outliers],
            "shortest": [_sample_stub(annotated[index]) for index in shortest],
            "longest": [_sample_stub(annotated[index]) for index in longest],
            "high_risk": [_sample_stub(annotated[index]) for index in high_risk],
        }
    return annotated, {
        "enabled": True,
        "roles": ["representative", "outlier", "shortest", "longest", "high_risk"],
        "per_cluster": per_cluster,
        "clusters": cluster_plan,
    }


def _refine_large_clusters_with_ptm(
    rows: list[dict[str, Any]],
    coarse_labels: list[str],
    *,
    min_cluster_size: int,
    components: int,
    metric: str,
    merge_layer: int,
    torque_cut: bool,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(coarse_labels):
        grouped[label].append(index)

    raw_labels = list(coarse_labels)
    refined_rows = [dict(row) for row in rows]
    attempts: list[dict[str, Any]] = []
    for parent_label, indexes in sorted(grouped.items(), key=lambda item: item[0]):
        for index in indexes:
            refined_rows[index]["coarse_cluster_id"] = parent_label
            refined_rows[index]["ptm_refined"] = False
            refined_rows[index]["ptm_refine_parent_cluster_id"] = parent_label
            refined_rows[index]["ptm_refine_subcluster_id"] = ""

        if len(indexes) < min_cluster_size:
            continue
        sub_rows = [rows[index] for index in indexes]
        sub_matrix, ptm_meta = _ptm_compact_matrix(sub_rows, components=components)
        attempt: dict[str, Any] = {
            "parent_cluster_id": parent_label,
            "parent_size": len(indexes),
            "ptm": ptm_meta,
            "refined": False,
        }
        if not sub_matrix:
            attempt["skip_reason"] = "no_compact_ptm_features"
            attempts.append(attempt)
            continue

        sub_distances = _distance_matrix(sub_matrix, metric=metric)
        effective_merge_layer: int | None = None
        if torque_cut:
            sub_raw, sub_noise, sub_diag = torque_clustering(
                sub_distances,
                k=0,
                detect_noise=True,
                adjustment_factor=None,
                merge_layer=None,
            )
            sub_labels = _relabel_by_size(sub_noise)
            mode = "torque_gap_cut"
        else:
            layer_preview = torque_merge_layer_preview(
                sub_distances,
                max_layer=merge_layer,
            )
            available_layers = [int(item["layer"]) for item in layer_preview]
            effective_merge_layer = min(merge_layer, max(available_layers or [0]))
            sub_raw, _sub_noise, sub_diag = torque_clustering(
                sub_distances,
                k=0,
                detect_noise=False,
                adjustment_factor=None,
                merge_layer=effective_merge_layer,
            )
            sub_labels = _relabel_by_size(sub_raw)
            mode = "merge_layer"
        sub_count = len(set(sub_labels))
        attempt.update(
            {
                "refined": sub_count > 1,
                "subcluster_count": sub_count,
                "mode": mode,
                "merge_layer_requested": merge_layer,
                "merge_layer": effective_merge_layer,
                "metric": metric,
                "backend": {
                    "algorithm": "torque",
                    "mode": mode,
                    "merge_layer_requested": merge_layer,
                    "merge_layer": effective_merge_layer,
                    "layer_cluster_counts": sub_diag.get("layer_cluster_counts"),
                    "autonomous_k": sub_diag["cluster_count"],
                    "cut_count": sub_diag["cut_count"],
                    "noise_count": sub_diag["noise_count"],
                    "detect_noise": sub_diag["detect_noise"],
                    "fast_merge_layer": bool(sub_diag.get("fast_merge_layer", False)),
                },
            }
        )
        attempts.append(attempt)
        if sub_count <= 1:
            continue
        for local_index, global_index in enumerate(indexes):
            sub_label = sub_labels[local_index]
            raw_labels[global_index] = f"{parent_label}::{sub_label}"
            refined_rows[global_index]["ptm_refined"] = True
            refined_rows[global_index]["ptm_refine_subcluster_id"] = sub_label

    final_labels = _relabel_string_labels_by_size(raw_labels)
    coarse_to_final: dict[str, Counter[str]] = defaultdict(Counter)
    for coarse_label, final_label in zip(coarse_labels, final_labels):
        coarse_to_final[coarse_label][final_label] += 1
    for row, final_label in zip(refined_rows, final_labels):
        row["cluster_id"] = final_label
        row["ptm_refine_final_cluster_id"] = final_label

    refined_parent_count = sum(1 for attempt in attempts if attempt.get("refined"))
    summary = {
        "enabled": True,
        "min_cluster_size": min_cluster_size,
        "components_requested": components,
        "metric": metric,
        "merge_layer": merge_layer,
        "torque_cut": torque_cut,
        "attempted_parent_count": len(attempts),
        "refined_parent_count": refined_parent_count,
        "attempts": attempts,
        "coarse_to_final_clusters": {
            coarse_label: dict(counter.most_common())
            for coarse_label, counter in coarse_to_final.items()
        },
    }
    return final_labels, summary, refined_rows


def _representatives(
    rows: list[dict[str, Any]],
    matrix: list[list[float]],
    labels: list[str],
    *,
    per_cluster: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    representatives: list[dict[str, Any]] = []
    for label, indexes in sorted(grouped.items(), key=lambda item: item[0]):
        centroid = [
            sum(matrix[index][col] for index in indexes) / max(1, len(indexes))
            for col in range(len(matrix[0]))
        ]
        ranked = sorted(
            indexes,
            key=lambda index: (
                _euclidean_distance(matrix[index], centroid),
                float(rows[index].get("start") or 0.0),
            ),
        )
        for rank, index in enumerate(ranked[:per_cluster]):
            row = rows[index]
            representatives.append(
                {
                    "cluster_id": label,
                    "representative_rank": rank,
                    "sample_id": row.get("sample_id", ""),
                    "chunk_index": row.get("chunk_index"),
                    "start": row.get("start"),
                    "end": row.get("end"),
                    "duration_s": row.get("duration_s"),
                    "text": row.get("text", ""),
                    "raw_text": row.get("raw_text", ""),
                    "text_preview": row.get("text_preview", ""),
                    "qc": row.get("qc", {}),
                    "text_features": row.get("text_features", {}),
                    "cluster_confidence": row.get("cluster_confidence"),
                    "cluster_noise": row.get("cluster_noise"),
                    "audit_sampling_roles": row.get("audit_sampling_roles", []),
                    "audit_sampling_score": row.get("audit_sampling_score"),
                    "cluster_centroid_distance": row.get("cluster_centroid_distance"),
                }
            )
    return representatives


def _cluster_summaries(
    rows: list[dict[str, Any]],
    labels: list[str],
    *,
    homogeneity: Mapping[str, Mapping[str, Any]] | None = None,
    tail_merge_suggestions: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, label in zip(rows, labels):
        grouped[label].append(row)
    summaries: list[dict[str, Any]] = []
    for label, members in sorted(grouped.items(), key=lambda item: item[0]):
        severity_counts = Counter(
            str((member.get("qc") or {}).get("severity") or "ok")
            for member in members
            if isinstance(member.get("qc"), Mapping)
        )
        observation_counts = Counter(
            "empty_text" if int(((member.get("text_features") or {}).get("char_count") or 0)) <= 0 else "text_present"
            for member in members
            if isinstance(member.get("text_features"), Mapping)
        )
        char_counts = [
            int(((member.get("text_features") or {}).get("char_count") or 0))
            for member in members
            if isinstance(member.get("text_features"), Mapping)
        ]
        durations = sorted(_duration_value(member) for member in members)
        risks = [_safe_float(member.get("audit_risk_score")) for member in members]
        role_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for member in sorted(
            members,
            key=lambda item: (
                -_safe_float(item.get("audit_sampling_score")),
                _safe_float(item.get("cluster_centroid_distance")),
                _safe_float(item.get("start")),
            ),
        ):
            for role in member.get("audit_sampling_roles") or []:
                if len(role_members[str(role)]) < 5:
                    role_members[str(role)].append(_sample_stub(member))
        empty_text = observation_counts.get("empty_text", 0)
        ptm_refined_count = sum(1 for member in members if bool(member.get("ptm_refined")))
        tail_suggestion = dict((tail_merge_suggestions or {}).get(label) or {})
        tail_merge_suggestion_count = 1 if tail_suggestion else 0
        homo = dict((homogeneity or {}).get(label) or {})
        homogeneity_signals = {
            "empty_text_ratio": round(empty_text / max(1, len(members)), 4),
            "ptm_refined_ratio": round(ptm_refined_count / max(1, len(members)), 4),
            "tail_merged_ratio": 0.0,
            "tail_merge_suggestion_ratio": round(tail_merge_suggestion_count / max(1, len(members)), 4),
            "dominant_qc_severity": severity_counts.most_common(1)[0][0] if severity_counts else "ok",
        }
        homogeneity_signals.update(homo)
        summaries.append(
            {
                "cluster_id": label,
                "count": len(members),
                "noise_count": sum(1 for member in members if bool(member.get("cluster_noise"))),
                "confidence_avg": round(
                    sum(float(member.get("cluster_confidence") or 0.0) for member in members)
                    / max(1, len(members)),
                    4,
                ),
                "qc_severity_counts": dict(severity_counts.most_common()),
                "text_observation_counts": dict(observation_counts.most_common()),
                "char_count_avg": round(sum(char_counts) / max(1, len(char_counts)), 3),
                "duration_min_s": round(durations[0], 3) if durations else 0.0,
                "duration_max_s": round(durations[-1], 3) if durations else 0.0,
                "duration_median_s": round(durations[len(durations) // 2], 3) if durations else 0.0,
                "audit_risk_avg": round(sum(risks) / max(1, len(risks)), 4),
                "audit_risk_max": round(max(risks) if risks else 0.0, 4),
                "ptm_refined_count": ptm_refined_count,
                "tail_merged_count": 0,
                "tail_merge_suggestion_count": tail_merge_suggestion_count,
                "tail_merge_suggestions": [tail_suggestion] if tail_suggestion else [],
                "homogeneity_signals": homogeneity_signals,
                "review_action": str(homo.get("review_action") or "review"),
                "review_sample_plan": {
                    role: samples
                    for role, samples in sorted(role_members.items())
                },
                "examples": [
                    {
                        "sample_id": member.get("sample_id", ""),
                        "start": member.get("start"),
                        "duration_s": member.get("duration_s"),
                        "text_preview": member.get("text_preview", ""),
                        "audit_sampling_roles": member.get("audit_sampling_roles", []),
                        "audit_sampling_score": member.get("audit_sampling_score"),
                    }
                    for member in sorted(
                        members,
                        key=lambda item: (
                            -_safe_float(item.get("audit_sampling_score")),
                            _safe_float(item.get("start")),
                        ),
                    )[:5]
                ],
            }
        )
    return summaries


def _audit_html(*, rows: list[dict[str, Any]], summaries: list[dict[str, Any]], title: str) -> str:
    rows_json = json.dumps(rows, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    summaries_json = json.dumps(summaries, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; background: #f6f7f5; color: #1d2421; font: 14px/1.45 system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }}
.app {{ display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: 100vh; }}
.side {{ border-right: 1px solid #d8ddd8; background: #fff; max-height: 100vh; overflow: auto; }}
.main {{ padding: 16px; max-height: 100vh; overflow: auto; }}
.head {{ position: sticky; top: 0; background: #fff; padding: 12px; border-bottom: 1px solid #d8ddd8; }}
h1 {{ margin: 0 0 8px; font-size: 16px; }}
input, select, textarea, button {{ font: inherit; }}
input, select {{ width: 100%; padding: 7px; border: 1px solid #d8ddd8; border-radius: 6px; }}
button {{ border: 1px solid #d8ddd8; border-radius: 6px; padding: 7px 10px; background: #fff; cursor: pointer; }}
button.active {{ background: #dff2ee; border-color: #0f766e; }}
.filters {{ display: grid; grid-template-columns: 1fr 140px; gap: 8px; }}
.item {{ padding: 10px 12px; border-bottom: 1px solid #d8ddd8; cursor: pointer; }}
.item.active, .item:hover {{ background: #eaf4f1; }}
.meta {{ color: #66706c; font-size: 12px; }}
.panel {{ background: #fff; border: 1px solid #d8ddd8; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
.kv {{ display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 6px 10px; }}
.text {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 18px; }}
.labels {{ display: flex; flex-wrap: wrap; gap: 8px; }}
textarea {{ width: 100%; min-height: 80px; border: 1px solid #d8ddd8; border-radius: 6px; padding: 8px; }}
code {{ background: #eef3ef; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div class="head">
      <h1>{html.escape(title)}</h1>
      <div class="filters">
        <input id="search" placeholder="搜索文本 / reason / sample_id">
        <select id="cluster"></select>
      </div>
      <p class="meta" id="summary"></p>
      <button id="download">下载 keep/drop 标签 JSONL</button>
    </div>
    <div id="list"></div>
  </aside>
  <main class="main">
    <section class="panel">
      <h2 id="sampleTitle"></h2>
      <p class="meta" id="sampleMeta"></p>
      <div class="text" id="text"></div>
    </section>
    <section class="panel">
      <h3>候选指标</h3>
      <div class="kv" id="metrics"></div>
    </section>
    <section class="panel">
      <h3>初始训练标签</h3>
      <p class="meta">聚类只用于第一次粗标签；每条样本只标 keep/drop，混簇噪声交给后续全量训练修正。</p>
      <div class="labels" id="displayButtons"></div>
      <label class="meta" for="notes">notes</label>
      <textarea id="notes"></textarea>
    </section>
  </main>
</div>
<script type="application/json" id="rows-json">{rows_json}</script>
<script type="application/json" id="summaries-json">{summaries_json}</script>
<script>
const ROWS = JSON.parse(document.getElementById("rows-json").textContent);
const SUMMARIES = JSON.parse(document.getElementById("summaries-json").textContent);
const STORAGE_KEY = "cueqc-cluster-audit:" + location.pathname;
const DISPLAY = [["keep","keep"],["drop","drop"]];
let annotations = loadAnnotations();
let filtered = [...ROWS];
let current = 0;
function loadAnnotations() {{ try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }} catch (_) {{ return {{}}; }} }}
function saveAnnotations() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations)); }}
function escapeHtml(text) {{ return String(text || "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch])); }}
function rowText(row) {{ return [row.sample_id,row.cluster_id,row.text,row.raw_text,JSON.stringify(row.qc||{{}}),JSON.stringify(row.text_features||{{}})].join("\\n").toLowerCase(); }}
function setupClusters() {{
  const select = document.getElementById("cluster");
  select.innerHTML = '<option value="">all clusters</option>';
  for (const summary of SUMMARIES) {{
    const option = document.createElement("option");
    option.value = summary.cluster_id;
    option.textContent = `${{summary.cluster_id}} (${{summary.count}})`;
    select.appendChild(option);
  }}
}}
function applyFilters() {{
  const q = document.getElementById("search").value.trim().toLowerCase();
  const cluster = document.getElementById("cluster").value;
  filtered = ROWS.filter(row => (!cluster || row.cluster_id === cluster) && (!q || rowText(row).includes(q)));
  if (!filtered.includes(ROWS[current])) current = Math.max(0, ROWS.indexOf(filtered[0] || ROWS[0]));
  renderList();
  renderCurrent();
}}
function renderList() {{
  const root = document.getElementById("list");
  root.innerHTML = "";
  for (const row of filtered) {{
    const div = document.createElement("div");
    div.className = "item" + (ROWS[current] === row ? " active" : "");
    div.onclick = () => {{ current = ROWS.indexOf(row); renderList(); renderCurrent(); }};
    const qc = row.qc || {{}};
    div.innerHTML = `<strong>${{escapeHtml(row.cluster_id)}} · chunk ${{row.chunk_index}}</strong>
      <div class="meta">${{Number(row.start||0).toFixed(2)}}s · ${{Number(row.duration_s||0).toFixed(2)}}s · QC=${{escapeHtml(qc.severity||"ok")}}</div>
      <div class="meta">${{escapeHtml(row.text_preview || row.text || "(empty)")}}</div>`;
    root.appendChild(div);
  }}
  document.getElementById("summary").textContent = `${{filtered.length}} / ${{ROWS.length}} samples · ${{SUMMARIES.length}} clusters`;
}}
function setButtons(rootId, options, key) {{
  const root = document.getElementById(rootId);
  root.innerHTML = "";
  const row = ROWS[current];
  const ann = annotations[row.sample_id] || {{}};
  for (const [value, label] of options) {{
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.className = ann[key] === value ? "active" : "";
    btn.onclick = () => {{
      annotations[row.sample_id] = {{...(annotations[row.sample_id] || {{}}), [key]: value, updated_at: new Date().toISOString()}};
      saveAnnotations();
      renderCurrent();
    }};
    root.appendChild(btn);
  }}
}}
function renderCurrent() {{
  const row = ROWS[current];
  if (!row) return;
  const tf = row.text_features || {{}};
  const cueFeatures = row.cue_features || {{}};
  const qc = row.qc || {{}};
  document.getElementById("sampleTitle").textContent = `${{row.cluster_id}} · ${{row.sample_id}}`;
  document.getElementById("sampleMeta").textContent = `chunk ${{row.chunk_index}} · ${{Number(row.start||0).toFixed(3)}}-${{Number(row.end||0).toFixed(3)}} · duration ${{Number(row.duration_s||0).toFixed(3)}}s`;
  document.getElementById("text").textContent = row.text || row.raw_text || "(empty)";
  const metrics = [
    ["cluster", row.cluster_id],
    ["qc", `${{qc.severity || "ok"}} · ${{(qc.reasons || []).join(", ")}}`],
    ["chars", `${{tf.char_count || 0}} unique=${{tf.unique_chars || 0}} kana=${{tf.kana_ratio || 0}} kanji=${{tf.kanji_ratio || 0}}`],
    ["stable_vocab", String(!!tf.has_stable_vocabulary)],
    ["repeat", JSON.stringify(tf.repeat_profile || {{}})],
    ["text_obs", JSON.stringify((cueFeatures.text_observation || row.text_observation || {{}}))],
    ["adjacency", JSON.stringify(row.adjacency || {{}})],
    ["audio", (row.audio || {{}}).path || ""]
  ];
  document.getElementById("metrics").innerHTML = metrics.map(([k,v]) => `<div>${{escapeHtml(k)}}</div><div>${{escapeHtml(v)}}</div>`).join("");
  setButtons("displayButtons", DISPLAY, "display_decision");
  document.getElementById("notes").value = (annotations[row.sample_id] || {{}}).notes || "";
}}
function exportRows() {{
  return ROWS.map(row => ({{
    sample_id: row.sample_id,
    cluster_id: row.cluster_id,
    chunk_index: row.chunk_index,
    start: row.start,
    end: row.end,
    duration_s: row.duration_s,
    text: row.text,
    raw_text: row.raw_text,
    display_decision: (annotations[row.sample_id] || {{}}).display_decision || "",
    ...(annotations[row.sample_id] || {{}})
  }}));
}}
document.getElementById("search").addEventListener("input", applyFilters);
document.getElementById("cluster").addEventListener("change", applyFilters);
document.getElementById("notes").addEventListener("input", () => {{
  const row = ROWS[current];
  annotations[row.sample_id] = {{...(annotations[row.sample_id] || {{}}), notes: document.getElementById("notes").value, updated_at: new Date().toISOString()}};
  saveAnnotations();
}});
document.getElementById("download").onclick = () => {{
  const blob = new Blob([exportRows().map(row => JSON.stringify(row)).join("\\n") + "\\n"], {{type:"application/jsonl;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "cueqc_cluster_labels.jsonl";
  a.click();
  URL.revokeObjectURL(a.href);
}};
setupClusters();
applyFilters();
</script>
</body>
</html>
"""


def _score_torc_layer_preview(layers: list[dict[str, Any]], *, total_count: int) -> dict[str, Any]:
    if not layers:
        return {
            "enabled": False,
            "reason": "no_layers",
            "selected_layer": None,
            "layers": [],
        }
    target = max(4, min(160, int(round(math.sqrt(max(1, total_count)) * 1.1))))
    scored: list[dict[str, Any]] = []
    for item in layers:
        cluster_count = max(1, int(item.get("cluster_count") or 1))
        max_size = max(0, int(item.get("cluster_size_max") or 0))
        min_size = max(0, int(item.get("cluster_size_min") or 0))
        max_ratio = max_size / max(1, total_count)
        count_score = math.exp(-abs(math.log(cluster_count / max(1, target))))
        largest_score = max(0.0, min(1.0, 1.0 - max(0.0, max_ratio - 0.34) / 0.66))
        tail_score = 1.0 if min_size >= 3 else 0.72
        score = 0.56 * count_score + 0.34 * largest_score + 0.10 * tail_score
        decision = "candidate"
        if max_ratio > 0.72:
            decision = "too_coarse"
        elif cluster_count > max(220, target * 3):
            decision = "too_fine"
        scored.append({
            **item,
            "score": round(max(0.0, min(1.0, score)), 4),
            "target_cluster_count": target,
            "max_cluster_ratio": round(max_ratio, 4),
            "tail_proxy": "ok" if min_size >= 3 else "has_small_tail",
            "audit_decision": decision,
        })
    selectable = [item for item in scored if item["audit_decision"] == "candidate"] or scored
    selected = max(selectable, key=lambda item: (float(item["score"]), -int(item["layer"])))
    for item in scored:
        item["selected"] = int(item["layer"]) == int(selected["layer"])
    return {
        "enabled": True,
        "schema": "cueqc_torc_layer_auto_selection_v1",
        "policy": "prefer_auditable_cluster_count_low_largest_cluster_ratio_and_small_tail_penalty",
        "target_cluster_count": target,
        "selected_layer": int(selected["layer"]),
        "selected_score": float(selected["score"]),
        "layers": scored,
    }


def cluster_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str = "euclidean",
    feature_space: str = "auto",
    representatives_per_cluster: int = 3,
    detect_noise: bool = True,
    adjustment_factor: float | None = None,
    merge_layer: int | None = None,
    ptm_refine_large_clusters: bool = False,
    ptm_refine_min_cluster_size: int = 500,
    ptm_refine_components: int = 16,
    ptm_refine_metric: str = "euclidean",
    ptm_refine_merge_layer: int = 1,
    ptm_refine_torque_cut: bool = False,
    coldstart_graph_k: int = 12,
    tail_merge_min_cluster_size: int | None = None,
    auto_select_layer: bool | None = None,
    auto_layer_max: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Cluster candidates with Torque Clustering.

    The cluster count is decided by TORC itself. By default the torque-gap cut
    on the final merge tree is used (factor-auto-tuned); pass ``merge_layer`` to
    instead take a partition from the TORC merge hierarchy (layer 0 = initial
    1-NN layer, layer 1 = after one merge pass, ...) which is far more stable on
    heavy-tailed distance distributions. Returns
    ``(clustered_rows, representatives, summaries, summary)``.
    """
    if not rows:
        return [], [], [], {
            "schema": "cueqc_cluster_summary_v1",
            "method": "torque",
            "cluster_count": 0,
            "candidate_count": 0,
            "feature_space": {"requested": feature_space, "resolved": feature_space},
        }
    precomputed_distances: np.ndarray | None = None
    if feature_space == "pre_asr_coldstart":
        matrix, precomputed_distances, feature_summary = _pre_asr_coldstart_inputs(
            rows,
            metric=metric,
            ptm_components=ptm_refine_components,
            graph_k=coldstart_graph_k,
        )
    else:
        matrix, feature_summary = _feature_matrix(rows, feature_space=feature_space)
    if not matrix:
        return [], [], [], {
            "schema": "cueqc_cluster_summary_v1",
            "method": "torque",
            "cluster_count": 0,
            "candidate_count": 0,
            "feature_space": feature_summary,
        }

    distance_matrix = precomputed_distances if precomputed_distances is not None else _distance_matrix(matrix, metric=metric)
    layer_selection: dict[str, Any] = {"enabled": False, "reason": "not_requested"}
    effective_merge_layer = merge_layer
    use_auto_layer = (
        (feature_space == "pre_asr_coldstart" if auto_select_layer is None else bool(auto_select_layer))
        and merge_layer is None
    )
    if use_auto_layer:
        raw_preview = torque_merge_layer_preview(distance_matrix, max_layer=auto_layer_max)
        layer_selection = _score_torc_layer_preview(raw_preview, total_count=len(rows))
        selected_layer = layer_selection.get("selected_layer")
        if selected_layer is not None:
            effective_merge_layer = int(selected_layer)
    labels_raw, labels_noise, diag = torque_clustering(
        distance_matrix, k=0, detect_noise=detect_noise,
        adjustment_factor=adjustment_factor, merge_layer=effective_merge_layer,
    )
    # merge_layer partitions carry no noise flag; only the final-tree cut does.
    labels = _relabel_by_size(labels_noise if (detect_noise and effective_merge_layer is None) else labels_raw)

    clustered_rows: list[dict[str, Any]] = []
    for row, label in zip(rows, labels):
        item = dict(row)
        item["cluster_id"] = label
        item["coarse_cluster_id"] = label
        item["cluster_method"] = "torque"
        item["cluster_backend"] = "torque"
        item["cluster_noise"] = label == "noise_00"
        item["cluster_confidence"] = 1.0
        item["ptm_refined"] = False
        item["ptm_refine_parent_cluster_id"] = label
        item["ptm_refine_subcluster_id"] = ""
        item["ptm_refine_final_cluster_id"] = label
        clustered_rows.append(item)

    ptm_refine_summary: dict[str, Any] = {
        "enabled": False,
        "reason": "not_requested",
    }
    if ptm_refine_large_clusters:
        labels, ptm_refine_summary, clustered_rows = _refine_large_clusters_with_ptm(
            clustered_rows,
            labels,
            min_cluster_size=ptm_refine_min_cluster_size,
            components=ptm_refine_components,
            metric=ptm_refine_metric,
            merge_layer=ptm_refine_merge_layer,
            torque_cut=ptm_refine_torque_cut,
        )
        for item, label in zip(clustered_rows, labels):
            item["cluster_method"] = "torque+ptm_compact_refine"
            item["cluster_backend"] = "torque"
            item["cluster_noise"] = label == "noise_00"
            item["cluster_confidence"] = 1.0

    effective_tail_merge_min = (
        3
        if tail_merge_min_cluster_size is None and feature_space == "pre_asr_coldstart"
        else int(tail_merge_min_cluster_size or 0)
    )
    tail_merge_summary: dict[str, Any] = {"enabled": False, "reason": "disabled", "suggestions": []}
    tail_suggestions_by_cluster: dict[str, dict[str, Any]] = {}
    if effective_tail_merge_min > 1:
        tail_merge_summary = _suggest_small_tail_merges(
            labels,
            distance_matrix,
            min_cluster_size=effective_tail_merge_min,
        )
        for suggestion in tail_merge_summary.get("suggestions") or []:
            if not isinstance(suggestion, Mapping):
                continue
            source_label = str(suggestion.get("from_cluster_id") or "")
            if source_label:
                tail_suggestions_by_cluster[source_label] = dict(suggestion)
        for item, label in zip(clustered_rows, labels):
            suggestion = tail_suggestions_by_cluster.get(label)
            item["tail_merged_from_cluster_id"] = ""
            item["tail_merge_suggested_target_cluster_id"] = (
                str(suggestion.get("target_cluster_id") or "") if suggestion else ""
            )
            item["tail_merge_suggestion_distance"] = (
                suggestion.get("mean_distance") if suggestion else None
            )
            item["tail_merge_requires_confirmation"] = bool(suggestion)

    clustered_rows, audit_sampling_summary = _annotate_audit_sampling(
        clustered_rows,
        matrix,
        labels,
        per_cluster=representatives_per_cluster,
    )
    homogeneity = _cluster_homogeneity(
        clustered_rows,
        labels,
        matrix,
        distance_matrix,
        feature_summary=feature_summary,
    )
    summaries = _cluster_summaries(
        clustered_rows,
        labels,
        homogeneity=homogeneity,
        tail_merge_suggestions=tail_suggestions_by_cluster,
    )
    representatives = _representatives(
        clustered_rows, matrix, labels, per_cluster=representatives_per_cluster
    )
    stable_count = len({label for label in labels if not label.startswith("noise_")})
    backend = {
        "algorithm": "torque",
        "mode": "merge_layer" if effective_merge_layer is not None else "torque_gap_cut",
        "merge_layer": effective_merge_layer,
        "merge_layer_requested": merge_layer,
        "layer_cluster_counts": diag.get("layer_cluster_counts"),
        "k_requested": diag["k_requested"],
        "autonomous_k": diag["cluster_count"],
        "cut_count": diag["cut_count"],
        "noise_count": diag["noise_count"],
        "detect_noise": diag["detect_noise"],
        "fast_merge_layer": bool(diag.get("fast_merge_layer", False)),
        "torque_max": round(float(max(diag["torque"])) if diag["torque"].size else 0.0, 6),
        "mass_sum": round(float(diag["mass"].sum()) if diag["mass"].size else 0.0, 6),
    }
    if "adjustment_factor" in diag:
        backend["adjustment_factor"] = round(float(diag["adjustment_factor"]), 4)
    summary = {
        "schema": "cueqc_cluster_summary_v1",
        "method": "torque",
        "metric": metric,
        "candidate_count": len(rows),
        "cluster_count": stable_count,
        "total_groups_including_noise": len(summaries),
        "cluster_counts": {s["cluster_id"]: s["count"] for s in summaries},
        "representatives_per_cluster": representatives_per_cluster,
        "feature_space": feature_summary,
        "backend": backend,
        "ptm_refine": ptm_refine_summary,
        "tail_merge": tail_merge_summary,
        "layer_selection": layer_selection,
        "audit_sampling": audit_sampling_summary,
        "coldstart_scope": (
            "pre_asr_cueqc_coldstart_labeling_only"
            if feature_space == "pre_asr_coldstart"
            else ""
        ),
    }
    return clustered_rows, representatives, summaries, summary


def preview_layers(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    feature_space: str,
    max_layer: int,
    coldstart_graph_k: int = 12,
    ptm_components: int = 16,
) -> dict[str, Any]:
    if feature_space == "pre_asr_coldstart":
        _matrix, distance_matrix, feature_summary = _pre_asr_coldstart_inputs(
            rows,
            metric=metric,
            ptm_components=ptm_components,
            graph_k=coldstart_graph_k,
        )
    else:
        matrix, feature_summary = _feature_matrix(rows, feature_space=feature_space)
        distance_matrix = _distance_matrix(matrix, metric=metric)
    layers = torque_merge_layer_preview(distance_matrix, max_layer=max_layer)
    layer_selection = _score_torc_layer_preview(layers, total_count=len(rows))
    return {
        "schema": "cueqc_torc_layer_preview_v1",
        "method": "torque",
        "metric": metric,
        "candidate_count": len(rows),
        "feature_space": feature_summary,
        "max_layer_requested": max_layer,
        "layers": layers,
        "layer_selection": layer_selection,
        "recommended_layer": layer_selection.get("selected_layer"),
    }


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    if args.max_items is not None:
        rows = rows[: args.max_items]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_preview_path = output_dir / "torc_layer_preview.json"
    if args.layer_preview_max is not None:
        preview = preview_layers(
            rows,
            metric=args.metric,
            feature_space=args.feature_space,
            max_layer=args.layer_preview_max,
            coldstart_graph_k=args.coldstart_graph_k,
            ptm_components=args.ptm_refine_components,
        )
        layer_preview_path.write_text(
            json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"layer_preview={layer_preview_path}")
        if args.layer_preview_only:
            print(
                "layers="
                + ", ".join(
                    f"{item['layer']}:{item['cluster_count']}"
                    for item in preview["layers"]
                )
            )
            return 0
    clustered, representatives, summaries, summary = cluster_rows(
        rows,
        metric=args.metric,
        feature_space=args.feature_space,
        representatives_per_cluster=args.representatives_per_cluster,
        detect_noise=not args.no_noise,
        adjustment_factor=args.adjustment_factor,
        merge_layer=args.merge_layer,
        ptm_refine_large_clusters=args.ptm_refine_large_clusters,
        ptm_refine_min_cluster_size=args.ptm_refine_min_cluster_size,
        ptm_refine_components=args.ptm_refine_components,
        ptm_refine_metric=args.ptm_refine_metric,
        ptm_refine_merge_layer=args.ptm_refine_merge_layer,
        ptm_refine_torque_cut=args.ptm_refine_torque_cut,
        coldstart_graph_k=args.coldstart_graph_k,
        tail_merge_min_cluster_size=args.tail_merge_min_cluster_size,
        auto_select_layer=True if args.auto_select_layer else None,
        auto_layer_max=args.auto_layer_max,
    )
    clusters_path = output_dir / "cueqc_clusters.jsonl"
    reps_path = output_dir / "cueqc_cluster_representatives.jsonl"
    summaries_path = output_dir / "cueqc_cluster_summaries.jsonl"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / "cluster_audit.html"
    write_jsonl(clusters_path, clustered)
    write_jsonl(reps_path, representatives)
    write_jsonl(summaries_path, summaries)
    summary.update(
        {
            "input": str(Path(args.input)),
            "clusters": str(clusters_path),
            "representatives": str(reps_path),
            "cluster_summaries": str(summaries_path),
            "html": str(html_path),
        }
    )
    if layer_preview_path.exists():
        summary["layer_preview"] = str(layer_preview_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_path.write_text(
        _audit_html(rows=clustered, summaries=summaries, title=args.title),
        encoding="utf-8",
    )
    print(f"clusters={clusters_path}")
    print(f"representatives={reps_path}")
    print(f"html={html_path}")
    print(f"cluster_count={summary['cluster_count']}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster CueQC candidates with parameter-free Torque Clustering."
    )
    parser.add_argument("--input", required=True, help="cueqc_candidates.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="euclidean")
    parser.add_argument(
        "--feature-space",
        choices=("auto", "pre_asr", "pre_asr_audit", "pre_asr_coldstart", "structured", "dense", "fused"),
        default="auto",
        help=(
            "auto uses Pre-ASR numeric features when present, then dense embeddings, "
            "then structured text features. Use pre_asr_audit for human review "
            "clusters: it removes absolute timeline coordinates and PTM bins, then "
            "adds audit-only text morphology. Use pre_asr_coldstart only for "
            "first-pass Pre-ASR CueQC labeling: it fuses structure, text morphology, "
            "and compact PTM views before TORC and is not a runtime feature."
        ),
    )
    parser.add_argument("--representatives-per-cluster", type=int, default=3)
    parser.add_argument("--no-noise", action="store_true", help="disable TORC noise detection")
    parser.add_argument(
        "--adjustment-factor",
        type=float,
        default=None,
        help="override TORC cut threshold in [0,1]; default auto-tunes from the distance matrix.",
    )
    parser.add_argument(
        "--merge-layer",
        type=int,
        default=None,
        help=(
            "take a partition from the TORC merge hierarchy instead of the "
            "torque-gap cut. layer 0 = initial 1-NN layer, layer 1 = one merge "
            "pass, etc. Far more stable than the factor-sensitive cut on "
            "heavy-tailed distance distributions."
        ),
    )
    parser.add_argument(
        "--ptm-refine-large-clusters",
        action="store_true",
        help=(
            "after the coarse feature-space clustering, split only large coarse "
            "clusters again with compact PTM SVD features. This is intended for "
            "human audit pages where structure-only TORC leaves a heavy-tailed "
            "mixed cluster."
        ),
    )
    parser.add_argument(
        "--ptm-refine-min-cluster-size",
        type=int,
        default=500,
        help="minimum coarse cluster size eligible for compact PTM second-stage splitting",
    )
    parser.add_argument(
        "--ptm-refine-components",
        type=int,
        default=16,
        help="SVD component count for compact PTM second-stage features",
    )
    parser.add_argument(
        "--ptm-refine-metric",
        choices=("cosine", "euclidean"),
        default="euclidean",
        help="distance metric for compact PTM second-stage splitting",
    )
    parser.add_argument(
        "--ptm-refine-merge-layer",
        type=int,
        default=1,
        help="TORC merge layer used inside each large coarse cluster for PTM refinement",
    )
    parser.add_argument(
        "--ptm-refine-torque-cut",
        action="store_true",
        help=(
            "inside each large coarse cluster, use TORC's torque-gap cut with "
            "noise/remainder instead of a fixed merge layer. This is slower than "
            "merge-layer mode but usually gives a middle granularity between "
            "layer 0 over-splitting and layer 1 heavy tails."
        ),
    )
    parser.add_argument(
        "--coldstart-graph-k",
        type=int,
        default=12,
        help="kNN size per view for pre_asr_coldstart multi-view graph fusion",
    )
    parser.add_argument(
        "--tail-merge-min-cluster-size",
        type=int,
        default=None,
        help=(
            "write merge suggestions for clusters smaller than this size. This never "
            "changes cluster_id or training labels without human confirmation. "
            "Default is 3 for pre_asr_coldstart and disabled for other feature spaces."
        ),
    )
    parser.add_argument(
        "--auto-select-layer",
        action="store_true",
        help=(
            "choose a TORC merge layer from layer preview scoring. pre_asr_coldstart "
            "uses this by default when --merge-layer is not set."
        ),
    )
    parser.add_argument(
        "--auto-layer-max",
        type=int,
        default=4,
        help="maximum TORC merge layer considered by automatic layer selection",
    )
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--title", default="CueQC cluster-first 审计")
    parser.add_argument(
        "--layer-preview-max",
        type=int,
        default=None,
        help="write torc_layer_preview.json with merge-layer cluster counts up to this layer",
    )
    parser.add_argument(
        "--layer-preview-only",
        action="store_true",
        help="only write torc_layer_preview.json; do not generate clustered candidates",
    )
    args = parser.parse_args(argv)
    if args.representatives_per_cluster <= 0:
        parser.error("--representatives-per-cluster must be positive")
    if args.adjustment_factor is not None and not 0.0 <= args.adjustment_factor <= 1.0:
        parser.error("--adjustment-factor must be in [0, 1]")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    if args.layer_preview_max is not None and args.layer_preview_max < 0:
        parser.error("--layer-preview-max must be non-negative")
    if args.layer_preview_only and args.layer_preview_max is None:
        parser.error("--layer-preview-only requires --layer-preview-max")
    if args.ptm_refine_min_cluster_size <= 1:
        parser.error("--ptm-refine-min-cluster-size must be greater than 1")
    if args.ptm_refine_components <= 0:
        parser.error("--ptm-refine-components must be positive")
    if args.ptm_refine_merge_layer < 0:
        parser.error("--ptm-refine-merge-layer must be non-negative")
    if args.coldstart_graph_k <= 0:
        parser.error("--coldstart-graph-k must be positive")
    if args.tail_merge_min_cluster_size is not None and args.tail_merge_min_cluster_size < 0:
        parser.error("--tail-merge-min-cluster-size must be non-negative")
    if args.auto_layer_max < 0:
        parser.error("--auto-layer-max must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
