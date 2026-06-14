#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.cueqc import (
    DEFAULT_CLUSTER_COUNT_MAX,
    DEFAULT_CLUSTER_COUNT_MIN,
    normalize_feature_matrix,
)


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


def _cosine_distance(left: list[float], right: list[float]) -> float:
    return 1.0 - (_dot(left, right) / (_norm(left) * _norm(right)))


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


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


def _stable_cluster_count(labels: Sequence[int]) -> int:
    return len({int(label) for label in labels if int(label) >= 0})


def _noise_count(labels: Sequence[int]) -> int:
    return sum(1 for label in labels if int(label) < 0)


def _mean_probability(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    values = [
        float(probability)
        for label, probability in zip(labels, probabilities)
        if int(label) >= 0
    ]
    return sum(values) / max(1, len(values))


def _hdbscan_min_cluster_sizes(
    *,
    row_count: int,
    min_clusters: int,
    max_clusters: int,
    explicit: int | None,
) -> list[int]:
    if explicit is not None and explicit > 0:
        return [max(2, min(explicit, max(2, row_count)))]
    if row_count <= 2:
        return [2]
    upper = min(50, max(2, row_count // max(1, min_clusters)), max(2, row_count // 2))
    seeds = {
        2,
        3,
        4,
        5,
        max(2, row_count // max(1, max_clusters)),
        max(2, row_count // max(1, (min_clusters + max_clusters) // 2)),
        max(2, row_count // max(1, min_clusters)),
    }
    sizes = {value for value in seeds if 2 <= value <= upper}
    if upper <= 30:
        sizes.update(range(2, upper + 1))
    else:
        step = max(1, upper // 16)
        sizes.update(range(2, upper + 1, step))
        sizes.add(upper)
    return sorted(sizes)


def _pca_reduce_matrix(
    matrix: list[list[float]],
    *,
    components: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    if components <= 0 or len(matrix) < 3 or not matrix or len(matrix[0]) < 3:
        return matrix, {"method": "none", "reason": "pca_not_applicable"}
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:  # pragma: no cover - exercised only without sklearn.
        return matrix, {"method": "none", "reason": f"pca_unavailable:{exc!r}"}
    width = len(matrix[0])
    n_components = min(max(2, components), width, len(matrix) - 1)
    if n_components < 2:
        return matrix, {"method": "none", "reason": "too_few_components"}
    pca = PCA(n_components=n_components, whiten=True, copy=True)
    reduced = pca.fit_transform(matrix)
    explained = getattr(pca, "explained_variance_ratio_", [])
    return reduced.tolist(), {
        "method": "pca_whiten",
        "components": n_components,
        "input_dim": width,
        "explained_variance_ratio_sum": round(float(sum(explained)), 6),
    }


def _hdbscan_score(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    min_clusters: int,
    max_clusters: int,
) -> tuple[float, float, float, float]:
    stable_count = _stable_cluster_count(labels)
    target = (min_clusters + max_clusters) / 2.0
    if min_clusters <= stable_count <= max_clusters:
        cluster_penalty = abs(stable_count - target) / max(1.0, target)
        range_penalty = 0.0
    elif stable_count < min_clusters:
        cluster_penalty = (min_clusters - stable_count) / max(1.0, min_clusters)
        range_penalty = 1.0
    else:
        cluster_penalty = (stable_count - max_clusters) / max(1.0, max_clusters)
        range_penalty = 1.0
    noise_ratio = _noise_count(labels) / max(1, len(labels))
    confidence_penalty = 1.0 - _mean_probability(labels, probabilities)
    return (range_penalty, cluster_penalty, noise_ratio, confidence_penalty)


def _relabel_hdbscan_by_size(labels: Sequence[int]) -> list[str]:
    stable_counts = Counter(int(label) for label in labels if int(label) >= 0)
    stable_order = {
        label: rank
        for rank, (label, _count) in enumerate(
            sorted(stable_counts.items(), key=lambda item: (-item[1], item[0]))
        )
    }
    out: list[str] = []
    for label in labels:
        raw = int(label)
        if raw < 0:
            out.append("noise_00")
        else:
            out.append(f"cluster_{stable_order[raw]:02d}")
    return out


def _sklearn_hdbscan_cluster(
    matrix: list[list[float]],
    *,
    metric: str,
    reducer: str,
    min_clusters: int,
    max_clusters: int,
    min_cluster_size: int | None,
    min_samples: int | None,
    selection_method: str,
    pca_components: int,
    umap_components: int,
    umap_neighbors: int,
    umap_min_dist: float,
    random_state: int,
) -> tuple[list[int], list[float], dict[str, Any]]:
    try:
        from sklearn.cluster import HDBSCAN
    except Exception as exc:
        raise RuntimeError(
            "sklearn HDBSCAN backend requires scikit-learn>=1.9; "
            "install it with `uv pip install scikit-learn>=1.9`"
        ) from exc
    if reducer == "auto":
        reducer = "umap"
    if reducer == "umap":
        reduced, reduction = _umap_reduce_matrix(
            matrix,
            metric=metric,
            components=umap_components,
            neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
    elif reducer == "pca":
        reduced, reduction = _pca_reduce_matrix(matrix, components=pca_components)
    elif reducer == "none":
        reduced, reduction = matrix, {"method": "none", "reason": "disabled"}
    else:
        raise ValueError(f"unsupported reducer: {reducer}")
    hdbscan_matrix = reduced
    effective_metric = "euclidean"
    if metric == "cosine":
        hdbscan_matrix = _l2_normalize_matrix(reduced)
    elif metric != "euclidean":
        raise ValueError(f"unsupported metric for sklearn_hdbscan: {metric}")

    selection_methods = (
        ["eom", "leaf"] if selection_method == "auto" else [selection_method]
    )
    min_sizes = _hdbscan_min_cluster_sizes(
        row_count=len(hdbscan_matrix),
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        explicit=min_cluster_size,
    )
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for method_name in selection_methods:
        for size in min_sizes:
            model = HDBSCAN(
                min_cluster_size=size,
                min_samples=min_samples,
                metric=effective_metric,
                cluster_selection_method=method_name,
                allow_single_cluster=False,
                copy=True,
            )
            model.fit(hdbscan_matrix)
            labels = [int(label) for label in model.labels_]
            probabilities = [
                float(value)
                for value in getattr(model, "probabilities_", [1.0] * len(labels))
            ]
            candidate = {
                "selection_method": method_name,
                "min_cluster_size": size,
                "min_samples": min_samples,
                "stable_cluster_count": _stable_cluster_count(labels),
                "noise_count": _noise_count(labels),
                "noise_ratio": round(_noise_count(labels) / max(1, len(labels)), 6),
                "mean_cluster_probability": round(
                    _mean_probability(labels, probabilities),
                    6,
                ),
                "score": _hdbscan_score(
                    labels,
                    probabilities,
                    min_clusters=min_clusters,
                    max_clusters=max_clusters,
                ),
                "labels": labels,
                "probabilities": probabilities,
            }
            candidates.append(candidate)
            if best is None or candidate["score"] < best["score"]:
                best = candidate
    if best is None or int(best["stable_cluster_count"]) <= 0:
        raise RuntimeError("sklearn HDBSCAN produced no stable clusters")
    diagnostics = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"labels", "probabilities", "score"}
        }
        | {"score": [round(float(value), 6) for value in candidate["score"]]}
        for candidate in candidates
    ]
    return list(best["labels"]), list(best["probabilities"]), {
        "backend": "umap_hdbscan" if reduction.get("method") == "umap" else (
            "sklearn_hdbscan_pca" if reduction.get("method") == "pca_whiten" else "sklearn_hdbscan"
        ),
        "metric": metric,
        "effective_metric": effective_metric,
        "reduction": reduction,
        "selection_method": best["selection_method"],
        "min_cluster_size": best["min_cluster_size"],
        "min_samples": best["min_samples"],
        "stable_cluster_count": best["stable_cluster_count"],
        "noise_count": best["noise_count"],
        "noise_ratio": best["noise_ratio"],
        "mean_cluster_probability": best["mean_cluster_probability"],
        "tuned_candidates": diagnostics,
    }


def _nearest_neighbors(matrix: list[list[float]], *, metric: str) -> list[int]:
    neighbors: list[int] = []
    for index, row in enumerate(matrix):
        best_index = index
        best_distance = math.inf
        for other_index, other in enumerate(matrix):
            if other_index == index:
                continue
            distance = _distance(row, other, metric)
            if distance < best_distance:
                best_index = other_index
                best_distance = distance
        neighbors.append(best_index)
    return neighbors


def _connected_components(edges: list[tuple[int, int]], count: int) -> list[int]:
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges:
        union(left, right)
    roots: dict[int, int] = {}
    labels: list[int] = []
    for index in range(count):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels


def _centroids(matrix: list[list[float]], labels: list[int]) -> dict[int, list[float]]:
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    for row, label in zip(matrix, labels):
        grouped[label].append(row)
    out: dict[int, list[float]] = {}
    for label, rows in grouped.items():
        width = len(rows[0]) if rows else 0
        out[label] = [
            sum(row[col] for row in rows) / max(1, len(rows))
            for col in range(width)
        ]
    return out


def finch_partitions(matrix: list[list[float]], *, metric: str) -> list[list[int]]:
    if not matrix:
        return []
    if len(matrix) == 1:
        return [[0]]
    partitions: list[list[int]] = []
    current_matrix = matrix
    original_to_current = list(range(len(matrix)))
    current_cluster_members: dict[int, list[int]] = {idx: [idx] for idx in range(len(matrix))}
    previous_count = len(matrix)
    while len(current_matrix) > 1:
        nn = _nearest_neighbors(current_matrix, metric=metric)
        edges: set[tuple[int, int]] = set()
        for index, neighbor in enumerate(nn):
            edges.add(tuple(sorted((index, neighbor))))
            if 0 <= neighbor < len(nn):
                edges.add(tuple(sorted((index, nn[neighbor]))))
        local_labels = _connected_components(sorted(edges), len(current_matrix))
        label_map: dict[int, int] = {}
        next_members: dict[int, list[int]] = {}
        for local_index, raw_label in enumerate(local_labels):
            next_label = label_map.setdefault(raw_label, len(label_map))
            members = current_cluster_members[local_index]
            next_members.setdefault(next_label, []).extend(members)
        original_labels = [0] * len(matrix)
        for label, members in next_members.items():
            for member in members:
                original_labels[member] = label
        cluster_count = len(set(original_labels))
        if cluster_count >= previous_count:
            break
        partitions.append(original_labels)
        previous_count = cluster_count
        centroids = _centroids(matrix, original_labels)
        current_matrix = [centroids[label] for label in sorted(centroids)]
        current_cluster_members = {
            label: next_members[label]
            for label in sorted(next_members)
        }
        original_to_current = [original_labels[index] for index in range(len(matrix))]
        if cluster_count == 1:
            break
    return partitions or [[index for index in range(len(matrix))]]


def _cluster_count(labels: list[int]) -> int:
    return len(set(labels))


def _relabel_by_size(labels: list[int]) -> list[str]:
    counts = Counter(labels)
    order = {
        label: rank
        for rank, (label, _count) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
    }
    return [f"cluster_{order[label]:02d}" for label in labels]


def _pick_partition(
    partitions: list[list[int]],
    *,
    min_clusters: int,
    max_clusters: int,
) -> list[int]:
    if not partitions:
        return []
    in_range = [
        labels
        for labels in partitions
        if min_clusters <= _cluster_count(labels) <= max_clusters
    ]
    if in_range:
        target = (min_clusters + max_clusters) / 2.0
        return min(in_range, key=lambda labels: abs(_cluster_count(labels) - target))
    target = max(min_clusters, min(max_clusters, _cluster_count(partitions[0])))
    return min(partitions, key=lambda labels: abs(_cluster_count(labels) - target))


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
    structured = normalize_feature_matrix(rows)
    dense, dense_meta = _dense_embedding_matrix(rows)
    resolved = feature_space
    if feature_space == "auto":
        resolved = "fused" if dense else "structured"
    if resolved == "structured":
        return structured, {
            "requested": feature_space,
            "resolved": resolved,
            "structured_dim": len(structured[0]) if structured else 0,
            "dense": dense_meta,
        }
    if resolved == "dense":
        if not dense:
            return structured, {
                "requested": feature_space,
                "resolved": "structured",
                "fallback_reason": "no_dense_embeddings",
                "structured_dim": len(structured[0]) if structured else 0,
                "dense": dense_meta,
            }
        dense_norm = _l2_normalize_matrix(_zscore_matrix(dense))
        return dense_norm, {
            "requested": feature_space,
            "resolved": resolved,
            "structured_dim": len(structured[0]) if structured else 0,
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
        "dense": dense_meta,
        "fused_dim": len(fused[0]) if fused else 0,
        "block_scaling": "l2_per_block",
    }


def _umap_reduce_matrix(
    matrix: list[list[float]],
    *,
    metric: str,
    components: int,
    neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    if components <= 0 or len(matrix) < 5:
        return matrix, {"method": "none", "reason": "umap_not_applicable"}
    try:
        import umap
    except Exception as exc:  # pragma: no cover - exercised only without umap-learn.
        raise RuntimeError(
            "UMAP backend requires umap-learn; install it with `uv pip install umap-learn`"
        ) from exc
    n_neighbors = min(max(2, neighbors), max(2, len(matrix) - 1))
    n_components = min(max(2, components), max(2, len(matrix) - 2), len(matrix[0]))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        transform_seed=random_state,
    )
    reduced = reducer.fit_transform(matrix)
    return reduced.tolist(), {
        "method": "umap",
        "components": n_components,
        "neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
        "input_dim": len(matrix[0]) if matrix else 0,
    }


def _representatives(rows: list[dict[str, Any]], matrix: list[list[float]], labels: list[str], *, per_cluster: int) -> list[dict[str, Any]]:
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
                }
            )
    return representatives


def _cluster_summaries(rows: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
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
        density_counts = Counter(
            str((((member.get("qc") or {}).get("text_density") or {}).get("level") or ""))
            for member in members
            if isinstance(member.get("qc"), Mapping)
        )
        char_counts = [
            int(((member.get("text_features") or {}).get("char_count") or 0))
            for member in members
            if isinstance(member.get("text_features"), Mapping)
        ]
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
                "text_density_counts": dict(density_counts.most_common()),
                "char_count_avg": round(sum(char_counts) / max(1, len(char_counts)), 3),
                "examples": [
                    {
                        "sample_id": member.get("sample_id", ""),
                        "start": member.get("start"),
                        "duration_s": member.get("duration_s"),
                        "text_preview": member.get("text_preview", ""),
                    }
                    for member in members[:5]
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
      <button id="download">下载人工标签 JSONL</button>
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
      <h3>人工标签</h3>
      <p class="meta">内容标签和决策标签分开标；低一致性簇先标 uncertain/review。</p>
      <div class="labels" id="contentButtons"></div>
      <div style="height:8px"></div>
      <div class="labels" id="displayButtons"></div>
      <div style="height:8px"></div>
      <div class="labels" id="alignButtons"></div>
      <div style="height:8px"></div>
      <div class="labels" id="qcButtons"></div>
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
const CONTENT = [["dialogue","dialogue"],["non_dialogue","non_dialogue"],["mixed","mixed"],["uncertain","uncertain"]];
const DISPLAY = [["keep","keep"],["drop","drop"],["compact","compact"],["review","review"]];
const ALIGN = [["align","align"],["skip_align_fallback","skip_align_fallback"]];
const QC = [["keep","keep"],["review","review"],["reject","reject"]];
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
    ["density", JSON.stringify(qc.text_density || {{}})],
    ["adjacency", JSON.stringify(row.adjacency || {{}})],
    ["audio", (row.audio || {{}}).path || ""]
  ];
  document.getElementById("metrics").innerHTML = metrics.map(([k,v]) => `<div>${{escapeHtml(k)}}</div><div>${{escapeHtml(v)}}</div>`).join("");
  setButtons("contentButtons", CONTENT, "content_label");
  setButtons("displayButtons", DISPLAY, "display_decision");
  setButtons("alignButtons", ALIGN, "alignment_policy");
  setButtons("qcButtons", QC, "qc_decision");
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
  a.download = "cueqc_manual_labels.jsonl";
  a.click();
  URL.revokeObjectURL(a.href);
}};
setupClusters();
applyFilters();
</script>
</body>
</html>
"""


def cluster_rows(
    rows: list[dict[str, Any]],
    *,
    method: str,
    metric: str,
    min_clusters: int,
    max_clusters: int,
    representatives_per_cluster: int,
    feature_space: str = "auto",
    reducer: str = "umap",
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    selection_method: str = "auto",
    pca_components: int = 8,
    umap_components: int = 8,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    random_state: int = 17,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if method not in {"auto", "umap_hdbscan", "sklearn_hdbscan", "finch_first_neighbor"}:
        raise ValueError(f"unsupported method: {method}")
    matrix, feature_summary = _feature_matrix(rows, feature_space=feature_space)
    if not matrix:
        return [], [], [], {
            "schema": "cueqc_cluster_summary_v1",
            "method": method,
            "cluster_count": 0,
            "candidate_count": 0,
            "feature_space": feature_summary,
        }
    resolved_method = "umap_hdbscan" if method == "auto" else method
    hdbscan_error = ""
    if resolved_method in {"umap_hdbscan", "sklearn_hdbscan"}:
        try:
            labels_raw_hdbscan, probabilities, backend_summary = _sklearn_hdbscan_cluster(
                matrix,
                metric=metric,
                reducer="umap" if resolved_method == "umap_hdbscan" else reducer,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                selection_method=selection_method,
                pca_components=pca_components,
                umap_components=umap_components,
                umap_neighbors=umap_neighbors,
                umap_min_dist=umap_min_dist,
                random_state=random_state,
            )
            labels = _relabel_hdbscan_by_size(labels_raw_hdbscan)
            clustered_rows = []
            for row, raw_label, label, probability in zip(
                rows,
                labels_raw_hdbscan,
                labels,
                probabilities,
            ):
                item = dict(row)
                item["cluster_id"] = label
                item["cluster_raw_label"] = int(raw_label)
                item["cluster_method"] = backend_summary["backend"]
                item["cluster_backend"] = backend_summary["backend"]
                item["cluster_noise"] = int(raw_label) < 0
                item["cluster_confidence"] = round(float(probability), 6)
                clustered_rows.append(item)
            summaries = _cluster_summaries(clustered_rows, labels)
            representatives = _representatives(
                clustered_rows,
                matrix,
                labels,
                per_cluster=representatives_per_cluster,
            )
            summary = {
                "schema": "cueqc_cluster_summary_v1",
                "method": backend_summary["backend"],
                "requested_method": method,
                "metric": metric,
                "candidate_count": len(rows),
                "cluster_count": len(
                    {label for label in labels if not label.startswith("noise_")}
                ),
                "total_groups_including_noise": len(summaries),
                "target_min_clusters": min_clusters,
                "target_max_clusters": max_clusters,
                "cluster_counts": {
                    summary["cluster_id"]: summary["count"] for summary in summaries
                },
                "representatives_per_cluster": representatives_per_cluster,
                "feature_space": feature_summary,
                "backend": backend_summary,
            }
            return clustered_rows, representatives, summaries, summary
        except Exception as exc:
            if method != "auto":
                raise
            hdbscan_error = repr(exc)
            resolved_method = "finch_first_neighbor"

    partitions = finch_partitions(matrix, metric=metric)
    labels_raw = _pick_partition(
        partitions,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )
    labels = _relabel_by_size(labels_raw)
    clustered_rows = []
    for row, label in zip(rows, labels):
        item = dict(row)
        item["cluster_id"] = label
        item["cluster_method"] = "finch_first_neighbor"
        item["cluster_backend"] = "finch_first_neighbor"
        item["cluster_noise"] = False
        item["cluster_confidence"] = 1.0
        clustered_rows.append(item)
    summaries = _cluster_summaries(clustered_rows, labels)
    representatives = _representatives(
        clustered_rows,
        matrix,
        labels,
        per_cluster=representatives_per_cluster,
    )
    summary = {
        "schema": "cueqc_cluster_summary_v1",
        "method": "finch_first_neighbor",
        "requested_method": method,
        "metric": metric,
        "candidate_count": len(rows),
        "cluster_count": len(summaries),
        "target_min_clusters": min_clusters,
        "target_max_clusters": max_clusters,
        "partition_cluster_counts": [_cluster_count(labels) for labels in partitions],
        "cluster_counts": {summary["cluster_id"]: summary["count"] for summary in summaries},
        "representatives_per_cluster": representatives_per_cluster,
        "feature_space": feature_summary,
    }
    if hdbscan_error:
        summary["fallback_reason"] = f"umap_hdbscan_unavailable:{hdbscan_error}"
    return clustered_rows, representatives, summaries, summary


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    if args.max_items is not None:
        rows = rows[: args.max_items]
    clustered, representatives, summaries, summary = cluster_rows(
        rows,
        method=args.method,
        metric=args.metric,
        feature_space=args.feature_space,
        reducer=args.reducer,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        representatives_per_cluster=args.representatives_per_cluster,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        selection_method=args.selection_method,
        pca_components=args.pca_components,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        random_state=args.random_state,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Cluster CueQC candidates for cluster-first manual audit.")
    parser.add_argument("--input", required=True, help="cueqc_candidates.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--method",
        choices=("auto", "umap_hdbscan", "sklearn_hdbscan", "finch_first_neighbor"),
        default="auto",
        help="auto resolves to fused embeddings + UMAP + HDBSCAN, with FINCH fallback.",
    )
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="euclidean")
    parser.add_argument(
        "--feature-space",
        choices=("auto", "structured", "dense", "fused"),
        default="auto",
        help="auto uses fused structured+dense embeddings when embeddings are present.",
    )
    parser.add_argument(
        "--reducer",
        choices=("auto", "umap", "pca", "none"),
        default="umap",
        help="dimensionality reducer for sklearn_hdbscan; umap_hdbscan always uses UMAP.",
    )
    parser.add_argument("--min-clusters", type=int, default=DEFAULT_CLUSTER_COUNT_MIN)
    parser.add_argument("--max-clusters", type=int, default=DEFAULT_CLUSTER_COUNT_MAX)
    parser.add_argument("--min-cluster-size", type=int)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument(
        "--selection-method",
        choices=("auto", "eom", "leaf"),
        default="auto",
        help="HDBSCAN cluster selection strategy; auto sweeps eom and leaf.",
    )
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--umap-components", type=int, default=8)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--representatives-per-cluster", type=int, default=3)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--title", default="CueQC cluster-first 审计")
    args = parser.parse_args(argv)
    if args.min_clusters <= 0 or args.max_clusters <= 0:
        parser.error("cluster bounds must be positive")
    if args.min_clusters > args.max_clusters:
        parser.error("--min-clusters must be <= --max-clusters")
    if args.representatives_per_cluster <= 0:
        parser.error("--representatives-per-cluster must be positive")
    if args.min_cluster_size is not None and args.min_cluster_size <= 1:
        parser.error("--min-cluster-size must be > 1")
    if args.min_samples is not None and args.min_samples <= 0:
        parser.error("--min-samples must be positive")
    if args.pca_components < 0:
        parser.error("--pca-components must be non-negative")
    if args.umap_components < 0:
        parser.error("--umap-components must be non-negative")
    if args.umap_neighbors <= 1:
        parser.error("--umap-neighbors must be > 1")
    if args.umap_min_dist < 0.0:
        parser.error("--umap-min-dist must be non-negative")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
