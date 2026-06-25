#!/usr/bin/env python3
"""Cluster CueQC candidates with Torque Clustering (TORC).

The previous HDBSCAN / FINCH / UMAP / PCA backends were removed and the whole
clustering path now routes through :mod:`tools.asr.cueqc.torque`, a single-file
port of Jie Yang's parameter-free Torque Clustering algorithm. TORC decides the
number of clusters autonomously from the distance matrix, so this module no
longer exposes ``--method``, ``--reducer``, ``--min-clusters`` /
``--max-clusters`` or any backend-specific tuning flags — only the feature
space (Pre-ASR numeric / structured / dense / fused embeddings) and the
distance metric remain.
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
                "dense": dense_meta,
            }
        pre_asr_norm = _l2_normalize_matrix(_zscore_matrix(pre_asr))
        return pre_asr_norm, {
            "requested": feature_space,
            "resolved": resolved,
            "pre_asr": pre_asr_meta,
            "dense": dense_meta,
            "normalization": "zscore_then_l2",
        }
    structured = normalize_feature_matrix(rows)
    if resolved == "structured":
        return structured, {
            "requested": feature_space,
            "resolved": resolved,
            "structured_dim": len(structured[0]) if structured else 0,
            "pre_asr": pre_asr_meta,
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
                "dense": dense_meta,
            }
        dense_norm = _l2_normalize_matrix(_zscore_matrix(dense))
        return dense_norm, {
            "requested": feature_space,
            "resolved": resolved,
            "structured_dim": len(structured[0]) if structured else 0,
            "pre_asr": pre_asr_meta,
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


def cluster_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str = "euclidean",
    feature_space: str = "auto",
    representatives_per_cluster: int = 3,
    detect_noise: bool = True,
    adjustment_factor: float | None = None,
    merge_layer: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Cluster candidates with Torque Clustering.

    The cluster count is decided by TORC itself. By default the torque-gap cut
    on the final merge tree is used (factor-auto-tuned); pass ``merge_layer`` to
    instead take a partition from the TORC merge hierarchy (layer 0 = initial
    1-NN layer, layer 1 = after one merge pass, ...) which is far more stable on
    heavy-tailed distance distributions. Returns
    ``(clustered_rows, representatives, summaries, summary)``.
    """
    matrix, feature_summary = _feature_matrix(rows, feature_space=feature_space)
    if not matrix:
        return [], [], [], {
            "schema": "cueqc_cluster_summary_v1",
            "method": "torque",
            "cluster_count": 0,
            "candidate_count": 0,
            "feature_space": feature_summary,
        }

    distance_matrix = _distance_matrix(matrix, metric=metric)
    labels_raw, labels_noise, diag = torque_clustering(
        distance_matrix, k=0, detect_noise=detect_noise,
        adjustment_factor=adjustment_factor, merge_layer=merge_layer,
    )
    # merge_layer partitions carry no noise flag; only the final-tree cut does.
    labels = _relabel_by_size(labels_noise if (detect_noise and merge_layer is None) else labels_raw)

    clustered_rows: list[dict[str, Any]] = []
    for row, label in zip(rows, labels):
        item = dict(row)
        item["cluster_id"] = label
        item["cluster_method"] = "torque"
        item["cluster_backend"] = "torque"
        item["cluster_noise"] = label == "noise_00"
        item["cluster_confidence"] = 1.0
        clustered_rows.append(item)

    summaries = _cluster_summaries(clustered_rows, labels)
    representatives = _representatives(
        clustered_rows, matrix, labels, per_cluster=representatives_per_cluster
    )
    stable_count = len({label for label in labels if not label.startswith("noise_")})
    backend = {
        "algorithm": "torque",
        "mode": "merge_layer" if merge_layer is not None else "torque_gap_cut",
        "merge_layer": merge_layer,
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
    }
    return clustered_rows, representatives, summaries, summary


def preview_layers(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    feature_space: str,
    max_layer: int,
) -> dict[str, Any]:
    matrix, feature_summary = _feature_matrix(rows, feature_space=feature_space)
    distance_matrix = _distance_matrix(matrix, metric=metric)
    layers = torque_merge_layer_preview(distance_matrix, max_layer=max_layer)
    return {
        "schema": "cueqc_torc_layer_preview_v1",
        "method": "torque",
        "metric": metric,
        "candidate_count": len(rows),
        "feature_space": feature_summary,
        "max_layer_requested": max_layer,
        "layers": layers,
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
        choices=("auto", "pre_asr", "structured", "dense", "fused"),
        default="auto",
        help="auto uses Pre-ASR numeric features when present, then dense embeddings, then structured text features.",
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
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
