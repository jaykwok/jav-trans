#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if method not in {"auto", "finch_first_neighbor"}:
        raise ValueError(f"unsupported method: {method}")
    matrix = normalize_feature_matrix(rows)
    if not matrix:
        return [], [], [], {
            "schema": "cueqc_cluster_summary_v1",
            "method": method,
            "cluster_count": 0,
            "candidate_count": 0,
        }
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
        "metric": metric,
        "candidate_count": len(rows),
        "cluster_count": len(summaries),
        "target_min_clusters": min_clusters,
        "target_max_clusters": max_clusters,
        "partition_cluster_counts": [_cluster_count(labels) for labels in partitions],
        "cluster_counts": {summary["cluster_id"]: summary["count"] for summary in summaries},
        "representatives_per_cluster": representatives_per_cluster,
    }
    return clustered_rows, representatives, summaries, summary


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    if args.max_items is not None:
        rows = rows[: args.max_items]
    clustered, representatives, summaries, summary = cluster_rows(
        rows,
        method=args.method,
        metric=args.metric,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        representatives_per_cluster=args.representatives_per_cluster,
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
        choices=("auto", "finch_first_neighbor"),
        default="auto",
        help="auto currently resolves to FINCH-style first-neighbor clustering.",
    )
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="euclidean")
    parser.add_argument("--min-clusters", type=int, default=DEFAULT_CLUSTER_COUNT_MIN)
    parser.add_argument("--max-clusters", type=int, default=DEFAULT_CLUSTER_COUNT_MAX)
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
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
