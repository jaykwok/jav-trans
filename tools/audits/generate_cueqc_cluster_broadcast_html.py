#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


CLUSTER_LABEL_SCHEMA = "cueqc_cluster_label_v1"
BROADCAST_LABEL_SCHEMA = "cueqc_cluster_broadcast_label_v1"
SOURCE_SUMMARY_KEYS = {
    "duration_top10",
    "hidden_nonlexical_summary",
    "rows",
    "sort",
    "subtitle_cue_count",
    "video_counts",
}


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(dict(payload))
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def json_for_script(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _trim_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\r", "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _compact_mapping(value: Any, *, max_items: int = 40) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= max_items:
            break
        if isinstance(item, Mapping):
            out[str(key)] = _compact_mapping(item, max_items=max_items)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            out[str(key)] = list(item[:20]) if hasattr(item, "__getitem__") else list(item)[:20]
        else:
            out[str(key)] = item
    return out


def _row_id(row: Mapping[str, Any]) -> str:
    for key in ("sample_id", "candidate_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    video_id = str(row.get("video_id") or row.get("audio_id") or "").strip()
    chunk_index = row.get("chunk_index", row.get("index", ""))
    if video_id and str(chunk_index).strip():
        return f"{video_id}#{chunk_index}"
    return ""


def project_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = _row_id(row)
        cluster_id = str(row.get("cluster_id") or "unclustered").strip() or "unclustered"
        start = _safe_float(row.get("start"))
        end = _safe_float(row.get("end"))
        duration = _safe_float(row.get("duration_s"), max(0.0, end - start))
        item = {
            "index": index,
            "sample_id": sample_id,
            "candidate_id": str(row.get("candidate_id") or ""),
            "cluster_id": cluster_id,
            "cluster_label": str(row.get("cluster_label") or ""),
            "cluster_method": str(row.get("cluster_method") or ""),
            "cluster_backend": str(row.get("cluster_backend") or ""),
            "cluster_confidence": _safe_float(row.get("cluster_confidence")),
            "cluster_noise": bool(row.get("cluster_noise")),
            "video_id": str(row.get("video_id") or row.get("audio_id") or ""),
            "video_label": str(row.get("video_label") or row.get("video_id") or row.get("audio_id") or ""),
            "audio_id": str(row.get("audio_id") or ""),
            "chunk_index": row.get("chunk_index", row.get("index", index)),
            "start": round(start, 6),
            "end": round(end, 6),
            "duration_s": round(duration, 6),
            "duration_rank": row.get("duration_rank"),
            "audit_sampling_score": _safe_float(row.get("audit_sampling_score"), _safe_float(row.get("cluster_confidence"))),
            "alignment_quality": str(row.get("alignment_quality") or ""),
            "alignment_mode": str(row.get("alignment_mode") or ""),
            "alignment_issue_type": str(row.get("alignment_issue_type") or ""),
            "alignment_issue_subtype": str(row.get("alignment_issue_subtype") or ""),
            "text": _trim_text(row.get("text") or row.get("raw_text")),
            "raw_text": _trim_text(row.get("raw_text") or row.get("text")),
            "text_preview": _trim_text(row.get("text_preview") or row.get("text") or row.get("raw_text"), 120),
            "text_features": _compact_mapping(row.get("text_features")),
            "cue_features": _compact_mapping(row.get("cue_features")),
            "boundary": _compact_mapping(row.get("boundary")),
            "adjacency": _compact_mapping(row.get("adjacency")),
            "asr_signals": _compact_mapping(row.get("asr_signals")),
            "pre_asr_cueqc": _compact_mapping(row.get("pre_asr_cueqc")),
        }
        audio = row.get("audio")
        if isinstance(audio, Mapping):
            item["audio"] = {
                "path": str(audio.get("path") or ""),
                "exists": bool(audio.get("exists", False)),
            }
        elif row.get("source_audio_path"):
            item["audio"] = {"path": str(row.get("source_audio_path") or ""), "exists": True}
        else:
            item["audio"] = {}
        projected.append(item)
    return projected


def cluster_entries(
    *,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    examples_per_cluster: int,
) -> list[dict[str, Any]]:
    rows_by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_cluster.setdefault(str(row.get("cluster_id") or "unclustered"), []).append(row)
    summary_by_cluster = {
        str(summary.get("cluster_id") or "unclustered"): dict(summary)
        for summary in summaries
        if isinstance(summary, Mapping)
    }
    cluster_ids = sorted(
        rows_by_cluster,
        key=lambda cluster_id: (
            -len(rows_by_cluster[cluster_id]),
            cluster_id,
        ),
    )
    entries: list[dict[str, Any]] = []
    for cluster_id in cluster_ids:
        members = list(rows_by_cluster[cluster_id])
        summary = dict(summary_by_cluster.get(cluster_id, {}))
        durations = [_safe_float(row.get("duration_s")) for row in members]
        confidences = [_safe_float(row.get("cluster_confidence")) for row in members]
        video_counts: dict[str, int] = {}
        for row in members:
            video_id = str(row.get("video_id") or row.get("audio_id") or "")
            if video_id:
                video_counts[video_id] = video_counts.get(video_id, 0) + 1
        examples = sorted(
            members,
            key=lambda row: (
                -_safe_float(row.get("duration_s")),
                str(row.get("video_id") or row.get("audio_id") or ""),
                _safe_float(row.get("start")),
            ),
        )[:examples_per_cluster]
        entries.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": str(summary.get("cluster_label") or members[0].get("cluster_label") or cluster_id),
                "cluster_method": str(summary.get("cluster_method") or members[0].get("cluster_method") or ""),
                "cluster_noise": bool(summary.get("cluster_noise", any(bool(row.get("cluster_noise")) for row in members))),
                "count": int(summary.get("count") or len(members)),
                "sample_count": len(members),
                "confidence_avg": round(
                    _safe_float(summary.get("confidence_avg"), sum(confidences) / max(1, len(confidences))),
                    6,
                ),
                "duration_min_s": round(_safe_float(summary.get("duration_min_s"), min(durations) if durations else 0.0), 6),
                "duration_max_s": round(_safe_float(summary.get("duration_max_s"), max(durations) if durations else 0.0), 6),
                "video_counts": summary.get("video_counts") if isinstance(summary.get("video_counts"), Mapping) else video_counts,
                "text_observation_counts": (
                    summary.get("text_observation_counts")
                    if isinstance(summary.get("text_observation_counts"), Mapping)
                    else {}
                ),
                "qc_severity_counts": (
                    summary.get("qc_severity_counts")
                    if isinstance(summary.get("qc_severity_counts"), Mapping)
                    else {}
                ),
                "examples": [
                    {
                        "sample_id": str(row.get("sample_id") or ""),
                        "candidate_id": str(row.get("candidate_id") or ""),
                        "video_id": str(row.get("video_id") or row.get("audio_id") or ""),
                        "chunk_index": row.get("chunk_index"),
                        "start": row.get("start"),
                        "end": row.get("end"),
                        "duration_s": row.get("duration_s"),
                        "text": row.get("text") or row.get("raw_text") or row.get("text_preview") or "",
                        "text_preview": row.get("text_preview") or row.get("text") or row.get("raw_text") or "",
                    }
                    for row in examples
                ],
            }
        )
    return entries


def _page_html(
    *,
    title: str,
    dataset_id: str,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> str:
    template = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f5f6f2;
  --panel: #fff;
  --ink: #202521;
  --muted: #68736d;
  --line: #d8ddd8;
  --accent: #0f766e;
  --accent-soft: #dff2ee;
  --danger: #a33b36;
  --danger-soft: #fae7e5;
  --amber: #a45d08;
  --amber-soft: #f7ead1;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
.app { display: grid; grid-template-columns: 420px minmax(0, 1fr); min-height: 100vh; }
.side { border-right: 1px solid var(--line); background: #fff; max-height: 100vh; overflow: auto; }
.main { padding: 16px; max-height: 100vh; overflow: auto; }
.head { position: sticky; top: 0; z-index: 2; background: #fff; padding: 12px; border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 8px; font-size: 16px; }
h2 { margin: 0 0 8px; font-size: 18px; }
h3 { margin: 0 0 8px; font-size: 14px; }
input, select, textarea, button { font: inherit; }
input, select, textarea { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); }
input, select { padding: 7px; }
textarea { min-height: 82px; padding: 8px; resize: vertical; }
button { border: 1px solid var(--line); border-radius: 6px; padding: 7px 10px; background: #fff; color: var(--ink); cursor: pointer; }
button.active { background: var(--accent-soft); border-color: var(--accent); color: #063d38; }
button.keep { border-color: var(--accent); }
button.drop { border-color: #d58b84; }
button.drop.active { background: var(--danger-soft); border-color: var(--danger); color: #78231f; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button.warn.active { background: var(--amber-soft); border-color: var(--amber); color: #6b3800; }
.filters { display: grid; grid-template-columns: minmax(0, 1fr) 130px; gap: 8px; }
.filter-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }
.toolbar, .actions, .pager { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.toolbar { margin-top: 10px; }
.status { margin-top: 8px; color: var(--muted); font-size: 12px; }
.cluster-list { border-top: 1px solid var(--line); }
.cluster-item { padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }
.cluster-item.active, .cluster-item:hover { background: #edf6f3; }
.cluster-title { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
.cluster-title strong { overflow-wrap: anywhere; }
.cluster-meta, .meta { color: var(--muted); font-size: 12px; }
.badges { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }
.badge { display: inline-block; border-radius: 999px; padding: 1px 7px; background: #eef3ef; color: var(--muted); font-size: 12px; }
.badge.keep { background: var(--accent-soft); color: #064842; }
.badge.drop { background: var(--danger-soft); color: #82221d; }
.badge.warn { background: var(--amber-soft); color: #6f3d00; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 12px; }
.grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr); gap: 12px; align-items: start; }
.kv { display: grid; grid-template-columns: 170px minmax(0, 1fr); gap: 6px 10px; }
.kv > div:nth-child(odd) { color: var(--muted); }
.example-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.example { border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #f8faf8; min-height: 70px; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 15px; }
.sample-controls { display: grid; grid-template-columns: minmax(0, 1fr) 140px 140px; gap: 8px; margin-bottom: 8px; }
.sample-row { display: grid; grid-template-columns: 88px 86px minmax(0, 1fr); gap: 8px; border-top: 1px solid var(--line); padding: 8px 0; }
.sample-row:first-child { border-top: 0; }
.sample-row .text { font-size: 14px; }
code { background: #eef3ef; border-radius: 4px; padding: 1px 4px; }
@media (max-width: 900px) {
  .app { grid-template-columns: 1fr; }
  .side, .main { max-height: none; }
  .grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div class="head">
      <h1>__TITLE__</h1>
      <div class="filters">
        <input id="clusterSearch" placeholder="搜索 cluster / 文本 / sample_id">
        <select id="stateFilter">
          <option value="">全部状态</option>
          <option value="unmarked">未标注</option>
          <option value="keep">保留并广播</option>
          <option value="drop">丢弃并广播</option>
          <option value="mixed_skip">混簇跳过</option>
          <option value="skip">跳过</option>
        </select>
      </div>
      <div class="filter-row">
        <select id="sortBy">
          <option value="count_desc">按样本数降序</option>
          <option value="duration_desc">按最长时长降序</option>
          <option value="confidence_desc">按置信度降序</option>
          <option value="cluster_id">按 cluster id</option>
        </select>
        <select id="noiseFilter">
          <option value="">全部簇</option>
          <option value="normal">普通簇</option>
          <option value="noise">noise 簇</option>
        </select>
      </div>
      <div class="toolbar">
        <button id="saveClusters" type="button">保存簇级标签</button>
        <button id="downloadClusters" type="button">下载簇级标签</button>
        <button id="saveBroadcast" class="primary" type="button">保存广播样本标签</button>
        <button id="downloadBroadcast" type="button">下载广播样本标签</button>
      </div>
      <div class="status" id="saveStatus"></div>
      <div class="status" id="progressText"></div>
    </div>
    <div class="cluster-list" id="clusterList"></div>
  </aside>
  <main class="main">
    <section class="panel">
      <h2 id="clusterTitle">未选择簇</h2>
      <p class="meta" id="clusterMeta"></p>
      <div class="actions" id="actionButtons"></div>
      <label class="meta" for="clusterNotes">notes</label>
      <textarea id="clusterNotes" placeholder="可选：记录为什么广播或为什么混簇跳过"></textarea>
    </section>
    <div class="grid">
      <section class="panel">
        <h3>代表样本</h3>
        <div class="example-grid" id="examples"></div>
      </section>
      <section class="panel">
        <h3>簇指标</h3>
        <div class="kv" id="metrics"></div>
      </section>
    </div>
    <section class="panel">
      <h3>样本明细</h3>
      <div class="sample-controls">
        <input id="sampleSearch" placeholder="搜索当前簇样本文本 / sample_id">
        <select id="sampleSort">
          <option value="duration_desc">时长降序</option>
          <option value="start_asc">时间升序</option>
          <option value="score_desc">score 降序</option>
          <option value="chunk_asc">chunk 升序</option>
        </select>
        <select id="sampleLimit">
          <option value="100">显示 100</option>
          <option value="300">显示 300</option>
          <option value="1000">显示 1000</option>
          <option value="all">显示全部</option>
        </select>
      </div>
      <div class="meta" id="sampleSummary"></div>
      <div id="sampleRows"></div>
    </section>
  </main>
</div>
<script type="application/json" id="dataset-json">__DATASET_ID_JSON__</script>
<script type="application/json" id="rows-json">__ROWS_JSON__</script>
<script type="application/json" id="summaries-json">__SUMMARIES_JSON__</script>
<script type="application/json" id="entries-json">__ENTRIES_JSON__</script>
<script>
const DATASET_ID = JSON.parse(document.getElementById("dataset-json").textContent);
const ROWS = JSON.parse(document.getElementById("rows-json").textContent);
const SUMMARIES = JSON.parse(document.getElementById("summaries-json").textContent);
const CLUSTER_ENTRIES = JSON.parse(document.getElementById("entries-json").textContent);
const STORAGE_KEY = "cueqc-cluster-broadcast:" + DATASET_ID + ":" + location.pathname;
const ACTIONS = [
  {seed_action: "use_seed", display_decision: "keep", label: "保留并广播", className: "keep"},
  {seed_action: "use_seed", display_decision: "drop", label: "丢弃并广播", className: "drop"},
  {seed_action: "mixed_skip", display_decision: "", label: "混簇跳过", className: "warn"},
  {seed_action: "skip", display_decision: "", label: "跳过", className: "warn"}
];
let annotations = loadAnnotations();
let filteredClusters = [...CLUSTER_ENTRIES];
let activeClusterId = CLUSTER_ENTRIES[0] ? CLUSTER_ENTRIES[0].cluster_id : "";

const rowsByCluster = new Map();
for (const row of ROWS) {
  const clusterId = String(row.cluster_id || "unclustered");
  if (!rowsByCluster.has(clusterId)) rowsByCluster.set(clusterId, []);
  rowsByCluster.get(clusterId).push(row);
}
const entryByCluster = new Map(CLUSTER_ENTRIES.map(entry => [entry.cluster_id, entry]));

function loadAnnotations() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (_) { return {}; }
}
function saveAnnotations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  updateProgress();
}
function escapeHtml(text) {
  const entities = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"};
  return String(text ?? "").replace(/[&<>"']/g, ch => entities[ch]);
}
function fmt(value, digits = 3) {
  const n = Number(value || 0);
  return Number.isFinite(n) ? n.toFixed(digits) : "0.000";
}
function clusterRows(clusterId) {
  return rowsByCluster.get(clusterId) || [];
}
function annotationFor(clusterId) {
  return annotations[clusterId] || {};
}
function selectedAction(ann) {
  const seed = String(ann.seed_action || "");
  const decision = String(ann.display_decision || "");
  if (seed === "use_seed" && (decision === "keep" || decision === "drop")) return decision;
  if (seed === "mixed_skip") return "mixed_skip";
  if (seed === "skip") return "skip";
  return "unmarked";
}
function trainingDecision(ann) {
  const decision = String(ann.display_decision || "");
  return ann.seed_action === "use_seed" && (decision === "keep" || decision === "drop") ? decision : "";
}
function actionLabel(state) {
  if (state === "keep") return "保留广播";
  if (state === "drop") return "丢弃广播";
  if (state === "mixed_skip") return "混簇跳过";
  if (state === "skip") return "跳过";
  return "未标注";
}
function clusterSearchBlob(entry) {
  const rows = clusterRows(entry.cluster_id);
  return [
    entry.cluster_id,
    entry.cluster_label,
    JSON.stringify(entry.video_counts || {}),
    ...rows.slice(0, 120).map(row => [row.sample_id, row.video_id, row.text, row.raw_text, row.text_preview].join(" "))
  ].join("\\n").toLowerCase();
}
function applyClusterFilters() {
  const q = document.getElementById("clusterSearch").value.trim().toLowerCase();
  const state = document.getElementById("stateFilter").value;
  const noise = document.getElementById("noiseFilter").value;
  const sortBy = document.getElementById("sortBy").value;
  filteredClusters = CLUSTER_ENTRIES.filter(entry => {
    const currentState = selectedAction(annotationFor(entry.cluster_id));
    if (state && currentState !== state) return false;
    if (noise === "noise" && !entry.cluster_noise) return false;
    if (noise === "normal" && entry.cluster_noise) return false;
    if (q && !clusterSearchBlob(entry).includes(q)) return false;
    return true;
  });
  filteredClusters.sort((a, b) => {
    if (sortBy === "duration_desc") return Number(b.duration_max_s || 0) - Number(a.duration_max_s || 0) || String(a.cluster_id).localeCompare(String(b.cluster_id));
    if (sortBy === "confidence_desc") return Number(b.confidence_avg || 0) - Number(a.confidence_avg || 0) || String(a.cluster_id).localeCompare(String(b.cluster_id));
    if (sortBy === "cluster_id") return String(a.cluster_id).localeCompare(String(b.cluster_id));
    return Number(b.sample_count || b.count || 0) - Number(a.sample_count || a.count || 0) || String(a.cluster_id).localeCompare(String(b.cluster_id));
  });
  if (!filteredClusters.some(entry => entry.cluster_id === activeClusterId)) {
    activeClusterId = filteredClusters[0] ? filteredClusters[0].cluster_id : "";
  }
  renderClusterList();
  renderActiveCluster();
}
function renderClusterList() {
  const root = document.getElementById("clusterList");
  root.innerHTML = "";
  for (const entry of filteredClusters) {
    const ann = annotationFor(entry.cluster_id);
    const state = selectedAction(ann);
    const div = document.createElement("div");
    div.className = "cluster-item" + (entry.cluster_id === activeClusterId ? " active" : "");
    div.onclick = () => {
      activeClusterId = entry.cluster_id;
      renderClusterList();
      renderActiveCluster();
    };
    const stateClass = state === "keep" ? "keep" : state === "drop" ? "drop" : state === "mixed_skip" || state === "skip" ? "warn" : "";
    div.innerHTML =
      '<div class="cluster-title"><strong>' + escapeHtml(entry.cluster_id) + '</strong><span class="badge ' + stateClass + '">' + escapeHtml(actionLabel(state)) + '</span></div>' +
      '<div class="cluster-meta">' + escapeHtml(entry.cluster_label || "") + '</div>' +
      '<div class="badges">' +
        '<span class="badge">' + Number(entry.sample_count || entry.count || 0) + ' samples</span>' +
        '<span class="badge">max ' + fmt(entry.duration_max_s, 2) + 's</span>' +
        '<span class="badge">conf ' + fmt(entry.confidence_avg, 3) + '</span>' +
        (entry.cluster_noise ? '<span class="badge warn">noise</span>' : '') +
      '</div>';
    root.appendChild(div);
  }
}
function setAction(action) {
  if (!activeClusterId) return;
  annotations[activeClusterId] = {
    ...(annotations[activeClusterId] || {}),
    seed_action: action.seed_action,
    display_decision: action.display_decision,
    training_label_included: action.seed_action === "use_seed" && (action.display_decision === "keep" || action.display_decision === "drop"),
    updated_at: new Date().toISOString()
  };
  saveAnnotations();
  renderClusterList();
  renderActiveCluster(false);
}
function renderActions() {
  const root = document.getElementById("actionButtons");
  const ann = annotationFor(activeClusterId);
  root.innerHTML = "";
  for (const action of ACTIONS) {
    const active = ann.seed_action === action.seed_action && String(ann.display_decision || "") === action.display_decision;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = action.label;
    button.className = (active ? "active " : "") + action.className;
    button.onclick = () => setAction(action);
    root.appendChild(button);
  }
  const clear = document.createElement("button");
  clear.type = "button";
  clear.textContent = "清除";
  clear.onclick = () => {
    delete annotations[activeClusterId];
    saveAnnotations();
    renderClusterList();
    renderActiveCluster(false);
  };
  root.appendChild(clear);
}
function renderExamples(entry) {
  const root = document.getElementById("examples");
  const examples = entry.examples && entry.examples.length ? entry.examples : clusterRows(entry.cluster_id).slice(0, 6);
  root.innerHTML = examples.slice(0, 8).map(example =>
    '<div class="example">' +
      '<div class="meta">' + escapeHtml(example.sample_id || "") + ' · ' + escapeHtml(example.video_id || "") + ' · ' + fmt(example.start, 2) + '-' + fmt(example.end, 2) + ' · ' + fmt(example.duration_s, 2) + 's</div>' +
      '<div class="text">' + escapeHtml(example.text || example.text_preview || "") + '</div>' +
    '</div>'
  ).join("") || '<div class="meta">没有代表样本。</div>';
}
function renderMetrics(entry) {
  const rows = clusterRows(entry.cluster_id);
  const sampleIds = rows.map(row => row.sample_id).filter(Boolean);
  const values = [
    ["dataset", DATASET_ID],
    ["cluster", entry.cluster_id],
    ["label", entry.cluster_label || ""],
    ["method", entry.cluster_method || ""],
    ["samples", String(sampleIds.length)],
    ["duration", fmt(entry.duration_min_s, 3) + "-" + fmt(entry.duration_max_s, 3) + "s"],
    ["confidence_avg", fmt(entry.confidence_avg, 4)],
    ["video_counts", JSON.stringify(entry.video_counts || {})],
    ["text_observation_counts", JSON.stringify(entry.text_observation_counts || {})],
    ["qc_severity_counts", JSON.stringify(entry.qc_severity_counts || {})],
    ["sample_id_preview", sampleIds.slice(0, 12).join(", ")]
  ];
  document.getElementById("metrics").innerHTML = values.map(([k, v]) => '<div>' + escapeHtml(k) + '</div><div>' + escapeHtml(v) + '</div>').join("");
}
function sampleBlob(row) {
  return [row.sample_id, row.candidate_id, row.video_id, row.text, row.raw_text, row.text_preview, JSON.stringify(row.text_features || {}), JSON.stringify(row.cue_features || {})].join("\\n").toLowerCase();
}
function renderSamples(entry) {
  const q = document.getElementById("sampleSearch").value.trim().toLowerCase();
  const sortBy = document.getElementById("sampleSort").value;
  const limitRaw = document.getElementById("sampleLimit").value;
  let rows = clusterRows(entry.cluster_id).filter(row => !q || sampleBlob(row).includes(q));
  rows.sort((a, b) => {
    if (sortBy === "start_asc") return Number(a.start || 0) - Number(b.start || 0);
    if (sortBy === "score_desc") return Number(b.audit_sampling_score || b.cluster_confidence || 0) - Number(a.audit_sampling_score || a.cluster_confidence || 0);
    if (sortBy === "chunk_asc") return Number(a.chunk_index || 0) - Number(b.chunk_index || 0);
    return Number(b.duration_s || 0) - Number(a.duration_s || 0);
  });
  const total = rows.length;
  if (limitRaw !== "all") rows = rows.slice(0, Number(limitRaw || 100));
  document.getElementById("sampleSummary").textContent = "显示 " + rows.length + " / " + total + " samples";
  document.getElementById("sampleRows").innerHTML = rows.map(row =>
    '<div class="sample-row">' +
      '<div class="meta">' + escapeHtml(row.video_label || row.video_id || "") + '<br>chunk ' + escapeHtml(row.chunk_index) + '</div>' +
      '<div class="meta">' + fmt(row.start, 2) + '-' + fmt(row.end, 2) + '<br>' + fmt(row.duration_s, 2) + 's</div>' +
      '<div><div class="meta">' + escapeHtml(row.sample_id || "") + '</div><div class="text">' + escapeHtml(row.text || row.raw_text || "") + '</div></div>' +
    '</div>'
  ).join("") || '<div class="meta">当前筛选下没有样本。</div>';
}
function renderActiveCluster(focusNotes = true) {
  const entry = entryByCluster.get(activeClusterId);
  if (!entry) {
    document.getElementById("clusterTitle").textContent = "没有匹配的簇";
    document.getElementById("clusterMeta").textContent = "";
    document.getElementById("examples").innerHTML = "";
    document.getElementById("metrics").innerHTML = "";
    document.getElementById("sampleRows").innerHTML = "";
    return;
  }
  const ann = annotationFor(entry.cluster_id);
  document.getElementById("clusterTitle").textContent = entry.cluster_id + " · " + actionLabel(selectedAction(ann));
  document.getElementById("clusterMeta").textContent = (entry.cluster_label || "") + " · " + Number(entry.sample_count || entry.count || 0) + " samples";
  renderActions();
  const notes = document.getElementById("clusterNotes");
  notes.value = ann.notes || ann.classification_reason || "";
  renderExamples(entry);
  renderMetrics(entry);
  renderSamples(entry);
  if (focusNotes) notes.blur();
}
function clusterLabelRow(entry) {
  const ann = annotationFor(entry.cluster_id);
  const decision = trainingDecision(ann);
  const rows = clusterRows(entry.cluster_id);
  const sampleIds = decision ? rows.map(row => row.sample_id).filter(Boolean) : [];
  return {
    schema: "cueqc_cluster_label_v1",
    dataset_id: DATASET_ID,
    cluster_id: entry.cluster_id,
    cluster_label: entry.cluster_label || "",
    display_decision: decision,
    seed_action: ann.seed_action || "",
    training_label_included: Boolean(decision),
    label_source: "cluster_broadcast_audit",
    notes: ann.notes || ann.classification_reason || "",
    updated_at: ann.updated_at || "",
    count: entry.sample_count || entry.count || rows.length,
    broadcast_sample_count: sampleIds.length,
    skipped_sample_count: decision ? 0 : rows.length,
    sample_ids: sampleIds,
    examples: rows.slice(0, 20).map(row => ({
      sample_id: row.sample_id,
      candidate_id: row.candidate_id || "",
      video_id: row.video_id || "",
      chunk_index: row.chunk_index,
      start: row.start,
      end: row.end,
      duration_s: row.duration_s,
      text: row.text || row.raw_text || ""
    }))
  };
}
function exportClusterRows() {
  return CLUSTER_ENTRIES.map(clusterLabelRow);
}
function broadcastRowsForEntry(entry) {
  const ann = annotationFor(entry.cluster_id);
  const decision = trainingDecision(ann);
  if (!(decision === "keep" || decision === "drop")) return [];
  return clusterRows(entry.cluster_id)
    .filter(row => row.sample_id)
    .map(row => ({
      schema: "cueqc_cluster_broadcast_label_v1",
      dataset_id: DATASET_ID,
      sample_id: row.sample_id,
      candidate_id: row.candidate_id || "",
      cluster_id: entry.cluster_id,
      video_id: row.video_id || "",
      audio_id: row.audio_id || "",
      chunk_index: row.chunk_index,
      start: row.start,
      end: row.end,
      duration_s: row.duration_s,
      display_decision: decision,
      label: decision,
      route: decision === "keep" ? "keep_for_asr" : "drop_before_asr",
      label_source: "cluster_broadcast",
      cluster_seed_action: "use_seed",
      source_cluster_label_schema: "cueqc_cluster_label_v1",
      notes: ann.notes || ann.classification_reason || "",
      updated_at: ann.updated_at || ""
    }));
}
function exportBroadcastRows() {
  return CLUSTER_ENTRIES.flatMap(broadcastRowsForEntry);
}
function downloadJsonl(filename, rows) {
  const text = rows.map(row => JSON.stringify(row)).join("\\n") + (rows.length ? "\\n" : "");
  downloadText(filename, text);
}
function downloadText(filename, text) {
  const blob = new Blob([text], {type: "application/jsonl;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
async function saveJsonl(filename, rows) {
  const text = rows.map(row => JSON.stringify(row)).join("\\n") + (rows.length ? "\\n" : "");
  const status = document.getElementById("saveStatus");
  status.textContent = "正在保存...";
  try {
    const response = await fetch("/__audit_api__/save-labels", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({href: location.pathname, filename, content: text})
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    status.textContent = "已保存到 " + payload.path;
  } catch (error) {
    status.textContent = "保存到审计目录失败，已改为下载：" + (error && error.message ? error.message : error);
    downloadText(filename, text);
  }
}
function updateProgress() {
  const counts = {keep: 0, drop: 0, mixed_skip: 0, skip: 0, unmarked: 0};
  let broadcast = 0;
  for (const entry of CLUSTER_ENTRIES) {
    const state = selectedAction(annotationFor(entry.cluster_id));
    counts[state] = (counts[state] || 0) + 1;
    if (state === "keep" || state === "drop") broadcast += clusterRows(entry.cluster_id).filter(row => row.sample_id).length;
  }
  document.getElementById("progressText").textContent =
    "簇 " + CLUSTER_ENTRIES.length +
    " · keep " + counts.keep +
    " · drop " + counts.drop +
    " · mixed " + counts.mixed_skip +
    " · skip " + counts.skip +
    " · 未标 " + counts.unmarked +
    " · 将广播 " + broadcast + " samples";
}
document.getElementById("clusterSearch").addEventListener("input", applyClusterFilters);
document.getElementById("stateFilter").addEventListener("change", applyClusterFilters);
document.getElementById("sortBy").addEventListener("change", applyClusterFilters);
document.getElementById("noiseFilter").addEventListener("change", applyClusterFilters);
document.getElementById("sampleSearch").addEventListener("input", () => renderActiveCluster(false));
document.getElementById("sampleSort").addEventListener("change", () => renderActiveCluster(false));
document.getElementById("sampleLimit").addEventListener("change", () => renderActiveCluster(false));
document.getElementById("clusterNotes").addEventListener("input", () => {
  if (!activeClusterId) return;
  annotations[activeClusterId] = {
    ...(annotations[activeClusterId] || {}),
    notes: document.getElementById("clusterNotes").value,
    updated_at: new Date().toISOString()
  };
  saveAnnotations();
});
document.getElementById("downloadClusters").onclick = () => downloadJsonl("cueqc_cluster_labels.jsonl", exportClusterRows());
document.getElementById("downloadBroadcast").onclick = () => downloadJsonl("cueqc_cluster_broadcast_labels.jsonl", exportBroadcastRows());
document.getElementById("saveClusters").onclick = () => saveJsonl("cueqc_cluster_labels.jsonl", exportClusterRows()).catch(error => alert(error.message || error));
document.getElementById("saveBroadcast").onclick = () => saveJsonl("cueqc_cluster_broadcast_labels.jsonl", exportBroadcastRows()).catch(error => alert(error.message || error));
applyClusterFilters();
updateProgress();
</script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", html.escape(title))
        .replace("__DATASET_ID_JSON__", json_for_script(dataset_id))
        .replace("__ROWS_JSON__", json_for_script(rows))
        .replace("__SUMMARIES_JSON__", json_for_script(summaries))
        .replace("__ENTRIES_JSON__", json_for_script(entries))
    )


def build_audit(
    *,
    clusters_jsonl: Path,
    summaries_jsonl: Path,
    output_dir: Path,
    title: str,
    dataset_id: str,
    summary_json: Path | None = None,
    examples_per_cluster: int = 8,
    refresh_nav: bool = False,
) -> dict[str, Any]:
    rows = project_rows(read_jsonl(clusters_jsonl))
    summaries = read_jsonl(summaries_jsonl)
    entries = cluster_entries(
        rows=rows,
        summaries=summaries,
        examples_per_cluster=examples_per_cluster,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    index_path.write_text(
        _page_html(
            title=title,
            dataset_id=dataset_id,
            rows=rows,
            summaries=summaries,
            entries=entries,
        ),
        encoding="utf-8",
    )

    summary_source = summary_json if summary_json else output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_source.exists():
        loaded = read_json(summary_source)
        if isinstance(loaded, Mapping):
            for key in SOURCE_SUMMARY_KEYS:
                if key in loaded:
                    summary[key] = loaded[key]
            summary["source_summary_json"] = project_rel(summary_source)
    summary.update(
        {
            "dataset_id": dataset_id,
            "title": title,
            "html": project_rel(index_path),
            "source_clusters": project_rel(clusters_jsonl),
            "source_cluster_summaries": project_rel(summaries_jsonl),
            "review_item_count": len(rows),
            "cluster_review_group_count": len(entries),
            "audit_generator": "tools/audits/generate_cueqc_cluster_broadcast_html.py",
            "media_enabled": False,
            "cluster_review_mode": "cluster_label_broadcast",
            "separate_from_audio_audit": True,
            "cluster_label_schema": CLUSTER_LABEL_SCHEMA,
            "cluster_label_export": "cueqc_cluster_labels.jsonl",
            "broadcast_label_schema": BROADCAST_LABEL_SCHEMA,
            "broadcast_label_export": "cueqc_cluster_broadcast_labels.jsonl",
            "cluster_action_options": ["keep", "drop", "mixed_skip", "skip"],
            "cluster_training_label_rule": (
                "only seed_action=use_seed with display_decision keep/drop broadcasts sample labels; "
                "mixed_skip/skip abstain and export no sample-level broadcast rows"
            ),
            "examples_per_cluster": examples_per_cluster,
            "clusters": project_rel(clusters_jsonl),
            "summaries": project_rel(summaries_jsonl),
        }
    )
    write_json(output_dir / "summary.json", summary)
    if refresh_nav:
        update_audit_entrypoints(latest_html=index_path, title=title)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a standalone CueQC cluster label broadcast audit page.")
    parser.add_argument("--clusters", required=True, help="Clustered candidate JSONL.")
    parser.add_argument("--summaries", required=True, help="Cluster summary JSONL.")
    parser.add_argument("--output-dir", required=True, help="Audit output directory.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--summary-json", help="Optional existing summary JSON to merge.")
    parser.add_argument("--examples-per-cluster", type=int, default=8)
    parser.add_argument("--refresh-nav", action="store_true", help="Rebuild agents/audits index and latest-audit.html")
    args = parser.parse_args(argv)
    if args.examples_per_cluster <= 0:
        parser.error("--examples-per-cluster must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_audit(
        clusters_jsonl=project_path(args.clusters),
        summaries_jsonl=project_path(args.summaries),
        output_dir=project_path(args.output_dir),
        title=args.title,
        dataset_id=args.dataset_id,
        summary_json=project_path(args.summary_json) if args.summary_json else None,
        examples_per_cluster=args.examples_per_cluster,
        refresh_nav=args.refresh_nav,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "html": summary.get("html"),
                "review_item_count": summary.get("review_item_count"),
                "cluster_review_group_count": summary.get("cluster_review_group_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
