#!/usr/bin/env python3
"""Generate a CueQC v3-Fusion false-drop audit page."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import ANON_LABELS, update_audit_entrypoints  # noqa: E402
from tools.audits.generate_cueqc_cluster_audit_html import (  # noqa: E402
    discover_media,
    enrich_rows,
    json_for_script,
    project_path,
    project_rel,
    read_jsonl,
    row_float,
    write_json,
)


LABEL_SCHEMA = "cueqc_false_drop_audit_label_v1"


def _confidence_bin(row: Mapping[str, Any]) -> str:
    confidence = row_float(row, "display_prob_drop", row_float(row, "confidence"))
    if confidence < 0.87:
        return "drop_085_087"
    if confidence < 0.89:
        return "drop_087_089"
    if confidence < 0.91:
        return "drop_089_091"
    return "drop_091_plus"


def _text_bucket(row: Mapping[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    compact = text.replace("…", ".").replace("・", "").strip()
    if not compact or set(compact) <= {".", "!", "?", "。", "、", " "}:
        return "punct_or_empty"
    if len(text) <= 4:
        return "short_text"
    if len(text) <= 16:
        return "medium_text"
    return "long_text"


def _sample_round_robin(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("video_id") or ""), _confidence_bin(row), _text_bucket(row))
        groups[key].append(row)
    for bucket in groups.values():
        bucket.sort(
            key=lambda item: (
                -row_float(item, "display_prob_drop", row_float(item, "confidence")),
                str(item.get("sample_id") or ""),
            )
        )
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while keys and len(selected) < limit:
        next_keys: list[tuple[str, str, str]] = []
        for key in keys:
            bucket = groups[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def select_audit_rows(
    predictions: list[dict[str, Any]],
    *,
    max_drop: int,
    min_drop_confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    drop_rows = [
        row
        for row in predictions
        if str(row.get("display_hint") or "") == "drop"
        and row_float(row, "display_prob_drop", row_float(row, "confidence")) >= min_drop_confidence
    ]
    sampled = _sample_round_robin(drop_rows, limit=max_drop, seed=seed)
    sampled.sort(
        key=lambda row: (
            str(row.get("video_id") or ""),
            row_float(row, "start"),
            int(row.get("chunk_index") or 0),
        )
    )
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(sampled):
        item = dict(row)
        start = row_float(item, "start")
        end = row_float(item, "end", start)
        item.update(
            {
                "audit_id": str(item.get("sample_id") or f"cueqc-pred-{index:05d}"),
                "audit_index": index,
                "audit_bucket": f"{str(item.get('video_id') or '')}:{_confidence_bin(item)}:{_text_bucket(item)}",
                "duration_s": max(0.0, end - start),
                "video_label": ANON_LABELS.get(str(item.get("video_id") or ""), str(item.get("video_id") or "")),
            }
        )
        rows.append(item)
    return rows


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<style>
:root {
  --bg: #f6f7f4;
  --panel: #fff;
  --ink: #202521;
  --muted: #687169;
  --line: #d9ded8;
  --accent: #0f766e;
  --accent-soft: #e1f2ef;
  --danger: #b42318;
  --warn: #9a6500;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
}
button, input, textarea, select { font: inherit; }
button {
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  padding: 8px 10px;
  cursor: pointer;
}
button.active, button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button.danger { color: var(--danger); }
.app { display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: 100vh; }
.sidebar { border-right: 1px solid var(--line); background: #fbfbf9; max-height: 100vh; overflow: auto; }
.side-head { position: sticky; top: 0; z-index: 2; background: #fbfbf9; border-bottom: 1px solid var(--line); padding: 12px; }
.side-head h1 { margin: 0 0 8px; font-size: 16px; }
.filters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.filters input, .filters select { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 7px 8px; background: #fff; }
.item { display: grid; gap: 4px; padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }
.item.active { background: var(--accent-soft); }
.item.done .item-title::before { content: "✓ "; color: var(--accent); }
.item.false-drop .item-title::before { content: "! "; color: var(--danger); }
.item-title { font-weight: 650; overflow-wrap: anywhere; }
.meta { color: var(--muted); font-size: 12px; }
.workspace { padding: 18px; max-height: 100vh; overflow: auto; }
.topbar { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.topbar h2 { margin: 0; font-size: 18px; overflow-wrap: anywhere; }
.actions, .labels, .toolbar, .tags { display: flex; flex-wrap: wrap; gap: 8px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 12px; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 12px; }
audio { width: 100%; margin: 8px 0 10px; }
.caption { min-height: 52px; border: 1px solid var(--line); border-radius: 6px; background: #101614; color: #f5f5f0; padding: 10px; white-space: pre-wrap; overflow-wrap: anywhere; }
.text-box, .cue-list { white-space: pre-wrap; overflow-wrap: anywhere; }
.cue { border-bottom: 1px solid var(--line); padding: 6px 0; }
.cue:last-child { border-bottom: 0; }
.kv { display: grid; grid-template-columns: 145px minmax(0, 1fr); gap: 6px 10px; overflow-wrap: anywhere; }
.badge { display: inline-flex; border: 1px solid var(--line); border-radius: 999px; padding: 2px 7px; color: var(--muted); font-size: 12px; }
textarea { width: 100%; min-height: 70px; border: 1px solid var(--line); border-radius: 6px; padding: 8px; resize: vertical; }
@media (max-width: 980px) {
  .app { grid-template-columns: 1fr; }
  .sidebar { max-height: none; }
  .grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="side-head">
      <h1>%%TITLE%%</h1>
      <div class="meta" id="progressText"></div>
      <div class="filters">
        <input id="searchInput" placeholder="搜索文本/影片">
        <select id="stateFilter">
          <option value="">全部</option>
          <option value="todo">未审</option>
          <option value="drop_ok">丢弃正确</option>
          <option value="false_drop_keep">误删应保留</option>
          <option value="uncertain">不确定</option>
        </select>
      </div>
    </div>
    <div id="itemList"></div>
  </aside>
  <main class="workspace">
    <div class="topbar">
      <div>
        <h2 id="rowTitle"></h2>
        <div class="meta" id="rowMeta"></div>
      </div>
      <div class="actions">
        <button id="prevBtn">上一条</button>
        <button id="nextBtn">下一条</button>
        <button id="copyBtn">复制 JSONL</button>
        <button id="downloadBtn" class="primary">导出 JSONL</button>
      </div>
    </div>
    <section class="panel">
      <div class="labels">
        <button data-label="drop_ok" class="primary">丢弃正确</button>
        <button data-label="false_drop_keep" class="danger">误删应保留</button>
        <button data-label="uncertain">不确定</button>
      </div>
      <div class="tags" style="margin-top:10px">
        <button data-tag="dialogue">正常对白</button>
        <button data-tag="vocalization">语气词/呻吟</button>
        <button data-tag="environment">环境/噪声</button>
        <button data-tag="overlap">多人/重叠</button>
        <button data-tag="short_fragment">短碎片</button>
      </div>
      <p class="meta" id="labelStatus"></p>
    </section>
    <div class="grid">
      <section class="panel">
        <audio id="media" controls preload="metadata"></audio>
        <div class="toolbar">
          <button id="playChunkBtn">播放片段</button>
          <button id="playContextBtn">播放上下文</button>
          <a id="vttLink" href="#">打开 VTT</a>
        </div>
        <h3>当前字幕</h3>
        <div class="caption" id="captionText"></div>
      </section>
      <aside class="panel">
        <h3>模型输出</h3>
        <div class="kv" id="metrics"></div>
      </aside>
    </div>
    <section class="panel">
      <h3>ASR 文本</h3>
      <div class="text-box" id="asrText"></div>
    </section>
    <section class="panel">
      <h3>字幕上下文</h3>
      <div class="grid">
        <div>
          <h3>片段内字幕</h3>
          <div class="cue-list" id="chunkCues"></div>
        </div>
        <div>
          <h3>上下文字幕</h3>
          <div class="cue-list" id="contextCues"></div>
        </div>
      </div>
    </section>
    <section class="panel">
      <h3>aligned segments</h3>
      <div class="cue-list" id="alignedSegments"></div>
    </section>
    <section class="panel">
      <h3>备注</h3>
      <textarea id="notes"></textarea>
    </section>
  </main>
</div>
<script type="application/json" id="rows-json">%%ROWS_JSON%%</script>
<script type="application/json" id="cues-json">%%CUES_JSON%%</script>
<script>
const ROWS = JSON.parse(document.getElementById("rows-json").textContent);
const CUES_BY_VIDEO = JSON.parse(document.getElementById("cues-json").textContent);
const DATASET_ID = %%DATASET_ID_JSON%%;
const OUTPUT_NAME = "cueqc_false_drop_audit_labels.jsonl";
const STORAGE_KEY = `cueqc-prediction-audit:${DATASET_ID}:${location.pathname}`;
let annotations = loadAnnotations();
let currentIndex = 0;
let activeVideo = "";
let rangeEnd = null;
const media = document.getElementById("media");

function loadAnnotations() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (_) { return {}; }
}
function saveAnnotations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  renderList();
  renderLabelState();
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[ch]));
}
function fmt(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "0.000";
}
function ann(row = ROWS[currentIndex]) {
  const key = row.audit_id;
  if (!annotations[key]) {
    annotations[key] = { audit_id: key, sample_id: row.sample_id, manual_decision: "", reason_tags: [], notes: "" };
  }
  return annotations[key];
}
function state(row) {
  const value = (annotations[row.audit_id] || {}).manual_decision || "";
  return value || "todo";
}
function visibleRows() {
  const query = document.getElementById("searchInput").value.trim().toLowerCase();
  const filter = document.getElementById("stateFilter").value;
  return ROWS.map((row, index) => ({row, index})).filter(item => {
    if (filter && state(item.row) !== filter) return false;
    if (!query) return true;
    return [item.row.video_label, item.row.video_id, item.row.text, item.row.sample_id, item.row.audit_bucket]
      .join("\\n").toLowerCase().includes(query);
  });
}
function renderList() {
  const root = document.getElementById("itemList");
  root.innerHTML = "";
  for (const item of visibleRows()) {
    const row = item.row;
    const st = state(row);
    const div = document.createElement("div");
    div.className = `item ${item.index === currentIndex ? "active" : ""} ${st !== "todo" ? "done" : ""} ${st === "false_drop_keep" ? "false-drop" : ""}`;
    div.onclick = () => loadRow(item.index);
    div.innerHTML = `
      <div class="item-title">${item.index + 1}. ${escapeHtml(row.video_label || row.video_id)} · ${fmt(row.confidence)}</div>
      <div class="meta">${escapeHtml(row.sample_id)} · ${fmt(row.start)}-${fmt(row.end)}</div>
      <div class="meta">${escapeHtml((row.text || "").slice(0, 90))}</div>`;
    root.appendChild(div);
  }
  const counts = ROWS.reduce((acc, row) => { acc[state(row)] = (acc[state(row)] || 0) + 1; return acc; }, {});
  document.getElementById("progressText").textContent =
    `${counts.drop_ok || 0} 丢弃正确 · ${counts.false_drop_keep || 0} 误删 · ${counts.uncertain || 0} 不确定 · ${counts.todo || 0} 未审`;
}
function renderCues(rootId, cues) {
  const root = document.getElementById(rootId);
  if (!cues || cues.length === 0) {
    root.innerHTML = '<div class="meta">无字幕重叠</div>';
    return;
  }
  root.innerHTML = cues.map(cue => `<div class="cue"><span class="meta">${fmt(cue.start)}-${fmt(cue.end)}</span><br>${escapeHtml(cue.text || "")}</div>`).join("");
}
function activeCueText(videoId, time) {
  return (CUES_BY_VIDEO[videoId] || [])
    .filter(cue => Number(cue.start) <= time && time <= Number(cue.end))
    .map(cue => cue.text || "")
    .join("\\n");
}
function renderMetrics(row) {
  const entries = [
    ["预测", row.display_hint],
    ["confidence", row.confidence],
    ["p_drop", row.display_prob_drop],
    ["p_keep", row.display_prob_keep],
    ["阈值", row.drop_threshold],
    ["bucket", row.audit_bucket],
    ["chunk", row.chunk_index],
    ["时长", row.duration_s]
  ];
  document.getElementById("metrics").innerHTML = entries
    .map(([key, value]) => `<div>${escapeHtml(key)}</div><div>${escapeHtml(value)}</div>`)
    .join("");
}
function renderLabelState() {
  const row = ROWS[currentIndex];
  const item = ann(row);
  document.querySelectorAll("[data-label]").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.label === item.manual_decision);
  });
  document.querySelectorAll("[data-tag]").forEach(btn => {
    btn.classList.toggle("active", (item.reason_tags || []).includes(btn.dataset.tag));
  });
  document.getElementById("notes").value = item.notes || "";
  const labelText = item.manual_decision ? item.manual_decision : "未审";
  document.getElementById("labelStatus").textContent = `${labelText} · ${(item.reason_tags || []).join(", ")}`;
}
function loadRow(index) {
  currentIndex = Math.max(0, Math.min(ROWS.length - 1, index));
  const row = ROWS[currentIndex];
  ann(row);
  if (activeVideo !== row.video_id) {
    activeVideo = row.video_id;
    media.src = (row.media || {}).audio_url || "";
  }
  rangeEnd = null;
  document.getElementById("rowTitle").textContent = `${currentIndex + 1} / ${ROWS.length} · ${row.video_label || row.video_id}`;
  document.getElementById("rowMeta").textContent = `${row.sample_id} · ${fmt(row.start)}-${fmt(row.end)} · p_drop=${fmt(row.display_prob_drop)}`;
  document.getElementById("asrText").textContent = row.text || "";
  document.getElementById("vttLink").href = (row.media || {}).vtt_url || "#";
  renderMetrics(row);
  renderCues("chunkCues", row.chunk_subtitle_cues || []);
  renderCues("contextCues", row.context_subtitle_cues || []);
  renderCues("alignedSegments", row.aligned_segments || []);
  renderLabelState();
  renderList();
  media.currentTime = Number(row.start || 0);
  updateCaption();
}
function setDecision(value) {
  const row = ROWS[currentIndex];
  const item = ann(row);
  item.manual_decision = value;
  item.updated_at = new Date().toISOString();
  saveAnnotations();
  if (currentIndex < ROWS.length - 1) loadRow(currentIndex + 1);
}
function toggleTag(value) {
  const item = ann();
  const tags = new Set(item.reason_tags || []);
  if (tags.has(value)) tags.delete(value); else tags.add(value);
  item.reason_tags = Array.from(tags);
  item.updated_at = new Date().toISOString();
  saveAnnotations();
}
function playRange(start, end) {
  rangeEnd = Number(end);
  media.currentTime = Number(start || 0);
  media.play();
}
function updateCaption() {
  const row = ROWS[currentIndex];
  const text = activeCueText(row.video_id, media.currentTime || 0);
  document.getElementById("captionText").textContent = text || "";
  if (rangeEnd !== null && media.currentTime >= rangeEnd) {
    media.pause();
    rangeEnd = null;
  }
}
function exportRows() {
  return ROWS.filter(row => (annotations[row.audit_id] || {}).manual_decision).map(row => {
    const item = annotations[row.audit_id] || {};
    return {
      schema: "%%LABEL_SCHEMA%%",
      dataset_id: DATASET_ID,
      audit_id: row.audit_id,
      sample_id: row.sample_id,
      video_id: row.video_id,
      video_label: row.video_label,
      chunk_index: row.chunk_index,
      start: row.start,
      end: row.end,
      text: row.text,
      model_display_hint: row.display_hint,
      display_prob_drop: row.display_prob_drop,
      display_prob_keep: row.display_prob_keep,
      confidence: row.confidence,
      manual_decision: item.manual_decision,
      is_false_drop: item.manual_decision === "false_drop_keep",
      reason_tags: item.reason_tags || [],
      notes: item.notes || "",
      updated_at: item.updated_at || ""
    };
  });
}
function exportJsonl() {
  return exportRows().map(row => JSON.stringify(row)).join("\\n") + "\\n";
}
function downloadJsonl() {
  const blob = new Blob([exportJsonl()], {type: "application/jsonl;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = OUTPUT_NAME;
  a.click();
  URL.revokeObjectURL(a.href);
}
async function copyJsonl() {
  await navigator.clipboard.writeText(exportJsonl());
}
document.querySelectorAll("[data-label]").forEach(btn => btn.onclick = () => setDecision(btn.dataset.label));
document.querySelectorAll("[data-tag]").forEach(btn => btn.onclick = () => toggleTag(btn.dataset.tag));
document.getElementById("notes").addEventListener("input", event => {
  const item = ann();
  item.notes = event.target.value;
  item.updated_at = new Date().toISOString();
  saveAnnotations();
});
document.getElementById("prevBtn").onclick = () => loadRow(currentIndex - 1);
document.getElementById("nextBtn").onclick = () => loadRow(currentIndex + 1);
document.getElementById("playChunkBtn").onclick = () => playRange(ROWS[currentIndex].start, ROWS[currentIndex].end);
document.getElementById("playContextBtn").onclick = () => playRange(ROWS[currentIndex].context_start, ROWS[currentIndex].context_end);
document.getElementById("downloadBtn").onclick = downloadJsonl;
document.getElementById("copyBtn").onclick = copyJsonl;
document.getElementById("searchInput").oninput = renderList;
document.getElementById("stateFilter").onchange = renderList;
media.addEventListener("timeupdate", updateCaption);
loadRow(0);
</script>
</body>
</html>
"""


def _page_html(
    *,
    title: str,
    dataset_id: str,
    rows: list[dict[str, Any]],
    cues_by_video: Mapping[str, list[dict[str, Any]]],
) -> str:
    return (
        HTML_TEMPLATE
        .replace("%%TITLE%%", title)
        .replace("%%ROWS_JSON%%", json_for_script(rows))
        .replace("%%CUES_JSON%%", json_for_script(cues_by_video))
        .replace("%%DATASET_ID_JSON%%", json.dumps(dataset_id, ensure_ascii=False))
        .replace("%%LABEL_SCHEMA%%", LABEL_SCHEMA)
    )


def build_audit(
    *,
    predictions_jsonl: Path,
    baseline_root: Path,
    output_dir: Path,
    title: str,
    dataset_id: str,
    max_drop: int = 200,
    min_drop_confidence: float = 0.85,
    seed: int = 20260617,
    refresh_nav: bool = False,
) -> dict[str, Any]:
    predictions = read_jsonl(predictions_jsonl)
    rows = select_audit_rows(
        predictions,
        max_drop=max_drop,
        min_drop_confidence=min_drop_confidence,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    media_by_video, cues_by_video, aligned_by_video = discover_media(
        baseline_root=baseline_root,
        output_dir=output_dir,
        rows=rows,
    )
    rows = enrich_rows(
        rows=rows,
        media_by_video=media_by_video,
        cues_by_video=cues_by_video,
        aligned_by_video=aligned_by_video,
    )

    index_path = output_dir / "index.html"
    index_path.write_text(
        _page_html(title=title, dataset_id=dataset_id, rows=rows, cues_by_video=cues_by_video),
        encoding="utf-8",
    )
    items_path = output_dir / "cueqc_prediction_audit_items.jsonl"
    items_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    counts = Counter(str(row.get("display_hint") or "") for row in predictions)
    sample_counts = Counter(str(row.get("video_id") or "") for row in rows)
    missing_media = [video_id for video_id, info in media_by_video.items() if not info.get("audio_exists")]
    missing_subtitles = [video_id for video_id, info in media_by_video.items() if not info.get("subtitle_exists")]
    summary = {
        "dataset_id": dataset_id,
        "title": title,
        "html": project_rel(index_path),
        "predictions_path": project_rel(predictions_jsonl),
        "audit_items_path": project_rel(items_path),
        "baseline_root": project_rel(baseline_root),
        "review_item_count": len(rows),
        "prediction_records": len(predictions),
        "prediction_counts": dict(counts),
        "sample_counts_by_video": dict(sample_counts),
        "sample_policy": {
            "max_drop": max_drop,
            "min_drop_confidence": min_drop_confidence,
            "seed": seed,
            "strata": ["video_id", "drop_confidence_bin", "text_bucket"],
        },
        "media_enabled": True,
        "media_mode": "audio",
        "media_by_video": media_by_video,
        "missing_media_videos": missing_media,
        "missing_subtitle_videos": missing_subtitles,
        "audit_generator": "tools/audits/generate_cueqc_prediction_audit_html.py",
        "label_schema": LABEL_SCHEMA,
        "label_export": "cueqc_false_drop_audit_labels.jsonl",
        "manual_decision_options": ["drop_ok", "false_drop_keep", "uncertain"],
    }
    write_json(output_dir / "summary.json", summary)
    if refresh_nav:
        update_audit_entrypoints(latest_html=index_path, title=title)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CueQC prediction false-drop audit page.")
    parser.add_argument("--predictions", required=True, help="cueqc_predictions.jsonl")
    parser.add_argument("--baseline-root", required=True, help="Baseline run root containing archived/ and jobs/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--max-drop", type=int, default=200)
    parser.add_argument("--min-drop-confidence", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--refresh-nav", action="store_true")
    args = parser.parse_args(argv)
    if args.max_drop < 0:
        parser.error("--max-drop must be non-negative")
    if not 0.0 <= args.min_drop_confidence <= 1.0:
        parser.error("--min-drop-confidence must be in [0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_audit(
        predictions_jsonl=project_path(args.predictions),
        baseline_root=project_path(args.baseline_root),
        output_dir=project_path(args.output_dir),
        title=args.title,
        dataset_id=args.dataset_id,
        max_drop=args.max_drop,
        min_drop_confidence=args.min_drop_confidence,
        seed=args.seed,
        refresh_nav=args.refresh_nav,
    )
    print(json.dumps({"ok": True, "html": summary["html"], "review_item_count": summary["review_item_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
