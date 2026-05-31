#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def row_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def audio_url(value: str, *, output_html: Path) -> str:
    if not value:
        return ""
    audio_path = project_path(value)
    try:
        return audio_path.resolve().relative_to(output_html.parent.resolve()).as_posix()
    except ValueError:
        return audio_path.as_uri()


def resolve_video_url(
    row: Mapping[str, Any],
    *,
    output_html: Path,
    video_paths: list[Path],
) -> str:
    if not video_paths:
        return ""
    if len(video_paths) == 1:
        selected = video_paths[0]
    else:
        lookup: dict[str, Path] = {}
        for path in video_paths:
            lookup[path.stem] = path
            lookup[path.stem.lower()] = path
            lookup[path.name] = path
            lookup[path.name.lower()] = path
        raw_video = str(row.get("video") or row.get("video_path") or "").strip()
        selected = lookup.get(raw_video) or lookup.get(raw_video.lower())
        if selected is None and raw_video:
            raw_path = Path(raw_video)
            selected = lookup.get(raw_path.stem) or lookup.get(raw_path.name)
        if selected is None:
            return ""
    try:
        return selected.resolve().relative_to(output_html.parent.resolve()).as_posix()
    except ValueError:
        return selected.as_uri()


def normalize_row(
    row: Mapping[str, Any],
    *,
    output_html: Path,
    index: int,
    video_paths: list[Path] | None = None,
) -> dict[str, Any]:
    duration_s = row_float(row, "duration_s")
    chunk_start_s = row_float(row, "chunk_start_s")
    chunk_end_s = row_float(row, "chunk_end_s")
    if chunk_end_s <= chunk_start_s:
        chunk_start_s = 0.0
        chunk_end_s = duration_s
    chunk_start_s = max(0.0, min(duration_s, chunk_start_s))
    chunk_end_s = max(chunk_start_s, min(duration_s, chunk_end_s))
    valid_start_s = row_float(row, "start")
    valid_end_s = row_float(row, "end")
    source_start_s = row_float(row, "source_start_s")
    source_end_s = row_float(row, "source_end_s")
    if valid_end_s <= valid_start_s and source_start_s > 0:
        valid_start_s = source_start_s + chunk_start_s
        valid_end_s = source_start_s + chunk_end_s
    materialized_pad_s = row_float(row, "materialized_pad_s")
    video_start_s = source_start_s or row_float(row, "materialized_from_start_s")
    video_end_s = source_end_s or row_float(row, "materialized_from_end_s")
    if video_start_s <= 0 and valid_start_s > 0:
        video_start_s = max(0.0, valid_start_s - materialized_pad_s)
    if video_end_s <= video_start_s and valid_end_s > 0:
        video_end_s = valid_end_s + materialized_pad_s
    if valid_start_s > 0:
        video_start_s = min(video_start_s, valid_start_s) if video_start_s > 0 else valid_start_s
    if valid_end_s > 0:
        video_end_s = max(video_end_s, valid_end_s)
    sample_id = str(row.get("sample_id") or f"sample-{index:04d}")
    return {
        "index": index,
        "sample_id": sample_id,
        "audio": str(row.get("audio") or ""),
        "audio_url": audio_url(str(row.get("audio") or ""), output_html=output_html),
        "video": str(row.get("video") or ""),
        "video_url": resolve_video_url(row, output_html=output_html, video_paths=video_paths or []),
        "video_start_s": round(video_start_s, 3),
        "video_end_s": round(video_end_s, 3),
        "valid_start_s": round(valid_start_s, 3),
        "valid_end_s": round(valid_end_s, 3),
        "duration_s": round(duration_s, 3),
        "chunk_start_s": round(chunk_start_s, 3),
        "chunk_end_s": round(chunk_end_s, 3),
        "source_start_s": source_start_s,
        "source_end_s": source_end_s,
        "materialized_from_start_s": row_float(row, "materialized_from_start_s"),
        "materialized_from_end_s": row_float(row, "materialized_from_end_s"),
        "review_type": str(row.get("review_type") or ""),
        "failure_bucket": str(row.get("failure_bucket") or ""),
        "alignment_quality": str(row.get("alignment_quality") or ""),
        "fallback_type": str(row.get("fallback_type") or ""),
        "asr_qc_severity": str(row.get("asr_qc_severity") or ""),
        "asr_qc_reasons": list(row.get("asr_qc_reasons") or []),
        "display_text": str(row.get("display_text") or row.get("text") or ""),
        "align_text": str(row.get("align_text") or ""),
        "raw_text": str(row.get("raw_text") or ""),
        "repetition_suggested_text": str(row.get("repetition_suggested_text") or ""),
        "low_information_level": str(row.get("low_information_level") or ""),
        "audit_reason": str(row.get("audit_reason") or ""),
        "case_label": str(row.get("case_label") or ""),
        "chunk_index": int(row.get("chunk_index") or 0),
        "position": int(row.get("position") or 0),
        "source_audio_path": str(row.get("source_audio_path") or ""),
    }


def html_template(
    *,
    title: str,
    dataset_id: str,
    output_jsonl_name: str,
    rows: list[dict[str, Any]],
    source_manifest: str,
) -> str:
    data_json = json.dumps(rows, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    title_json = json.dumps(title, ensure_ascii=False)
    dataset_json = json.dumps(dataset_id, ensure_ascii=False)
    output_jsonl_json = json.dumps(output_jsonl_name, ensure_ascii=False)
    source_manifest_json = json.dumps(source_manifest, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f5f6f4;
  --panel: #ffffff;
  --ink: #1d2421;
  --muted: #66706c;
  --line: #d8ddd8;
  --accent: #0f766e;
  --accent-soft: #dff2ee;
  --warn: #9a6500;
  --danger: #b42318;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}}
button, input, select, textarea {{ font: inherit; }}
button {{
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 10px;
  background: #fff;
  color: var(--ink);
  cursor: pointer;
}}
button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
button.danger {{ color: var(--danger); }}
button.active {{ background: var(--accent-soft); border-color: var(--accent); }}
.app {{ display: grid; grid-template-columns: 390px minmax(0, 1fr); min-height: 100vh; }}
.sidebar {{
  max-height: 100vh;
  overflow: auto;
  border-right: 1px solid var(--line);
  background: #fbfbf9;
}}
.side-head {{
  position: sticky;
  top: 0;
  z-index: 2;
  padding: 12px;
  border-bottom: 1px solid var(--line);
  background: #fbfbf9;
}}
h1 {{ margin: 0 0 8px; font-size: 16px; }}
h2 {{ margin: 0; font-size: 18px; overflow-wrap: anywhere; }}
h3 {{ margin: 0 0 10px; font-size: 14px; }}
.filters {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }}
.filters input, .filters select {{
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 8px;
  background: #fff;
}}
.item {{
  display: grid;
  gap: 4px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  cursor: pointer;
}}
.item:hover {{ background: #eef3ef; }}
.item.active {{ background: var(--accent-soft); }}
.item.done .item-title::before {{ content: "✓ "; color: var(--accent); }}
.item.drop .item-title::before {{ content: "× "; color: var(--danger); }}
.item-title {{ font-weight: 650; overflow-wrap: anywhere; }}
.meta {{ color: var(--muted); font-size: 12px; }}
.workspace {{ max-height: 100vh; overflow: auto; padding: 18px; }}
.topbar {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  margin-bottom: 12px;
}}
.actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 12px;
}}
.grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 12px; }}
audio {{ width: 100%; margin-bottom: 10px; }}
.timeline {{
  position: relative;
  height: 48px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #111815;
  overflow: hidden;
  cursor: pointer;
}}
.valid-range {{
  position: absolute;
  top: 0;
  bottom: 0;
  background: rgba(15, 118, 110, 0.42);
}}
.cursor {{
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #fff;
}}
.timeline-labels {{
  display: flex;
  justify-content: space-between;
  margin-top: 4px;
  color: var(--muted);
  font-size: 12px;
}}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
.label-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
.label-grid button {{ min-height: 38px; text-align: left; }}
.text-box {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fbfbf9;
  padding: 9px;
  min-height: 44px;
}}
.text-box.suggested {{ border-color: #c9b27a; background: #fffaf0; }}
.kv {{ display: grid; grid-template-columns: 138px minmax(0, 1fr); gap: 6px 10px; font-size: 13px; }}
.kv div:nth-child(odd) {{ color: var(--muted); }}
textarea, input[type="text"] {{
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
}}
textarea {{ min-height: 84px; resize: vertical; }}
.status-line {{ display: flex; gap: 12px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }}
.hint {{ color: var(--muted); font-size: 12px; }}
@media (max-width: 980px) {{
  .app {{ grid-template-columns: 1fr; }}
  .sidebar {{ max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--line); }}
  .workspace {{ max-height: none; }}
  .grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="side-head">
      <h1 id="pageTitle"></h1>
      <div class="status-line">
        <span id="progressText"></span>
        <span id="savedText"></span>
      </div>
      <div class="filters">
        <input id="searchInput" placeholder="搜索文本 / ID / bucket">
        <select id="typeFilter"><option value="">全部类型</option></select>
      </div>
    </div>
    <div id="candidateList"></div>
  </aside>
  <main class="workspace">
    <div class="topbar">
      <div>
        <h2 id="clipTitle"></h2>
        <div class="meta" id="clipMeta"></div>
      </div>
      <div class="actions">
        <button id="prevBtn">上一条</button>
        <button id="nextBtn">下一条</button>
        <button id="copyBtn">复制 JSONL</button>
        <button id="downloadBtn">下载 JSONL</button>
        <button class="primary" id="saveBtn">保存 JSONL</button>
      </div>
    </div>

    <div class="grid">
      <section class="panel">
        <audio id="audio" controls preload="metadata"></audio>
        <div class="timeline" id="timeline">
          <div class="valid-range" id="validRange"></div>
          <div class="cursor" id="cursor"></div>
        </div>
        <div class="timeline-labels">
          <span id="rangeStart"></span>
          <span id="nowText"></span>
          <span id="rangeEnd"></span>
        </div>
        <div class="toolbar">
          <button id="playRangeBtn" class="primary">播放有效区间</button>
          <button id="playAllBtn">播放含 padding 全段</button>
          <button id="pauseBtn">暂停</button>
          <button id="replayBtn">重播</button>
        </div>
        <p class="hint">点击左侧条目会自动从有效区间起点播放，并在有效区间终点暂停；音频前后各含少量 padding，便于判断边界。</p>
      </section>

      <aside class="panel">
        <h3>元数据</h3>
        <div class="kv" id="metrics"></div>
      </aside>
    </div>

    <section class="panel">
      <h3>人工标签</h3>
      <div class="label-grid" id="labelButtons"></div>
      <div class="toolbar">
        <button id="markReviewedBtn">仅标记已审</button>
        <button class="danger" id="clearLabelBtn">清除本条标注</button>
      </div>
    </section>

    <section class="panel">
      <h3>ASR 显示文本</h3>
      <div class="text-box" id="displayText"></div>
    </section>

    <section class="panel" id="suggestedPanel">
      <h3>重复修复建议</h3>
      <div class="text-box suggested" id="suggestedText"></div>
    </section>

    <section class="panel">
      <h3>Align Text</h3>
      <div class="text-box" id="alignText"></div>
    </section>

    <section class="panel">
      <h3>人工修订</h3>
      <label class="hint" for="manualText">人工文本 / 建议文本</label>
      <textarea id="manualText"></textarea>
      <label class="hint" for="notes">备注</label>
      <textarea id="notes"></textarea>
    </section>
  </main>
</div>

<script>
const PAGE_TITLE = {title_json};
const DATASET_ID = {dataset_json};
const OUTPUT_JSONL_NAME = {output_jsonl_json};
const SOURCE_MANIFEST = {source_manifest_json};
const CANDIDATES = {data_json};
const STORAGE_KEY = "fusionvad-ja-alignment-audit:" + DATASET_ID;

const LABELS = [
  ["keep_text", "保留文本", "当前文本基本可用"],
  ["use_suggested_text", "采用重复修复", "重复循环建议看起来合理"],
  ["drop_non_speech", "非语音/无字幕", "纯噪声、BGM、无字幕价值声音"],
  ["coarse_timing_ok", "粗时间轴可接受", "VAD coarse fallback 可接受"],
  ["needs_split", "需要再切分", "一条里有多段或边界太粗"],
  ["needs_realign", "需要重新对齐", "文本可用但时间轴不可靠"],
  ["bad_asr", "ASR 文本错误", "音频有人声但文本明显不对"],
  ["uncertain", "不确定", "需要后续复核"]
];

let currentIndex = 0;
let annotations = loadAnnotations();
let playMode = "range";

const audio = document.getElementById("audio");
const timeline = document.getElementById("timeline");

function loadAnnotations() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
  catch (_err) {{ return {{}}; }}
}}

function persist() {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  document.getElementById("savedText").textContent = "本地缓存已保存";
  renderList();
}}

function baseAnnotation(row) {{
  return {{
    sample_id: row.sample_id,
    audio: row.audio,
    review_type: row.review_type,
    failure_bucket: row.failure_bucket,
    chunk_start_s: row.chunk_start_s,
    chunk_end_s: row.chunk_end_s,
    display_text: row.display_text,
    repetition_suggested_text: row.repetition_suggested_text,
    manual_label: "",
    manual_text: "",
    reviewed: false,
    notes: ""
  }};
}}

function ann(row = CANDIDATES[currentIndex]) {{
  if (!annotations[row.sample_id]) annotations[row.sample_id] = baseAnnotation(row);
  return annotations[row.sample_id];
}}

function escapeHtml(value) {{
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}}

function fmt(value) {{
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "0.000";
}}

function stateFor(row) {{
  const item = annotations[row.sample_id];
  if (!item || !item.reviewed) return "todo";
  return item.manual_label === "drop_non_speech" ? "drop" : "done";
}}

function renderTypeFilter() {{
  const filter = document.getElementById("typeFilter");
  const types = Array.from(new Set(CANDIDATES.map(row => row.review_type).filter(Boolean))).sort();
  for (const type of types) {{
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type;
    filter.appendChild(option);
  }}
}}

function visibleCandidates() {{
  const query = document.getElementById("searchInput").value.trim().toLowerCase();
  const type = document.getElementById("typeFilter").value;
  return CANDIDATES
    .map((row, index) => ({{ row, index }}))
    .filter(item => {{
      if (type && item.row.review_type !== type) return false;
      if (!query) return true;
      const haystack = [
        item.row.sample_id,
        item.row.review_type,
        item.row.failure_bucket,
        item.row.display_text,
        item.row.repetition_suggested_text,
        item.row.align_text
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    }});
}}

function renderList() {{
  const list = document.getElementById("candidateList");
  list.innerHTML = "";
  const visible = visibleCandidates();
  for (const item of visible) {{
    const row = item.row;
    const div = document.createElement("div");
    div.className = "item " + (item.index === currentIndex ? "active " : "") + stateFor(row);
    div.addEventListener("click", () => loadCandidate(item.index, true));
    div.innerHTML = `
      <div class="item-title">${{item.index + 1}}. ${{escapeHtml(row.review_type || "review")}}</div>
      <div class="meta">${{escapeHtml(row.failure_bucket)}} · chunk ${{row.chunk_index}} · ${{fmt(row.chunk_end_s - row.chunk_start_s)}}s</div>
      <div class="meta">${{escapeHtml((row.display_text || "").slice(0, 90))}}</div>
    `;
    list.appendChild(div);
  }}
  const done = CANDIDATES.filter(row => stateFor(row) === "done").length;
  const dropped = CANDIDATES.filter(row => stateFor(row) === "drop").length;
  document.getElementById("progressText").textContent = `${{done}} 已审，${{dropped}} 非语音，${{CANDIDATES.length}} 总计`;
}}

function renderMetrics(row) {{
  const entries = [
    ["review_type", row.review_type],
    ["failure_bucket", row.failure_bucket],
    ["alignment_quality", row.alignment_quality],
    ["fallback_type", row.fallback_type],
    ["asr_qc_severity", row.asr_qc_severity],
    ["asr_qc_reasons", (row.asr_qc_reasons || []).join(", ")],
    ["clip duration", fmt(row.duration_s)],
    ["play range", `${{fmt(row.chunk_start_s)}} - ${{fmt(row.chunk_end_s)}}`],
    ["source range", `${{fmt(row.materialized_from_start_s)}} - ${{fmt(row.materialized_from_end_s)}}`],
    ["sample_id", row.sample_id],
    ["source_manifest", SOURCE_MANIFEST]
  ];
  document.getElementById("metrics").innerHTML = entries
    .map(([key, value]) => `<div>${{escapeHtml(key)}}</div><div>${{escapeHtml(value)}}</div>`)
    .join("");
}}

function renderLabels() {{
  const row = CANDIDATES[currentIndex];
  const item = ann(row);
  const box = document.getElementById("labelButtons");
  box.innerHTML = "";
  for (const [value, label, title] of LABELS) {{
    const button = document.createElement("button");
    button.className = item.manual_label === value ? "active" : "";
    button.textContent = label;
    button.title = title;
    button.addEventListener("click", () => {{
      const current = ann(row);
      current.manual_label = value;
      current.reviewed = true;
      if (value === "use_suggested_text" && !current.manual_text) {{
        current.manual_text = row.repetition_suggested_text || row.display_text || "";
        document.getElementById("manualText").value = current.manual_text;
      }}
      persist();
      renderLabels();
    }});
    box.appendChild(button);
  }}
}}

function updateTimeline() {{
  const row = CANDIDATES[currentIndex];
  const duration = Math.max(Number(row.duration_s) || audio.duration || 0, 0.001);
  const startPct = row.chunk_start_s / duration * 100;
  const endPct = row.chunk_end_s / duration * 100;
  const range = document.getElementById("validRange");
  range.style.left = `${{startPct}}%`;
  range.style.width = `${{Math.max(0, endPct - startPct)}}%`;
  const cursorPct = Math.max(0, Math.min(100, (audio.currentTime || 0) / duration * 100));
  document.getElementById("cursor").style.left = `${{cursorPct}}%`;
  document.getElementById("rangeStart").textContent = `有效起点 ${{fmt(row.chunk_start_s)}}s`;
  document.getElementById("rangeEnd").textContent = `有效终点 ${{fmt(row.chunk_end_s)}}s`;
  document.getElementById("nowText").textContent = `当前 ${{fmt(audio.currentTime || 0)}}s`;
}}

function playRange() {{
  const row = CANDIDATES[currentIndex];
  playMode = "range";
  audio.currentTime = row.chunk_start_s;
  audio.play().catch(() => {{}});
  updateTimeline();
}}

function loadCandidate(index, autoplay = false) {{
  currentIndex = Math.max(0, Math.min(CANDIDATES.length - 1, index));
  const row = CANDIDATES[currentIndex];
  const item = ann(row);
  audio.src = row.audio_url || "";
  document.getElementById("clipTitle").textContent = `${{currentIndex + 1}} / ${{CANDIDATES.length}} · ${{row.sample_id}}`;
  document.getElementById("clipMeta").textContent = `${{row.review_type}} · ${{row.failure_bucket}} · chunk ${{row.chunk_index}}`;
  document.getElementById("displayText").textContent = row.display_text || "";
  document.getElementById("alignText").textContent = row.align_text || "";
  const suggestedPanel = document.getElementById("suggestedPanel");
  suggestedPanel.style.display = row.repetition_suggested_text ? "" : "none";
  document.getElementById("suggestedText").textContent = row.repetition_suggested_text || "";
  document.getElementById("manualText").value = item.manual_text || "";
  document.getElementById("notes").value = item.notes || "";
  renderMetrics(row);
  renderLabels();
  renderList();
  updateTimeline();
  if (autoplay) {{
    audio.addEventListener("loadedmetadata", playRange, {{ once: true }});
    audio.load();
  }}
}}

function exportRows() {{
  return CANDIDATES.map(row => {{
    const item = ann(row);
    return {{
      sample_id: row.sample_id,
      audio: row.audio,
      review_type: row.review_type,
      failure_bucket: row.failure_bucket,
      chunk_start_s: row.chunk_start_s,
      chunk_end_s: row.chunk_end_s,
      source_start_s: row.materialized_from_start_s,
      source_end_s: row.materialized_from_end_s,
      display_text: row.display_text,
      align_text: row.align_text,
      repetition_suggested_text: row.repetition_suggested_text,
      manual_label: item.manual_label || "",
      manual_text: item.manual_text || "",
      reviewed: Boolean(item.reviewed),
      notes: item.notes || ""
    }};
  }});
}}

function exportJsonl() {{
  return exportRows().map(row => JSON.stringify(row)).join("\\n") + "\\n";
}}

function downloadJsonl() {{
  const blob = new Blob([exportJsonl()], {{ type: "application/jsonl;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = OUTPUT_JSONL_NAME;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}}

async function saveJsonl() {{
  const text = exportJsonl();
  if (window.showSaveFilePicker) {{
    const handle = await window.showSaveFilePicker({{
      suggestedName: OUTPUT_JSONL_NAME,
      types: [{{ description: "JSONL", accept: {{ "application/jsonl": [".jsonl"] }} }}]
    }});
    const writable = await handle.createWritable();
    await writable.write(text);
    await writable.close();
  }} else {{
    downloadJsonl();
  }}
}}

audio.addEventListener("timeupdate", () => {{
  const row = CANDIDATES[currentIndex];
  if (playMode === "range" && audio.currentTime >= row.chunk_end_s) {{
    audio.pause();
    audio.currentTime = row.chunk_end_s;
  }}
  updateTimeline();
}});
audio.addEventListener("loadedmetadata", updateTimeline);

timeline.addEventListener("click", event => {{
  const row = CANDIDATES[currentIndex];
  const rect = timeline.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  audio.currentTime = ratio * row.duration_s;
  updateTimeline();
}});

document.getElementById("playRangeBtn").addEventListener("click", playRange);
document.getElementById("playAllBtn").addEventListener("click", () => {{
  playMode = "all";
  audio.currentTime = 0;
  audio.play().catch(() => {{}});
}});
document.getElementById("pauseBtn").addEventListener("click", () => audio.pause());
document.getElementById("replayBtn").addEventListener("click", playRange);
document.getElementById("prevBtn").addEventListener("click", () => loadCandidate(currentIndex - 1, true));
document.getElementById("nextBtn").addEventListener("click", () => loadCandidate(currentIndex + 1, true));
document.getElementById("copyBtn").addEventListener("click", async () => {{
  await navigator.clipboard.writeText(exportJsonl());
}});
document.getElementById("downloadBtn").addEventListener("click", downloadJsonl);
document.getElementById("saveBtn").addEventListener("click", () => saveJsonl().catch(console.error));
document.getElementById("markReviewedBtn").addEventListener("click", () => {{
  const item = ann();
  item.reviewed = true;
  persist();
  renderLabels();
}});
document.getElementById("clearLabelBtn").addEventListener("click", () => {{
  const row = CANDIDATES[currentIndex];
  delete annotations[row.sample_id];
  persist();
  loadCandidate(currentIndex, false);
}});
document.getElementById("manualText").addEventListener("input", event => {{
  const item = ann();
  item.manual_text = event.target.value;
  item.reviewed = true;
  persist();
}});
document.getElementById("notes").addEventListener("input", event => {{
  const item = ann();
  item.notes = event.target.value;
  item.reviewed = true;
  persist();
}});
document.getElementById("searchInput").addEventListener("input", renderList);
document.getElementById("typeFilter").addEventListener("change", renderList);
document.addEventListener("keydown", event => {{
  if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  if (event.key === " ") {{ event.preventDefault(); audio.paused ? playRange() : audio.pause(); }}
  if (event.key.toLowerCase() === "j") loadCandidate(currentIndex - 1, true);
  if (event.key.toLowerCase() === "k") loadCandidate(currentIndex + 1, true);
  if (event.key.toLowerCase() === "r") playRange();
  if (event.key === "1") document.querySelector('[title="当前文本基本可用"]').click();
  if (event.key === "2") document.querySelector('[title="重复循环建议看起来合理"]').click();
  if (event.key === "3") document.querySelector('[title="纯噪声、BGM、无字幕价值声音"]').click();
}});

document.getElementById("pageTitle").textContent = PAGE_TITLE;
renderTypeFilter();
loadCandidate(0, false);
</script>
</body>
</html>
"""


def video_html_template(
    *,
    title: str,
    dataset_id: str,
    output_jsonl_name: str,
    rows: list[dict[str, Any]],
    source_manifest: str,
) -> str:
    data_json = json.dumps(rows, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE_HTML__</title>
<style>
:root {
  --bg: #f4f6f5;
  --panel: #ffffff;
  --ink: #1e2422;
  --muted: #66716d;
  --line: #d9dedb;
  --accent: #0f766e;
  --accent-soft: #dff2ee;
  --danger: #b42318;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
button, input, select, textarea { font: inherit; }
button {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 10px;
  background: #fff;
  color: var(--ink);
  cursor: pointer;
}
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button.danger { color: var(--danger); }
button.active { background: var(--accent-soft); border-color: var(--accent); }
.app { display: grid; grid-template-columns: 390px minmax(0, 1fr); min-height: 100vh; }
.sidebar { max-height: 100vh; overflow: auto; border-right: 1px solid var(--line); background: #fbfbf9; }
.side-head { position: sticky; top: 0; z-index: 2; padding: 12px; border-bottom: 1px solid var(--line); background: #fbfbf9; }
h1 { margin: 0 0 8px; font-size: 16px; }
h2 { margin: 0; font-size: 18px; overflow-wrap: anywhere; }
h3 { margin: 0 0 10px; font-size: 14px; }
.filters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.filters input, .filters select { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 7px 8px; background: #fff; }
.item { display: grid; gap: 4px; padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }
.item:hover { background: #eef3ef; }
.item.active { background: var(--accent-soft); }
.item.done .item-title::before { content: "✓ "; color: var(--accent); }
.item.drop .item-title::before { content: "× "; color: var(--danger); }
.item-title { font-weight: 650; overflow-wrap: anywhere; }
.meta { color: var(--muted); font-size: 12px; }
.workspace { max-height: 100vh; overflow: auto; padding: 18px; }
.topbar { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 12px; }
.actions, .toolbar { display: flex; flex-wrap: wrap; gap: 8px; }
.actions { justify-content: flex-end; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 12px; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 12px; }
.video-shell { position: relative; background: #080b0a; border-radius: 8px; overflow: hidden; }
video { display: block; width: 100%; max-height: 72vh; background: #080b0a; }
.caption-overlay {
  position: absolute;
  left: 3%;
  right: 3%;
  bottom: 26px;
  padding: 8px 10px;
  border-radius: 6px;
  background: rgba(0, 0, 0, 0.68);
  color: #fff;
  text-align: center;
  font-size: clamp(16px, 2.3vw, 26px);
  line-height: 1.35;
  text-shadow: 0 1px 2px #000;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.caption-overlay.dim { opacity: 0.42; }
.timeline { position: relative; height: 48px; border: 1px solid var(--line); border-radius: 6px; background: #111815; overflow: hidden; cursor: pointer; margin-top: 10px; }
.valid-range { position: absolute; top: 0; bottom: 0; background: rgba(15, 118, 110, 0.45); }
.cursor { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; }
.timeline-labels { display: flex; justify-content: space-between; margin-top: 4px; color: var(--muted); font-size: 12px; }
.label-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.label-grid button { min-height: 38px; text-align: left; }
.text-box { white-space: pre-wrap; overflow-wrap: anywhere; border: 1px solid var(--line); border-radius: 6px; background: #fbfbf9; padding: 9px; min-height: 44px; }
.text-box.suggested { border-color: #c9b27a; background: #fffaf0; }
.kv { display: grid; grid-template-columns: 138px minmax(0, 1fr); gap: 6px 10px; font-size: 13px; }
.kv div:nth-child(odd) { color: var(--muted); }
textarea { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px; min-height: 84px; resize: vertical; }
.status-line, .hint { color: var(--muted); font-size: 12px; }
.status-line { display: flex; gap: 12px; flex-wrap: wrap; }
@media (max-width: 980px) {
  .app { grid-template-columns: 1fr; }
  .sidebar { max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--line); }
  .workspace { max-height: none; }
  .grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="side-head">
      <h1 id="pageTitle"></h1>
      <div class="status-line">
        <span id="progressText"></span>
        <span id="savedText"></span>
      </div>
      <div class="filters">
        <input id="searchInput" placeholder="搜索文本 / ID / bucket">
        <select id="typeFilter"><option value="">全部类型</option></select>
      </div>
    </div>
    <div id="candidateList"></div>
  </aside>

  <main class="workspace">
    <div class="topbar">
      <div>
        <h2 id="clipTitle"></h2>
        <div class="meta" id="clipMeta"></div>
      </div>
      <div class="actions">
        <button id="prevBtn">上一条</button>
        <button id="nextBtn">下一条</button>
        <button id="copyBtn">复制 JSONL</button>
        <button id="downloadBtn">下载 JSONL</button>
        <button class="primary" id="saveBtn">保存 JSONL</button>
      </div>
    </div>

    <div class="grid">
      <section class="panel">
        <div class="video-shell">
          <video id="video" controls preload="metadata" playsinline></video>
          <div class="caption-overlay" id="captionOverlay"></div>
        </div>
        <div class="timeline" id="timeline">
          <div class="valid-range" id="validRange"></div>
          <div class="cursor" id="cursor"></div>
        </div>
        <div class="timeline-labels">
          <span id="rangeStart"></span>
          <span id="nowText"></span>
          <span id="rangeEnd"></span>
        </div>
        <div class="toolbar" style="margin-top:10px">
          <button id="playRangeBtn" class="primary">播放有效区间</button>
          <button id="playContextBtn">播放含 padding 视频片段</button>
          <button id="pauseBtn">暂停</button>
          <button id="replayBtn">重播</button>
        </div>
        <p class="hint">绿色区间是当前字幕对应的有效时间轴；字幕会叠加在原视频上，padding 段会降低字幕透明度。</p>
      </section>

      <aside class="panel">
        <h3>元数据</h3>
        <div class="kv" id="metrics"></div>
      </aside>
    </div>

    <section class="panel">
      <h3>人工标签</h3>
      <div class="label-grid" id="labelButtons"></div>
      <div class="toolbar" style="margin-top:10px">
        <button id="markReviewedBtn">仅标记已审</button>
        <button class="danger" id="clearLabelBtn">清除本条标注</button>
      </div>
    </section>

    <section class="panel">
      <h3>ASR 显示文本</h3>
      <div class="text-box" id="displayText"></div>
    </section>

    <section class="panel" id="suggestedPanel">
      <h3>重复修复建议</h3>
      <div class="text-box suggested" id="suggestedText"></div>
    </section>

    <section class="panel">
      <h3>Align Text</h3>
      <div class="text-box" id="alignText"></div>
    </section>

    <section class="panel">
      <h3>人工修订</h3>
      <label class="hint" for="manualText">人工文本 / 建议文本</label>
      <textarea id="manualText"></textarea>
      <label class="hint" for="notes">备注</label>
      <textarea id="notes"></textarea>
    </section>
  </main>
</div>

<script>
const PAGE_TITLE = __TITLE_JSON__;
const DATASET_ID = __DATASET_JSON__;
const OUTPUT_JSONL_NAME = __OUTPUT_JSONL_JSON__;
const SOURCE_MANIFEST = __SOURCE_MANIFEST_JSON__;
const CANDIDATES = __DATA_JSON__;
const STORAGE_KEY = "fusionvad-ja-video-alignment-audit:" + DATASET_ID;

const LABELS = [
  ["keep_text", "保留文本", "当前文本基本可用"],
  ["use_suggested_text", "采用重复修复", "重复循环建议看起来合理"],
  ["drop_non_speech", "非语音/无字幕", "纯噪声、BGM、无字幕价值声音"],
  ["coarse_timing_ok", "粗时间轴可接受", "VAD coarse fallback 可接受"],
  ["needs_split", "需要再切分", "一条里有多段或边界太粗"],
  ["needs_realign", "需要重新对齐", "文本可用但时间轴不可靠"],
  ["bad_asr", "ASR 文本错误", "视频有人声但文本明显不对"],
  ["uncertain", "不确定", "需要后续复核"]
];

let currentIndex = 0;
let annotations = loadAnnotations();
let playMode = "range";

const video = document.getElementById("video");
const timeline = document.getElementById("timeline");
const captionOverlay = document.getElementById("captionOverlay");

function loadAnnotations() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
  catch (_err) { return {}; }
}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  document.getElementById("savedText").textContent = "本地缓存已保存";
  renderList();
}

function baseAnnotation(row) {
  return {
    sample_id: row.sample_id,
    video: row.video,
    video_start_s: row.video_start_s,
    video_end_s: row.video_end_s,
    valid_start_s: row.valid_start_s,
    valid_end_s: row.valid_end_s,
    review_type: row.review_type,
    failure_bucket: row.failure_bucket,
    display_text: row.display_text,
    repetition_suggested_text: row.repetition_suggested_text,
    manual_label: "",
    manual_text: "",
    reviewed: false,
    notes: ""
  };
}

function ann(row = CANDIDATES[currentIndex]) {
  if (!annotations[row.sample_id]) annotations[row.sample_id] = baseAnnotation(row);
  return annotations[row.sample_id];
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmt(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "0.000";
}

function stateFor(row) {
  const item = annotations[row.sample_id];
  if (!item || !item.reviewed) return "todo";
  return item.manual_label === "drop_non_speech" ? "drop" : "done";
}

function renderTypeFilter() {
  const filter = document.getElementById("typeFilter");
  const types = Array.from(new Set(CANDIDATES.map(row => row.review_type).filter(Boolean))).sort();
  for (const type of types) {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type;
    filter.appendChild(option);
  }
}

function visibleCandidates() {
  const query = document.getElementById("searchInput").value.trim().toLowerCase();
  const type = document.getElementById("typeFilter").value;
  return CANDIDATES
    .map((row, index) => ({ row, index }))
    .filter(item => {
      if (type && item.row.review_type !== type) return false;
      if (!query) return true;
      const haystack = [
        item.row.sample_id,
        item.row.video,
        item.row.review_type,
        item.row.failure_bucket,
        item.row.display_text,
        item.row.repetition_suggested_text,
        item.row.align_text
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    });
}

function renderList() {
  const list = document.getElementById("candidateList");
  list.innerHTML = "";
  for (const item of visibleCandidates()) {
    const row = item.row;
    const div = document.createElement("div");
    div.className = "item " + (item.index === currentIndex ? "active " : "") + stateFor(row);
    div.addEventListener("click", () => loadCandidate(item.index, true));
    div.innerHTML = `
      <div class="item-title">${item.index + 1}. ${escapeHtml(row.review_type || "review")}</div>
      <div class="meta">${escapeHtml(row.failure_bucket)} · chunk ${row.chunk_index} · ${fmt(row.valid_end_s - row.valid_start_s)}s</div>
      <div class="meta">${escapeHtml((row.display_text || "").slice(0, 90))}</div>
    `;
    list.appendChild(div);
  }
  const done = CANDIDATES.filter(row => stateFor(row) === "done").length;
  const dropped = CANDIDATES.filter(row => stateFor(row) === "drop").length;
  document.getElementById("progressText").textContent = `${done} 已审，${dropped} 非语音，${CANDIDATES.length} 总计`;
}

function renderMetrics(row) {
  const entries = [
    ["video", row.video],
    ["review_type", row.review_type],
    ["failure_bucket", row.failure_bucket],
    ["alignment_quality", row.alignment_quality],
    ["fallback_type", row.fallback_type],
    ["asr_qc_severity", row.asr_qc_severity],
    ["asr_qc_reasons", (row.asr_qc_reasons || []).join(", ")],
    ["valid video range", `${fmt(row.valid_start_s)} - ${fmt(row.valid_end_s)}`],
    ["context video range", `${fmt(row.video_start_s)} - ${fmt(row.video_end_s)}`],
    ["audio clip range", `${fmt(row.chunk_start_s)} - ${fmt(row.chunk_end_s)}`],
    ["sample_id", row.sample_id],
    ["source_manifest", SOURCE_MANIFEST]
  ];
  document.getElementById("metrics").innerHTML = entries
    .map(([key, value]) => `<div>${escapeHtml(key)}</div><div>${escapeHtml(value)}</div>`)
    .join("");
}

function renderLabels() {
  const row = CANDIDATES[currentIndex];
  const item = ann(row);
  const box = document.getElementById("labelButtons");
  box.innerHTML = "";
  for (const [value, label, title] of LABELS) {
    const button = document.createElement("button");
    button.className = item.manual_label === value ? "active" : "";
    button.textContent = label;
    button.title = title;
    button.addEventListener("click", () => {
      const current = ann(row);
      current.manual_label = value;
      current.reviewed = true;
      if (value === "use_suggested_text" && !current.manual_text) {
        current.manual_text = row.repetition_suggested_text || row.display_text || "";
        document.getElementById("manualText").value = current.manual_text;
      }
      persist();
      renderLabels();
    });
    box.appendChild(button);
  }
}

function updateTimeline() {
  const row = CANDIDATES[currentIndex];
  const contextDuration = Math.max(row.video_end_s - row.video_start_s, 0.001);
  const validStartPct = (row.valid_start_s - row.video_start_s) / contextDuration * 100;
  const validEndPct = (row.valid_end_s - row.video_start_s) / contextDuration * 100;
  const cursorPct = (video.currentTime - row.video_start_s) / contextDuration * 100;
  document.getElementById("validRange").style.left = `${Math.max(0, Math.min(100, validStartPct))}%`;
  document.getElementById("validRange").style.width = `${Math.max(0, Math.min(100, validEndPct) - Math.max(0, validStartPct))}%`;
  document.getElementById("cursor").style.left = `${Math.max(0, Math.min(100, cursorPct))}%`;
  document.getElementById("rangeStart").textContent = `有效起点 ${fmt(row.valid_start_s)}s`;
  document.getElementById("rangeEnd").textContent = `有效终点 ${fmt(row.valid_end_s)}s`;
  document.getElementById("nowText").textContent = `当前 ${fmt(video.currentTime || 0)}s`;
  const inValid = video.currentTime >= row.valid_start_s && video.currentTime <= row.valid_end_s;
  captionOverlay.classList.toggle("dim", !inValid);
  captionOverlay.textContent = row.display_text || "(空文本)";
}

function seekAndPlay(startS, mode) {
  const row = CANDIDATES[currentIndex];
  playMode = mode;
  video.currentTime = Math.max(0, startS);
  video.play().catch(() => {});
  updateTimeline();
}

function playRange() {
  const row = CANDIDATES[currentIndex];
  seekAndPlay(row.valid_start_s, "range");
}

function playContext() {
  const row = CANDIDATES[currentIndex];
  seekAndPlay(row.video_start_s, "context");
}

function loadCandidate(index, autoplay = false) {
  currentIndex = Math.max(0, Math.min(CANDIDATES.length - 1, index));
  const row = CANDIDATES[currentIndex];
  const item = ann(row);
  video.src = row.video_url || "";
  document.getElementById("clipTitle").textContent = `${currentIndex + 1} / ${CANDIDATES.length} · ${row.sample_id}`;
  document.getElementById("clipMeta").textContent = `${row.video || "video"} · ${row.review_type} · ${row.failure_bucket} · chunk ${row.chunk_index}`;
  document.getElementById("displayText").textContent = row.display_text || "";
  document.getElementById("alignText").textContent = row.align_text || "";
  const suggestedPanel = document.getElementById("suggestedPanel");
  suggestedPanel.style.display = row.repetition_suggested_text ? "" : "none";
  document.getElementById("suggestedText").textContent = row.repetition_suggested_text || "";
  document.getElementById("manualText").value = item.manual_text || "";
  document.getElementById("notes").value = item.notes || "";
  renderMetrics(row);
  renderLabels();
  renderList();
  updateTimeline();
  if (autoplay) {
    video.addEventListener("loadedmetadata", playRange, { once: true });
    video.load();
  }
}

function exportRows() {
  return CANDIDATES.map(row => {
    const item = ann(row);
    return {
      sample_id: row.sample_id,
      video: row.video,
      review_type: row.review_type,
      failure_bucket: row.failure_bucket,
      valid_start_s: row.valid_start_s,
      valid_end_s: row.valid_end_s,
      video_start_s: row.video_start_s,
      video_end_s: row.video_end_s,
      display_text: row.display_text,
      align_text: row.align_text,
      repetition_suggested_text: row.repetition_suggested_text,
      manual_label: item.manual_label || "",
      manual_text: item.manual_text || "",
      reviewed: Boolean(item.reviewed),
      notes: item.notes || ""
    };
  });
}

function exportJsonl() {
  return exportRows().map(row => JSON.stringify(row)).join("\\n") + "\\n";
}

function downloadJsonl() {
  const blob = new Blob([exportJsonl()], { type: "application/jsonl;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = OUTPUT_JSONL_NAME;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function saveJsonl() {
  const text = exportJsonl();
  if (window.showSaveFilePicker) {
    const handle = await window.showSaveFilePicker({
      suggestedName: OUTPUT_JSONL_NAME,
      types: [{ description: "JSONL", accept: { "application/jsonl": [".jsonl"] } }]
    });
    const writable = await handle.createWritable();
    await writable.write(text);
    await writable.close();
  } else {
    downloadJsonl();
  }
}

video.addEventListener("timeupdate", () => {
  const row = CANDIDATES[currentIndex];
  if (playMode === "range" && video.currentTime >= row.valid_end_s) {
    video.pause();
    video.currentTime = row.valid_end_s;
  }
  if (playMode === "context" && video.currentTime >= row.video_end_s) {
    video.pause();
    video.currentTime = row.video_end_s;
  }
  updateTimeline();
});
video.addEventListener("loadedmetadata", updateTimeline);

timeline.addEventListener("click", event => {
  const row = CANDIDATES[currentIndex];
  const rect = timeline.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  video.currentTime = row.video_start_s + ratio * (row.video_end_s - row.video_start_s);
  updateTimeline();
});

document.getElementById("playRangeBtn").addEventListener("click", playRange);
document.getElementById("playContextBtn").addEventListener("click", playContext);
document.getElementById("pauseBtn").addEventListener("click", () => video.pause());
document.getElementById("replayBtn").addEventListener("click", playRange);
document.getElementById("prevBtn").addEventListener("click", () => loadCandidate(currentIndex - 1, true));
document.getElementById("nextBtn").addEventListener("click", () => loadCandidate(currentIndex + 1, true));
document.getElementById("copyBtn").addEventListener("click", async () => {
  await navigator.clipboard.writeText(exportJsonl());
});
document.getElementById("downloadBtn").addEventListener("click", downloadJsonl);
document.getElementById("saveBtn").addEventListener("click", () => saveJsonl().catch(console.error));
document.getElementById("markReviewedBtn").addEventListener("click", () => {
  const item = ann();
  item.reviewed = true;
  persist();
  renderLabels();
});
document.getElementById("clearLabelBtn").addEventListener("click", () => {
  const row = CANDIDATES[currentIndex];
  delete annotations[row.sample_id];
  persist();
  loadCandidate(currentIndex, false);
});
document.getElementById("manualText").addEventListener("input", event => {
  const item = ann();
  item.manual_text = event.target.value;
  item.reviewed = true;
  persist();
});
document.getElementById("notes").addEventListener("input", event => {
  const item = ann();
  item.notes = event.target.value;
  item.reviewed = true;
  persist();
});
document.getElementById("searchInput").addEventListener("input", renderList);
document.getElementById("typeFilter").addEventListener("change", renderList);
document.addEventListener("keydown", event => {
  if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  if (event.key === " ") { event.preventDefault(); video.paused ? playRange() : video.pause(); }
  if (event.key.toLowerCase() === "j") loadCandidate(currentIndex - 1, true);
  if (event.key.toLowerCase() === "k") loadCandidate(currentIndex + 1, true);
  if (event.key.toLowerCase() === "r") playRange();
  if (event.key === "1") document.querySelector('[title="当前文本基本可用"]').click();
  if (event.key === "2") document.querySelector('[title="重复循环建议看起来合理"]').click();
  if (event.key === "3") document.querySelector('[title="纯噪声、BGM、无字幕价值声音"]').click();
});

document.getElementById("pageTitle").textContent = PAGE_TITLE;
renderTypeFilter();
loadCandidate(0, false);
</script>
</body>
</html>
"""
    return (
        template.replace("__TITLE_HTML__", html.escape(title))
        .replace("__TITLE_JSON__", json.dumps(title, ensure_ascii=False))
        .replace("__DATASET_JSON__", json.dumps(dataset_id, ensure_ascii=False))
        .replace("__OUTPUT_JSONL_JSON__", json.dumps(output_jsonl_name, ensure_ascii=False))
        .replace("__SOURCE_MANIFEST_JSON__", json.dumps(source_manifest, ensure_ascii=False))
        .replace("__DATA_JSON__", data_json)
    )


def write_html(
    *,
    manifest: Path,
    output_html: Path,
    title: str,
    dataset_id: str,
    output_jsonl_name: str,
    video_paths: list[Path] | None = None,
) -> dict[str, Any]:
    rows = [
        normalize_row(row, output_html=output_html, index=index, video_paths=video_paths)
        for index, row in enumerate(read_jsonl(manifest))
    ]
    output_html.parent.mkdir(parents=True, exist_ok=True)
    if video_paths:
        html_text = video_html_template(
            title=title,
            dataset_id=dataset_id,
            output_jsonl_name=output_jsonl_name,
            rows=rows,
            source_manifest=project_rel(manifest),
        )
    else:
        html_text = html_template(
            title=title,
            dataset_id=dataset_id,
            output_jsonl_name=output_jsonl_name,
            rows=rows,
            source_manifest=project_rel(manifest),
        )
    output_html.write_text(html_text, encoding="utf-8")
    summary = {
        "source_manifest": project_rel(manifest),
        "html": project_rel(output_html),
        "rows": len(rows),
        "dataset_id": dataset_id,
        "output_jsonl_name": output_jsonl_name,
        "video_paths": [project_rel(path) for path in video_paths or []],
    }
    summary_path = output_html.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local HTML page for alignment failure manual audit.")
    parser.add_argument("--manifest", required=True, help="alignment_failure_audio_manifest.jsonl")
    parser.add_argument(
        "--output-html",
        default="agents/audits/fusionvad-ja/alignment-failure-audit.html",
    )
    parser.add_argument("--title", default="FusionVAD-JA Alignment Failure 人工审计")
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--output-jsonl-name", default="manual_alignment_failure_labels.jsonl")
    parser.add_argument(
        "--video-path",
        action="append",
        default=[],
        help="Original video path. Can be repeated; matched by manifest row 'video' stem. If one path is supplied, it is used for all rows.",
    )
    args = parser.parse_args(argv)
    if not args.dataset_id:
        args.dataset_id = project_path(args.manifest).stem
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = write_html(
        manifest=project_path(args.manifest),
        output_html=project_path(args.output_html),
        title=args.title,
        dataset_id=args.dataset_id,
        output_jsonl_name=args.output_jsonl_name,
        video_paths=[project_path(path) for path in args.video_path],
    )
    print(f"html={summary['html']}")
    print(f"rows={summary['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
