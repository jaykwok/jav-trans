#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        for key in ("teacher_segment_counts", "teacher_speech_ratios", "teacher_segments"):
            value = row.get(key)
            if isinstance(value, str) and value.strip().startswith("{"):
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        for key in (
            "duration_s",
            "active_frame_ratio",
            "ignored_frame_ratio",
            "conflict_frame_ratio",
            "weighted_speech_frame_ratio",
            "weighted_negative_frame_ratio",
        ):
            if key in row:
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    row[key] = 0.0
    return rows


def read_candidates(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    return read_jsonl(path)


def resolve_audio_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def safe_audio_name(row: Mapping[str, Any], audio_path: Path) -> str:
    audio_id = str(row.get("audio_id") or audio_path.stem or "audio")
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in audio_id)
    suffix = audio_path.suffix if audio_path.suffix else ".wav"
    return f"{safe}{suffix}"


def audio_url_for_row(
    row: Mapping[str, Any],
    *,
    output_dir: Path,
    copy_audio: bool,
) -> str:
    audio_value = str(row.get("audio") or "")
    if not audio_value:
        return ""
    audio_path = resolve_audio_path(audio_value)
    if copy_audio:
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        target = audio_dir / safe_audio_name(row, audio_path)
        if audio_path.exists() and not target.exists():
            shutil.copy2(audio_path, target)
        return target.relative_to(output_dir).as_posix()
    return audio_path.as_uri()


def normalize_candidate(row: Mapping[str, Any], *, output_dir: Path, copy_audio: bool) -> dict[str, Any]:
    duration_s = float(row.get("duration_s") or 0.0)
    teacher_segments = row.get("teacher_segments") if isinstance(row.get("teacher_segments"), Mapping) else {}
    return {
        "audio_id": str(row.get("audio_id") or ""),
        "audio": str(row.get("audio") or ""),
        "audio_url": audio_url_for_row(row, output_dir=output_dir, copy_audio=copy_audio),
        "duration_s": duration_s,
        "source": str(row.get("source") or ""),
        "text": str(row.get("text") or ""),
        "reason": str(row.get("reason") or ""),
        "label_quality": str(row.get("label_quality") or ""),
        "active_frame_ratio": float(row.get("active_frame_ratio") or 0.0),
        "ignored_frame_ratio": float(row.get("ignored_frame_ratio") or 0.0),
        "conflict_frame_ratio": float(row.get("conflict_frame_ratio") or 0.0),
        "weighted_speech_frame_ratio": float(row.get("weighted_speech_frame_ratio") or 0.0),
        "weighted_negative_frame_ratio": float(row.get("weighted_negative_frame_ratio") or 0.0),
        "teacher_segment_counts": row.get("teacher_segment_counts") or {},
        "teacher_speech_ratios": row.get("teacher_speech_ratios") or {},
        "teacher_segments": teacher_segments,
    }


def has_teacher_segments(candidates: list[dict[str, Any]]) -> bool:
    for candidate in candidates:
        teacher_segments = candidate.get("teacher_segments")
        if not isinstance(teacher_segments, Mapping):
            continue
        for segments in teacher_segments.values():
            if not isinstance(segments, list):
                continue
            for segment in segments:
                if not isinstance(segment, Mapping):
                    continue
                try:
                    if float(segment.get("end", 0.0)) > float(segment.get("start", 0.0)):
                        return True
                except (TypeError, ValueError):
                    continue
    return False


def html_template(*, title: str, dataset_id: str, output_jsonl_name: str, candidates: list[dict[str, Any]]) -> str:
    data_json = json.dumps(candidates, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    title_json = json.dumps(title, ensure_ascii=False)
    dataset_json = json.dumps(dataset_id, ensure_ascii=False)
    output_name_json = json.dumps(output_jsonl_name, ensure_ascii=False)
    teacher_buttons = ""
    if has_teacher_segments(candidates):
        teacher_buttons = """
          <button id="teacherUnionBtn">Teacher 并集</button>
          <button id="teacherIntersectionBtn">Teacher 交集</button>"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f7f4;
  --panel: #ffffff;
  --ink: #1f2523;
  --muted: #66706d;
  --line: #d8ddd8;
  --accent: #0f766e;
  --danger: #b42318;
  --warn: #9a6500;
  --speech: rgba(15, 118, 110, 0.28);
  --teacher: rgba(154, 101, 0, 0.22);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}}
button, input, textarea, select {{
  font: inherit;
}}
button {{
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 6px;
  padding: 7px 10px;
  cursor: pointer;
}}
button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
button.danger {{ color: var(--danger); }}
button:disabled {{ opacity: 0.5; cursor: default; }}
.app {{
  display: grid;
  grid-template-columns: 360px minmax(0, 1fr);
  min-height: 100vh;
}}
.sidebar {{
  border-right: 1px solid var(--line);
  background: #fbfbf9;
  overflow: auto;
  max-height: 100vh;
}}
.side-head {{
  position: sticky;
  top: 0;
  z-index: 2;
  background: #fbfbf9;
  border-bottom: 1px solid var(--line);
  padding: 12px;
}}
.side-head h1 {{ font-size: 16px; margin: 0 0 8px; }}
.filters {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
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
.item.active {{ background: #e7f4f1; }}
.item.done .item-title::before {{ content: "✓ "; color: var(--accent); }}
.item.skip .item-title::before {{ content: "↷ "; color: var(--warn); }}
.item-title {{ font-weight: 650; overflow-wrap: anywhere; }}
.meta {{ color: var(--muted); font-size: 12px; }}
.workspace {{ padding: 18px; overflow: auto; max-height: 100vh; }}
.topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}}
.title h2 {{ margin: 0; font-size: 18px; overflow-wrap: anywhere; }}
.title .meta {{ margin-top: 3px; }}
.actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 12px;
}}
.grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 12px;
}}
audio {{ width: 100%; margin: 8px 0 10px; }}
canvas {{
  display: block;
  width: 100%;
  height: 180px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #101614;
}}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
.kv {{
  display: grid;
  grid-template-columns: 135px minmax(0, 1fr);
  gap: 6px 10px;
  font-size: 13px;
}}
.kv div:nth-child(odd) {{ color: var(--muted); }}
.text-box {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #fbfbf9;
  min-height: 44px;
}}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 7px 5px; text-align: left; }}
th {{ color: var(--muted); font-weight: 600; }}
td input {{
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 5px;
}}
textarea {{
  width: 100%;
  min-height: 72px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  resize: vertical;
}}
.status-line {{
  display: flex;
  gap: 12px;
  color: var(--muted);
  font-size: 12px;
  flex-wrap: wrap;
}}
.hint {{ color: var(--muted); font-size: 12px; }}
@media (max-width: 900px) {{
  .app {{ grid-template-columns: 1fr; }}
  .sidebar {{ max-height: 44vh; border-right: 0; border-bottom: 1px solid var(--line); }}
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
        <input id="searchInput" placeholder="搜索文本 / ID">
        <select id="reasonFilter"><option value="">全部原因</option></select>
      </div>
    </div>
    <div id="candidateList"></div>
  </aside>
  <main class="workspace">
    <div class="topbar">
      <div class="title">
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
        <canvas id="wave" width="1200" height="220"></canvas>
        <div class="toolbar">
          <button id="playPauseBtn">播放/暂停</button>
          <button id="startBtn">设起点 [</button>
          <button id="endBtn">设终点 ]</button>
          <button id="allSpeechBtn">全段语音</button>
          <button id="startToHereBtn">开头到当前点</button>
          <button id="hereToEndBtn">当前点到末尾</button>{teacher_buttons}
          <button class="danger" id="nonSpeechBtn">非语音</button>
          <button id="reviewedBtn">标记已审</button>
        </div>
        <p class="hint">快捷键：空格播放/暂停，[ 设起点，] 设终点，A 全段语音，N 非语音，B 开头到当前点，E 当前点到末尾，J/K 上一条/下一条。快速审计优先使用四类结果：全段语音、非语音、开头到当前点、当前点到末尾；少数多段对白再用片段表格精修。只设起点时默认到音频末尾，只设终点时默认从音频开头开始。修改会自动缓存在当前浏览器。标注口径：凡是希望送入 ASR 并可能生成字幕的人声、拟声、短促发声都可标为语音；纯 BGM、静音、环境/机械声和无字幕价值残留标为非语音。背景音乐、底噪或环境声垫在对白下面时仍按可字幕化人声区间标为语音。</p>
      </section>

      <aside class="panel">
        <div class="kv" id="metrics"></div>
      </aside>
    </div>

    <section class="panel">
      <h3>文本</h3>
      <div class="text-box" id="clipText"></div>
    </section>

    <section class="panel">
      <h3>人工语音片段</h3>
      <table>
        <thead><tr><th style="width:36%">开始</th><th style="width:36%">结束</th><th></th></tr></thead>
        <tbody id="segmentRows"></tbody>
      </table>
      <div class="toolbar">
        <button id="addEmptyBtn">添加空行</button>
        <button class="danger" id="clearBtn">清空片段</button>
      </div>
    </section>

    <section class="panel">
      <h3>备注</h3>
      <label class="hint" for="skipReason">跳过原因（不确定或不纳入训练时填写）</label>
      <input id="skipReason" list="skipReasonOptions" placeholder="例如 moan_only / human_nonverbal / bad_audio" style="width:100%;border:1px solid var(--line);border-radius:6px;padding:8px;margin:4px 0 10px;">
      <datalist id="skipReasonOptions">
        <option value="moan_only">纯呻吟/喘息，无可辨识语言</option>
        <option value="human_nonverbal">笑声、哭声、尖叫等非语言人声</option>
        <option value="no_dialogue">无可辨识对白</option>
        <option value="pure_bgm">纯背景音乐或环境声</option>
        <option value="bad_audio">音频损坏或噪声过重</option>
        <option value="uncertain">边界或类别不确定</option>
        <option value="duplicate">重复样本</option>
        <option value="out_of_domain">样本不符合本轮任务</option>
      </datalist>
      <label class="hint" for="notes">标注备注</label>
      <textarea id="notes"></textarea>
    </section>
  </main>
</div>

<script>
const PAGE_TITLE = {title_json};
const DATASET_ID = {dataset_json};
const OUTPUT_JSONL_NAME = {output_name_json};
const CANDIDATES = {data_json};
const STORAGE_KEY = "fusionvad-ja-manual-audit:" + DATASET_ID;

let currentIndex = 0;
let annotations = loadAnnotations();
let pendingStart = null;
let decodedPeaks = null;
let saveHandle = null;

const audio = document.getElementById("audio");
const wave = document.getElementById("wave");
const ctx = wave.getContext("2d");

function loadAnnotations() {{
  try {{
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
  }} catch (_err) {{
    return {{}};
  }}
}}

function persist() {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  document.getElementById("savedText").textContent = "本地缓存已保存";
  renderList();
  drawWaveform();
}}

function baseAnnotation(candidate) {{
  return {{
    audio_id: candidate.audio_id,
    audio: candidate.audio,
    duration_s: candidate.duration_s,
    source: candidate.source,
    text: candidate.text,
    reason: candidate.reason,
    label_quality: "manual",
    speech_segments: [],
    reviewed: false,
    skip_reason: "",
    notes: ""
  }};
}}

function ann(candidate = CANDIDATES[currentIndex]) {{
  if (!annotations[candidate.audio_id]) {{
    annotations[candidate.audio_id] = baseAnnotation(candidate);
  }}
  return annotations[candidate.audio_id];
}}

function clampTime(value, duration) {{
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(duration || 0, number));
}}

function hasTimeValue(value) {{
  return value !== undefined && value !== null && String(value).trim() !== "" && Number.isFinite(Number(value));
}}

function normalizeSegments(segments, duration) {{
  const clipDuration = Number(duration) || 0;
  return (segments || [])
    .map(seg => {{
      const hasStart = hasTimeValue(seg.start);
      const hasEnd = hasTimeValue(seg.end);
      if (!hasStart && !hasEnd) return null;
      return {{
        start: clampTime(hasStart ? seg.start : 0, clipDuration),
        end: clampTime(hasEnd ? seg.end : clipDuration, clipDuration)
      }};
    }})
    .filter(seg => seg && seg.end > seg.start)
    .sort((a, b) => a.start - b.start)
    .reduce((merged, seg) => {{
      const last = merged[merged.length - 1];
      if (last && seg.start <= last.end) {{
        last.end = Math.max(last.end, seg.end);
      }} else {{
        merged.push({{start: seg.start, end: seg.end}});
      }}
      return merged;
    }}, []);
}}

function setSegments(segments, reviewed = true) {{
  const candidate = CANDIDATES[currentIndex];
  const item = ann(candidate);
  item.speech_segments = normalizeSegments(segments, candidate.duration_s);
  item.reviewed = reviewed;
  persist();
  renderSegments();
}}

function reviewedState(candidate) {{
  const item = annotations[candidate.audio_id];
  if (!item) return "todo";
  if (item.skip_reason) return "skip";
  if (item.reviewed) return "done";
  return "todo";
}}

function renderReasonFilter() {{
  const filter = document.getElementById("reasonFilter");
  const reasons = Array.from(new Set(CANDIDATES.map(row => row.reason).filter(Boolean))).sort();
  for (const reason of reasons) {{
    const option = document.createElement("option");
    option.value = reason;
    option.textContent = reason;
    filter.appendChild(option);
  }}
}}

function visibleCandidates() {{
  const query = document.getElementById("searchInput").value.trim().toLowerCase();
  const reason = document.getElementById("reasonFilter").value;
  return CANDIDATES
    .map((candidate, index) => ({{candidate, index}}))
    .filter(item => {{
      if (reason && item.candidate.reason !== reason) return false;
      if (!query) return true;
      const haystack = [
        item.candidate.audio_id,
        item.candidate.text,
        item.candidate.reason,
        item.candidate.label_quality
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    }});
}}

function renderList() {{
  const list = document.getElementById("candidateList");
  list.innerHTML = "";
  const visible = visibleCandidates();
  for (const item of visible) {{
    const candidate = item.candidate;
    const state = reviewedState(candidate);
    const div = document.createElement("div");
    div.className = "item " + (item.index === currentIndex ? "active " : "") + state;
    div.addEventListener("click", () => loadCandidate(item.index));
    div.innerHTML = `
      <div class="item-title">${{item.index + 1}}. ${{escapeHtml(candidate.reason || "候选")}}</div>
      <div class="meta">${{escapeHtml(candidate.audio_id)}} · ${{fmt(candidate.duration_s)}}s · ${{escapeHtml(candidate.label_quality)}}</div>
      <div class="meta">${{escapeHtml((candidate.text || "").slice(0, 80))}}</div>
    `;
    list.appendChild(div);
  }}
  const done = CANDIDATES.filter(row => reviewedState(row) === "done").length;
  const skipped = CANDIDATES.filter(row => reviewedState(row) === "skip").length;
  document.getElementById("progressText").textContent = `${{done}} 已审，${{skipped}} 已跳过，${{CANDIDATES.length}} 总计`;
}}

function fmt(value) {{
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "0.000";
}}

function escapeHtml(value) {{
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}}

function loadCandidate(index) {{
  if (index !== currentIndex) commitPendingStart();
  currentIndex = Math.max(0, Math.min(CANDIDATES.length - 1, index));
  const candidate = CANDIDATES[currentIndex];
  pendingStart = null;
  decodedPeaks = null;
  ann(candidate);
  audio.src = candidate.audio_url || "";
  document.getElementById("clipTitle").textContent = `${{currentIndex + 1}} / ${{CANDIDATES.length}} · ${{candidate.audio_id}}`;
  document.getElementById("clipMeta").textContent = `${{candidate.reason}} · ${{fmt(candidate.duration_s)}}s · ${{candidate.label_quality}}`;
  document.getElementById("clipText").textContent = candidate.text || "";
  renderMetrics(candidate);
  renderSegments();
  renderNotes();
  renderList();
  drawWaveform();
  decodeWaveform(candidate);
}}

function renderMetrics(candidate) {{
  const entries = [
    ["active 比例", candidate.active_frame_ratio],
    ["ignore 比例", candidate.ignored_frame_ratio],
    ["冲突比例", candidate.conflict_frame_ratio],
    ["加权语音", candidate.weighted_speech_frame_ratio],
    ["加权非语音", candidate.weighted_negative_frame_ratio]
  ];
  if (hasTeacherData(candidate)) {{
    entries.push(
      ["teacher 片段数", JSON.stringify(candidate.teacher_segment_counts || {{}})],
      ["teacher 语音占比", JSON.stringify(candidate.teacher_speech_ratios || {{}})]
    );
  }}
  document.getElementById("metrics").innerHTML = entries
    .map(([key, value]) => `<div>${{escapeHtml(key)}}</div><div>${{escapeHtml(typeof value === "number" ? fmt(value) : value)}}</div>`)
    .join("");
}}

function renderSegments() {{
  const candidate = CANDIDATES[currentIndex];
  const item = ann(candidate);
  item.speech_segments = normalizeSegments(item.speech_segments, candidate.duration_s);
  const body = document.getElementById("segmentRows");
  body.innerHTML = "";
  item.speech_segments.forEach((seg, index) => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input type="number" step="0.001" min="0" value="${{fmt(seg.start)}}"></td>
      <td><input type="number" step="0.001" min="0" value="${{fmt(seg.end)}}"></td>
      <td><button class="danger">删除</button></td>
    `;
    const inputs = tr.querySelectorAll("input");
    inputs[0].addEventListener("change", event => {{
      item.speech_segments[index].start = event.target.value.trim() === "" ? null : Number(event.target.value);
      setSegments(item.speech_segments, true);
    }});
    inputs[1].addEventListener("change", event => {{
      item.speech_segments[index].end = event.target.value.trim() === "" ? null : Number(event.target.value);
      setSegments(item.speech_segments, true);
    }});
    tr.querySelector("button").addEventListener("click", () => {{
      item.speech_segments.splice(index, 1);
      setSegments(item.speech_segments, true);
    }});
    body.appendChild(tr);
  }});
  drawWaveform();
}}

function renderNotes() {{
  const item = ann();
  document.getElementById("skipReason").value = item.skip_reason || "";
  document.getElementById("notes").value = item.notes || "";
}}

function teacherSegments(candidate) {{
  const out = [];
  const groups = candidate.teacher_segments || {{}};
  for (const segments of Object.values(groups)) {{
    for (const seg of segments || []) {{
      if (Number(seg.end) > Number(seg.start)) {{
        out.push({{start: Number(seg.start), end: Number(seg.end)}});
      }}
    }}
  }}
  return out;
}}

function hasTeacherData(candidate) {{
  return teacherSegments(candidate).length > 0;
}}

function teacherIntersection(candidate) {{
  const groups = Object.values(candidate.teacher_segments || {{}})
    .map(segments => (segments || []).map(seg => ({{start: Number(seg.start), end: Number(seg.end)}})).filter(seg => seg.end > seg.start));
  if (!groups.length) return [];
  let intersections = groups[0];
  for (const group of groups.slice(1)) {{
    const next = [];
    for (const left of intersections) {{
      for (const right of group) {{
        const start = Math.max(left.start, right.start);
        const end = Math.min(left.end, right.end);
        if (end > start) next.push({{start, end}});
      }}
    }}
    intersections = next;
  }}
  return intersections;
}}

async function decodeWaveform(candidate) {{
  drawWaveform();
  if (!candidate.audio_url) return;
  try {{
    const response = await fetch(candidate.audio_url);
    const buffer = await response.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioContext.decodeAudioData(buffer);
    const channel = decoded.getChannelData(0);
    const width = wave.width;
    const block = Math.max(1, Math.floor(channel.length / width));
    decodedPeaks = [];
    for (let x = 0; x < width; x++) {{
      let min = 1;
      let max = -1;
      const start = x * block;
      const end = Math.min(channel.length, start + block);
      for (let i = start; i < end; i++) {{
        const value = channel[i];
        if (value < min) min = value;
        if (value > max) max = value;
      }}
      decodedPeaks.push([min, max]);
    }}
    drawWaveform();
    audioContext.close();
  }} catch (_err) {{
    decodedPeaks = null;
    drawWaveform();
  }}
}}

function drawWaveform() {{
  const candidate = CANDIDATES[currentIndex];
  const item = ann(candidate);
  const width = wave.width;
  const height = wave.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101614";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#4b5c57";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  if (decodedPeaks) {{
    ctx.strokeStyle = "#d7ebe7";
    ctx.beginPath();
    decodedPeaks.forEach(([min, max], x) => {{
      ctx.moveTo(x, (1 - max) * height / 2);
      ctx.lineTo(x, (1 - min) * height / 2);
    }});
    ctx.stroke();
  }} else {{
    ctx.fillStyle = "#d7ebe7";
    ctx.fillText("波形会在音频解码后显示；如果这里为空，播放和标注仍可正常使用。", 16, 26);
  }}

  if (hasTeacherData(candidate)) {{
    drawSegmentBands(teacherSegments(candidate), "rgba(154,101,0,0.24)", 0.0, 0.34);
  }}
  drawSegmentBands(item.speech_segments, "rgba(15,118,110,0.38)", 0.34, 1.0);
  const cursor = audio.currentTime || 0;
  if (candidate.duration_s > 0) {{
    const x = cursor / candidate.duration_s * width;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }}
  if (pendingStart !== null && candidate.duration_s > 0) {{
    const x = pendingStart / candidate.duration_s * width;
    ctx.strokeStyle = "#f8d66d";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }}
}}

function drawSegmentBands(segments, color, topRatio, bottomRatio) {{
  const candidate = CANDIDATES[currentIndex];
  if (!candidate.duration_s) return;
  const y = wave.height * topRatio;
  const h = wave.height * (bottomRatio - topRatio);
  ctx.fillStyle = color;
  for (const seg of segments || []) {{
    const start = clampTime(seg.start, candidate.duration_s);
    const end = clampTime(seg.end, candidate.duration_s);
    if (end <= start) continue;
    const x = start / candidate.duration_s * wave.width;
    const w = Math.max(1, (end - start) / candidate.duration_s * wave.width);
    ctx.fillRect(x, y, w, h);
  }}
}}

function addSegment(start, end) {{
  const candidate = CANDIDATES[currentIndex];
  const item = ann(candidate);
  item.speech_segments.push({{start, end}});
  item.skip_reason = "";
  setSegments(item.speech_segments, true);
  renderNotes();
}}

function commitPendingStart() {{
  if (pendingStart === null) return false;
  const candidate = CANDIDATES[currentIndex];
  const start = clampTime(pendingStart, candidate.duration_s);
  const end = Number(candidate.duration_s) || 0;
  pendingStart = null;
  if (end > start) {{
    addSegment(start, end);
    return true;
  }}
  drawWaveform();
  return false;
}}

function exportRows(annotatedOnly = true) {{
  commitPendingStart();
  const rows = [];
  for (const candidate of CANDIDATES) {{
    const item = annotations[candidate.audio_id] || baseAnnotation(candidate);
    const reviewed = Boolean(item.reviewed || item.skip_reason);
    if (annotatedOnly && !reviewed) continue;
    rows.push({{
      audio_id: candidate.audio_id,
      audio: candidate.audio,
      duration_s: candidate.duration_s,
      source: candidate.source,
      text: candidate.text,
      reason: candidate.reason,
      label_quality: "manual",
      speech_segments: normalizeSegments(item.speech_segments, candidate.duration_s),
      reviewed,
      skip_reason: item.skip_reason || "",
      notes: item.notes || ""
    }});
  }}
  return rows;
}}

function jsonlText() {{
  return exportRows(true).map(row => JSON.stringify(row)).join("\\n") + "\\n";
}}

async function saveJsonl() {{
  const text = jsonlText();
  if ("showSaveFilePicker" in window) {{
    saveHandle = saveHandle || await window.showSaveFilePicker({{
      suggestedName: OUTPUT_JSONL_NAME,
      types: [{{description: "JSONL 标注文件", accept: {{"application/jsonl": [".jsonl"], "text/plain": [".jsonl"]}}}}]
    }});
    const writable = await saveHandle.createWritable();
    await writable.write(text);
    await writable.close();
    document.getElementById("savedText").textContent = "JSONL 已保存";
  }} else {{
    downloadJsonl();
  }}
}}

function downloadJsonl() {{
  const blob = new Blob([jsonlText()], {{type: "application/jsonl;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = OUTPUT_JSONL_NAME;
  link.click();
  URL.revokeObjectURL(url);
}}

async function copyJsonl() {{
  await navigator.clipboard.writeText(jsonlText());
  document.getElementById("savedText").textContent = "JSONL 已复制";
}}

document.getElementById("pageTitle").textContent = PAGE_TITLE;
document.getElementById("searchInput").addEventListener("input", renderList);
document.getElementById("reasonFilter").addEventListener("change", renderList);
document.getElementById("prevBtn").addEventListener("click", () => loadCandidate(currentIndex - 1));
document.getElementById("nextBtn").addEventListener("click", () => loadCandidate(currentIndex + 1));
document.getElementById("playPauseBtn").addEventListener("click", () => audio.paused ? audio.play() : audio.pause());
document.getElementById("startBtn").addEventListener("click", () => {{ pendingStart = audio.currentTime || 0; drawWaveform(); }});
document.getElementById("endBtn").addEventListener("click", () => {{
  const end = audio.currentTime || 0;
  if (pendingStart === null) {{
    addSegment(0, end);
  }} else {{
    addSegment(Math.min(pendingStart, end), Math.max(pendingStart, end));
  }}
  pendingStart = null;
}});
document.getElementById("allSpeechBtn").addEventListener("click", () => setSegments([{{start: 0, end: CANDIDATES[currentIndex].duration_s}}], true));
document.getElementById("startToHereBtn").addEventListener("click", () => {{
  const end = clampTime(audio.currentTime || 0, CANDIDATES[currentIndex].duration_s);
  pendingStart = null;
  setSegments([{{start: 0, end}}], true);
}});
document.getElementById("hereToEndBtn").addEventListener("click", () => {{
  const start = clampTime(audio.currentTime || 0, CANDIDATES[currentIndex].duration_s);
  pendingStart = null;
  setSegments([{{start, end: CANDIDATES[currentIndex].duration_s}}], true);
}});
const teacherUnionBtn = document.getElementById("teacherUnionBtn");
if (teacherUnionBtn) {{
  teacherUnionBtn.addEventListener("click", () => setSegments(teacherSegments(CANDIDATES[currentIndex]), true));
}}
const teacherIntersectionBtn = document.getElementById("teacherIntersectionBtn");
if (teacherIntersectionBtn) {{
  teacherIntersectionBtn.addEventListener("click", () => setSegments(teacherIntersection(CANDIDATES[currentIndex]), true));
}}
document.getElementById("nonSpeechBtn").addEventListener("click", () => {{
  const item = ann();
  item.speech_segments = [];
  item.skip_reason = "";
  item.reviewed = true;
  persist();
  renderSegments();
  renderNotes();
}});
document.getElementById("reviewedBtn").addEventListener("click", () => {{
  commitPendingStart();
  ann().reviewed = true;
  persist();
}});
document.getElementById("addEmptyBtn").addEventListener("click", () => addSegment(audio.currentTime || 0, Math.min(CANDIDATES[currentIndex].duration_s, (audio.currentTime || 0) + 0.2)));
document.getElementById("clearBtn").addEventListener("click", () => setSegments([], false));
document.getElementById("saveBtn").addEventListener("click", () => saveJsonl().catch(err => alert(err.message || err)));
document.getElementById("downloadBtn").addEventListener("click", downloadJsonl);
document.getElementById("copyBtn").addEventListener("click", () => copyJsonl().catch(err => alert(err.message || err)));
document.getElementById("skipReason").addEventListener("input", event => {{
  const item = ann();
  item.skip_reason = event.target.value;
  item.reviewed = Boolean(item.skip_reason) || item.reviewed;
  persist();
}});
document.getElementById("notes").addEventListener("input", event => {{
  ann().notes = event.target.value;
  persist();
}});
audio.addEventListener("timeupdate", drawWaveform);
wave.addEventListener("click", event => {{
  const candidate = CANDIDATES[currentIndex];
  const rect = wave.getBoundingClientRect();
  const ratio = (event.clientX - rect.left) / rect.width;
  audio.currentTime = clampTime(ratio * candidate.duration_s, candidate.duration_s);
  drawWaveform();
}});
document.addEventListener("keydown", event => {{
  if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  if (event.code === "Space") {{ event.preventDefault(); audio.paused ? audio.play() : audio.pause(); }}
  if (event.key === "[") {{ pendingStart = audio.currentTime || 0; drawWaveform(); }}
  if (event.key === "]") {{ document.getElementById("endBtn").click(); }}
  if (event.key.toLowerCase() === "a") {{ document.getElementById("allSpeechBtn").click(); }}
  if (event.key.toLowerCase() === "n") {{ document.getElementById("nonSpeechBtn").click(); }}
  if (event.key.toLowerCase() === "b") {{ document.getElementById("startToHereBtn").click(); }}
  if (event.key.toLowerCase() === "e") {{ document.getElementById("hereToEndBtn").click(); }}
  if (event.key.toLowerCase() === "j") {{ loadCandidate(currentIndex - 1); }}
  if (event.key.toLowerCase() === "k") {{ loadCandidate(currentIndex + 1); }}
  if (event.key === "ArrowLeft") {{ audio.currentTime = Math.max(0, (audio.currentTime || 0) - 0.05); }}
  if (event.key === "ArrowRight") {{ audio.currentTime = Math.min(CANDIDATES[currentIndex].duration_s, (audio.currentTime || 0) + 0.05); }}
}});

renderReasonFilter();
renderList();
loadCandidate(0);
</script>
</body>
</html>
"""


def run(args: argparse.Namespace) -> None:
    candidates_path = Path(args.candidates)
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    rows = read_candidates(candidates_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    candidates = [
        normalize_candidate(row, output_dir=output_html.parent, copy_audio=args.copy_audio)
        for row in rows
    ]
    dataset_id = args.dataset_id or str(candidates_path.resolve())
    html = html_template(
        title=args.title,
        dataset_id=dataset_id,
        output_jsonl_name=args.output_jsonl_name,
        candidates=candidates,
    )
    output_html.write_text(html, encoding="utf-8")
    summary = {
        "candidates": str(candidates_path),
        "candidate_count": len(candidates),
        "copy_audio": args.copy_audio,
        "output_html": str(output_html),
        "output_jsonl_name": args.output_jsonl_name,
    }
    summary_path = output_html.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"html={output_html}")
    print(f"summary={summary_path}")
    print(f"candidate_count={len(candidates)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate standalone HTML for manual FusionVAD-JA audit labeling.")
    parser.add_argument("--candidates", required=True, help="audit_candidates.jsonl or audit_candidates.csv.")
    parser.add_argument(
        "--output-html",
        default=str(PROJECT_ROOT / "datasets" / "val" / "fusionvad-ja" / "v1-3" / "manual-audit-galgame" / "manual_audit.html"),
    )
    parser.add_argument("--output-jsonl-name", default="manual_labels.jsonl")
    parser.add_argument("--title", default="FusionVAD-JA 人工审计标注")
    parser.add_argument("--dataset-id", help="Stable browser localStorage key. Defaults to candidates absolute path.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--copy-audio", action="store_true", help="Copy candidate audio next to the HTML for reliable local playback.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
