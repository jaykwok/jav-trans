#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)


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


def rel_url(path: Path, *, from_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), from_dir.resolve())).as_posix()
    except ValueError:
        return path.as_posix()


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def row_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def compact_text(value: Any, *, max_chars: int = 520) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1] + "..."
    return text


def parse_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(".")
    millis = int((millis + "000")[:3])
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + millis / 1000.0


def srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    millis = int(round((seconds - int(seconds)) * 1000))
    whole = int(seconds)
    if millis >= 1000:
        whole += 1
        millis -= 1000
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def clean_cue_text(value: Any) -> str:
    return str(value or "").replace("\r", "").strip() or "..."


def read_srt_cues(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    cues: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        match: re.Match[str] | None = None
        time_line_index = -1
        for index, line in enumerate(lines[:3]):
            match = TIME_RE.search(line)
            if match:
                time_line_index = index
                break
        if not match or time_line_index < 0:
            continue
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        if end <= start:
            continue
        cues.append(
            {
                "index": len(cues),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": clean_cue_text("\n".join(lines[time_line_index + 1 :])),
            }
        )
    return cues


def write_vtt(path: Path, cues: list[dict[str, Any]]) -> None:
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.extend(
            [
                f"{vtt_timestamp(cue['start'])} --> {vtt_timestamp(cue['end'])}",
                clean_cue_text(cue.get("text")),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return min(a_end, b_end) - max(a_start, b_start) > 0.01


def cues_for_range(cues: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for cue in cues:
        cue_start = row_float(cue, "start")
        cue_end = row_float(cue, "end")
        if overlaps(start, end, cue_start, cue_end):
            selected.append(cue)
    return selected


def cue_text_for_range(cues: list[dict[str, Any]], start: float, end: float) -> str:
    return compact_text("\n".join(clean_cue_text(cue.get("text")) for cue in cues_for_range(cues, start, end)))


def build_review_items(
    *,
    long_chunk_rows: list[dict[str, Any]],
    subtitle_cues: list[dict[str, Any]],
    context_pad_s: float,
    max_items: int | None,
) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        long_chunk_rows,
        key=lambda row: (
            -row_float(row, "duration_s"),
            row_float(row, "start"),
            int(row.get("chunk_index") or 0),
        ),
    )
    items: list[dict[str, Any]] = []
    for row in sorted_rows:
        start = row_float(row, "start")
        end = row_float(row, "end")
        if end <= start:
            continue
        context_start = max(0.0, start - context_pad_s)
        context_end = end + context_pad_s
        cue_rows = cues_for_range(subtitle_cues, start, end)
        items.append(
            {
                "index": len(items),
                "sample_id": f"r21-long-fallback-{len(items):03d}-chunk{int(row.get('chunk_index') or 0):04d}",
                "video_label": "匿名样片 A",
                "chunk_index": int(row.get("chunk_index") or 0),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_s": round(row_float(row, "duration_s") or (end - start), 3),
                "context_start": round(context_start, 3),
                "context_end": round(context_end, 3),
                "core_start": round(row_float(row, "core_start"), 3),
                "core_end": round(row_float(row, "core_end"), 3),
                "core_duration_s": round(row_float(row, "core_duration_s"), 3),
                "split_reason": str(row.get("split_reason") or ""),
                "alignment_quality": str(row.get("alignment_quality") or ""),
                "fallback_subtype": str(row.get("fallback_subtype") or row.get("fallback_reason") or ""),
                "failure_bucket": str(row.get("failure_bucket") or ""),
                "speech_island_count": int(row.get("speech_island_count") or 0),
                "internal_gap_count": int(row.get("internal_gap_count") or 0),
                "internal_gap_max_s": round(row_float(row, "internal_gap_max_s"), 3),
                "internal_gap_total_s": round(row_float(row, "internal_gap_total_s"), 3),
                "longest_silence_s": round(row_float(row, "longest_silence_s"), 3),
                "total_silence_s": round(row_float(row, "total_silence_s"), 3),
                "silence_ratio": row.get("silence_ratio"),
                "risk_reasons": list(row.get("risk_reasons") or []),
                "display_text": compact_text(row.get("display_text") or ""),
                "subtitle_text": cue_text_for_range(subtitle_cues, start, end),
                "subtitle_cues": [
                    {
                        "start": cue["start"],
                        "end": cue["end"],
                        "text": cue["text"],
                    }
                    for cue in cue_rows
                ],
            }
        )
        if max_items is not None and len(items) >= max_items:
            break
    return items


def write_latest_audit_redirect(*, audit_root: Path, latest_html: Path, title: str) -> None:
    latest_rel = rel_url(latest_html, from_dir=audit_root)
    (audit_root / "latest-audit.html").write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={html.escape(latest_rel)}">
<title>跳转到最新审计页</title>
</head>
<body>
<p>正在跳转到 <a href="{html.escape(latest_rel)}">{html.escape(title)}</a>。</p>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_audit_index(*, audit_root: Path, latest_html: Path, title: str, summary: Mapping[str, Any]) -> None:
    latest_rel = rel_url(latest_html, from_dir=audit_root)
    entries: list[tuple[str, str, str]] = [
        (latest_rel, "最新审计页", title),
    ]
    for index_path in sorted(audit_root.glob("*/index.html")):
        rel = rel_url(index_path, from_dir=audit_root)
        if rel == latest_rel:
            continue
        desc = "审计页面"
        summary_path = index_path.parent / "summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                parts = []
                count = payload.get("review_item_count")
                if count is not None:
                    parts.append(f"{count} 条")
                if payload.get("video_label"):
                    parts.append(str(payload["video_label"]))
                elif payload.get("video"):
                    parts.append("视频审计")
                if parts:
                    desc = " · ".join(parts)
            except Exception:
                pass
        entries.append((rel, index_path.parent.name, desc))

    cards = "\n".join(
        f'  <a class="entry" href="{html.escape(href)}">\n'
        f"    <strong>{html.escape(label)}</strong>\n"
        f"    {html.escape(desc)}\n"
        "  </a>"
        for href, label, desc in entries
    )
    latest_meta = (
        f"当前 latest: {project_rel(latest_html)}；"
        f"long_chunks={summary.get('review_item_count', 0)}；"
        f"subtitle_cues={summary.get('subtitle_cue_count', 0)}"
    )
    (audit_root / "index.html").write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FusionVAD-JA 审计入口</title>
<style>
body {{
  margin: 0;
  background: #f5f6f4;
  color: #1d2421;
  font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}}
main {{ max-width: 920px; margin: 0 auto; padding: 32px 18px; }}
h1 {{ margin: 0 0 18px; font-size: 24px; }}
.entry {{
  display: block;
  margin: 12px 0;
  padding: 14px 16px;
  border: 1px solid #d8ddd8;
  border-radius: 8px;
  background: #fff;
  color: inherit;
  text-decoration: none;
}}
.entry strong {{ display: block; margin-bottom: 4px; color: #0f766e; }}
.muted {{ color: #66706c; font-size: 13px; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
</head>
<body>
<main>
  <h1>FusionVAD-JA 审计入口</h1>
{cards}
  <p class="muted">{html.escape(latest_meta)}</p>
  <p class="muted">后续人工审计统一放在 <code>agents/audits/</code>。</p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def maybe_update_audit_entrypoints(*, output_dir: Path, title: str, summary: Mapping[str, Any]) -> None:
    audit_root = PROJECT_ROOT / "agents" / "audits" / "fusionvad-ja"
    try:
        output_dir.resolve().relative_to(audit_root.resolve())
    except ValueError:
        return
    write_latest_audit_redirect(audit_root=audit_root, latest_html=output_dir / "index.html", title=title)
    write_audit_index(audit_root=audit_root, latest_html=output_dir / "index.html", title=title, summary=summary)


def page_template(
    *,
    title: str,
    video_url: str,
    vtt_url: str,
    items: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    items_json = json.dumps(items, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    cues_json = json.dumps(cues, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    summary_json = json.dumps(summary, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f4f6f5;
  --panel: #ffffff;
  --ink: #1d2421;
  --muted: #66706c;
  --line: #d8ddd8;
  --accent: #0f766e;
  --accent-soft: #dff2ee;
  --danger: #b42318;
  --warn: #9a6500;
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
button.active {{ background: var(--accent-soft); border-color: var(--accent); }}
.app {{ display: grid; grid-template-columns: 410px minmax(0, 1fr); min-height: 100vh; }}
.sidebar {{ max-height: 100vh; overflow: auto; border-right: 1px solid var(--line); background: #fbfbf9; }}
.side-head {{ position: sticky; top: 0; z-index: 2; padding: 12px; border-bottom: 1px solid var(--line); background: #fbfbf9; }}
h1 {{ margin: 0 0 8px; font-size: 16px; }}
h2 {{ margin: 0; font-size: 18px; overflow-wrap: anywhere; }}
h3 {{ margin: 0 0 10px; font-size: 14px; }}
.filters {{ display: grid; grid-template-columns: 1fr 128px; gap: 8px; margin-top: 8px; }}
.filters input, .filters select {{
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 8px;
  background: #fff;
}}
.item {{ display: grid; gap: 4px; padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }}
.item:hover {{ background: #eef3ef; }}
.item.active {{ background: var(--accent-soft); }}
.item-title {{ font-weight: 650; overflow-wrap: anywhere; }}
.meta, .hint {{ color: var(--muted); font-size: 12px; }}
.badge {{
  display: inline-block;
  border-radius: 999px;
  padding: 2px 7px;
  font-size: 12px;
  border: 1px solid var(--line);
  background: #fff;
}}
.badge.warn {{ color: var(--warn); border-color: #e4c27f; background: #fff7df; }}
.badge.danger {{ color: var(--danger); border-color: #efb5af; background: #fff0ee; }}
.workspace {{ max-height: 100vh; overflow: auto; padding: 16px; }}
.topbar {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 12px; }}
.actions, .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.actions {{ justify-content: flex-end; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 12px; }}
.video-shell {{ position: relative; background: #070a09; border-radius: 8px; overflow: hidden; }}
video {{ display: block; width: 100%; max-height: 66vh; background: #070a09; }}
.caption-overlay {{
  position: absolute;
  left: 8%;
  right: 8%;
  bottom: 20px;
  max-height: 24%;
  overflow: auto;
  padding: 6px 10px;
  border-radius: 6px;
  background: rgba(0, 0, 0, 0.68);
  color: #fff;
  text-align: center;
  font-size: clamp(13px, 1.25vw, 18px);
  line-height: 1.3;
  text-shadow: 0 1px 2px #000;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.caption-overlay:empty {{ display: none; }}
.timeline {{ position: relative; height: 48px; border: 1px solid var(--line); border-radius: 6px; background: #101715; overflow: hidden; cursor: pointer; margin-top: 10px; }}
.fallback-range {{ position: absolute; top: 0; bottom: 0; background: rgba(15, 118, 110, 0.55); }}
.core-range {{ position: absolute; top: 10px; bottom: 10px; background: rgba(255, 255, 255, 0.22); border-top: 1px solid rgba(255,255,255,0.6); border-bottom: 1px solid rgba(255,255,255,0.6); }}
.cursor {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; }}
.timeline-labels {{ display: flex; justify-content: space-between; margin-top: 4px; color: var(--muted); font-size: 12px; }}
.kv {{ display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 6px 10px; font-size: 13px; }}
.kv div:nth-child(odd) {{ color: var(--muted); }}
.text-box {{ white-space: pre-wrap; overflow-wrap: anywhere; border: 1px solid var(--line); border-radius: 6px; background: #fbfbf9; padding: 9px; min-height: 44px; max-height: 190px; overflow: auto; }}
.cue-list {{ display: grid; gap: 8px; max-height: 240px; overflow: auto; }}
.cue {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fbfbf9; }}
.label-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
textarea {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px; min-height: 70px; resize: vertical; }}
.error {{ color: var(--danger); }}
@media (max-width: 1080px) {{
  .app {{ grid-template-columns: 1fr; }}
  .sidebar {{ max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--line); }}
  .workspace {{ max-height: none; }}
  .grid, .label-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="side-head">
      <h1>{html.escape(title)}</h1>
      <div class="meta" id="summaryLine"></div>
      <div class="filters">
        <input id="searchInput" placeholder="搜索 chunk / 文本 / reason">
        <select id="sortSelect">
          <option value="duration">按时长</option>
          <option value="time">按时间</option>
          <option value="islands">按 island 数</option>
        </select>
      </div>
    </div>
    <div id="itemList"></div>
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
        <button class="primary" id="playChunkBtn">播放长 chunk</button>
        <button id="playContextBtn">播放上下文</button>
        <button id="downloadBtn">下载审计 JSONL</button>
      </div>
    </div>
    <div class="grid">
      <section class="panel">
        <div class="video-shell">
          <video id="video" controls preload="metadata" playsinline>
            <source src="{html.escape(video_url)}" type="video/mp4">
            <track kind="subtitles" label="完整日语字幕" srclang="ja" src="{html.escape(vtt_url)}" default>
          </video>
          <div class="caption-overlay" id="captionOverlay"></div>
        </div>
        <div class="timeline" id="timeline">
          <div class="fallback-range" id="fallbackRange"></div>
          <div class="core-range" id="coreRange"></div>
          <div class="cursor" id="cursor"></div>
        </div>
        <div class="timeline-labels">
          <span id="rangeStart"></span>
          <span id="nowText"></span>
          <span id="rangeEnd"></span>
        </div>
        <div class="toolbar" style="margin-top:10px">
          <button id="pauseBtn">暂停</button>
          <button id="replayBtn">重播当前模式</button>
          <a href="{html.escape(video_url)}">直接打开视频</a>
          <a href="{html.escape(vtt_url)}">完整 VTT</a>
        </div>
        <p class="hint">绿色是 fallback chunk，浅色内条是 core 区间；播放会在当前模式终点自动暂停。画面字幕为完整日语字幕当前 cue。</p>
        <p class="hint error" id="videoError"></p>
      </section>
      <aside class="panel">
        <h3>长 chunk 指标</h3>
        <div class="kv" id="metrics"></div>
      </aside>
    </div>
    <section class="panel">
      <h3>完整字幕在该 chunk 内的 cue</h3>
      <div class="cue-list" id="cueList"></div>
    </section>
    <section class="panel">
      <h3>chunk ASR 显示文本</h3>
      <div class="text-box" id="displayText"></div>
    </section>
    <section class="panel">
      <h3>人工标签</h3>
      <div class="label-grid" id="labelButtons"></div>
      <label class="hint" for="notes">备注</label>
      <textarea id="notes"></textarea>
    </section>
  </main>
</div>
<script type="application/json" id="items-json">{items_json}</script>
<script type="application/json" id="cues-json">{cues_json}</script>
<script type="application/json" id="summary-json">{summary_json}</script>
<script>
const ITEMS = JSON.parse(document.getElementById("items-json").textContent);
const CUES = JSON.parse(document.getElementById("cues-json").textContent);
const SUMMARY = JSON.parse(document.getElementById("summary-json").textContent);
const STORAGE_KEY = "fusionvad-ja-long-fallback-audit:" + SUMMARY.dataset_id;
const LABELS = [
  ["needs_split", "需要切分"],
  ["ok_long_utterance", "确实长句"],
  ["multiple_islands", "多段语音揉一起"],
  ["gap_or_noise", "中间噪音/无声"],
  ["non_speech", "无有效语音"],
  ["bad_asr", "ASR 文本异常"],
  ["uncertain", "不确定"]
];
let currentIndex = 0;
let filtered = [...ITEMS];
let playMode = "chunk";
let annotations = loadAnnotations();
const video = document.getElementById("video");
const itemList = document.getElementById("itemList");
const searchInput = document.getElementById("searchInput");
const sortSelect = document.getElementById("sortSelect");
const notes = document.getElementById("notes");
const timeline = document.getElementById("timeline");
const fallbackRange = document.getElementById("fallbackRange");
const coreRange = document.getElementById("coreRange");
const cursor = document.getElementById("cursor");

function loadAnnotations() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
  catch (_) {{ return {{}}; }}
}}
function saveAnnotations() {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
}}
function escapeHtml(text) {{
  return String(text || "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}
function fmt(t) {{
  const v = Math.max(0, Number(t || 0));
  const h = Math.floor(v / 3600);
  const m = Math.floor((v - h * 3600) / 60);
  const s = (v - h * 3600 - m * 60).toFixed(2).padStart(5, "0");
  return h ? `${{h}}:${{String(m).padStart(2, "0")}}:${{s}}` : `${{m}}:${{s}}`;
}}
function activeCueText(t) {{
  return CUES
    .filter(cue => Number(cue.start) <= t && t <= Number(cue.end))
    .map(cue => cue.text)
    .join("\\n");
}}
function itemSearchText(item) {{
  return [
    item.sample_id,
    item.chunk_index,
    item.split_reason,
    item.fallback_subtype,
    item.risk_reasons.join(","),
    item.display_text,
    item.subtitle_text
  ].join("\\n").toLowerCase();
}}
function sortItems(rows) {{
  const mode = sortSelect.value;
  const cloned = [...rows];
  if (mode === "time") cloned.sort((a, b) => a.start - b.start);
  else if (mode === "islands") cloned.sort((a, b) => (b.speech_island_count - a.speech_island_count) || (b.duration_s - a.duration_s));
  else cloned.sort((a, b) => (b.duration_s - a.duration_s) || (a.start - b.start));
  return cloned;
}}
function applyFilters() {{
  const query = searchInput.value.trim().toLowerCase();
  filtered = sortItems(ITEMS.filter(item => !query || itemSearchText(item).includes(query)));
  if (!filtered.includes(ITEMS[currentIndex])) {{
    currentIndex = ITEMS.indexOf(filtered[0] || ITEMS[0] || null);
    if (currentIndex < 0) currentIndex = 0;
  }}
  renderList();
  renderCurrent(false);
}}
function renderList() {{
  itemList.innerHTML = "";
  for (const item of filtered) {{
    const div = document.createElement("div");
    div.className = "item" + (ITEMS[currentIndex] === item ? " active" : "");
    div.onclick = () => {{
      currentIndex = ITEMS.indexOf(item);
      renderCurrent(true);
      renderList();
    }};
    div.innerHTML = `
      <div class="item-title">#${{item.index + 1}} chunk ${{item.chunk_index}} <span class="badge danger">${{Number(item.duration_s).toFixed(2)}}s</span></div>
      <div class="meta">${{fmt(item.start)}}-${{fmt(item.end)}} · island=${{item.speech_island_count}} · gap=${{item.internal_gap_count}} · ${{escapeHtml(item.split_reason)}}</div>
      <div class="meta">${{escapeHtml(item.subtitle_text || item.display_text || "(empty)")}}</div>
    `;
    itemList.appendChild(div);
  }}
}}
function setMetrics(item) {{
  const rows = [
    ["视频", item.video_label],
    ["chunk", item.chunk_index],
    ["chunk 时间", `${{fmt(item.start)}}-${{fmt(item.end)}}`],
    ["chunk 时长", `${{Number(item.duration_s).toFixed(2)}}s`],
    ["core", `${{fmt(item.core_start)}}-${{fmt(item.core_end)}} (${{Number(item.core_duration_s).toFixed(2)}}s)`],
    ["split_reason", item.split_reason],
    ["fallback", item.fallback_subtype],
    ["speech_islands", item.speech_island_count],
    ["internal_gap", `${{item.internal_gap_count}} / max ${{Number(item.internal_gap_max_s).toFixed(2)}}s / total ${{Number(item.internal_gap_total_s).toFixed(2)}}s`],
    ["silence", `longest ${{Number(item.longest_silence_s).toFixed(2)}}s / total ${{Number(item.total_silence_s).toFixed(2)}}s`],
    ["risk", (item.risk_reasons || []).join(", ")]
  ];
  document.getElementById("metrics").innerHTML = rows
    .map(([key, value]) => `<div>${{escapeHtml(key)}}</div><div>${{escapeHtml(value)}}</div>`)
    .join("");
}}
function renderCueList(item) {{
  const root = document.getElementById("cueList");
  if (!item.subtitle_cues || item.subtitle_cues.length === 0) {{
    root.innerHTML = `<div class="hint">该 chunk 内没有重叠字幕 cue。</div>`;
    return;
  }}
  root.innerHTML = item.subtitle_cues.map(cue => `
    <div class="cue">
      <div class="meta">${{fmt(cue.start)}}-${{fmt(cue.end)}}</div>
      <div>${{escapeHtml(cue.text)}}</div>
    </div>
  `).join("");
}}
function renderLabels(item) {{
  const root = document.getElementById("labelButtons");
  root.innerHTML = "";
  const ann = annotations[item.sample_id] || {{}};
  for (const [value, label] of LABELS) {{
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.className = ann.label === value ? "active" : "";
    btn.onclick = () => {{
      annotations[item.sample_id] = {{...(annotations[item.sample_id] || {{}}), label, updated_at: new Date().toISOString()}};
      saveAnnotations();
      renderLabels(item);
    }};
    root.appendChild(btn);
  }}
  notes.value = ann.notes || "";
}}
function updateTimeline() {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  const width = Math.max(0.001, item.context_end - item.context_start);
  const setRange = (el, start, end, minWidth) => {{
    const left = Math.max(0, Math.min(100, ((start - item.context_start) / width) * 100));
    const right = Math.max(0, Math.min(100, ((end - item.context_start) / width) * 100));
    el.style.left = `${{left}}%`;
    el.style.width = `${{Math.max(minWidth, right - left)}}%`;
  }};
  setRange(fallbackRange, item.start, item.end, 0.5);
  setRange(coreRange, item.core_start || item.start, item.core_end || item.end, 0.5);
  const t = video.currentTime || 0;
  const pct = Math.max(0, Math.min(100, ((t - item.context_start) / width) * 100));
  cursor.style.left = `${{pct}}%`;
  document.getElementById("nowText").textContent = fmt(t);
  document.getElementById("captionOverlay").textContent = activeCueText(t);
  const stopAt = playMode === "context" ? item.context_end : item.end;
  if (!video.paused && t >= stopAt) video.pause();
}}
function renderCurrent(seek) {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  document.getElementById("clipTitle").textContent = `#${{item.index + 1}} 长 fallback chunk ${{item.chunk_index}}`;
  document.getElementById("clipMeta").textContent = `${{fmt(item.start)}}-${{fmt(item.end)}} · ${{Number(item.duration_s).toFixed(2)}}s · ${{item.fallback_subtype || "fallback"}}`;
  document.getElementById("rangeStart").textContent = fmt(item.context_start);
  document.getElementById("rangeEnd").textContent = fmt(item.context_end);
  document.getElementById("displayText").textContent = item.display_text || "(empty)";
  setMetrics(item);
  renderCueList(item);
  renderLabels(item);
  if (seek) video.currentTime = item.start;
  updateTimeline();
}}
function playCurrent(mode) {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  playMode = mode;
  video.currentTime = mode === "context" ? item.context_start : item.start;
  video.play();
}}
function exportRows() {{
  return ITEMS.map(item => ({{
    sample_id: item.sample_id,
    video_label: item.video_label,
    chunk_index: item.chunk_index,
    start: item.start,
    end: item.end,
    duration_s: item.duration_s,
    speech_island_count: item.speech_island_count,
    internal_gap_count: item.internal_gap_count,
    display_text: item.display_text,
    subtitle_text: item.subtitle_text,
    ...(annotations[item.sample_id] || {{}})
  }}));
}}
function downloadJsonl() {{
  const text = exportRows().map(row => JSON.stringify(row)).join("\\n") + "\\n";
  const blob = new Blob([text], {{type: "application/jsonl;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "manual_long_fallback_chunk_labels.jsonl";
  a.click();
  URL.revokeObjectURL(a.href);
}}
notes.addEventListener("input", () => {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  annotations[item.sample_id] = {{...(annotations[item.sample_id] || {{}}), notes: notes.value, updated_at: new Date().toISOString()}};
  saveAnnotations();
}});
video.addEventListener("timeupdate", updateTimeline);
video.addEventListener("seeked", updateTimeline);
video.addEventListener("error", () => {{
  document.getElementById("videoError").textContent = "视频无法加载。请确认 live-server 从项目根目录启动，且 video 文件存在。";
}});
timeline.addEventListener("click", event => {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  const rect = timeline.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  video.currentTime = item.context_start + ratio * (item.context_end - item.context_start);
}});
document.getElementById("playChunkBtn").onclick = () => playCurrent("chunk");
document.getElementById("playContextBtn").onclick = () => playCurrent("context");
document.getElementById("pauseBtn").onclick = () => video.pause();
document.getElementById("replayBtn").onclick = () => playCurrent(playMode);
document.getElementById("downloadBtn").onclick = downloadJsonl;
document.getElementById("prevBtn").onclick = () => {{
  const pos = filtered.indexOf(ITEMS[currentIndex]);
  if (pos > 0) currentIndex = ITEMS.indexOf(filtered[pos - 1]);
  renderCurrent(true);
  renderList();
}};
document.getElementById("nextBtn").onclick = () => {{
  const pos = filtered.indexOf(ITEMS[currentIndex]);
  if (pos >= 0 && pos < filtered.length - 1) currentIndex = ITEMS.indexOf(filtered[pos + 1]);
  renderCurrent(true);
  renderList();
}};
searchInput.addEventListener("input", applyFilters);
sortSelect.addEventListener("change", applyFilters);
document.getElementById("summaryLine").textContent = `${{ITEMS.length}} 条 20-30s 长 fallback · 完整字幕 ${{CUES.length}} cues`;
try {{
  for (const track of video.textTracks) track.mode = "hidden";
}} catch (_) {{}}
applyFilters();
renderCurrent(true);
</script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a video audit page for long unsafe fallback chunks."
    )
    parser.add_argument("--long-chunks", required=True, help="unsafe_fallback_chunks.jsonl")
    parser.add_argument("--subtitle-srt", required=True, help="Full Japanese SRT")
    parser.add_argument("--video", required=True, help="Original video path")
    parser.add_argument(
        "--output-dir",
        default="agents/audits/fusionvad-ja/long-fallback-r21-dropgap512-th080",
    )
    parser.add_argument("--title", default="FusionVAD-JA 长 chunk 审计 · R21")
    parser.add_argument("--context-pad-s", type=float, default=2.0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--update-entrypoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.context_pad_s < 0:
        parser.error("--context-pad-s must be non-negative")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    long_chunks_path = project_path(args.long_chunks)
    subtitle_srt_path = project_path(args.subtitle_srt)
    video_path = project_path(args.video)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_chunk_rows = read_jsonl(long_chunks_path)
    subtitle_cues = read_srt_cues(subtitle_srt_path)
    if not long_chunk_rows:
        raise SystemExit(f"No long chunk rows found: {long_chunks_path}")
    if not subtitle_cues:
        raise SystemExit(f"No subtitle cues parsed: {subtitle_srt_path}")

    items = build_review_items(
        long_chunk_rows=long_chunk_rows,
        subtitle_cues=subtitle_cues,
        context_pad_s=args.context_pad_s,
        max_items=args.max_items,
    )
    if not items:
        raise SystemExit("No review items produced")

    full_vtt = output_dir / "full.ja.vtt"
    write_vtt(full_vtt, subtitle_cues)
    write_jsonl(output_dir / "long_fallback_review_items.jsonl", items)
    summary = {
        "dataset_id": output_dir.name,
        "title": args.title,
        "video_label": "匿名样片 A",
        "video_path": project_rel(video_path),
        "long_chunks": project_rel(long_chunks_path),
        "subtitle_srt_source": project_rel(subtitle_srt_path),
        "html": project_rel(output_dir / "index.html"),
        "full_vtt": project_rel(full_vtt),
        "review_item_count": len(items),
        "subtitle_cue_count": len(subtitle_cues),
        "duration_p50_s": sorted(item["duration_s"] for item in items)[len(items) // 2],
        "duration_max_s": max(item["duration_s"] for item in items),
    }
    write_json(output_dir / "summary.json", summary)
    html_text = page_template(
        title=args.title,
        video_url=rel_url(video_path, from_dir=output_dir),
        vtt_url=rel_url(full_vtt, from_dir=output_dir),
        items=items,
        cues=subtitle_cues,
        summary=summary,
    )
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    if args.update_entrypoints:
        maybe_update_audit_entrypoints(output_dir=output_dir, title=args.title, summary=summary)
    print(f"html={project_rel(output_dir / 'index.html')}")
    print(f"items={len(items)}")
    print(f"summary={project_rel(output_dir / 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
