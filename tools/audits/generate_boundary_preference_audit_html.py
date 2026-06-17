#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAYBACK_PREROLL_S = 1.5
PLAYBACK_POSTROLL_S = 1.5
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.boundary.boundary_preference import (  # noqa: E402
    BLIND_ITEM_SCHEMA,
    LABEL_SCHEMA,
    read_jsonl,
)


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path) -> str:
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


def _page_html(*, title: str, dataset_id: str, items: list[dict[str, Any]]) -> str:
    data_json = json.dumps(items, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f3f5f4;
  --panel: #fff;
  --ink: #202624;
  --muted: #68716d;
  --line: #d8deda;
  --accent: #176b62;
  --accent-soft: #def1ed;
  --a: #a65312;
  --b: #176b62;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}}
button, textarea {{ font: inherit; }}
button {{
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 8px 11px;
  background: #fff;
  color: var(--ink);
  cursor: pointer;
}}
button.active {{ border-color: var(--accent); background: var(--accent-soft); }}
button.primary {{ border-color: var(--accent); background: var(--accent); color: #fff; }}
.app {{ display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: 100vh; }}
.sidebar {{ max-height: 100vh; overflow: auto; border-right: 1px solid var(--line); background: #fafbf9; }}
.side-head {{ position: sticky; top: 0; z-index: 3; padding: 13px; border-bottom: 1px solid var(--line); background: #fafbf9; }}
h1 {{ margin: 0 0 6px; font-size: 17px; }}
h2 {{ margin: 0; font-size: 19px; }}
h3 {{ margin: 0 0 9px; font-size: 14px; }}
.meta, .hint {{ color: var(--muted); font-size: 12px; }}
.item {{ padding: 9px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }}
.item:hover {{ background: #edf2ef; }}
.item.active {{ background: var(--accent-soft); }}
.item.done .item-title::after {{ content: " · 已标"; color: var(--accent); }}
.item-title {{ font-weight: 650; }}
.workspace {{ max-height: 100vh; overflow: auto; padding: 16px; }}
.topbar {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 12px; }}
.actions, .button-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.panel {{ margin-bottom: 12px; padding: 12px; border: 1px solid var(--line); border-radius: 9px; background: var(--panel); }}
.media-shell {{ position: relative; border-radius: 8px; overflow: hidden; background: #060908; }}
video {{ display: block; width: 100%; max-height: 61vh; background: #060908; }}
.overlay {{
  position: absolute;
  left: 7%;
  right: 7%;
  bottom: 18px;
  padding: 7px 10px;
  border-radius: 6px;
  background: rgba(0,0,0,.68);
  color: #fff;
  text-align: center;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.timeline {{ position: relative; height: 44px; margin-top: 10px; border-radius: 7px; background: #edf0ed; cursor: pointer; }}
.range {{ position: absolute; top: 7px; height: 12px; border-radius: 5px; opacity: .78; }}
.range.a {{ background: var(--a); }}
.range.b {{ top: 25px; background: var(--b); }}
.cursor {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #111; }}
.compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.choice {{ border: 2px solid var(--line); border-radius: 8px; padding: 11px; background: #fbfcfa; }}
.choice.a {{ border-color: color-mix(in srgb, var(--a) 55%, white); }}
.choice.b {{ border-color: color-mix(in srgb, var(--b) 55%, white); }}
.choice h3 {{ display: flex; justify-content: space-between; gap: 8px; }}
.text {{ min-height: 70px; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 16px; }}
.metrics {{ display: grid; grid-template-columns: 145px 1fr; gap: 5px 10px; margin-top: 10px; font-size: 12px; }}
.labels {{ display: flex; flex-wrap: wrap; gap: 7px; }}
textarea {{ width: 100%; min-height: 75px; resize: vertical; border: 1px solid var(--line); border-radius: 7px; padding: 8px; }}
.progress {{ height: 7px; margin-top: 8px; border-radius: 5px; background: #e2e7e3; overflow: hidden; }}
.progress > div {{ height: 100%; background: var(--accent); }}
@media (max-width: 900px) {{
  .app {{ grid-template-columns: 1fr; }}
  .sidebar {{ max-height: 34vh; border-right: 0; border-bottom: 1px solid var(--line); }}
  .workspace {{ max-height: none; }}
  .compare {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="side-head">
      <h1>{html.escape(title)}</h1>
      <div class="meta" id="progressText"></div>
      <div class="progress"><div id="progressBar"></div></div>
    </div>
    <div id="itemList"></div>
  </aside>
  <main class="workspace">
    <div class="topbar">
      <div>
        <h2 id="itemTitle"></h2>
        <div class="meta" id="itemMeta"></div>
      </div>
      <div class="actions">
        <button id="prevBtn">上一条</button>
        <button id="nextBtn">下一条</button>
        <button class="primary" id="downloadBtn">导出 JSONL</button>
      </div>
    </div>
    <section class="panel">
      <div class="media-shell">
        <video id="media" controls preload="metadata" playsinline></video>
        <div class="overlay" id="overlay"></div>
      </div>
      <div class="timeline" id="timeline">
        <div class="range a" id="rangeA"></div>
        <div class="range b" id="rangeB"></div>
        <div class="cursor" id="cursor"></div>
      </div>
      <div class="button-row" style="margin-top:10px">
        <button id="playABtn">播放 A</button>
        <button id="playBBtn">播放 B</button>
        <button id="playContextBtn">播放上下文</button>
        <button id="pauseBtn">暂停</button>
      </div>
      <div class="hint">A/B 已随机化。请比较语音起止、是否截字/吞字、是否跨入相邻噪声或台词，以及转写是否更完整稳定。</div>
    </section>
    <section class="compare">
      <div class="choice a">
        <h3><span>A</span><span id="timeA" class="meta"></span></h3>
        <div class="text" id="textA"></div>
        <div class="metrics" id="metricsA"></div>
      </div>
      <div class="choice b">
        <h3><span>B</span><span id="timeB" class="meta"></span></h3>
        <div class="text" id="textB"></div>
        <div class="metrics" id="metricsB"></div>
      </div>
    </section>
    <section class="panel">
      <h3>主标签（单选，快捷键 1-5）</h3>
      <div class="labels" id="primaryLabels"></div>
    </section>
    <section class="panel">
      <h3>Timing 原因（多选）</h3>
      <div class="labels" id="timingLabels"></div>
    </section>
    <section class="panel">
      <h3>备注</h3>
      <textarea id="notes" placeholder="只记录必要观察，例如某个音节被截断、尾音拖入下一句。"></textarea>
    </section>
  </main>
</div>
<script id="dataset" type="application/json">{data_json}</script>
<script>
const ITEMS = JSON.parse(document.getElementById("dataset").textContent);
const DATASET_ID = {json.dumps(dataset_id)};
const STORAGE_KEY = `boundary-preference:${{DATASET_ID}}`;
const PLAYBACK_PREROLL_S = {PLAYBACK_PREROLL_S};
const PLAYBACK_POSTROLL_S = {PLAYBACK_POSTROLL_S};
const PRIMARY = [
  ["a_better", "A better"],
  ["b_better", "B better"],
  ["tie", "tie"],
  ["both_bad", "both bad"],
  ["uncertain", "uncertain"]
];
const TIMING = [
  ["a_start_better", "A start 更准"],
  ["b_start_better", "B start 更准"],
  ["a_end_better", "A end 更准"],
  ["b_end_better", "B end 更准"],
  ["a_cuts_speech", "A 截断语音"],
  ["b_cuts_speech", "B 截断语音"],
  ["a_includes_extra", "A 带入多余声音"],
  ["b_includes_extra", "B 带入多余声音"],
  ["a_crosses_gap", "A 跨 gap/噪声"],
  ["b_crosses_gap", "B 跨 gap/噪声"],
  ["a_text_better", "A 文本更完整"],
  ["b_text_better", "B 文本更完整"],
  ["repeat_or_nonlexical", "重复/非词影响判断"]
];
const media = document.getElementById("media");
const notes = document.getElementById("notes");
let currentIndex = 0;
let playMode = "context";
let annotations = {{}};
try {{ annotations = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}") || {{}}; }} catch (_) {{}}

function esc(value) {{
  return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}
function fmt(value) {{
  const seconds = Math.max(0, Number(value || 0));
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(2).padStart(5, "0");
  return `${{m}}:${{s}}`;
}}
function current() {{ return ITEMS[currentIndex]; }}
function save() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations)); updateProgress(); renderList(); }}
function annotation(item) {{ return annotations[item.item_id] || {{}}; }}
function playbackBounds(item, mode) {{
  if (mode === "context") return {{start: item.context_start, end: item.context_end}};
  const side = mode === "a" ? item.a : item.b;
  return {{
    start: Math.max(item.context_start, side.start - PLAYBACK_PREROLL_S),
    end: Math.min(item.context_end, side.end + PLAYBACK_POSTROLL_S)
  }};
}}
function setMedia(item) {{
  const expected = new URL(item.media_url, location.href).href;
  if (media.src !== expected) {{
    media.src = item.media_url;
    media.load();
  }}
}}
function renderList() {{
  document.getElementById("itemList").innerHTML = ITEMS.map((item, index) => {{
    const done = Boolean(annotation(item).primary_label);
    return `<div class="item ${{index === currentIndex ? "active" : ""}} ${{done ? "done" : ""}}" data-index="${{index}}">
      <div class="item-title">#${{index + 1}} · ${{esc(item.video_label || item.video_id)}}</div>
      <div class="meta">${{fmt(item.context_start)}} - ${{fmt(item.context_end)}}</div>
    </div>`;
  }}).join("");
  for (const el of document.querySelectorAll(".item")) {{
    el.onclick = () => {{ currentIndex = Number(el.dataset.index); renderCurrent(true); renderList(); }};
  }}
}}
function updateProgress() {{
  const done = ITEMS.filter(item => Boolean(annotation(item).primary_label)).length;
  document.getElementById("progressText").textContent = `${{done}} / ${{ITEMS.length}} 已标`;
  document.getElementById("progressBar").style.width = `${{ITEMS.length ? done / ITEMS.length * 100 : 0}}%`;
}}
function metricRows(side) {{
  return [
    ["alignment", `${{side.alignment_quality || "-"}} / ${{side.fallback_subtype || "none"}}`],
    ["sentinel", side.sentinel ? "yes" : "no"],
    ["nonlexical", side.nonlexical_text ? "yes" : "no"],
    ["repeat", `${{(side.repeat_profile || {{}}).unit || "-"}} x${{(side.repeat_profile || {{}}).run || 0}}`],
    ["cue density", Number(side.cue_density_cps || 0).toFixed(2) + " cps"]
  ].map(([key, value]) => `<div>${{esc(key)}}</div><div>${{esc(value)}}</div>`).join("");
}}
function renderButtons(item) {{
  const ann = annotation(item);
  const primaryRoot = document.getElementById("primaryLabels");
  primaryRoot.innerHTML = "";
  PRIMARY.forEach(([value, label], index) => {{
    const button = document.createElement("button");
    button.textContent = `${{index + 1}} · ${{label}}`;
    button.className = ann.primary_label === value ? "active" : "";
    button.onclick = () => {{
      annotations[item.item_id] = {{...annotation(item), primary_label: value, updated_at: new Date().toISOString()}};
      save();
      renderButtons(item);
    }};
    primaryRoot.appendChild(button);
  }});
  const timingRoot = document.getElementById("timingLabels");
  timingRoot.innerHTML = "";
  const selected = new Set(Array.isArray(ann.timing_reasons) ? ann.timing_reasons : []);
  for (const [value, label] of TIMING) {{
    const button = document.createElement("button");
    button.textContent = label;
    button.className = selected.has(value) ? "active" : "";
    button.onclick = () => {{
      const next = new Set(Array.isArray(annotation(item).timing_reasons) ? annotation(item).timing_reasons : []);
      if (next.has(value)) next.delete(value); else next.add(value);
      annotations[item.item_id] = {{...annotation(item), timing_reasons: Array.from(next), updated_at: new Date().toISOString()}};
      save();
      renderButtons(item);
    }};
    timingRoot.appendChild(button);
  }}
  notes.value = ann.notes || "";
}}
function setRange(el, start, end, item) {{
  const width = Math.max(0.001, item.context_end - item.context_start);
  const left = Math.max(0, Math.min(100, (start - item.context_start) / width * 100));
  const right = Math.max(0, Math.min(100, (end - item.context_start) / width * 100));
  el.style.left = `${{left}}%`;
  el.style.width = `${{Math.max(0.4, right - left)}}%`;
}}
function updateTimeline() {{
  const item = current();
  if (!item) return;
  setRange(document.getElementById("rangeA"), item.a.start, item.a.end, item);
  setRange(document.getElementById("rangeB"), item.b.start, item.b.end, item);
  const width = Math.max(0.001, item.context_end - item.context_start);
  const pct = Math.max(0, Math.min(100, (media.currentTime - item.context_start) / width * 100));
  document.getElementById("cursor").style.left = `${{pct}}%`;
  const side = playMode === "a" ? item.a : playMode === "b" ? item.b : null;
  const overlay = document.getElementById("overlay");
  const subtitleVisible = side && media.currentTime >= side.start && media.currentTime < side.end;
  overlay.textContent = subtitleVisible ? side.text || "(empty)" : "";
  overlay.style.display = subtitleVisible ? "block" : "none";
  const bounds = playbackBounds(item, playMode);
  if (!media.paused && media.currentTime >= bounds.end) media.pause();
}}
function renderCurrent(seek) {{
  const item = current();
  if (!item) return;
  setMedia(item);
  document.getElementById("itemTitle").textContent = `#${{currentIndex + 1}} · ${{item.video_label || item.video_id}}`;
  document.getElementById("itemMeta").textContent = `上下文 ${{fmt(item.context_start)}}-${{fmt(item.context_end)}} · 边界附近 ${{fmt(item.boundary_time_s)}}`;
  document.getElementById("timeA").textContent = `${{fmt(item.a.start)}}-${{fmt(item.a.end)}} · ${{Number(item.a.duration_s).toFixed(3)}}s`;
  document.getElementById("timeB").textContent = `${{fmt(item.b.start)}}-${{fmt(item.b.end)}} · ${{Number(item.b.duration_s).toFixed(3)}}s`;
  document.getElementById("textA").textContent = item.a.text || "(empty)";
  document.getElementById("textB").textContent = item.b.text || "(empty)";
  document.getElementById("metricsA").innerHTML = metricRows(item.a);
  document.getElementById("metricsB").innerHTML = metricRows(item.b);
  renderButtons(item);
  if (seek) media.currentTime = item.context_start;
  updateTimeline();
}}
function play(mode) {{
  const item = current();
  playMode = mode;
  media.currentTime = playbackBounds(item, mode).start;
  updateTimeline();
  media.play().catch(() => {{}});
}}
function exportRows() {{
  return ITEMS.map(item => ({{
    schema: {json.dumps(LABEL_SCHEMA)},
    dataset_id: DATASET_ID,
    item_id: item.item_id,
    display_index: item.display_index,
    video_id: item.video_id,
    ...(annotation(item))
  }}));
}}
function downloadJsonl() {{
  const text = exportRows().map(row => JSON.stringify(row)).join("\\n") + "\\n";
  const blob = new Blob([text], {{type: "application/jsonl;charset=utf-8"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "manual_boundary_preference_labels.jsonl";
  link.click();
  URL.revokeObjectURL(link.href);
}}
notes.addEventListener("input", () => {{
  const item = current();
  annotations[item.item_id] = {{...annotation(item), notes: notes.value, updated_at: new Date().toISOString()}};
  save();
}});
media.addEventListener("timeupdate", updateTimeline);
media.addEventListener("seeked", updateTimeline);
document.getElementById("timeline").onclick = event => {{
  const item = current();
  const rect = event.currentTarget.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  media.currentTime = item.context_start + ratio * (item.context_end - item.context_start);
}};
document.getElementById("playABtn").onclick = () => play("a");
document.getElementById("playBBtn").onclick = () => play("b");
document.getElementById("playContextBtn").onclick = () => play("context");
document.getElementById("pauseBtn").onclick = () => media.pause();
document.getElementById("prevBtn").onclick = () => {{ currentIndex = Math.max(0, currentIndex - 1); renderCurrent(true); renderList(); }};
document.getElementById("nextBtn").onclick = () => {{ currentIndex = Math.min(ITEMS.length - 1, currentIndex + 1); renderCurrent(true); renderList(); }};
document.getElementById("downloadBtn").onclick = downloadJsonl;
document.addEventListener("keydown", event => {{
  if (event.target && ["TEXTAREA", "INPUT"].includes(event.target.tagName)) return;
  const number = Number(event.key);
  if (number >= 1 && number <= 5) {{
    const button = document.getElementById("primaryLabels").children[number - 1];
    if (button) button.click();
  }}
  if (event.key === "j") document.getElementById("prevBtn").click();
  if (event.key === "k") document.getElementById("nextBtn").click();
}});
renderList();
updateProgress();
renderCurrent(true);
</script>
</body>
</html>
"""


def generate_audit(
    *,
    blind_manifest: Path,
    output_dir: Path,
    title: str,
    dataset_id: str,
    update_entrypoints: bool,
) -> dict[str, Any]:
    rows = read_jsonl(blind_manifest)
    if not rows:
        raise ValueError(f"no blind review rows found: {blind_manifest}")
    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for row in rows:
        if row.get("schema") != BLIND_ITEM_SCHEMA:
            raise ValueError(f"unsupported blind item schema: {row.get('schema')!r}")
        item = dict(row)
        media_path = project_path(str(item.get("media_path") or ""))
        if not media_path.exists():
            raise FileNotFoundError(media_path)
        sides = [item.get("a") or {}, item.get("b") or {}]
        earliest_start = min(float(side.get("start") or 0.0) for side in sides)
        latest_end = max(float(side.get("end") or 0.0) for side in sides)
        item["context_start"] = round(
            max(
                0.0,
                min(
                    float(item.get("context_start") or earliest_start),
                    earliest_start - PLAYBACK_PREROLL_S,
                ),
            ),
            6,
        )
        item["context_end"] = round(
            max(
                float(item.get("context_end") or latest_end),
                latest_end + PLAYBACK_POSTROLL_S,
            ),
            6,
        )
        item["media_url"] = rel_url(media_path, from_dir=output_dir)
        items.append(item)
    summary = {
        "dataset_id": dataset_id,
        "title": title,
        "label_schema": LABEL_SCHEMA,
        "blind_manifest": project_rel(blind_manifest),
        "html": project_rel(output_dir / "index.html"),
        "review_item_count": len(items),
        "video_counts": dict(Counter(str(item.get("video_id") or "") for item in items)),
        "primary_labels": [
            "a_better",
            "b_better",
            "tie",
            "both_bad",
            "uncertain",
        ],
        "playback_preroll_s": PLAYBACK_PREROLL_S,
        "playback_postroll_s": PLAYBACK_POSTROLL_S,
    }
    (output_dir / "index.html").write_text(
        _page_html(title=title, dataset_id=dataset_id, items=items),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if update_entrypoints:
        update_audit_entrypoints(latest_html=output_dir / "index.html", title=title)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a blinded A/B audit page for boundary perturbation preferences."
    )
    parser.add_argument("--blind-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="True v5 边界偏好盲测")
    parser.add_argument("--dataset-id", default="true-v5-boundary-preference-pilot")
    parser.add_argument("--update-entrypoints", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = project_path(args.output_dir)
    summary = generate_audit(
        blind_manifest=project_path(args.blind_manifest),
        output_dir=output_dir,
        title=args.title,
        dataset_id=args.dataset_id,
        update_entrypoints=args.update_entrypoints,
    )
    print(f"html={output_dir / 'index.html'}")
    print(f"summary={output_dir / 'summary.json'}")
    print(f"items={summary['review_item_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
