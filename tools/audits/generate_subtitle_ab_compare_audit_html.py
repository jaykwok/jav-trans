#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


TIMESTAMP_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})(?:[,.](?P<ms>\d{1,3}))?"
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_timestamp(value: str) -> float:
    match = TIMESTAMP_RE.search(value.strip())
    if not match:
        raise ValueError(f"invalid timestamp: {value!r}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int((match.group("ms") or "0").ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def srt_timestamp(seconds: float, *, decimal: str = ",") -> str:
    seconds = max(0.0, float(seconds))
    millis = int(round((seconds - int(seconds)) * 1000))
    whole = int(seconds)
    if millis >= 1000:
        whole += 1
        millis -= 1000
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def read_srt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", text.strip())
    cues: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        left, right = lines[0].split("-->", 1)
        start = parse_timestamp(left)
        end = parse_timestamp(right.split()[0])
        cue_text = "\n".join(lines[1:]).strip()
        if end <= start:
            continue
        cues.append(
            {
                "index": len(cues) + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": cue_text,
            }
        )
    return cues


def write_vtt(path: Path, cues: list[dict[str, Any]]) -> None:
    lines = ["WEBVTT", ""]
    for cue in cues:
        text = str(cue.get("text") or "").replace("\r", "").strip() or "..."
        lines.extend(
            [
                f"{srt_timestamp(float(cue['start']), decimal='.')} --> {srt_timestamp(float(cue['end']), decimal='.')}",
                text,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def overlapping(cues: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [cue for cue in cues if float(cue["end"]) > start and float(cue["start"]) < end]


def preview(cues: list[dict[str, Any]], *, max_chars: int = 180) -> str:
    text = " / ".join(str(cue.get("text") or "").replace("\n", " ") for cue in cues)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def make_hotspots(
    *,
    old_cues: list[dict[str, Any]],
    new_cues: list[dict[str, Any]],
    duration_s: float,
    window_s: float = 30.0,
    limit: int = 180,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window_count = max(1, math.ceil(duration_s / window_s))
    for index in range(window_count):
        start = index * window_s
        end = min(duration_s, start + window_s)
        old_rows = overlapping(old_cues, start, end)
        new_rows = overlapping(new_cues, start, end)
        old_chars = sum(len(str(row.get("text") or "").replace("\n", "")) for row in old_rows)
        new_chars = sum(len(str(row.get("text") or "").replace("\n", "")) for row in new_rows)
        count_delta = len(old_rows) - len(new_rows)
        char_delta = old_chars - new_chars
        score = abs(count_delta) * 2.0 + abs(char_delta) / 18.0
        if len(old_rows) >= 10:
            score += 1.5
        if len(new_rows) >= 10:
            score += 1.0
        if score < 2.0 and preview(old_rows) == preview(new_rows):
            continue
        rows.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "old_count": len(old_rows),
                "new_count": len(new_rows),
                "old_chars": old_chars,
                "new_chars": new_chars,
                "count_delta": count_delta,
                "char_delta": char_delta,
                "score": round(score, 3),
                "old_preview": preview(old_rows),
                "new_preview": preview(new_rows),
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), float(row["start"])))
    selected = rows[:limit]
    selected.sort(key=lambda row: float(row["start"]))
    return selected


def load_quality(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def page_html(*, title: str, dataset: dict[str, Any]) -> str:
    data_json = json.dumps(dataset, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    media_mode = str(dataset.get("media_mode") or "video")
    if media_mode == "audio":
        stage_html = """
    <div class="stage audio-stage">
      <div class="subtitle audio-subtitle old"><span class="tag">旧</span><span id="oldOverlay"></span></div>
      <audio id="video" controls preload="metadata"></audio>
      <div class="subtitle audio-subtitle new"><span class="tag">新</span><span id="newOverlay"></span></div>
    </div>
"""
        media_hint = "旧字幕显示在音频控件上方，新字幕显示在音频控件下方。这里直接比较完整日语 SRT 的显示效果。"
        media_link_label = "音频源"
    else:
        stage_html = """
    <div class="stage">
      <video id="video" controls preload="metadata" playsinline></video>
      <div class="subtitle-layer">
        <div class="subtitle old"><span class="tag">旧</span><span id="oldOverlay"></span></div>
        <div class="subtitle new"><span class="tag">新</span><span id="newOverlay"></span></div>
      </div>
    </div>
"""
        media_hint = "旧字幕显示在视频上方，新字幕显示在视频下方。这里直接比较完整日语 SRT 的显示效果，浏览器原生字幕轨不参与渲染。"
        media_link_label = "视频"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f4f6f5;
  --panel: #ffffff;
  --ink: #1f2523;
  --muted: #66706c;
  --line: #d8dfdc;
  --old: #b45309;
  --new: #0f766e;
  --accent: #1f5f57;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}}
button, select {{ font: inherit; }}
button, select {{
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 10px;
  background: #fff;
  color: var(--ink);
}}
button {{ cursor: pointer; }}
button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
main {{ max-width: 1220px; margin: 0 auto; padding: 18px; }}
.topbar {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 12px; }}
h1 {{ margin: 0 0 6px; font-size: 20px; }}
.meta, .hint {{ color: var(--muted); font-size: 12px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: flex-end; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
.stage {{ position: relative; background: #050706; border-radius: 8px; overflow: hidden; }}
video {{ display: block; width: 100%; max-height: 72vh; background: #050706; }}
audio {{ display: block; width: min(900px, 100%); margin: 16px auto; }}
.audio-stage {{ padding: 16px; }}
.subtitle-layer {{ position: absolute; inset: 12px 20px 58px; pointer-events: none; display: flex; flex-direction: column; justify-content: space-between; gap: 20px; }}
.subtitle {{
  max-width: min(980px, 94%);
  margin: 0 auto;
  padding: 8px 12px;
  border-radius: 6px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0,0,0,.9);
  font-weight: 650;
  font-size: clamp(18px, 2.2vw, 30px);
  line-height: 1.32;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: rgba(0,0,0,.58);
}}
.subtitle.old {{ border-top: 3px solid var(--old); }}
.subtitle.new {{ border-bottom: 3px solid var(--new); }}
.audio-subtitle {{ min-height: 74px; }}
.tag {{ display: inline-block; margin-right: 8px; font-size: 12px; color: #fff; opacity: .82; vertical-align: middle; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 12px; }}
.now-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.cue-box {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfbf9; padding: 10px; min-height: 88px; }}
.cue-title {{ display: flex; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
.cue-title strong {{ color: var(--ink); }}
.cue-text {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 15px; line-height: 1.4; }}
.hotspots {{ max-height: 72vh; overflow: auto; }}
.hotspot {{ display: grid; gap: 4px; width: 100%; text-align: left; margin-bottom: 8px; border: 1px solid var(--line); background: #fbfbf9; }}
.hotspot.active {{ border-color: var(--accent); background: #e9f3ef; }}
.hotspot-time {{ font-weight: 700; }}
.hotspot-preview {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
.metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #fbfbf9; }}
.metric b {{ display: block; font-size: 16px; }}
.old-label {{ color: var(--old); }}
.new-label {{ color: var(--new); }}
.links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }}
a {{ color: #0f5f57; }}
@media (max-width: 920px) {{
  .grid, .now-grid, .metrics {{ grid-template-columns: 1fr; }}
  .subtitle-layer {{ inset: 8px 10px 54px; }}
  .subtitle {{ font-size: 18px; }}
}}
</style>
</head>
<body>
<main>
  <div class="topbar">
    <div>
      <h1>{html.escape(title)}</h1>
      <div class="meta" id="summaryLine"></div>
    </div>
    <div class="controls">
      <select id="videoSelect"></select>
      <button id="prevBtn">上一处</button>
      <button id="nextBtn">下一处</button>
      <button class="primary" id="playWindowBtn">播放当前窗口</button>
    </div>
  </div>
  <section class="panel">
{stage_html}
    <div class="links" id="links"></div>
    <p class="hint">{html.escape(media_hint)}</p>
  </section>
  <section class="grid">
    <div>
      <section class="panel">
        <div class="metrics" id="metrics"></div>
      </section>
      <section class="panel">
        <div class="now-grid">
          <div class="cue-box">
            <div class="cue-title"><strong class="old-label">旧字幕当前 cue</strong><span id="oldCount"></span></div>
            <div class="cue-text" id="oldNow"></div>
          </div>
          <div class="cue-box">
            <div class="cue-title"><strong class="new-label">新字幕当前 cue</strong><span id="newCount"></span></div>
            <div class="cue-text" id="newNow"></div>
          </div>
        </div>
      </section>
    </div>
    <aside class="panel hotspots" id="hotspotList"></aside>
  </section>
</main>
<script id="dataset" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("dataset").textContent);
const video = document.getElementById("video");
const videoSelect = document.getElementById("videoSelect");
const hotspotList = document.getElementById("hotspotList");
let videoIndex = 0;
let hotspotIndex = 0;
let stopAt = null;

function fmtTime(value) {{
  const seconds = Math.max(0, Number(value || 0));
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${{m}}:${{s}}`;
}}
function escapeHtml(value) {{
  return String(value || "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}
function activeCues(cues, time) {{
  return cues.filter(cue => cue.start <= time && cue.end >= time);
}}
function cueText(cues) {{
  return cues.map(cue => cue.text || "").filter(Boolean).join("\\n");
}}
function currentCase() {{ return DATA.cases[videoIndex]; }}
function currentHotspot() {{ return currentCase().hotspots[hotspotIndex] || null; }}
function renderSelect() {{
  videoSelect.innerHTML = DATA.cases.map((item, index) => `<option value="${{index}}">${{escapeHtml(item.label)}}</option>`).join("");
  videoSelect.value = String(videoIndex);
}}
function renderMetrics(item) {{
  const oldQ = item.old.quality || {{}};
  const newQ = item.new.quality || {{}};
  const oldM = item.old.metrics || {{}};
  const newM = item.new.metrics || {{}};
  const rows = [
    ["chunks", oldM.chunks, newM.chunks],
    ["blocks", oldM.blocks, newM.blocks],
    ["ASR+align", `${{oldM.asr_align_s || "?"}}s`, `${{newM.asr_align_s || "?"}}s`],
    ["per min", oldQ.per_min_subtitle_count, newQ.per_min_subtitle_count],
    ["short ratio", oldQ.short_segment_ratio, newQ.short_segment_ratio],
    ["fallback", oldQ.alignment_fallback_ratio, newQ.alignment_fallback_ratio],
    ["ASR gen errors", oldM.asr_generation_errors, newM.asr_generation_errors],
    ["ASR gen overflow", oldM.asr_generation_overflow, newM.asr_generation_overflow],
  ];
  document.getElementById("metrics").innerHTML = rows.map(([name, oldValue, newValue]) => `
    <div class="metric">
      <span>${{escapeHtml(name)}}</span>
      <b><span class="old-label">${{escapeHtml(oldValue ?? "")}}</span> → <span class="new-label">${{escapeHtml(newValue ?? "")}}</span></b>
    </div>
  `).join("");
}}
function renderHotspots() {{
  const item = currentCase();
  hotspotList.innerHTML = `
    <h2 style="margin:0 0 10px;font-size:15px;">差异窗口</h2>
    ${{item.hotspots.map((row, index) => `
      <button class="hotspot ${{index === hotspotIndex ? "active" : ""}}" data-index="${{index}}">
        <span class="hotspot-time">${{fmtTime(row.start)}} - ${{fmtTime(row.end)}} · 旧 ${{row.old_count}} / 新 ${{row.new_count}}</span>
        <span class="hotspot-preview"><b>旧</b> ${{escapeHtml(row.old_preview || "(empty)")}}</span>
        <span class="hotspot-preview"><b>新</b> ${{escapeHtml(row.new_preview || "(empty)")}}</span>
      </button>
    `).join("")}}
  `;
  for (const button of hotspotList.querySelectorAll(".hotspot")) {{
    button.addEventListener("click", () => {{
      hotspotIndex = Number(button.dataset.index || 0);
      seekHotspot(true);
      renderHotspots();
    }});
  }}
}}
function renderLinks(item) {{
  document.getElementById("links").innerHTML = `
    <a href="${{escapeHtml(item.media)}}">{html.escape(media_link_label)}</a>
    <a href="${{escapeHtml(item.old.srt_url)}}">旧 SRT</a>
    <a href="${{escapeHtml(item.new.srt_url)}}">新 SRT</a>
    <a href="${{escapeHtml(item.old.vtt_url)}}">旧 VTT</a>
    <a href="${{escapeHtml(item.new.vtt_url)}}">新 VTT</a>
  `;
}}
function loadCase(index) {{
  videoIndex = index;
  hotspotIndex = 0;
  const item = currentCase();
  video.src = item.media;
  video.load();
  document.getElementById("summaryLine").textContent = DATA.summary;
  renderSelect();
  renderMetrics(item);
  renderLinks(item);
  renderHotspots();
  updateCurrentText();
}}
function updateCurrentText() {{
  const item = currentCase();
  const t = video.currentTime || 0;
  const oldActive = activeCues(item.old.cues, t);
  const newActive = activeCues(item.new.cues, t);
  const oldText = cueText(oldActive);
  const newText = cueText(newActive);
  document.getElementById("oldOverlay").textContent = oldText;
  document.getElementById("newOverlay").textContent = newText;
  document.getElementById("oldNow").textContent = oldText || "(无当前字幕)";
  document.getElementById("newNow").textContent = newText || "(无当前字幕)";
  document.getElementById("oldCount").textContent = oldActive.length ? `${{oldActive.length}} cue` : "";
  document.getElementById("newCount").textContent = newActive.length ? `${{newActive.length}} cue` : "";
  if (stopAt !== null && t >= stopAt) {{
    video.pause();
    stopAt = null;
  }}
}}
function seekHotspot(play) {{
  const row = currentHotspot();
  if (!row) return;
  video.currentTime = Math.max(0, row.start - 0.3);
  stopAt = row.end + 0.3;
  updateCurrentText();
  if (play) video.play();
}}
videoSelect.addEventListener("change", () => loadCase(Number(videoSelect.value || 0)));
video.addEventListener("timeupdate", updateCurrentText);
video.addEventListener("seeked", updateCurrentText);
document.getElementById("playWindowBtn").addEventListener("click", () => seekHotspot(true));
document.getElementById("prevBtn").addEventListener("click", () => {{
  hotspotIndex = Math.max(0, hotspotIndex - 1);
  seekHotspot(true);
  renderHotspots();
}});
document.getElementById("nextBtn").addEventListener("click", () => {{
  hotspotIndex = Math.min(currentCase().hotspots.length - 1, hotspotIndex + 1);
  seekHotspot(true);
  renderHotspots();
}});
document.addEventListener("keydown", event => {{
  if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  if (event.key === "j") document.getElementById("prevBtn").click();
  if (event.key === "k") document.getElementById("nextBtn").click();
  if (event.key === " ") {{ event.preventDefault(); video.paused ? video.play() : video.pause(); }}
}});
loadCase(0);
</script>
</body>
</html>
"""


def case_from_args(values: list[str], output_dir: Path, *, old_label: str, new_label: str) -> dict[str, Any]:
    if len(values) != 8:
        raise ValueError("--case expects 8 values: id label media old_srt new_srt old_quality_json new_quality_json old_summary_json:new_summary_json")
    case_id, label, media, old_srt, new_srt, old_quality, new_quality, summary_pair = values
    old_summary_path_raw, new_summary_path_raw = summary_pair.split(":", 1)
    media_path = project_path(media)
    old_srt_path = project_path(old_srt)
    new_srt_path = project_path(new_srt)
    old_cues = read_srt(old_srt_path)
    new_cues = read_srt(new_srt_path)
    duration_s = max(
        [0.0]
        + [float(cue["end"]) for cue in old_cues]
        + [float(cue["end"]) for cue in new_cues]
    )
    old_vtt = output_dir / f"{case_id}.old.vtt"
    new_vtt = output_dir / f"{case_id}.new.vtt"
    write_vtt(old_vtt, old_cues)
    write_vtt(new_vtt, new_cues)
    return {
        "id": case_id,
        "label": label,
        "duration_s": round(duration_s, 3),
        "media": rel_url(media_path, from_dir=output_dir),
        "source_media": project_rel(media_path),
        "old": {
            "label": old_label,
            "srt": project_rel(old_srt_path),
            "srt_url": rel_url(old_srt_path, from_dir=output_dir),
            "vtt": project_rel(old_vtt),
            "vtt_url": rel_url(old_vtt, from_dir=output_dir),
            "cues": old_cues,
            "quality": load_quality(project_path(old_quality)),
            "metrics": metrics_for_case(project_path(old_summary_path_raw), case_id),
        },
        "new": {
            "label": new_label,
            "srt": project_rel(new_srt_path),
            "srt_url": rel_url(new_srt_path, from_dir=output_dir),
            "vtt": project_rel(new_vtt),
            "vtt_url": rel_url(new_vtt, from_dir=output_dir),
            "cues": new_cues,
            "quality": load_quality(project_path(new_quality)),
            "metrics": metrics_for_case(project_path(new_summary_path_raw), case_id),
        },
        "hotspots": make_hotspots(old_cues=old_cues, new_cues=new_cues, duration_s=duration_s),
    }


def metrics_for_case(summary_path: Path, case_id: str) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    payload = read_json(summary_path)
    for row in payload.get("results") or []:
        if Path(str(row.get("video") or "")).stem != case_id:
            continue
        counts = row.get("counts") or {}
        asr_generation = row.get("asr_generation") or {}
        stage = row.get("stage_timings") or {}
        return {
            "chunks": counts.get("transcript_chunks"),
            "blocks": counts.get("blocks"),
            "asr_align_s": round(float(stage.get("asr_alignment_total_s") or 0.0), 1),
            "total_s": round(float(stage.get("pipeline_total_s") or 0.0), 1),
            "asr_generation_errors": asr_generation.get("error_count"),
            "asr_generation_overflow": asr_generation.get("overflow_count"),
        }
    return {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full-video old/new subtitle A/B audit page.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Boundary Refiner 字幕 A/B 对比审计")
    parser.add_argument("--dataset-id", default="subtitle-ab-compare")
    parser.add_argument("--media-mode", choices=("video", "audio"), default="video")
    parser.add_argument("--old-label", default="旧")
    parser.add_argument("--new-label", default="新")
    parser.add_argument(
        "--case",
        nargs=8,
        action="append",
        required=True,
        metavar=("ID", "LABEL", "MEDIA", "OLD_SRT", "NEW_SRT", "OLD_QUALITY", "NEW_QUALITY", "OLD_SUMMARY:NEW_SUMMARY"),
    )
    parser.add_argument("--update-audit-nav", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        case_from_args(values, output_dir, old_label=args.old_label, new_label=args.new_label)
        for values in args.case
    ]
    summary = {
        "dataset_id": args.dataset_id,
        "title": args.title,
        "rows": sum(len(case["hotspots"]) for case in cases),
        "review_item_count": sum(len(case["hotspots"]) for case in cases),
        "subtitle_cue_count": sum(len(case["old"]["cues"]) + len(case["new"]["cues"]) for case in cases),
        "case_count": len(cases),
        "video_label": " / ".join(case["label"] for case in cases),
        "cases": [
            {
                "id": case["id"],
                "label": case["label"],
                "old_cues": len(case["old"]["cues"]),
                "new_cues": len(case["new"]["cues"]),
                "hotspots": len(case["hotspots"]),
                "media": case["source_media"],
                "old_srt": case["old"]["srt"],
                "new_srt": case["new"]["srt"],
            }
            for case in cases
        ],
    }
    dataset = {
        "dataset_id": args.dataset_id,
        "media_mode": args.media_mode,
        "title": args.title,
        "summary": f"{len(cases)} videos · {summary['review_item_count']} 差异窗口 · {summary['subtitle_cue_count']} cues",
        "cases": cases,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "index.html").write_text(page_html(title=args.title, dataset=dataset), encoding="utf-8")
    if args.update_audit_nav:
        update_audit_entrypoints(latest_html=output_dir / "index.html", title=args.title)
    print(f"html={output_dir / 'index.html'}")
    print(f"summary={output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
