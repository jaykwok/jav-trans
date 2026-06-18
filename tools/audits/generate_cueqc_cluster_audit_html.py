#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import ANON_LABELS, update_audit_entrypoints  # noqa: E402


TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
PREROLL_S = 2.0
POSTROLL_S = 2.0


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
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def json_for_script(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")


def parse_timestamp(value: str) -> float:
    match = re.search(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})(?:[,.](?P<ms>\d{1,3}))?", value.strip())
    if not match:
        raise ValueError(f"invalid timestamp: {value!r}")
    millis = int((match.group("ms") or "0").ljust(3, "0")[:3])
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + millis / 1000.0
    )


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


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds, decimal=".")


def clean_cue_text(value: Any) -> str:
    return str(value or "").replace("\r", "").strip() or "..."


def read_srt_cues(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
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
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.extend(
            [
                f"{vtt_timestamp(float(cue['start']))} --> {vtt_timestamp(float(cue['end']))}",
                clean_cue_text(cue.get("text")),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return min(a_end, b_end) - max(a_start, b_start) > 0.01


def cues_for_range(cues: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [
        cue
        for cue in cues
        if overlaps(start, end, row_float(cue, "start"), row_float(cue, "end"))
    ]


def audio_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".m4a":
        return "audio/mp4"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".ogg":
        return "audio/ogg"
    if suffix == ".opus":
        return "audio/opus"
    if suffix == ".flac":
        return "audio/flac"
    if suffix == ".webm":
        return "audio/webm"
    if suffix == ".aac":
        return "audio/aac"
    return "audio/wav"


def discover_media(*, baseline_root: Path, output_dir: Path, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    video_ids = sorted({str(row.get("video_id") or "").strip() for row in rows if row.get("video_id")})
    media_by_video: dict[str, Any] = {}
    cues_by_video: dict[str, list[dict[str, Any]]] = {}
    aligned_by_video: dict[str, list[dict[str, Any]]] = {}
    subtitles_dir = output_dir / "subtitles"

    for video_id in video_ids:
        label = ANON_LABELS.get(video_id, video_id)
        archived_dir = baseline_root / "archived" / video_id
        srt_path = archived_dir / f"{video_id}.ja.srt"
        aligned_path = archived_dir / f"{video_id}.aligned_segments.json"
        audio_dir = baseline_root / "jobs" / f"{video_id}_b5" / "audio"
        audio_candidates = sorted(
            path for path in audio_dir.glob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )
        audio_path = audio_candidates[0] if audio_candidates else None

        cues: list[dict[str, Any]] = []
        vtt_path = subtitles_dir / f"{video_id}.ja.vtt"
        if srt_path.exists():
            cues = read_srt_cues(srt_path)
            write_vtt(vtt_path, cues)

        aligned_segments: list[dict[str, Any]] = []
        if aligned_path.exists():
            payload = read_json(aligned_path)
            raw_segments = payload.get("segments", []) if isinstance(payload, Mapping) else []
            if isinstance(raw_segments, list):
                for index, segment in enumerate(raw_segments):
                    if not isinstance(segment, Mapping):
                        continue
                    start = row_float(segment, "start")
                    end = row_float(segment, "end")
                    if end <= start:
                        continue
                    aligned_segments.append(
                        {
                            "index": index,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "text": clean_cue_text(segment.get("text")),
                            "alignment_quality": segment.get("alignment_quality", ""),
                            "fallback_subtype": segment.get("fallback_subtype", ""),
                        }
                    )

        media_by_video[video_id] = {
            "video_id": video_id,
            "video_label": label,
            "audio_path": project_rel(audio_path) if audio_path else "",
            "audio_url": rel_url(audio_path, from_dir=output_dir) if audio_path else "",
            "audio_mime": audio_mime(audio_path) if audio_path else "",
            "audio_exists": bool(audio_path is not None and audio_path.exists()),
            "srt_path": project_rel(srt_path) if srt_path.exists() else "",
            "vtt_path": project_rel(vtt_path) if vtt_path.exists() else "",
            "vtt_url": rel_url(vtt_path, from_dir=output_dir) if vtt_path.exists() else "",
            "subtitle_exists": bool(srt_path.exists() and vtt_path.exists()),
            "subtitle_cue_count": len(cues),
            "aligned_path": project_rel(aligned_path) if aligned_path.exists() else "",
            "aligned_segment_count": len(aligned_segments),
        }
        cues_by_video[video_id] = cues
        aligned_by_video[video_id] = aligned_segments
    return media_by_video, cues_by_video, aligned_by_video


def absolute_fallback_range(row: Mapping[str, Any], start: float, end: float) -> tuple[float, float]:
    timing = row.get("subtitle_timing")
    if not isinstance(timing, Mapping):
        return start, end
    rel_start = timing.get("fallback_window_start_s")
    rel_end = timing.get("fallback_window_end_s")
    try:
        fallback_start = start + float(rel_start) if rel_start is not None else start
        fallback_end = start + float(rel_end) if rel_end is not None else end
    except (TypeError, ValueError):
        return start, end
    if fallback_end <= fallback_start:
        return start, end
    return max(0.0, fallback_start), fallback_end


def enrich_rows(
    *,
    rows: list[dict[str, Any]],
    media_by_video: Mapping[str, Any],
    cues_by_video: Mapping[str, list[dict[str, Any]]],
    aligned_by_video: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        video_id = str(item.get("video_id") or "")
        start = row_float(item, "start")
        end = row_float(item, "end")
        if end <= start:
            end = start + max(0.0, row_float(item, "duration_s"))
        fallback_start, fallback_end = absolute_fallback_range(item, start, end)
        context_start = max(0.0, min(start, fallback_start) - PREROLL_S)
        context_end = max(end, fallback_end) + POSTROLL_S
        cues = cues_by_video.get(video_id, [])
        aligned_segments = aligned_by_video.get(video_id, [])
        media = dict(media_by_video.get(video_id) or {})

        item.update(
            {
                "index": index,
                "video_label": media.get("video_label", video_id),
                "media": media,
                "context_start": round(context_start, 3),
                "context_end": round(context_end, 3),
                "fallback_window_start": round(fallback_start, 3),
                "fallback_window_end": round(fallback_end, 3),
                "chunk_subtitle_cues": cues_for_range(cues, start, end),
                "fallback_subtitle_cues": cues_for_range(cues, fallback_start, fallback_end),
                "context_subtitle_cues": cues_for_range(cues, context_start, context_end),
                "aligned_segments": cues_for_range(aligned_segments, start, end),
            }
        )
        enriched.append(item)
    return enriched


def _page_html(
    *,
    title: str,
    dataset_id: str,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    media_by_video: Mapping[str, Any],
    cues_by_video: Mapping[str, list[dict[str, Any]]],
) -> str:
    rows_json = json_for_script(rows)
    summaries_json = json_for_script(summaries)
    media_json = json_for_script(media_by_video)
    cues_json = json_for_script(cues_by_video)
    dataset_id_json = json_for_script(dataset_id)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f5f6f2;
  --panel: #fff;
  --ink: #202521;
  --muted: #67716a;
  --line: #d8ddd8;
  --accent: #0f766e;
  --accent-soft: #dff2ee;
  --amber: #b45309;
  --amber-soft: #f6e6c8;
  --danger: #a33b36;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }}
.app {{ display: grid; grid-template-columns: 380px minmax(0, 1fr); min-height: 100vh; }}
.side {{ border-right: 1px solid var(--line); background: #fff; max-height: 100vh; overflow: auto; }}
.main {{ padding: 16px; max-height: 100vh; overflow: auto; }}
.head {{ position: sticky; top: 0; z-index: 3; background: #fff; padding: 12px; border-bottom: 1px solid var(--line); }}
h1 {{ margin: 0 0 8px; font-size: 16px; }}
h2, h3 {{ margin: 0 0 8px; }}
h2 {{ font-size: 17px; }}
h3 {{ font-size: 14px; }}
input, select, textarea, button {{ font: inherit; }}
input, select {{ width: 100%; padding: 7px; border: 1px solid var(--line); border-radius: 6px; background: #fff; }}
button {{ border: 1px solid var(--line); border-radius: 6px; padding: 7px 10px; background: #fff; cursor: pointer; }}
button.active {{ background: var(--accent-soft); border-color: var(--accent); color: #063d38; }}
button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
summary {{ cursor: pointer; font-weight: 700; }}
a {{ color: var(--accent); text-decoration: none; }}
.filters {{ display: grid; grid-template-columns: 1fr 150px; gap: 8px; }}
.item {{ padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }}
.item.active, .item:hover {{ background: #eaf4f1; }}
.item-title {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 1px 7px; background: #eef3ef; color: var(--muted); font-size: 12px; }}
.badge.warn {{ background: var(--amber-soft); color: #7a3f00; }}
.badge.noise {{ background: #eee7df; color: #76533b; }}
.meta {{ color: var(--muted); font-size: 12px; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(300px, 0.75fr); gap: 12px; align-items: start; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
.kv {{ display: grid; grid-template-columns: 170px minmax(0, 1fr); gap: 6px 10px; }}
.kv > div:nth-child(odd) {{ color: var(--muted); }}
.text {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 18px; }}
.text-box {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f7faf8; border: 1px solid var(--line); border-radius: 6px; padding: 8px; min-height: 38px; }}
.labels, .actions, .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.label-block {{ border-top: 1px solid var(--line); padding-top: 10px; margin-top: 10px; }}
.label-block:first-of-type {{ border-top: 0; padding-top: 0; margin-top: 0; }}
.label-heading {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 7px; }}
.label-title {{ font-weight: 700; }}
.label-block.disabled {{ opacity: 0.55; }}
.label-block.disabled button {{ pointer-events: none; }}
.cluster-review {{ display: grid; gap: 10px; }}
.cluster-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfb; }}
.cluster-card.active {{ border-color: var(--accent); background: #f3faf8; }}
.cluster-card.complete .cluster-title::after {{ content: " 已标注"; color: var(--accent); font-size: 12px; font-weight: 500; }}
.cluster-card-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }}
.cluster-title {{ font-weight: 700; }}
.cluster-examples {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }}
.cluster-example {{ text-align: left; min-height: 66px; }}
.cluster-fields {{ display: grid; grid-template-columns: minmax(150px, 0.35fr) minmax(240px, 1fr); gap: 8px; margin-top: 8px; }}
.cluster-fields textarea {{ min-height: 52px; }}
.cluster-status {{ min-height: 18px; margin-top: 8px; }}
.cluster-nav {{ display: grid; gap: 8px; }}
.cluster-nav-button {{ width: 100%; text-align: left; border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fff; }}
.cluster-nav-button.active {{ background: var(--accent-soft); border-color: var(--accent); }}
.cluster-nav-button.complete {{ background: #f3faf8; }}
.cluster-nav-top {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px; flex-wrap: wrap; }}
.cluster-nav-index {{ font-weight: 700; }}
.cluster-nav-sub {{ margin-top: 4px; color: var(--muted); font-size: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }}
.cluster-detail-summary {{ color: var(--muted); font-size: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }}
.cluster-detail-fields {{ display: grid; gap: 8px; margin-top: 10px; }}
.cluster-detail-fields textarea {{ min-height: 86px; }}
.cluster-audio-list {{ display: grid; gap: 10px; max-height: min(78vh, 1200px); overflow: auto; padding-right: 4px; }}
.cluster-audio-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfb; display: grid; gap: 8px; }}
.cluster-audio-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px; flex-wrap: wrap; }}
.cluster-audio-meta {{ color: var(--muted); font-size: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }}
.cluster-audio-text {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
.legacy-ui[hidden] {{ display: none !important; }}
#labelStatus {{ min-height: 18px; margin: 8px 0 0; }}
textarea {{ width: 100%; min-height: 80px; border: 1px solid var(--line); border-radius: 6px; padding: 8px; resize: vertical; }}
audio {{ width: 100%; }}
.caption-preview {{ margin-top: 10px; min-height: 54px; display: flex; align-items: center; justify-content: center; text-align: center; white-space: pre-wrap; overflow-wrap: anywhere; border: 1px solid var(--line); border-radius: 8px; background: #17201d; color: #fff; font-size: 20px; padding: 10px; }}
.timeline {{ position: relative; height: 22px; margin-top: 12px; border: 1px solid var(--line); border-radius: 999px; background: #edf1ee; overflow: hidden; cursor: pointer; }}
.context-range {{ position: absolute; top: 0; bottom: 0; left: 0; right: 0; background: #e5ece8; }}
.chunk-range, .fallback-range {{ position: absolute; top: 3px; bottom: 3px; border-radius: 999px; }}
.chunk-range {{ background: rgba(15, 118, 110, 0.28); border: 1px solid rgba(15, 118, 110, 0.5); }}
.fallback-range {{ background: rgba(180, 83, 9, 0.34); border: 1px solid rgba(180, 83, 9, 0.55); }}
.cursor {{ position: absolute; top: 0; bottom: 0; width: 2px; background: var(--danger); transform: translateX(-1px); }}
.timeline-labels {{ display: flex; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 12px; margin-top: 4px; }}
.cue-list {{ display: grid; gap: 6px; }}
.cue {{ border: 1px solid var(--line); border-radius: 6px; padding: 7px 8px; background: #fbfcfb; }}
.cue.active {{ border-color: var(--accent); background: var(--accent-soft); }}
.error {{ color: var(--danger); }}
code {{ background: #eef3ef; padding: 1px 4px; border-radius: 4px; }}
@media (max-width: 980px) {{
  .app {{ grid-template-columns: 1fr; }}
  .side, .main {{ max-height: none; }}
  .grid {{ grid-template-columns: 1fr; }}
  .cluster-examples, .cluster-fields {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div class="head">
      <h1>{html.escape(title)}</h1>
      <div class="toolbar">
        <button class="primary" id="downloadClusters">下载簇级标注 JSONL</button>
      </div>
      <p class="meta" id="clusterProgress"></p>
      <p class="meta" id="summaryLine"></p>
    </div>
    <div class="cluster-nav" id="clusterNav"></div>
  </aside>
  <main class="main">
    <section class="panel">
      <div class="label-heading">
        <h2 id="clusterTitle">簇级粗标签</h2>
        <span class="meta" id="clusterCount"></span>
      </div>
      <div class="cluster-detail-summary" id="clusterSummary"></div>
      <div class="cluster-detail-fields">
        <div class="label-heading"><span class="label-title">训练标签</span><span class="meta">必选 · 只导出 keep/drop · 后续全量训练修正混簇噪声</span></div>
        <div class="labels" id="clusterDisplayButtons"></div>
        <textarea id="activeClusterReason" placeholder="备注 / 原因（可选，不进训练）"></textarea>
      </div>
      <p class="meta cluster-status" id="clusterStatus"></p>
    </section>
    <section class="panel">
      <h3>全部音频</h3>
      <div class="cluster-audio-list" id="clusterAudioList"></div>
    </section>
    <div class="legacy-ui" hidden>
      <div class="filters">
        <input id="search" placeholder="搜索文本 / sample_id">
        <select id="cluster"></select>
      </div>
      <div id="list"></div>
      <div class="cluster-review" id="clusterReview"></div>
    <section class="panel">
      <h2 id="sampleTitle"></h2>
      <p class="meta" id="sampleMeta"></p>
      <div class="text" id="text"></div>
    </section>
    <div class="grid">
      <section class="panel">
        <div class="actions" style="margin-bottom:10px">
          <button id="prevBtn">上一条</button>
          <button id="nextBtn">下一条</button>
          <button class="primary" id="playChunkBtn">播放 chunk</button>
          <button id="playContextBtn">播放上下文</button>
        </div>
        <audio id="media" controls preload="metadata"></audio>
        <div class="caption-preview" id="captionOverlay"></div>
        <div class="timeline" id="timeline">
          <div class="context-range"></div>
          <div class="chunk-range" id="chunkRange"></div>
          <div class="fallback-range" id="fallbackRange"></div>
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
          <a id="mediaLink" href="#">打开音频</a>
          <a id="vttLink" href="#">打开 VTT</a>
        </div>
        <p class="meta error" id="mediaError"></p>
      </section>
      <aside class="panel">
        <h3>候选指标</h3>
        <div class="kv" id="metrics"></div>
      </aside>
    </div>
    <section class="panel">
      <h3>字幕上下文</h3>
      <div class="grid">
        <div>
          <h3>chunk 内字幕</h3>
          <div class="cue-list" id="chunkCueList"></div>
        </div>
        <div>
          <h3>字幕时间轴窗口内字幕</h3>
          <div class="cue-list" id="fallbackCueList"></div>
        </div>
      </div>
      <h3 style="margin-top:12px">上下文字幕</h3>
      <div class="cue-list" id="contextCueList"></div>
    </section>
    <section class="panel">
      <h3>字幕片段</h3>
      <div class="cue-list" id="alignedList"></div>
    </section>
    </div>
  </main>
</div>
<script type="application/json" id="rows-json">{rows_json}</script>
<script type="application/json" id="summaries-json">{summaries_json}</script>
<script type="application/json" id="media-json">{media_json}</script>
<script type="application/json" id="cues-json">{cues_json}</script>
<script>
const ROWS = JSON.parse(document.getElementById("rows-json").textContent);
const SUMMARIES = JSON.parse(document.getElementById("summaries-json").textContent);
const MEDIA_BY_VIDEO = JSON.parse(document.getElementById("media-json").textContent);
const CUES_BY_VIDEO = JSON.parse(document.getElementById("cues-json").textContent);
const DATASET_ID = {dataset_id_json};
const STORAGE_VERSION = "cluster-display-seed-v1";
const STORAGE_KEY = `cueqc-cluster-audit:${{STORAGE_VERSION}}:${{DATASET_ID}}:${{location.pathname}}`;
const CLUSTER_STORAGE_KEY = STORAGE_KEY + ":cluster-labels-v1";
const CLUSTER_EXAMPLES_PER_CLUSTER = 3;
const CLUSTER_ACTIVE_KEY = STORAGE_KEY + ":cluster-active-v1";
const LEGACY_STORAGE_KEY = "cueqc-cluster-audit:" + location.pathname;
try {{
  localStorage.removeItem(LEGACY_STORAGE_KEY);
  localStorage.removeItem(LEGACY_STORAGE_KEY + ":custom-options");
  localStorage.removeItem(LEGACY_STORAGE_KEY + ":custom-group");
}} catch (_) {{}}
const DISPLAY = [{{value:"keep", label:"保留"}},{{value:"drop", label:"丢弃"}}];
const CLUSTER_DISPLAY = [{{value:"keep", label:"保留（进入字幕）"}},{{value:"drop", label:"丢弃（不进入字幕）"}}];
function clusterOrder(summary) {{
  const match = String(summary.cluster_id || "").match(/^cluster_(\\d+)$/);
  return match ? Number(match[1]) : 1000;
}}
const CLUSTER_ENTRIES = [...SUMMARIES].sort((a, b) => clusterOrder(a) - clusterOrder(b)).map((summary, index) => ({{
  summary,
  index,
  clusterId: summary.cluster_id || "",
  displayId: `cluster_${{String(index).padStart(2, "0")}}`
}}));
const CLUSTER_BY_ID = new Map(CLUSTER_ENTRIES.map(entry => [entry.clusterId, entry]));
let annotations = loadAnnotations();
let clusterAnnotations = loadClusterAnnotations();
let activeClusterId = loadActiveClusterId();
let filtered = [...ROWS];
let current = 0;
let activeVideo = "";
let playMode = "chunk";
const media = document.getElementById("media");
const labelStatus = document.getElementById("labelStatus");
const timeline = document.getElementById("timeline");
const chunkRange = document.getElementById("chunkRange");
const fallbackRange = document.getElementById("fallbackRange");
const cursor = document.getElementById("cursor");
function loadAnnotations() {{ try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }} catch (_) {{ return {{}}; }} }}
function saveAnnotations() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations)); }}
function loadClusterAnnotations() {{ try {{ return JSON.parse(localStorage.getItem(CLUSTER_STORAGE_KEY) || "{{}}"); }} catch (_) {{ return {{}}; }} }}
function saveClusterAnnotations() {{ localStorage.setItem(CLUSTER_STORAGE_KEY, JSON.stringify(clusterAnnotations)); }}
function loadActiveClusterId() {{ try {{ return localStorage.getItem(CLUSTER_ACTIVE_KEY) || ""; }} catch (_) {{ return ""; }} }}
function saveActiveClusterId(clusterId) {{ try {{ localStorage.setItem(CLUSTER_ACTIVE_KEY, clusterId || ""); }} catch (_) {{}} }}
function escapeHtml(text) {{ return String(text || "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch])); }}
function fmt(t) {{
  const v = Math.max(0, Number(t || 0));
  const h = Math.floor(v / 3600);
  const m = Math.floor((v - h * 3600) / 60);
  const s = (v - h * 3600 - m * 60).toFixed(2).padStart(5, "0");
  return h ? `${{h}}:${{String(m).padStart(2, "0")}}:${{s}}` : `${{m}}:${{s}}`;
}}
function activeCueText(videoId, t) {{
  const cues = CUES_BY_VIDEO[videoId] || [];
  const lines = [];
  for (const cue of cues) {{
    if (Number(cue.start) <= t && t <= Number(cue.end)) lines.push(cue.text);
  }}
  return lines.join("\\n");
}}
function rowText(row) {{
  return [row.sample_id,row.video_id,row.video_label,row.cluster_id,row.text,row.raw_text,JSON.stringify(row.qc||{{}}),JSON.stringify(row.text_features||{{}})].join("\\n").toLowerCase();
}}
function clusterDisplayId(index) {{
  return `cluster_${{String(index).padStart(2, "0")}}`;
}}
function getClusterEntry(clusterId) {{
  return CLUSTER_BY_ID.get(clusterId) || CLUSTER_ENTRIES[0] || null;
}}
function getActiveClusterEntry() {{
  if (activeClusterId && CLUSTER_BY_ID.has(activeClusterId)) return CLUSTER_BY_ID.get(activeClusterId) || null;
  return CLUSTER_ENTRIES[0] || null;
}}
function clusterRows(clusterId) {{
  return ROWS.filter(row => row.cluster_id === clusterId);
}}
function clusterRowsForDisplay(clusterId) {{
  return clusterRows(clusterId).sort((a, b) => {{
    const confidenceDelta = Number(b.cluster_confidence || 0) - Number(a.cluster_confidence || 0);
    if (Math.abs(confidenceDelta) > 1e-9) return confidenceDelta;
    const videoDelta = String(a.video_label || a.video_id || "").localeCompare(String(b.video_label || b.video_id || ""));
    if (videoDelta !== 0) return videoDelta;
    return Number(a.chunk_index || 0) - Number(b.chunk_index || 0);
  }});
}}
function clusterExampleRows(summary, limit = 3) {{
  const clusterId = summary.cluster_id || "";
  const rowsBySample = new Map(ROWS.map(row => [row.sample_id, row]));
  const preferred = (summary.examples || []).map(item => rowsBySample.get(item.sample_id)).filter(Boolean);
  const candidates = [...preferred, ...clusterRows(clusterId).sort((a, b) => Number(b.cluster_confidence || 0) - Number(a.cluster_confidence || 0))];
  const selected = [];
  const seenSamples = new Set();
  const seenVideos = new Set();
  const add = (row, preferNewVideo) => {{
    if (!row || seenSamples.has(row.sample_id)) return false;
    if (preferNewVideo && seenVideos.has(row.video_id || "")) return false;
    selected.push(row);
    seenSamples.add(row.sample_id);
    seenVideos.add(row.video_id || "");
    return true;
  }};
  for (const row of candidates) {{
    if (selected.length >= limit) break;
    add(row, true);
  }}
  for (const row of candidates) {{
    if (selected.length >= limit) break;
    add(row, false);
  }}
  return selected;
}}
function clusterSummaryText(summary) {{
  const density = summary.text_density_counts || {{}};
  const densityText = Object.entries(density).map(([k, v]) => `${{k}}=${{v}}`).join(", ");
  return `count=${{summary.count || 0}} · chars_avg=${{Number(summary.char_count_avg || 0).toFixed(2)}} · conf=${{Number(summary.confidence_avg || 0).toFixed(3)}} · density ${{densityText || "n/a"}}`;
}}
function setClusterButtons(rootId, options, clusterId, key) {{
  const root = document.getElementById(rootId);
  if (!root) return;
  root.innerHTML = "";
  const ann = clusterAnnotations[clusterId] || {{}};
  for (const option of options) {{
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = option.label;
    btn.className = ann[key] === option.value ? "active" : "";
    btn.onclick = () => {{
      updateClusterAnnotation(clusterId, key, option.value);
      renderClusterDetail();
    }};
    root.appendChild(btn);
  }}
}}
function updateClusterAnnotation(clusterId, key, value) {{
  clusterAnnotations[clusterId] = {{
    ...(clusterAnnotations[clusterId] || {{}}),
    [key]: value,
    updated_at: new Date().toISOString()
  }};
  saveClusterAnnotations();
  updateClusterProgress();
  renderClusterNav();
  if (clusterId === activeClusterId) {{
    const status = document.getElementById("clusterStatus");
    if (status) {{
      status.textContent = isClusterComplete(clusterId)
        ? "已选显示策略"
        : "先选显示策略（保留/丢弃），再看下面的音频。";
    }}
  }}
}}
function isClusterComplete(clusterId) {{
  const ann = clusterAnnotations[clusterId] || {{}};
  return ["keep", "drop"].includes(String(ann.display_decision || ""));
}}
function updateClusterProgress() {{
  const complete = SUMMARIES.filter(summary => isClusterComplete(summary.cluster_id)).length;
  const total = SUMMARIES.length;
  document.getElementById("clusterProgress").textContent = `${{complete}} / ${{total}} 簇已选显示策略`;
}}
function selectCluster(clusterId) {{
  const entry = getClusterEntry(clusterId);
  if (!entry) return;
  activeClusterId = entry.clusterId;
  saveActiveClusterId(activeClusterId);
  renderClusterNav();
  renderClusterDetail();
}}
function renderClusterNav() {{
  const root = document.getElementById("clusterNav");
  if (!root) return;
  root.innerHTML = "";
  CLUSTER_ENTRIES.forEach(entry => {{
    const ann = clusterAnnotations[entry.clusterId] || {{}};
    const button = document.createElement("button");
    button.type = "button";
    const classes = ["cluster-nav-button"];
    if (entry.clusterId === activeClusterId) classes.push("active");
    if (isClusterComplete(entry.clusterId)) classes.push("complete");
    button.className = classes.join(" ");
    button.innerHTML = `
      <div class="cluster-nav-top">
        <span class="cluster-nav-index">${{escapeHtml(entry.displayId)}}</span>
        <span class="badge">${{escapeHtml(String(entry.summary.count || 0))}} 条</span>
      </div>
      <div class="cluster-nav-sub">
        ${{escapeHtml(entry.clusterId)}}
        ${{ann.display_decision ? ` · ${{escapeHtml(ann.display_decision === "drop" ? "丢弃" : "保留")}}` : ""}}
      </div>`;
    button.onclick = () => selectCluster(entry.clusterId);
    root.appendChild(button);
  }});
  const summaryLine = document.getElementById("summaryLine");
  if (summaryLine) summaryLine.textContent = `${{ROWS.length}} 条样本 · ${{CLUSTER_ENTRIES.length}} 个簇`;
}}
function renderClusterAudioList(entry) {{
  const root = document.getElementById("clusterAudioList");
  if (!root || !entry) return;
  const examples = clusterRowsForDisplay(entry.clusterId);
  root.innerHTML = "";
  if (examples.length === 0) {{
    root.innerHTML = '<div class="meta">没有可展示的代表样本。</div>';
    return;
  }}
  const count = document.createElement("div");
  count.className = "meta";
  count.textContent = `共 ${{examples.length}} 条音频`;
  root.appendChild(count);
  for (const row of examples) {{
    const info = row.media || {{}};
    const card = document.createElement("div");
    card.className = "cluster-audio-card";
    const audioId = `audio-${{row.sample_id}}`;
    card.innerHTML = `
      <div class="cluster-audio-head">
        <strong>${{escapeHtml(row.video_label || row.video_id || "")}} · chunk ${{escapeHtml(row.chunk_index)}} · ${{Number(row.duration_s || 0).toFixed(2)}}s</strong>
        <span class="meta">${{escapeHtml(row.sample_id)}}</span>
      </div>
      <div class="cluster-audio-meta">${{fmt(row.start)}} - ${{fmt(row.end)}}${{info.audio_url ? "" : " · 音频缺失"}}${{info.vtt_url ? "" : " · 字幕缺失"}}</div>
      <div id="${{audioId}}"></div>
      <div class="cluster-audio-text">${{escapeHtml(row.text_preview || row.text || row.raw_text || "(empty)")}}</div>
    `;
    root.appendChild(card);
    const slot = card.querySelector(`#${{CSS.escape(audioId)}}`);
    if (slot) {{
      if (info.audio_url) {{
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "metadata";
        const source = document.createElement("source");
        source.src = info.audio_url;
        source.type = info.audio_mime || "audio/wav";
        audio.appendChild(source);
        if (info.vtt_url) {{
          const track = document.createElement("track");
          track.kind = "subtitles";
          track.label = "完整日语字幕";
          track.srclang = "ja";
          track.src = info.vtt_url;
          track.default = true;
          audio.appendChild(track);
        }}
        const start = Number(row.start || 0);
        const end = Number(row.end || 0);
        const seekToChunkStart = () => {{
          try {{
            if (Number.isFinite(start) && start >= 0 && start < audio.duration) {{
              audio.currentTime = start;
            }}
          }} catch (_) {{}}
        }};
        audio.addEventListener("loadedmetadata", seekToChunkStart, {{ once: true }});
        audio.addEventListener("play", () => {{
          if (Number.isFinite(start) && Number.isFinite(end) && end > start) {{
            if (audio.currentTime < start || audio.currentTime >= end) seekToChunkStart();
          }}
        }});
        audio.addEventListener("timeupdate", () => {{
          if (Number.isFinite(end) && end > start && audio.currentTime >= end) audio.pause();
        }});
        slot.appendChild(audio);
      }} else {{
        slot.textContent = "音频文件未找到。";
      }}
    }}
  }}
}}
function renderClusterDetail() {{
  const entry = getActiveClusterEntry();
  if (!entry) return;
  const summary = entry.summary || {{}};
  const ann = clusterAnnotations[entry.clusterId] || {{}};
  const title = document.getElementById("clusterTitle");
  if (title) title.textContent = `${{entry.displayId}}`;
  const count = document.getElementById("clusterCount");
  if (count) count.textContent = `${{summary.count || 0}} 条 · ${{entry.clusterId}}`;
  const summaryNode = document.getElementById("clusterSummary");
  if (summaryNode) summaryNode.textContent = clusterSummaryText(summary);
  setClusterButtons("clusterDisplayButtons", CLUSTER_DISPLAY, entry.clusterId, "display_decision");
  const reasonInput = document.getElementById("activeClusterReason");
  if (reasonInput) {{
    reasonInput.value = ann.classification_reason || "";
    reasonInput.oninput = () => updateClusterAnnotation(entry.clusterId, "classification_reason", reasonInput.value);
  }}
  const status = document.getElementById("clusterStatus");
  if (status) {{
    status.textContent = isClusterComplete(entry.clusterId)
      ? "已选显示策略"
      : "先选显示策略（保留/丢弃），再看下面的音频。";
  }}
  renderClusterAudioList(entry);
}}
function selectSample(sampleId, autoplay = true) {{
  const row = ROWS.find(item => item.sample_id === sampleId);
  if (!row) return;
  document.getElementById("search").value = "";
  document.getElementById("cluster").value = row.cluster_id || "";
  filtered = ROWS.filter(item => item.cluster_id === row.cluster_id);
  current = ROWS.indexOf(row);
  renderList();
  renderCurrent(true);
  document.getElementById("sampleTitle").scrollIntoView({{block: "start"}});
  if (autoplay) setTimeout(() => playCurrent("chunk"), 0);
}}
function renderClusterReview() {{
  const root = document.getElementById("clusterReview");
  root.innerHTML = "";
  for (const summary of SUMMARIES) {{
    const clusterId = summary.cluster_id || "";
    const ann = clusterAnnotations[clusterId] || {{}};
    const examples = clusterExampleRows(summary, CLUSTER_EXAMPLES_PER_CLUSTER);
    const card = document.createElement("div");
    const cardClasses = ["cluster-card"];
    if (clusterId === (ROWS[current] || {{}}).cluster_id) cardClasses.push("active");
    if (isClusterComplete(clusterId)) cardClasses.push("complete");
    card.className = cardClasses.join(" ");
    card.innerHTML = `
      <div class="cluster-card-head">
        <div class="cluster-title">${{escapeHtml(clusterId)}} <span class="badge">${{escapeHtml(String(summary.count || 0))}} 条</span></div>
        <div class="meta">${{escapeHtml(clusterSummaryText(summary))}}</div>
      </div>
      <div class="cluster-examples">
        ${{examples.map(row => `
          <button class="cluster-example" type="button" data-example-sample="${{escapeHtml(row.sample_id)}}">
            <strong>${{escapeHtml(row.video_label || row.video_id || "")}} · chunk ${{escapeHtml(row.chunk_index)}}</strong>
            <div class="meta">${{fmt(row.start)}}-${{fmt(row.end)}} · ${{Number(row.duration_s || 0).toFixed(2)}}s</div>
            <div>${{escapeHtml(row.text_preview || row.text || "(empty)")}}</div>
          </button>
        `).join("")}}
      </div>
      <div class="cluster-fields">
        <input data-cluster-label="${{escapeHtml(clusterId)}}" placeholder="簇标签" value="${{escapeHtml(ann.cluster_label || "")}}">
        <textarea data-cluster-reason="${{escapeHtml(clusterId)}}" placeholder="分类原因 / 共同特征 / 例外">${{escapeHtml(ann.classification_reason || "")}}</textarea>
      </div>`;
    root.appendChild(card);
  }}
  root.querySelectorAll("[data-example-sample]").forEach(button => {{
    button.onclick = () => selectSample(button.getAttribute("data-example-sample") || "", true);
  }});
  root.querySelectorAll("[data-cluster-label]").forEach(input => {{
    input.oninput = () => updateClusterAnnotation(input.getAttribute("data-cluster-label") || "", "cluster_label", input.value);
  }});
  root.querySelectorAll("[data-cluster-reason]").forEach(input => {{
    input.oninput = () => updateClusterAnnotation(input.getAttribute("data-cluster-reason") || "", "classification_reason", input.value);
  }});
  updateClusterProgress();
}}
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
  renderCurrent(false);
}}
function renderList() {{
  const root = document.getElementById("list");
  root.innerHTML = "";
  for (const row of filtered) {{
    const div = document.createElement("div");
    const cueFeatures = row.cue_features || {{}};
    const density = ((cueFeatures.text_density || row.text_density || {{}}).level) || "";
    div.className = "item" + (ROWS[current] === row ? " active" : "");
    div.onclick = () => {{ current = ROWS.indexOf(row); renderList(); renderCurrent(true); }};
    const badgeClass = row.cluster_noise ? "badge noise" : "badge";
    div.innerHTML = `<div class="item-title"><strong>${{escapeHtml(row.cluster_id)}} · chunk ${{row.chunk_index}}</strong><span class="${{badgeClass}}">${{escapeHtml(row.video_label || row.video_id || "")}}</span></div>
      <div class="meta">${{fmt(row.start)}}-${{fmt(row.end)}} · ${{Number(row.duration_s||0).toFixed(2)}}s · ${{escapeHtml(density)}}</div>
      <div class="meta">${{escapeHtml(row.text_preview || row.text || "(empty)")}}</div>`;
    root.appendChild(div);
  }}
  document.getElementById("summaryLine").textContent = `${{filtered.length}} / ${{ROWS.length}} samples · ${{SUMMARIES.length}} clusters`;
}}
function fixedOptions(options) {{
  const merged = [];
  const seen = new Set();
  for (const option of options) {{
    const value = String(option.value || "").trim();
    const label = String(option.label || value).trim();
    if (!value || seen.has(value)) continue;
    seen.add(value);
    merged.push({{value, label}});
  }}
  return merged;
}}
function labelForValue(key, value, options) {{
  const found = fixedOptions(options).find(option => option.value === value);
  return found ? found.label : value;
}}
function setAnnotation(row, key, value, label) {{
  const next = {{
    ...(annotations[row.sample_id] || {{}}),
    [key]: value,
    [`${{key}}_zh`]: label || value,
    updated_at: new Date().toISOString()
  }};
  annotations[row.sample_id] = {{
    ...next
  }};
  saveAnnotations();
}}
function advanceFromRow(row) {{
  const pos = filtered.indexOf(row);
  if (pos >= 0 && pos < filtered.length - 1) {{
    current = ROWS.indexOf(filtered[pos + 1]);
    renderList();
    renderCurrent(true);
  }} else {{
    renderCurrent(false);
  }}
}}
function shouldAdvanceAfterRequiredLabels(row) {{
  const ann = annotations[row.sample_id] || {{}};
  return !!ann.display_decision;
}}
function markDropAndAdvance(row) {{
  const existing = annotations[row.sample_id] || {{}};
  annotations[row.sample_id] = {{
    ...existing,
    display_decision: "drop",
    display_decision_zh: "丢弃",
    terminal_decision: true,
    terminal_reason: "display_drop",
    updated_at: new Date().toISOString()
  }};
  saveAnnotations();
  advanceFromRow(row);
}}
function setButtons(rootId, options, key) {{
  const root = document.getElementById(rootId);
  if (!root) return;
  root.innerHTML = "";
  const row = ROWS[current];
  const ann = annotations[row.sample_id] || {{}};
  for (const option of fixedOptions(options)) {{
    const btn = document.createElement("button");
    btn.textContent = option.label;
    btn.title = `${{option.label}} / ${{option.value}}`;
    btn.className = ann[key] === option.value ? "active" : "";
    btn.onclick = () => {{
      if (key === "display_decision" && option.value === "drop") {{
        markDropAndAdvance(row);
        return;
      }}
      setAnnotation(row, key, option.value, option.label);
      if (shouldAdvanceAfterRequiredLabels(row)) {{
        if ((annotations[row.sample_id] || {{}}).display_decision === "drop") {{
          annotations[row.sample_id] = {{
            ...(annotations[row.sample_id] || {{}}),
            terminal_decision: true,
            terminal_reason: "display_drop",
            qc_ignored: true,
            needs_quality_label: false,
            updated_at: new Date().toISOString()
          }};
          saveAnnotations();
        }}
        advanceFromRow(row);
      }} else {{
        renderCurrent(false);
      }}
    }};
    root.appendChild(btn);
  }}
}}
function setMediaForItem(row, seek) {{
  const videoId = row.video_id || "";
  const info = MEDIA_BY_VIDEO[videoId] || {{}};
  const error = document.getElementById("mediaError");
  error.textContent = "";
  if (activeVideo !== videoId) {{
    media.pause();
    media.innerHTML = "";
    if (info.audio_url) {{
      const source = document.createElement("source");
      source.src = info.audio_url;
      source.type = info.audio_mime || "audio/wav";
      media.appendChild(source);
    }}
    if (info.vtt_url) {{
      const track = document.createElement("track");
      track.kind = "subtitles";
      track.label = "完整日语字幕";
      track.srclang = "ja";
      track.src = info.vtt_url;
      track.default = true;
      media.appendChild(track);
    }}
    activeVideo = videoId;
    media.load();
    try {{ for (const track of media.textTracks) track.mode = "hidden"; }} catch (_) {{}}
  }}
  document.getElementById("mediaLink").href = info.audio_url || "#";
  document.getElementById("vttLink").href = info.vtt_url || "#";
  if (!info.audio_url) error.textContent = "音频文件未找到。";
  if (!info.vtt_url) error.textContent = [error.textContent, "字幕 VTT 未找到。"].filter(Boolean).join(" ");
  if (seek && info.audio_url) media.currentTime = Number(row.start || 0);
}}
function setMetrics(row) {{
  const tf = row.text_features || {{}};
  const cueFeatures = row.cue_features || {{}};
  const mediaInfo = row.media || {{}};
  const rows = [
    ["video", `${{row.video_label || ""}} · ${{row.video_id || ""}}`],
    ["cluster", `${{row.cluster_id}} · confidence=${{Number(row.cluster_confidence || 0).toFixed(3)}}`],
    ["chunk", `${{row.chunk_index}} · ${{fmt(row.start)}}-${{fmt(row.end)}}`],
    ["context", `${{fmt(row.context_start)}}-${{fmt(row.context_end)}}`],
    ["字幕时间轴窗口", `${{fmt(row.fallback_window_start)}}-${{fmt(row.fallback_window_end)}}`],
    ["density", JSON.stringify(cueFeatures.text_density || row.text_density || {{}})],
    ["chars", `${{tf.char_count || 0}} unique=${{tf.unique_chars || 0}} kana=${{tf.kana_ratio || 0}} kanji=${{tf.kanji_ratio || 0}}`],
    ["repeat", JSON.stringify(tf.repeat_profile || {{}})],
    ["adjacency", JSON.stringify(row.adjacency || {{}})],
    ["audio", mediaInfo.audio_path || ""],
    ["subtitle", mediaInfo.srt_path || ""]
  ];
  document.getElementById("metrics").innerHTML = rows.map(([k,v]) => `<div>${{escapeHtml(k)}}</div><div>${{escapeHtml(v)}}</div>`).join("");
}}
function renderCueGroup(rootId, cues, emptyText) {{
  const root = document.getElementById(rootId);
  if (!cues || cues.length === 0) {{
    root.innerHTML = `<div class="meta">${{escapeHtml(emptyText)}}</div>`;
    return;
  }}
  const t = media.currentTime || 0;
  root.innerHTML = cues.map(cue => {{
    const active = Number(cue.start) <= t && t <= Number(cue.end) ? " active" : "";
    return `<div class="cue${{active}}"><div class="meta">${{fmt(cue.start)}}-${{fmt(cue.end)}}</div><div>${{escapeHtml(cue.text)}}</div></div>`;
  }}).join("");
}}
function renderCueLists(row) {{
  renderCueGroup("chunkCueList", row.chunk_subtitle_cues, "该 chunk 内没有字幕 cue。");
  renderCueGroup("fallbackCueList", row.fallback_subtitle_cues, "该字幕时间轴窗口内没有字幕 cue。");
  renderCueGroup("contextCueList", row.context_subtitle_cues, "该上下文内没有字幕 cue。");
  renderCueGroup("alignedList", row.aligned_segments, "该 chunk 内没有字幕片段。");
}}
function updateTimeline() {{
  const row = ROWS[current];
  if (!row) return;
  const width = Math.max(0.001, Number(row.context_end) - Number(row.context_start));
  const setRange = (el, start, end, minWidth) => {{
    const left = Math.max(0, Math.min(100, ((Number(start) - Number(row.context_start)) / width) * 100));
    const right = Math.max(0, Math.min(100, ((Number(end) - Number(row.context_start)) / width) * 100));
    el.style.left = `${{left}}%`;
    el.style.width = `${{Math.max(minWidth, right - left)}}%`;
  }};
  setRange(chunkRange, row.start, row.end, 0.5);
  setRange(fallbackRange, row.fallback_window_start, row.fallback_window_end, 0.5);
  const t = media.currentTime || 0;
  const pct = Math.max(0, Math.min(100, ((t - Number(row.context_start)) / width) * 100));
  cursor.style.left = `${{pct}}%`;
  document.getElementById("nowText").textContent = fmt(t);
  document.getElementById("captionOverlay").textContent = activeCueText(row.video_id || "", t);
  const stopAt = playMode === "context" ? Number(row.context_end) : Number(row.end);
  if (!media.paused && t >= stopAt) media.pause();
  renderCueLists(row);
}}
function renderCurrent(seek) {{
  const row = ROWS[current];
  if (!row) return;
  document.getElementById("sampleTitle").textContent = `${{row.cluster_id}} · ${{row.sample_id}}`;
  document.getElementById("sampleMeta").textContent = `${{row.video_label || row.video_id || ""}} · chunk ${{row.chunk_index}} · ${{fmt(row.start)}}-${{fmt(row.end)}} · duration ${{Number(row.duration_s||0).toFixed(3)}}s`;
  document.getElementById("text").textContent = row.text || row.raw_text || "(empty)";
  document.getElementById("rangeStart").textContent = fmt(row.context_start);
  document.getElementById("rangeEnd").textContent = fmt(row.context_end);
  setMediaForItem(row, seek);
  setMetrics(row);
  renderCueLists(row);
  setButtons("displayButtons", DISPLAY, "display_decision");
  updateTimeline();
  renderClusterReview();
}}
function playCurrent(mode) {{
  const row = ROWS[current];
  if (!row) return;
  playMode = mode;
  setMediaForItem(row, false);
  media.currentTime = mode === "context" ? Number(row.context_start) : Number(row.start);
  media.play().catch(() => {{}});
}}
function exportClusterRows() {{
  return SUMMARIES.map(summary => {{
    const clusterId = summary.cluster_id || "";
    const ann = clusterAnnotations[clusterId] || {{}};
    const examples = clusterExampleRows(summary, CLUSTER_EXAMPLES_PER_CLUSTER).map(row => ({{
      sample_id: row.sample_id,
      video_id: row.video_id,
      video_label: row.video_label,
      chunk_index: row.chunk_index,
      start: row.start,
      end: row.end,
      duration_s: row.duration_s,
      text: row.text,
      raw_text: row.raw_text,
      text_preview: row.text_preview
    }}));
    return {{
      schema: "cueqc_cluster_label_v1",
      dataset_id: DATASET_ID,
      cluster_id: clusterId,
      display_decision: ann.display_decision || "",
      notes: ann.classification_reason || "",
      updated_at: ann.updated_at || "",
      count: summary.count || 0,
      char_count_avg: summary.char_count_avg || 0,
      confidence_avg: summary.confidence_avg || 0,
      text_density_counts: summary.text_density_counts || {{}},
      examples
    }};
  }});
}}
function downloadClusterJsonl() {{
  const blob = new Blob([exportClusterRows().map(row => JSON.stringify(row)).join("\\n") + "\\n"], {{type:"application/jsonl;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "cueqc_cluster_labels.jsonl";
  a.click();
  URL.revokeObjectURL(a.href);
}}
document.getElementById("search").addEventListener("input", applyFilters);
document.getElementById("cluster").addEventListener("change", applyFilters);
document.getElementById("downloadClusters").onclick = downloadClusterJsonl;
document.getElementById("playChunkBtn").onclick = () => playCurrent("chunk");
document.getElementById("playContextBtn").onclick = () => playCurrent("context");
document.getElementById("pauseBtn").onclick = () => media.pause();
document.getElementById("replayBtn").onclick = () => playCurrent(playMode);
document.getElementById("prevBtn").onclick = () => {{
  const pos = filtered.indexOf(ROWS[current]);
  if (pos > 0) current = ROWS.indexOf(filtered[pos - 1]);
  renderList();
  renderCurrent(true);
}};
document.getElementById("nextBtn").onclick = () => {{
  const pos = filtered.indexOf(ROWS[current]);
  if (pos >= 0 && pos < filtered.length - 1) current = ROWS.indexOf(filtered[pos + 1]);
  renderList();
  renderCurrent(true);
}};
media.addEventListener("timeupdate", updateTimeline);
media.addEventListener("seeked", updateTimeline);
media.addEventListener("loadedmetadata", updateTimeline);
media.addEventListener("error", () => {{
  document.getElementById("mediaError").textContent = "媒体无法加载。";
}});
timeline.addEventListener("click", event => {{
  const row = ROWS[current];
  if (!row) return;
  const rect = timeline.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  media.currentTime = Number(row.context_start) + ratio * (Number(row.context_end) - Number(row.context_start));
}});
if (!activeClusterId || !CLUSTER_BY_ID.has(activeClusterId)) {{
  activeClusterId = (CLUSTER_ENTRIES[0] && CLUSTER_ENTRIES[0].clusterId) || "";
}}
renderClusterNav();
renderClusterDetail();
updateClusterProgress();
</script>
</body>
</html>
"""


def build_audit(
    *,
    clusters_jsonl: Path,
    summaries_jsonl: Path,
    baseline_root: Path,
    output_dir: Path,
    title: str,
    dataset_id: str,
    summary_json: Path | None = None,
    refresh_nav: bool = False,
) -> dict[str, Any]:
    rows = read_jsonl(clusters_jsonl)
    summaries = read_jsonl(summaries_jsonl)
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
        _page_html(
            title=title,
            dataset_id=dataset_id,
            rows=rows,
            summaries=summaries,
            media_by_video=media_by_video,
            cues_by_video=cues_by_video,
        ),
        encoding="utf-8",
    )

    summary_source = summary_json if summary_json else output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_source.exists():
        loaded = read_json(summary_source)
        if isinstance(loaded, dict):
            summary.update(loaded)
    missing_media = [video_id for video_id, info in media_by_video.items() if not info.get("audio_exists")]
    missing_subtitles = [video_id for video_id, info in media_by_video.items() if not info.get("subtitle_exists")]
    summary.update(
        {
            "dataset_id": dataset_id,
            "title": title,
            "review_item_count": len(rows),
            "html": project_rel(index_path),
            "source_clusters": project_rel(clusters_jsonl),
            "source_cluster_summaries": project_rel(summaries_jsonl),
            "baseline_root": project_rel(baseline_root),
            "media_enabled": True,
            "media_mode": "audio",
            "media_by_video": media_by_video,
            "subtitle_vtt_by_video": {
                video_id: info.get("vtt_path", "") for video_id, info in media_by_video.items()
            },
            "subtitle_cue_counts_by_video": {
                video_id: info.get("subtitle_cue_count", 0) for video_id, info in media_by_video.items()
            },
            "aligned_segment_counts_by_video": {
                video_id: info.get("aligned_segment_count", 0) for video_id, info in media_by_video.items()
            },
            "missing_media_videos": missing_media,
            "missing_subtitle_videos": missing_subtitles,
            "audit_generator": "tools/audits/generate_cueqc_cluster_audit_html.py",
            "label_schema": "cueqc_cluster_labels.jsonl",
            "label_schema_version": "cluster_display_seed_v1",
            "cluster_label_schema": "cueqc_cluster_label_v1",
            "cluster_label_export": "cueqc_cluster_labels.jsonl",
            "cluster_label_storage_suffix": "cluster-labels-v1",
            "cluster_display_decision_options": ["keep", "drop"],
            "cluster_display_decision_required": True,
            "cluster_review_enabled": True,
            "cluster_review_group_count": len(summaries),
            "cluster_review_examples_per_cluster": 3,
            "cluster_review_layout": "left_nav_single_cluster_audio_list_all_v2",
            "cluster_review_audio_render_mode": "all",
            "cluster_review_scope": "natural_clusters_including_noise",
            "training_label_fields": ["display_decision"],
        }
    )
    write_json(output_dir / "summary.json", summary)
    if refresh_nav:
        update_audit_entrypoints(latest_html=index_path, title=title)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CueQC cluster audit page with synchronized audio and subtitles.")
    parser.add_argument("--clusters", required=True, help="cueqc_clusters.jsonl")
    parser.add_argument("--summaries", required=True, help="cueqc_cluster_summaries.jsonl")
    parser.add_argument("--baseline-root", required=True, help="Baseline run root containing archived/ and jobs/")
    parser.add_argument("--output-dir", required=True, help="Audit output directory")
    parser.add_argument("--title", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--summary-json", help="Optional existing summary JSON to merge")
    parser.add_argument("--refresh-nav", action="store_true", help="Rebuild agents/audits index and latest-audit.html")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_audit(
        clusters_jsonl=project_path(args.clusters),
        summaries_jsonl=project_path(args.summaries),
        baseline_root=project_path(args.baseline_root),
        output_dir=project_path(args.output_dir),
        title=args.title,
        dataset_id=args.dataset_id,
        summary_json=project_path(args.summary_json) if args.summary_json else None,
        refresh_nav=args.refresh_nav,
    )
    print(json.dumps({"ok": True, "html": summary.get("html"), "review_item_count": summary.get("review_item_count")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
