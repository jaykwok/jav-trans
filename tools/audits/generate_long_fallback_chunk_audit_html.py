#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints

TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}


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
    return "audio/mp4"


def video_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".mkv":
        return "video/x-matroska"
    return "video/mp4"


def materialize_audio_media(
    *,
    source_path: Path,
    output_dir: Path,
    ffmpeg_bin: str,
) -> Path:
    if source_path.suffix.lower() in AUDIO_SUFFIXES:
        return source_path
    output_path = output_dir / "audit_audio.m4a"
    if output_path.exists() and output_path.stat().st_mtime >= source_path.stat().st_mtime:
        return output_path
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"ffmpeg not found: {ffmpeg_bin}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg audio extraction failed: {' '.join(cmd)}") from exc
    return output_path


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
            -(row_float(row, "fallback_duration_s") or row_float(row, "duration_s")),
            row_float(row, "fallback_window_start") or row_float(row, "start"),
            int(row.get("chunk_index") or 0),
        ),
    )
    items: list[dict[str, Any]] = []
    for row in sorted_rows:
        padded_start = row_float(row, "start")
        padded_end = row_float(row, "end")
        if padded_end <= padded_start:
            continue
        core_start = row_float(row, "core_start") or padded_start
        core_end = row_float(row, "core_end") or padded_end
        fallback_start = row_float(row, "fallback_window_start") or core_start
        fallback_end = row_float(row, "fallback_window_end") or core_end
        if fallback_end <= fallback_start:
            fallback_start = padded_start
            fallback_end = padded_end
        fallback_duration_s = row_float(row, "fallback_duration_s") or (fallback_end - fallback_start)
        padded_duration_s = row_float(row, "duration_s") or (padded_end - padded_start)
        context_start = max(0.0, min(padded_start, fallback_start) - context_pad_s)
        context_end = max(padded_end, fallback_end) + context_pad_s
        cue_rows = cues_for_range(subtitle_cues, fallback_start, fallback_end)
        items.append(
            {
                "index": len(items),
                "sample_id": f"fallback-window-{len(items):03d}-chunk{int(row.get('chunk_index') or 0):04d}",
                "video_label": "匿名样片 A",
                "chunk_index": int(row.get("chunk_index") or 0),
                "start": round(fallback_start, 3),
                "end": round(fallback_end, 3),
                "duration_s": round(fallback_duration_s, 3),
                "fallback_window_start": round(fallback_start, 3),
                "fallback_window_end": round(fallback_end, 3),
                "fallback_window_source": str(row.get("fallback_window_source") or ""),
                "fallback_duration_s": round(fallback_duration_s, 3),
                "padded_start": round(padded_start, 3),
                "padded_end": round(padded_end, 3),
                "padded_duration_s": round(padded_duration_s, 3),
                "context_start": round(context_start, 3),
                "context_end": round(context_end, 3),
                "core_start": round(core_start, 3),
                "core_end": round(core_end, 3),
                "core_duration_s": round(row_float(row, "core_duration_s") or (core_end - core_start), 3),
                "split_reason": str(row.get("split_reason") or ""),
                "alignment_quality": str(row.get("alignment_quality") or ""),
                "fallback_subtype": str(row.get("fallback_subtype") or row.get("fallback_reason") or ""),
                "failure_bucket": str(row.get("failure_bucket") or ""),
                "fallback_safe": bool(row.get("fallback_safe")),
                "speech_island_count": int(row.get("speech_island_count") or 0),
                "internal_gap_count": int(row.get("internal_gap_count") or 0),
                "internal_gap_max_s": round(row_float(row, "internal_gap_max_s"), 3),
                "internal_gap_total_s": round(row_float(row, "internal_gap_total_s"), 3),
                "longest_silence_s": round(row_float(row, "longest_silence_s"), 3),
                "total_silence_s": round(row_float(row, "total_silence_s"), 3),
                "silence_ratio": row.get("silence_ratio"),
                "risk_reasons": list(row.get("risk_reasons") or []),
                "display_text": compact_text(row.get("display_text") or ""),
                "subtitle_text": cue_text_for_range(subtitle_cues, fallback_start, fallback_end),
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>最新审计页入口</title>
</head>
<body>
<p>自动跳转已关闭。打开 <a href="{html.escape(latest_rel)}">{html.escape(title)}</a>。</p>
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
<title>SpeechBoundary-JA 审计入口</title>
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
  <h1>SpeechBoundary-JA 审计入口</h1>
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
    update_audit_entrypoints(latest_html=output_dir / "index.html", title=title)


def page_template(
    *,
    title: str,
    media_url: str,
    media_mode: str,
    media_mime: str,
    vtt_url: str,
    items: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    items_json = json.dumps(items, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    cues_json = json.dumps(cues, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    summary_json = json.dumps(summary, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    media_kind_label = "音频" if media_mode == "audio" else "视频"
    direct_media_label = "直接打开音频" if media_mode == "audio" else "直接打开视频"
    if media_mode == "audio":
        media_markup = f"""
        <div class="media-shell audio-shell">
          <audio id="media" controls preload="metadata">
            <source src="{html.escape(media_url)}" type="{html.escape(media_mime)}">
            <track kind="subtitles" label="完整日语字幕" srclang="ja" src="{html.escape(vtt_url)}" default>
          </audio>
        </div>
        <div class="audio-caption-preview" id="captionOverlay"></div>
"""
    else:
        media_markup = f"""
        <div class="media-shell video-shell">
          <video id="media" controls preload="metadata" playsinline>
            <source src="{html.escape(media_url)}" type="{html.escape(media_mime)}">
            <track kind="subtitles" label="完整日语字幕" srclang="ja" src="{html.escape(vtt_url)}" default>
          </video>
          <div class="caption-overlay" id="captionOverlay"></div>
        </div>
"""
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
.media-shell {{ position: relative; background: #070a09; border-radius: 8px; overflow: hidden; }}
video {{ display: block; width: 100%; max-height: 66vh; background: #070a09; }}
audio {{ display: block; width: 100%; min-height: 48px; background: #fff; }}
.audio-shell {{ background: #fff; border: 1px solid var(--line); padding: 12px; }}
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
.audio-caption-preview {{
  margin-top: 10px;
  min-height: 64px;
  max-height: 160px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fbfbf9;
  padding: 10px 12px;
  font-size: 17px;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.audio-caption-preview:empty::before {{ content: "当前没有同步字幕"; color: var(--muted); font-size: 13px; }}
.timeline {{ position: relative; height: 48px; border: 1px solid var(--line); border-radius: 6px; background: #101715; overflow: hidden; cursor: pointer; margin-top: 10px; }}
.fallback-range {{ position: absolute; top: 0; bottom: 0; background: rgba(15, 118, 110, 0.55); }}
.padded-range {{ position: absolute; top: 4px; bottom: 4px; border: 1px solid rgba(255,255,255,0.32); background: rgba(255, 255, 255, 0.08); }}
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
          <option value="duration">按 fallback 时长</option>
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
        <button class="primary" id="playChunkBtn">播放 fallback 窗口</button>
        <button id="playContextBtn">播放上下文</button>
        <button id="downloadBtn">下载审计 JSONL</button>
      </div>
    </div>
    <div class="grid">
      <section class="panel">
        {media_markup}
        <div class="timeline" id="timeline">
          <div class="padded-range" id="paddedRange"></div>
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
          <a href="{html.escape(media_url)}">{html.escape(direct_media_label)}</a>
          <a href="{html.escape(vtt_url)}">完整 VTT</a>
        </div>
        <p class="hint">绿色是实际 fallback 时间窗，浅色内条是 speech core，暗色外框是 ASR padded chunk；播放会在当前模式终点自动暂停。{html.escape(media_kind_label)}下方/画面字幕为完整日语字幕当前 cue。</p>
        <p class="hint error" id="mediaError"></p>
      </section>
      <aside class="panel">
        <h3>fallback 窗口指标</h3>
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
const STORAGE_KEY = "speech-boundary-ja-long-fallback-audit:" + SUMMARY.label_schema_version + ":" + SUMMARY.dataset_id;
const LABELS = [
  ["timing_accurate", "时间轴准确"],
  ["needs_realign", "时间轴需重对齐"],
  ["timing_start_early", "开头偏早"],
  ["timing_start_late", "开头偏晚"],
  ["timing_end_early", "结尾偏早"],
  ["timing_end_late", "结尾偏晚"],
  ["timing_window_too_long", "时间窗过长"],
  ["timing_window_too_short", "时间窗过短"],
  ["timing_crosses_gap_noise", "跨无声/噪声"],
  ["text_ok", "文本可用"],
  ["needs_split", "需要继续切分"],
  ["ok_long_utterance", "确实长句"],
  ["multiple_islands", "多段语音揉一起"],
  ["gap_or_noise", "中间噪音/无声"],
  ["low_info_vocal", "低信息人声"],
  ["non_speech", "无字幕价值"],
  ["bad_asr", "ASR 文本异常"],
  ["uncertain", "不确定"]
];
let currentIndex = 0;
let filtered = [...ITEMS];
let playMode = "chunk";
let annotations = loadAnnotations();
const media = document.getElementById("media");
const itemList = document.getElementById("itemList");
const searchInput = document.getElementById("searchInput");
const sortSelect = document.getElementById("sortSelect");
const notes = document.getElementById("notes");
const timeline = document.getElementById("timeline");
const paddedRange = document.getElementById("paddedRange");
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
  else if (mode === "islands") cloned.sort((a, b) => (b.speech_island_count - a.speech_island_count) || (b.fallback_duration_s - a.fallback_duration_s));
  else cloned.sort((a, b) => (b.fallback_duration_s - a.fallback_duration_s) || (a.start - b.start));
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
      <div class="item-title">#${{item.index + 1}} chunk ${{item.chunk_index}} <span class="badge danger">fallback ${{Number(item.fallback_duration_s).toFixed(2)}}s</span></div>
      <div class="meta">${{fmt(item.start)}}-${{fmt(item.end)}} · padded ${{Number(item.padded_duration_s).toFixed(2)}}s · island=${{item.speech_island_count}} · gap=${{item.internal_gap_count}} · ${{escapeHtml(item.split_reason)}}</div>
      <div class="meta">${{escapeHtml(item.subtitle_text || item.display_text || "(empty)")}}</div>
    `;
    itemList.appendChild(div);
  }}
}}
function setMetrics(item) {{
  const rows = [
    ["媒体", item.video_label],
    ["chunk", item.chunk_index],
    ["fallback 时间窗", `${{fmt(item.fallback_window_start)}}-${{fmt(item.fallback_window_end)}}`],
    ["fallback 时长", `${{Number(item.fallback_duration_s).toFixed(2)}}s`],
    ["fallback source", item.fallback_window_source],
    ["padded chunk", `${{fmt(item.padded_start)}}-${{fmt(item.padded_end)}} (${{Number(item.padded_duration_s).toFixed(2)}}s)`],
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
  const labels = Array.isArray(ann.labels) ? ann.labels : (ann.label ? [ann.label] : []);
  for (const [value, label] of LABELS) {{
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.className = labels.includes(value) ? "active" : "";
    btn.onclick = () => {{
      const current = new Set(Array.isArray((annotations[item.sample_id] || {{}}).labels) ? annotations[item.sample_id].labels : []);
      if (current.has(value)) current.delete(value);
      else current.add(value);
      annotations[item.sample_id] = {{
        ...(annotations[item.sample_id] || {{}}),
        labels: Array.from(current),
        updated_at: new Date().toISOString()
      }};
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
  setRange(paddedRange, item.padded_start, item.padded_end, 0.5);
  setRange(fallbackRange, item.fallback_window_start, item.fallback_window_end, 0.5);
  setRange(coreRange, item.core_start || item.start, item.core_end || item.end, 0.5);
  const t = media.currentTime || 0;
  const pct = Math.max(0, Math.min(100, ((t - item.context_start) / width) * 100));
  cursor.style.left = `${{pct}}%`;
  document.getElementById("nowText").textContent = fmt(t);
  document.getElementById("captionOverlay").textContent = activeCueText(t);
  const stopAt = playMode === "context" ? item.context_end : item.fallback_window_end;
  if (!media.paused && t >= stopAt) media.pause();
}}
function renderCurrent(seek) {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  document.getElementById("clipTitle").textContent = `#${{item.index + 1}} fallback window chunk ${{item.chunk_index}}`;
  document.getElementById("clipMeta").textContent = `${{fmt(item.fallback_window_start)}}-${{fmt(item.fallback_window_end)}} · ${{Number(item.fallback_duration_s).toFixed(2)}}s · padded ${{Number(item.padded_duration_s).toFixed(2)}}s · ${{item.fallback_subtype || "fallback"}}`;
  document.getElementById("rangeStart").textContent = fmt(item.context_start);
  document.getElementById("rangeEnd").textContent = fmt(item.context_end);
  document.getElementById("displayText").textContent = item.display_text || "(empty)";
  setMetrics(item);
  renderCueList(item);
  renderLabels(item);
  if (seek) media.currentTime = item.fallback_window_start;
  updateTimeline();
}}
function playCurrent(mode) {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  playMode = mode;
  media.currentTime = mode === "context" ? item.context_start : item.fallback_window_start;
  media.play().catch(() => {{}});
}}
function exportRows() {{
  return ITEMS.map(item => ({{
    sample_id: item.sample_id,
    video_label: item.video_label,
    chunk_index: item.chunk_index,
    start: item.start,
    end: item.end,
    fallback_duration_s: item.fallback_duration_s,
    fallback_window_source: item.fallback_window_source,
    padded_start: item.padded_start,
    padded_end: item.padded_end,
    padded_duration_s: item.padded_duration_s,
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
  a.download = "manual_fallback_window_risk_labels.jsonl";
  a.click();
  URL.revokeObjectURL(a.href);
}}
notes.addEventListener("input", () => {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  annotations[item.sample_id] = {{...(annotations[item.sample_id] || {{}}), notes: notes.value, updated_at: new Date().toISOString()}};
  saveAnnotations();
}});
media.addEventListener("timeupdate", updateTimeline);
media.addEventListener("seeked", updateTimeline);
media.addEventListener("error", () => {{
  document.getElementById("mediaError").textContent = "媒体无法加载。请确认 live-server 从项目根目录启动，且媒体文件存在。";
}});
timeline.addEventListener("click", event => {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  const rect = timeline.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  media.currentTime = item.context_start + ratio * (item.context_end - item.context_start);
}});
document.getElementById("playChunkBtn").onclick = () => playCurrent("chunk");
document.getElementById("playContextBtn").onclick = () => playCurrent("context");
document.getElementById("pauseBtn").onclick = () => media.pause();
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
document.getElementById("summaryLine").textContent = `${{ITEMS.length}} 条 fallback-window 风险样本 · 完整字幕 ${{CUES.length}} cues`;
try {{
  for (const track of media.textTracks) track.mode = "hidden";
}} catch (_) {{}}
applyFilters();
renderCurrent(true);
</script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an audio/video audit page for unsafe fallback chunks."
    )
    parser.add_argument("--long-chunks", required=True, help="unsafe_fallback_chunks.jsonl")
    parser.add_argument("--subtitle-srt", required=True, help="Full Japanese SRT")
    parser.add_argument("--video", required=True, help="Source video/audio path")
    parser.add_argument(
        "--media-mode",
        choices=("video", "audio"),
        default="video",
        help="Generate a video audit page or extract audio-only media for review.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg binary used for audio extraction")
    parser.add_argument(
        "--output-dir",
        default="agents/audits/long-fallback-audit",
    )
    parser.add_argument("--title", default="SpeechBoundary-JA fallback-window 审计")
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
    source_media_path = project_path(args.video)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.media_mode == "audio":
        media_path = materialize_audio_media(
            source_path=source_media_path,
            output_dir=output_dir,
            ffmpeg_bin=args.ffmpeg_bin,
        )
        media_mime = audio_mime(media_path)
    else:
        media_path = source_media_path
        media_mime = video_mime(media_path)

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
        "label_schema_version": "fallback_window_timing_v2",
        "video_label": "匿名样片 A",
        "media_mode": args.media_mode,
        "media_path": project_rel(media_path),
        "media_mime": media_mime,
        "source_media_path": project_rel(source_media_path),
        "video_path": project_rel(source_media_path),
        "fallback_rows": project_rel(long_chunks_path),
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
        media_url=rel_url(media_path, from_dir=output_dir),
        media_mode=args.media_mode,
        media_mime=media_mime,
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
