#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


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


def _candidate_audio_paths(rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        raw_audio_values = [row.get("source_audio_path")]
        audio_payload = row.get("audio")
        if isinstance(audio_payload, Mapping):
            raw_audio_values.append(audio_payload.get("path"))
        for raw_audio in raw_audio_values:
            if not raw_audio:
                continue
            path = project_path(str(raw_audio))
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists() and path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                paths.append(path)
    return paths


def _media_root_audio_paths(*, video_id: str, rows: Sequence[Mapping[str, Any]], media_roots: Sequence[Path]) -> list[Path]:
    keys = {video_id.lower()}
    for row in rows:
        for value in (row.get("audio_id"), row.get("sample_id")):
            raw = str(value or "").strip().lower()
            if raw:
                keys.add(raw)
                keys.add(Path(raw).stem)
    paths: list[Path] = []
    seen: set[str] = set()
    for root in media_roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            haystack = f"{path.stem} {path.parent.as_posix()}".lower()
            if not any(key and key in haystack for key in keys):
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return sorted(paths, key=lambda item: item.as_posix())


def discover_media(
    *,
    archived_root: Path,
    media_roots: Sequence[Path],
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    video_ids = sorted({str(row.get("video_id") or "").strip() for row in rows if row.get("video_id")})
    rows_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        video_id = str(row.get("video_id") or "").strip()
        if video_id:
            rows_by_video.setdefault(video_id, []).append(row)
    media_by_video: dict[str, Any] = {}
    cues_by_video: dict[str, list[dict[str, Any]]] = {}
    aligned_by_video: dict[str, list[dict[str, Any]]] = {}
    subtitles_dir = output_dir / "subtitles"

    for video_id in video_ids:
        label = ANON_LABELS.get(video_id, video_id)
        archived_dir = archived_root / video_id
        srt_path = archived_dir / f"{video_id}.ja.srt"
        aligned_path = archived_dir / f"{video_id}.aligned_segments.json"
        video_rows = rows_by_video.get(video_id, [])
        audio_candidates = [
            *_candidate_audio_paths(video_rows),
            *_media_root_audio_paths(video_id=video_id, rows=video_rows, media_roots=media_roots),
        ]
        deduped_audio_candidates: list[Path] = []
        seen_audio: set[str] = set()
        for path in audio_candidates:
            resolved = str(path.resolve())
            if resolved in seen_audio:
                continue
            seen_audio.add(resolved)
            deduped_audio_candidates.append(path)
        audio_path = deduped_audio_candidates[0] if deduped_audio_candidates else None

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
                            "alignment_issue_subtype": segment.get("alignment_issue_subtype", ""),
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

def absolute_subtitle_window_range(row: Mapping[str, Any], start: float, end: float) -> tuple[float, float]:
    timing = row.get("subtitle_timing")
    if not isinstance(timing, Mapping):
        return start, end
    raw_start = timing.get("alignment_window_start_s")
    raw_end = timing.get("alignment_window_end_s")
    try:
        window_start = float(raw_start) if raw_start is not None else start
        window_end = float(raw_end) if raw_end is not None else end
    except (TypeError, ValueError):
        return start, end
    if window_end <= window_start:
        return start, end
    source = str(timing.get("alignment_window_source") or "")
    looks_absolute = (
        source == "final_subtitle"
        or (window_start >= max(0.0, start - 1.0) and window_end <= end + 1.0)
    )
    if not looks_absolute:
        window_start = start + window_start
        window_end = start + window_end
    return max(0.0, window_start), window_end


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
        subtitle_window_start, subtitle_window_end = absolute_subtitle_window_range(item, start, end)
        context_start = max(0.0, min(start, subtitle_window_start) - PREROLL_S)
        context_end = max(end, subtitle_window_end) + POSTROLL_S
        cues = cues_by_video.get(video_id, [])
        aligned_segments = aligned_by_video.get(video_id, [])
        media = dict(media_by_video.get(video_id) or {})

        item.update(
            {
                "index": index,
                "video_label": media.get("video_label", video_id),
                "media": media,
                "alignment_issue_type": item.get("alignment_issue_type", ""),
                "alignment_issue_subtype": item.get("alignment_issue_subtype", ""),
                "context_start": round(context_start, 3),
                "context_end": round(context_end, 3),
                "subtitle_window_start": round(subtitle_window_start, 3),
                "subtitle_window_end": round(subtitle_window_end, 3),
                "chunk_subtitle_cues": cues_for_range(cues, start, end),
                "subtitle_window_cues": cues_for_range(cues, subtitle_window_start, subtitle_window_end),
                "context_subtitle_cues": cues_for_range(cues, context_start, context_end),
                "aligned_segments": cues_for_range(aligned_segments, start, end),
            }
        )
        enriched.append(item)
    return enriched


def rows_for_page(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    page_keys = {
        "adjacency",
        "alignment_issue_subtype",
        "alignment_issue_type",
        "alignment_mode",
        "alignment_quality",
        "audio",
        "audio_id",
        "audit_sampling_score",
        "chunk_index",
        "cluster_confidence",
        "cluster_id",
        "cluster_label",
        "cluster_noise",
        "confidence",
        "context_end",
        "context_start",
        "cue_features",
        "display_prob_drop",
        "display_prob_keep",
        "duration_rank",
        "duration_rank_key",
        "duration_s",
        "end",
        "index",
        "labels",
        "media",
        "position",
        "qc",
        "raw_text",
        "sample_id",
        "start",
        "subtitle_window_end",
        "subtitle_window_start",
        "text",
        "text_features",
        "text_observation",
        "text_preview",
        "video_id",
        "video_label",
    }
    return [
        {key: row[key] for key in page_keys if key in row}
        for row in rows
    ]


def _page_html(
    *,
    title: str,
    dataset_id: str,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    media_by_video: Mapping[str, Any],
    cues_by_video: Mapping[str, list[dict[str, Any]]],
    aligned_by_video: Mapping[str, list[dict[str, Any]]],
) -> str:
    rows_json = json_for_script(rows_for_page(rows))
    summaries_json = json_for_script(summaries)
    media_json = json_for_script(media_by_video)
    cues_json = json_for_script(cues_by_video)
    aligned_json = json_for_script(aligned_by_video)
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
.main {{ display: flex; flex-direction: column; padding: 16px; max-height: 100vh; overflow: auto; }}
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
.audit-controls {{ display: grid; gap: 8px; margin-top: 10px; }}
.audit-controls .wide {{ grid-column: 1 / -1; }}
.control-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
.control-grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.sample-list {{ border-top: 1px solid var(--line); }}
.sidebar-details {{ border-top: 1px solid var(--line); padding: 10px 12px; }}
.sidebar-details .cluster-nav {{ margin-top: 10px; max-height: 280px; overflow: auto; padding-right: 4px; }}
.item {{ padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }}
.item.active, .item:hover {{ background: #eaf4f1; }}
.item-title {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
.item-metrics {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 1px 7px; background: #eef3ef; color: var(--muted); font-size: 12px; }}
.badge.warn {{ background: var(--amber-soft); color: #7a3f00; }}
.badge.noise {{ background: #eee7df; color: #76533b; }}
.meta {{ color: var(--muted); font-size: 12px; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(300px, 0.75fr); gap: 12px; align-items: start; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
.main > .panel {{ order: 5; }}
.main > .sample-detail-panel {{ order: 1; }}
.main > .grid {{ order: 2; }}
.main > .cluster-admin-panel {{ order: 20; }}
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
.cluster-audio-actions {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.cluster-inline-player {{ display: grid; gap: 6px; }}
.cluster-inline-player audio {{ width: 100%; height: 32px; }}
.cluster-inline-status {{ color: var(--muted); font-size: 12px; min-height: 16px; }}
.legacy-ui[hidden] {{ display: none !important; }}
#labelStatus {{ min-height: 18px; margin: 8px 0 0; }}
textarea {{ width: 100%; min-height: 80px; border: 1px solid var(--line); border-radius: 6px; padding: 8px; resize: vertical; }}
audio {{ width: 100%; }}
.caption-preview {{ margin-top: 10px; min-height: 88px; display: grid; align-items: center; border: 1px solid var(--line); border-radius: 8px; background: #17201d; color: #fff; padding: 12px; }}
.caption-preview.empty {{ color: #bac7c0; }}
.caption-stack {{ display: grid; gap: 8px; width: 100%; }}
.caption-line {{ white-space: pre-wrap; overflow-wrap: anywhere; text-align: center; line-height: 1.45; }}
.caption-line.dim {{ color: rgba(255, 255, 255, 0.58); font-size: 14px; }}
.caption-line.current {{ color: #fff; font-size: 22px; font-weight: 700; }}
.caption-empty {{ text-align: center; color: #bac7c0; font-size: 15px; }}
.caption-meta {{ margin-top: 6px; min-height: 18px; color: var(--muted); font-size: 12px; text-align: center; }}
.timeline {{ position: relative; height: 22px; margin-top: 12px; border: 1px solid var(--line); border-radius: 999px; background: #edf1ee; overflow: hidden; cursor: pointer; }}
.context-range {{ position: absolute; top: 0; bottom: 0; left: 0; right: 0; background: #e5ece8; }}
.chunk-range, .subtitle-window-range {{ position: absolute; top: 3px; bottom: 3px; border-radius: 999px; }}
.chunk-range {{ background: rgba(15, 118, 110, 0.28); border: 1px solid rgba(15, 118, 110, 0.5); }}
.subtitle-window-range {{ background: rgba(180, 83, 9, 0.34); border: 1px solid rgba(180, 83, 9, 0.55); }}
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
      <div class="audit-controls">
        <input class="wide" id="search" placeholder="搜索文本 / sample_id / 视频 / 指标 JSON">
        <div class="control-grid">
          <select id="videoFilter"></select>
          <select id="cluster"></select>
        </div>
        <div class="control-grid">
          <select id="alignmentFilter"></select>
          <select id="issueFilter"></select>
        </div>
        <div class="control-grid three">
          <input id="minDuration" inputmode="decimal" placeholder="最小时长 s">
          <input id="maxDuration" inputmode="decimal" placeholder="最大时长 s">
          <input id="minConfidence" inputmode="decimal" placeholder="最低置信度">
        </div>
        <div class="control-grid">
          <select id="sortBy">
            <option value="duration_desc">时长 ↓</option>
            <option value="duration_asc">时长 ↑</option>
            <option value="confidence_desc">置信度/采样分 ↓</option>
            <option value="confidence_asc">置信度/采样分 ↑</option>
            <option value="chars_desc">字符数 ↓</option>
            <option value="chars_per_sec_desc">字符密度 ↓</option>
            <option value="start_asc">时间轴 ↑</option>
            <option value="video_start">视频 + 时间轴 ↑</option>
            <option value="chunk_asc">chunk index ↑</option>
          </select>
          <select id="pageSize">
            <option value="200">显示 200</option>
            <option value="500">显示 500</option>
            <option value="1000">显示 1000</option>
            <option value="all">显示全部</option>
          </select>
        </div>
      </div>
    </div>
    <div class="sample-list" id="list"></div>
    <details class="sidebar-details">
      <summary>簇级导航</summary>
      <div class="cluster-nav" id="clusterNav"></div>
    </details>
  </aside>
  <main class="main">
    <section class="panel cluster-admin-panel">
      <div class="label-heading">
        <h2 id="clusterTitle">簇级粗标签</h2>
        <span class="meta" id="clusterCount"></span>
      </div>
      <div class="cluster-detail-summary" id="clusterSummary"></div>
      <div class="cluster-detail-fields">
        <div class="label-heading"><span class="label-title">种子控制</span><span class="meta">只有“用作种子 + keep/drop”会进入 cold-start 训练</span></div>
        <div class="labels" id="clusterSeedActionButtons"></div>
        <div class="label-heading"><span class="label-title">训练标签</span><span class="meta">CueQC 训练目标仍只导出 keep/drop；混簇和跳过不导出标签</span></div>
        <div class="labels" id="clusterDisplayButtons"></div>
        <textarea id="activeClusterReason" placeholder="备注 / 原因（可选，不进训练）"></textarea>
      </div>
      <p class="meta cluster-status" id="clusterStatus"></p>
    </section>
    <section class="panel cluster-admin-panel">
      <h3>全部音频</h3>
      <div class="cluster-audio-list" id="clusterAudioList"></div>
    </section>
    <div class="legacy-ui" hidden>
      <div class="cluster-review" id="clusterReview"></div>
    </div>
    <section class="panel sample-detail-panel">
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
        <audio id="media" controls preload="none"></audio>
        <div class="caption-preview" id="captionOverlay"></div>
        <div class="caption-meta" id="captionMeta"></div>
        <div class="timeline" id="timeline">
          <div class="context-range"></div>
          <div class="chunk-range" id="chunkRange"></div>
          <div class="subtitle-window-range" id="subtitleWindowRange"></div>
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
          <div class="cue-list" id="subtitleWindowCueList"></div>
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
<script type="application/json" id="aligned-json">{aligned_json}</script>
<script>
const ROWS = JSON.parse(document.getElementById("rows-json").textContent);
const SUMMARIES = JSON.parse(document.getElementById("summaries-json").textContent);
const MEDIA_BY_VIDEO = JSON.parse(document.getElementById("media-json").textContent);
const CUES_BY_VIDEO = JSON.parse(document.getElementById("cues-json").textContent);
const ALIGNED_BY_VIDEO = JSON.parse(document.getElementById("aligned-json").textContent);
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
const CLUSTER_SEED_ACTIONS = [{{value:"use_seed", label:"用作种子"}},{{value:"mixed_skip", label:"混簇跳过"}},{{value:"skip", label:"跳过"}}];
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
let lastCueRenderKey = "";
const CLUSTER_AUDIO_PAGE_SIZE = 80;
const clusterAudioVisible = new Map();
const media = document.getElementById("media");
const labelStatus = document.getElementById("labelStatus");
const timeline = document.getElementById("timeline");
const chunkRange = document.getElementById("chunkRange");
const subtitleWindowRange = document.getElementById("subtitleWindowRange");
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
function cueWindow(videoId, t) {{
  const cues = CUES_BY_VIDEO[videoId] || [];
  let prev = null;
  let next = null;
  let activeIndex = -1;
  for (let index = 0; index < cues.length; index += 1) {{
    const cue = cues[index];
    const start = Number(cue.start || 0);
    const end = Number(cue.end || start);
    if (start <= t && t <= end) {{
      activeIndex = index;
      break;
    }}
    if (end < t) prev = cue;
    else if (!next && start > t) next = cue;
  }}
  if (activeIndex >= 0) {{
    const current = [];
    let tail = activeIndex;
    for (let index = activeIndex; index < cues.length; index += 1) {{
      const cue = cues[index];
      const start = Number(cue.start || 0);
      const end = Number(cue.end || start);
      if (!(start <= t && t <= end)) break;
      current.push(cue);
      tail = index;
    }}
    return {{
      prev: cues[activeIndex - 1] || prev,
      current,
      next: cues[tail + 1] || next,
    }};
  }}
  return {{ prev, current: [], next }};
}}
function cueText(cues) {{
  if (!Array.isArray(cues)) return "";
  return cues.map(cue => String(cue && cue.text ? cue.text : "").trim()).filter(Boolean).join("\\n");
}}
function cueTimeRange(cues) {{
  if (!Array.isArray(cues) || cues.length === 0) return "";
  const start = Number(cues[0].start || 0);
  const end = Number(cues[cues.length - 1].end || cues[0].end || start);
  return `${{fmt(start)}} - ${{fmt(end)}}`;
}}
function renderCaptionPreview(row) {{
  const root = document.getElementById("captionOverlay");
  const meta = document.getElementById("captionMeta");
  if (!root) return;
  const t = media.currentTime || 0;
  const window = cueWindow(row.video_id || "", t);
  const prevText = window.prev ? String(window.prev.text || "").trim() : "";
  const currentText = cueText(window.current);
  const nextText = window.next ? String(window.next.text || "").trim() : "";
  const lines = [];
  if (prevText) {{
    lines.push(`<div class="caption-line dim">${{escapeHtml(prevText)}}</div>`);
  }}
  if (currentText) {{
    lines.push(`<div class="caption-line current">${{escapeHtml(currentText)}}</div>`);
  }} else {{
    lines.push(`<div class="caption-empty">当前时刻没有字幕</div>`);
  }}
  if (nextText) {{
    lines.push(`<div class="caption-line dim">${{escapeHtml(nextText)}}</div>`);
  }}
  root.className = `caption-preview${{currentText ? "" : " empty"}}`;
  root.innerHTML = `<div class="caption-stack">${{lines.join("")}}</div>`;
  if (meta) {{
    meta.textContent = currentText ? `当前字幕 ${{cueTimeRange(window.current)}}` : "当前时刻没有字幕 cue";
  }}
}}
function rowText(row) {{
  return [
    row.sample_id,
    row.video_id,
    row.video_label,
    row.audio_id,
    row.cluster_id,
    row.cluster_label,
    row.text,
    row.raw_text,
    row.alignment_quality,
    row.alignment_mode,
    row.alignment_issue_type,
    row.alignment_issue_subtype,
    JSON.stringify(row.qc || {{}}),
    JSON.stringify(row.text_features || {{}}),
    JSON.stringify(row.cue_features || {{}}),
    JSON.stringify(row.adjacency || {{}})
  ].join("\\n").toLowerCase();
}}
function textObservation(row) {{
  const cueFeatures = row.cue_features || {{}};
  return cueFeatures.text_observation || row.text_observation || {{}};
}}
function rowNumber(row, key, fallback = 0) {{
  const value = Number(row && row[key]);
  return Number.isFinite(value) ? value : fallback;
}}
function optionalNumberInput(id) {{
  const raw = document.getElementById(id).value.trim();
  if (!raw) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}}
function rowConfidence(row) {{
  const values = [
    row.cluster_confidence,
    row.audit_sampling_score,
    row.confidence,
    row.display_prob_drop,
    row.display_prob_keep
  ].map(Number).filter(Number.isFinite);
  return values.length ? values[0] : 0;
}}
function rowCharCount(row) {{
  const obs = textObservation(row);
  return Number(obs.char_count ?? obs.raw_char_count ?? 0) || 0;
}}
function rowCharsPerSec(row) {{
  const obs = textObservation(row);
  const direct = Number(obs.chars_per_sec);
  if (Number.isFinite(direct)) return direct;
  const duration = Math.max(0.001, rowNumber(row, "duration_s", rowNumber(row, "end") - rowNumber(row, "start")));
  return rowCharCount(row) / duration;
}}
function compareRows(a, b) {{
  const sortBy = document.getElementById("sortBy") ? document.getElementById("sortBy").value : "duration_desc";
  const videoCompare = String(a.video_label || a.video_id || "").localeCompare(String(b.video_label || b.video_id || ""));
  const startCompare = rowNumber(a, "start") - rowNumber(b, "start");
  if (sortBy === "duration_asc") return rowNumber(a, "duration_s") - rowNumber(b, "duration_s") || videoCompare || startCompare;
  if (sortBy === "confidence_desc") return rowConfidence(b) - rowConfidence(a) || rowNumber(b, "duration_s") - rowNumber(a, "duration_s") || videoCompare || startCompare;
  if (sortBy === "confidence_asc") return rowConfidence(a) - rowConfidence(b) || rowNumber(b, "duration_s") - rowNumber(a, "duration_s") || videoCompare || startCompare;
  if (sortBy === "chars_desc") return rowCharCount(b) - rowCharCount(a) || rowNumber(b, "duration_s") - rowNumber(a, "duration_s") || videoCompare || startCompare;
  if (sortBy === "chars_per_sec_desc") return rowCharsPerSec(b) - rowCharsPerSec(a) || rowNumber(b, "duration_s") - rowNumber(a, "duration_s") || videoCompare || startCompare;
  if (sortBy === "start_asc") return startCompare || videoCompare || rowNumber(a, "chunk_index") - rowNumber(b, "chunk_index");
  if (sortBy === "video_start") return videoCompare || startCompare || rowNumber(a, "chunk_index") - rowNumber(b, "chunk_index");
  if (sortBy === "chunk_asc") return videoCompare || rowNumber(a, "chunk_index") - rowNumber(b, "chunk_index") || startCompare;
  return rowNumber(b, "duration_s") - rowNumber(a, "duration_s") || videoCompare || startCompare;
}}
function sortRows(rows) {{
  return [...rows].sort(compareRows);
}}
function currentRenderLimit() {{
  const raw = document.getElementById("pageSize") ? document.getElementById("pageSize").value : "200";
  if (raw === "all") return Number.POSITIVE_INFINITY;
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : 200;
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
  return sortRows(clusterRows(clusterId));
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
  const obs = summary.text_observation_counts || {{}};
  const obsText = Object.entries(obs).map(([k, v]) => `${{k}}=${{v}}`).join(", ");
  return `count=${{summary.count || 0}} · chars_avg=${{Number(summary.char_count_avg || 0).toFixed(2)}} · conf=${{Number(summary.confidence_avg || 0).toFixed(3)}} · text ${{obsText || "n/a"}}`;
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
    const currentValue = key === "seed_action" ? clusterSeedAction(ann) : ann[key];
    btn.className = currentValue === option.value ? "active" : "";
    btn.onclick = () => {{
      updateClusterAnnotation(clusterId, key, option.value);
      renderClusterDetail();
    }};
    root.appendChild(btn);
  }}
}}
function updateClusterAnnotation(clusterId, key, value) {{
  const next = {{ ...(clusterAnnotations[clusterId] || {{}}) }};
  next[key] = value;
  if (key === "display_decision" && ["keep", "drop"].includes(String(value || ""))) {{
    next.seed_action = "use_seed";
  }}
  if (key === "seed_action" && value !== "use_seed") {{
    next.display_decision = "";
  }}
  next.updated_at = new Date().toISOString();
  clusterAnnotations[clusterId] = next;
  saveClusterAnnotations();
  updateClusterProgress();
  renderClusterNav();
  if (clusterId === activeClusterId) {{
    const status = document.getElementById("clusterStatus");
    if (status) {{
      status.textContent = isClusterComplete(clusterId)
        ? "已完成种子决策"
        : "选择用作种子并给 keep/drop，或标为混簇/跳过。";
    }}
  }}
}}
function clusterSeedAction(ann) {{
  return ann.seed_action || (ann.display_decision ? "use_seed" : "");
}}
function clusterTrainingDecision(ann) {{
  return clusterSeedAction(ann) === "use_seed" && ["keep", "drop"].includes(String(ann.display_decision || ""))
    ? ann.display_decision
    : "";
}}
function isClusterComplete(clusterId) {{
  const ann = clusterAnnotations[clusterId] || {{}};
  const action = clusterSeedAction(ann);
  return ["mixed_skip", "skip"].includes(action) || Boolean(clusterTrainingDecision(ann));
}}
function updateClusterProgress() {{
  const complete = SUMMARIES.filter(summary => isClusterComplete(summary.cluster_id)).length;
  const total = SUMMARIES.length;
  document.getElementById("clusterProgress").textContent = `${{complete}} / ${{total}} 簇已完成种子决策`;
}}
function selectCluster(clusterId) {{
  const entry = getClusterEntry(clusterId);
  if (!entry) return;
  activeClusterId = entry.clusterId;
  saveActiveClusterId(activeClusterId);
  const clusterSelect = document.getElementById("cluster");
  if (clusterSelect) clusterSelect.value = entry.clusterId;
  applyFilters();
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
        ${{clusterSeedAction(ann) ? ` · ${{escapeHtml(clusterSeedAction(ann) === "use_seed" ? (ann.display_decision === "drop" ? "种子:丢弃" : "种子:保留") : (clusterSeedAction(ann) === "mixed_skip" ? "混簇跳过" : "跳过"))}}` : ""}}
      </div>`;
    button.onclick = () => selectCluster(entry.clusterId);
    root.appendChild(button);
  }});
}}
function clusterAudioVisibleCount(clusterId, total) {{
  const currentLimit = clusterAudioVisible.get(clusterId) || CLUSTER_AUDIO_PAGE_SIZE;
  return Math.min(total, currentLimit);
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
  const visibleCount = clusterAudioVisibleCount(entry.clusterId, examples.length);
  const count = document.createElement("div");
  count.className = "meta";
  count.textContent = `共 ${{examples.length}} 条音频 · 当前显示 ${{visibleCount}} 条 · 每条独立播放器`;
  root.appendChild(count);
  for (const row of examples.slice(0, visibleCount)) {{
    const info = row.media || {{}};
    const sampleId = escapeHtml(row.sample_id);
    const card = document.createElement("div");
    card.className = "cluster-audio-card";
    card.innerHTML = `
      <div class="cluster-audio-head">
        <strong>${{escapeHtml(row.video_label || row.video_id || "")}} · chunk ${{escapeHtml(row.chunk_index)}} · ${{Number(row.duration_s || 0).toFixed(2)}}s</strong>
        <span class="meta">${{sampleId}}</span>
      </div>
      <div class="cluster-audio-meta">${{fmt(row.start)}} - ${{fmt(row.end)}}${{info.audio_url ? "" : " · 音频缺失"}}${{info.vtt_url ? "" : " · 字幕缺失"}}</div>
      <div class="cluster-audio-actions">
        <button class="primary" type="button" data-play-sample="${{sampleId}}" data-play-mode="chunk"${{info.audio_url ? "" : " disabled"}}>播放 chunk</button>
        <button type="button" data-play-sample="${{sampleId}}" data-play-mode="context"${{info.audio_url ? "" : " disabled"}}>播放上下文</button>
        <button type="button" data-open-sample="${{sampleId}}">打开详情</button>
      </div>
      <div class="cluster-inline-player">
        <audio controls preload="none" data-inline-audio="${{sampleId}}" src="${{escapeHtml(info.audio_url || "")}}"></audio>
        <div class="cluster-inline-status" data-inline-status="${{sampleId}}"></div>
      </div>
      <div class="cluster-audio-text">${{escapeHtml(row.text_preview || row.text || row.raw_text || "(empty)")}}</div>
    `;
    root.appendChild(card);
  }}
  if (visibleCount < examples.length) {{
    const loadMore = document.createElement("button");
    loadMore.type = "button";
    loadMore.textContent = `再显示 ${{Math.min(CLUSTER_AUDIO_PAGE_SIZE, examples.length - visibleCount)}} 条`;
    loadMore.onclick = () => {{
      clusterAudioVisible.set(entry.clusterId, visibleCount + CLUSTER_AUDIO_PAGE_SIZE);
      renderClusterAudioList(entry);
    }};
    root.appendChild(loadMore);
  }}
  root.querySelectorAll("[data-open-sample]").forEach(button => {{
    button.onclick = () => selectSample(button.getAttribute("data-open-sample") || "", false);
  }});
  root.querySelectorAll("[data-play-sample]").forEach(button => {{
    button.onclick = () => {{
      const sampleId = button.getAttribute("data-play-sample") || "";
      const mode = button.getAttribute("data-play-mode") || "chunk";
      playInlineAudio(sampleId, mode);
    }};
  }});
}}
function stopOtherInlineAudio(activeAudio) {{
  document.querySelectorAll("[data-inline-audio]").forEach(audio => {{
    if (audio !== activeAudio) audio.pause();
  }});
}}
function inlineAudioForSample(sampleId) {{
  return Array.from(document.querySelectorAll("[data-inline-audio]")).find(
    audio => audio.getAttribute("data-inline-audio") === sampleId
  );
}}
function inlineStatusForSample(sampleId) {{
  return Array.from(document.querySelectorAll("[data-inline-status]")).find(
    node => node.getAttribute("data-inline-status") === sampleId
  );
}}
function playInlineAudio(sampleId, mode) {{
  const row = ROWS.find(item => item.sample_id === sampleId);
  if (!row) return;
  const audio = inlineAudioForSample(sampleId);
  const status = inlineStatusForSample(sampleId);
  if (!audio) return;
  const info = row.media || {{}};
  if (!info.audio_url) {{
    if (status) status.textContent = "音频文件未找到。";
    return;
  }}
  stopOtherInlineAudio(audio);
  const start = mode === "context" ? Number(row.context_start ?? row.start ?? 0) : Number(row.start ?? 0);
  const end = mode === "context" ? Number(row.context_end ?? row.end ?? start) : Number(row.end ?? start);
  audio.dataset.stopAt = Number.isFinite(end) ? String(end) : "";
  audio.dataset.sampleId = sampleId;
  audio.dataset.playMode = mode;
  const begin = () => {{
    try {{ audio.currentTime = Number.isFinite(start) ? start : 0; }} catch (_) {{}}
    if (status) status.textContent = `${{mode === "context" ? "上下文" : "chunk"}} ${{fmt(start)}} - ${{fmt(end)}}`;
    const promise = audio.play();
    if (promise && promise.catch) {{
      promise.catch(err => {{
        if (status) status.textContent = `播放失败：${{err && err.message ? err.message : err}}`;
      }});
    }}
  }};
  if (audio.readyState < 1) {{
    audio.addEventListener("loadedmetadata", begin, {{once: true}});
    audio.load();
  }} else {{
    begin();
  }}
}}
document.addEventListener("timeupdate", event => {{
  const audio = event.target;
  if (!(audio instanceof HTMLAudioElement) || !audio.matches("[data-inline-audio]")) return;
  const stopAt = Number(audio.dataset.stopAt || 0);
  if (stopAt > 0 && !audio.paused && audio.currentTime >= stopAt) audio.pause();
}}, true);
document.addEventListener("error", event => {{
  const audio = event.target;
  if (!(audio instanceof HTMLAudioElement) || !audio.matches("[data-inline-audio]")) return;
  const sampleId = audio.getAttribute("data-inline-audio") || "";
  const status = inlineStatusForSample(sampleId);
  if (status) status.textContent = "媒体无法加载。";
}}, true);
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
  setClusterButtons("clusterSeedActionButtons", CLUSTER_SEED_ACTIONS, entry.clusterId, "seed_action");
  setClusterButtons("clusterDisplayButtons", CLUSTER_DISPLAY, entry.clusterId, "display_decision");
  const reasonInput = document.getElementById("activeClusterReason");
  if (reasonInput) {{
    reasonInput.value = ann.classification_reason || "";
    reasonInput.oninput = () => updateClusterAnnotation(entry.clusterId, "classification_reason", reasonInput.value);
  }}
  const status = document.getElementById("clusterStatus");
  if (status) {{
    status.textContent = isClusterComplete(entry.clusterId)
      ? "已完成种子决策"
      : "选择用作种子并给 keep/drop，或标为混簇/跳过。";
  }}
  renderClusterAudioList(entry);
}}
function selectSample(sampleId, autoplay = true) {{
  const row = ROWS.find(item => item.sample_id === sampleId);
  if (!row) return;
  document.getElementById("search").value = "";
  document.getElementById("cluster").value = row.cluster_id || "";
  filtered = sortRows(ROWS.filter(item => item.cluster_id === row.cluster_id));
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
function fillSelect(selectId, options, allLabel) {{
  const select = document.getElementById(selectId);
  if (!select) return;
  select.innerHTML = "";
  const all = document.createElement("option");
  all.value = "";
  all.textContent = allLabel;
  select.appendChild(all);
  for (const option of options) {{
    const node = document.createElement("option");
    node.value = option.value;
    node.textContent = option.label;
    select.appendChild(node);
  }}
}}
function uniqueOptions(rows, key, labelFn) {{
  const values = new Map();
  for (const row of rows) {{
    const value = String(row[key] ?? "").trim();
    if (!value || values.has(value)) continue;
    values.set(value, labelFn ? labelFn(value, row) : value);
  }}
  return [...values.entries()]
    .sort((a, b) => String(a[1]).localeCompare(String(b[1])))
    .map(([value, label]) => ({{value, label}}));
}}
function setupFilters() {{
  fillSelect(
    "cluster",
    SUMMARIES.map(summary => ({{value: summary.cluster_id || "", label: `${{summary.cluster_id}} (${{summary.count || 0}})`}})),
    "all duration buckets / clusters"
  );
  fillSelect(
    "videoFilter",
    uniqueOptions(ROWS, "video_id", (_value, row) => row.video_label || row.video_id || ""),
    "all videos"
  );
  fillSelect("alignmentFilter", uniqueOptions(ROWS, "alignment_quality"), "all alignment quality");
  fillSelect("issueFilter", uniqueOptions(ROWS, "alignment_issue_subtype"), "all alignment issue");
}}
function applyFilters() {{
  const q = document.getElementById("search").value.trim().toLowerCase();
  const cluster = document.getElementById("cluster").value;
  const video = document.getElementById("videoFilter").value;
  const alignment = document.getElementById("alignmentFilter").value;
  const issue = document.getElementById("issueFilter").value;
  const minDuration = optionalNumberInput("minDuration");
  const maxDuration = optionalNumberInput("maxDuration");
  const minConfidence = optionalNumberInput("minConfidence");
  filtered = sortRows(ROWS.filter(row => {{
    const duration = rowNumber(row, "duration_s", rowNumber(row, "end") - rowNumber(row, "start"));
    if (cluster && row.cluster_id !== cluster) return false;
    if (video && row.video_id !== video) return false;
    if (alignment && String(row.alignment_quality || "") !== alignment) return false;
    if (issue && String(row.alignment_issue_subtype || "") !== issue) return false;
    if (minDuration !== null && duration < minDuration) return false;
    if (maxDuration !== null && duration > maxDuration) return false;
    if (minConfidence !== null && rowConfidence(row) < minConfidence) return false;
    if (q && !rowText(row).includes(q)) return false;
    return true;
  }}));
  if (cluster && CLUSTER_BY_ID.has(cluster) && cluster !== activeClusterId) {{
    activeClusterId = cluster;
    saveActiveClusterId(activeClusterId);
    renderClusterNav();
    renderClusterDetail();
  }}
  current = Math.max(0, ROWS.indexOf(filtered[0] || ROWS[0]));
  renderList();
  renderCurrent(false);
}}
function renderList() {{
  const root = document.getElementById("list");
  root.innerHTML = "";
  const limit = currentRenderLimit();
  const visibleRows = filtered.slice(0, limit);
  for (const row of visibleRows) {{
    const div = document.createElement("div");
    const textObs = textObservation(row);
    div.className = "item" + (ROWS[current] === row ? " active" : "");
    div.onclick = () => {{ current = ROWS.indexOf(row); renderList(); renderCurrent(true); }};
    const badgeClass = row.cluster_noise ? "badge noise" : "badge";
    div.innerHTML = `<div class="item-title"><strong>${{escapeHtml(row.cluster_id)}} · chunk ${{row.chunk_index}}</strong><span class="${{badgeClass}}">${{escapeHtml(row.video_label || row.video_id || "")}}</span></div>
      <div class="meta">${{fmt(row.start)}}-${{fmt(row.end)}} · ${{Number(row.duration_s||0).toFixed(2)}}s · conf=${{rowConfidence(row).toFixed(3)}} · rank=${{escapeHtml(String(row.duration_rank ?? row.index ?? ""))}}</div>
      <div class="item-metrics">
        <span class="badge">chars=${{escapeHtml(String(textObs.char_count ?? ""))}}</span>
        <span class="badge">cps=${{rowCharsPerSec(row).toFixed(2)}}</span>
        <span class="badge">${{escapeHtml(row.alignment_quality || "align:n/a")}}</span>
        <span class="badge">${{escapeHtml(row.alignment_issue_subtype || "issue:n/a")}}</span>
      </div>
      <div class="meta">${{escapeHtml(row.text_preview || row.text || "(empty)")}}</div>`;
    root.appendChild(div);
  }}
  if (visibleRows.length < filtered.length) {{
    const more = document.createElement("button");
    more.type = "button";
    more.textContent = `当前显示 ${{visibleRows.length}} / ${{filtered.length}}，切换“显示数量”可查看更多`;
    more.onclick = () => {{
      document.getElementById("pageSize").value = "all";
      renderList();
    }};
    root.appendChild(more);
  }}
  document.getElementById("summaryLine").textContent = `${{filtered.length}} / ${{ROWS.length}} samples · showing ${{visibleRows.length}} · ${{SUMMARIES.length}} clusters`;
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
    const currentValue = key === "seed_action" ? clusterSeedAction(ann) : ann[key];
    btn.className = currentValue === option.value ? "active" : "";
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
  const textObs = textObservation(row);
  const mediaInfo = row.media || {{}};
  const rows = [
    ["video", `${{row.video_label || ""}} · ${{row.video_id || ""}}`],
    ["cluster", `${{row.cluster_id}} · confidence=${{Number(row.cluster_confidence || 0).toFixed(3)}}`],
    ["sampling", `score=${{Number(row.audit_sampling_score ?? rowConfidence(row)).toFixed(3)}} · rank=${{row.duration_rank ?? row.index ?? ""}}`],
    ["chunk", `${{row.chunk_index}} · ${{fmt(row.start)}}-${{fmt(row.end)}}`],
    ["duration", `${{Number(row.duration_s || 0).toFixed(3)}}s · cps=${{rowCharsPerSec(row).toFixed(3)}}`],
    ["context", `${{fmt(row.context_start)}}-${{fmt(row.context_end)}}`],
    ["字幕时间轴窗口", `${{fmt(row.subtitle_window_start)}}-${{fmt(row.subtitle_window_end)}}`],
    ["text_obs", JSON.stringify(textObs)],
    ["chars", `${{tf.char_count || 0}} unique=${{tf.unique_chars || 0}} kana=${{tf.kana_ratio || 0}} kanji=${{tf.kanji_ratio || 0}}`],
    ["repeat", JSON.stringify(tf.repeat_profile || {{}})],
    ["alignment", `${{row.alignment_quality || ""}} · mode=${{row.alignment_mode || ""}}`],
    ["alignment issue", `${{row.alignment_issue_type || ""}} · ${{row.alignment_issue_subtype || ""}}`],
    ["adjacency", JSON.stringify(row.adjacency || {{}})],
    ["cue_features", JSON.stringify(cueFeatures)],
    ["audio", mediaInfo.audio_path || ""],
    ["subtitle", mediaInfo.srt_path || ""]
  ];
  document.getElementById("metrics").innerHTML = rows.map(([k,v]) => `<div>${{escapeHtml(k)}}</div><div>${{escapeHtml(v)}}</div>`).join("");
}}
function overlapsRange(aStart, aEnd, bStart, bEnd) {{
  return Math.min(Number(aEnd), Number(bEnd)) - Math.max(Number(aStart), Number(bStart)) > 0.01;
}}
function rowsForRange(items, start, end) {{
  if (!Array.isArray(items)) return [];
  return items.filter(item => overlapsRange(start, end, item.start, item.end));
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
  const cues = CUES_BY_VIDEO[row.video_id || ""] || [];
  const aligned = ALIGNED_BY_VIDEO[row.video_id || ""] || [];
  renderCueGroup("chunkCueList", rowsForRange(cues, row.start, row.end), "该 chunk 内没有字幕 cue。");
  renderCueGroup("subtitleWindowCueList", rowsForRange(cues, row.subtitle_window_start, row.subtitle_window_end), "该字幕时间轴窗口内没有字幕 cue。");
  renderCueGroup("contextCueList", rowsForRange(cues, row.context_start, row.context_end), "该上下文内没有字幕 cue。");
  renderCueGroup("alignedList", rowsForRange(aligned, row.start, row.end), "该 chunk 内没有字幕片段。");
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
  setRange(subtitleWindowRange, row.subtitle_window_start, row.subtitle_window_end, 0.5);
  const t = media.currentTime || 0;
  const pct = Math.max(0, Math.min(100, ((t - Number(row.context_start)) / width) * 100));
  cursor.style.left = `${{pct}}%`;
  document.getElementById("nowText").textContent = fmt(t);
  renderCaptionPreview(row);
  const stopAt = playMode === "context" ? Number(row.context_end) : Number(row.end);
  if (!media.paused && t >= stopAt) media.pause();
  const cueRenderKey = String(row.sample_id || "") + "|" + String(Math.floor(t * 2));
  if (cueRenderKey !== lastCueRenderKey) {{
    lastCueRenderKey = cueRenderKey;
    renderCueLists(row);
  }}
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
  lastCueRenderKey = "";
  renderCueLists(row);
  setButtons("displayButtons", DISPLAY, "display_decision");
  updateTimeline();
  const legacy = document.querySelector(".legacy-ui");
  if (legacy && !legacy.hidden) renderClusterReview();
}}
function playCurrent(mode) {{
  const row = ROWS[current];
  if (!row) return;
  const error = document.getElementById("mediaError");
  playMode = mode;
  setMediaForItem(row, false);
  const target = mode === "context" ? Number(row.context_start) : Number(row.start);
  const startPlayback = () => {{
    try {{ media.currentTime = Number.isFinite(target) ? target : 0; }} catch (_) {{}}
    error.textContent = "";
    const promise = media.play();
    if (promise && promise.catch) {{
      promise.catch(err => {{
        error.textContent = `播放失败：${{err && err.message ? err.message : err}}`;
      }});
    }}
  }};
  if (!media.querySelector("source")) {{
    error.textContent = "音频文件未找到。";
    return;
  }}
  if (media.readyState < 1) {{
    error.textContent = "正在加载音频...";
    media.addEventListener("loadedmetadata", startPlayback, {{ once: true }});
    media.load();
  }} else {{
    startPlayback();
  }}
}}
function exportClusterRows() {{
  return SUMMARIES.map(summary => {{
    const clusterId = summary.cluster_id || "";
    const ann = clusterAnnotations[clusterId] || {{}};
    const seedAction = clusterSeedAction(ann);
    const trainingDecision = clusterTrainingDecision(ann);
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
      display_decision: trainingDecision,
      seed_action: seedAction,
      training_label_included: Boolean(trainingDecision),
      notes: ann.classification_reason || "",
      updated_at: ann.updated_at || "",
      count: summary.count || 0,
      char_count_avg: summary.char_count_avg || 0,
      confidence_avg: summary.confidence_avg || 0,
      text_observation_counts: summary.text_observation_counts || {{}},
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
document.getElementById("videoFilter").addEventListener("change", applyFilters);
document.getElementById("alignmentFilter").addEventListener("change", applyFilters);
document.getElementById("issueFilter").addEventListener("change", applyFilters);
document.getElementById("minDuration").addEventListener("input", applyFilters);
document.getElementById("maxDuration").addEventListener("input", applyFilters);
document.getElementById("minConfidence").addEventListener("input", applyFilters);
document.getElementById("sortBy").addEventListener("change", () => {{
  applyFilters();
}});
document.getElementById("pageSize").addEventListener("change", renderList);
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
setupFilters();
applyFilters();
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
    archived_root: Path,
    media_roots: Sequence[Path],
    output_dir: Path,
    title: str,
    dataset_id: str,
    summary_json: Path | None = None,
    refresh_nav: bool = False,
) -> dict[str, Any]:
    if not archived_root.exists():
        raise FileNotFoundError(f"archived root not found: {archived_root}")
    if not media_roots:
        raise ValueError("at least one --media-root is required")
    missing_media_roots = [path for path in media_roots if not path.exists()]
    if missing_media_roots:
        raise FileNotFoundError("media root not found: " + ", ".join(str(path) for path in missing_media_roots))

    rows = read_jsonl(clusters_jsonl)
    summaries = read_jsonl(summaries_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    media_by_video, cues_by_video, aligned_by_video = discover_media(
        archived_root=archived_root,
        media_roots=media_roots,
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
            aligned_by_video=aligned_by_video,
        ),
        encoding="utf-8",
    )

    summary_source = summary_json if summary_json else output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_source.exists():
        loaded = read_json(summary_source)
        if isinstance(loaded, dict):
            summary.update(loaded)
    summary.pop("baseline_root", None)
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
            "archived_root": project_rel(archived_root),
            "media_roots": [project_rel(path) for path in media_roots],
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
            "cluster_seed_action_options": ["use_seed", "mixed_skip", "skip"],
            "cluster_display_decision_required": False,
            "cluster_training_label_rule": "only seed_action=use_seed with display_decision keep/drop is broadcast into training; mixed_skip/skip abstain",
            "cluster_review_enabled": True,
            "cluster_review_group_count": len(summaries),
            "cluster_review_examples_per_cluster": 3,
            "cluster_review_layout": "left_nav_single_cluster_inline_players_v4",
            "cluster_review_audio_render_mode": "per_chunk_inline_audio",
            "cluster_audio_page_size": 80,
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
    parser.add_argument("--archived-root", required=True, help="Directory containing <video_id>/<video_id>.ja.srt and aligned_segments.json.")
    parser.add_argument("--media-root", action="append", required=True, help="Job/audio/media root to search recursively for source audio. Repeatable.")
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
        archived_root=project_path(args.archived_root),
        media_roots=[project_path(path) for path in args.media_root],
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
