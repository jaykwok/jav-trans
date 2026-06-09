#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints
from tools.audits.generate_long_fallback_chunk_audit_html import (
    audio_mime,
    clean_cue_text,
    compact_text,
    cue_text_for_range,
    cues_for_range,
    materialize_audio_media,
    project_path,
    project_rel,
    read_jsonl,
    read_srt_cues,
    rel_url,
    row_float,
    video_mime,
    write_json,
    write_jsonl,
    write_vtt,
)


BucketPredicate = Callable[[Mapping[str, Any]], bool]


def read_aligned_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("segments") or []
    else:
        rows = []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def aligned_by_chunk(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            chunk_index = int(row.get("source_chunk_index"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(chunk_index, []).append(
            {
                "start": round(row_float(row, "start"), 3),
                "end": round(row_float(row, "end"), 3),
                "text": compact_text(row.get("text") or "", max_chars=320),
            }
        )
    return grouped


def row_chunk_index(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("chunk_index"))
    except (TypeError, ValueError):
        return -1


def repetition_changed(row: Mapping[str, Any]) -> bool:
    repair = row.get("repetition_repair")
    return isinstance(repair, Mapping) and bool(repair.get("changed"))


def text_density_level(row: Mapping[str, Any]) -> str:
    return str(row.get("text_density_level") or "")


def density_review(row: Mapping[str, Any]) -> bool:
    return row_float(row, "core_cps") > 4.0 or row_float(row, "fallback_cps") > 4.0


def bucket_definitions() -> list[tuple[str, str, BucketPredicate, Callable[[Mapping[str, Any]], tuple[Any, ...]]]]:
    return [
        (
            "density_over_4cps",
            "4 CPS 密度审计",
            lambda row: density_review(row),
            lambda row: (
                not bool(row.get("chunk_text_leak_risk")),
                not bool(row.get("chunk_repetition_risk")),
                str(row.get("asr_qc_severity") or "") != "reject",
                -row_float(row, "core_cps"),
                row_float(row, "start"),
            ),
        ),
        (
            "repeat_or_qc_reject",
            "重复循环 / QC reject",
            lambda row: str(row.get("asr_qc_severity") or "") == "reject"
            or repetition_changed(row)
            or str(row.get("failure_bucket") or "") == "repeat_repair_suggested",
            lambda row: (
                str(row.get("asr_qc_severity") or "") != "reject",
                not repetition_changed(row),
                -int((row.get("repetition_repair") or {}).get("run") or 0)
                if isinstance(row.get("repetition_repair"), Mapping)
                else 0,
                row_float(row, "start"),
            ),
        ),
        (
            "nonlexical_empty",
            "空文本 / 非词 fallback",
            lambda row: str(row.get("fallback_subtype") or "") == "nonlexical_text"
            or bool(row.get("align_text_empty"))
            or text_density_level(row) == "empty_or_punctuation",
            lambda row: (-row_float(row, "duration_s"), row_float(row, "start")),
        ),
        (
            "sentinel_fallback",
            "aligner sentinel fallback",
            lambda row: str(row.get("fallback_subtype") or "") == "proportional_after_sentinel",
            lambda row: (
                -row_float(row, "fallback_duration_s"),
                -row_float(row, "duration_s"),
                row_float(row, "start"),
            ),
        ),
        (
            "low_info_vocal",
            "低信息人声 / 语气词",
            lambda row: text_density_level(row)
            in {
                "short_kana_dialogue_candidate",
                "short_vocalization_candidate",
                "repeated_vocalization_candidate",
                "long_sparse_text",
            },
            lambda row: (
                text_density_level(row) != "repeated_vocalization_candidate",
                -row_float(row, "duration_s"),
                row_float(row, "start"),
            ),
        ),
        (
            "asr_qc_warn",
            "ASR QC warn",
            lambda row: str(row.get("asr_qc_severity") or "") == "warn"
            and str(row.get("fallback_subtype") or "") != "nonlexical_text",
            lambda row: (-row_float(row, "duration_s"), row_float(row, "start")),
        ),
        (
            "forced_control",
            "forced 正常对照",
            lambda row: str(row.get("alignment_quality") or "") == "forced"
            and str(row.get("fallback_type") or "") == "none"
            and str(row.get("asr_qc_severity") or "") == "ok"
            and text_density_level(row) == "normal_dialogue",
            lambda row: (row_float(row, "start"),),
        ),
    ]


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_per_bucket: int,
    max_items: int | None,
) -> list[tuple[str, str, dict[str, Any]]]:
    selected: list[tuple[str, str, dict[str, Any]]] = []
    used_chunks: set[int] = set()
    for bucket_id, bucket_label, predicate, sort_key in bucket_definitions():
        members = [row for row in rows if row_chunk_index(row) not in used_chunks and predicate(row)]
        members.sort(key=sort_key)
        for row in members[:max_per_bucket]:
            selected.append((bucket_id, bucket_label, row))
            used_chunks.add(row_chunk_index(row))
            if max_items is not None and len(selected) >= max_items:
                return selected
    selected.sort(key=lambda item: (row_float(item[2], "start"), item[0]))
    return selected


def reason_hints(row: Mapping[str, Any]) -> list[str]:
    hints: list[str] = []
    if str(row.get("asr_qc_severity") or "") == "reject":
        hints.append("asr_qc_reject")
    if str(row.get("asr_qc_severity") or "") == "warn":
        hints.append("asr_qc_warn")
    if density_review(row):
        hints.append("subtitle_density_over_4cps")
    if bool(row.get("chunk_text_leak_risk")):
        hints.append("chunk_text_leak_risk")
    if bool(row.get("chunk_repetition_risk")):
        hints.append("chunk_repetition_risk")
    if repetition_changed(row):
        hints.append("repeat_repair_suggested")
    if str(row.get("fallback_subtype") or "") == "proportional_after_sentinel":
        hints.append("aligner_sentinel")
    if str(row.get("fallback_subtype") or "") == "nonlexical_text":
        hints.append("nonlexical_text")
    level = text_density_level(row)
    if level and level != "normal_dialogue":
        hints.append(f"text_density:{level}")
    for reason in row.get("asr_qc_reasons") or []:
        if reason not in hints:
            hints.append(str(reason))
    for reason in row.get("failure_reasons") or []:
        if reason not in hints:
            hints.append(str(reason))
    return hints


def build_review_items(
    *,
    diagnostics_rows: list[dict[str, Any]],
    aligned_segments_by_chunk: dict[int, list[dict[str, Any]]],
    subtitle_cues: list[dict[str, Any]],
    context_margin_s: float,
    max_per_bucket: int,
    max_items: int | None,
    video_label: str,
) -> list[dict[str, Any]]:
    sampled = sample_rows(
        diagnostics_rows,
        max_per_bucket=max_per_bucket,
        max_items=max_items,
    )
    items: list[dict[str, Any]] = []
    bucket_ordinals: Counter[str] = Counter()
    for bucket_id, bucket_label, row in sampled:
        chunk_index = row_chunk_index(row)
        start = row_float(row, "start")
        end = row_float(row, "end")
        if end <= start:
            continue
        fallback_start = row_float(row, "fallback_window_start") or start
        fallback_end = row_float(row, "fallback_window_end") or end
        if fallback_end <= fallback_start:
            fallback_start, fallback_end = start, end
        context_start = max(0.0, start - context_margin_s)
        context_end = end + context_margin_s
        chunk_cue_rows = cues_for_range(subtitle_cues, start, end)
        fallback_cue_rows = cues_for_range(subtitle_cues, fallback_start, fallback_end)
        context_cue_rows = cues_for_range(subtitle_cues, context_start, context_end)
        aligned_rows = aligned_segments_by_chunk.get(chunk_index, [])
        ordinal = bucket_ordinals[bucket_id]
        bucket_ordinals[bucket_id] += 1
        repair = row.get("repetition_repair") if isinstance(row.get("repetition_repair"), Mapping) else {}
        repair_changed = bool(repair.get("changed"))
        text_density = row.get("text_density") if isinstance(row.get("text_density"), Mapping) else {}
        suggested_text = (
            row.get("repetition_suggested_text")
            or repair.get("suggested_text")
            or ""
        ) if repair_changed else ""
        item = {
            "index": len(items),
            "sample_id": f"asrattr-{bucket_id}-{ordinal:02d}-chunk{chunk_index:04d}",
            "bucket": bucket_id,
            "bucket_label": bucket_label,
            "video_label": video_label,
            "chunk_index": chunk_index,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration_s": round(end - start, 3),
            "context_start": round(context_start, 3),
            "context_end": round(context_end, 3),
            "fallback_window_start": round(fallback_start, 3),
            "fallback_window_end": round(fallback_end, 3),
            "fallback_duration_s": round(row_float(row, "fallback_duration_s") or (fallback_end - fallback_start), 3),
            "fallback_window_source": str(row.get("fallback_window_source") or ""),
            "chunk_duration_s": round(row_float(row, "chunk_duration_s") or (end - start), 3),
            "core_duration_s": round(row_float(row, "core_duration_s") or row_float(row, "fallback_duration_s") or (fallback_end - fallback_start), 3),
            "chunk_extra_to_fallback_s": round(row_float(row, "chunk_extra_to_fallback_s"), 3),
            "fallback_to_chunk_duration_ratio": row_float(row, "fallback_to_chunk_duration_ratio"),
            "chunk_to_fallback_duration_ratio": row_float(row, "chunk_to_fallback_duration_ratio"),
            "alignment_quality": str(row.get("alignment_quality") or ""),
            "alignment_mode": str(row.get("alignment_mode") or ""),
            "fallback_type": str(row.get("fallback_type") or ""),
            "fallback_subtype": str(row.get("fallback_subtype") or ""),
            "asr_qc_severity": str(row.get("asr_qc_severity") or ""),
            "asr_qc_reasons": list(row.get("asr_qc_reasons") or []),
            "failure_bucket": str(row.get("failure_bucket") or ""),
            "failure_reasons": list(row.get("failure_reasons") or []),
            "text_density_level": text_density_level(row),
            "text_density": dict(text_density),
            "nonlexical_text": bool(row.get("nonlexical_text")),
            "align_text_empty": bool(row.get("align_text_empty")),
            "chars_per_sec": row_float(row, "chars_per_sec"),
            "chunk_cps": row_float(row, "chunk_cps") or row_float(row, "chars_per_sec"),
            "core_cps": row_float(row, "core_cps") or row_float(row, "fallback_cps"),
            "fallback_cps": row_float(row, "fallback_cps") or row_float(row, "core_cps"),
            "chunk_text_leak_risk": bool(row.get("chunk_text_leak_risk")),
            "chunk_repetition_risk": bool(row.get("chunk_repetition_risk")),
            "repeat_profile": (
                dict(row.get("repeat_profile"))
                if isinstance(row.get("repeat_profile"), Mapping)
                else {}
            ),
            "vocalization_repetition": (
                dict(row.get("vocalization_repetition"))
                if isinstance(row.get("vocalization_repetition"), Mapping)
                else {}
            ),
            "preserve_nonlexical_repetition": bool(row.get("preserve_nonlexical_repetition")),
            "signal_quality_verdict": str(row.get("signal_quality_verdict") or ""),
            "signal_quality_reason": str(row.get("signal_quality_reason") or ""),
            "avg_logprob": row.get("avg_logprob"),
            "compression_ratio": row.get("compression_ratio"),
            "no_speech_prob": row.get("no_speech_prob"),
            "compact_chars": int(row.get("compact_chars") or 0),
            "display_text": compact_text(row.get("display_text") or row.get("text") or ""),
            "raw_text": compact_text(row.get("raw_text") or ""),
            "analysis_text": compact_text(row.get("analysis_text") or ""),
            "align_text": compact_text(row.get("align_text") or ""),
            "repetition_repair": dict(repair),
            "repetition_suggested_text": compact_text(suggested_text),
            "reason_hints": reason_hints(row),
            "subtitle_text": cue_text_for_range(subtitle_cues, start, end),
            "subtitle_context_text": cue_text_for_range(subtitle_cues, context_start, context_end),
            "subtitle_cues": [
                {
                    "start": cue["start"],
                    "end": cue["end"],
                    "text": clean_cue_text(cue.get("text")),
                }
                for cue in context_cue_rows
            ],
            "final_chunk_cues": [
                {
                    "start": cue["start"],
                    "end": cue["end"],
                    "text": clean_cue_text(cue.get("text")),
                }
                for cue in chunk_cue_rows
            ],
            "final_fallback_cues": [
                {
                    "start": cue["start"],
                    "end": cue["end"],
                    "text": clean_cue_text(cue.get("text")),
                }
                for cue in fallback_cue_rows
            ],
            "aligned_segments": aligned_rows,
        }
        items.append(item)
    return items


def count_by_key(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows).most_common())


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
    if media_mode == "audio":
        media_markup = f"""
        <div class="media-shell audio-shell">
          <audio id="media" controls preload="metadata">
            <source src="{html.escape(media_url)}" type="{html.escape(media_mime)}">
            <track kind="subtitles" label="完整日语字幕" srclang="ja" src="{html.escape(vtt_url)}" default>
          </audio>
        </div>
        <div class="caption-preview" id="captionOverlay"></div>
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
a {{ color: var(--accent); }}
.app {{ display: grid; grid-template-columns: 430px minmax(0, 1fr); min-height: 100vh; }}
.sidebar {{ max-height: 100vh; overflow: auto; border-right: 1px solid var(--line); background: #fbfbf9; }}
.side-head {{ position: sticky; top: 0; z-index: 2; padding: 12px; border-bottom: 1px solid var(--line); background: #fbfbf9; }}
h1 {{ margin: 0 0 8px; font-size: 16px; }}
h2 {{ margin: 0; font-size: 18px; overflow-wrap: anywhere; }}
h3 {{ margin: 0 0 10px; font-size: 14px; }}
.filters {{ display: grid; grid-template-columns: minmax(0, 1fr) 150px; gap: 8px; margin-top: 8px; }}
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
.grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 390px; gap: 12px; }}
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
.caption-preview {{
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
.caption-preview:empty::before {{ content: "当前没有同步字幕"; color: var(--muted); font-size: 13px; }}
.timeline {{ position: relative; height: 48px; border: 1px solid var(--line); border-radius: 6px; background: #101715; overflow: hidden; cursor: pointer; margin-top: 10px; }}
.chunk-range {{ position: absolute; top: 4px; bottom: 4px; border: 1px solid rgba(255,255,255,0.32); background: rgba(255, 255, 255, 0.10); }}
.fallback-range {{ position: absolute; top: 0; bottom: 0; background: rgba(15, 118, 110, 0.55); }}
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
@media (max-width: 1100px) {{
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
        <input id="searchInput" placeholder="搜索 bucket / chunk / 文本 / reason">
        <select id="bucketSelect">
          <option value="">全部 bucket</option>
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
        <button class="primary" id="playChunkBtn">播放 ASR chunk</button>
        <button id="playFallbackBtn">播放 speech/fallback core</button>
        <button id="playContextBtn">播放上下文</button>
        <button id="downloadBtn">下载审计 JSONL</button>
      </div>
    </div>
    <div class="grid">
      <section class="panel">
        {media_markup}
        <div class="timeline" id="timeline">
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
          <a href="{html.escape(media_url)}">直接打开媒体</a>
          <a href="{html.escape(vtt_url)}">完整 VTT</a>
        </div>
        <p class="hint">浅色外框是送入 ASR 的 chunk；绿色是 speech/fallback core window，代表当前 chunk 的实际语音/回退计时范围。播放会在当前模式终点自动暂停。</p>
        <p class="hint error" id="mediaError"></p>
      </section>
      <aside class="panel">
        <h3>诊断指标</h3>
        <div class="kv" id="metrics"></div>
      </aside>
    </div>
    <section class="panel">
      <h3>文本对照</h3>
      <div class="kv" style="grid-template-columns: 120px minmax(0, 1fr);">
        <div>ASR display</div><div class="text-box" id="displayText"></div>
        <div>ASR raw</div><div class="text-box" id="rawText"></div>
        <div>align_text</div><div class="text-box" id="alignText"></div>
        <div>analysis_text</div><div class="text-box" id="analysisText"></div>
        <div>重复修复</div><div class="text-box" id="suggestedText"></div>
      </div>
      <p class="hint">ASR raw/display 是模型转写；aligned segments / 完整字幕是 forced alignment 与后处理后的结果。若 ASR 文本有一部分没有被 aligner 对齐，它可能不会出现在最终字幕轨里。</p>
    </section>
    <section class="panel">
      <h3>最终字幕成效（送往 LLM 前的日文 cue）</h3>
      <div class="grid">
        <div>
          <h3>ASR chunk 内最终字幕</h3>
          <div class="cue-list" id="finalChunkCueList"></div>
        </div>
        <div>
          <h3>speech/fallback core 内最终字幕</h3>
          <div class="cue-list" id="finalFallbackCueList"></div>
        </div>
      </div>
      <h3 style="margin-top:12px">上下文最终字幕</h3>
      <div class="cue-list" id="cueList"></div>
      <p class="hint">这里展示的是最终日文字幕 cue plan，即送往 LLM 翻译前的字幕效果。视频上的同步字幕也来自同一份完整 VTT。</p>
    </section>
    <section class="panel">
      <h3>aligned segments</h3>
      <div class="cue-list" id="alignedList"></div>
    </section>
    <section class="panel">
      <h3>人工归因标签</h3>
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
const STORAGE_KEY = "javtrans-asr-attribution-audit:" + SUMMARY.label_schema_version + ":" + SUMMARY.dataset_id;
const LABELS = [
  ["true_low_info_vocal", "真实低信息人声"],
  ["asr_hallucination", "ASR 幻觉/错听"],
  ["non_speech_noise", "非语音噪声/BGM"],
  ["overlap_or_multi_speaker", "多人/重叠语音"],
  ["weak_voice", "轻声/弱人声"],
  ["boundary_too_short_context", "上下文切太短"],
  ["boundary_too_long_mixed", "chunk 混入多句/噪声"],
  ["text_ok", "文本可用"],
  ["timing_ok", "时间轴准确"],
  ["timing_start_early", "开头偏早"],
  ["timing_start_late", "开头偏晚"],
  ["timing_end_early", "结尾偏早"],
  ["timing_end_late", "结尾偏晚"],
  ["boundary_too_split", "边界切得过碎"],
  ["drop_or_review", "需要人工复核"],
  ["uncertain", "不确定"]
];
let currentIndex = 0;
let filtered = [...ITEMS];
let playMode = "chunk";
let annotations = loadAnnotations();
const media = document.getElementById("media");
const itemList = document.getElementById("itemList");
const searchInput = document.getElementById("searchInput");
const bucketSelect = document.getElementById("bucketSelect");
const notes = document.getElementById("notes");
const timeline = document.getElementById("timeline");
const chunkRange = document.getElementById("chunkRange");
const fallbackRange = document.getElementById("fallbackRange");
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
    item.bucket,
    item.bucket_label,
    item.chunk_index,
    item.alignment_quality,
    item.fallback_subtype,
    item.asr_qc_severity,
    (item.asr_qc_reasons || []).join(","),
    (item.reason_hints || []).join(","),
    item.display_text,
    item.subtitle_text
  ].join("\\n").toLowerCase();
}}
function setupBuckets() {{
  const buckets = [];
  for (const item of ITEMS) {{
    if (!buckets.some(row => row[0] === item.bucket)) buckets.push([item.bucket, item.bucket_label]);
  }}
  for (const [value, label] of buckets) {{
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    bucketSelect.appendChild(option);
  }}
}}
function applyFilters() {{
  const query = searchInput.value.trim().toLowerCase();
  const bucket = bucketSelect.value;
  filtered = ITEMS.filter(item => (!bucket || item.bucket === bucket) && (!query || itemSearchText(item).includes(query)));
  if (!filtered.includes(ITEMS[currentIndex])) {{
    currentIndex = ITEMS.indexOf(filtered[0] || ITEMS[0] || null);
    if (currentIndex < 0) currentIndex = 0;
  }}
  renderList();
  renderCurrent(false);
}}
function badgeClass(item) {{
  if (item.chunk_text_leak_risk || item.chunk_repetition_risk) return "danger";
  if (item.asr_qc_severity === "reject") return "danger";
  if (item.asr_qc_severity === "warn" || item.fallback_type !== "none") return "warn";
  return "";
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
      <div class="item-title">#${{item.index + 1}} chunk ${{item.chunk_index}} <span class="badge ${{badgeClass(item)}}">${{escapeHtml(item.bucket_label)}}</span></div>
      <div class="meta">core ${{fmt(item.fallback_window_start)}}-${{fmt(item.fallback_window_end)}} · chunk ${{fmt(item.start)}}-${{fmt(item.end)}} · core CPS=${{Number(item.core_cps || item.fallback_cps || 0).toFixed(2)}} · chunk CPS=${{Number(item.chunk_cps || item.chars_per_sec || 0).toFixed(2)}} · ${{escapeHtml(item.alignment_quality)}}/${{escapeHtml(item.fallback_subtype || "none")}} · QC=${{escapeHtml(item.asr_qc_severity || "ok")}}</div>
      <div class="meta">${{escapeHtml(item.display_text || item.subtitle_text || "(empty)")}}</div>
    `;
    itemList.appendChild(div);
  }}
}}
function setMetrics(item) {{
  const rows = [
    ["bucket", item.bucket_label],
    ["媒体", item.video_label],
    ["chunk", item.chunk_index],
    ["ASR chunk", `${{fmt(item.start)}}-${{fmt(item.end)}} (${{Number(item.duration_s).toFixed(2)}}s)`],
    ["speech/fallback core", `${{fmt(item.fallback_window_start)}}-${{fmt(item.fallback_window_end)}} (${{Number(item.fallback_duration_s).toFixed(2)}}s) · source=${{item.fallback_window_source || "-"}}`],
    ["chunk/core", `${{Number(item.chunk_duration_s || item.duration_s || 0).toFixed(2)}}s / ${{Number(item.core_duration_s || item.fallback_duration_s || 0).toFixed(2)}}s · ratio=${{Number(item.fallback_to_chunk_duration_ratio || 0).toFixed(3)}}`],
    ["alignment", `${{item.alignment_quality}} / ${{item.alignment_mode}}`],
    ["fallback", `${{item.fallback_type}} / ${{item.fallback_subtype}}`],
    ["ASR QC", `${{item.asr_qc_severity}} · ${{(item.asr_qc_reasons || []).join(", ")}}`],
    ["signal", `${{item.signal_quality_verdict || "-"}} · ${{item.signal_quality_reason || "-"}}`],
    ["avg_logprob", item.avg_logprob === null || item.avg_logprob === undefined ? "-" : Number(item.avg_logprob).toFixed(3)],
    ["no_speech", item.no_speech_prob === null || item.no_speech_prob === undefined ? "-" : Number(item.no_speech_prob).toFixed(3)],
    ["compression", item.compression_ratio === null || item.compression_ratio === undefined ? "-" : Number(item.compression_ratio).toFixed(3)],
    ["text_density", `${{item.text_density_level}}`],
    ["chunk CPS", Number(item.chunk_cps || item.chars_per_sec || 0).toFixed(2)],
    ["core/fallback CPS", Number(item.core_cps || item.fallback_cps || 0).toFixed(2)],
    ["chunk text leak risk", item.chunk_text_leak_risk ? "yes" : "no"],
    ["chunk repetition risk", item.chunk_repetition_risk ? "yes" : "no"],
    ["repeat profile", `${{(item.repeat_profile || {{}}).unit || ""}} x${{(item.repeat_profile || {{}}).run || 0}} · ratio=${{Number((item.repeat_profile || {{}}).ratio || 0).toFixed(2)}}`],
    ["nonlexical repeat", item.preserve_nonlexical_repetition ? "preserve_with_review" : "standard"],
    ["compact chars", item.compact_chars],
    ["failure", `${{item.failure_bucket}} · ${{(item.failure_reasons || []).join(", ")}}`],
    ["hints", (item.reason_hints || []).join(", ")]
  ];
  document.getElementById("metrics").innerHTML = rows
    .map(([key, value]) => `<div>${{escapeHtml(key)}}</div><div>${{escapeHtml(value)}}</div>`)
    .join("");
}}
function renderCueGroup(rootId, cues, emptyText) {{
  const root = document.getElementById(rootId);
  if (!cues || cues.length === 0) {{
    root.innerHTML = `<div class="hint">${{escapeHtml(emptyText)}}</div>`;
  }} else {{
    root.innerHTML = cues.map(cue => `
      <div class="cue">
        <div class="meta">${{fmt(cue.start)}}-${{fmt(cue.end)}}</div>
        <div>${{escapeHtml(cue.text)}}</div>
      </div>
    `).join("");
  }}
}}
function renderCueList(item) {{
  renderCueGroup("finalChunkCueList", item.final_chunk_cues, "该 ASR chunk 内没有最终字幕 cue。");
  renderCueGroup("finalFallbackCueList", item.final_fallback_cues, "该 speech/fallback core 内没有最终字幕 cue。");
  renderCueGroup("cueList", item.subtitle_cues, "该上下文内没有最终字幕 cue。");
  const aligned = document.getElementById("alignedList");
  if (!item.aligned_segments || item.aligned_segments.length === 0) {{
    aligned.innerHTML = `<div class="hint">该 chunk 没有 aligned segment。</div>`;
  }} else {{
    aligned.innerHTML = item.aligned_segments.map(seg => `
      <div class="cue">
        <div class="meta">aligned ${{fmt(seg.start)}}-${{fmt(seg.end)}}</div>
        <div>${{escapeHtml(seg.text)}}</div>
      </div>
    `).join("");
  }}
}}
function renderLabels(item) {{
  const root = document.getElementById("labelButtons");
  root.innerHTML = "";
  const ann = annotations[item.sample_id] || {{}};
  const labels = Array.isArray(ann.labels) ? ann.labels : [];
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
  setRange(chunkRange, item.start, item.end, 0.5);
  setRange(fallbackRange, item.fallback_window_start, item.fallback_window_end, 0.5);
  const t = media.currentTime || 0;
  const pct = Math.max(0, Math.min(100, ((t - item.context_start) / width) * 100));
  cursor.style.left = `${{pct}}%`;
  document.getElementById("nowText").textContent = fmt(t);
  document.getElementById("captionOverlay").textContent = activeCueText(t);
  const stopAt = playMode === "context" ? item.context_end : playMode === "fallback" ? item.fallback_window_end : item.end;
  if (!media.paused && t >= stopAt) media.pause();
}}
function renderCurrent(seek) {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  document.getElementById("clipTitle").textContent = `#${{item.index + 1}} ${{item.bucket_label}} · chunk ${{item.chunk_index}}`;
  document.getElementById("clipMeta").textContent = `core ${{fmt(item.fallback_window_start)}}-${{fmt(item.fallback_window_end)}} · chunk ${{fmt(item.start)}}-${{fmt(item.end)}} · fallback ${{item.fallback_subtype || "none"}} · QC ${{item.asr_qc_severity || "ok"}}`;
  document.getElementById("rangeStart").textContent = fmt(item.context_start);
  document.getElementById("rangeEnd").textContent = fmt(item.context_end);
  document.getElementById("displayText").textContent = item.display_text || "(empty)";
  document.getElementById("rawText").textContent = item.raw_text || "(empty)";
  document.getElementById("alignText").textContent = item.align_text || "(empty)";
  document.getElementById("analysisText").textContent = item.analysis_text || "(empty)";
  const repair = item.repetition_repair || {{}};
  document.getElementById("suggestedText").textContent =
    repair.changed && item.repetition_suggested_text
      ? `重复修复建议：${{item.repetition_suggested_text}}`
      : "无重复修复建议";
  setMetrics(item);
  renderCueList(item);
  renderLabels(item);
  if (seek) media.currentTime = item.start;
  updateTimeline();
}}
function playCurrent(mode) {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  playMode = mode;
  media.currentTime = mode === "context" ? item.context_start : mode === "fallback" ? item.fallback_window_start : item.start;
  media.play().catch(() => {{}});
}}
function exportRows() {{
  return ITEMS.map(item => ({{
    sample_id: item.sample_id,
    bucket: item.bucket,
    bucket_label: item.bucket_label,
    video_label: item.video_label,
    chunk_index: item.chunk_index,
    start: item.start,
    end: item.end,
    duration_s: item.duration_s,
    fallback_window_start: item.fallback_window_start,
    fallback_window_end: item.fallback_window_end,
    chunk_duration_s: item.chunk_duration_s,
    core_duration_s: item.core_duration_s,
    chunk_extra_to_fallback_s: item.chunk_extra_to_fallback_s,
    fallback_to_chunk_duration_ratio: item.fallback_to_chunk_duration_ratio,
    alignment_quality: item.alignment_quality,
    fallback_subtype: item.fallback_subtype,
    asr_qc_severity: item.asr_qc_severity,
    asr_qc_reasons: item.asr_qc_reasons,
    text_density_level: item.text_density_level,
    chunk_cps: item.chunk_cps,
    core_cps: item.core_cps,
    fallback_cps: item.fallback_cps,
    chunk_text_leak_risk: item.chunk_text_leak_risk,
    chunk_repetition_risk: item.chunk_repetition_risk,
    repeat_profile: item.repeat_profile,
    vocalization_repetition: item.vocalization_repetition,
    preserve_nonlexical_repetition: item.preserve_nonlexical_repetition,
    signal_quality_verdict: item.signal_quality_verdict,
    signal_quality_reason: item.signal_quality_reason,
    avg_logprob: item.avg_logprob,
    compression_ratio: item.compression_ratio,
    no_speech_prob: item.no_speech_prob,
    display_text: item.display_text,
    repetition_suggested_text: item.repetition_suggested_text,
    subtitle_text: item.subtitle_text,
    reason_hints: item.reason_hints,
    ...(annotations[item.sample_id] || {{}})
  }}));
}}
function downloadJsonl() {{
  const text = exportRows().map(row => JSON.stringify(row)).join("\\n") + "\\n";
  const blob = new Blob([text], {{type: "application/jsonl;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "manual_asr_attribution_labels.jsonl";
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
document.getElementById("playFallbackBtn").onclick = () => playCurrent("fallback");
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
bucketSelect.addEventListener("change", applyFilters);
document.getElementById("summaryLine").textContent = `${{ITEMS.length}} 条归因样本 · 完整字幕 ${{CUES.length}} cues`;
try {{
  for (const track of media.textTracks) track.mode = "hidden";
}} catch (_) {{}}
setupBuckets();
applyFilters();
renderCurrent(true);
</script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ASR attribution audit page from alignment diagnostics."
    )
    parser.add_argument("--diagnostics", required=True, help="diagnostics.jsonl")
    parser.add_argument("--aligned", required=True, help="aligned_segments.json")
    parser.add_argument("--subtitle-srt", required=True, help="Full Japanese SRT")
    parser.add_argument("--media", required=True, help="Source video/audio path")
    parser.add_argument(
        "--media-mode",
        choices=("audio", "video"),
        default="audio",
        help="Generate an audio-only or video audit page.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--output-dir", default="agents/audits/asr-attribution-audit")
    parser.add_argument("--title", default="ASR 低信息/幻觉归因审计")
    parser.add_argument("--video-label", default="")
    parser.add_argument("--context-margin-s", type=float, default=2.0)
    parser.add_argument("--max-per-bucket", type=int, default=14)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--update-entrypoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.context_margin_s < 0:
        parser.error("--context-margin-s must be non-negative")
    if args.max_per_bucket <= 0:
        parser.error("--max-per-bucket must be positive")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    diagnostics_path = project_path(args.diagnostics)
    aligned_path = project_path(args.aligned)
    subtitle_srt_path = project_path(args.subtitle_srt)
    source_media_path = project_path(args.media)
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

    diagnostics_rows = read_jsonl(diagnostics_path)
    aligned_rows = read_aligned_segments(aligned_path)
    subtitle_cues = read_srt_cues(subtitle_srt_path)
    if not diagnostics_rows:
        raise SystemExit(f"No diagnostics rows found: {diagnostics_path}")
    if not subtitle_cues:
        raise SystemExit(f"No subtitle cues parsed: {subtitle_srt_path}")

    video_label = args.video_label or str(diagnostics_rows[0].get("video") or source_media_path.stem)
    items = build_review_items(
        diagnostics_rows=diagnostics_rows,
        aligned_segments_by_chunk=aligned_by_chunk(aligned_rows),
        subtitle_cues=subtitle_cues,
        context_margin_s=args.context_margin_s,
        max_per_bucket=args.max_per_bucket,
        max_items=args.max_items,
        video_label=video_label,
    )
    if not items:
        raise SystemExit("No review items produced")

    full_vtt = output_dir / "full.ja.vtt"
    write_vtt(full_vtt, subtitle_cues)
    manifest_path = output_dir / "asr_attribution_review_items.jsonl"
    write_jsonl(manifest_path, items)
    summary = {
        "dataset_id": output_dir.name,
        "title": args.title,
        "label_schema_version": "asr_attribution_v1",
        "video_label": video_label,
        "media_mode": args.media_mode,
        "media_path": project_rel(media_path),
        "media_mime": media_mime,
        "source_media_path": project_rel(source_media_path),
        "diagnostics_source": project_rel(diagnostics_path),
        "aligned_source": project_rel(aligned_path),
        "subtitle_srt_source": project_rel(subtitle_srt_path),
        "html": project_rel(output_dir / "index.html"),
        "full_vtt": project_rel(full_vtt),
        "review_items": project_rel(manifest_path),
        "review_item_count": len(items),
        "subtitle_cue_count": len(subtitle_cues),
        "diagnostics_row_count": len(diagnostics_rows),
        "bucket_counts": dict(Counter(item["bucket"] for item in items).most_common()),
        "density_over_4cps_count": sum(
            1
            for row in diagnostics_rows
            if row_float(row, "core_cps") > 4.0 or row_float(row, "fallback_cps") > 4.0
        ),
        "chunk_text_leak_risk_count": sum(
            1 for row in diagnostics_rows if bool(row.get("chunk_text_leak_risk"))
        ),
        "chunk_repetition_risk_count": sum(
            1 for row in diagnostics_rows if bool(row.get("chunk_repetition_risk"))
        ),
        "alignment_quality_counts": count_by_key(diagnostics_rows, "alignment_quality"),
        "fallback_subtype_counts": count_by_key(diagnostics_rows, "fallback_subtype"),
        "asr_qc_severity_counts": count_by_key(diagnostics_rows, "asr_qc_severity"),
        "text_density_counts": count_by_key(diagnostics_rows, "text_density_level"),
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
        update_audit_entrypoints(latest_html=output_dir / "index.html", title=args.title)
    print(f"html={project_rel(output_dir / 'index.html')}")
    print(f"items={len(items)}")
    print(f"summary={project_rel(output_dir / 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
