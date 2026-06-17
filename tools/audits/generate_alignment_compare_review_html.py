#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints


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
<p>
  自动跳转已关闭。打开
  <a href="{html.escape(latest_rel)}">{html.escape(title)}</a>。
</p>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_audit_index(*, audit_root: Path, latest_html: Path, title: str, summary: Mapping[str, Any]) -> None:
    latest_rel = rel_url(latest_html, from_dir=audit_root)
    known_entries = [
        (
            latest_rel,
            "最新审计页",
            title,
        ),
        (
            "current-video-audit/index.html",
            "历史视频审计页",
            "原视频片段 + ASR 字幕叠加，alignment failure priority 样本。",
        ),
    ]
    generated_entries: list[tuple[str, str, str]] = []
    for index_path in sorted(audit_root.glob("*/index.html")):
        rel = rel_url(index_path, from_dir=audit_root)
        if rel in {entry[0] for entry in known_entries}:
            continue
        entry_title = index_path.parent.name
        summary_path = index_path.parent / "summary.json"
        desc = "审计页面"
        if summary_path.exists():
            try:
                payload = read_json(summary_path)
                if isinstance(payload, Mapping):
                    count = payload.get("review_item_count")
                    video = payload.get("video")
                    parts = []
                    if count is not None:
                        parts.append(f"{count} 条")
                    if video:
                        parts.append(str(video))
                    if parts:
                        desc = " · ".join(parts)
            except Exception:
                pass
        generated_entries.append((rel, entry_title, desc))

    def entry_html(href: str, label: str, desc: str) -> str:
        return (
            f'  <a class="entry" href="{html.escape(href)}">\n'
            f"    <strong>{html.escape(label)}</strong>\n"
            f"    {html.escape(desc)}\n"
            "  </a>"
        )

    entries_html = "\n".join(
        entry_html(*entry) for entry in [*known_entries, *generated_entries]
    )
    outcome_counts = summary.get("outcome_counts") or {}
    outcome_text = ", ".join(f"{key}={value}" for key, value in outcome_counts.items())
    latest_meta = (
        f"当前 latest: {project_rel(latest_html)}"
        + (f"；{outcome_text}" if outcome_text else "")
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
main {{
  max-width: 920px;
  margin: 0 auto;
  padding: 32px 18px;
}}
h1 {{
  margin: 0 0 18px;
  font-size: 24px;
}}
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
.entry strong {{
  display: block;
  margin-bottom: 4px;
  color: #0f766e;
}}
.muted {{
  color: #66706c;
  font-size: 13px;
}}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
</style>
</head>
<body>
<main>
  <h1>SpeechBoundary-JA 审计入口</h1>
{entries_html}
  <p class="muted">
    {html.escape(latest_meta)}
  </p>
  <p class="muted">
    后续人工审计统一放在 <code>agents/audits/</code>。
    云端训练监控脚本仍保留在 <code>agents/temp/check_cloud_qwen_sft.sh</code>。
  </p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def maybe_update_audit_entrypoints(*, output_dir: Path, title: str, summary: Mapping[str, Any]) -> None:
    update_audit_entrypoints(latest_html=output_dir / "index.html", title=title)


def row_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def compact_text(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def is_baseline_fallback_target(row: Mapping[str, Any]) -> bool:
    fallback_type = str(row.get("fallback_type") or "")
    return fallback_type not in {"", "none"}


def normalized_row(row: Mapping[str, Any]) -> dict[str, Any]:
    start = row_float(row, "start")
    end = row_float(row, "end")
    duration = row_float(row, "duration_s") or max(0.0, end - start)
    return {
        "chunk_index": int(row.get("chunk_index") or 0),
        "position": int(row.get("position") or 0),
        "start": round(start, 3),
        "end": round(end, 3),
        "duration_s": round(duration, 3),
        "alignment_quality": str(row.get("alignment_quality") or ""),
        "alignment_mode": str(row.get("alignment_mode") or ""),
        "fallback_type": str(row.get("fallback_type") or ""),
        "fallback_subtype": str(row.get("fallback_subtype") or ""),
        "failure_bucket": str(row.get("failure_bucket") or ""),
        "failure_reasons": list(row.get("failure_reasons") or []),
        "display_text": compact_text(row.get("display_text") or row.get("text") or ""),
        "align_text": compact_text(row.get("align_text") or ""),
        "raw_text": compact_text(row.get("raw_text") or ""),
        "chars_per_sec": row.get("chars_per_sec"),
        "word_count": (row.get("word_timing") or {}).get("word_count"),
    }


def outcome_for(candidate_rows: list[dict[str, Any]]) -> str:
    if not candidate_rows:
        return "no_match"
    qualities = {str(row.get("alignment_quality") or "") for row in candidate_rows}
    fallback_types = {str(row.get("fallback_type") or "") for row in candidate_rows}
    has_forced = "forced" in qualities or "partial" in qualities
    has_fallback = bool(fallback_types - {"", "none"})
    has_review = "drop_or_review" in qualities
    if has_forced and not has_fallback:
        return "resolved"
    if has_forced and has_fallback:
        return "mixed"
    if has_fallback:
        return "still_fallback"
    if has_review:
        return "review"
    return "changed"


def build_review_items(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    context_margin_s: float,
    max_items: int | None,
) -> list[dict[str, Any]]:
    selected = [row for row in baseline_rows if is_baseline_fallback_target(row)]
    selected.sort(key=lambda row: (row_float(row, "start"), int(row.get("chunk_index") or 0)))
    items: list[dict[str, Any]] = []
    for index, baseline_row in enumerate(selected):
        b_start = row_float(baseline_row, "start")
        b_end = row_float(baseline_row, "end")
        matched = [
            row
            for row in candidate_rows
            if interval_overlap(b_start, b_end, row_float(row, "start"), row_float(row, "end")) >= 0.05
        ]
        matched.sort(key=lambda row: (row_float(row, "start"), row_float(row, "end")))
        all_starts = [b_start, *(row_float(row, "start") for row in matched)]
        all_ends = [b_end, *(row_float(row, "end") for row in matched)]
        item_start = max(0.0, min(all_starts) - context_margin_s)
        item_end = max(all_ends) + context_margin_s
        baseline_norm = normalized_row(baseline_row)
        candidate_norm = [normalized_row(row) for row in matched]
        outcome = outcome_for(candidate_norm)
        items.append(
            {
                "index": index,
                "sample_id": f"sample-a-fallback-{index:04d}-chunk{baseline_norm['chunk_index']:04d}",
                "video": str(baseline_row.get("video") or "sample-a"),
                "review_start": round(item_start, 3),
                "review_end": round(item_end, 3),
                "baseline": baseline_norm,
                "candidate": candidate_norm,
                "outcome": outcome,
                "candidate_count": len(candidate_norm),
            }
        )
        if max_items is not None and len(items) >= max_items:
            break
    return items


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


def clean_cue_text(text: Any) -> str:
    return str(text or "").replace("\r", "").strip() or "..."


def segment_rows(aligned_path: Path) -> list[dict[str, Any]]:
    data = read_json(aligned_path)
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(data.get("segments") or []):
        if not isinstance(segment, Mapping):
            continue
        start = row_float(segment, "start")
        end = row_float(segment, "end")
        if end <= start:
            continue
        rows.append(
            {
                "index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": clean_cue_text(segment.get("text")),
                "source_chunk_index": segment.get("source_chunk_index"),
            }
        )
    return rows


def write_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                str(index),
                f"{srt_timestamp(segment['start'])} --> {srt_timestamp(segment['end'])}",
                clean_cue_text(segment.get("text")),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.extend(
            [
                f"{vtt_timestamp(segment['start'])} --> {vtt_timestamp(segment['end'])}",
                clean_cue_text(segment.get("text")),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def page_template(
    *,
    title: str,
    video_url: str,
    baseline_vtt_url: str,
    candidate_vtt_url: str,
    items: list[dict[str, Any]],
    baseline_segments: list[dict[str, Any]],
    candidate_segments: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    data_json = json.dumps(items, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    baseline_json = json.dumps(baseline_segments, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    candidate_json = json.dumps(candidate_segments, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    summary_json = json.dumps(summary, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f5f6f4;
  --panel: #ffffff;
  --ink: #1f2523;
  --muted: #66706c;
  --line: #d9dedb;
  --accent: #0f766e;
  --accent-soft: #dff2ee;
  --warn: #9a6500;
  --danger: #b42318;
  --ok: #16794c;
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
.app {{ display: grid; grid-template-columns: 390px minmax(0, 1fr); min-height: 100vh; }}
.sidebar {{ max-height: 100vh; overflow: auto; border-right: 1px solid var(--line); background: #fbfbf9; }}
.side-head {{ position: sticky; top: 0; z-index: 2; padding: 12px; border-bottom: 1px solid var(--line); background: #fbfbf9; }}
h1 {{ margin: 0 0 8px; font-size: 16px; }}
h2 {{ margin: 0; font-size: 18px; overflow-wrap: anywhere; }}
h3 {{ margin: 0 0 8px; font-size: 14px; }}
.filters {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }}
.filters input, .filters select {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 7px 8px; background: #fff; }}
.item {{ display: grid; gap: 4px; padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }}
.item:hover {{ background: #eef3ef; }}
.item.active {{ background: var(--accent-soft); }}
.item-title {{ font-weight: 650; overflow-wrap: anywhere; }}
.meta, .hint {{ color: var(--muted); font-size: 12px; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 2px 7px; font-size: 12px; border: 1px solid var(--line); background: #fff; }}
.badge.resolved {{ color: var(--ok); border-color: #a7d9c1; background: #eef8f2; }}
.badge.mixed {{ color: var(--warn); border-color: #dfc987; background: #fff8e3; }}
.badge.still_fallback {{ color: var(--danger); border-color: #efb5af; background: #fff0ee; }}
.badge.review {{ color: var(--warn); border-color: #dfc987; background: #fff8e3; }}
.workspace {{ max-height: 100vh; overflow: auto; padding: 16px; }}
.topbar {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 12px; }}
.actions, .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.actions {{ justify-content: flex-end; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
.video-shell {{ position: relative; background: #070a09; border-radius: 8px; overflow: hidden; }}
video {{ display: block; width: 100%; max-height: 58vh; background: #070a09; }}
.dual-captions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }}
.caption-box {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px; min-height: 64px; background: #fbfbf9; }}
.caption-title {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
.caption-text {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px; line-height: 1.35; max-height: 120px; overflow: auto; }}
.timeline {{ position: relative; height: 42px; border: 1px solid var(--line); border-radius: 6px; background: #111815; overflow: hidden; cursor: pointer; margin-top: 10px; }}
.range {{ position: absolute; top: 0; bottom: 0; background: rgba(15, 118, 110, 0.42); }}
.cursor {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; }}
.timeline-labels {{ display: flex; justify-content: space-between; margin-top: 4px; color: var(--muted); font-size: 12px; }}
.compare-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.run-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfbf9; }}
.run-head {{ display: flex; justify-content: space-between; gap: 8px; align-items: baseline; margin-bottom: 8px; }}
.run-title {{ font-weight: 700; }}
.chunk {{ border-top: 1px solid var(--line); padding-top: 8px; margin-top: 8px; }}
.kv {{ display: grid; grid-template-columns: 122px minmax(0, 1fr); gap: 4px 8px; font-size: 12px; }}
.kv div:nth-child(odd) {{ color: var(--muted); }}
.text {{ margin-top: 8px; white-space: pre-wrap; overflow-wrap: anywhere; }}
.label-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }}
textarea {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px; min-height: 72px; resize: vertical; }}
@media (max-width: 980px) {{
  .app {{ grid-template-columns: 1fr; }}
  .sidebar {{ max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--line); }}
  .workspace {{ max-height: none; }}
  .compare-grid, .dual-captions, .label-grid {{ grid-template-columns: 1fr; }}
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
        <input id="searchInput" placeholder="搜索文本 / chunk / 状态">
        <select id="outcomeFilter"><option value="">全部状态</option></select>
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
        <button class="primary" id="playBtn">播放片段</button>
        <button id="downloadBtn">下载审计 JSONL</button>
      </div>
    </div>
    <section class="panel">
      <div class="video-shell">
        <video id="video" controls preload="metadata" playsinline>
          <source src="{html.escape(video_url)}" type="video/mp4">
          <track kind="subtitles" label="baseline" srclang="ja" src="{html.escape(baseline_vtt_url)}">
          <track kind="subtitles" label="island split" srclang="ja" src="{html.escape(candidate_vtt_url)}">
        </video>
      </div>
      <div class="dual-captions">
        <div class="caption-box">
          <div class="caption-title">Baseline 当前字幕</div>
          <div class="caption-text" id="baselineNow"></div>
        </div>
        <div class="caption-box">
          <div class="caption-title">Opt-in 当前字幕</div>
          <div class="caption-text" id="candidateNow"></div>
        </div>
      </div>
      <div class="timeline" id="timeline">
        <div class="range" id="range"></div>
        <div class="cursor" id="cursor"></div>
      </div>
      <div class="timeline-labels">
        <span id="rangeStart"></span>
        <span id="nowText"></span>
        <span id="rangeEnd"></span>
      </div>
      <p class="hint">页面展示 baseline fallback chunk 与 opt-in Boundary Refiner 后的时间重叠片段；这里是审计视图，不代表最终字幕分块。</p>
    </section>
    <section class="panel">
      <h3>本条对比</h3>
      <div class="compare-grid">
        <div class="run-card">
          <div class="run-head"><span class="run-title">Baseline</span><span id="baselineBadge" class="badge"></span></div>
          <div id="baselineChunks"></div>
        </div>
        <div class="run-card">
          <div class="run-head"><span class="run-title">Opt-in Island Split</span><span id="candidateBadge" class="badge"></span></div>
          <div id="candidateChunks"></div>
        </div>
      </div>
    </section>
    <section class="panel">
      <h3>人工结论</h3>
      <div class="label-grid" id="labelButtons"></div>
      <textarea id="notes" placeholder="备注"></textarea>
    </section>
  </main>
</div>
<script>
const ITEMS = {data_json};
const BASELINE_SEGMENTS = {baseline_json};
const CANDIDATE_SEGMENTS = {candidate_json};
const SUMMARY = {summary_json};
const STORAGE_KEY = "speech-boundary-ja-alignment-compare-review:" + SUMMARY.dataset_id;
const LABELS = [
  ["opt_in_better", "Opt-in 更好"],
  ["same", "差不多"],
  ["opt_in_worse", "Opt-in 更差"],
  ["non_speech", "片段无有效语音"],
  ["needs_timing_fix", "仍需时间轴修正"],
  ["uncertain", "不确定"]
];
let currentIndex = 0;
let filtered = [...ITEMS];
let annotations = loadAnnotations();
const video = document.getElementById("video");
const itemList = document.getElementById("itemList");
const searchInput = document.getElementById("searchInput");
const outcomeFilter = document.getElementById("outcomeFilter");
const range = document.getElementById("range");
const cursor = document.getElementById("cursor");
const timeline = document.getElementById("timeline");
const notes = document.getElementById("notes");

function loadAnnotations() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
  catch (_) {{ return {{}}; }}
}}
function saveAnnotations() {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
}}
function fmt(t) {{
  const v = Math.max(0, Number(t || 0));
  const m = Math.floor(v / 60);
  const s = (v - m * 60).toFixed(2).padStart(5, "0");
  return `${{m}}:${{s}}`;
}}
function badgeClass(outcome) {{
  if (outcome === "resolved") return "resolved";
  if (outcome === "mixed") return "mixed";
  if (outcome === "still_fallback") return "still_fallback";
  if (outcome === "review") return "review";
  return "";
}}
function outcomeText(outcome) {{
  return {{
    resolved: "已改善",
    mixed: "部分改善",
    still_fallback: "仍 fallback",
    review: "需复核",
    changed: "已变化",
    no_match: "无匹配"
  }}[outcome] || outcome;
}}
function textForSearch(item) {{
  return [
    item.sample_id,
    item.outcome,
    item.baseline.display_text,
    item.baseline.align_text,
    ...item.candidate.map(c => c.display_text),
    ...item.candidate.map(c => c.align_text)
  ].join("\\n").toLowerCase();
}}
function renderFilters() {{
  const values = [...new Set(ITEMS.map(i => i.outcome))].sort();
  for (const value of values) {{
    const option = document.createElement("option");
    option.value = value;
    option.textContent = outcomeText(value);
    outcomeFilter.appendChild(option);
  }}
}}
function applyFilters() {{
  const q = searchInput.value.trim().toLowerCase();
  const outcome = outcomeFilter.value;
  filtered = ITEMS.filter(item => (!outcome || item.outcome === outcome) && (!q || textForSearch(item).includes(q)));
  if (!filtered.includes(ITEMS[currentIndex])) currentIndex = ITEMS.indexOf(filtered[0] || ITEMS[0] || null);
  if (currentIndex < 0) currentIndex = 0;
  renderList();
  renderCurrent();
}}
function renderList() {{
  itemList.innerHTML = "";
  for (const item of filtered) {{
    const div = document.createElement("div");
    div.className = "item" + (ITEMS[currentIndex] === item ? " active" : "");
    div.onclick = () => {{ currentIndex = ITEMS.indexOf(item); renderCurrent(); renderList(); }};
    const base = item.baseline;
    div.innerHTML = `
      <div class="item-title">#${{item.index + 1}} chunk ${{base.chunk_index}} <span class="badge ${{badgeClass(item.outcome)}}">${{outcomeText(item.outcome)}}</span></div>
      <div class="meta">${{fmt(item.review_start)}}-${{fmt(item.review_end)}} · opt-in chunks=${{item.candidate_count}}</div>
      <div class="meta">${{escapeHtml(base.display_text || base.align_text || "(empty)")}}</div>
    `;
    itemList.appendChild(div);
  }}
}}
function escapeHtml(text) {{
  return String(text || "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}
function chunkHtml(chunk) {{
  return `<div class="chunk">
    <div class="kv">
      <div>chunk</div><div>${{chunk.chunk_index}}</div>
      <div>time</div><div>${{fmt(chunk.start)}}-${{fmt(chunk.end)}} (${{Number(chunk.duration_s || 0).toFixed(2)}}s)</div>
      <div>quality</div><div>${{escapeHtml(chunk.alignment_quality)}} / ${{escapeHtml(chunk.fallback_subtype || "none")}}</div>
    </div>
    <div class="text">${{escapeHtml(chunk.display_text || chunk.align_text || chunk.raw_text || "(empty)")}}</div>
  </div>`;
}}
function activeText(segments, t) {{
  return segments
    .filter(seg => Number(seg.start) <= t && t <= Number(seg.end))
    .map(seg => seg.text)
    .join("\\n");
}}
function updateCaptions() {{
  const t = video.currentTime || 0;
  document.getElementById("baselineNow").textContent = activeText(BASELINE_SEGMENTS, t);
  document.getElementById("candidateNow").textContent = activeText(CANDIDATE_SEGMENTS, t);
  const item = ITEMS[currentIndex];
  if (item) {{
    const width = Math.max(0.001, item.review_end - item.review_start);
    const pct = Math.max(0, Math.min(100, ((t - item.review_start) / width) * 100));
    cursor.style.left = `${{pct}}%`;
    document.getElementById("nowText").textContent = fmt(t);
    if (!video.paused && t >= item.review_end) video.pause();
  }}
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
      annotations[item.sample_id] = {{...(annotations[item.sample_id] || {{}}), label: value, updated_at: new Date().toISOString()}};
      saveAnnotations();
      renderLabels(item);
    }};
    root.appendChild(btn);
  }}
  notes.value = ann.notes || "";
}}
function renderCurrent() {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  document.getElementById("clipTitle").textContent = `#${{item.index + 1}} baseline chunk ${{item.baseline.chunk_index}}`;
  document.getElementById("clipMeta").textContent = `${{fmt(item.review_start)}}-${{fmt(item.review_end)}} · ${{outcomeText(item.outcome)}}`;
  document.getElementById("baselineBadge").textContent = item.baseline.alignment_quality + " / " + (item.baseline.fallback_subtype || "none");
  document.getElementById("candidateBadge").textContent = outcomeText(item.outcome);
  document.getElementById("candidateBadge").className = "badge " + badgeClass(item.outcome);
  document.getElementById("baselineChunks").innerHTML = chunkHtml(item.baseline);
  document.getElementById("candidateChunks").innerHTML = item.candidate.length ? item.candidate.map(chunkHtml).join("") : "<div class='hint'>无重叠 opt-in chunk</div>";
  document.getElementById("rangeStart").textContent = fmt(item.review_start);
  document.getElementById("rangeEnd").textContent = fmt(item.review_end);
  range.style.left = "0%";
  range.style.width = "100%";
  renderLabels(item);
  video.currentTime = item.review_start;
  updateCaptions();
}}
function playCurrent() {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  video.currentTime = item.review_start;
  video.play();
}}
function exportRows() {{
  return ITEMS.map(item => ({{
    sample_id: item.sample_id,
    video: item.video,
    review_start: item.review_start,
    review_end: item.review_end,
    baseline_chunk_index: item.baseline.chunk_index,
    outcome: item.outcome,
    candidate_chunk_indices: item.candidate.map(c => c.chunk_index),
    ...(annotations[item.sample_id] || {{}})
  }}));
}}
function downloadJsonl() {{
  const text = exportRows().map(row => JSON.stringify(row)).join("\\n") + "\\n";
  const blob = new Blob([text], {{type: "application/jsonl;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "manual_alignment_compare_labels.jsonl";
  a.click();
  URL.revokeObjectURL(a.href);
}}
notes.addEventListener("input", () => {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  annotations[item.sample_id] = {{...(annotations[item.sample_id] || {{}}), notes: notes.value, updated_at: new Date().toISOString()}};
  saveAnnotations();
}});
video.addEventListener("timeupdate", updateCaptions);
video.addEventListener("seeked", updateCaptions);
timeline.addEventListener("click", event => {{
  const item = ITEMS[currentIndex];
  if (!item) return;
  const rect = timeline.getBoundingClientRect();
  const ratio = (event.clientX - rect.left) / rect.width;
  video.currentTime = item.review_start + Math.max(0, Math.min(1, ratio)) * (item.review_end - item.review_start);
}});
document.getElementById("playBtn").onclick = playCurrent;
document.getElementById("prevBtn").onclick = () => {{
  const pos = filtered.indexOf(ITEMS[currentIndex]);
  if (pos > 0) currentIndex = ITEMS.indexOf(filtered[pos - 1]);
  renderCurrent(); renderList();
}};
document.getElementById("nextBtn").onclick = () => {{
  const pos = filtered.indexOf(ITEMS[currentIndex]);
  if (pos >= 0 && pos < filtered.length - 1) currentIndex = ITEMS.indexOf(filtered[pos + 1]);
  renderCurrent(); renderList();
}};
document.getElementById("downloadBtn").onclick = downloadJsonl;
searchInput.addEventListener("input", applyFilters);
outcomeFilter.addEventListener("change", applyFilters);
document.getElementById("summaryLine").textContent = `${{ITEMS.length}} 条 fallback 对比 · baseline fallback ${{SUMMARY.baseline_fallback_count}} → opt-in fallback ${{SUMMARY.candidate_fallback_count}}`;
renderFilters();
renderList();
renderCurrent();
</script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a video review page comparing baseline alignment fallback against opt-in island split."
    )
    parser.add_argument("--baseline-diagnostics-dir", required=True)
    parser.add_argument("--candidate-diagnostics-dir", required=True)
    parser.add_argument("--baseline-aligned", required=True)
    parser.add_argument("--candidate-aligned", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--output-dir",
        default="agents/audits/alignment-compare-review",
    )
    parser.add_argument("--title", default="SpeechBoundary-JA fallback 对比审计")
    parser.add_argument("--context-margin-s", type=float, default=1.0)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args(argv)
    if args.context_margin_s < 0:
        parser.error("--context-margin-s must be non-negative")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_dir = project_path(args.baseline_diagnostics_dir)
    candidate_dir = project_path(args.candidate_diagnostics_dir)
    baseline_rows = read_jsonl(baseline_dir / "diagnostics.jsonl")
    candidate_rows = read_jsonl(candidate_dir / "diagnostics.jsonl")
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_segments = segment_rows(project_path(args.baseline_aligned))
    candidate_segments = segment_rows(project_path(args.candidate_aligned))
    baseline_srt = output_dir / "baseline.full.srt"
    candidate_srt = output_dir / "island_split.full.srt"
    baseline_vtt = output_dir / "baseline.full.vtt"
    candidate_vtt = output_dir / "island_split.full.vtt"
    write_srt(baseline_srt, baseline_segments)
    write_srt(candidate_srt, candidate_segments)
    write_vtt(baseline_vtt, baseline_segments)
    write_vtt(candidate_vtt, candidate_segments)

    items = build_review_items(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        context_margin_s=args.context_margin_s,
        max_items=args.max_items,
    )
    write_jsonl(output_dir / "alignment_compare_review_items.jsonl", items)
    summary = {
        "dataset_id": output_dir.name,
        "baseline_diagnostics_dir": project_rel(baseline_dir),
        "candidate_diagnostics_dir": project_rel(candidate_dir),
        "baseline_fallback_count": sum(1 for row in baseline_rows if is_baseline_fallback_target(row)),
        "candidate_fallback_count": sum(1 for row in candidate_rows if is_baseline_fallback_target(row)),
        "review_item_count": len(items),
        "outcome_counts": {
            key: sum(1 for item in items if item["outcome"] == key)
            for key in sorted({item["outcome"] for item in items})
        },
        "video": project_rel(args.video),
        "html": project_rel(output_dir / "index.html"),
        "baseline_srt": project_rel(baseline_srt),
        "candidate_srt": project_rel(candidate_srt),
        "baseline_vtt": project_rel(baseline_vtt),
        "candidate_vtt": project_rel(candidate_vtt),
    }
    write_json(output_dir / "summary.json", summary)
    html_text = page_template(
        title=args.title,
        video_url=rel_url(project_path(args.video), from_dir=output_dir),
        baseline_vtt_url=rel_url(baseline_vtt, from_dir=output_dir),
        candidate_vtt_url=rel_url(candidate_vtt, from_dir=output_dir),
        items=items,
        baseline_segments=baseline_segments,
        candidate_segments=candidate_segments,
        summary=summary,
    )
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    maybe_update_audit_entrypoints(output_dir=output_dir, title=args.title, summary=summary)
    print(f"html={project_rel(output_dir / 'index.html')}")
    print(f"items={len(items)}")
    print(f"summary={project_rel(output_dir / 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
