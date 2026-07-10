#!/usr/bin/env python3
"""Generate the Pre-ASR CueQC v12 D.7 repair audit page."""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import ANON_LABELS, project_rel, update_audit_entrypoints  # noqa: E402


LABEL_SCHEMA = "pre_asr_v12_repair_manual_verdict_v1"
MANUAL_VERDICTS_FILENAME = "manual_verdicts.jsonl"
DEFAULT_SOURCE_WINDOWS = (
    "agents/temp/20260707_204500_pre-asr-chunk-reexport-full/source_windows.jsonl"
)
DEFAULT_T050_PAIRED = (
    "agents/temp/20260707_234100_pre-asr-v12-gate-t050/paired_decisions.jsonl"
)
DEFAULT_T095_PAIRED = (
    "agents/temp/20260707_234000_pre-asr-v12-gate-t095/paired_decisions.jsonl"
)
DEFAULT_LONG_FALSE_DROP_PATHS = (
    "agents/temp/20260707_234100_pre-asr-v12-gate-t040/v12_false_drops_ge_0p8s.jsonl",
    "agents/temp/20260707_234100_pre-asr-v12-gate-t045/v12_false_drops_ge_0p8s.jsonl",
    "agents/temp/20260707_234100_pre-asr-v12-gate-t050/v12_false_drops_ge_0p8s.jsonl",
)
POOL_LABELS = {
    "A1_t050_residual_false_keep": "A1 t=0.50 残余 false-keep 全审",
    "A2_t095_exclusive_false_keep_sample": "A2 t=0.95 独有 false-keep 抽审",
    "B_low_threshold_long_false_drop": "B false-drop 全审",
}
VERDICT_TO_LABEL = {
    "drop": "definite_drop",
    "keep": "definite_keep",
    "unsure": "ambiguous_ignore",
}


def _resolve_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(payload))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def row_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def json_for_script(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def candidate_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("id") or row.get("candidate_id") or row.get("sample_id") or "").strip()
    if value:
        return value
    audio_id = str(row.get("audio_id") or row.get("window_id") or "").strip()
    return f"preasr-{audio_id}-chunk{row_int(row, 'chunk_index'):05d}"


def video_id_from_window(window_id: str) -> str:
    return window_id.rsplit("-w", 1)[0] if "-w" in window_id else window_id


def direction_for(row: Mapping[str, Any]) -> str:
    truth = str(row.get("truth") or "")
    prediction = str(row.get("v12_prediction") or row.get("prediction") or "")
    if truth == "keep" and prediction == "drop":
        return "A"
    if truth == "drop" and prediction == "keep":
        return "B"
    return ""


def is_false_keep(row: Mapping[str, Any]) -> bool:
    return str(row.get("truth")) == "drop" and str(row.get("v12_prediction")) == "keep"


def is_false_drop(row: Mapping[str, Any]) -> bool:
    return str(row.get("truth")) == "keep" and str(row.get("v12_prediction")) == "drop"


def probability_bin(prob_drop: float) -> str:
    if prob_drop < 0.60:
        return "p_drop_050_060"
    if prob_drop < 0.70:
        return "p_drop_060_070"
    if prob_drop < 0.80:
        return "p_drop_070_080"
    if prob_drop < 0.90:
        return "p_drop_080_090"
    return "p_drop_090_095"


def duration_bin(duration_s: float) -> str:
    if duration_s < 0.5:
        return "dur_lt_0p5"
    if duration_s < 1.0:
        return "dur_0p5_1"
    if duration_s < 3.0:
        return "dur_1_3"
    if duration_s < 8.0:
        return "dur_3_8"
    return "dur_ge_8"


def sample_balanced_false_keep_noise(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    if limit <= 0 or len(materialized) <= limit:
        return materialized
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        groups[
            (
                probability_bin(row_float(row, "v12_prob_drop")),
                duration_bin(row_float(row, "duration_s", row_float(row, "end") - row_float(row, "start"))),
            )
        ].append(row)
    for group in groups.values():
        group.sort(
            key=lambda item: (
                str(item.get("audio_id") or ""),
                row_int(item, "chunk_index"),
                candidate_id(item),
            )
        )
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while keys and len(selected) < limit:
        next_keys: list[tuple[str, str]] = []
        for key in keys:
            bucket = groups[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    selected.sort(
        key=lambda item: (
            probability_bin(row_float(item, "v12_prob_drop")),
            duration_bin(row_float(item, "duration_s", row_float(item, "end") - row_float(item, "start"))),
            str(item.get("audio_id") or ""),
            row_int(item, "chunk_index"),
        )
    )
    return selected


def threshold_from_path(path: Path) -> str:
    match = re.search(r"t(\d{3})", path.as_posix())
    if not match:
        return ""
    raw = match.group(1)
    return f"0.{raw[-2:]}"


def select_repair_pools(
    *,
    t050_rows: Sequence[Mapping[str, Any]],
    t095_rows: Sequence[Mapping[str, Any]],
    long_false_drop_rows_by_path: Mapping[str, Sequence[Mapping[str, Any]]],
    a2_limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a1 = [dict(row) for row in t050_rows if is_false_keep(row)]
    a1_ids = {candidate_id(row) for row in a1}
    a2_population = [
        dict(row)
        for row in t095_rows
        if is_false_keep(row) and candidate_id(row) not in a1_ids
    ]
    a2 = sample_balanced_false_keep_noise(a2_population, limit=a2_limit, seed=seed)

    long_by_id: dict[str, dict[str, Any]] = {}
    for path_text, rows in long_false_drop_rows_by_path.items():
        threshold = threshold_from_path(Path(path_text))
        for row in rows:
            if not is_false_drop(row):
                continue
            key = candidate_id(row)
            item = long_by_id.setdefault(key, dict(row))
            thresholds = item.setdefault("audit_threshold_sources", [])
            if threshold and threshold not in thresholds:
                thresholds.append(threshold)
            sources = item.setdefault("audit_manifest_sources", [])
            if path_text not in sources:
                sources.append(path_text)
    b_rows = sorted(
        long_by_id.values(),
        key=lambda item: (
            str(item.get("audio_id") or ""),
            row_int(item, "chunk_index"),
            candidate_id(item),
        ),
    )

    selected: list[dict[str, Any]] = []
    selected.extend(
        annotate_pool_rows(
            a1,
            source_pool="A1_t050_residual_false_keep",
            audit_kind="false_keep",
            threshold_context="0.50",
        )
    )
    selected.extend(
        annotate_pool_rows(
            a2,
            source_pool="A2_t095_exclusive_false_keep_sample",
            audit_kind="false_keep",
            threshold_context="0.95_exclusive",
        )
    )
    selected.extend(
        annotate_pool_rows(
            b_rows,
            source_pool="B_low_threshold_long_false_drop",
            audit_kind="long_false_drop",
            threshold_context="0.40/0.45/0.50",
        )
    )
    summary = {
        "a1_t050_residual_false_keep_count": len(a1),
        "a2_t095_exclusive_false_keep_population": len(a2_population),
        "a2_t095_exclusive_false_keep_sample_count": len(a2),
        "b_low_threshold_long_false_drop_count": len(b_rows),
        "pool_counts": dict(Counter(row["source_pool"] for row in selected)),
        "review_item_count": len(selected),
    }
    return selected, summary


def annotate_pool_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_pool: str,
    audit_kind: str,
    threshold_context: str,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item.update(
            {
                "candidate_id": candidate_id(item),
                "source_pool": source_pool,
                "source_pool_label": POOL_LABELS[source_pool],
                "audit_kind": audit_kind,
                "audit_pool_index": index,
                "threshold_context": threshold_context,
                "direction": direction_for(item),
            }
        )
        annotated.append(item)
    return annotated


def load_source_windows(source_windows_jsonl: Path) -> dict[str, dict[str, Any]]:
    windows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(source_windows_jsonl):
        window_id = str(row.get("window_id") or "").strip()
        if not window_id:
            raise ValueError(f"source window without window_id: {row}")
        if window_id in windows:
            raise ValueError(f"duplicate source window: {window_id}")
        windows[window_id] = row
    return windows


def display_video_label(video_id: str) -> str:
    label = ANON_LABELS.get(video_id)
    if label:
        return label
    return video_id


def safe_filename(value: str, *, limit: int = 130) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not slug:
        slug = "item"
    return slug[:limit]


def rel_url(path: Path, *, from_dir: Path) -> str:
    try:
        return path.resolve().relative_to(from_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def cut_audio_clip(
    *,
    source_audio: Path,
    output_path: Path,
    start_s: float,
    end_s: float,
    force: bool,
) -> None:
    duration_s = max(0.05, end_s - start_s)
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, start_s):.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def enrich_rows_with_audio(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_windows: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    context_pad_s: float,
    cut_audio: bool,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clips_dir = output_dir / "clips"
    enriched: list[dict[str, Any]] = []
    clip_count = 0
    missing_audio: list[str] = []
    for page_index, row in enumerate(rows, start=1):
        item = dict(row)
        window_id = str(item.get("audio_id") or item.get("window_id") or "").strip()
        window = source_windows.get(window_id)
        if window is None:
            raise ValueError(f"candidate references unknown source window {window_id}: {candidate_id(item)}")

        video_id = str(window.get("video_id") or video_id_from_window(window_id))
        source_audio = _resolve_path(str(window.get("audio_wav") or ""))
        if not source_audio.exists():
            missing_audio.append(str(source_audio))
            if cut_audio:
                raise FileNotFoundError(f"source audio not found for {candidate_id(item)}: {source_audio}")

        start_s = row_float(item, "start")
        end_s = row_float(item, "end")
        if end_s <= start_s:
            end_s = start_s + max(0.05, row_float(item, "duration_s", 0.05))
        window_duration_s = row_float(window, "duration_s", max(end_s, 0.0))
        context_start = max(0.0, start_s - context_pad_s)
        context_end = min(max(window_duration_s, end_s), end_s + context_pad_s)

        slug = safe_filename(f"{page_index:03d}_{item['source_pool']}_{candidate_id(item)}")
        chunk_clip = clips_dir / f"{slug}_chunk.wav"
        context_clip = clips_dir / f"{slug}_context.wav"
        if cut_audio:
            cut_audio_clip(
                source_audio=source_audio,
                output_path=chunk_clip,
                start_s=start_s,
                end_s=end_s,
                force=force,
            )
            cut_audio_clip(
                source_audio=source_audio,
                output_path=context_clip,
                start_s=context_start,
                end_s=context_end,
                force=force,
            )
            clip_count += 2

        item.update(
            {
                "page_index": page_index,
                "audio_id": window_id,
                "window_id": window_id,
                "video_id": video_id,
                "video_label": display_video_label(video_id),
                "chunk_index": row_int(item, "chunk_index"),
                "start": round(start_s, 6),
                "end": round(end_s, 6),
                "duration_s": round(max(0.0, end_s - start_s), 6),
                "context_start": round(context_start, 6),
                "context_end": round(context_end, 6),
                "model_prob_drop": row_float(item, "v12_prob_drop"),
                "v12_prob_drop": row_float(item, "v12_prob_drop"),
                "v12_prediction": str(item.get("v12_prediction") or ""),
                "truth": str(item.get("truth") or ""),
                "omni_label": "definite_keep" if str(item.get("truth")) == "keep" else "definite_drop",
                "source_audio_wav": project_rel(source_audio),
                "source_video": str(window.get("source_video") or ""),
                "source_start_s": row_float(window, "source_start_s"),
                "source_end_s": row_float(window, "source_end_s"),
                "chunk_clip": rel_url(chunk_clip, from_dir=output_dir),
                "context_clip": rel_url(context_clip, from_dir=output_dir),
                "clip": rel_url(chunk_clip, from_dir=output_dir),
            }
        )
        enriched.append(item)
    summary = {
        "clip_count": clip_count,
        "cut_audio": cut_audio,
        "missing_source_audio_count": len(missing_audio),
        "missing_source_audio": missing_audio[:20],
    }
    return enriched, summary


def row_for_page(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "audit_kind",
        "audit_manifest_sources",
        "audit_pool_index",
        "audit_threshold_sources",
        "audio_id",
        "candidate_id",
        "chunk_clip",
        "chunk_index",
        "clip",
        "context_clip",
        "context_end",
        "context_start",
        "direction",
        "duration_s",
        "end",
        "legacy_prediction",
        "legacy_prob_drop",
        "legacy_reason",
        "model_prob_drop",
        "omni_label",
        "page_index",
        "source_pool",
        "source_pool_label",
        "source_start_s",
        "source_video",
        "start",
        "threshold_context",
        "truth",
        "v12_prediction",
        "v12_prob_drop",
        "v12_reason",
        "video_id",
        "video_label",
        "window_id",
    }
    return {key: row.get(key) for key in sorted(keys) if key in row}


def article_html(row: Mapping[str, Any]) -> str:
    cid = html.escape(str(row["candidate_id"]))
    pool = html.escape(str(row["source_pool"]))
    direction = html.escape(str(row.get("direction") or ""))
    prob = row_float(row, "v12_prob_drop")
    title = (
        f"{row_int(row, 'page_index')}. "
        f"{html.escape(str(row.get('source_pool_label') or row.get('source_pool')))}"
    )
    if row.get("audit_kind") == "false_keep":
        task_hint = "Omni/merged truth=drop，但 v12 保留；请判断这段最终应该 drop、keep，还是不确定。"
    else:
        task_hint = "merged truth=keep，但 v12 在低阈值会删除；请判断这段最终应该 drop、keep，还是不确定。"
    threshold_sources = row.get("audit_threshold_sources")
    threshold_text = ""
    if isinstance(threshold_sources, list) and threshold_sources:
        threshold_text = f" ｜ thresholds {html.escape(', '.join(str(value) for value in threshold_sources))}"
    return f"""
<article class="audit-item" data-cid="{cid}" data-pool="{pool}" data-direction="{direction}">
  <h3>{title}</h3>
  <p><code>{cid}</code></p>
  <p>
    <b>{html.escape(str(row.get("video_label") or row.get("video_id") or ""))}</b>
    ｜ window <code>{html.escape(str(row.get("window_id") or ""))}</code>
    ｜ chunk {row_int(row, "chunk_index")}
    ｜ {row_float(row, "start"):.3f}-{row_float(row, "end"):.3f}s
    ｜ duration {row_float(row, "duration_s"):.3f}s
  </p>
  <p>
    truth <b>{html.escape(str(row.get("truth") or ""))}</b>
    ｜ v12 <b>{html.escape(str(row.get("v12_prediction") or ""))}</b>
    ｜ p_drop <b>{prob:.6f}</b>
    ｜ direction <b>{direction}</b>{threshold_text}
  </p>
  <p class="hint">{html.escape(task_hint)}</p>
  <div class="audio-grid">
    <div>
      <div class="audio-label">chunk</div>
      <audio controls preload="none" src="{html.escape(str(row.get("chunk_clip") or ""))}"></audio>
    </div>
    <div>
      <div class="audio-label">context</div>
      <audio controls preload="none" src="{html.escape(str(row.get("context_clip") or ""))}"></audio>
    </div>
  </div>
  <div class="verdict">
    <button type="button" data-v="drop">drop</button>
    <button type="button" data-v="keep">keep</button>
    <button type="button" data-v="unsure">不确定</button>
    <input type="text" class="note" placeholder="备注（可选）">
  </div>
</article>"""


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<style>
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #f6f7f4;
  color: #1d2421;
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
}
main { max-width: 1180px; margin: 0 auto; padding: 18px 14px 40px; }
h1 { margin: 0 0 8px; font-size: 22px; }
h2 { margin: 28px 0 8px; font-size: 18px; }
h3 { margin: 0 0 8px; font-size: 15px; }
code { background: #eef1ee; padding: 2px 4px; border-radius: 4px; overflow-wrap: anywhere; }
button, input, select { font: inherit; }
button {
  border: 1px solid #b8c2bd;
  border-radius: 6px;
  background: #fff;
  color: #1d2421;
  cursor: pointer;
  padding: 7px 12px;
}
button.primary, .verdict button.active[data-v="keep"] {
  background: #0f766e;
  border-color: #0f766e;
  color: #fff;
}
.verdict button.active[data-v="drop"] {
  background: #b42318;
  border-color: #b42318;
  color: #fff;
}
.verdict button.active[data-v="unsure"] {
  background: #946800;
  border-color: #946800;
  color: #fff;
}
.toolbar {
  position: sticky;
  top: 0;
  z-index: 2;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  align-items: center;
  margin-bottom: 14px;
  border: 1px solid #d8ddd8;
  border-radius: 8px;
  background: #fff;
  padding: 10px 12px;
}
.toolbar-actions, .filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.filters select {
  border: 1px solid #b8c2bd;
  border-radius: 6px;
  background: #fff;
  padding: 7px 8px;
  min-width: 180px;
}
.progress { color: #52605a; }
.hint { color: #66706c; }
.pool-summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 12px 0;
}
.pool-card {
  border: 1px solid #d8ddd8;
  border-radius: 8px;
  background: #fff;
  padding: 10px 12px;
}
.pool-card strong { display: block; margin-bottom: 2px; }
.audit-item {
  border: 1px solid #d8ddd8;
  border-radius: 8px;
  background: #fff;
  margin: 12px 0;
  padding: 12px 14px;
}
.audit-item.done { border-color: #0f766e; background: #effaf7; }
.audit-item.hidden { display: none; }
.audio-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 10px;
  margin: 8px 0 10px;
}
audio { width: 100%; }
.audio-label { color: #52605a; font-weight: 650; margin-bottom: 4px; }
.verdict { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.note {
  min-width: 260px;
  flex: 1 1 280px;
  border: 1px solid #b8c2bd;
  border-radius: 6px;
  padding: 7px 8px;
}
.status { color: #52605a; margin-top: 8px; white-space: pre-wrap; }
@media (max-width: 760px) {
  .toolbar { grid-template-columns: 1fr; }
  .pool-summary, .audio-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<main>
  <h1>%%TITLE%%</h1>
  <p class="hint">目标：先修标签口径，再重算 v12/iter1 在修正标签下的 gate 上界。人工裁决只使用 drop / keep / 不确定，不恢复规则 fallback。</p>
  <div class="toolbar">
    <div>
      <div class="progress" id="progress"></div>
      <div class="status" id="saveStatus"></div>
    </div>
    <div class="toolbar-actions">
      <div class="filters">
        <select id="poolFilter">
          <option value="">全部审计池</option>
%%POOL_OPTIONS%%
        </select>
        <select id="stateFilter">
          <option value="">全部状态</option>
          <option value="todo">未审</option>
          <option value="drop">drop</option>
          <option value="keep">keep</option>
          <option value="unsure">不确定</option>
        </select>
      </div>
      <button type="button" id="copyBtn">复制 JSONL</button>
      <button type="button" id="downloadBtn">下载 JSONL</button>
      <button type="button" id="saveBtn" class="primary">保存审计结果</button>
    </div>
  </div>
  <section class="pool-summary">
%%POOL_CARDS%%
  </section>
%%ARTICLES%%
</main>
<script type="application/json" id="rows-json">%%ROWS_JSON%%</script>
<script>
const ROWS = JSON.parse(document.getElementById("rows-json").textContent);
const LABEL_SCHEMA = "%%LABEL_SCHEMA%%";
const OUTPUT_NAME = "%%OUTPUT_NAME%%";
const STORAGE_KEY = `pre-asr-v12-repair-audit:${location.pathname}`;
const VERDICT_TO_LABEL = {drop: "definite_drop", keep: "definite_keep", unsure: "ambiguous_ignore"};
let annotations = loadAnnotations();

function loadAnnotations() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (_) { return {}; }
}
function saveAnnotations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  refreshUi();
}
function rowByCandidate(candidateId) {
  return ROWS.find(row => row.candidate_id === candidateId) || {};
}
function annotation(candidateId) {
  if (!annotations[candidateId]) annotations[candidateId] = {verdict: "", note: "", updated_at: ""};
  return annotations[candidateId];
}
function setVerdict(article, verdict) {
  const cid = article.dataset.cid;
  const ann = annotation(cid);
  ann.verdict = verdict;
  ann.note = article.querySelector(".note").value || "";
  ann.updated_at = new Date().toISOString();
  saveAnnotations();
}
function refreshArticle(article) {
  const cid = article.dataset.cid;
  const ann = annotations[cid] || {};
  article.classList.toggle("done", Boolean(ann.verdict));
  for (const button of article.querySelectorAll("[data-v]")) {
    button.classList.toggle("active", button.dataset.v === ann.verdict);
  }
  const note = article.querySelector(".note");
  if (document.activeElement !== note) note.value = ann.note || "";
}
function stateFor(row) {
  return (annotations[row.candidate_id] || {}).verdict || "todo";
}
function refreshVisibility() {
  const pool = document.getElementById("poolFilter").value;
  const state = document.getElementById("stateFilter").value;
  for (const article of document.querySelectorAll(".audit-item")) {
    const row = rowByCandidate(article.dataset.cid);
    const visible = (!pool || row.source_pool === pool) && (!state || stateFor(row) === state);
    article.classList.toggle("hidden", !visible);
  }
}
function refreshProgress() {
  const counts = ROWS.reduce((acc, row) => {
    const state = stateFor(row);
    acc[state] = (acc[state] || 0) + 1;
    return acc;
  }, {});
  document.getElementById("progress").textContent =
    `${counts.drop || 0} drop · ${counts.keep || 0} keep · ${counts.unsure || 0} 不确定 · ${counts.todo || 0} 未审 / ${ROWS.length}`;
}
function refreshUi() {
  for (const article of document.querySelectorAll(".audit-item")) refreshArticle(article);
  refreshVisibility();
  refreshProgress();
}
function exportRows() {
  return ROWS
    .filter(row => (annotations[row.candidate_id] || {}).verdict)
    .map(row => {
      const ann = annotations[row.candidate_id] || {};
      const verdict = ann.verdict || "";
      return {
        schema: LABEL_SCHEMA,
        candidate_id: row.candidate_id,
        sample_id: row.candidate_id,
        verdict,
        manual_label: verdict,
        label: VERDICT_TO_LABEL[verdict] || "",
        note: ann.note || "",
        updated_at: ann.updated_at || "",
        direction: row.direction || "",
        source_pool: row.source_pool || "",
        audit_kind: row.audit_kind || "",
        window_id: row.window_id || row.audio_id || "",
        audio_id: row.audio_id || row.window_id || "",
        video_id: row.video_id || "",
        video_label: row.video_label || "",
        chunk_index: row.chunk_index,
        start: row.start,
        end: row.end,
        duration_s: row.duration_s,
        truth: row.truth,
        omni_label: row.omni_label,
        v12_prediction: row.v12_prediction,
        model_prob_drop: row.model_prob_drop,
        v12_prob_drop: row.v12_prob_drop,
        threshold_context: row.threshold_context || "",
        audit_threshold_sources: row.audit_threshold_sources || [],
        clip: row.chunk_clip || row.clip || "",
        context_clip: row.context_clip || ""
      };
    });
}
function exportJsonl() {
  const rows = exportRows();
  return rows.map(row => JSON.stringify(row)).join("\\n") + (rows.length ? "\\n" : "");
}
function downloadJsonl() {
  const blob = new Blob([exportJsonl()], {type: "application/jsonl;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = OUTPUT_NAME;
  a.click();
  URL.revokeObjectURL(a.href);
}
async function copyJsonl() {
  await navigator.clipboard.writeText(exportJsonl());
  document.getElementById("saveStatus").textContent = "已复制 JSONL。";
}
async function saveJsonl() {
  const content = exportJsonl();
  const response = await fetch("/__audit_api__/save-labels", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({href: location.pathname, filename: OUTPUT_NAME, content})
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  document.getElementById("saveStatus").textContent = `已保存到 ${payload.path}`;
}
for (const article of document.querySelectorAll(".audit-item")) {
  for (const button of article.querySelectorAll("[data-v]")) {
    button.addEventListener("click", () => setVerdict(article, button.dataset.v));
  }
  article.querySelector(".note").addEventListener("input", event => {
    const ann = annotation(article.dataset.cid);
    ann.note = event.target.value || "";
    ann.updated_at = new Date().toISOString();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  });
}
document.getElementById("poolFilter").addEventListener("change", refreshVisibility);
document.getElementById("stateFilter").addEventListener("change", refreshVisibility);
document.getElementById("downloadBtn").addEventListener("click", downloadJsonl);
document.getElementById("copyBtn").addEventListener("click", () => copyJsonl().catch(error => {
  document.getElementById("saveStatus").textContent = `复制失败：${error.message || error}`;
}));
document.getElementById("saveBtn").addEventListener("click", () => saveJsonl().catch(error => {
  document.getElementById("saveStatus").textContent = `保存失败，已改为下载：${error.message || error}`;
  downloadJsonl();
}));
refreshUi();
</script>
</body>
</html>
"""


def page_html(*, title: str, rows: Sequence[Mapping[str, Any]]) -> str:
    pool_counts = Counter(str(row.get("source_pool") or "") for row in rows)
    pool_options = "\n".join(
        f'          <option value="{html.escape(pool)}">{html.escape(POOL_LABELS.get(pool, pool))}</option>'
        for pool in POOL_LABELS
        if pool_counts.get(pool, 0)
    )
    pool_cards = "\n".join(
        f'    <div class="pool-card"><strong>{html.escape(POOL_LABELS.get(pool, pool))}</strong><span>{pool_counts.get(pool, 0)} 条</span></div>'
        for pool in POOL_LABELS
        if pool_counts.get(pool, 0)
    )
    articles = "\n".join(article_html(row) for row in rows)
    page_rows = [row_for_page(row) for row in rows]
    return (
        HTML_TEMPLATE.replace("%%TITLE%%", html.escape(title))
        .replace("%%POOL_OPTIONS%%", pool_options)
        .replace("%%POOL_CARDS%%", pool_cards)
        .replace("%%ARTICLES%%", articles)
        .replace("%%ROWS_JSON%%", json_for_script(page_rows))
        .replace("%%LABEL_SCHEMA%%", LABEL_SCHEMA)
        .replace("%%OUTPUT_NAME%%", MANUAL_VERDICTS_FILENAME)
    )


def build_audit(
    *,
    source_windows_jsonl: Path,
    t050_paired_jsonl: Path,
    t095_paired_jsonl: Path,
    long_false_drop_jsonls: Sequence[Path],
    output_dir: Path,
    title: str = "Pre-ASR CueQC v12 repair audit",
    a2_limit: int = 100,
    seed: int = 20260708,
    context_pad_s: float = 1.5,
    cut_audio: bool = True,
    force: bool = False,
    refresh_nav: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    t050_rows = read_jsonl(t050_paired_jsonl)
    t095_rows = read_jsonl(t095_paired_jsonl)
    long_rows_by_path = {str(path): read_jsonl(path) for path in long_false_drop_jsonls}
    selected, pool_summary = select_repair_pools(
        t050_rows=t050_rows,
        t095_rows=t095_rows,
        long_false_drop_rows_by_path=long_rows_by_path,
        a2_limit=a2_limit,
        seed=seed,
    )
    source_windows = load_source_windows(source_windows_jsonl)
    enriched, audio_summary = enrich_rows_with_audio(
        rows=selected,
        source_windows=source_windows,
        output_dir=output_dir,
        context_pad_s=context_pad_s,
        cut_audio=cut_audio,
        force=force,
    )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row_for_page(row), ensure_ascii=False, sort_keys=True) + "\n")

    index_path = output_dir / "index.html"
    index_path.write_text(page_html(title=title, rows=enriched), encoding="utf-8")

    summary = {
        "schema": "pre_asr_v12_repair_audit_summary_v1",
        "title": title,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label_schema": LABEL_SCHEMA,
        "manual_verdicts_filename": MANUAL_VERDICTS_FILENAME,
        "manifest": project_rel(manifest_path),
        "index": project_rel(index_path),
        "source_windows": project_rel(source_windows_jsonl),
        "t050_paired": project_rel(t050_paired_jsonl),
        "t095_paired": project_rel(t095_paired_jsonl),
        "long_false_drop_jsonls": [project_rel(path) for path in long_false_drop_jsonls],
        "a2_limit": a2_limit,
        "seed": seed,
        "context_pad_s": context_pad_s,
        "pool_labels": POOL_LABELS,
        "upstream_freeze": {
            "qwen_1_7b_proposer": "promoted proposer v1",
            "semantic_split": "promoted Split v2",
            "policy": "frozen during D7 repair; no upstream retrain/replacement/operating-point change",
        },
        "oom_discipline": {
            "gpu_oom_definition": "physical dedicated VRAM * 0.95; shared VRAM is not available budget",
            "ram_oom_definition": "physical RAM * 0.95",
        },
        **pool_summary,
        **audio_summary,
    }
    write_json(output_dir / "summary.json", summary)
    if refresh_nav:
        update_audit_entrypoints(latest_html=index_path, title=title)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-windows", default=DEFAULT_SOURCE_WINDOWS)
    parser.add_argument("--t050-paired", default=DEFAULT_T050_PAIRED)
    parser.add_argument("--t095-paired", default=DEFAULT_T095_PAIRED)
    parser.add_argument(
        "--long-false-drop",
        action="append",
        default=[],
        help="May be repeated. Defaults to t040/t045/t050 long false-drop manifests.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--title", default="Pre-ASR CueQC v12 repair audit")
    parser.add_argument("--a2-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--context-pad-s", type=float, default=1.5)
    parser.add_argument("--skip-audio-cut", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-nav", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir:
        output_dir = _resolve_path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "agents" / "audits" / f"{stamp}_pre-asr-v12-repair-audit"
    long_paths = args.long_false_drop or list(DEFAULT_LONG_FALSE_DROP_PATHS)
    summary = build_audit(
        source_windows_jsonl=_resolve_path(args.source_windows),
        t050_paired_jsonl=_resolve_path(args.t050_paired),
        t095_paired_jsonl=_resolve_path(args.t095_paired),
        long_false_drop_jsonls=[_resolve_path(path) for path in long_paths],
        output_dir=output_dir,
        title=args.title,
        a2_limit=args.a2_limit,
        seed=args.seed,
        context_pad_s=args.context_pad_s,
        cut_audio=not args.skip_audio_cut,
        force=args.force,
        refresh_nav=args.refresh_nav,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
