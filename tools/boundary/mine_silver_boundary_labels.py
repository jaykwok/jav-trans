#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.asr.diagnostics.diagnose_asr_alignment import (  # noqa: E402
    diagnose_case,
    discover_aligned_jsons,
)
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


SILVER_LABEL_SCHEMA = "speech_boundary_silver_display_v1"
HARD_CASE_SCHEMA = "speech_boundary_hard_case_v1"


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def aligned_path_to_segments(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return []
    return [row for row in payload.get("segments") or [] if isinstance(row, dict)]


def words_by_chunk(segments: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        for word in segment.get("words") or []:
            if not isinstance(word, dict):
                continue
            try:
                chunk_index = int(word.get("source_chunk_index", segment.get("source_chunk_index")))
                start = float(word.get("start"))
                end = float(word.get("end"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            grouped[chunk_index].append(
                {
                    "start": start,
                    "end": end,
                    "word": str(word.get("word") or word.get("text") or ""),
                }
            )
    for values in grouped.values():
        values.sort(key=lambda item: (item["start"], item["end"]))
    return dict(grouped)


def _numeric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return default


def _is_silver_candidate(row: dict[str, Any], words: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reject: list[str] = []
    if str(row.get("alignment_quality") or "") != "forced":
        reject.append("alignment_not_forced")
    if str(row.get("fallback_type") or "none") != "none":
        reject.append("fallback_active")
    if str(row.get("fallback_subtype") or "none") != "none":
        reject.append("fallback_subtype_not_none")
    if row.get("nonlexical_text"):
        reject.append("nonlexical_text")
    if row.get("align_text_empty"):
        reject.append("align_text_empty")
    if row.get("asr_review_uncertain"):
        reject.append("asr_review_uncertain")
    if str(row.get("asr_qc_severity") or "") not in {"", "ok"}:
        reject.append("asr_qc_not_ok")
    if not str(row.get("analysis_text") or row.get("text") or "").strip():
        reject.append("empty_text")
    if not words:
        reject.append("no_positive_duration_words")
    density_level = str(row.get("text_density_level") or "")
    if density_level in {
        "empty_or_punctuation",
        "short_vocalization_candidate",
        "repeated_vocalization_candidate",
    }:
        reject.append(f"text_density_{density_level}")
    if row.get("word_timing_failure_reasons"):
        reject.append("word_timing_failure")
    return not reject, reject


def _hard_case_reasons(row: dict[str, Any], reject_reasons: list[str]) -> list[str]:
    reasons = list(row.get("failure_reasons") or [])
    quality = str(row.get("alignment_quality") or "")
    fallback_subtype = str(row.get("fallback_subtype") or "")
    if quality and quality != "forced":
        reasons.append(f"alignment_quality_{quality}")
    if fallback_subtype and fallback_subtype != "none":
        reasons.append(f"fallback_{fallback_subtype}")
    reasons.extend(reject_reasons)
    if _numeric(row, "duration_s") >= 5.0:
        reasons.append("long_chunk")
    return list(dict.fromkeys(reason for reason in reasons if reason))


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "max": None}
    values = sorted(values)

    def percentile(p: float) -> float:
        if len(values) == 1:
            return values[0]
        idx = (len(values) - 1) * p
        lo = int(idx)
        hi = min(len(values) - 1, lo + 1)
        frac = idx - lo
        return values[lo] * (1.0 - frac) + values[hi] * frac

    return {
        "p50": round(percentile(0.50), 6),
        "p90": round(percentile(0.90), 6),
        "p95": round(percentile(0.95), 6),
        "max": round(values[-1], 6),
    }


def mine_case(aligned_path: Path, workflow_root: Path | None) -> tuple[list[dict], list[dict], dict]:
    diagnostic_rows, case_summary = diagnose_case(
        aligned_path=aligned_path,
        workflow_root=workflow_root,
    )
    segments = aligned_path_to_segments(aligned_path)
    grouped_words = words_by_chunk(segments)
    silver_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    start_errors: list[float] = []
    end_errors: list[float] = []

    for row in diagnostic_rows:
        try:
            chunk_index = int(row.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        words = grouped_words.get(chunk_index, [])
        is_silver, reject_reasons = _is_silver_candidate(row, words)
        if is_silver:
            first_word = words[0]
            last_word = words[-1]
            speech_core_start = _numeric(row, "fallback_window_start")
            speech_core_end = _numeric(row, "fallback_window_end")
            start_label = float(first_word["start"])
            end_label = float(last_word["end"])
            start_error = speech_core_start - start_label
            end_error = speech_core_end - end_label
            start_errors.append(start_error)
            end_errors.append(end_error)
            silver_rows.append(
                {
                    "schema": SILVER_LABEL_SCHEMA,
                    "case_label": row.get("case_label", ""),
                    "video": row.get("video", ""),
                    "aligned_path": row.get("aligned_path", ""),
                    "source_audio_path": row.get("source_audio_path", ""),
                    "chunk_index": chunk_index,
                    "start": row.get("start"),
                    "end": row.get("end"),
                    "duration_s": row.get("duration_s"),
                    "text": row.get("analysis_text") or row.get("text") or "",
                    "display_start_label": round(start_label, 6),
                    "display_end_label": round(end_label, 6),
                    "speech_core_start": round(speech_core_start, 6),
                    "speech_core_end": round(speech_core_end, 6),
                    "start_error_s": round(start_error, 6),
                    "end_error_s": round(end_error, 6),
                    "word_count": len(words),
                    "first_word": first_word,
                    "last_word": last_word,
                    "alignment_quality": row.get("alignment_quality", ""),
                    "fallback_subtype": row.get("fallback_subtype", ""),
                    "asr_qc_severity": row.get("asr_qc_severity", ""),
                    "text_density_level": row.get("text_density_level", ""),
                    "label_policy": {
                        "start_weight": 1.0,
                        "end_weight": 0.35,
                        "end_may_be_shorter": True,
                    },
                }
            )
        else:
            reasons = _hard_case_reasons(row, reject_reasons)
            if reasons:
                hard_rows.append(
                    {
                        "schema": HARD_CASE_SCHEMA,
                        "case_label": row.get("case_label", ""),
                        "video": row.get("video", ""),
                        "aligned_path": row.get("aligned_path", ""),
                        "chunk_index": chunk_index,
                        "start": row.get("start"),
                        "end": row.get("end"),
                        "duration_s": row.get("duration_s"),
                        "text": row.get("analysis_text") or row.get("text") or "",
                        "hard_case_reasons": reasons,
                        "alignment_quality": row.get("alignment_quality", ""),
                        "fallback_type": row.get("fallback_type", ""),
                        "fallback_subtype": row.get("fallback_subtype", ""),
                        "asr_qc_severity": row.get("asr_qc_severity", ""),
                        "asr_qc_reasons": row.get("asr_qc_reasons", []),
                        "text_density_level": row.get("text_density_level", ""),
                        "fallback_window_start": row.get("fallback_window_start"),
                        "fallback_window_end": row.get("fallback_window_end"),
                        "fallback_duration_s": row.get("fallback_duration_s"),
                    }
                )

    summary = {
        **case_summary,
        "silver_count": len(silver_rows),
        "hard_case_count": len(hard_rows),
        "silver_start_error_s": _percentiles(start_errors),
        "silver_end_error_s": _percentiles(end_errors),
        "hard_case_reason_counts": dict(
            Counter(reason for item in hard_rows for reason in item["hard_case_reasons"]).most_common()
        ),
    }
    return silver_rows, hard_rows, summary


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Silver Boundary Labels",
        "",
        f"- cases: {summary['case_count']}",
        f"- chunks: {summary['chunk_count']}",
        f"- silver labels: {summary['silver_count']}",
        f"- hard cases: {summary['hard_case_count']}",
        "",
        "## Error Distribution",
        "",
        f"- start error: `{summary['silver_start_error_s']}`",
        f"- end error: `{summary['silver_end_error_s']}`",
        "",
        "## Alignment Quality",
        "",
    ]
    for key, count in (summary.get("alignment_quality_counts") or {}).items():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Hard Case Reasons", ""])
    for key, count in (summary.get("hard_case_reason_counts") or {}).items():
        lines.append(f"- `{key}`: {count}")
    return "\n".join(lines) + "\n"


def _preview_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return rows[: max(0, limit)]


def _write_audit_html(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    silver_rows: list[dict[str, Any]],
    hard_rows: list[dict[str, Any]],
    audit_output_dir: Path | None,
    update_nav: bool,
) -> str:
    if audit_output_dir is None:
        return ""
    audit_output_dir.mkdir(parents=True, exist_ok=True)
    preview = {
        "summary": summary,
        "silver": _preview_rows(
            sorted(
                silver_rows,
                key=lambda row: abs(float(row.get("start_error_s") or 0.0)),
                reverse=True,
            ),
            limit=120,
        ),
        "hard": _preview_rows(hard_rows, limit=160),
        "paths": {
            "summary": project_rel(output_dir / "summary.md"),
            "silver": project_rel(output_dir / "silver_boundary_labels.jsonl"),
            "hard": project_rel(output_dir / "hard_cases.jsonl"),
        },
    }
    json_payload = json.dumps(preview, ensure_ascii=False).replace("</", "<\\/")
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silver Boundary Labels</title>
<style>
body {{ margin: 0; background: #f6f7f5; color: #18211d; font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px 18px 40px; }}
h1 {{ margin: 0 0 12px; font-size: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 14px 0 18px; }}
.metric, .row {{ background: #fff; border: 1px solid #dce2dd; border-radius: 8px; padding: 10px 12px; }}
.metric b {{ display: block; font-size: 20px; }}
.tabs {{ display: flex; gap: 8px; margin: 18px 0 10px; }}
button {{ border: 1px solid #b7c2bc; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
button.active {{ border-color: #0f766e; background: #e9f6f4; color: #0f766e; }}
.row {{ margin: 8px 0; }}
.title {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; }}
.badge {{ border: 1px solid #c8d0cb; border-radius: 999px; padding: 1px 7px; color: #4d5a54; font-size: 12px; }}
.forced {{ border-color: #0f766e; color: #0f766e; }}
.hard {{ border-color: #b42318; color: #b42318; }}
.text {{ margin: 6px 0; font-size: 15px; }}
.meta {{ color: #607069; font-size: 12px; overflow-wrap: anywhere; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
@media (max-width: 760px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
</style>
</head>
<body>
<main>
  <h1>Silver Boundary Labels</h1>
  <div class="meta">高置信 forced word timing 用作真实域 display-boundary silver label；hard cases 用于后续审计和 RL/DPO 负样本。</div>
  <div class="grid" id="metrics"></div>
  <div class="meta">产物：<code id="paths"></code></div>
  <div class="tabs">
    <button type="button" data-tab="silver" class="active">Silver start drift</button>
    <button type="button" data-tab="hard">Hard cases</button>
  </div>
  <div id="rows"></div>
</main>
<script id="payload" type="application/json">{json_payload}</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const metrics = document.getElementById("metrics");
const rows = document.getElementById("rows");
const paths = document.getElementById("paths");
paths.textContent = `${{DATA.paths.summary}} · ${{DATA.paths.silver}} · ${{DATA.paths.hard}}`;
function fmt(value) {{
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return value.toFixed(3);
  return String(value);
}}
function renderMetrics() {{
  const s = DATA.summary;
  const items = [
    ["cases", s.case_count],
    ["chunks", s.chunk_count],
    ["silver", s.silver_count],
    ["hard", s.hard_case_count],
    ["start p90", s.silver_start_error_s?.p90],
    ["start p95", s.silver_start_error_s?.p95],
    ["end p90", s.silver_end_error_s?.p90],
    ["end p95", s.silver_end_error_s?.p95],
  ];
  metrics.innerHTML = items.map(([label, value]) => `<div class="metric"><span>${{label}}</span><b>${{fmt(value)}}</b></div>`).join("");
}}
function renderSilver() {{
  rows.innerHTML = DATA.silver.map((item, idx) => `
    <div class="row">
      <div class="title"><b>#${{idx + 1}} chunk ${{item.chunk_index}}</b><span class="badge forced">forced</span><span class="badge">start err ${{fmt(item.start_error_s)}}s</span><span class="badge">end err ${{fmt(item.end_error_s)}}s</span></div>
      <div class="text">${{escapeHtml(item.text || "")}}</div>
      <div class="meta">${{item.video}} · ${{fmt(item.display_start_label)}}-${{fmt(item.display_end_label)}} · speech_core ${{fmt(item.speech_core_start)}}-${{fmt(item.speech_core_end)}} · words=${{item.word_count}}</div>
    </div>`).join("");
}}
function renderHard() {{
  rows.innerHTML = DATA.hard.map((item, idx) => `
    <div class="row">
      <div class="title"><b>#${{idx + 1}} chunk ${{item.chunk_index}}</b><span class="badge hard">${{escapeHtml(item.alignment_quality || "")}}</span><span class="badge">${{escapeHtml(item.fallback_subtype || "none")}}</span></div>
      <div class="text">${{escapeHtml(item.text || "")}}</div>
      <div class="meta">${{item.video}} · ${{fmt(item.start)}}-${{fmt(item.end)}} · ${{escapeHtml((item.hard_case_reasons || []).join(", "))}}</div>
    </div>`).join("");
}}
function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
for (const button of document.querySelectorAll("button[data-tab]")) {{
  button.addEventListener("click", () => {{
    document.querySelectorAll("button[data-tab]").forEach(item => item.classList.remove("active"));
    button.classList.add("active");
    button.dataset.tab === "hard" ? renderHard() : renderSilver();
  }});
}}
renderMetrics();
renderSilver();
</script>
</body>
</html>
"""
    index_path = audit_output_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    write_json(audit_output_dir / "summary.json", summary)
    if update_nav:
        update_audit_entrypoints(
            latest_html=index_path,
            title="Silver Boundary Labels 审计",
        )
    return project_rel(index_path)


def run(
    *,
    workflow_root: Path | None,
    aligned_jsons: list[Path],
    output_dir: Path,
    audit_output_dir: Path | None = None,
    update_nav: bool = False,
) -> dict[str, Any]:
    paths = discover_aligned_jsons(workflow_root, aligned_jsons)
    all_silver: list[dict[str, Any]] = []
    all_hard: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for path in paths:
        silver_rows, hard_rows, case_summary = mine_case(path, workflow_root)
        all_silver.extend(silver_rows)
        all_hard.extend(hard_rows)
        case_summaries.append(case_summary)

    start_errors = [float(row["start_error_s"]) for row in all_silver]
    end_errors = [float(row["end_error_s"]) for row in all_silver]
    summary = {
        "schema": "speech_boundary_silver_mining_summary_v1",
        "workflow_root": project_rel(workflow_root),
        "case_count": len(case_summaries),
        "chunk_count": sum(int(item.get("chunk_count") or 0) for item in case_summaries),
        "silver_count": len(all_silver),
        "hard_case_count": len(all_hard),
        "silver_start_error_s": _percentiles(start_errors),
        "silver_end_error_s": _percentiles(end_errors),
        "alignment_quality_counts": dict(
            Counter(
                key
                for case in case_summaries
                for key, count in (case.get("alignment_quality_counts") or {}).items()
                for _ in range(int(count))
            ).most_common()
        ),
        "fallback_subtype_counts": dict(
            Counter(
                key
                for case in case_summaries
                for key, count in (case.get("fallback_subtype_counts") or {}).items()
                for _ in range(int(count))
            ).most_common()
        ),
        "hard_case_reason_counts": dict(
            Counter(reason for item in all_hard for reason in item["hard_case_reasons"]).most_common()
        ),
        "cases": case_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "silver_boundary_labels.jsonl", all_silver)
    write_jsonl(output_dir / "hard_cases.jsonl", all_hard)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    audit_path = _write_audit_html(
        output_dir,
        summary=summary,
        silver_rows=all_silver,
        hard_rows=all_hard,
        audit_output_dir=audit_output_dir,
        update_nav=update_nav,
    )
    if audit_path:
        summary["audit_html"] = audit_path
        write_json(output_dir / "summary.json", summary)
        write_json(audit_output_dir / "summary.json", summary)  # type: ignore[arg-type]
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine high-confidence real-domain silver labels from workflow artifacts."
    )
    parser.add_argument("--workflow-root", default="")
    parser.add_argument("--aligned-json", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "silver-boundary-labels-v1"),
    )
    parser.add_argument("--audit-output-dir", default="")
    parser.add_argument("--update-audit-nav", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workflow_root = project_path(args.workflow_root) if args.workflow_root else None
    aligned_jsons = [project_path(path) for path in args.aligned_json]
    summary = run(
        workflow_root=workflow_root,
        aligned_jsons=aligned_jsons,
        output_dir=project_path(args.output_dir),
        audit_output_dir=project_path(args.audit_output_dir)
        if args.audit_output_dir
        else None,
        update_nav=bool(args.update_audit_nav),
    )
    print(
        "silver={silver_count} hard_cases={hard_case_count} cases={case_count}".format(
            **summary
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
