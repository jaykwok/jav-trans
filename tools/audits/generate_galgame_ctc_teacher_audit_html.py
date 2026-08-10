#!/usr/bin/env python3
"""Build a deterministic listening audit for Galgame Grok CTC supervision."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.run_grok_ctc_teacher import _compile_canonical_text_crops  # noqa: E402


STATUS_LABELS = {
    "accepted": "接受：可用时间裁剪",
    "teacher_acoustic_empty": "拒绝：Grok 空白",
    "cer_above_maximum": "拒绝：文本差异过大",
    "teacher_match_below_minimum": "拒绝：教师匹配率不足",
    "canonical_match_below_minimum": "拒绝：原文覆盖率不足",
    "no_eligible_island": "拒绝：没有可靠连续区间",
    "canonical_acoustic_empty": "拒绝：原文无可发音字符",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_number(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _spread(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Take evenly spaced cases after sorting by CER, duration, then hash."""
    if limit <= 0 or len(rows) <= limit:
        return rows
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get("cer") or -1.0),
            float(row["duration_s"]),
            _stable_number(str(row["source_id"])),
        ),
    )
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indexes = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indexes]


def _audit_row(result: dict[str, Any]) -> dict[str, Any]:
    crops, report = _compile_canonical_text_crops(
        result,
        context_s=0.25,
        maximum_cer=0.30,
        minimum_teacher_match_share=0.70,
        minimum_canonical_match_share=0.70,
        minimum_island_match_share=0.60,
        minimum_crop_chars=2,
        minimum_crop_s=0.50,
        max_crops_per_source=1,
        canonical_merge_gap_s=0.35,
    )
    response = result.get("response") or {}
    return {
        "source_id": str(result["source_id"]),
        "audio": str(result["audio"]),
        "duration_s": float(result["source_duration_s"]),
        "partition": str(result.get("partition") or ""),
        "canonical_text": str(result.get("canonical_text") or ""),
        "teacher_text": str(response.get("transcript") or ""),
        "words": list(response.get("words") or []),
        "status": str(report.get("reason") or "unknown"),
        "cer": report.get("cer"),
        "teacher_match_share": report.get("teacher_match_share"),
        "canonical_match_share": report.get("canonical_match_share"),
        "crop": crops[0] if crops else None,
    }


def select_audit_rows(
    results: list[dict[str, Any]],
    *,
    accepted: int,
    empty: int,
    high_cer: int,
    no_island: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = [_audit_row(result) for result in results]
    by_status: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_status.setdefault(str(row["status"]), []).append(row)
    selected = (
        _spread(by_status.get("accepted", []), accepted)
        + _spread(by_status.get("teacher_acoustic_empty", []), empty)
        + _spread(by_status.get("cer_above_maximum", []), high_cer)
        + _spread(by_status.get("no_eligible_island", []), no_island)
    )
    selected.sort(key=lambda row: (str(row["status"]), str(row["source_id"])))
    return selected, Counter(str(row["status"]) for row in rows)


def _metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _timeline(row: dict[str, Any]) -> str:
    duration = max(0.001, float(row["duration_s"]))
    spans: list[str] = []
    for word in row["words"]:
        start = max(0.0, float(word["start_s"]))
        end = min(duration, float(word["end_s"]))
        left = 100.0 * start / duration
        width = max(0.25, 100.0 * (end - start) / duration)
        label = html.escape(str(word.get("text") or ""), quote=True)
        title = html.escape(f"{label}  {start:.3f}–{end:.3f}s", quote=True)
        spans.append(
            f'<span class="word" style="left:{left:.4f}%;width:{width:.4f}%" '
            f'title="{title}">{label}</span>'
        )
    crop = row.get("crop")
    crop_html = ""
    if crop:
        start = float(crop["source_start_s"])
        end = float(crop["source_end_s"])
        left = 100.0 * start / duration
        width = 100.0 * (end - start) / duration
        crop_html = (
            f'<span class="crop" style="left:{left:.4f}%;width:{width:.4f}%" '
            f'title="训练裁剪 {start:.3f}–{end:.3f}s"></span>'
        )
    return f'<div class="timeline">{crop_html}{"".join(spans)}</div>'


def build_page(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir(exist_ok=True)
    cards: list[str] = []
    exported: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        source = Path(str(row["audio"]))
        suffix = source.suffix.lower() or ".ogg"
        media_name = f"{index:03d}_{row['source_id']}{suffix}"
        shutil.copy2(source, media_dir / media_name)
        exported_row = {**row, "audio": f"media/{media_name}"}
        exported.append(exported_row)
        status = str(row["status"])
        crop = row.get("crop")
        crop_text = (
            f"<b>训练目标：</b>{html.escape(str(crop['text']))}　"
            f"<b>裁剪：</b>{float(crop['source_start_s']):.3f}–"
            f"{float(crop['source_end_s']):.3f}s"
            if crop
            else "不进入训练 manifest"
        )
        cards.append(
            f'<section class="card {html.escape(status)}">'
            f'<header><h2>样本 {index:02d}</h2><span>{html.escape(STATUS_LABELS.get(status, status))}</span></header>'
            f'<audio controls preload="metadata" src="media/{html.escape(media_name)}"></audio>'
            f'<p><b>数据集原文：</b>{html.escape(str(row["canonical_text"]))}</p>'
            f'<p><b>Grok 转写：</b>{html.escape(str(row["teacher_text"])) or "（空白）"}</p>'
            f'<p class="metrics">CER {_metric(row.get("cer"))}　教师匹配 {_metric(row.get("teacher_match_share"))}　原文覆盖 {_metric(row.get("canonical_match_share"))}</p>'
            f'{_timeline(row)}<p class="crop-note">{crop_text}</p>'
            f'<details><summary>来源 ID</summary><code>{html.escape(str(row["source_id"]))}</code></details>'
            "</section>"
        )
    status_counts = Counter(str(row["status"]) for row in rows)
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Galgame Grok CTC 教师审计</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;background:#f3f5f7;color:#17212b}}
.intro,.card{{background:#fff;border:1px solid #d9e0e6;border-radius:12px;padding:16px;margin:0 0 14px;box-shadow:0 2px 10px #21354712}}
h1{{margin-top:0}}header{{display:flex;justify-content:space-between;align-items:center;gap:12px}}header h2{{font-size:18px;margin:0}}header span{{font-weight:700}}audio{{width:100%;margin:12px 0}}p{{line-height:1.65;margin:8px 0}}.metrics{{color:#50606e}}.timeline{{position:relative;height:54px;background:#e9edf1;border-radius:7px;overflow:hidden;margin:12px 0}}.word{{position:absolute;top:8px;height:24px;min-width:3px;background:#3386b5;color:white;font-size:11px;overflow:hidden;text-align:center;border-radius:3px}}.crop{{position:absolute;top:36px;height:10px;background:#49a367;opacity:.85;border-radius:5px}}.crop-note{{background:#eef6ef;padding:8px 10px;border-radius:6px}}.teacher_acoustic_empty header span,.cer_above_maximum header span,.no_eligible_island header span{{color:#a13d3d}}.accepted header span{{color:#237443}}details{{color:#65727d}}code{{overflow-wrap:anywhere}}
</style></head><body>
<section class="intro"><h1>Galgame Grok CTC 教师审计</h1>
<p>蓝色是 Grok 的字符级时间单元，绿色是实际进入 CTC manifest 的裁剪范围。训练文字始终来自数据集原文，Grok 只提供时间证据。</p>
<p>本页抽样 {len(rows)} 条：{html.escape(json.dumps(dict(status_counts), ensure_ascii=False))}。请重点试听绿色裁剪是否完整包住对应文字，以及拒绝样本是否确实不值得进入训练。</p></section>
{"".join(cards)}</body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    _write_jsonl(output_dir / "audit_rows.jsonl", exported)
    return {"rows": len(rows), "selected_by_status": dict(status_counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accepted", type=int, default=24)
    parser.add_argument("--empty", type=int, default=12)
    parser.add_argument("--high-cer", type=int, default=12)
    parser.add_argument("--no-island", type=int, default=4)
    args = parser.parse_args()
    selected, population = select_audit_rows(
        _read_jsonl(Path(args.results)),
        accepted=args.accepted,
        empty=args.empty,
        high_cer=args.high_cer,
        no_island=args.no_island,
    )
    output_dir = Path(args.output_dir)
    summary = {
        "schema": "galgame_grok_ctc_teacher_audit_v1",
        "results": str(args.results),
        "population_by_status": dict(population),
        **build_page(selected, output_dir),
        "page": "index.html",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
