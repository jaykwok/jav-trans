#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate import (  # noqa: E402
    OUTPUT_ROOT,
    PROJECT_ROOT,
    Cue,
    Match,
    align_modes,
    class_for_similarity,
    fmt_time,
    html_page,
    load_json_file,
    load_srt_file,
    project_rel,
    safe_delta,
    write_json,
)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def display_path(path: Path) -> str:
    try:
        return project_rel(path)
    except ValueError:
        return str(path)


def unique_label(label: str, used: set[str]) -> str:
    clean = label.strip() or "asr"
    if clean not in used:
        used.add(clean)
        return clean
    suffix = 2
    while f"{clean}-{suffix}" in used:
        suffix += 1
    final = f"{clean}-{suffix}"
    used.add(final)
    return final


def parse_input_spec(spec: str, used: set[str]) -> tuple[str, Path]:
    if "=" in spec:
        label, path_text = spec.split("=", 1)
        path = resolve_project_path(path_text.strip()).resolve()
        return unique_label(label, used), path
    path = resolve_project_path(spec.strip()).resolve()
    return unique_label(path.stem, used), path


def load_source(label: str, path: Path) -> list[Cue]:
    if not path.exists():
        raise SystemExit(f"ASR subtitle not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".srt":
        cues = load_srt_file(path, label)
    elif suffix == ".json":
        cues = load_json_file(path, label)
    else:
        raise SystemExit(f"unsupported ASR subtitle file type: {path}")
    if not cues:
        raise SystemExit(f"no subtitle cues parsed from: {path}")
    return cues


def match_cell(match: Match) -> str:
    if match.cue is None:
        return '<td class="missing">-</td>'
    cue = match.cue
    cls = class_for_similarity(match.text_similarity)
    meta = (
        f"#{cue.index + 1} {fmt_time(cue.start)}-{fmt_time(cue.end)} "
        f"ja={match.text_similarity:.2f} "
        f"ds={safe_delta(match.time_delta_start)} de={safe_delta(match.time_delta_end)}"
    )
    return (
        f'<td class="{cls}">'
        f'<div class="jp">{html.escape(cue.ja_text)}</div>'
        f'<div class="meta">{html.escape(meta)}</div>'
        "</td>"
    )


def build_report(
    *,
    video_stem: str,
    input_paths: dict[str, Path],
    sources: list[str],
    base_source: str,
    rows: list[tuple[Cue, dict[str, Match]]],
) -> str:
    title = f"{video_stem} ASR backend Japanese sentence compare"
    counts = {"same": 0, "close": 0, "diff": 0, "missing": 0}
    table_rows = []
    for base, matches in rows:
        cells = []
        row_classes = []
        for source in sources:
            match = matches[source]
            if source != base_source:
                if match.cue is None:
                    counts["missing"] += 1
                    row_classes.append("missing-row")
                else:
                    counts[class_for_similarity(match.text_similarity)] += 1
                    if match.text_similarity < 0.82:
                        row_classes.append("diff")
            cells.append(match_cell(match))
        table_rows.append(
            f'<tr class="{" ".join(row_classes)}">'
            f'<td class="time"><div>{fmt_time(base.start)}</div><div>{fmt_time(base.end)}</div><div>{base.duration:.2f}s</div></td>'
            + "".join(cells)
            + "</tr>"
        )

    headers = "".join(f"<th>{html.escape(source)}</th>" for source in sources)
    stat = ", ".join(f"{key}={value}" for key, value in counts.items())
    inputs = "".join(
        f"<li><strong>{html.escape(label)}</strong>: {html.escape(display_path(path))}</li>"
        for label, path in input_paths.items()
    )
    body = f"""
<header>
<h1>{html.escape(title)}</h1>
<p>Base ASR: {html.escape(base_source)}. Rows are aligned by cue time overlap, then Japanese text similarity.</p>
<p>Stats: {html.escape(stat)}</p>
</header>
<main>
<section class="panel">
<p>Inputs:</p>
<ul>{inputs}</ul>
</section>
<table>
<thead><tr><th class="time">time</th>{headers}</tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</main>
"""
    return html_page(title, body)


def build_review_items(
    rows: list[tuple[Cue, dict[str, Match]]],
    sources: list[str],
    base_source: str,
    limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for base, matches in rows:
        labels = []
        reasons = []
        missing_sources = []
        variants: dict[str, dict[str, Any]] = {}
        worst_similarity = 1.0
        max_shift = 0.0
        for source in sources:
            match = matches[source]
            cue = match.cue
            variants[source] = {
                "matched": cue is not None,
                "index": cue.index if cue else None,
                "start": cue.start if cue else None,
                "end": cue.end if cue else None,
                "ja_text": cue.ja_text if cue else "",
                "text_similarity": match.text_similarity,
                "time_delta_start": None if math.isnan(match.time_delta_start) else match.time_delta_start,
                "time_delta_end": None if math.isnan(match.time_delta_end) else match.time_delta_end,
            }
            if source == base_source:
                continue
            if cue is None:
                missing_sources.append(source)
                continue
            worst_similarity = min(worst_similarity, match.text_similarity)
            max_shift = max(max_shift, abs(match.time_delta_start), abs(match.time_delta_end))

        if missing_sources:
            labels.append("missing-backend-match")
            reasons.append(f"missing in {', '.join(missing_sources)}")
        if worst_similarity < 0.68:
            labels.append("large-asr-text-diff")
            reasons.append(f"worst JA similarity {worst_similarity:.2f}")
        elif worst_similarity < 0.84:
            labels.append("moderate-asr-text-diff")
            reasons.append(f"worst JA similarity {worst_similarity:.2f}")
        if max_shift > 2.0:
            labels.append("time-shift")
            reasons.append(f"max boundary shift {max_shift:.2f}s")
        if not base.ja_text:
            labels.append("empty-base-ja")
            reasons.append("base Japanese text is empty")
        if not labels:
            continue

        severity = "high" if missing_sources or worst_similarity < 0.68 else "medium" if max_shift > 2.0 else "low"
        items.append(
            {
                "item_id": f"{base.index + 1:05d}",
                "start": max(0.0, base.start - 1.0),
                "end": base.end + 1.25,
                "severity": severity,
                "labels": labels,
                "reason": "; ".join(reasons),
                "base_index": base.index,
                "base_ja": base.ja_text,
                "variants": variants,
            }
        )
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda item: (severity_rank.get(str(item["severity"]), 9), float(item["start"])))
    return items[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Japanese transcripts from different ASR backends.")
    parser.add_argument("--video", required=True, help="Path to source video, used for output folder naming")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="ASR subtitle input as label=path or path. Repeat at least twice.",
    )
    parser.add_argument("--base", help="Input label to use as base. Defaults to the first --input label.")
    parser.add_argument("--review-limit", type=int, default=200)
    parser.add_argument("--output-prefix", default="asr_backend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = resolve_project_path(args.video).resolve()
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")
    if len(args.input) < 2:
        raise SystemExit("at least two --input values are required")

    used_labels: set[str] = set()
    input_paths: dict[str, Path] = {}
    cues_by_source: dict[str, list[Cue]] = {}
    for spec in args.input:
        label, path = parse_input_spec(spec, used_labels)
        input_paths[label] = path
        cues_by_source[label] = load_source(label, path)

    sources = list(cues_by_source)
    base_source = args.base or sources[0]
    if base_source not in cues_by_source:
        raise SystemExit(f"base {base_source!r} not in inputs: {sources}")

    rows = align_modes(cues_by_source, base_source)
    review_items = build_review_items(rows, sources, base_source, args.review_limit)

    output_dir = OUTPUT_ROOT / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"{args.output_prefix}_japanese_compare.html"
    review_name = f"{args.output_prefix}_review_items.json"
    summary_name = f"{args.output_prefix}_summary.json"

    (output_dir / report_name).write_text(
        build_report(
            video_stem=video_path.stem,
            input_paths=input_paths,
            sources=sources,
            base_source=base_source,
            rows=rows,
        ),
        encoding="utf-8",
    )
    write_json(output_dir / review_name, review_items)
    write_json(
        output_dir / summary_name,
        {
            "video": display_path(video_path),
            "base_source": base_source,
            "sources": sources,
            "counts": {source: len(cues) for source, cues in cues_by_source.items()},
            "review_item_count": len(review_items),
            "outputs": {
                "asr_backend_japanese_compare": report_name,
                "asr_backend_review_items": review_name,
            },
        },
    )

    print(output_dir.relative_to(PROJECT_ROOT))
    print(output_dir / report_name)


if __name__ == "__main__":
    main()
