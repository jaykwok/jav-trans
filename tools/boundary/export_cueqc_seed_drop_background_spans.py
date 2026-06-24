#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CANDIDATE_SCHEMA = "speech_boundary_hard_negative_candidate_from_cueqc_v1"
SUMMARY_SCHEMA = "cueqc_seed_drop_background_span_export_summary_v1"
FRAME_NEGATIVE_ROUTE = "speech_boundary_frame_negative_candidate"


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_display_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        return str(raw)


def local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def discover_latest_seed_candidates() -> Path:
    paths = sorted(
        (PROJECT_ROOT / "agents" / "temp").glob(
            "*_scorer-v5-native-realneg-candidates-from-cluster-seed/cueqc_confirmed_drop_candidates.jsonl"
        )
    )
    if not paths:
        paths = sorted(
            (PROJECT_ROOT / "agents" / "temp").glob(
                "*_speech-boundary-hard-negative-candidates-from-cueqc-cluster-seed/cueqc_confirmed_drop_candidates.jsonl"
            )
        )
    if not paths:
        raise FileNotFoundError("no CueQC cluster-seed drop candidate manifest found under agents/temp")
    return paths[-1]


def discover_latest_all_candidates() -> Path:
    paths = sorted((PROJECT_ROOT / "agents" / "temp").glob("*_cueqc17b-v4-fresh-10film-candidates/cueqc_candidates.jsonl"))
    if not paths:
        raise FileNotFoundError("no fresh CueQC candidate pool found under agents/temp")
    return paths[-1]


def row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def row_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default) or default)
    except (TypeError, ValueError):
        return int(default)


def text_char_count(row: Mapping[str, Any]) -> int:
    for key in ("text_features", "cue_features"):
        features = row.get(key)
        if not isinstance(features, Mapping):
            continue
        if key == "cue_features":
            features = features.get("text_observation")
        if isinstance(features, Mapping):
            return row_int(features, "char_count", 0)
    return 0


def is_empty_observation(row: Mapping[str, Any]) -> bool:
    if text_char_count(row) > 0:
        return False
    text = str(row.get("compact_text") or row.get("text_preview") or row.get("text") or "").strip()
    return text in {"", "...", "…"}


def source_audio_path(row: Mapping[str, Any]) -> str:
    value = row.get("source_audio_path")
    if value:
        return repo_display_path(project_path(str(value)))
    audio = row.get("audio")
    if isinstance(audio, Mapping) and audio.get("path"):
        return repo_display_path(project_path(str(audio["path"])))
    return ""


def duration_bucket(duration_s: float) -> str:
    if duration_s < 1.0:
        return "<1s"
    if duration_s < 2.0:
        return "1-2s"
    if duration_s < 4.0:
        return "2-4s"
    if duration_s < 8.0:
        return "4-8s"
    return ">=8s"


def duration_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "total_s": 0.0, "p50_s": 0.0, "p90_s": 0.0, "p99_s": 0.0, "max_s": 0.0}
    ordered = sorted(float(value) for value in values)

    def quantile(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return ordered[index]

    return {
        "count": len(ordered),
        "total_s": round(sum(ordered), 6),
        "p50_s": round(quantile(0.50), 6),
        "p90_s": round(quantile(0.90), 6),
        "p99_s": round(quantile(0.99), 6),
        "max_s": round(max(ordered), 6),
    }


def load_seed_candidates(path: Path) -> tuple[set[str], dict[str, dict[str, Any]], list[str]]:
    rows = read_jsonl(path)
    seed_ids: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    label_paths: set[str] = set()
    for row in rows:
        schema = str(row.get("schema") or "")
        if schema and schema != CANDIDATE_SCHEMA:
            raise ValueError(f"unsupported seed candidate schema {schema!r}")
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        seed_ids.add(sample_id)
        by_id[sample_id] = dict(row)
        for value in row.get("source_label_paths") or []:
            label_paths.add(repo_display_path(project_path(str(value))))
    return seed_ids, by_id, sorted(label_paths)


def grouped_candidates(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        audio_path = source_audio_path(row)
        if not audio_path:
            continue
        key = (audio_path, str(row.get("video_id") or ""), str(row.get("audio_id") or ""))
        groups[key].append(dict(row))
    return groups


def nonempty_boundaries(
    intervals: Sequence[tuple[float, float]],
    *,
    start: float,
    end: float,
) -> tuple[bool, float | None, float | None]:
    previous_end: float | None = None
    next_start: float | None = None
    for non_start, non_end in intervals:
        if non_end <= start:
            previous_end = max(previous_end or 0.0, non_end)
            continue
        if non_start >= end:
            next_start = min(next_start if next_start is not None else non_start, non_start)
            continue
        if non_end > start and non_start < end:
            return True, previous_end, next_start
    return False, previous_end, next_start


def merge_seeded_empty_spans(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed_sample_ids: set[str],
    max_gap_s: float,
    guard_s: float,
    min_duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    emitted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    groups = grouped_candidates(rows)
    for (audio_path, video_id, audio_id), group_rows in sorted(groups.items(), key=lambda item: item[0]):
        empty_intervals: list[dict[str, Any]] = []
        nonempty_intervals: list[tuple[float, float]] = []
        for row in sorted(group_rows, key=lambda item: (row_float(item, "start"), row_float(item, "end"))):
            start = row_float(row, "start")
            end = row_float(row, "end", start)
            if end <= start:
                skipped.append({"sample_id": row.get("sample_id"), "reason": "non_positive_candidate_span"})
                continue
            if is_empty_observation(row):
                sample_id = str(row.get("sample_id") or "")
                empty_intervals.append(
                    {
                        "start": start,
                        "end": end,
                        "sample_ids": [sample_id] if sample_id else [],
                        "seed_sample_ids": [sample_id] if sample_id in seed_sample_ids else [],
                        "chunk_indexes": [row_int(row, "chunk_index", 0)],
                    }
                )
            else:
                nonempty_intervals.append((start, end))

        merged: list[dict[str, Any]] = []
        for interval in empty_intervals:
            if not merged or float(interval["start"]) > float(merged[-1]["end"]) + max_gap_s:
                merged.append({**interval})
                continue
            current = merged[-1]
            current["end"] = max(float(current["end"]), float(interval["end"]))
            current["sample_ids"].extend(interval["sample_ids"])
            current["seed_sample_ids"].extend(interval["seed_sample_ids"])
            current["chunk_indexes"].extend(interval["chunk_indexes"])

        for span in merged:
            seed_ids = [str(item) for item in span["seed_sample_ids"] if item]
            if not seed_ids:
                skipped.append({"video_id": video_id, "start": span["start"], "end": span["end"], "reason": "no_seed_anchor"})
                continue
            start = float(span["start"])
            end = float(span["end"])
            overlaps, previous_end, next_start = nonempty_boundaries(nonempty_intervals, start=start, end=end)
            if overlaps:
                skipped.append(
                    {
                        "video_id": video_id,
                        "start": round(start, 6),
                        "end": round(end, 6),
                        "reason": "overlaps_nonempty_text_candidate",
                    }
                )
                continue
            if previous_end is not None:
                start = max(start, previous_end + guard_s)
            if next_start is not None:
                end = min(end, next_start - guard_s)
            duration_s = max(0.0, end - start)
            if duration_s < min_duration_s:
                skipped.append(
                    {
                        "video_id": video_id,
                        "start": round(start, 6),
                        "end": round(end, 6),
                        "duration_s": round(duration_s, 6),
                        "reason": "too_short_after_guard",
                        "seed_anchor_count": len(seed_ids),
                    }
                )
                continue
            emitted.append(
                {
                    "audio_id": audio_id,
                    "chunk_index": min(int(value) for value in span["chunk_indexes"]) if span["chunk_indexes"] else 0,
                    "duration_s": round(duration_s, 6),
                    "empty_chunk_count": len(span["sample_ids"]),
                    "end": round(end, 6),
                    "seed_sample_ids": seed_ids,
                    "source_audio_path": audio_path,
                    "start": round(start, 6),
                    "video_id": video_id,
                }
            )
    return emitted, skipped


def build_candidate(
    *,
    span: Mapping[str, Any],
    index: int,
    seed_candidates_path: Path,
    all_candidates_path: Path,
    source_label_paths: Sequence[str],
    max_gap_s: float,
    guard_s: float,
) -> dict[str, Any]:
    video_id = str(span.get("video_id") or "unknown")
    start = row_float(span, "start")
    end = row_float(span, "end", start)
    duration_s = max(0.0, end - start)
    seed_sample_ids = [str(item) for item in span.get("seed_sample_ids") or []]
    sample_id = f"cueqc-bgspan-{video_id}-{index:05d}"
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": f"cueqc-seeddrop-background-span-{video_id}-{index:05d}",
        "source": "cueqc_cluster_seed_drop_background_span",
        "source_label_paths": list(source_label_paths),
        "source_candidate_path": repo_display_path(all_candidates_path),
        "source_seed_candidate_path": repo_display_path(seed_candidates_path),
        "source_label_count": len(seed_sample_ids),
        "source_evidence": [
            {
                "seed_candidate_path": repo_display_path(seed_candidates_path),
                "all_candidates_path": repo_display_path(all_candidates_path),
                "seed_sample_ids": seed_sample_ids[:50],
                "seed_anchor_count": len(seed_sample_ids),
                "empty_chunk_count": int(span.get("empty_chunk_count") or 0),
                "max_gap_s": max_gap_s,
                "guard_s": guard_s,
            }
        ],
        "sample_id": sample_id,
        "audit_id": sample_id,
        "video_id": video_id,
        "video_label": video_id,
        "chunk_index": int(span.get("chunk_index") or 0),
        "start": round(start, 6),
        "end": round(end, 6),
        "duration_s": round(duration_s, 6),
        "duration_bucket": duration_bucket(duration_s),
        "text": "",
        "raw_text": "",
        "text_observation_bucket": "empty_contiguous_background",
        "manual_decision": "cluster_seed_drop_background_span",
        "reason_tags": ["cluster_seed_drop_background_span", "empty_text_contiguous"],
        "notes": "Merged contiguous empty-text CueQC candidates around audited seed-drop anchors.",
        "display_prob_drop_min": 1.0,
        "display_prob_drop_max": 1.0,
        "display_prob_drop_mean": 1.0,
        "display_prob_keep_mean": 0.0,
        "candidate_route": FRAME_NEGATIVE_ROUTE,
        "route_reason": "empty-text contiguous span contains audited CueQC seed-drop anchors and no text-present overlap",
        "hard_negative_status": "candidate_requires_audio_materialization",
        "required_conversion": "convert to SpeechBoundary-JA frame-negative labels",
        "source_audio_path": str(span.get("source_audio_path") or ""),
        "audio_id": str(span.get("audio_id") or ""),
        "seed_sample_ids": seed_sample_ids,
        "empty_chunk_count": int(span.get("empty_chunk_count") or 0),
    }


def export_cueqc_seed_drop_background_spans(
    *,
    seed_candidates_path: Path,
    all_candidates_path: Path,
    output_dir: Path,
    max_gap_s: float = 0.2,
    guard_s: float = 0.12,
    min_duration_s: float = 1.2,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    seed_candidates_path = seed_candidates_path.resolve()
    all_candidates_path = all_candidates_path.resolve()
    output_dir = output_dir.resolve()
    if not seed_candidates_path.exists():
        raise FileNotFoundError(f"missing seed candidates: {repo_display_path(seed_candidates_path)}")
    if not all_candidates_path.exists():
        raise FileNotFoundError(f"missing all candidates: {repo_display_path(all_candidates_path)}")
    seed_sample_ids, _seed_rows, source_label_paths = load_seed_candidates(seed_candidates_path)
    all_rows = read_jsonl(all_candidates_path)
    spans, skipped = merge_seeded_empty_spans(
        all_rows,
        seed_sample_ids=seed_sample_ids,
        max_gap_s=max_gap_s,
        guard_s=guard_s,
        min_duration_s=min_duration_s,
    )
    candidates = [
        build_candidate(
            span=span,
            index=index,
            seed_candidates_path=seed_candidates_path,
            all_candidates_path=all_candidates_path,
            source_label_paths=source_label_paths,
            max_gap_s=max_gap_s,
            guard_s=guard_s,
        )
        for index, span in enumerate(sorted(spans, key=lambda row: (str(row.get("video_id") or ""), row_float(row, "start"))))
    ]
    if require_nonempty and not candidates:
        raise ValueError("no CueQC seed-drop background spans were exported")

    candidates_path = output_dir / "cueqc_seed_drop_background_span_candidates.jsonl"
    skipped_path = output_dir / "cueqc_seed_drop_background_span_skipped.json"
    summary_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    write_jsonl(candidates_path, candidates)
    write_json(skipped_path, skipped)

    durations = [row_float(row, "duration_s") for row in candidates]
    video_counts = Counter(str(row.get("video_id") or "") for row in candidates)
    duration_counts = Counter(str(row.get("duration_bucket") or "") for row in candidates)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed_candidates_path": repo_display_path(seed_candidates_path),
        "all_candidates_path": repo_display_path(all_candidates_path),
        "output_dir": repo_display_path(output_dir),
        "params": {
            "max_gap_s": max_gap_s,
            "guard_s": guard_s,
            "min_duration_s": min_duration_s,
        },
        "outputs": {
            "background_span_candidates": repo_display_path(candidates_path),
            "skipped": repo_display_path(skipped_path),
            "summary_json": repo_display_path(summary_path),
            "summary_md": repo_display_path(summary_md_path),
        },
        "counts": {
            "seed_sample_ids": len(seed_sample_ids),
            "all_candidates": len(all_rows),
            "exported_candidates": len(candidates),
            "skipped_spans": len(skipped),
        },
        "duration_stats": duration_stats(durations),
        "candidate_video_counts": dict(video_counts),
        "candidate_duration_bucket_counts": dict(duration_counts),
        "speech_boundary_hard_negative": {
            "frame_negative_candidates_emitted": True,
            "direct_boundary_refiner_dataset_emitted": False,
            "policy": (
                "Merge contiguous empty-text CueQC candidates only when the merged span contains audited "
                "seed-drop anchors and does not overlap text-present candidates."
            ),
            "next_conversion": [
                "run tools.boundary.prepare_cueqc_drop_hard_negative_sources with --candidates background_span_candidates",
                "pass speech_boundary_negative_manifest.json to tools.boundary.ja.build_scorer_v5_native_dataset",
            ],
        },
    }
    write_json(summary_path, summary)
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    stats = summary["duration_stats"]
    lines = [
        "# CueQC Seed-Drop Background Span Export",
        "",
        f"- Seed candidates: `{summary['seed_candidates_path']}`",
        f"- All candidates: `{summary['all_candidates_path']}`",
        f"- Output: `{summary['output_dir']}`",
        "",
        "## Counts",
        "",
        f"- Seed sample ids: `{counts['seed_sample_ids']}`",
        f"- All candidates: `{counts['all_candidates']}`",
        f"- Exported background spans: `{counts['exported_candidates']}`",
        f"- Duration total/p50/p90/max: `{stats['total_s']}` / `{stats['p50_s']}` / `{stats['p90_s']}` / `{stats['max_s']}`",
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in summary["outputs"].items())
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Emits SpeechBoundary-JA hard-negative candidates only.",
            "- It does not emit Boundary Refiner rows.",
            "- Text-present overlaps are rejected; neighboring text-present chunks are separated by guard time.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export longer SpeechBoundary-JA hard-negative candidates by merging contiguous empty-text "
            "CueQC chunks around audited cluster seed-drop anchors."
        )
    )
    parser.add_argument("--seed-candidates", default="", help="cueqc_confirmed_drop_candidates.jsonl from cluster seed export.")
    parser.add_argument("--all-candidates", default="", help="Full fresh cueqc_candidates.jsonl pool.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/YYYYMMDD_HHMMSS_speech-boundary-bgspan-candidates-from-cueqc-seed-drop.",
    )
    parser.add_argument("--max-gap-s", type=float, default=0.2)
    parser.add_argument("--guard-s", type=float, default=0.12)
    parser.add_argument("--min-duration-s", type=float, default=1.2)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args(argv)
    if args.max_gap_s < 0.0:
        parser.error("--max-gap-s must be non-negative")
    if args.guard_s < 0.0:
        parser.error("--guard-s must be non-negative")
    if args.min_duration_s <= 0.0:
        parser.error("--min-duration-s must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_candidates_path = project_path(args.seed_candidates) if args.seed_candidates else discover_latest_seed_candidates()
    all_candidates_path = project_path(args.all_candidates) if args.all_candidates else discover_latest_all_candidates()
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "agents"
        / "temp"
        / f"{local_timestamp()}_speech-boundary-bgspan-candidates-from-cueqc-seed-drop"
    )
    summary = export_cueqc_seed_drop_background_spans(
        seed_candidates_path=seed_candidates_path,
        all_candidates_path=all_candidates_path,
        output_dir=output_dir,
        max_gap_s=args.max_gap_s,
        guard_s=args.guard_s,
        min_duration_s=args.min_duration_s,
        require_nonempty=not args.allow_empty,
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"candidates={summary['outputs']['background_span_candidates']}")
    print(
        "spans={count} total_s={total_s} p90_s={p90_s}".format(
            count=summary["counts"]["exported_candidates"],
            total_s=summary["duration_stats"]["total_s"],
            p90_s=summary["duration_stats"]["p90_s"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
