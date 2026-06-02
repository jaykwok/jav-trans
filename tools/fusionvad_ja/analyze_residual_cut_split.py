#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.chunk_packer import (  # noqa: E402
    FramePackingConfig,
    PackedChunk,
    _split_pre_asr_risk_chunk,
)
from vad.base import SpeechSegment  # noqa: E402


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


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def compact_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "p50": round(quantile(values, 0.50), 6),
        "p90": round(quantile(values, 0.90), 6),
        "p95": round(quantile(values, 0.95), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
    }


def scores_from_payload(payload: dict[str, Any], keys: tuple[str, ...]) -> list[float]:
    for key in keys:
        raw = payload.get(key)
        if raw is not None:
            return [float(value) for value in raw]
    return []


def unsafe_indices_from_rows(rows: list[dict[str, Any]]) -> set[int]:
    indices: set[int] = set()
    for row in rows:
        try:
            index = int(row.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        indices.add(index)
    return indices


def sentinel_indices_from_diagnostics(rows: list[dict[str, Any]]) -> set[int]:
    indices: set[int] = set()
    for row in rows:
        if str(row.get("fallback_subtype") or "") != "vad_coarse_after_sentinel":
            continue
        try:
            indices.add(int(row.get("chunk_index")))
        except (TypeError, ValueError):
            continue
    return indices


def frame_hop_from_cache(cache: dict[str, Any], default: float = 1.0 / 29.97) -> float:
    runtime = cache.get("runtime_vad_signature", {}).get("chunk_packing") or {}
    signature = cache.get("signature", {}).get("chunk") or {}
    for value in (
        runtime.get("frame_hop_s"),
        signature.get("pack_frame_hop_s"),
        signature.get("frame_hop_s"),
    ):
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0.0:
            return parsed
    return default


def chunk_from_span(item: dict[str, Any]) -> PackedChunk:
    vad_segments: list[SpeechSegment] = []
    for segment in item.get("vad_segments") or []:
        if not isinstance(segment, dict):
            continue
        start = row_float(segment, "start")
        end = row_float(segment, "end", start)
        if end <= start:
            continue
        score = segment.get("score")
        vad_segments.append(
            SpeechSegment(
                start=start,
                end=end,
                score=None if score is None else float(score),
            )
        )
    start = row_float(item, "start")
    end = row_float(item, "end", start)
    core_start = row_float(
        item,
        "core_start",
        vad_segments[0].start if vad_segments else start,
    )
    core_end = row_float(
        item,
        "core_end",
        vad_segments[-1].end if vad_segments else end,
    )
    return PackedChunk(
        start=start,
        end=end,
        vad_segments=vad_segments,
        duration=max(0.0, end - start),
        left_padding_s=row_float(item, "left_padding_s", max(0.0, core_start - start)),
        right_padding_s=row_float(item, "right_padding_s", max(0.0, end - core_end)),
        split_reason=str(item.get("split_reason") or ""),
        core_start=core_start,
        core_end=core_end,
        internal_gap_count=row_int(item, "internal_gap_count"),
        internal_gap_max_s=row_float(item, "internal_gap_max_s"),
        split_policy=str(item.get("split_policy") or ""),
        valley_split_count=row_int(item, "valley_split_count"),
        valley_score_min=item.get("valley_score_min"),
        cut_split_count=row_int(item, "cut_split_count"),
        cut_score_max=item.get("cut_score_max"),
        drop_gap_split_count=row_int(item, "drop_gap_split_count"),
        drop_gap_score_max=item.get("drop_gap_score_max"),
        risk_split_count=row_int(item, "risk_split_count"),
        risk_score=item.get("risk_score"),
        risk_reasons=tuple(str(value) for value in (item.get("risk_reasons") or ())),
    )


def child_payload(child: PackedChunk) -> dict[str, Any]:
    core_start = child.core_start if child.core_start is not None else child.start
    core_end = child.core_end if child.core_end is not None else child.end
    return {
        "start": round(float(child.start), 6),
        "end": round(float(child.end), 6),
        "duration_s": round(max(0.0, float(child.end) - float(child.start)), 6),
        "core_start": round(float(core_start), 6),
        "core_end": round(float(core_end), 6),
        "core_duration_s": round(max(0.0, float(core_end) - float(core_start)), 6),
        "split_reason": child.split_reason,
        "split_policy": child.split_policy,
        "risk_score": child.risk_score,
        "risk_reasons": list(child.risk_reasons),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    vad_cache = project_path(args.vad_cache)
    frame_scores_path = project_path(args.frame_scores)
    diagnostics_path = project_path(args.diagnostics) if args.diagnostics else None
    unsafe_path = project_path(args.unsafe_fallback_chunks) if args.unsafe_fallback_chunks else None
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = read_json(vad_cache)
    if not isinstance(cache, dict):
        raise ValueError(f"VAD cache must be a JSON object: {vad_cache}")
    score_payload = read_json(frame_scores_path)
    if not isinstance(score_payload, dict):
        raise ValueError(f"frame scores must be a JSON object: {frame_scores_path}")

    frame_scores = scores_from_payload(score_payload, ("scores", "frame_scores"))
    cut_scores = scores_from_payload(score_payload, ("cut_scores", "cut_frame_scores"))
    score_frame_hop_s = float(score_payload.get("frame_hop_s") or 0.02)
    if not frame_scores:
        raise ValueError("frame scores payload has no speech scores")
    if not cut_scores:
        raise ValueError("frame scores payload has no cut scores")

    unsafe_indices = unsafe_indices_from_rows(read_jsonl(unsafe_path))
    sentinel_indices = sentinel_indices_from_diagnostics(read_jsonl(diagnostics_path))
    target_indices = unsafe_indices or sentinel_indices
    frame_hop_s = frame_hop_from_cache(cache)
    config = FramePackingConfig(
        frame_hop_s=frame_hop_s,
        score_frame_hop_s=score_frame_hop_s,
        window_frames=args.window_frames,
        reserve_frames=args.reserve_frames,
        target_padding_frames=args.target_padding_frames,
        gap_merge_frames=args.gap_merge_frames,
        pre_asr_risk_split_enabled=True,
        pre_asr_risk_split_min_core_frames=args.min_core_frames,
        pre_asr_risk_split_target_core_frames=args.target_core_frames,
        pre_asr_risk_split_safe_core_frames=args.safe_core_frames,
        pre_asr_risk_split_min_gap_frames=args.min_gap_frames,
        pre_asr_risk_split_min_child_frames=args.min_child_frames,
        pre_asr_risk_split_max_children=args.max_children,
        pre_asr_risk_split_threshold=args.risk_threshold,
        pre_asr_risk_split_continuous_threshold=args.continuous_threshold,
        pre_asr_risk_split_valley_threshold=args.valley_threshold,
        pre_asr_risk_split_cut_threshold=args.cut_threshold,
        pre_asr_cut_split_min_cut_frames=args.min_cut_frames,
        pre_asr_valley_split_min_valley_frames=args.min_valley_frames,
    )

    rows: list[dict[str, Any]] = []
    spans = cache.get("processing_spans") or []
    for index, item in enumerate(spans):
        if not isinstance(item, dict):
            continue
        chunk = chunk_from_span(item)
        children = _split_pre_asr_risk_chunk(
            chunk,
            parent_index=index,
            config=config,
            frame_scores=frame_scores,
            cut_frame_scores=cut_scores,
        )
        child_rows = [child_payload(child) for child in children]
        max_child_duration = max((row["duration_s"] for row in child_rows), default=0.0)
        max_child_core = max((row["core_duration_s"] for row in child_rows), default=0.0)
        core_start = chunk.core_start if chunk.core_start is not None else chunk.start
        core_end = chunk.core_end if chunk.core_end is not None else chunk.end
        rows.append(
            {
                "chunk_index": index,
                "is_target": index in target_indices,
                "is_unsafe": index in unsafe_indices,
                "is_sentinel": index in sentinel_indices,
                "old_start": round(chunk.start, 6),
                "old_end": round(chunk.end, 6),
                "old_duration_s": round(chunk.duration, 6),
                "old_core_start": round(float(core_start), 6),
                "old_core_end": round(float(core_end), 6),
                "old_core_duration_s": round(max(0.0, float(core_end) - float(core_start)), 6),
                "old_split_reason": chunk.split_reason,
                "old_split_policy": chunk.split_policy,
                "old_vad_segment_count": len(chunk.vad_segments),
                "old_internal_gap_count": chunk.internal_gap_count,
                "old_internal_gap_max_s": chunk.internal_gap_max_s,
                "child_count": len(child_rows),
                "max_child_duration_s": max_child_duration,
                "max_child_core_duration_s": max_child_core,
                "children": child_rows,
                "risk_reasons": child_rows[0]["risk_reasons"] if child_rows else [],
            }
        )

    target_rows = [row for row in rows if row["is_target"]]
    split_rows = [row for row in rows if int(row["child_count"]) > 1]
    target_split_rows = [row for row in target_rows if int(row["child_count"]) > 1]
    total_children = sum(int(row["child_count"]) for row in rows)
    risk_reason_counts: Counter[str] = Counter()
    for row in split_rows:
        risk_reason_counts.update(str(value) for value in row.get("risk_reasons") or [])

    summary = {
        "vad_cache": project_rel(vad_cache),
        "frame_scores": project_rel(frame_scores_path),
        "diagnostics": project_rel(diagnostics_path),
        "unsafe_fallback_chunks": project_rel(unsafe_path),
        "output_dir": project_rel(output_dir),
        "frame_hop_s": frame_hop_s,
        "score_frame_hop_s": score_frame_hop_s,
        "old_chunk_count": len(rows),
        "simulated_child_count": total_children,
        "chunk_growth_ratio": (total_children / len(rows)) if rows else 0.0,
        "target_chunk_count": len(target_rows),
        "target_split_count": len(target_split_rows),
        "target_split_ratio": (len(target_split_rows) / len(target_rows)) if target_rows else 0.0,
        "all_split_count": len(split_rows),
        "all_split_ratio": (len(split_rows) / len(rows)) if rows else 0.0,
        "target_old_duration_s": compact_stats([row["old_duration_s"] for row in target_rows]),
        "target_old_core_duration_s": compact_stats([row["old_core_duration_s"] for row in target_rows]),
        "target_max_child_duration_s": compact_stats(
            [row["max_child_duration_s"] for row in target_rows]
        ),
        "target_max_child_core_duration_s": compact_stats(
            [row["max_child_core_duration_s"] for row in target_rows]
        ),
        "all_max_child_duration_s": compact_stats([row["max_child_duration_s"] for row in rows]),
        "all_max_child_core_duration_s": compact_stats(
            [row["max_child_core_duration_s"] for row in rows]
        ),
        "old_split_reason_counts": dict(
            Counter(str(row["old_split_reason"]) for row in rows).most_common()
        ),
        "old_split_policy_counts": dict(
            Counter(str(row["old_split_policy"]) for row in rows).most_common()
        ),
        "risk_reason_counts": dict(risk_reason_counts.most_common()),
        "config": {
            "window_frames": args.window_frames,
            "reserve_frames": args.reserve_frames,
            "target_padding_frames": args.target_padding_frames,
            "gap_merge_frames": args.gap_merge_frames,
            "min_core_frames": args.min_core_frames,
            "target_core_frames": args.target_core_frames,
            "safe_core_frames": args.safe_core_frames,
            "min_gap_frames": args.min_gap_frames,
            "min_child_frames": args.min_child_frames,
            "max_children": args.max_children,
            "risk_threshold": args.risk_threshold,
            "continuous_threshold": args.continuous_threshold,
            "valley_threshold": args.valley_threshold,
            "cut_threshold": args.cut_threshold,
            "min_cut_frames": args.min_cut_frames,
            "min_valley_frames": args.min_valley_frames,
        },
        "note": (
            "Offline estimate only. This starts from existing processing_spans "
            "and simulates residual risk splitting; it does not rerun ASR or alignment."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "residual_split_plan.jsonl", rows)
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Residual Cut Split Analysis",
        "",
        f"- vad cache: `{summary['vad_cache']}`",
        f"- frame scores: `{summary['frame_scores']}`",
        f"- target chunks: {summary['target_chunk_count']}",
        f"- target split: {summary['target_split_count']} ({summary['target_split_ratio']:.3f})",
        f"- chunk growth: {summary['chunk_growth_ratio']:.3f}x",
        "",
        "## Target Chunks",
        "",
        f"- old duration: `{summary['target_old_duration_s']}`",
        f"- old core duration: `{summary['target_old_core_duration_s']}`",
        f"- max child duration: `{summary['target_max_child_duration_s']}`",
        f"- max child core duration: `{summary['target_max_child_core_duration_s']}`",
        "",
        "## All Chunks",
        "",
        f"- max child duration: `{summary['all_max_child_duration_s']}`",
        f"- max child core duration: `{summary['all_max_child_core_duration_s']}`",
        f"- old split reasons: `{summary['old_split_reason_counts']}`",
        f"- old split policies: `{summary['old_split_policy_counts']}`",
        f"- risk reasons: `{summary['risk_reason_counts']}`",
        "",
        "> Offline estimate only. This does not rerun ASR or forced alignment.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate residual risk splitting from existing processing_spans. "
            "Use this after a workflow has already applied cut/valley packing."
        )
    )
    parser.add_argument("--vad-cache", required=True)
    parser.add_argument("--frame-scores", required=True)
    parser.add_argument("--diagnostics", default="")
    parser.add_argument("--unsafe-fallback-chunks", default="")
    parser.add_argument("--output-dir", default="agents/temp/fusionvad-ja/residual-cut-split-analysis")
    parser.add_argument("--window-frames", type=int, default=899)
    parser.add_argument("--reserve-frames", type=int, default=45)
    parser.add_argument("--target-padding-frames", type=int, default=60)
    parser.add_argument("--gap-merge-frames", type=int, default=45)
    parser.add_argument("--min-core-frames", type=int, default=420)
    parser.add_argument("--target-core-frames", type=int, default=270)
    parser.add_argument("--safe-core-frames", type=int, default=360)
    parser.add_argument("--min-gap-frames", type=int, default=6)
    parser.add_argument("--min-child-frames", type=int, default=45)
    parser.add_argument("--max-children", type=int, default=8)
    parser.add_argument("--risk-threshold", type=float, default=1.0)
    parser.add_argument("--continuous-threshold", type=float, default=2.0)
    parser.add_argument("--valley-threshold", type=float, default=0.20)
    parser.add_argument("--cut-threshold", type=float, default=0.50)
    parser.add_argument("--min-cut-frames", type=int, default=3)
    parser.add_argument("--min-valley-frames", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = analyze(parse_args(argv))
    print(f"summary={summary['output_dir']}/summary.json")
    print(
        "target_split={}/{} growth={:.3f} max_child_p90={:.3f}s".format(
            summary["target_split_count"],
            summary["target_chunk_count"],
            summary["chunk_growth_ratio"],
            summary["target_max_child_duration_s"]["p90"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
