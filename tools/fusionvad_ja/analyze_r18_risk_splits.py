#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.chunk_packer import PackedChunk, pack_vad_segments  # noqa: E402
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
        return {"count": 0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "p50": round(quantile(values, 0.5), 6),
        "p90": round(quantile(values, 0.9), 6),
        "p95": round(quantile(values, 0.95), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
    }


def overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def normalize_segments(raw_segments: Any) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    for item in raw_segments or []:
        if not isinstance(item, dict):
            continue
        start = row_float(item, "start")
        end = row_float(item, "end", start)
        if end <= start:
            continue
        score = item.get("score")
        segments.append(
            SpeechSegment(
                start=start,
                end=end,
                score=None if score is None else float(score),
            )
        )
    return sorted(segments, key=lambda item: (item.start, item.end))


def normalize_chunks(raw_chunks: Any) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index, item in enumerate(raw_chunks or []):
        if not isinstance(item, dict):
            continue
        start = row_float(item, "start")
        end = row_float(item, "end", start)
        vad_segments = normalize_segments(item.get("vad_segments") or [])
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
        chunks.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "duration_s": max(0.0, end - start),
                "core_start": core_start,
                "core_end": core_end,
                "core_duration_s": max(0.0, core_end - core_start),
                "split_reason": str(item.get("split_reason") or ""),
                "split_policy": str(item.get("split_policy") or ""),
                "vad_segments": vad_segments,
                "vad_seg_count": len(vad_segments),
                "internal_gap_count": row_int(item, "internal_gap_count"),
                "internal_gap_max_s": row_float(item, "internal_gap_max_s"),
            }
        )
    return chunks


def chunk_to_row(index: int, chunk: PackedChunk) -> dict[str, Any]:
    core_start = chunk.core_start if chunk.core_start is not None else chunk.start
    core_end = chunk.core_end if chunk.core_end is not None else chunk.end
    return {
        "index": index,
        "start": round(chunk.start, 6),
        "end": round(chunk.end, 6),
        "duration_s": round(max(0.0, chunk.end - chunk.start), 6),
        "core_start": round(core_start, 6),
        "core_end": round(core_end, 6),
        "core_duration_s": round(max(0.0, core_end - core_start), 6),
        "split_reason": chunk.split_reason,
        "split_policy": chunk.split_policy,
        "parent_chunk_id": chunk.parent_chunk_id,
        "island_id": chunk.island_id,
        "island_count": chunk.island_count,
        "vad_seg_count": len(chunk.vad_segments),
        "internal_gap_count": chunk.internal_gap_count,
        "internal_gap_max_s": round(chunk.internal_gap_max_s, 6),
        "risk_split_count": chunk.risk_split_count,
        "risk_score": chunk.risk_score,
        "risk_reasons": list(chunk.risk_reasons),
        "cut_score_max": chunk.cut_score_max,
        "valley_score_min": chunk.valley_score_min,
    }


def scores_from_payload(payload: dict[str, Any], keys: Sequence[str]) -> list[float] | None:
    for key in keys:
        raw = payload.get(key)
        if raw is not None:
            return [float(value) for value in raw]
    return None


def base_chunk_config(cache_payload: dict[str, Any]) -> dict[str, Any]:
    runtime = cache_payload.get("runtime_vad_signature", {}).get("chunk_packing") or {}
    signature_chunk = cache_payload.get("signature", {}).get("chunk") or {}

    def cfg_float(runtime_key: str, signature_key: str, default: float) -> float:
        value = runtime.get(runtime_key)
        if value is None:
            value = signature_chunk.get(signature_key)
        try:
            return float(value if value is not None else default)
        except (TypeError, ValueError):
            return default

    def cfg_int(runtime_key: str, signature_key: str, default: int) -> int:
        value = runtime.get(runtime_key)
        if value is None:
            value = signature_chunk.get(signature_key)
        try:
            return int(value if value is not None else default)
        except (TypeError, ValueError):
            return default

    return {
        "frame_hop_s": cfg_float("frame_hop_s", "pack_frame_hop_s", 1.0 / 29.97),
        "window_frames": cfg_int("window_frames", "pack_window_frames", 899),
        "reserve_frames": cfg_int("reserve_frames", "pack_reserve_frames", 45),
        "target_padding_frames": cfg_int(
            "target_padding_frames",
            "pack_target_padding_frames",
            60,
        ),
        "gap_merge_frames": cfg_int("gap_merge_frames", "pack_gap_merge_frames", 45),
        "max_core_frames": cfg_int("max_core_frames", "pack_max_core_frames", 0),
        "pre_asr_island_split_enabled": False,
        "pre_asr_valley_split_enabled": False,
        "pre_asr_cut_split_enabled": False,
    }


def threshold_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    thresholds = [8.0, 12.0, 14.0, 20.0, 28.0]
    return {
        f"gt_{threshold:g}s": sum(1 for row in rows if row_float(row, key) > threshold)
        for threshold in thresholds
    }


def count_counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows).most_common())


def diagnostic_chunk_index(row: dict[str, Any]) -> int | None:
    value = row.get("chunk_index")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def sentinel_indices(diagnostics: list[dict[str, Any]]) -> set[int]:
    indices: set[int] = set()
    for row in diagnostics:
        if str(row.get("fallback_subtype") or "") != "vad_coarse_after_sentinel":
            continue
        index = diagnostic_chunk_index(row)
        if index is not None:
            indices.add(index)
    return indices


def overlap_new_chunks(
    old_chunk: dict[str, Any],
    new_rows: list[dict[str, Any]],
    *,
    min_overlap_s: float,
) -> list[dict[str, Any]]:
    old_start = row_float(old_chunk, "core_start", row_float(old_chunk, "start"))
    old_end = row_float(old_chunk, "core_end", row_float(old_chunk, "end", old_start))
    matches: list[dict[str, Any]] = []
    for row in new_rows:
        new_start = row_float(row, "core_start", row_float(row, "start"))
        new_end = row_float(row, "core_end", row_float(row, "end", new_start))
        overlap = overlap_s(old_start, old_end, new_start, new_end)
        if overlap >= min_overlap_s:
            enriched = dict(row)
            enriched["_overlap_s"] = round(overlap, 6)
            matches.append(enriched)
    return matches


def build_plan_rows(
    original_chunks: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    risky_indices: set[int],
    *,
    min_overlap_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for old in original_chunks:
        matches = overlap_new_chunks(old, new_rows, min_overlap_s=min_overlap_s)
        max_child_duration = max(
            [row_float(match, "duration_s") for match in matches] or [0.0]
        )
        max_child_core_duration = max(
            [row_float(match, "core_duration_s") for match in matches] or [0.0]
        )
        rows.append(
            {
                "old_chunk_index": old["index"],
                "old_start": round(row_float(old, "start"), 6),
                "old_end": round(row_float(old, "end"), 6),
                "old_duration_s": round(row_float(old, "duration_s"), 6),
                "old_core_duration_s": round(row_float(old, "core_duration_s"), 6),
                "old_split_reason": old["split_reason"],
                "old_vad_seg_count": old["vad_seg_count"],
                "old_internal_gap_max_s": round(row_float(old, "internal_gap_max_s"), 6),
                "sentinel_risk": old["index"] in risky_indices,
                "new_overlap_chunk_count": len(matches),
                "new_max_child_duration_s": round(max_child_duration, 6),
                "new_max_child_core_duration_s": round(max_child_core_duration, 6),
                "new_overlap_chunks": [
                    {
                        "index": match["index"],
                        "start": match["start"],
                        "end": match["end"],
                        "duration_s": match["duration_s"],
                        "core_duration_s": match["core_duration_s"],
                        "split_reason": match["split_reason"],
                        "split_policy": match["split_policy"],
                        "risk_score": match["risk_score"],
                        "risk_reasons": match["risk_reasons"],
                        "overlap_s": match["_overlap_s"],
                    }
                    for match in matches
                ],
            }
        )
    return rows


def summarize(
    *,
    vad_cache: Path,
    frame_scores_path: Path | None,
    diagnostics_path: Path | None,
    original_chunks: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    risky_indices: set[int],
    config: dict[str, Any],
    score_frame_hop_s: float | None,
) -> dict[str, Any]:
    original_durations = [row_float(row, "duration_s") for row in original_chunks]
    original_core_durations = [row_float(row, "core_duration_s") for row in original_chunks]
    new_durations = [row_float(row, "duration_s") for row in new_rows]
    new_core_durations = [row_float(row, "core_duration_s") for row in new_rows]
    new_risk_rows = [row for row in new_rows if row_int(row, "risk_split_count") > 0]
    risky_plan_rows = [row for row in plan_rows if row.get("sentinel_risk")]
    risky_split_rows = [
        row for row in risky_plan_rows if row_int(row, "new_overlap_chunk_count") > 1
    ]
    risky_max_child_durations = [
        row_float(row, "new_max_child_duration_s") for row in risky_plan_rows
    ]
    risky_max_child_core_durations = [
        row_float(row, "new_max_child_core_duration_s") for row in risky_plan_rows
    ]
    risk_reason_counts: Counter[str] = Counter()
    for row in new_risk_rows:
        risk_reason_counts.update(str(value) for value in row.get("risk_reasons") or [])

    return {
        "vad_cache": project_rel(vad_cache),
        "frame_scores": project_rel(frame_scores_path),
        "diagnostics": project_rel(diagnostics_path),
        "output_dir": "",
        "note": (
            "Offline estimate only. It re-packs cached VAD segments and maps old "
            "diagnostic chunks by core overlap; it does not rerun ASR or forced alignment."
        ),
        "has_frame_scores": bool(frame_scores_path),
        "score_frame_hop_s": score_frame_hop_s,
        "config": config,
        "original_chunk_count": len(original_chunks),
        "new_chunk_count": len(new_rows),
        "chunk_growth_ratio": (len(new_rows) / len(original_chunks)) if original_chunks else 0.0,
        "risk_split_child_count": len(new_risk_rows),
        "risk_split_parent_count": len(
            {
                int(row["parent_chunk_id"])
                for row in new_risk_rows
                if row.get("parent_chunk_id") is not None
            }
        ),
        "sentinel_risk_old_chunk_count": len(risky_indices),
        "sentinel_risk_old_split_count": len(risky_split_rows),
        "sentinel_risk_old_split_ratio": (
            len(risky_split_rows) / len(risky_indices) if risky_indices else 0.0
        ),
        "original_duration_s": compact_stats(original_durations),
        "original_core_duration_s": compact_stats(original_core_durations),
        "new_duration_s": compact_stats(new_durations),
        "new_core_duration_s": compact_stats(new_core_durations),
        "sentinel_new_max_child_duration_s": compact_stats(risky_max_child_durations),
        "sentinel_new_max_child_core_duration_s": compact_stats(
            risky_max_child_core_durations
        ),
        "original_duration_threshold_counts": threshold_counts(original_chunks, "duration_s"),
        "new_duration_threshold_counts": threshold_counts(new_rows, "duration_s"),
        "original_core_threshold_counts": threshold_counts(original_chunks, "core_duration_s"),
        "new_core_threshold_counts": threshold_counts(new_rows, "core_duration_s"),
        "original_split_reason_counts": count_counter(original_chunks, "split_reason"),
        "new_split_reason_counts": count_counter(new_rows, "split_reason"),
        "new_split_policy_counts": count_counter(new_rows, "split_policy"),
        "new_risk_reason_counts": dict(risk_reason_counts.most_common()),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# R18 Risk Split Offline Recalculation",
        "",
        f"- vad cache: `{summary['vad_cache']}`",
        f"- frame scores: `{summary['frame_scores']}`",
        f"- diagnostics: `{summary['diagnostics']}`",
        f"- original chunks: {summary['original_chunk_count']}",
        f"- simulated chunks: {summary['new_chunk_count']}",
        f"- growth: {summary['chunk_growth_ratio']:.3f}x",
        f"- risk split parents: {summary['risk_split_parent_count']}",
        f"- risk split children: {summary['risk_split_child_count']}",
        (
            "- sentinel-risk old chunks split: "
            f"{summary['sentinel_risk_old_split_count']}/"
            f"{summary['sentinel_risk_old_chunk_count']} "
            f"({summary['sentinel_risk_old_split_ratio']:.3f})"
        ),
        "",
        "> Offline estimate only: this re-packs cached VAD segments and maps old "
        "diagnostic chunks by core overlap. It does not rerun ASR or forced alignment.",
        "",
        "## Duration",
        "",
        f"- original duration: `{summary['original_duration_s']}`",
        f"- simulated duration: `{summary['new_duration_s']}`",
        f"- original core: `{summary['original_core_duration_s']}`",
        f"- simulated core: `{summary['new_core_duration_s']}`",
        f"- sentinel max child duration: `{summary['sentinel_new_max_child_duration_s']}`",
        f"- sentinel max child core: `{summary['sentinel_new_max_child_core_duration_s']}`",
        "",
        "## Threshold Counts",
        "",
        f"- original duration: `{summary['original_duration_threshold_counts']}`",
        f"- simulated duration: `{summary['new_duration_threshold_counts']}`",
        f"- original core: `{summary['original_core_threshold_counts']}`",
        f"- simulated core: `{summary['new_core_threshold_counts']}`",
        "",
        "## Split Counts",
        "",
        f"- original split reasons: `{summary['original_split_reason_counts']}`",
        f"- simulated split reasons: `{summary['new_split_reason_counts']}`",
        f"- simulated split policies: `{summary['new_split_policy_counts']}`",
        f"- risk reasons: `{summary['new_risk_reason_counts']}`",
        "",
    ]
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    vad_cache = project_path(args.vad_cache)
    diagnostics_path = project_path(args.diagnostics) if args.diagnostics else None
    frame_scores_path = project_path(args.frame_scores) if args.frame_scores else None
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_payload = read_json(vad_cache)
    if not isinstance(cache_payload, dict):
        raise ValueError(f"VAD cache must be a JSON object: {vad_cache}")
    original_chunks = normalize_chunks(cache_payload.get("processing_spans") or [])
    segments = normalize_segments(cache_payload.get("vad_segments") or [])
    diagnostics = read_jsonl(diagnostics_path)
    risky_indices = sentinel_indices(diagnostics)
    config = base_chunk_config(cache_payload)

    frame_scores: list[float] | None = None
    cut_scores: list[float] | None = None
    score_frame_hop_s: float | None = None
    if frame_scores_path:
        scores_payload = read_json(frame_scores_path)
        if not isinstance(scores_payload, dict):
            raise ValueError(f"Frame scores must be a JSON object: {frame_scores_path}")
        frame_scores = scores_from_payload(scores_payload, ("scores", "frame_scores"))
        cut_scores = scores_from_payload(scores_payload, ("cut_scores", "cut_frame_scores"))
        score_frame_hop_s = float(scores_payload.get("frame_hop_s") or config["frame_hop_s"])

    config.update(
        {
            "score_frame_hop_s": score_frame_hop_s,
            "pre_asr_risk_split_enabled": True,
            "pre_asr_risk_split_min_core_frames": args.min_core_frames,
            "pre_asr_risk_split_target_core_frames": args.target_core_frames,
            "pre_asr_risk_split_safe_core_frames": args.safe_core_frames,
            "pre_asr_risk_split_min_gap_frames": args.min_gap_frames,
            "pre_asr_risk_split_min_child_frames": args.min_child_frames,
            "pre_asr_risk_split_max_children": args.max_children,
            "pre_asr_risk_split_threshold": args.risk_threshold,
            "pre_asr_risk_split_continuous_threshold": args.continuous_threshold,
            "pre_asr_risk_split_valley_threshold": args.valley_threshold,
            "pre_asr_risk_split_cut_threshold": args.cut_threshold,
        }
    )
    simulated = pack_vad_segments(
        segments,
        frame_scores=frame_scores,
        cut_frame_scores=cut_scores,
        **config,
    )
    new_rows = [chunk_to_row(index, chunk) for index, chunk in enumerate(simulated)]
    plan_rows = build_plan_rows(
        original_chunks,
        new_rows,
        risky_indices,
        min_overlap_s=args.min_overlap_s,
    )
    summary = summarize(
        vad_cache=vad_cache,
        frame_scores_path=frame_scores_path,
        diagnostics_path=diagnostics_path,
        original_chunks=original_chunks,
        new_rows=new_rows,
        plan_rows=plan_rows,
        risky_indices=risky_indices,
        config=config,
        score_frame_hop_s=score_frame_hop_s,
    )
    summary["output_dir"] = project_rel(output_dir)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "simulated_chunks.jsonl", new_rows)
    write_jsonl(output_dir / "risk_split_plan.jsonl", plan_rows)
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline R18 estimate: re-pack cached VAD segments with risk-aware "
            "pre-ASR splitting and compare chunk distributions."
        )
    )
    parser.add_argument("--vad-cache", required=True)
    parser.add_argument("--diagnostics", default="")
    parser.add_argument("--frame-scores", default="")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "r18-risk-split-offline"),
    )
    parser.add_argument("--min-core-frames", type=int, default=420)
    parser.add_argument("--target-core-frames", type=int, default=270)
    parser.add_argument("--safe-core-frames", type=int, default=360)
    parser.add_argument("--min-gap-frames", type=int, default=6)
    parser.add_argument("--min-child-frames", type=int, default=45)
    parser.add_argument("--max-children", type=int, default=8)
    parser.add_argument("--risk-threshold", type=float, default=1.0)
    parser.add_argument("--continuous-threshold", type=float, default=2.0)
    parser.add_argument("--valley-threshold", type=float, default=0.20)
    parser.add_argument("--cut-threshold", type=float, default=0.94)
    parser.add_argument("--min-overlap-s", type=float, default=0.02)
    args = parser.parse_args(argv)
    for name in (
        "min_core_frames",
        "target_core_frames",
        "safe_core_frames",
        "min_gap_frames",
        "min_child_frames",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_children <= 0:
        parser.error("--max-children must be positive")
    if args.risk_threshold < 0.0:
        parser.error("--risk-threshold must be non-negative")
    if args.continuous_threshold < 0.0:
        parser.error("--continuous-threshold must be non-negative")
    if args.valley_threshold < 0.0:
        parser.error("--valley-threshold must be non-negative")
    if args.cut_threshold < 0.0:
        parser.error("--cut-threshold must be non-negative")
    if args.min_overlap_s < 0.0:
        parser.error("--min-overlap-s must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    summary = analyze(parse_args(argv))
    print(f"summary={summary['output_dir']}/summary.json")
    print(
        "chunks={old}->{new} growth={growth:.3f} sentinel_split={split}/{total}".format(
            old=summary["original_chunk_count"],
            new=summary["new_chunk_count"],
            growth=summary["chunk_growth_ratio"],
            split=summary["sentinel_risk_old_split_count"],
            total=summary["sentinel_risk_old_chunk_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
