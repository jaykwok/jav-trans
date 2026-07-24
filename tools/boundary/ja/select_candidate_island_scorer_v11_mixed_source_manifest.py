#!/usr/bin/env python3
"""Freeze a one-source-per-video real train manifest for mixed-source review.

This selector is deliberately evidence-free: it never creates labels and never
promotes an Omni/Teacher result to truth.  It only chooses complete train source
windows for a subsequent independent dual-evidence Teacher run.  Existing
selected sources may be supplied through ``--include-manifest``; new sources are
chosen deterministically from videos not already represented, preferring source
windows with more upstream candidate events.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import wave
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_ID = "boundary_acoustic_binary_v12"
SOURCE_SCHEMA = "candidate_island_scorer_v11_train_teacher_source_v1"
SOURCE_WINDOW_SCHEMA = "joint_boundary_omni_source_window_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_mixed_source_manifest_summary_v1"
STATS_SCHEMA = "candidate_island_scorer_v11_mixed_source_manifest_stats_v1"
FRAME_HOP_S = 0.02
FRAME_SAMPLES = 320
SAMPLE_RATE = 16000


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve(value: str | Path, *, owner: Path | None = None) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    candidates = []
    if owner is not None:
        candidates.append(owner.parent / raw)
    candidates.append(PROJECT_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _index(rows: Sequence[Mapping[str, Any]], *, key: str, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in result:
            raise ValueError(f"{name} requires unique non-empty {key}: {value!r}")
        result[value] = dict(row)
    return result


def _audio_geometry(path: Path) -> tuple[int, int, float]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        samples = int(handle.getnframes())
    if sample_rate != SAMPLE_RATE or channels != 1 or samples <= 0:
        raise ValueError(
            f"source must be non-empty mono 16k WAV: {path} "
            f"rate={sample_rate} channels={channels} samples={samples}"
        )
    return sample_rate, samples, samples / SAMPLE_RATE


def _validate_source(
    row: Mapping[str, Any],
    *,
    source_manifest: Path,
    source_windows: Mapping[str, Mapping[str, Any]],
    require_audio_sha: bool = True,
) -> dict[str, Any]:
    source_id = str(row.get("source_id") or "")
    if row.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"wrong train teacher source schema: {source_id}")
    if row.get("boundary_serialization_contract_id") != CONTRACT_ID:
        raise ValueError(f"wrong central contract: {source_id}")
    if row.get("partition") != "train":
        raise ValueError(f"mixed source manifest accepts train only: {source_id}")
    video_id = str(row.get("video_id") or "")
    if not source_id or not video_id:
        raise ValueError(f"source/video identity is missing: {source_id!r}")
    window = source_windows.get(source_id)
    if window is None:
        raise ValueError(f"source manifest is missing source window: {source_id}")
    if window.get("schema") != SOURCE_WINDOW_SCHEMA:
        raise ValueError(f"wrong source-window schema: {source_id}")
    if str(window.get("window_id") or "") != source_id:
        raise ValueError(f"source/window identity mismatch: {source_id}")
    if str(window.get("video_id") or "") != video_id:
        raise ValueError(f"source/window video mismatch: {source_id}")
    audio = _resolve(str(row.get("audio") or ""), owner=source_manifest)
    if not audio.is_file():
        raise FileNotFoundError(audio)
    sample_rate, samples, duration_s = _audio_geometry(audio)
    expected_sha = str(row.get("audio_sha256") or "")
    actual_sha = _sha256(audio)
    if require_audio_sha and (len(expected_sha) != 64 or expected_sha != actual_sha):
        raise ValueError(f"source audio SHA mismatch: {source_id}")
    frame_count = int(row.get("frame_count") or 0)
    expected_frames = samples // FRAME_SAMPLES
    if frame_count != expected_frames or frame_count <= 0:
        raise ValueError(
            f"source frame geometry mismatch: {source_id} "
            f"manifest={frame_count} audio={expected_frames}"
        )
    if abs(float(row.get("duration_s") or 0.0) - duration_s) > 1e-6:
        raise ValueError(f"source duration mismatch: {source_id}")
    return {
        "schema": STATS_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "source_id": source_id,
        "video_id": video_id,
        "audio": _display(audio),
        "audio_sha256": actual_sha,
        "sample_rate": sample_rate,
        "sample_count": samples,
        "duration_s": duration_s,
        "frame_count": frame_count,
        "frame_hop_s": FRAME_HOP_S,
        "candidate_count": int(window.get("candidate_count") or 0),
        "span_count": int(window.get("span_count") or 0),
        "source_start_s": float(window.get("source_start_s") or 0.0),
        "source_end_s": float(window.get("source_end_s") or 0.0),
    }


def _rank_key(stat: Mapping[str, Any]) -> tuple[int, int, float, str]:
    return (
        -int(stat.get("candidate_count") or 0),
        -int(stat.get("span_count") or 0),
        -float(stat.get("duration_s") or 0.0),
        str(stat.get("source_id") or ""),
    )


def select_manifest(
    *,
    source_manifest: Path,
    source_windows: Path,
    include_manifest: Path | None,
    exclude_manifest: Path | None,
    new_video_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    source_windows = source_windows.resolve()
    include_manifest = include_manifest.resolve() if include_manifest else None
    exclude_manifest = exclude_manifest.resolve() if exclude_manifest else None
    for path in (source_manifest, source_windows, include_manifest, exclude_manifest):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if new_video_count < 0:
        raise ValueError("new_video_count must be non-negative")

    source_rows = _index(_rows(source_manifest), key="source_id", name="train source manifest")
    window_rows = _index(_rows(source_windows), key="window_id", name="source windows")
    include_ids: list[str] = []
    if include_manifest is not None:
        include_ids = [str(row.get("source_id") or "") for row in _rows(include_manifest)]
        if any(not value for value in include_ids):
            raise ValueError("include manifest contains an empty source_id")
    if len(include_ids) != len(set(include_ids)):
        raise ValueError("include manifest contains duplicate source IDs")
    missing_include = sorted(set(include_ids) - set(source_rows))
    if missing_include:
        raise ValueError(f"included sources are absent from source manifest: {missing_include[:5]}")

    excluded_video_ids: set[str] = set()
    if exclude_manifest is not None:
        for row in _rows(exclude_manifest):
            video_id = str(row.get("video_id") or "")
            if video_id:
                excluded_video_ids.add(video_id)

    selected_stats: list[dict[str, Any]] = []
    selected_video_ids: set[str] = set()
    for source_id in include_ids:
        stat = _validate_source(
            source_rows[source_id], source_manifest=source_manifest, source_windows=window_rows
        )
        if stat["video_id"] in selected_video_ids:
            raise ValueError(f"include manifest must contain one source per video: {stat['video_id']}")
        selected_video_ids.add(stat["video_id"])
        selected_stats.append({**stat, "selection_role": "included_existing_dual_evidence"})

    candidates_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_id, row in source_rows.items():
        if source_id in include_ids:
            continue
        stat = _validate_source(row, source_manifest=source_manifest, source_windows=window_rows)
        if stat["video_id"] in selected_video_ids or stat["video_id"] in excluded_video_ids:
            continue
        candidates_by_video[stat["video_id"]].append(stat)

    best_by_video = {
        video_id: sorted(rows, key=_rank_key)[0]
        for video_id, rows in candidates_by_video.items()
    }
    ordered_videos = sorted(
        best_by_video,
        key=lambda video_id: (_rank_key(best_by_video[video_id]), video_id),
    )
    if new_video_count > len(ordered_videos):
        raise ValueError(
            f"requested {new_video_count} new videos but only {len(ordered_videos)} are available"
        )
    chosen_videos = ordered_videos[:new_video_count]
    for video_id in chosen_videos:
        selected_stats.append({**best_by_video[video_id], "selection_role": "new_candidate_by_upstream_count"})

    if not selected_stats:
        raise ValueError("mixed source manifest cannot be empty")
    if len({row["source_id"] for row in selected_stats}) != len(selected_stats):
        raise ValueError("mixed source manifest contains duplicate source IDs")
    if len({row["video_id"] for row in selected_stats}) != len(selected_stats):
        raise ValueError("mixed source manifest contains duplicate video IDs")

    selected_ids = {row["source_id"] for row in selected_stats}
    output_rows = [dict(source_rows[source_id]) for source_id in sorted(selected_ids)]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "mixed_source_manifest.jsonl"
    _write_jsonl(manifest_path, output_rows)
    stats_path = output_dir / "selection_stats.jsonl"
    _write_jsonl(
        stats_path,
        sorted(selected_stats, key=lambda row: (str(row["video_id"]), str(row["source_id"]))),
    )
    candidate_pool_path = output_dir / "candidate_pool_best_by_video.jsonl"
    _write_jsonl(
        candidate_pool_path,
        [best_by_video[video_id] for video_id in ordered_videos],
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "source_manifest": _display(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_windows": _display(source_windows),
        "source_windows_sha256": _sha256(source_windows),
        "include_manifest": _display(include_manifest) if include_manifest else None,
        "include_manifest_sha256": _sha256(include_manifest) if include_manifest else None,
        "exclude_manifest": _display(exclude_manifest) if exclude_manifest else None,
        "exclude_manifest_sha256": _sha256(exclude_manifest) if exclude_manifest else None,
        "selection_policy": "retain_included_sources_then_one_new_source_per_video_ranked_by_candidate_count_v1",
        "new_video_count_requested": new_video_count,
        "new_video_count_selected": len(chosen_videos),
        "included_source_count": len(include_ids),
        "source_count": len(output_rows),
        "video_count": len(selected_stats),
        "selected_source_ids": [row["source_id"] for row in sorted(selected_stats, key=lambda row: row["source_id"])],
        "selected_video_ids": [row["video_id"] for row in sorted(selected_stats, key=lambda row: row["video_id"])],
        "new_selected_source_ids": [best_by_video[video_id]["source_id"] for video_id in chosen_videos],
        "candidate_video_pool_count": len(ordered_videos),
        "candidate_pool_best_by_video": _display(candidate_pool_path),
        "candidate_pool_best_by_video_sha256": _sha256(candidate_pool_path),
        "selection_stats": _display(stats_path),
        "selection_stats_sha256": _sha256(stats_path),
        "mixed_source_manifest": _display(manifest_path),
        "mixed_source_manifest_sha256": _sha256(manifest_path),
        "source_stats_schema": STATS_SCHEMA,
        "one_source_per_video": True,
        "train_only": True,
        "teacher_output_used_as_truth": False,
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--include-manifest")
    parser.add_argument("--exclude-manifest")
    parser.add_argument("--new-video-count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    select_manifest(
        source_manifest=Path(args.source_manifest),
        source_windows=Path(args.source_windows),
        include_manifest=Path(args.include_manifest) if args.include_manifest else None,
        exclude_manifest=Path(args.exclude_manifest) if args.exclude_manifest else None,
        new_video_count=args.new_video_count,
        output_dir=Path(args.output_dir),
    )
