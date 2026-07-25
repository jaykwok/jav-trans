#!/usr/bin/env python3
"""Freeze the identity-only 25-source Scorer v12 Teacher pilot manifest.

This tool may reuse v11 artifacts only as lists of source identities.  It
re-resolves audio and partition metadata from the frozen full-source dataset
and never copies v11 labels, spans, ASR text, or semantic classifications.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import wave
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA,
)


CONTRACT_ID = "boundary_acoustic_binary_v12"
FRAME_HOP_S = 0.02
FRAME_SAMPLES = 320
MAX_SHORT_SOURCE_DRIFT_S = 1.0
PARTITION_SCHEMA = "candidate_island_scorer_v11_partition_manifest_v1"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_pilot_manifest_summary_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(
    rows: Sequence[Mapping[str, Any]], key: str, *, name: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in result:
            raise ValueError(f"{name} requires unique non-empty {key}: {value!r}")
        result[value] = dict(row)
    return result


def _source_id(row: Mapping[str, Any]) -> str:
    return str(row.get("source_id") or row.get("window_id") or "")


def _resolve_audio(source: Mapping[str, Any], *, source_windows: Path) -> Path:
    value = str(source.get("audio_wav") or source.get("audio") or "")
    if not value:
        raise ValueError(f"source window has no audio path: {_source_id(source)}")
    raw = Path(value)
    candidates = (
        (raw,)
        if raw.is_absolute()
        else (source_windows.parent / raw, PROJECT_ROOT / raw)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[0])


def _audio_geometry(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as handle:
        channels = int(handle.getnchannels())
        sample_rate = int(handle.getframerate())
        sample_count = int(handle.getnframes())
        sample_width = int(handle.getsampwidth())
    if channels != 1 or sample_rate != 16000:
        raise ValueError(
            f"v12 source audio must be mono 16 kHz: {path} "
            f"(channels={channels}, sample_rate={sample_rate})"
        )
    if sample_count <= 0:
        raise ValueError(f"v12 source audio is empty: {path}")
    duration_s = sample_count / sample_rate
    frame_count = (sample_count + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    if frame_count <= 0 or abs(frame_count * FRAME_HOP_S - duration_s) > FRAME_HOP_S:
        raise ValueError(f"v12 source frame geometry is invalid: {path}")
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "sample_width_bytes": sample_width,
        "duration_s": duration_s,
        "frame_count": frame_count,
    }


def _ranked_ids(
    values: Sequence[str], *, seed: int, namespace: str
) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"vocal-envelope-v12-pilot\0{seed}\0{namespace}\0{value}".encode(
                "utf-8"
            )
        ).hexdigest(),
    )


def _validated_audio_identity(
    source_id: str,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    source_windows: Path,
) -> tuple[Path, dict[str, int | float], str, float]:
    source = sources.get(source_id)
    if source is None:
        raise ValueError(f"v12 source identity is absent from frozen sources: {source_id}")
    audio = _resolve_audio(source, source_windows=source_windows)
    geometry = _audio_geometry(audio)
    actual_sha = _sha256(audio)
    declared_sha = str(
        source.get("audio_wav_sha256") or source.get("audio_sha256") or ""
    )
    if not declared_sha or actual_sha != declared_sha:
        raise ValueError(f"v12 source audio SHA mismatch: {source_id}")
    declared_duration = float(source.get("duration_s") or 0.0)
    actual_duration = float(geometry["duration_s"])
    duration_delta = actual_duration - declared_duration
    if declared_duration <= 0.0:
        raise ValueError(f"v12 source manifest duration is invalid: {source_id}")
    if duration_delta > FRAME_HOP_S or duration_delta < -MAX_SHORT_SOURCE_DRIFT_S:
        raise ValueError(
            f"v12 source audio duration mismatch: {source_id} "
            f"(declared={declared_duration:.6f}, actual={actual_duration:.6f})"
        )
    return audio, geometry, actual_sha, duration_delta


def build_pilot_manifest(
    *,
    source_windows: Path,
    partition_manifest: Path,
    output_dir: Path,
    train_count: int = 13,
    heldout_count: int = 12,
    seed: int = 117,
) -> dict[str, Any]:
    source_windows = source_windows.resolve()
    partition_manifest = partition_manifest.resolve()
    if train_count <= 0 or heldout_count <= 0:
        raise ValueError("v12 pilot train/heldout counts must be positive")

    source_rows = _jsonl(source_windows)
    sources = _index(
        ({**row, "source_id": _source_id(row)} for row in source_rows),
        "source_id",
        name="full-source windows",
    )
    partition_rows = _jsonl(partition_manifest)
    partitions = _index(partition_rows, "source_id", name="partition manifest")
    for row in partition_rows:
        if row.get("schema") != PARTITION_SCHEMA:
            raise ValueError("unexpected frozen partition schema")
        if row.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError("frozen partition central contract mismatch")

    by_partition_video: dict[str, dict[str, list[str]]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    for source_id, row in partitions.items():
        partition = str(row.get("partition") or "")
        video_id = str(row.get("video_id") or "")
        if partition not in by_partition_video or not video_id:
            raise ValueError(f"invalid frozen source partition identity: {source_id}")
        by_partition_video[partition].setdefault(video_id, []).append(source_id)

    rejected_identities: list[dict[str, str]] = []
    valid_by_partition_video: dict[str, dict[str, list[str]]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    for partition, videos in by_partition_video.items():
        for video_id, source_ids in videos.items():
            valid: list[str] = []
            for source_id in _ranked_ids(
                source_ids,
                seed=seed,
                namespace=f"{partition}:{video_id}:source",
            ):
                try:
                    _validated_audio_identity(
                        source_id,
                        sources=sources,
                        source_windows=source_windows,
                    )
                except (FileNotFoundError, ValueError, wave.Error) as error:
                    rejected_identities.append(
                        {
                            "source_id": source_id,
                            "partition": partition,
                            "reason": str(error),
                        }
                    )
                    continue
                valid.append(source_id)
            if valid:
                valid_by_partition_video[partition][video_id] = valid

    ranked_train_videos = _ranked_ids(
        list(valid_by_partition_video["train"]),
        seed=seed,
        namespace="train-video",
    )
    if len(ranked_train_videos) < train_count:
        raise ValueError(
            "not enough audio-valid frozen train videos for v12 pilot; "
            f"rejected={rejected_identities}"
        )
    selected_train = [
        valid_by_partition_video["train"][video_id][0]
        for video_id in ranked_train_videos[:train_count]
    ]

    heldout_available = {
        partition: sum(
            len(source_ids)
            for source_ids in valid_by_partition_video[partition].values()
        )
        for partition in ("val", "test")
    }
    heldout_total_available = sum(heldout_available.values())
    if heldout_total_available < heldout_count:
        raise ValueError("not enough audio-valid frozen held-out identities for v12 pilot")
    val_count = int(round(heldout_count * heldout_available["val"] / heldout_total_available))
    val_count = max(1, min(heldout_count - 1, val_count))
    heldout_quotas = {"val": val_count, "test": heldout_count - val_count}
    selected_heldout: list[str] = []
    for partition in ("val", "test"):
        quota = heldout_quotas[partition]
        videos = _ranked_ids(
            list(valid_by_partition_video[partition]),
            seed=seed,
            namespace=f"{partition}-video",
        )
        chosen: list[str] = []
        depth = 0
        while len(chosen) < quota:
            added = False
            for video_id in videos:
                candidates = valid_by_partition_video[partition][video_id]
                if depth < len(candidates):
                    chosen.append(candidates[depth])
                    added = True
                    if len(chosen) == quota:
                        break
            if not added:
                break
            depth += 1
        if len(chosen) != quota:
            raise ValueError(f"not enough balanced {partition} sources for v12 pilot")
        selected_heldout.extend(chosen)

    heldout_candidates = selected_heldout
    selected_ids = [*selected_train, *heldout_candidates]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("v12 pilot train and held-out identities overlap")

    output_rows: list[dict[str, Any]] = []
    video_partitions: dict[str, str] = {}
    core_ids: set[str] = set()
    partition_counts: Counter[str] = Counter()
    train_videos: set[str] = set()
    for source_id in selected_ids:
        source = sources.get(source_id)
        partition_row = partitions.get(source_id)
        if source is None or partition_row is None:
            raise ValueError(f"v12 pilot identity is absent from frozen sources: {source_id}")
        partition = str(partition_row.get("partition") or "")
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"invalid v12 pilot partition: {source_id}")
        if source_id in selected_train and partition != "train":
            raise ValueError(f"train identity is not in train partition: {source_id}")
        if source_id in heldout_candidates and partition not in {"val", "test"}:
            raise ValueError(f"held-out identity is not val/test: {source_id}")
        video_id = str(partition_row.get("video_id") or source.get("video_id") or "")
        if not video_id or video_id != str(source.get("video_id") or ""):
            raise ValueError(f"v12 pilot video identity mismatch: {source_id}")
        previous = video_partitions.setdefault(video_id, partition)
        if previous != partition:
            raise ValueError(f"v12 pilot video crosses partitions: {video_id}")
        if partition == "train" and video_id in train_videos:
            raise ValueError(f"v12 pilot train selection reuses a video: {video_id}")
        if partition == "train":
            train_videos.add(video_id)
        core_id = source_id
        if core_id in core_ids:
            raise ValueError(f"v12 pilot core is reused: {core_id}")
        core_ids.add(core_id)

        audio, geometry, actual_sha, duration_delta = _validated_audio_identity(
            source_id,
            sources=sources,
            source_windows=source_windows,
        )

        partition_counts[partition] += 1
        output_rows.append(
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "source_id": source_id,
                "video_id": video_id,
                "partition": partition,
                "core_ids": [core_id],
                "audio": str(audio),
                "audio_sha256": actual_sha,
                "duration_s": float(geometry["duration_s"]),
                "source_manifest_duration_s": float(source.get("duration_s") or 0.0),
                "audio_duration_delta_s": round(duration_delta, 9),
                "frame_count": int(geometry["frame_count"]),
                "frame_hop_s": FRAME_HOP_S,
                "sample_rate": int(geometry["sample_rate"]),
                "sample_count": int(geometry["sample_count"]),
                "channels": int(geometry["channels"]),
                "sample_width_bytes": int(geometry["sample_width_bytes"]),
                "source_kind": "real_full_source",
                "synthetic_composite": False,
                "pilot_role": "train" if partition == "train" else "heldout",
                "identity_source": "frozen_full_source_and_partition_manifests",
                "identity_reused_from_v11_only": False,
                "v11_truth_inherited": False,
                "v11_span_inherited": False,
                "asr_text_used_as_truth": False,
                "training_manifest_allowed": False,
            }
        )

    if set(partition_counts) != {"train", "val", "test"}:
        raise ValueError("v12 pilot must contain train, val, and test partitions")
    if partition_counts["train"] != train_count:
        raise ValueError("v12 pilot train count mismatch")
    if partition_counts["val"] + partition_counts["test"] != heldout_count:
        raise ValueError("v12 pilot held-out count mismatch")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "source_manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in output_rows
        ),
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_count": len(output_rows),
        "partition_counts": dict(sorted(partition_counts.items())),
        "video_count": len(video_partitions),
        "core_count": len(core_ids),
        "seed": seed,
        "selection_policy": "frozen_partition_identity_only_video_balanced_sha256_v1",
        "selected_train_source_ids": selected_train,
        "selected_heldout_source_ids": heldout_candidates,
        "heldout_partition_quotas": heldout_quotas,
        "rejected_identities": rejected_identities,
        "source_windows": str(source_windows),
        "source_windows_sha256": _sha256(source_windows),
        "partition_manifest": str(partition_manifest),
        "partition_manifest_sha256": _sha256(partition_manifest),
        "identity_selection_uses_labels": False,
        "v11_truth_inherited": False,
        "v11_span_inherited": False,
        "asr_text_used_as_truth": False,
        "training_manifest_allowed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=13)
    parser.add_argument("--heldout-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=117)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_pilot_manifest(
                source_windows=Path(args.source_windows),
                partition_manifest=Path(args.partition_manifest),
                output_dir=Path(args.output_dir),
                train_count=args.train_count,
                heldout_count=args.heldout_count,
                seed=args.seed,
            ),
            ensure_ascii=False,
        )
    )
