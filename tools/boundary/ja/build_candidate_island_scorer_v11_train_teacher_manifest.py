#!/usr/bin/env python3
"""Build a train-only real-workflow source manifest for Scorer v11 teachers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


SCHEMA = "candidate_island_scorer_v11_train_teacher_source_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_train_teacher_manifest_summary_v1"
SOURCE_SCHEMA = "joint_boundary_omni_source_window_v1"
PARTITION_SCHEMA = "candidate_island_scorer_v11_partition_manifest_v1"
SAMPLE_RATE = 16000
FRAME_HOP_S = 0.02
FRAME_SAMPLES = 320


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str, *, owner: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [owner.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def _index(rows: list[dict[str, Any]], key: str, *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in result:
            raise ValueError(f"{name} has missing or duplicate {key}: {value!r}")
        result[value] = row
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_windows = Path(args.source_windows).resolve()
    partition_manifest = Path(args.partition_manifest).resolve()
    for path in (source_windows, partition_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    sources = _index(_rows(source_windows), "window_id", name="source windows")
    partitions = _index(_rows(partition_manifest), "source_id", name="partition manifest")
    output_rows: list[dict[str, Any]] = []
    video_counts: Counter[str] = Counter()
    for source_id, partition in partitions.items():
        if partition.get("schema") != PARTITION_SCHEMA:
            raise ValueError(f"wrong Scorer v11 partition schema: {source_id}")
        if partition.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"wrong central Boundary contract: {source_id}")
        if partition.get("partition") != "train":
            continue
        source = sources.get(source_id)
        if source is None or source.get("schema") != SOURCE_SCHEMA:
            raise ValueError(f"train partition source is missing or has wrong schema: {source_id}")
        video_id = str(source.get("video_id") or "")
        if video_id != str(partition.get("video_id") or ""):
            raise ValueError(f"source/partition video identity mismatch: {source_id}")
        audio = _resolve(str(source.get("audio_wav") or ""), owner=source_windows)
        info = sf.info(audio)
        if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
            raise ValueError(f"teacher source must be non-empty mono 16k WAV: {source_id}")
        frame_count = int(info.frames) // FRAME_SAMPLES
        if frame_count <= 0:
            raise ValueError(f"teacher source has no complete 20ms frame: {source_id}")
        output_rows.append(
            {
                "schema": SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "input_distribution": "real_workflow_source_window",
                "audio": str(audio),
                "audio_sha256": _sha256(audio),
                "sample_rate": SAMPLE_RATE,
                "sample_count": int(info.frames),
                "duration_s": int(info.frames) / SAMPLE_RATE,
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "teacher_only": True,
                "training_manifest_allowed": False,
            }
        )
        video_counts[video_id] += 1
    if not output_rows:
        raise ValueError("partition manifest contains no train sources")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "train_teacher_sources.jsonl"
    _write_jsonl(manifest, output_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "source_windows": str(source_windows),
        "source_windows_sha256": _sha256(source_windows),
        "partition_manifest": str(partition_manifest),
        "partition_manifest_sha256": _sha256(partition_manifest),
        "source_count": len(output_rows),
        "video_count": len(video_counts),
        "video_source_counts": dict(sorted(video_counts.items())),
        "duration_s_total": sum(float(row["duration_s"]) for row in output_rows),
        "train_teacher_sources": str(manifest),
        "train_teacher_sources_sha256": _sha256(manifest),
        "heldout_audio_used": False,
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
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
