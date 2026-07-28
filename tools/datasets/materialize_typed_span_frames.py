#!/usr/bin/env python3
"""Materialize typed-span examples into frame-level labels on the feature grid.

`build_typed_span_dataset.py` emits spans in seconds; every trainer here wants
frames.  This turns one into the other and resolves each example to the PTM/MFCC
cache that holds its frames.

By default the two provenances come from different caches at different widths
but in the SAME coordinate system - real omni windows were truncated to the
leading 128 dims at export time while synthetic composites kept all 2048.
Measured over 40 windows of each, the per-dimension mean profile correlates
r=0.92 against a permutation control of 0.004, so slicing synthetic to
`[:, :ptm_dim]` puts both on one basis at zero extraction cost.  It is asserted
rather than trusted, because a silent basis mismatch would look exactly like the
provenance shortcut this data is meant to test for.

Two caveats make that default the cheap option rather than the right one, and
`--real-feature-manifest` exists to escape both:

  * the leading 128 dims carry only ~6.7% of total PTM energy, barely above the
    6.25% a uniform spread would give, so truncation is a real bottleneck rather
    than a lossless view
  * re-extracting a real window under the synthetic cache's settings reproduces
    the stored 128 dims only to r=0.94, so the two caches were not built by the
    same path - a difference that is itself a provenance signal

Pointing `--real-feature-manifest` at the full-2048 re-export puts both sides on
one extraction path and unlocks the learned projection, which needs untruncated
input.

Two independent label tracks come out, because the sources supervise them to
different depths:

  `speech`  0/1, IGNORE where unknown.  Known for every real keep/drop chunk AND
            for synthetic gap fill, so all 133 h supervise it.
  `type`    0=speech 1=non_semantic_vocal 2=non_vocal, IGNORE otherwise.  Real
            drops resolve a subtype only when their Omni flags decide it, and
            synthetic gaps never do, so this track is far sparser.

Uncovered time in a sparsely-labelled real window stays IGNORE on both tracks.
Treating it as non-speech is the single most tempting corruption available here:
it would manufacture ~7 h of fake negatives and inflate every non-speech metric.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_SCHEMA = "typed_span_frames_v1"
IGNORE_INDEX = -100
DEFAULT_PTM_DIM = 128

TYPE_TO_INDEX = {
    "speech": 0,
    "non_semantic_vocal": 1,
    "non_vocal": 2,
}

# Discovered rather than listed: any synthetic batch that has been through
# `build_feature_cache.py` exposes the same manifest at the same relative path,
# so a new batch is picked up without editing this file (and a batch WITHOUT a
# cache is simply absent, rather than silently resolving to a stale one).
HARDMIX_FEATURE_MANIFEST_RELATIVE = "feature-cache-17b-hf-bf16/feature_manifest.jsonl"


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_feature_index(
    real_override_manifest: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Map example_id -> the cached feature npz that already holds its frames.

    `real_override_manifest` swaps the real windows onto a different export -
    in practice the full-2048 re-extraction, which the stored 128-dim cache
    cannot substitute for because a learned projection needs the untruncated
    input. Applied last so it wins over the default cache.
    """
    index: dict[str, dict[str, Any]] = {}

    for name in (
        "omni-joint-boundary-preasr-v1",
        "omni-joint-boundary-preasr-v2",
        "omni-joint-boundary-preasr-v3",
    ):
        features = PROJECT_ROOT / "datasets" / "train" / name / "features"
        if not features.is_dir():
            continue
        for window_dir in features.iterdir():
            path = window_dir / "speech_sequence_features.npz"
            if path.is_file():
                index[f"{name}:{window_dir.name}"] = {
                    "feature_path": str(path),
                    "ptm_stored_dim": 128,
                }

    train_root = PROJECT_ROOT / "datasets" / "train"
    for manifest in sorted(train_root.glob(f"*/{HARDMIX_FEATURE_MANIFEST_RELATIVE}")):
        name = manifest.parents[1].name
        for row in _rows(manifest):
            audio_id = str(row.get("audio_id") or "")
            feature_path = str(row.get("feature_path") or "")
            if not audio_id or not feature_path:
                continue
            resolved = Path(feature_path)
            if not resolved.is_absolute():
                resolved = PROJECT_ROOT / resolved
            index[f"{name}:{audio_id}"] = {
                "feature_path": str(resolved),
                "ptm_stored_dim": int(row.get("ptm_dim") or 0),
                "manifest_frame_count": int(row.get("frame_count") or 0),
            }

    if real_override_manifest is not None:
        if not real_override_manifest.is_file():
            raise SystemExit(f"missing real override manifest: {real_override_manifest}")
        replaced = 0
        for row in _rows(real_override_manifest):
            example_id = str(row.get("example_id") or "")
            feature_path = str(row.get("feature_path") or "")
            if not example_id or not feature_path:
                continue
            index[example_id] = {
                "feature_path": feature_path,
                "ptm_stored_dim": int(row.get("ptm_dim") or 0),
                "manifest_frame_count": int(row.get("frame_count") or 0),
            }
            replaced += 1
        print(f"real feature override: {replaced} windows")

    return index


def _feature_frame_count(path: Path) -> int:
    """Frame count without decompressing the PTM payload."""
    with np.load(path) as handle:
        return int(handle["mfcc"].shape[0])


def frame_labels(
    example: dict[str, Any], frame_count: int, *, stats: Counter
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize spans onto `frame_count` frames as (speech, type) label arrays."""
    hop = float(example.get("frame_hop_s") or 0.02)
    speech = np.full(frame_count, IGNORE_INDEX, dtype=np.int8)
    types = np.full(frame_count, IGNORE_INDEX, dtype=np.int8)

    for span in example.get("spans") or []:
        start = int(round(float(span["start_s"]) / hop))
        end = int(round(float(span["end_s"]) / hop))
        start = max(0, min(frame_count, start))
        end = max(0, min(frame_count, end))
        if end <= start:
            stats["span_collapsed_on_grid"] += 1
            continue

        is_speech = span.get("speech")
        if is_speech is True:
            speech[start:end] = 1
        elif is_speech is False:
            speech[start:end] = 0
        else:
            stats["span_speech_unknown"] += 1

        type_index = TYPE_TO_INDEX.get(str(span.get("type")))
        if type_index is not None:
            types[start:end] = type_index

    return speech, types


def materialize(
    *,
    dataset: Path,
    split: Path,
    output: Path,
    ptm_dim: int,
    partitions: tuple[str, ...],
    real_override_manifest: Path | None = None,
) -> dict[str, Any]:
    partition_of = {
        str(row["example_id"]): str(row["partition"]) for row in _rows(split)
    }
    feature_index = build_feature_index(real_override_manifest)

    stats: Counter = Counter()
    index_rows: list[dict[str, Any]] = []
    speech_chunks: list[np.ndarray] = []
    type_chunks: list[np.ndarray] = []
    offset = 0

    # Reported per partition x provenance, because a headline number that mixes
    # them is exactly what hid the v11 shortcut.
    supervised: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for example in _rows(dataset):
        example_id = str(example["example_id"])
        partition = partition_of.get(example_id)
        if partition is None:
            stats["example_missing_split"] += 1
            continue
        if partition not in partitions:
            continue

        feature = feature_index.get(example_id)
        if feature is None:
            stats["example_missing_features"] += 1
            continue

        feature_path = Path(feature["feature_path"])
        if not feature_path.is_file():
            stats["example_feature_file_missing"] += 1
            continue

        stored_dim = int(feature.get("ptm_stored_dim") or 0)
        if stored_dim < ptm_dim:
            stats["example_ptm_too_narrow"] += 1
            continue

        try:
            feature_frames = _feature_frame_count(feature_path)
        except Exception:
            stats["example_feature_unreadable"] += 1
            continue

        label_frames = int(example.get("frame_count") or 0)
        frame_count = min(feature_frames, label_frames)
        if frame_count <= 0:
            stats["example_empty_grid"] += 1
            continue
        if abs(feature_frames - label_frames) > 1:
            # One frame of slack is rounding; more means the cached features and
            # the label manifest disagree about the audio, which must not be
            # silently truncated into alignment.
            stats["example_frame_count_mismatch"] += 1
            continue

        speech, types = frame_labels(example, frame_count, stats=stats)
        provenance = str(example.get("provenance"))

        counter = supervised[(partition, provenance)]
        counter["frames"] += frame_count
        counter["speech_known"] += int(np.count_nonzero(speech != IGNORE_INDEX))
        counter["speech_positive"] += int(np.count_nonzero(speech == 1))
        counter["type_known"] += int(np.count_nonzero(types != IGNORE_INDEX))
        for name, value in TYPE_TO_INDEX.items():
            counter[f"type_{name}"] += int(np.count_nonzero(types == value))

        speech_chunks.append(speech)
        type_chunks.append(types)
        index_rows.append(
            {
                "example_id": example_id,
                "partition": partition,
                "provenance": provenance,
                "dataset": str(example.get("dataset")),
                "feature_path": str(feature_path),
                "ptm_stored_dim": stored_dim,
                "ptm_use_dim": ptm_dim,
                "frame_offset": offset,
                "frame_count": frame_count,
                "frame_hop_s": float(example.get("frame_hop_s") or 0.02),
                "group_id": str(example.get("group_id")),
            }
        )
        offset += frame_count
        stats["examples_written"] += 1

    if not speech_chunks:
        raise SystemExit("no examples materialized; check --dataset/--split inputs")

    output.mkdir(parents=True, exist_ok=True)
    speech_all = np.concatenate(speech_chunks)
    type_all = np.concatenate(type_chunks)
    np.savez_compressed(
        output / "frame_labels.npz",
        speech=speech_all,
        type=type_all,
    )
    with (output / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema": OUTPUT_SCHEMA,
        "dataset": str(dataset),
        "split": str(split),
        "output": str(output),
        "ptm_use_dim": ptm_dim,
        "ignore_index": IGNORE_INDEX,
        "type_to_index": TYPE_TO_INDEX,
        "total_frames": int(speech_all.shape[0]),
        "stats": dict(stats),
        "supervised": {
            f"{partition}/{provenance}": dict(counter)
            for (partition, provenance), counter in sorted(supervised.items())
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"materialized {stats['examples_written']} examples, "
          f"{speech_all.shape[0]} frames -> {output}")
    print()
    print(f"{'partition/provenance':<40} {'hours':>8} {'speech%':>9} "
          f"{'typed%':>8}")
    for (partition, provenance), counter in sorted(supervised.items()):
        frames = counter["frames"] or 1
        known = counter["speech_known"] or 1
        hours = frames * 0.02 / 3600.0
        print(
            f"{partition + '/' + provenance:<40} {hours:>8.2f} "
            f"{100.0 * counter['speech_positive'] / known:>8.1f}% "
            f"{100.0 * counter['type_known'] / frames:>7.1f}%"
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ptm-dim", type=int, default=DEFAULT_PTM_DIM)
    parser.add_argument(
        "--partitions",
        default="train,val,test",
        help="Comma-separated partitions to materialize.",
    )
    parser.add_argument(
        "--real-feature-manifest",
        default="",
        help="Override real-window feature paths (e.g. the full-2048 re-export).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    materialize(
        dataset=Path(args.dataset).expanduser().resolve(),
        split=Path(args.split).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        ptm_dim=max(1, int(args.ptm_dim)),
        partitions=tuple(
            part.strip() for part in args.partitions.split(",") if part.strip()
        ),
        real_override_manifest=(
            Path(args.real_feature_manifest).expanduser().resolve()
            if args.real_feature_manifest
            else None
        ),
    )
