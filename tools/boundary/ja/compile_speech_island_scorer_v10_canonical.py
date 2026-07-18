#!/usr/bin/env python3
"""Compile auditable Scorer v10 full-source canonical supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT,
    SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
    SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
)


SOURCE_SCHEMA = "speech_scorer_v10_canonical_source_v1"
PREPARE_SUMMARY_SCHEMA = "speech_scorer_v10_canonical_prepare_summary_v1"
FINALIZE_SUMMARY_SCHEMA = "speech_scorer_v10_canonical_finalize_summary_v1"
CANONICAL_LABEL_SCHEMA = "speech_scorer_canonical_frames_v1"
CANONICAL_LABELS = {"background": 0, "speech": 1, "unsure": 2}
PARTITIONS = ("train", "val", "test")
DEFAULT_FRAME_HOP_S = 0.02


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not result:
        raise ValueError(f"identity cannot be converted to a safe filename: {value!r}")
    return result


def _frame_count(duration_s: float, frame_hop_s: float) -> int:
    if duration_s <= 0.0 or frame_hop_s <= 0.0:
        raise ValueError("duration and frame hop must be positive")
    return int(math.ceil((duration_s / frame_hop_s) - 1e-9))


def canonical_frame_labels(
    source: dict[str, Any],
    *,
    frame_count: int | None = None,
    frame_hop_s: float = DEFAULT_FRAME_HOP_S,
) -> np.ndarray:
    """Project exact sample spans to frames; mixed boundary cells remain unsure."""

    sample_rate = int(source["sample_rate"])
    sample_count = int(source["sample_count"])
    duration_s = sample_count / sample_rate
    count = frame_count or _frame_count(duration_s, frame_hop_s)
    spans = list(source.get("canonical_spans") or ())
    if not spans:
        raise ValueError(f"canonical source has no spans: {source.get('source_id')}")
    labels = np.full(count, CANONICAL_LABELS["unsure"], dtype=np.int64)
    for frame_index in range(count):
        frame_start = frame_index * frame_hop_s
        frame_end = min(duration_s, (frame_index + 1) * frame_hop_s)
        observed = {
            str(span["label"])
            for span in spans
            if min(frame_end, int(span["end_sample"]) / sample_rate)
            > max(frame_start, int(span["start_sample"]) / sample_rate)
        }
        if not observed:
            raise ValueError(
                f"canonical spans do not cover frame {frame_index}: {source.get('source_id')}"
            )
        unknown = observed - CANONICAL_LABELS.keys()
        if unknown:
            raise ValueError(f"invalid canonical labels: {sorted(unknown)}")
        labels[frame_index] = (
            CANONICAL_LABELS[next(iter(observed))]
            if len(observed) == 1
            else CANONICAL_LABELS["unsure"]
        )
    return labels


def _validate_exact_spans(source: dict[str, Any]) -> None:
    spans = sorted(
        [dict(span) for span in source.get("canonical_spans") or ()],
        key=lambda span: int(span["start_sample"]),
    )
    if not spans:
        raise ValueError(f"source has no canonical spans: {source.get('source_id')}")
    if int(spans[0]["start_sample"]) != 0:
        raise ValueError("canonical spans must begin at sample zero")
    if int(spans[-1]["end_sample"]) != int(source["sample_count"]):
        raise ValueError("canonical spans must cover the complete source")
    for left, right in zip(spans, spans[1:], strict=False):
        if int(left["end_sample"]) != int(right["start_sample"]):
            raise ValueError("canonical spans must be contiguous and non-overlapping")
    for span in spans:
        if int(span["end_sample"]) <= int(span["start_sample"]):
            raise ValueError("canonical spans must have positive length")
        if str(span.get("label") or "") not in CANONICAL_LABELS:
            raise ValueError("canonical span has an invalid label")


def _strict_negative_rows(path: Path, *, min_confidence: float) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    seen: set[str] = set()
    video_partitions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        identity = str(row.get("audio_id") or "")
        partition = str(row.get("source_partition") or "")
        video_id = str(row.get("video_id") or row.get("window_id") or "")
        if not identity or identity in seen:
            raise ValueError("negative manifest requires unique audio_id values")
        seen.add(identity)
        if partition not in PARTITIONS:
            raise ValueError("negative manifest requires fixed train/val/test partitions")
        if str(row.get("source") or "") != "omni_definite_drop":
            raise ValueError("background controls require strict Omni definite-drop rows")
        if float(row.get("omni_confidence") or 0.0) < min_confidence:
            raise ValueError("background control confidence is below the canonical floor")
        if not video_id:
            raise ValueError("background controls require video identity")
        video_partitions[video_id].add(partition)
        audio = Path(str(row.get("audio") or ""))
        if not audio.is_file():
            raise ValueError(f"background control audio is missing: {audio}")
    if any(len(values) != 1 for values in video_partitions.values()):
        raise ValueError("background video identity crosses dataset partitions")
    return rows


def _select_background_rows(
    rows: Sequence[dict[str, Any]], *, maximum_per_partition: int
) -> list[dict[str, Any]]:
    if maximum_per_partition < 0:
        raise ValueError("background maximum must be non-negative")
    result: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        available = sorted(
            (row for row in rows if row["source_partition"] == partition),
            key=lambda row: (float(row.get("duration_s") or 0.0), str(row["audio_id"])),
        )
        if not available:
            raise ValueError(f"negative manifest has no {partition} controls")
        if maximum_per_partition and len(available) > maximum_per_partition:
            indexes = np.linspace(
                0, len(available) - 1, maximum_per_partition, dtype=np.int64
            )
            available = [available[int(index)] for index in indexes]
        result.extend(available)
    return result


def _background_details(row: dict[str, Any]) -> list[dict[str, Any]]:
    details = [
        dict(row["negative_unit_span"]["source"]),
        *(dict(value) for value in row["inter_unit_gaps"]["sources"]),
    ]
    if row.get("additive_overlay"):
        details.append(dict(row["additive_overlay"]["source"]))
    return details


def _speech_source(
    row: dict[str, Any], *, negative_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    partition = str(row.get("source_partition") or "")
    if partition not in PARTITIONS:
        raise ValueError("speech composites require fixed train/val/test partitions")
    audio = Path(str(row.get("audio") or ""))
    if not audio.is_file():
        raise ValueError(f"speech composite audio is missing: {audio}")
    if int(row.get("sample_rate") or 0) != 16000:
        raise ValueError("Scorer v10 canonical sources must be 16 kHz")
    spans: list[dict[str, Any]] = []
    for core in row.get("core_spans") or ():
        spans.append(
            {
                "start_sample": int(core["start_sample"]),
                "end_sample": int(core["end_sample"]),
                "label": "speech",
                "label_source": "teacher_approved_galgame_speech_core_composition_extent",
                "core_id": str(core["core_id"]),
            }
        )
    negative = row["negative_unit_span"]
    spans.append(
        {
            "start_sample": int(negative["start_sample"]),
            "end_sample": int(negative["end_sample"]),
            "label": "background",
            "label_source": "strict_cueqc_definite_drop_synthetic_unit",
            "background_id": str(negative["source"]["audio_id"]),
        }
    )
    gaps = row["inter_unit_gaps"]
    gap_sources = list(gaps["sources"])
    for name, detail in (("left", gap_sources[0]), ("right", gap_sources[1])):
        spans.append(
            {
                "start_sample": int(gaps[f"{name}_start_sample"]),
                "end_sample": int(gaps[f"{name}_end_sample"]),
                "label": "background",
                "label_source": "strict_cueqc_definite_drop_synthetic_gap",
                "background_id": str(detail["audio_id"]),
            }
        )
    details = _background_details(row)
    background_ids = sorted({str(detail["audio_id"]) for detail in details})
    background_videos: set[str] = set()
    for identity in background_ids:
        canonical = negative_by_id.get(identity)
        if canonical is None:
            raise ValueError(f"speech composite background is not canonical: {identity}")
        if str(canonical["source_partition"]) != partition:
            raise ValueError("embedded background identity crosses source partition")
        background_videos.add(str(canonical["video_id"]))
    source = {
        "schema": SOURCE_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "source_id": str(row["sample_id"]),
        "audio": str(audio),
        "row_role": "speech",
        "partition": partition,
        "core_ids": [str(core["core_id"]) for core in row["core_spans"]],
        "background_id": "",
        "background_source_ids": background_ids,
        "background_source_video_ids": sorted(background_videos),
        "sample_rate": int(row["sample_rate"]),
        "sample_count": int(row["sample_count"]),
        "duration_s": float(row["sample_count"]) / int(row["sample_rate"]),
        "input_distribution": "full_source_windows",
        "canonical_spans": sorted(spans, key=lambda span: int(span["start_sample"])),
        "additive_overlay": row.get("additive_overlay"),
        "source_contract": str(row.get("label_contract") or ""),
    }
    _validate_exact_spans(source)
    return source


def _background_source(row: dict[str, Any]) -> dict[str, Any]:
    audio = Path(str(row["audio"]))
    sample_rate = int(row.get("sample_rate") or 16000)
    if sample_rate != 16000:
        raise ValueError("Scorer v10 canonical sources must be 16 kHz")
    # The exported asset duration is authoritative until feature extraction rechecks it.
    sample_count = max(1, int(round(float(row["duration_s"]) * sample_rate)))
    identity = str(row["audio_id"])
    source = {
        "schema": SOURCE_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "source_id": f"scorer-v10-background-{_safe_id(identity)}",
        "audio": str(audio),
        "row_role": "all_background",
        "partition": str(row["source_partition"]),
        "core_ids": [],
        "background_id": identity,
        "background_source_ids": [identity],
        "background_source_video_ids": [str(row["video_id"])],
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "duration_s": sample_count / sample_rate,
        "input_distribution": "full_source_windows",
        "canonical_spans": [
            {
                "start_sample": 0,
                "end_sample": sample_count,
                "label": "background",
                "label_source": "strict_cueqc_definite_drop_full_control",
                "background_id": identity,
            }
        ],
        "background_type": str(row.get("background_type") or "omni_drop"),
        "omni_flags": list(row.get("omni_flags") or ()),
        "omni_confidence": float(row.get("omni_confidence") or 0.0),
    }
    _validate_exact_spans(source)
    return source


def _validate_sources(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source_ids: set[str] = set()
    core_counts: Counter[str] = Counter()
    core_partitions: dict[str, set[str]] = defaultdict(set)
    background_partitions: dict[str, set[str]] = defaultdict(set)
    video_partitions: dict[str, set[str]] = defaultdict(set)
    presence = {name: Counter() for name in PARTITIONS}
    for row in rows:
        source_id = str(row["source_id"])
        partition = str(row["partition"])
        if source_id in source_ids:
            raise ValueError(f"canonical source is duplicated: {source_id}")
        source_ids.add(source_id)
        if partition not in PARTITIONS:
            raise ValueError("canonical source has an invalid partition")
        row_role = str(row.get("row_role") or "")
        core_ids = [str(value) for value in row.get("core_ids") or ()]
        background_id = str(row.get("background_id") or "")
        if row_role == "speech" and (not core_ids or background_id):
            raise ValueError("canonical speech rows require cores and no background_id")
        if row_role == "all_background" and (core_ids or not background_id):
            raise ValueError("canonical all-background rows require background_id only")
        if row_role not in {"speech", "all_background"}:
            raise ValueError("canonical source has an invalid row role")
        if row.get("boundary_serialization_contract_id") != ACOUSTIC_BINARY_V12_CONTRACT.contract_id:
            raise ValueError("canonical source has the wrong central contract")
        _validate_exact_spans(row)
        presence[partition][row_role] += 1
        for core_id in core_ids:
            core_counts[core_id] += 1
            core_partitions[core_id].add(partition)
        for identity in row.get("background_source_ids") or ():
            background_partitions[str(identity)].add(partition)
        for video_id in row.get("background_source_video_ids") or ():
            video_partitions[str(video_id)].add(partition)
    if max(core_counts.values(), default=0) > 1:
        raise ValueError("canonical Scorer cores may be used at most once")
    if any(len(values) != 1 for values in core_partitions.values()):
        raise ValueError("canonical core identity crosses partitions")
    if any(len(values) != 1 for values in background_partitions.values()):
        raise ValueError("canonical background identity crosses partitions")
    if any(len(values) != 1 for values in video_partitions.values()):
        raise ValueError("canonical background video crosses partitions")
    if any(not presence[name]["speech"] or not presence[name]["all_background"] for name in PARTITIONS):
        raise ValueError("every partition requires speech and all-background sources")
    return {
        "source_count": len(source_ids),
        "core_count": len(core_counts),
        "max_core_use_count": max(core_counts.values(), default=0),
        "background_identity_count": len(background_partitions),
        "background_video_count": len(video_partitions),
        "partition_role_counts": {
            name: dict(sorted(presence[name].items())) for name in PARTITIONS
        },
    }


def prepare_dataset(
    *,
    speech_manifest: Path,
    negative_manifest: Path,
    output_dir: Path,
    frame_hop_s: float = DEFAULT_FRAME_HOP_S,
    min_negative_confidence: float = 0.90,
    background_max_per_partition: int = 0,
) -> dict[str, Any]:
    negatives = _strict_negative_rows(
        negative_manifest, min_confidence=min_negative_confidence
    )
    negative_by_id = {str(row["audio_id"]): row for row in negatives}
    speech_sources = [
        _speech_source(row, negative_by_id=negative_by_id)
        for row in _read_jsonl(speech_manifest)
    ]
    selected_background = _select_background_rows(
        negatives, maximum_per_partition=background_max_per_partition
    )
    sources = [*speech_sources, *(_background_source(row) for row in selected_background)]
    dataset_summary = _validate_sources(sources)

    feature_labels: list[dict[str, Any]] = []
    audio_manifest: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for source in sources:
        labels = canonical_frame_labels(source, frame_hop_s=frame_hop_s)
        label_counts.update(
            background=int(np.sum(labels == CANONICAL_LABELS["background"])),
            speech=int(np.sum(labels == CANONICAL_LABELS["speech"])),
            unsure=int(np.sum(labels == CANONICAL_LABELS["unsure"])),
        )
        weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0)
        feature_labels.append(
            {
                "audio_id": source["source_id"],
                "source": "scorer_v10_canonical_full_source",
                "duration_s": source["duration_s"],
                "text": "",
                "teacher_segments": {},
                "frame_hop_s": frame_hop_s,
                "speech_frames": (labels == CANONICAL_LABELS["speech"]).astype(int).tolist(),
                "label_quality": "negative" if source["row_role"] == "all_background" else "supervised",
                "frame_weights": weights.tolist(),
                "boundary_metadata": {
                    "schema": SOURCE_SCHEMA,
                    "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
                    "row_role": source["row_role"],
                    "partition": source["partition"],
                    "unsure_frame_count": int(np.sum(labels == CANONICAL_LABELS["unsure"])),
                },
            }
        )
        audio_manifest.append(
            {
                "audio_id": source["source_id"],
                "audio": source["audio"],
                "partition": source["partition"],
                "row_role": source["row_role"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sources_path = output_dir / "canonical_sources.jsonl"
    labels_path = output_dir / "feature_cache_labels.jsonl"
    audio_manifest_path = output_dir / "audio_manifest.json"
    _write_jsonl(sources_path, sources)
    _write_jsonl(labels_path, feature_labels)
    audio_manifest_path.write_text(
        json.dumps(audio_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    used_in_speech = {
        identity
        for source in speech_sources
        for identity in source["background_source_ids"]
    }
    summary = {
        "schema": PREPARE_SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "speech_manifest": str(speech_manifest),
        "negative_manifest": str(negative_manifest),
        "frame_hop_s": frame_hop_s,
        "minimum_negative_confidence": min_negative_confidence,
        "background_control_asset_overlap_with_speech_augmentation": len(
            used_in_speech & {str(row["audio_id"]) for row in selected_background}
        ),
        "canonical_frame_counts": dict(label_counts),
        "unsure_training_mapping": -100,
        "dataset": dataset_summary,
        "canonical_sources": str(sources_path),
        "feature_cache_labels": str(labels_path),
        "audio_manifest": str(audio_manifest_path),
        "feature_cache_ready": True,
        "training_ready": False,
        "promotion_ready": False,
    }
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def finalize_dataset(
    *,
    canonical_sources: Path,
    feature_manifest: Path,
    manual_gate_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    gate = json.loads(manual_gate_summary.read_text(encoding="utf-8-sig"))
    if gate.get("schema") != "speech_scorer_v10_canonical_manual_gate_v1":
        raise ValueError("Scorer v10 canonical finalize requires the manual gate schema")
    canonical_sha256 = hashlib.sha256(canonical_sources.read_bytes()).hexdigest()
    if gate.get("canonical_sources_sha256") != canonical_sha256:
        raise ValueError("Scorer v10 canonical manual gate is bound to another manifest")
    if gate.get("manual_gate_pass") is not True:
        raise ValueError("Scorer v10 canonical manual gate has not passed")
    sources = _read_jsonl(canonical_sources)
    dataset_summary = _validate_sources(sources)
    features: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(feature_manifest):
        identity = str(row.get("audio_id") or "")
        if not identity or identity in features:
            raise ValueError("feature manifest requires unique audio_id values")
        features[identity] = row
    expected = {str(row["source_id"]) for row in sources}
    if set(features) != expected:
        missing = sorted(expected - set(features))
        extra = sorted(set(features) - expected)
        raise ValueError(
            f"feature manifest identity mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    training_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for source in sources:
        feature = features[str(source["source_id"])]
        if str(feature.get("ptm") or "") != QWEN_ASR_17B_REPO_ID:
            raise ValueError("Scorer v10 feature cache must use the 1.7B PTM")
        if int(feature.get("ptm_dim") or 0) != SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM:
            raise ValueError("Scorer v10 feature cache must retain raw PTM2048")
        if int(feature.get("mfcc_dim") or 0) != SPEECH_ISLAND_SCORER_V10_MFCC_DIM:
            raise ValueError("Scorer v10 feature cache must contain MFCC40")
        if abs(float(feature.get("frame_hop_s") or 0.0) - DEFAULT_FRAME_HOP_S) > 1e-9:
            raise ValueError("Scorer v10 feature cache requires a 20 ms frame hop")
        feature_path = Path(str(feature.get("feature_path") or ""))
        if not feature_path.is_file():
            raise ValueError(f"Scorer v10 feature cache is missing: {feature_path}")
        with np.load(feature_path) as payload:
            ptm_shape = tuple(np.asarray(payload["ptm"]).shape)
            mfcc_shape = tuple(np.asarray(payload["mfcc"]).shape)
        frame_count = int(feature["frame_count"])
        if ptm_shape != (frame_count, SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM):
            raise ValueError("Scorer v10 cached PTM shape does not match its manifest")
        if mfcc_shape != (frame_count, SPEECH_ISLAND_SCORER_V10_MFCC_DIM):
            raise ValueError("Scorer v10 cached MFCC shape does not match its manifest")
        labels = canonical_frame_labels(
            source, frame_count=frame_count, frame_hop_s=DEFAULT_FRAME_HOP_S
        )
        weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0).astype(
            np.float32
        )
        label_path = labels_dir / f"{_safe_id(str(source['source_id']))}.labels.npz"
        np.savez_compressed(
            label_path,
            canonical_labels=labels,
            frame_weights=weights,
            canonical_label_schema=np.asarray([CANONICAL_LABEL_SCHEMA]),
            boundary_serialization_contract_id=np.asarray(
                [ACOUSTIC_BINARY_V12_CONTRACT.contract_id]
            ),
        )
        label_counts.update(
            background=int(np.sum(labels == CANONICAL_LABELS["background"])),
            speech=int(np.sum(labels == CANONICAL_LABELS["speech"])),
            unsure=int(np.sum(labels == CANONICAL_LABELS["unsure"])),
        )
        training_rows.append(
            {
                "schema": "speech_scorer_v10_binary_training_row_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source["source_id"],
                "audio": source["audio"],
                "row_role": source["row_role"],
                "partition": source["partition"],
                "core_ids": list(source.get("core_ids") or ()),
                "background_id": str(source.get("background_id") or ""),
                "background_source_ids": list(source.get("background_source_ids") or ()),
                "input_distribution": SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT[
                    "input_distribution"
                ],
                "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
                "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "feature_path": str(feature_path),
                "label_path": str(label_path),
                "frame_count": frame_count,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_manifest = output_dir / "training_manifest.jsonl"
    _write_jsonl(training_manifest, training_rows)
    from tools.boundary.ja.train_speech_island_scorer_v10_binary import (  # noqa: E402
        summarize_partition_labels,
        validate_dataset_rows,
    )

    trainer_summary = validate_dataset_rows(training_rows)
    presence, replay_counts = summarize_partition_labels(training_rows)
    summary = {
        "schema": FINALIZE_SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "canonical_sources_sha256": canonical_sha256,
        "manual_gate_summary": str(manual_gate_summary),
        "dataset": dataset_summary,
        "trainer_dataset": trainer_summary,
        "partition_label_presence": presence,
        "canonical_frame_counts": dict(label_counts),
        "trainer_replay_frame_counts": dict(replay_counts),
        "excluded_training_count": int(label_counts["unsure"]),
        "training_manifest": str(training_manifest),
        "training_ready": True,
        "numeric_gate_pass": False,
        "manual_zero_clipping_gate": "required_before_promotion",
        "promotion_ready": False,
    }
    (output_dir / "finalize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--speech-manifest", required=True)
    prepare.add_argument("--negative-manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--frame-hop-s", type=float, default=DEFAULT_FRAME_HOP_S)
    prepare.add_argument("--min-negative-confidence", type=float, default=0.90)
    prepare.add_argument("--background-max-per-partition", type=int, default=0)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--canonical-sources", required=True)
    finalize.add_argument("--feature-manifest", required=True)
    finalize.add_argument("--manual-gate-summary", required=True)
    finalize.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_dataset(
            speech_manifest=Path(args.speech_manifest),
            negative_manifest=Path(args.negative_manifest),
            output_dir=Path(args.output_dir),
            frame_hop_s=float(args.frame_hop_s),
            min_negative_confidence=float(args.min_negative_confidence),
            background_max_per_partition=int(args.background_max_per_partition),
        )
    else:
        result = finalize_dataset(
            canonical_sources=Path(args.canonical_sources),
            feature_manifest=Path(args.feature_manifest),
            manual_gate_summary=Path(args.manual_gate_summary),
            output_dir=Path(args.output_dir),
        )
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
