#!/usr/bin/env python3
"""Compile fully reviewed Scorer v11 candidate-membership canonical sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_SYNTHETIC_TRAIN_SOURCE_SCHEMA,
)


SUMMARY_SCHEMA = "candidate_island_scorer_v11_canonical_compile_summary_v1"
PARTITION_SCHEMA = "candidate_island_scorer_v11_partition_manifest_v1"
HELDOUT_VERDICT_SCHEMA = "candidate_island_scorer_v11_heldout_manual_verdict_v1"
MANUAL_VERDICT_SCHEMA = "candidate_island_scorer_v11_manual_verdict_v1"
REAL_TRAIN_OUTSIDE_SCHEMA = "candidate_island_scorer_v11_real_train_outside_source_v1"
FRAME_HOP_S = 0.02
LABELS = {"outside_candidate": 0, "inside_candidate": 1, "unsure": 2}
PARTITIONS = {"train", "val", "test"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _index_unique(rows: Sequence[dict[str, Any]], key: str, *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity:
            raise ValueError(f"{name} row is missing {key}")
        if identity in result:
            raise ValueError(f"duplicate {name} {key}: {identity}")
        result[identity] = row
    return result


def _validate_spans(
    spans: Sequence[dict[str, Any]], *, source_id: str, frame_count: int
) -> list[dict[str, Any]]:
    if not spans:
        raise ValueError(f"manual verdict has no full-source spans: {source_id}")
    normalized: list[dict[str, Any]] = []
    cursor = 0
    for span in spans:
        label = str(span.get("label") or "")
        start = int(span.get("start_frame", -1))
        end = int(span.get("end_frame", -1))
        if label not in LABELS:
            raise ValueError(f"invalid v11 canonical label for {source_id}: {label!r}")
        if start != cursor or end <= start or end > frame_count:
            raise ValueError(
                f"v11 canonical spans must be contiguous full-source coverage: {source_id}"
            )
        normalized.append(
            {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "start_s": round(start * FRAME_HOP_S, 6),
                "end_s": round(end * FRAME_HOP_S, 6),
            }
        )
        cursor = end
    if cursor != frame_count:
        raise ValueError(f"v11 canonical spans do not cover the source tail: {source_id}")
    return normalized


def _wav_geometry(path: Path) -> tuple[int, float, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_count = int(handle.getnframes())
    if sample_rate != 16000 or channels != 1 or sample_count <= 0:
        raise ValueError(
            "Scorer v11 held-out audio must be non-empty 16k mono PCM WAV: "
            f"{path} rate={sample_rate} channels={channels} samples={sample_count}"
        )
    return sample_count, sample_count / sample_rate, (sample_count + 319) // 320


def _clip_reviewed_spans_to_audio(
    spans: Sequence[dict[str, Any]],
    *,
    source_id: str,
    reviewed_frame_count: int,
    audio_frame_count: int,
) -> tuple[list[dict[str, Any]], int, str]:
    if audio_frame_count <= 0:
        raise ValueError(f"Scorer v11 held-out audio has no frames: {source_id}")
    if audio_frame_count > reviewed_frame_count + 1:
        raise ValueError(
            "Scorer v11 held-out audio extends beyond the reviewed frame grid: "
            f"{source_id} reviewed={reviewed_frame_count} audio={audio_frame_count}"
        )
    effective_frames = min(reviewed_frame_count, audio_frame_count)
    if effective_frames == reviewed_frame_count:
        return (
            _validate_spans(
                spans, source_id=source_id, frame_count=reviewed_frame_count
            ),
            effective_frames,
            "exact_or_subframe_audio_tail_ignored_v1",
        )
    clipped: list[dict[str, Any]] = []
    for span in spans:
        start = int(span.get("start_frame", -1))
        end = min(int(span.get("end_frame", -1)), effective_frames)
        if start >= effective_frames:
            break
        if end > start:
            clipped.append(
                {
                    "label": str(span.get("label") or ""),
                    "start_frame": start,
                    "end_frame": end,
                }
            )
    return (
        _validate_spans(
            clipped, source_id=source_id, frame_count=effective_frames
        ),
        effective_frames,
        "trim_unavailable_review_grid_tail_to_decoded_audio_v1",
    )


def canonical_frame_labels(source: dict[str, Any]) -> np.ndarray:
    if source.get("schema") != CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA:
        raise ValueError("wrong Scorer v11 canonical source schema")
    frame_count = int(source.get("frame_count") or 0)
    labels = np.full(frame_count, -1, dtype=np.int64)
    for span in source.get("canonical_spans") or ():
        labels[int(span["start_frame"]) : int(span["end_frame"])] = LABELS[
            str(span["label"])
        ]
    if frame_count <= 0 or np.any(labels < 0):
        raise ValueError(f"invalid canonical frame coverage: {source.get('source_id')}")
    return labels


def compile_canonical(
    *,
    synthetic_train_sources: Path,
    real_train_outside_sources: Path,
    source_windows: Path,
    partition_manifest: Path,
    manual_verdicts: Sequence[Path],
    output_dir: Path,
    verify_audio: bool = True,
) -> dict[str, Any]:
    synthetic_train_sources = synthetic_train_sources.resolve()
    real_train_outside_sources = real_train_outside_sources.resolve()
    source_windows = source_windows.resolve()
    partition_manifest = partition_manifest.resolve()
    verdict_paths = [path.resolve() for path in manual_verdicts]
    if not verdict_paths:
        raise ValueError("Scorer v11 canonical compile requires manual verdict files")
    for path in (
        synthetic_train_sources,
        real_train_outside_sources,
        source_windows,
        partition_manifest,
        *verdict_paths,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    sources = _index_unique(_read_jsonl(source_windows), "window_id", name="source")
    partitions = _index_unique(
        _read_jsonl(partition_manifest), "source_id", name="partition"
    )
    if not set(partitions).issubset(sources):
        missing_sources = sorted(set(partitions) - set(sources))
        raise ValueError(
            "partition manifest references missing source windows: "
            f"missing_sources={missing_sources[:5]}"
        )
    sources = {source_id: sources[source_id] for source_id in partitions}
    for source_id, partition_row in partitions.items():
        if partition_row.get("schema") != PARTITION_SCHEMA:
            raise ValueError(f"wrong v11 partition schema: {source_id}")
        if partition_row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"wrong central boundary contract: {source_id}")
        partition = str(partition_row.get("partition") or "")
        if partition not in PARTITIONS:
            raise ValueError(f"invalid Scorer v11 partition: {source_id}")
        source = sources[source_id]
        if source.get("schema") != "joint_boundary_omni_source_window_v1":
            raise ValueError(f"wrong Scorer v11 source-window schema: {source_id}")
        if str(source.get("video_id") or "") != str(
            partition_row.get("video_id") or ""
        ):
            raise ValueError(f"video identity mismatch: {source_id}")
    heldout_partitions = {
        source_id: row
        for source_id, row in partitions.items()
        if str(row.get("partition") or "") in {"val", "test"}
    }
    heldout_sources = {
        source_id: sources[source_id] for source_id in heldout_partitions
    }
    if not heldout_sources:
        raise ValueError("Scorer v11 canonical requires frozen real val/test sources")

    verdict_rows: list[dict[str, Any]] = []
    verdict_file_by_source: dict[str, Path] = {}
    for path in verdict_paths:
        for row in _read_jsonl(path):
            source_id = str(row.get("source_id") or "")
            if source_id in verdict_file_by_source:
                raise ValueError(f"duplicate manual verdict source: {source_id}")
            verdict_file_by_source[source_id] = path
            verdict_rows.append(row)
    verdicts = _index_unique(verdict_rows, "source_id", name="manual verdict")
    missing = sorted(set(heldout_sources) - set(verdicts))
    extra = sorted(set(verdicts) - set(heldout_sources))
    if missing or extra:
        raise ValueError(
            "Scorer v11 requires current-duty full-source truth for every frozen source; "
            f"missing={len(missing)} {missing[:8]}, extra={len(extra)} {extra[:8]}"
        )

    synthetic_train_sha = _sha256(synthetic_train_sources)
    real_train_outside_sha = _sha256(real_train_outside_sources)
    source_windows_sha = _sha256(source_windows)
    partition_sha = _sha256(partition_manifest)
    verdict_shas = {_display(path): _sha256(path) for path in verdict_paths}
    compiled: list[dict[str, Any]] = []
    partition_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    seen_video_partition: dict[str, str] = {}
    seen_core_partition: dict[str, str] = {}
    seen_core_source: dict[str, str] = {}
    seen_source_partition: dict[str, str] = {}

    synthetic_rows = _index_unique(
        _read_jsonl(synthetic_train_sources), "source_id", name="synthetic train source"
    )
    if not synthetic_rows:
        raise ValueError("Scorer v11 canonical requires non-empty synthetic train sources")
    for source_id in sorted(synthetic_rows):
        source = synthetic_rows[source_id]
        if source.get("schema") != CANDIDATE_ISLAND_SCORER_V11_SYNTHETIC_TRAIN_SOURCE_SCHEMA:
            raise ValueError(f"wrong Scorer v11 synthetic train schema: {source_id}")
        if source.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"wrong central boundary contract: {source_id}")
        if source.get("partition") != "train":
            raise ValueError(f"synthetic Scorer v11 source is not train-only: {source_id}")
        if not bool(source.get("synthetic_composite")) or not bool(
            source.get("training_manifest_allowed")
        ):
            raise ValueError(f"synthetic Scorer v11 source is not approved: {source_id}")
        if source.get("source_kind") not in {
            "semantic_composite_candidate",
            "isolated_human_vocal_candidate",
            "clear_nonvocal_all_background",
        }:
            raise ValueError(f"unknown Scorer v11 synthetic source kind: {source_id}")
        frame_count = int(source.get("frame_count") or 0)
        if (
            frame_count <= 0
            or float(source.get("frame_hop_s") or 0.0) != FRAME_HOP_S
            or int(source.get("sample_rate") or 0) != 16000
            or int(source.get("sample_count") or 0) != frame_count * 320
            or abs(float(source.get("duration_s") or 0.0) - frame_count * FRAME_HOP_S)
            > 1e-9
        ):
            raise ValueError(f"synthetic Scorer v11 frame geometry mismatch: {source_id}")
        spans = _validate_spans(
            list(source.get("canonical_spans") or ()),
            source_id=source_id,
            frame_count=frame_count,
        )
        if any(span["label"] == "unsure" for span in spans):
            raise ValueError(f"synthetic Scorer v11 truth cannot contain unsure: {source_id}")
        audio = _resolve(str(source.get("audio") or ""))
        if not audio.exists():
            raise FileNotFoundError(audio)
        audio_sha = str(source.get("audio_sha256") or "")
        if len(audio_sha) != 64 or (verify_audio and _sha256(audio) != audio_sha):
            raise ValueError(f"synthetic Scorer v11 audio SHA256 mismatch: {source_id}")
        if source_id in heldout_sources:
            raise ValueError(f"Scorer v11 source identity crosses partitions: {source_id}")
        seen_source_partition[source_id] = "train"
        core_ids = [str(value) for value in source.get("core_ids") or ()]
        if any(not value for value in core_ids) or len(set(core_ids)) != len(core_ids):
            raise ValueError(f"invalid synthetic core identities: {source_id}")
        for core_id in core_ids:
            previous_source = seen_core_source.setdefault(core_id, source_id)
            if previous_source != source_id:
                raise ValueError(
                    f"Scorer v11 core identity is reused: {core_id} in "
                    f"{previous_source} and {source_id}"
                )
            previous = seen_core_partition.setdefault(core_id, "train")
            if previous != "train":
                raise ValueError(f"Scorer v11 core identity crosses partitions: {core_id}")
        for span in spans:
            label_counts[str(span["label"])] += int(span["end_frame"]) - int(
                span["start_frame"]
            )
        partition_counts["train"] += 1
        compiled.append(
            {
                "schema": CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "canonical_label_schema": (
                    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA
                ),
                "source_id": source_id,
                "video_id": "",
                "core_ids": core_ids,
                "core_identity_kind": "frozen_synthetic_component_v1",
                "partition": "train",
                "input_distribution": str(source.get("input_distribution") or ""),
                "source_kind": str(source.get("source_kind") or ""),
                "synthetic_composite": True,
                "audio": _display(audio),
                "audio_sha256": audio_sha,
                "duration_s": float(source["duration_s"]),
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "canonical_spans": spans,
                "annotation_provenance": "exact_composition_train_truth",
                "synthetic_train_sources_sha256": synthetic_train_sha,
                "candidate_sample_span": source.get("candidate_sample_span"),
                "candidate_source": source.get("candidate_source"),
                "outside_brackets": source.get("outside_brackets"),
                "composition_provenance": source.get("composition_provenance"),
                "training_manifest_allowed": True,
            }
        )

    real_train_rows = _index_unique(
        _read_jsonl(real_train_outside_sources),
        "source_id",
        name="real train outside source",
    )
    if not real_train_rows:
        raise ValueError("Scorer v11 canonical requires real train outside sources")
    for source_id in sorted(real_train_rows):
        source = real_train_rows[source_id]
        if source.get("schema") != REAL_TRAIN_OUTSIDE_SCHEMA:
            raise ValueError(f"wrong Scorer v11 real train outside schema: {source_id}")
        if source.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"wrong central boundary contract: {source_id}")
        if source.get("partition") != "train" or not bool(
            source.get("training_manifest_allowed")
        ):
            raise ValueError(f"real Scorer v11 outside source is not approved train data: {source_id}")
        if bool(source.get("gemini_output_used_as_inside_truth")):
            raise ValueError(f"Gemini inside cannot become Scorer v11 truth: {source_id}")
        if bool(source.get("asr_text_used_as_inside_truth")) or bool(
            source.get("asr_empty_used_without_gemini_outside")
        ):
            raise ValueError(f"ASR cannot independently define Scorer v11 truth: {source_id}")
        if int(source.get("unsure_training_label", 0)) != -100:
            raise ValueError(f"real Scorer v11 unsure must map to -100: {source_id}")
        if source_id not in sources or source_id not in partitions:
            raise ValueError(f"real train outside source is outside the frozen source scope: {source_id}")
        partition_row = partitions[source_id]
        source_window = sources[source_id]
        if partition_row.get("partition") != "train":
            raise ValueError(f"real outside source crosses into held-out: {source_id}")
        video_id = str(source.get("video_id") or "")
        if not video_id or video_id != str(partition_row.get("video_id") or "") or video_id != str(
            source_window.get("video_id") or ""
        ):
            raise ValueError(f"real train outside video identity mismatch: {source_id}")
        previous_partition = seen_video_partition.setdefault(video_id, "train")
        if previous_partition != "train":
            raise ValueError(f"video identity crosses partitions: {video_id}")
        if source_id in seen_source_partition or source_id in heldout_sources:
            raise ValueError(f"Scorer v11 source identity crosses partitions: {source_id}")
        seen_source_partition[source_id] = "train"
        frame_count = int(source.get("frame_count") or 0)
        if frame_count <= 0 or float(source.get("frame_hop_s") or 0.0) != FRAME_HOP_S:
            raise ValueError(f"real train outside frame geometry mismatch: {source_id}")
        spans = _validate_spans(
            list(source.get("canonical_spans") or ()),
            source_id=source_id,
            frame_count=frame_count,
        )
        labels_present = {str(span["label"]) for span in spans}
        if not labels_present.issubset({"outside_candidate", "unsure"}) or (
            "outside_candidate" not in labels_present
        ):
            raise ValueError(f"real train outside truth has invalid duties: {source_id}")
        audio = _resolve(str(source.get("audio") or ""))
        if not audio.exists():
            raise FileNotFoundError(audio)
        audio_sha = str(source.get("audio_sha256") or "")
        expected_sha = str(source_window.get("audio_wav_sha256") or "")
        if len(audio_sha) != 64 or audio_sha != expected_sha:
            raise ValueError(f"real train outside audio SHA identity mismatch: {source_id}")
        if verify_audio and _sha256(audio) != audio_sha:
            raise ValueError(f"real train outside audio SHA256 mismatch: {source_id}")
        sample_count, duration_s, audio_frame_count = _wav_geometry(audio)
        spans, effective_frame_count, geometry_policy = _clip_reviewed_spans_to_audio(
            spans,
            source_id=source_id,
            reviewed_frame_count=frame_count,
            audio_frame_count=audio_frame_count,
        )
        core_ids = [str(value) for value in source.get("core_ids") or ()]
        if core_ids != [f"real-train-outside-source::{source_id}"]:
            raise ValueError(f"real train outside core identity mismatch: {source_id}")
        core_id = core_ids[0]
        previous_core_source = seen_core_source.setdefault(core_id, source_id)
        if previous_core_source != source_id:
            raise ValueError(f"Scorer v11 core identity is reused: {core_id}")
        previous_core_partition = seen_core_partition.setdefault(core_id, "train")
        if previous_core_partition != "train":
            raise ValueError(f"Scorer v11 core identity crosses partitions: {core_id}")
        for span in spans:
            label_counts[str(span["label"])] += int(span["end_frame"]) - int(
                span["start_frame"]
            )
        partition_counts["train"] += 1
        compiled.append(
            {
                "schema": CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "canonical_label_schema": CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA,
                "source_id": source_id,
                "video_id": video_id,
                "core_ids": core_ids,
                "core_identity_kind": "real_train_outside_source_v1",
                "partition": "train",
                "input_distribution": str(source.get("input_distribution") or ""),
                "source_kind": "real_train_outside_masked",
                "synthetic_composite": False,
                "audio": _display(audio),
                "audio_sha256": audio_sha,
                "duration_s": duration_s,
                "frame_count": effective_frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "audio_sample_count": sample_count,
                "reviewed_nominal_frame_count": frame_count,
                "audio_geometry_policy": geometry_policy,
                "canonical_spans": spans,
                "annotation_provenance": str(source.get("annotation_provenance") or ""),
                "real_train_outside_sources_sha256": real_train_outside_sha,
                "gemini_output_used_as_inside_truth": False,
                "asr_text_used_as_inside_truth": False,
                "asr_empty_used_without_gemini_outside": False,
                "unsure_training_label": -100,
                "training_manifest_allowed": True,
            }
        )

    for source_id in sorted(heldout_sources):
        source = heldout_sources[source_id]
        partition_row = heldout_partitions[source_id]
        verdict = verdicts[source_id]
        if partition_row.get("schema") != PARTITION_SCHEMA:
            raise ValueError(f"wrong v11 partition schema: {source_id}")
        if verdict.get("schema") not in {HELDOUT_VERDICT_SCHEMA, MANUAL_VERDICT_SCHEMA}:
            raise ValueError(f"wrong v11 manual verdict schema: {source_id}")
        for row in (partition_row, verdict):
            if row.get("boundary_serialization_contract_id") != (
                ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            ):
                raise ValueError(f"wrong central boundary contract: {source_id}")
        partition = str(partition_row.get("partition") or "")
        if partition not in {"val", "test"} or verdict.get("partition") != partition:
            raise ValueError(f"partition mismatch: {source_id}")
        if not bool(verdict.get("reviewed_full_source")):
            raise ValueError(f"manual full-source review is incomplete: {source_id}")
        provenance = str(verdict.get("review_provenance") or "")
        if (
            (provenance and not provenance.startswith("human_"))
            or bool(verdict.get("human_review_required"))
            or bool(verdict.get("preaudit_provenance"))
        ):
            raise ValueError(f"Omni-only or unconfirmed verdict cannot become truth: {source_id}")
        if float(verdict.get("frame_hop_s") or 0.0) != FRAME_HOP_S:
            raise ValueError(f"frame hop mismatch: {source_id}")
        frame_count = int(verdict.get("frame_count") or 0)
        expected_frames = int(round(float(source.get("duration_s") or 0.0) / FRAME_HOP_S))
        if frame_count <= 0 or frame_count != expected_frames:
            raise ValueError(f"frame count mismatch: {source_id}")
        reviewed_spans = _validate_spans(
            list(verdict.get("spans") or ()),
            source_id=source_id,
            frame_count=frame_count,
        )
        video_id = str(source.get("video_id") or "")
        if not video_id or video_id != str(partition_row.get("video_id") or ""):
            raise ValueError(f"video identity mismatch: {source_id}")
        previous_partition = seen_video_partition.setdefault(video_id, partition)
        if previous_partition != partition:
            raise ValueError(f"video identity crosses partitions: {video_id}")
        if source_id in seen_source_partition:
            raise ValueError(f"Scorer v11 source identity crosses partitions: {source_id}")
        seen_source_partition[source_id] = partition
        real_core_id = f"real-source-window::{source_id}"
        previous_core_source = seen_core_source.setdefault(real_core_id, source_id)
        if previous_core_source != source_id:
            raise ValueError(f"Scorer v11 core identity is reused: {real_core_id}")
        previous_core_partition = seen_core_partition.setdefault(real_core_id, partition)
        if previous_core_partition != partition:
            raise ValueError(f"Scorer v11 core identity crosses partitions: {real_core_id}")
        audio = _resolve(str(source.get("audio_wav") or ""))
        if not audio.exists():
            raise FileNotFoundError(audio)
        audio_sha = str(source.get("audio_wav_sha256") or "")
        if len(audio_sha) != 64:
            raise ValueError(f"source audio SHA256 is missing: {source_id}")
        if verify_audio and _sha256(audio) != audio_sha:
            raise ValueError(f"source audio SHA256 mismatch: {source_id}")
        audio_sample_count, audio_duration_s, audio_frame_count = _wav_geometry(audio)
        spans, effective_frame_count, geometry_policy = _clip_reviewed_spans_to_audio(
            reviewed_spans,
            source_id=source_id,
            reviewed_frame_count=frame_count,
            audio_frame_count=audio_frame_count,
        )

        for span in spans:
            label_counts[str(span["label"])] += int(span["end_frame"]) - int(
                span["start_frame"]
            )
        partition_counts[partition] += 1
        compiled.append(
            {
                "schema": CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "canonical_label_schema": (
                    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA
                ),
                "source_id": source_id,
                "video_id": video_id,
                "core_ids": [real_core_id],
                "core_identity_kind": "real_source_window_v1",
                "partition": partition,
                "input_distribution": "real_workflow_source_windows",
                "synthetic_composite": False,
                "audio": _display(audio),
                "audio_sha256": audio_sha,
                "duration_s": audio_duration_s,
                "frame_count": effective_frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "audio_sample_count": audio_sample_count,
                "reviewed_nominal_frame_count": frame_count,
                "audio_geometry_policy": geometry_policy,
                "canonical_spans": spans,
                "annotation_provenance": "human_full_source_review",
                "manual_verdict_file": _display(verdict_file_by_source[source_id]),
                "manual_verdict_file_sha256": verdict_shas[
                    _display(verdict_file_by_source[source_id])
                ],
                "source_windows_sha256": source_windows_sha,
                "partition_manifest_sha256": partition_sha,
                "training_manifest_allowed": True,
            }
        )

    if set(partition_counts) != PARTITIONS:
        raise ValueError(
            f"Scorer v11 canonical requires non-empty train/val/test: {dict(partition_counts)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "canonical_sources.jsonl"
    _write_jsonl(canonical_path, compiled)
    canonical_sha = _sha256(canonical_path)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA,
        "canonical_sources": _display(canonical_path),
        "canonical_sources_sha256": canonical_sha,
        "synthetic_train_sources": _display(synthetic_train_sources),
        "synthetic_train_sources_sha256": synthetic_train_sha,
        "real_train_outside_sources": _display(real_train_outside_sources),
        "real_train_outside_sources_sha256": real_train_outside_sha,
        "source_windows": _display(source_windows),
        "source_windows_sha256": source_windows_sha,
        "partition_manifest": _display(partition_manifest),
        "partition_manifest_sha256": partition_sha,
        "manual_verdict_files_sha256": verdict_shas,
        "source_count": len(compiled),
        "partition_counts": dict(sorted(partition_counts.items())),
        "canonical_frame_counts": dict(sorted(label_counts.items())),
        "all_heldout_sources_human_confirmed": True,
        "synthetic_train_truth_exact_composition": True,
        "real_train_outside_truth_requires_asr_empty": True,
        "asr_text_used_as_inside_truth": False,
        "asr_empty_used_without_gemini_outside": False,
        "real_train_unsure_excluded_from_training": True,
        "omni_preverdicts_used_as_truth": False,
        "training_manifest_allowed": True,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-train-sources", required=True)
    parser.add_argument("--real-train-outside-sources", required=True)
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--manual-verdicts", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-audio-content-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return compile_canonical(
        synthetic_train_sources=Path(args.synthetic_train_sources),
        real_train_outside_sources=Path(args.real_train_outside_sources),
        source_windows=Path(args.source_windows),
        partition_manifest=Path(args.partition_manifest),
        manual_verdicts=[Path(path) for path in args.manual_verdicts],
        output_dir=Path(args.output_dir),
        verify_audio=not args.skip_audio_content_check,
    )


if __name__ == "__main__":
    main()
