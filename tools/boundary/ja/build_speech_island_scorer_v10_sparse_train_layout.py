#!/usr/bin/env python3
"""Rebuild train-only Scorer v10 sources with sparse hard-speech layouts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (  # noqa: E402
    CANONICAL_LABELS,
    CANONICAL_LABEL_SCHEMA,
    SOURCE_SCHEMA,
    _validate_sources,
    canonical_frame_labels,
)


SUMMARY_SCHEMA = "speech_scorer_v10_sparse_train_layout_summary_v1"
INPUT_SUMMARY_SCHEMA = "speech_scorer_v10_corrected_canonical_r5_summary_v1"
FRAME_HOP_S = 0.02
SAMPLE_RATE = 16000
LAYOUT_LABEL_SOURCE = "train_sparse_acoustic_layout_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
        while chunk := handle.read(8 * 1024 * 1024):
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


def _rms(values: np.ndarray) -> float:
    audio = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-8)))


def _candidate(source: dict[str, Any]) -> bool:
    spans = list(source.get("canonical_spans") or ())
    if (
        source.get("partition") != "train"
        or source.get("row_role") != "speech"
        or source.get("additive_overlay") is not None
        or [span.get("label") for span in spans]
        != ["speech", "background", "background", "background", "speech"]
    ):
        return False
    if any(
        str(span.get("label_source") or "").startswith("manual_") for span in spans
    ):
        return False
    sample_rate = int(source.get("sample_rate") or 0)
    if sample_rate != SAMPLE_RATE:
        return False
    speech_durations = [
        (int(span["end_sample"]) - int(span["start_sample"])) / sample_rate
        for span in spans
        if span["label"] == "speech"
    ]
    return (
        len(speech_durations) == 2
        and min(speech_durations) >= 0.18
        and max(speech_durations) <= 2.5
    )


def _selection_key(source_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()
    return digest, source_id


def select_sources(
    sources: Sequence[dict[str, Any]], *, source_count: int, seed: int
) -> list[dict[str, Any]]:
    candidates = [source for source in sources if _candidate(source)]
    candidates.sort(key=lambda row: _selection_key(str(row["source_id"]), seed))
    if source_count <= 0 or len(candidates) < source_count:
        raise ValueError(
            f"sparse layout requires {source_count} eligible train sources; "
            f"found {len(candidates)}"
        )
    return candidates[:source_count]


def _load_audio(source: dict[str, Any]) -> np.ndarray:
    path = _resolve(str(source["audio"]))
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != SAMPLE_RATE or len(audio) != int(source["sample_count"]):
        raise ValueError(f"sparse-layout audio identity mismatch: {source['source_id']}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _rebuild_source(
    source: dict[str, Any],
    *,
    output_audio: Path,
    target_db_values: Sequence[float],
    source_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_audio_path = _resolve(str(source["audio"]))
    audio = _load_audio(source)
    original_spans = [copy.deepcopy(span) for span in source["canonical_spans"]]
    pieces = [
        np.ascontiguousarray(
            audio[int(span["start_sample"]): int(span["end_sample"])],
            dtype=np.float32,
        )
        for span in original_spans
    ]
    order = (1, 0, 2, 4, 3)
    reordered = [pieces[index].copy() for index in order]
    reordered_spans = [copy.deepcopy(original_spans[index]) for index in order]
    target_details: list[dict[str, Any]] = []
    for local_index in (1, 3):
        target_db = float(
            target_db_values[(source_index * 2 + len(target_details)) % len(target_db_values)]
        )
        context = np.concatenate((reordered[local_index - 1], reordered[local_index + 1]))
        speech_rms_before = _rms(reordered[local_index])
        context_rms = _rms(context)
        if speech_rms_before <= 1e-8 or context_rms <= 1e-8:
            raise ValueError(f"sparse-layout source has silent material: {source['source_id']}")
        target_rms = context_rms * (10.0 ** (target_db / 20.0))
        scale = target_rms / speech_rms_before
        reordered[local_index] *= np.float32(scale)
        target_details.append(
            {
                "core_id": str(reordered_spans[local_index].get("core_id") or ""),
                "target_speech_to_adjacent_db": target_db,
                "pre_layout_speech_rms_dbfs": _db(speech_rms_before),
                "adjacent_background_rms_dbfs": _db(context_rms),
                "speech_scale": float(scale),
            }
        )

    rebuilt_audio = np.ascontiguousarray(np.concatenate(reordered), dtype=np.float32)
    peak = float(np.max(np.abs(rebuilt_audio), initial=0.0))
    limiter_gain = min(1.0, 0.98 / peak) if peak else 1.0
    rebuilt_audio *= np.float32(limiter_gain)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio, rebuilt_audio, SAMPLE_RATE, subtype="PCM_16")
    written, written_rate = sf.read(output_audio, dtype="float32", always_2d=False)
    if written_rate != SAMPLE_RATE or len(written) != len(audio):
        raise ValueError("sparse-layout output audio identity changed")

    cursor = 0
    for local_index, (span, piece) in enumerate(zip(reordered_spans, reordered, strict=True)):
        original_start = int(span["start_sample"])
        original_end = int(span["end_sample"])
        span["origin_start_sample"] = original_start
        span["origin_end_sample"] = original_end
        span["origin_label_source"] = str(span.get("label_source") or "")
        span["start_sample"] = cursor
        span["end_sample"] = cursor + len(piece)
        span["label_source"] = LAYOUT_LABEL_SOURCE
        span["sparse_layout_role"] = "speech" if local_index in (1, 3) else "background"
        cursor += len(piece)
    if cursor != len(audio):
        raise ValueError("sparse-layout reconstruction changed sample count")

    for detail, local_index in zip(target_details, (1, 3), strict=True):
        start = int(reordered_spans[local_index]["start_sample"])
        end = int(reordered_spans[local_index]["end_sample"])
        left = reordered_spans[local_index - 1]
        right = reordered_spans[local_index + 1]
        written_context = np.concatenate(
            (
                written[int(left["start_sample"]): int(left["end_sample"])],
                written[int(right["start_sample"]): int(right["end_sample"])],
            )
        )
        achieved = _db(_rms(written[start:end])) - _db(_rms(written_context))
        detail["achieved_speech_to_adjacent_db"] = achieved
        if abs(achieved - float(detail["target_speech_to_adjacent_db"])) > 0.05:
            raise ValueError("sparse-layout PCM output missed the requested acoustic ratio")

    rebuilt = copy.deepcopy(source)
    rebuilt["audio"] = _display(output_audio)
    rebuilt["canonical_spans"] = reordered_spans
    rebuilt["additive_overlay"] = None
    rebuilt["training_distribution_reconstruction"] = {
        "schema": "speech_scorer_v10_sparse_train_layout_v1",
        "train_only": True,
        "source_identity_preserved": True,
        "core_identity_preserved": True,
        "partition_identity_preserved": True,
        "original_audio": _display(original_audio_path),
        "original_audio_sha256": _sha256(original_audio_path),
        "component_order": list(order),
        "limiter_gain": float(limiter_gain),
        "speech_targets": target_details,
    }
    detail = {
        "source_id": str(source["source_id"]),
        "partition": str(source["partition"]),
        "core_ids": list(source["core_ids"]),
        "original_audio": _display(original_audio_path),
        "rebuilt_audio": _display(output_audio),
        "original_audio_sha256": _sha256(original_audio_path),
        "rebuilt_audio_sha256": _sha256(output_audio),
        "sample_count": len(written),
        "speech_targets": target_details,
    }
    return rebuilt, detail


def _feature_label(source: dict[str, Any]) -> dict[str, Any]:
    labels = canonical_frame_labels(source, frame_hop_s=FRAME_HOP_S)
    weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0)
    return {
        "audio_id": source["source_id"],
        "source": "scorer_v10_sparse_train_layout_v1",
        "duration_s": source["duration_s"],
        "text": "",
        "teacher_segments": {},
        "frame_hop_s": FRAME_HOP_S,
        "speech_frames": (labels == CANONICAL_LABELS["speech"]).astype(int).tolist(),
        "label_quality": (
            "negative" if source["row_role"] == "all_background" else "supervised"
        ),
        "frame_weights": weights.tolist(),
        "boundary_metadata": {
            "schema": SOURCE_SCHEMA,
            "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
            "row_role": source["row_role"],
            "partition": source["partition"],
            "unsure_frame_count": int(np.sum(labels == CANONICAL_LABELS["unsure"])),
            "training_distribution_reconstruction": bool(
                source.get("training_distribution_reconstruction")
            ),
        },
    }


def _frame_counts(sources: Sequence[dict[str, Any]]) -> Counter[str]:
    result: Counter[str] = Counter()
    inverse = {value: key for key, value in CANONICAL_LABELS.items()}
    for source in sources:
        for value, count in zip(
            *np.unique(canonical_frame_labels(source, frame_hop_s=FRAME_HOP_S), return_counts=True),
            strict=True,
        ):
            result[inverse[int(value)]] += int(count)
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not args.target_db:
        raise ValueError("sparse layout requires at least one target dB value")
    r5_summary_path = Path(args.r5_summary).resolve()
    r5 = _read_json(r5_summary_path)
    if r5.get("schema") != INPUT_SUMMARY_SCHEMA:
        raise ValueError("sparse layout requires the corrected-r5 canonical summary")
    if r5.get("boundary_serialization_contract_id") != ACOUSTIC_BINARY_V12_CONTRACT.contract_id:
        raise ValueError("sparse layout requires the central Boundary contract")
    canonical_path = _resolve(str(r5.get("canonical_sources") or ""))
    if _sha256(canonical_path) != str(r5.get("canonical_sources_sha256") or ""):
        raise ValueError("corrected-r5 canonical SHA256 mismatch")
    sources = _read_jsonl(canonical_path)
    _validate_sources(sources)
    selected = select_sources(sources, source_count=args.source_count, seed=args.seed)
    selected_ids = {str(source["source_id"]) for source in selected}

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    rebuilt_by_id: dict[str, dict[str, Any]] = {}
    changes: list[dict[str, Any]] = []
    for index, source in enumerate(selected):
        rebuilt, detail = _rebuild_source(
            source,
            output_audio=audio_dir / f"{source['source_id']}.wav",
            target_db_values=args.target_db,
            source_index=index,
        )
        rebuilt_by_id[str(source["source_id"])] = rebuilt
        changes.append(detail)
    rebuilt_sources = [rebuilt_by_id.get(str(row["source_id"]), row) for row in sources]
    dataset = _validate_sources(rebuilt_sources)

    for before, after in zip(sources, rebuilt_sources, strict=True):
        changed = str(before["source_id"]) in selected_ids
        if changed:
            if before["partition"] != "train" or after["partition"] != "train":
                raise ValueError("sparse layout may only alter train sources")
            if before["core_ids"] != after["core_ids"]:
                raise ValueError("sparse layout changed core identity")
        elif before != after:
            raise ValueError("sparse layout changed an unselected source")

    labels = [_feature_label(source) for source in rebuilt_sources]
    labels_by_id = {str(row["audio_id"]): row for row in labels}
    audio_manifest = [
        {
            "audio_id": source["source_id"],
            "audio": source["audio"],
            "partition": source["partition"],
            "row_role": source["row_role"],
        }
        for source in rebuilt_sources
    ]
    changed_audio_manifest = [
        row for row in audio_manifest if str(row["audio_id"]) in selected_ids
    ]
    changed_labels = [labels_by_id[source_id] for source_id in selected_ids]
    changed_labels.sort(key=lambda row: str(row["audio_id"]))
    changed_audio_manifest.sort(key=lambda row: str(row["audio_id"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output = output_dir / "canonical_sources.jsonl"
    labels_output = output_dir / "feature_cache_labels.jsonl"
    audio_output = output_dir / "audio_manifest.json"
    changed_sources_output = output_dir / "changed_sources.jsonl"
    changed_labels_output = output_dir / "changed_feature_cache_labels.jsonl"
    changed_audio_output = output_dir / "changed_audio_manifest.json"
    _write_jsonl(canonical_output, rebuilt_sources)
    _write_jsonl(labels_output, labels)
    _write_jsonl(changed_sources_output, changes)
    _write_jsonl(changed_labels_output, changed_labels)
    audio_output.write_text(
        json.dumps(audio_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    changed_audio_output.write_text(
        json.dumps(changed_audio_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    before_counts = _frame_counts(sources)
    after_counts = _frame_counts(rebuilt_sources)
    achieved_values = [
        float(target["achieved_speech_to_adjacent_db"])
        for change in changes
        for target in change["speech_targets"]
    ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "input_r5_summary": _display(r5_summary_path),
        "input_r5_summary_sha256": _sha256(r5_summary_path),
        "input_r5_canonical_sources": _display(canonical_path),
        "input_r5_canonical_sources_sha256": _sha256(canonical_path),
        "dataset": dataset,
        "selected_source_count": len(selected_ids),
        "selected_core_count": sum(len(source["core_ids"]) for source in selected),
        "selected_source_ids": sorted(selected_ids),
        "selection_seed": int(args.seed),
        "selection_contract": "eligible_clean_train_source_stable_hash_v1",
        "layout_contract": "background_speech_background_speech_background_v1",
        "target_speech_to_adjacent_db": [float(value) for value in args.target_db],
        "achieved_speech_to_adjacent_db_min": min(achieved_values),
        "achieved_speech_to_adjacent_db_max": max(achieved_values),
        "canonical_frame_counts_before": dict(before_counts),
        "canonical_frame_counts_after": dict(after_counts),
        "canonical_frame_count_delta": {
            label: int(after_counts[label] - before_counts[label])
            for label in ("speech", "background", "unsure")
        },
        "canonical_sources": _display(canonical_output),
        "canonical_sources_sha256": _sha256(canonical_output),
        "feature_cache_labels": _display(labels_output),
        "feature_cache_labels_sha256": _sha256(labels_output),
        "audio_manifest": _display(audio_output),
        "audio_manifest_sha256": _sha256(audio_output),
        "changed_sources": _display(changed_sources_output),
        "changed_sources_sha256": _sha256(changed_sources_output),
        "changed_feature_cache_labels": _display(changed_labels_output),
        "changed_feature_cache_labels_sha256": _sha256(changed_labels_output),
        "changed_audio_manifest": _display(changed_audio_output),
        "changed_audio_manifest_sha256": _sha256(changed_audio_output),
        "audio_bytes_changed": True,
        "changed_partition": "train",
        "heldout_audio_identity_changed": False,
        "source_identity_changed": False,
        "core_identity_changed": False,
        "partition_identity_changed": False,
        "max_core_use_count": int(dataset["max_core_use_count"]),
        "unsure_training_mapping": -100,
        "feature_cache_labels_ready": True,
        "incremental_feature_extraction_required": True,
        "training_manifest_ready": False,
        "checkpoint_promotion_authorized": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-count", type=int, default=22)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--target-db", action="append", type=float)
    args = parser.parse_args(argv)
    if args.target_db is None:
        args.target_db = [-8.0, -4.0, 0.0, 3.0]
    return args


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False))
