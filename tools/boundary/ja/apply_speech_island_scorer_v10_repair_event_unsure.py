#!/usr/bin/env python3
"""Apply explicit human unsure verdicts to every occurrence of a repair event."""
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.boundary.ja.build_speech_island_scorer_v10_sparse_train_layout import (  # noqa: E402
    SUMMARY_SCHEMA as R6_SUMMARY_SCHEMA,
)
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (  # noqa: E402
    CANONICAL_LABELS,
    CANONICAL_LABEL_SCHEMA,
    SOURCE_SCHEMA,
    _validate_sources,
    canonical_frame_labels,
)


VERDICT_SCHEMA = "speech_scorer_v10_repair_event_ambiguity_manual_verdict_v1"
SUMMARY_SCHEMA = "speech_scorer_v10_repair_event_unsure_summary_v1"
FRAME_HOP_S = 0.02


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def _occurrence_key(source_id: str, span: dict[str, Any]) -> tuple[str, int, int]:
    return source_id, int(span["start_sample"]), int(span["end_sample"])


def _feature_label(source: dict[str, Any]) -> dict[str, Any]:
    labels = canonical_frame_labels(source, frame_hop_s=FRAME_HOP_S)
    weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0)
    return {
        "audio_id": source["source_id"],
        "source": "scorer_v10_repair_event_unsure_v1",
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
            "repair_event_unsure_override": bool(
                source.get("repair_event_unsure_override")
            ),
        },
    }


def _frame_counts(sources: Sequence[dict[str, Any]]) -> Counter[str]:
    inverse = {value: key for key, value in CANONICAL_LABELS.items()}
    result: Counter[str] = Counter()
    for source in sources:
        values, counts = np.unique(
            canonical_frame_labels(source, frame_hop_s=FRAME_HOP_S), return_counts=True
        )
        for value, count in zip(values, counts, strict=True):
            result[inverse[int(value)]] += int(count)
    return result


def apply_unsure(
    *, input_summary_path: Path, verdicts_path: Path, output_dir: Path
) -> dict[str, Any]:
    source_summary = _json(input_summary_path)
    if source_summary.get("schema") != R6_SUMMARY_SCHEMA:
        raise ValueError("repair-event unsure currently requires sparse-layout r6")
    if source_summary.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("repair-event unsure requires the central Boundary contract")
    canonical_path = _resolve(str(source_summary.get("canonical_sources") or ""))
    if _sha256(canonical_path) != str(source_summary.get("canonical_sources_sha256") or ""):
        raise ValueError("input canonical SHA256 mismatch")
    sources = _rows(canonical_path)
    _validate_sources(sources)

    verdicts = _rows(verdicts_path)
    if not verdicts:
        raise ValueError("repair-event unsure verdict file is empty")
    by_event: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        event_id = str(verdict.get("repair_event_id") or "")
        if (
            verdict.get("schema") != VERDICT_SCHEMA
            or verdict.get("verdict") != "unsure"
            or not event_id
            or event_id in by_event
        ):
            raise ValueError("repair-event unsure verdict is invalid or duplicated")
        by_event[event_id] = verdict

    actual: dict[str, set[tuple[str, int, int]]] = {event: set() for event in by_event}
    ignored_core_ids: set[str] = set()
    corrected: list[dict[str, Any]] = []
    changed_sources: set[str] = set()
    for original in sources:
        source = copy.deepcopy(original)
        retained_core_ids = list(source.get("core_ids") or ())
        source_events: set[str] = set()
        for span in source["canonical_spans"]:
            event_id = str(span.get("repair_event_id") or "")
            if event_id not in by_event:
                continue
            if span.get("label") != "speech" or not span.get("core_id"):
                raise ValueError("repair-event unsure target is not a registered speech span")
            key = _occurrence_key(str(source["source_id"]), span)
            if key in actual[event_id]:
                raise ValueError("repair-event occurrence is duplicated")
            actual[event_id].add(key)
            core_id = str(span.pop("core_id"))
            ignored_core_ids.add(core_id)
            retained_core_ids = [value for value in retained_core_ids if value != core_id]
            span["label"] = "unsure"
            span["label_source"] = "manual_repair_event_ambiguity_unsure_v1"
            span["ignored_core_id"] = core_id
            span["ambiguity_reason"] = str(by_event[event_id].get("reason") or "")
            span["ambiguity_verdict_schema"] = VERDICT_SCHEMA
            source_events.add(event_id)
        if source_events:
            changed_sources.add(str(source["source_id"]))
            source["core_ids"] = retained_core_ids
            source["repair_event_unsure_override"] = {
                "verdicts": _display(verdicts_path),
                "verdicts_sha256": _sha256(verdicts_path),
                "repair_event_ids": sorted(source_events),
                "training_mapping": -100,
            }
            if not any(span["label"] == "speech" for span in source["canonical_spans"]):
                if retained_core_ids:
                    raise ValueError("speech-free unsure row retains a core identity")
                background_id = str(
                    source.get("repaired_background_id")
                    or next(
                        (
                            span.get("origin_background_id")
                            for span in source["canonical_spans"]
                            if span.get("origin_background_id")
                        ),
                        "",
                    )
                )
                if not background_id:
                    raise ValueError("speech-free unsure row cannot restore background identity")
                source["row_role"] = "all_background"
                source["background_id"] = background_id
        corrected.append(source)

    for event_id, verdict in by_event.items():
        expected = {
            (
                str(item.get("source_id") or ""),
                int(item.get("start_sample") or -1),
                int(item.get("end_sample") or -1),
            )
            for item in verdict.get("reviewed_occurrences") or ()
        }
        if not expected or expected != actual[event_id]:
            raise ValueError(
                f"repair-event unsure reviewed/actual occurrences differ: {event_id}"
            )
    dataset = _validate_sources(corrected)
    if not changed_sources:
        raise ValueError("repair-event unsure verdict changed no canonical source")

    labels = [_feature_label(source) for source in corrected]
    audio_manifest = [
        {
            "audio_id": source["source_id"],
            "audio": source["audio"],
            "partition": source["partition"],
            "row_role": source["row_role"],
        }
        for source in corrected
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output = output_dir / "canonical_sources.jsonl"
    labels_output = output_dir / "feature_cache_labels.jsonl"
    audio_output = output_dir / "audio_manifest.json"
    _write_rows(canonical_output, corrected)
    _write_rows(labels_output, labels)
    audio_output.write_text(
        json.dumps(audio_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    before = _frame_counts(sources)
    after = _frame_counts(corrected)
    result = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "input_summary": _display(input_summary_path),
        "input_summary_sha256": _sha256(input_summary_path),
        "input_canonical_sources": _display(canonical_path),
        "input_canonical_sources_sha256": _sha256(canonical_path),
        "manual_verdicts": _display(verdicts_path),
        "manual_verdicts_sha256": _sha256(verdicts_path),
        "repair_event_ids": sorted(by_event),
        "changed_source_ids": sorted(changed_sources),
        "changed_source_count": len(changed_sources),
        "ignored_core_ids": sorted(ignored_core_ids),
        "ignored_core_count": len(ignored_core_ids),
        "dataset": dataset,
        "canonical_frame_counts_before": dict(before),
        "canonical_frame_counts_after": dict(after),
        "canonical_frame_count_delta": {
            label: int(after[label] - before[label])
            for label in ("speech", "background", "unsure")
        },
        "canonical_sources": _display(canonical_output),
        "canonical_sources_sha256": _sha256(canonical_output),
        "feature_cache_labels": _display(labels_output),
        "feature_cache_labels_sha256": _sha256(labels_output),
        "audio_manifest": _display(audio_output),
        "audio_manifest_sha256": _sha256(audio_output),
        "audio_bytes_changed": False,
        "source_identity_changed": False,
        "partition_identity_changed": False,
        "heldout_audio_identity_changed": False,
        "unsure_training_mapping": -100,
        "feature_cache_reuse_pending_rebind": True,
        "training_manifest_ready": False,
        "checkpoint_promotion_authorized": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", required=True)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            apply_unsure(
                input_summary_path=Path(args.input_summary),
                verdicts_path=Path(args.verdicts),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
