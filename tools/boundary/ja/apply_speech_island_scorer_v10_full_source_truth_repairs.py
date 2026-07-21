#!/usr/bin/env python3
"""Apply complete human full-source truth to Scorer v10 background rows."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.audits.evaluate_scorer_v10_full_source_span_audit import (  # noqa: E402
    DECISION_SCHEMA,
    GATE_SCHEMA,
)
from tools.boundary.ja.apply_speech_island_scorer_v10_repair_event_unsure import (  # noqa: E402
    SUMMARY_SCHEMA as REPAIR_EVENT_UNSURE_SUMMARY_SCHEMA,
)
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (  # noqa: E402
    CANONICAL_LABELS,
    CANONICAL_LABEL_SCHEMA,
    SOURCE_SCHEMA,
    _validate_sources,
    canonical_frame_labels,
)


SUMMARY_SCHEMA = "speech_scorer_v10_full_source_truth_repair_summary_v1"
REPAIR_METADATA_SCHEMA = "speech_scorer_v10_full_source_truth_repair_v1"
FRAME_HOP_S = 0.02
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = 320
LABELS = {"background", "speech", "unsure"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def _require_bound_file(payload: Mapping[str, Any], field: str) -> Path:
    path = _resolve(str(payload.get(field) or ""))
    if not path.is_file():
        raise ValueError(f"full-source repair evidence is missing: {field}")
    if _sha256(path) != str(payload.get(f"{field}_sha256") or ""):
        raise ValueError(f"full-source repair {field} SHA256 mismatch")
    return path


def _validate_decision_spans(
    decision: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    source_id = str(decision.get("source_id") or "")
    frame_count = int(decision.get("frame_count") or 0)
    if frame_count <= 0:
        raise ValueError(f"full-source decision has no frames: {source_id}")
    if abs(float(decision.get("frame_hop_s") or 0.0) - FRAME_HOP_S) > 1e-9:
        raise ValueError(f"full-source decision frame hop changed: {source_id}")
    spans = decision.get("spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"full-source decision has no spans: {source_id}")
    normalized: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    cursor = 0
    previous = ""
    for index, raw in enumerate(spans):
        if not isinstance(raw, dict):
            raise ValueError(f"full-source decision span is invalid: {source_id}")
        label = str(raw.get("label") or "")
        start = int(raw.get("start_frame") if raw.get("start_frame") is not None else -1)
        end = int(raw.get("end_frame") if raw.get("end_frame") is not None else -1)
        if label not in LABELS:
            raise ValueError(f"full-source decision label is invalid: {source_id}:{label}")
        if start != cursor or end <= start or end > frame_count:
            raise ValueError(
                f"full-source decision must be ordered and gap-free: {source_id}:{index}"
            )
        if label == previous:
            raise ValueError(
                f"full-source decision has adjacent identical labels: {source_id}:{index}"
            )
        start_s = round(start * FRAME_HOP_S, 6)
        end_s = round(end * FRAME_HOP_S, 6)
        if abs(float(raw.get("start_s") or 0.0) - start_s) > 1e-6:
            raise ValueError(f"full-source decision start_s mismatch: {source_id}:{index}")
        if abs(float(raw.get("end_s") or 0.0) - end_s) > 1e-6:
            raise ValueError(f"full-source decision end_s mismatch: {source_id}:{index}")
        normalized.append(
            {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "start_s": start_s,
                "end_s": end_s,
            }
        )
        counts[label] += end - start
        cursor = end
        previous = label
    if cursor != frame_count:
        raise ValueError(f"full-source decision does not reach the final frame: {source_id}")
    expected_verdict = (
        "complete_with_target_speech"
        if counts["speech"]
        else (
            "complete_with_unsure_only"
            if counts["unsure"]
            else "complete_all_background"
        )
    )
    if str(decision.get("verdict") or "") != expected_verdict:
        raise ValueError(f"full-source decision verdict/labels differ: {source_id}")
    return normalized, counts


def _load_gate(
    manual_gate_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    gate = _json(manual_gate_path)
    if gate.get("schema") != GATE_SCHEMA:
        raise ValueError("full-source repair requires the strict manual gate")
    if gate.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("full-source repair gate uses another Boundary contract")
    if (
        gate.get("manual_gate_passed") is not True
        or gate.get("canonical_recompile_allowed") is not True
        or gate.get("training_manifest_allowed") is not False
        or gate.get("all_reviewed_sources_have_gap_free_full_coverage") is not True
        or gate.get("model_output_used_as_truth") is not False
        or gate.get("asr_output_used_as_truth") is not False
        or list(gate.get("unreviewed_source_ids") or ())
    ):
        raise ValueError("full-source repair gate is incomplete or unsafe")
    _require_bound_file(gate, "audit_manifest")
    _require_bound_file(gate, "manual_verdicts")
    decisions_path = _require_bound_file(gate, "decisions")
    decisions = _rows(decisions_path)
    if not decisions:
        raise ValueError("full-source repair decision manifest is empty")

    seen: set[str] = set()
    label_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []
    for raw in decisions:
        source_id = str(raw.get("source_id") or "")
        if (
            raw.get("schema") != DECISION_SCHEMA
            or raw.get("boundary_serialization_contract_id")
            != ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            or raw.get("model_output_used_as_truth") is not False
            or raw.get("asr_output_used_as_truth") is not False
            or not source_id
            or source_id in seen
        ):
            raise ValueError("full-source repair decision is invalid or duplicated")
        spans, counts = _validate_decision_spans(raw)
        decision = dict(raw)
        decision["spans"] = spans
        normalized.append(decision)
        seen.add(source_id)
        label_counts.update(counts)
        verdict_counts[str(raw["verdict"])] += 1
    if (
        int(gate.get("source_count") or 0) != len(normalized)
        or int(gate.get("reviewed_source_count") or 0) != len(normalized)
        or dict(sorted(label_counts.items()))
        != dict(sorted((gate.get("label_frame_counts") or {}).items()))
        or dict(sorted(verdict_counts.items()))
        != dict(sorted((gate.get("verdict_counts") or {}).items()))
    ):
        raise ValueError("full-source repair gate summary does not match its decisions")
    return gate, normalized, _sha256(manual_gate_path), _sha256(decisions_path)


def _frame_counts(sources: Sequence[dict[str, Any]]) -> Counter[str]:
    inverse = {value: key for key, value in CANONICAL_LABELS.items()}
    result: Counter[str] = Counter()
    for source in sources:
        values, counts = np.unique(canonical_frame_labels(source), return_counts=True)
        for value, count in zip(values, counts, strict=True):
            result[inverse[int(value)]] += int(count)
    return result


def _core_id(
    *, gate_sha256: str, decisions_sha256: str, source_id: str, start: int, end: int
) -> str:
    payload = (
        f"{REPAIR_METADATA_SCHEMA}\0{ACOUSTIC_BINARY_V12_CONTRACT.contract_id}\0"
        f"{gate_sha256}\0{decisions_sha256}\0{source_id}\0{start}\0{end}"
    ).encode("utf-8")
    return "scorer-v10-full-source-core-" + hashlib.sha256(payload).hexdigest()


def _feature_label(
    source: dict[str, Any], *, gate_path: Path, gate_sha256: str
) -> dict[str, Any]:
    labels = canonical_frame_labels(source)
    weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0)
    return {
        "audio_id": source["source_id"],
        "source": "scorer_v10_full_source_truth_repair_v1",
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
            "unsure_frame_count": int(
                np.sum(labels == CANONICAL_LABELS["unsure"])
            ),
            "full_source_truth_repair": {
                "schema": REPAIR_METADATA_SCHEMA,
                "manual_gate": _display(gate_path),
                "manual_gate_sha256": gate_sha256,
                "model_output_used_as_truth": False,
                "asr_output_used_as_truth": False,
            },
        },
    }


def apply_repairs(
    *, input_summary_path: Path, manual_gate_path: Path, output_dir: Path
) -> dict[str, Any]:
    source_summary = _json(input_summary_path)
    if source_summary.get("schema") not in {
        REPAIR_EVENT_UNSURE_SUMMARY_SCHEMA,
        SUMMARY_SCHEMA,
    }:
        raise ValueError("full-source truth repair requires an approved canonical summary")
    if source_summary.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("input canonical uses another Boundary contract")
    if (
        source_summary.get("audio_bytes_changed") is not False
        or source_summary.get("source_identity_changed") is not False
        or source_summary.get("partition_identity_changed") is not False
        or source_summary.get("training_manifest_ready") is not False
        or source_summary.get("checkpoint_promotion_authorized") is not False
    ):
        raise ValueError("input canonical is not a closed label-only r7 revision")
    canonical_path = _require_bound_file(source_summary, "canonical_sources")
    audio_manifest_path = _require_bound_file(source_summary, "audio_manifest")
    feature_labels_path = _require_bound_file(source_summary, "feature_cache_labels")
    sources = _rows(canonical_path)
    input_dataset = _validate_sources(sources)
    for source in sources:
        if (
            source.get("schema") != SOURCE_SCHEMA
            or source.get("canonical_label_schema") != CANONICAL_LABEL_SCHEMA
        ):
            raise ValueError("input canonical source schema changed")

    audio_manifest = _json(audio_manifest_path)
    feature_labels = _rows(feature_labels_path)
    if not isinstance(audio_manifest, list):
        raise ValueError("input audio manifest is not a list")
    source_ids = [str(source["source_id"]) for source in sources]
    if [str(row.get("audio_id") or "") for row in audio_manifest] != source_ids:
        raise ValueError("input audio manifest order/identity differs from canonical")
    if [str(row.get("audio_id") or "") for row in feature_labels] != source_ids:
        raise ValueError("input feature label order/identity differs from canonical")
    for source, audio_row in zip(sources, audio_manifest, strict=True):
        for source_field, audio_field in (
            ("audio", "audio"),
            ("partition", "partition"),
            ("row_role", "row_role"),
        ):
            if source.get(source_field) != audio_row.get(audio_field):
                raise ValueError(f"input audio manifest {audio_field} mismatch")

    gate, decisions, gate_sha256, decisions_sha256 = _load_gate(manual_gate_path)
    by_source = {str(row["source_id"]): row for row in decisions}
    available = {str(row["source_id"]) for row in sources}
    missing = sorted(set(by_source) - available)
    if missing:
        raise ValueError(f"full-source repair target is missing: {missing}")

    corrected: list[dict[str, Any]] = []
    corrected_audio_manifest = copy.deepcopy(audio_manifest)
    corrected_feature_labels = copy.deepcopy(feature_labels)
    added_core_ids: list[str] = []
    target_transition_counts: Counter[str] = Counter()
    requested_counts: Counter[str] = Counter()
    changed_source_ids: list[str] = []
    allowed_changed_fields = {
        "background_id",
        "canonical_repair_contract",
        "canonical_repair_gate",
        "canonical_spans",
        "core_ids",
        "full_source_truth_repair",
        "row_role",
    }
    for index, original in enumerate(sources):
        source_id = str(original["source_id"])
        decision = by_source.get(source_id)
        if decision is None:
            corrected.append(copy.deepcopy(original))
            continue
        if str(decision.get("partition") or "") != str(original["partition"]):
            raise ValueError(f"full-source repair partition mismatch: {source_id}")
        if int(original.get("sample_rate") or 0) != SAMPLE_RATE:
            raise ValueError(f"full-source repair requires 16 kHz audio: {source_id}")
        before_labels = canonical_frame_labels(original)
        if int(decision["frame_count"]) != len(before_labels):
            raise ValueError(f"full-source repair frame extent mismatch: {source_id}")
        original_spans = list(original.get("canonical_spans") or ())
        background_id = str(original.get("background_id") or "")
        if (
            original.get("row_role") != "all_background"
            or list(original.get("core_ids") or ())
            or not background_id
            or len(original_spans) != 1
            or original_spans[0].get("label") != "background"
            or int(
                original_spans[0].get("start_sample")
                if original_spans[0].get("start_sample") is not None
                else -1
            )
            != 0
            or int(
                original_spans[0].get("end_sample")
                if original_spans[0].get("end_sample") is not None
                else -1
            )
            != int(original["sample_count"])
        ):
            raise ValueError(
                f"full-source truth may only replace an exact all-background row: {source_id}"
            )

        source = copy.deepcopy(original)
        canonical_spans: list[dict[str, Any]] = []
        core_ids: list[str] = []
        expected_labels: list[int] = []
        for span in decision["spans"]:
            label = str(span["label"])
            start_frame = int(span["start_frame"])
            end_frame = int(span["end_frame"])
            start_sample = min(int(source["sample_count"]), start_frame * SAMPLES_PER_FRAME)
            end_sample = min(int(source["sample_count"]), end_frame * SAMPLES_PER_FRAME)
            if end_sample <= start_sample:
                raise ValueError(f"full-source repair produced an empty sample span: {source_id}")
            canonical_span: dict[str, Any] = {
                "start_sample": start_sample,
                "end_sample": end_sample,
                "label": label,
                "label_source": "manual_full_source_truth_repair_v1",
                "full_source_start_frame": start_frame,
                "full_source_end_frame": end_frame,
                "manual_gate_sha256": gate_sha256,
                "decisions_sha256": decisions_sha256,
            }
            if label == "speech":
                core_id = _core_id(
                    gate_sha256=gate_sha256,
                    decisions_sha256=decisions_sha256,
                    source_id=source_id,
                    start=start_frame,
                    end=end_frame,
                )
                canonical_span["core_id"] = core_id
                canonical_span["origin_background_id"] = background_id
                core_ids.append(core_id)
                added_core_ids.append(core_id)
            elif label == "background":
                canonical_span["background_id"] = background_id
            else:
                canonical_span["origin_background_id"] = background_id
                canonical_span["training_label"] = -100
            canonical_spans.append(canonical_span)
            expected_labels.extend(
                [CANONICAL_LABELS[label]] * (end_frame - start_frame)
            )
            requested_counts[label] += end_frame - start_frame

        source["canonical_spans"] = canonical_spans
        source["core_ids"] = core_ids
        source["row_role"] = "speech" if core_ids else "all_background"
        source["background_id"] = "" if core_ids else background_id
        source["canonical_repair_contract"] = REPAIR_METADATA_SCHEMA
        source["canonical_repair_gate"] = _display(manual_gate_path)
        source["full_source_truth_repair"] = {
            "schema": REPAIR_METADATA_SCHEMA,
            "manual_gate": _display(manual_gate_path),
            "manual_gate_sha256": gate_sha256,
            "decisions": _display(_resolve(str(gate["decisions"]))),
            "decisions_sha256": decisions_sha256,
            "model_output_used_as_truth": False,
            "asr_output_used_as_truth": False,
            "unmarked_complement_used_only_after_full_review": True,
        }
        changed_fields = {
            key
            for key in set(original) | set(source)
            if original.get(key) != source.get(key)
        }
        if not changed_fields <= allowed_changed_fields:
            raise AssertionError(f"full-source repair changed protected fields: {source_id}")
        after_labels = canonical_frame_labels(source)
        if after_labels.tolist() != expected_labels:
            raise ValueError(f"full-source repair frame projection mismatch: {source_id}")
        inverse = {value: name for name, value in CANONICAL_LABELS.items()}
        for before, after in zip(before_labels, after_labels, strict=True):
            target_transition_counts[f"{inverse[int(before)]}_to_{inverse[int(after)]}"] += 1
        corrected.append(source)
        corrected_audio_manifest[index]["row_role"] = source["row_role"]
        corrected_feature_labels[index] = _feature_label(
            source, gate_path=manual_gate_path, gate_sha256=gate_sha256
        )
        changed_source_ids.append(source_id)

    if set(changed_source_ids) != set(by_source):
        raise AssertionError("full-source repair did not apply every decision")
    if not changed_source_ids:
        raise ValueError("full-source repair changed no source")
    if len(added_core_ids) != len(set(added_core_ids)):
        raise ValueError("full-source repair generated a duplicate core identity")
    if dict(sorted(requested_counts.items())) != dict(
        sorted((gate.get("label_frame_counts") or {}).items())
    ):
        raise ValueError("full-source repair applied different labels than the gate")
    output_dataset = _validate_sources(corrected)
    if [str(row["source_id"]) for row in corrected] != source_ids:
        raise AssertionError("full-source repair changed source order or identity")
    for original, source in zip(sources, corrected, strict=True):
        if str(original["source_id"]) not in by_source and original != source:
            raise AssertionError("full-source repair changed a non-target source")

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output = output_dir / "canonical_sources.jsonl"
    labels_output = output_dir / "feature_cache_labels.jsonl"
    audio_output = output_dir / "audio_manifest.json"
    _write_rows(canonical_output, corrected)
    _write_rows(labels_output, corrected_feature_labels)
    audio_output.write_text(
        json.dumps(corrected_audio_manifest, ensure_ascii=False, indent=2) + "\n",
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
        "manual_gate": _display(manual_gate_path),
        "manual_gate_sha256": gate_sha256,
        "decisions": _display(_resolve(str(gate["decisions"]))),
        "decisions_sha256": decisions_sha256,
        "model_output_used_as_truth": False,
        "asr_output_used_as_truth": False,
        "changed_source_ids": sorted(changed_source_ids),
        "changed_source_count": len(changed_source_ids),
        "added_core_ids": sorted(added_core_ids),
        "added_core_count": len(added_core_ids),
        "requested_label_frame_counts": dict(sorted(requested_counts.items())),
        "changed_frame_transition_counts": dict(
            sorted(target_transition_counts.items())
        ),
        "input_dataset": input_dataset,
        "dataset": output_dataset,
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
        "source_order_changed": False,
        "partition_identity_changed": False,
        "heldout_audio_identity_changed": False,
        "non_target_source_rows_changed": 0,
        "max_core_use_count": int(output_dataset["max_core_use_count"]),
        "unsure_training_mapping": -100,
        "feature_cache_reuse_pending_rebind": True,
        "training_manifest_ready": False,
        "checkpoint_promotion_authorized": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", required=True)
    parser.add_argument("--manual-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            apply_repairs(
                input_summary_path=Path(args.input_summary),
                manual_gate_path=Path(args.manual_gate),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
