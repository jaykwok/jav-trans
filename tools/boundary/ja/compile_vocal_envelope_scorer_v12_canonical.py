#!/usr/bin/env python3
"""Compile validated Scorer v12 dual-evidence into a frozen canonical set."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_DATASET_CONTRACT,
    VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    VOCAL_ENVELOPE_SCORER_V12_LABELS,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
)

CONTRACT_ID = "boundary_acoustic_binary_v12"
FRAME_HOP_S = 0.02
EXPECTED_MODEL = "google/gemini-3.6-flash"
EXPECTED_PROFILE = "gemini"
EXPECTED_REASONING = "medium"
EXPECTED_MAX_TOKENS = 8192
EXPECTED_TIMESTAMP_CONTRACT = "omni_audio_timestamp_mmss_mmm_v1"
EXPECTED_EXECUTION_CONTRACT = "gemini_openrouter_reasoning_require_parameters_v1"
EXPECTED_PROMPT_PROFILE = "vocal-envelope-protect-nonvocal-v1"
EXPECTED_PROTECT_PROMPT_VERSION = "vocal-envelope-protect-v1-gemini36-medium-mmss"
EXPECTED_NONVOCAL_PROMPT_VERSION = "vocal-envelope-nonvocal-v1-gemini36-medium-mmss"
EXPECTED_PROMPT_VERSION = f"{EXPECTED_PROTECT_PROMPT_VERSION}__{EXPECTED_NONVOCAL_PROMPT_VERSION}"
OUTPUT_SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_canonical_compile_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


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


def _index(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in output:
            raise ValueError(f"{label} requires unique non-empty {key}: {value!r}")
        output[value] = dict(row)
    return output


def _normalize_spans(evidence: Mapping[str, Any], *, frame_count: int, source_id: str) -> list[dict[str, Any]]:
    raw = [
        *list(evidence.get("vocal_spans") or ()),
        *list(evidence.get("non_vocal_spans") or ()),
        *list(evidence.get("unsure_spans") or ()),
    ]
    spans = sorted(
        (
            {
                "label": str(item.get("label") or "unsure"),
                "start_frame": int(item.get("start_frame") or 0),
                "end_frame": int(item.get("end_frame") or 0),
                "start_s": round(int(item.get("start_frame") or 0) * FRAME_HOP_S, 6),
                "end_s": round(int(item.get("end_frame") or 0) * FRAME_HOP_S, 6),
                **({"category": str(item["category"])} if item.get("category") else {}),
                **({"reason": str(item["reason"])} if item.get("reason") else {}),
            }
            for item in raw
        ),
        key=lambda item: (item["start_frame"], item["end_frame"], item["label"]),
    )
    cursor = 0
    for item in spans:
        if item["label"] == "vocal_candidate":
            expected = VOCAL_ENVELOPE_SCORER_V12_LABELS[1]
        elif item["label"] == "non_vocal_candidate":
            expected = VOCAL_ENVELOPE_SCORER_V12_LABELS[0]
        elif item["label"] == "unsure":
            expected = "unsure"
        else:
            raise ValueError(f"unsupported v12 label {item['label']!r}: {source_id}")
        item["label"] = expected
        if item["start_frame"] != cursor or item["end_frame"] <= cursor or item["end_frame"] > frame_count:
            raise ValueError(f"v12 evidence must form contiguous source coverage: {source_id}")
        cursor = item["end_frame"]
    if cursor != frame_count:
        raise ValueError(f"v12 evidence misses source frames: {source_id}")
    return spans


def _validate_partition_and_core(sources: Mapping[str, Mapping[str, Any]]) -> None:
    seen_core: set[str] = set()
    video_partitions: dict[str, str] = {}
    for source_id, source in sources.items():
        partition = str(source.get("partition") or "")
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"invalid v12 partition: {source_id}")
        video_id = str(source.get("video_id") or "")
        if not video_id:
            raise ValueError(f"v12 source has no frozen video_id: {source_id}")
        previous = video_partitions.setdefault(video_id, partition)
        if previous != partition:
            raise ValueError(f"v12 video crosses partitions: {video_id}")
        cores = source.get("core_ids") or source.get("core_id") or []
        if isinstance(cores, str):
            cores = [cores]
        values = [str(item) for item in cores if str(item)]
        if len(values) != 1 or values[0] in seen_core:
            raise ValueError(f"v12 core is missing or reused: {source_id}")
        seen_core.add(values[0])


def compile_canonical(*, manifest: Path, preaudit: Path, output_dir: Path, allow_teacher_supervision: bool = False) -> dict[str, Any]:
    manifest = manifest.resolve()
    preaudit = preaudit.resolve()
    sources = _index(_rows(manifest), "source_id", "v12 source manifest")
    labels = _index(_rows(preaudit), "source_id", "v12 preaudit")
    if set(sources) != set(labels):
        raise ValueError("v12 source manifest and preaudit IDs must match exactly")
    _validate_partition_and_core(sources)
    manifest_sha = _sha256(manifest)
    preaudit_sha = _sha256(preaudit)
    compiled: list[dict[str, Any]] = []
    totals = Counter()
    partitions = Counter()
    for source_id in sorted(sources):
        source = sources[source_id]
        evidence = labels[source_id]
        if evidence.get("schema") != VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA:
            raise ValueError(f"wrong v12 preaudit schema: {source_id}")
        if evidence.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError(f"wrong v12 central contract: {source_id}")
        if evidence.get("teacher_failed_closed") is True:
            raise ValueError(f"failed-closed v12 evidence cannot compile: {source_id}")
        if evidence.get("model") != EXPECTED_MODEL or evidence.get("provider_profile") != EXPECTED_PROFILE:
            raise ValueError(f"v12 teacher model/profile mismatch: {source_id}")
        for field, expected in (("reasoning_effort", EXPECTED_REASONING), ("max_tokens", EXPECTED_MAX_TOKENS), ("teacher_timestamp_contract_id", EXPECTED_TIMESTAMP_CONTRACT), ("teacher_execution_contract_id", EXPECTED_EXECUTION_CONTRACT), ("source_manifest_sha256", manifest_sha)):
            if evidence.get(field) != expected:
                raise ValueError(f"v12 teacher {field} mismatch: {source_id}")
        for field, expected in (
            ("prompt_profile", EXPECTED_PROMPT_PROFILE),
            ("prompt_version", EXPECTED_PROMPT_VERSION),
            ("protect_prompt_version", EXPECTED_PROTECT_PROMPT_VERSION),
            ("nonvocal_prompt_version", EXPECTED_NONVOCAL_PROMPT_VERSION),
        ):
            if evidence.get(field) != expected:
                raise ValueError(f"v12 teacher {field} mismatch: {source_id}")
        if any(evidence.get(field) is not None for field in ("temperature", "top_p", "top_k")):
            raise ValueError(f"v12 teacher sampling parameters must be omitted: {source_id}")
        if evidence.get("partition") != source.get("partition"):
            raise ValueError(f"v12 teacher partition mismatch: {source_id}")
        if str(evidence.get("video_id") or "") != str(source.get("video_id") or ""):
            raise ValueError(f"v12 teacher video mismatch: {source_id}")
        source_cores = source.get("core_ids") or source.get("core_id") or []
        if isinstance(source_cores, str):
            source_cores = [source_cores]
        if list(evidence.get("core_ids") or ()) != [str(value) for value in source_cores]:
            raise ValueError(f"v12 teacher core mismatch: {source_id}")
        audio = Path(str(source.get("audio") or evidence.get("audio") or ""))
        if not audio.is_absolute():
            audio = (manifest.parent / audio).resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        actual_audio_sha = _sha256(audio)
        declared_audio_sha = str(source.get("audio_sha256") or "")
        if not declared_audio_sha or actual_audio_sha != declared_audio_sha:
            raise ValueError(f"v12 source audio SHA mismatch: {source_id}")
        if str(evidence.get("audio_sha256") or "") != declared_audio_sha:
            raise ValueError(f"v12 teacher audio SHA mismatch: {source_id}")
        frame_count = int(source.get("frame_count") or evidence.get("frame_count") or 0)
        if frame_count <= 0 or int(evidence.get("frame_count") or 0) != frame_count:
            raise ValueError(f"v12 frame geometry mismatch: {source_id}")
        duration_s = float(source.get("duration_s") or 0.0)
        if duration_s <= 0.0 or abs(float(evidence.get("duration_s") or 0.0) - duration_s) > 1e-9:
            raise ValueError(f"v12 duration mismatch: {source_id}")
        spans = _normalize_spans(evidence, frame_count=frame_count, source_id=source_id)
        counts = Counter(str(span["label"]) for span in spans)
        for label, count in counts.items():
            totals[label] += sum(int(span["end_frame"]) - int(span["start_frame"]) for span in spans if span["label"] == label)
        partition = str(source["partition"])
        partitions[partition] += 1
        compiled.append({
            "schema": VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA,
            "boundary_serialization_contract_id": CONTRACT_ID,
            "canonical_label_schema": VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
            "source_id": source_id,
            "video_id": str(source.get("video_id") or ""),
            "partition": partition,
            "core_ids": list(source.get("core_ids") or ([source.get("core_id")] if source.get("core_id") else [])),
            "source_kind": str(source.get("source_kind") or "real_full_source"),
            "synthetic_composite": bool(source.get("synthetic_composite", False)),
            "audio": _display(audio),
            "audio_sha256": declared_audio_sha,
            "duration_s": duration_s,
            "frame_count": frame_count,
            "frame_hop_s": FRAME_HOP_S,
            "canonical_spans": spans,
            "labels": list(VOCAL_ENVELOPE_SCORER_V12_LABELS),
            "unsure_training_label": VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
            "annotation_provenance": "gemini36_medium_independent_vocal_nonvocal_evidence_v1",
            "teacher_output_used_as_truth": bool(allow_teacher_supervision),
            "teacher_output_used_as_calibrated_evidence": True,
            "training_manifest_allowed": bool(allow_teacher_supervision),
            "v11_label_inheritance": False,
            "v11_complement_conversion": False,
            "preaudit": str(preaudit),
            "preaudit_sha256": preaudit_sha,
            "source_manifest": str(manifest),
            "source_manifest_sha256": manifest_sha,
        })
    if not totals["vocal_candidate"] or not totals["non_vocal_candidate"]:
        raise ValueError("v12 canonical requires both vocal and non-vocal definite frames")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "canonical_sources.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in compiled:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": OUTPUT_SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "canonical_label_schema": VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
        "dataset_contract": VOCAL_ENVELOPE_SCORER_V12_DATASET_CONTRACT,
        "manifest": str(manifest), "manifest_sha256": manifest_sha,
        "preaudit": str(preaudit), "preaudit_sha256": preaudit_sha,
        "output": str(output_path), "output_sha256": _sha256(output_path),
        "source_count": len(compiled), "partition_counts": dict(partitions),
        "frame_counts": dict(sorted(totals.items())),
        "teacher_output_used_as_truth": bool(allow_teacher_supervision),
        "training_manifest_allowed": bool(allow_teacher_supervision),
        "v11_complement_conversion": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preaudit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-teacher-supervision", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(compile_canonical(manifest=Path(args.manifest), preaudit=Path(args.preaudit), output_dir=Path(args.output_dir), allow_teacher_supervision=args.allow_teacher_supervision), ensure_ascii=False))
