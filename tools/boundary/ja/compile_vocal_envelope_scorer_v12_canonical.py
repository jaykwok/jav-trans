#!/usr/bin/env python3
"""Compile validated Scorer v12 single-pass tri-state evidence."""
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
    VOCAL_ENVELOPE_SCORER_V12_ENVELOPE_STRUCTURE_OPTIONS,
    VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_NONVOCAL_SAFETY_OPTIONS,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
    VOCAL_ENVELOPE_SCORER_V12_VOCAL_COVERAGE_OPTIONS,
    VOCAL_ENVELOPE_SCORER_V12_VOCAL_PURITY_OPTIONS,
    vocal_envelope_v12_manual_verdict_is_approved,
)
from tools.omni.gemini_native import (  # noqa: E402
    GEMINI_NATIVE_EXECUTION_CONTRACT,
    GEMINI_NATIVE_MODEL,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_calibration import (  # noqa: E402
    CALIBRATION_ARTIFACT_SHA256,
    evidence_span_signature,
    load_approved_calibration,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_teacher_contract import (  # noqa: E402
    teacher_contract_fingerprint_fields,
)

CONTRACT_ID = "boundary_acoustic_binary_v12"
FRAME_HOP_S = 0.02
EXPECTED_REASONING = "medium"
EXPECTED_MAX_TOKENS = 8192
EXPECTED_TIMESTAMP_CONTRACT = "omni_audio_timestamp_mmss_mmm_v1"
PROVIDER_CONTRACTS: dict[str, dict[str, str]] = {
    "openrouter": {
        "model": "google/gemini-3.6-flash",
        "execution_contract": "openrouter_gemini36_reasoning_require_parameters_v1",
    },
    "gemini": {
        "model": GEMINI_NATIVE_MODEL,
        "execution_contract": GEMINI_NATIVE_EXECUTION_CONTRACT,
    },
}
EXPECTED_PROMPT_PROFILE = "voice-envelope-single-pass-tristate-v4"
EXPECTED_PROMPT_VERSION = "voice-envelope-single-pass-tristate-v4-voice-only-gemini36-medium-mmss"
OUTPUT_SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_canonical_compile_summary_v2"
EXPECTED_TEACHER_CONTRACT_FINGERPRINTS = teacher_contract_fingerprint_fields()


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


def _validate_manual_verdict(
    verdict: Mapping[str, Any],
    *,
    source_id: str,
    source: Mapping[str, Any],
    manifest_sha: str,
    preaudit_sha: str,
) -> bool:
    partition = str(source.get("partition") or "")
    declared_audio_sha = str(source.get("audio_sha256") or "")
    frame_count = int(source.get("frame_count") or 0)
    duration_s = float(source.get("duration_s") or 0.0)
    if verdict.get("schema") != VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA:
        raise ValueError(f"wrong v12 manual verdict schema: {source_id}")
    if verdict.get("boundary_serialization_contract_id") != CONTRACT_ID:
        raise ValueError(f"wrong v12 manual verdict contract: {source_id}")
    for field, expected in (
        ("task_semantics", VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS),
        ("source_manifest_sha256", manifest_sha),
        ("preaudit_sha256", preaudit_sha),
        ("partition", partition),
        ("video_id", str(source.get("video_id") or "")),
        ("audio_sha256", declared_audio_sha),
        ("frame_count", frame_count),
    ):
        if verdict.get(field) != expected:
            raise ValueError(f"v12 manual verdict {field} mismatch: {source_id}")
    if abs(float(verdict.get("duration_s") or 0.0) - duration_s) > 1e-9:
        raise ValueError(f"v12 manual verdict duration mismatch: {source_id}")
    if (
        str(verdict.get("vocal_coverage") or "")
        not in VOCAL_ENVELOPE_SCORER_V12_VOCAL_COVERAGE_OPTIONS
    ):
        raise ValueError(f"invalid v12 vocal coverage verdict: {source_id}")
    if (
        str(verdict.get("non_vocal_safety") or "")
        not in VOCAL_ENVELOPE_SCORER_V12_NONVOCAL_SAFETY_OPTIONS
    ):
        raise ValueError(f"invalid v12 non-vocal safety verdict: {source_id}")
    if (
        str(verdict.get("vocal_purity") or "")
        not in VOCAL_ENVELOPE_SCORER_V12_VOCAL_PURITY_OPTIONS
    ):
        raise ValueError(f"invalid v12 vocal purity verdict: {source_id}")
    if (
        str(verdict.get("envelope_structure") or "")
        not in VOCAL_ENVELOPE_SCORER_V12_ENVELOPE_STRUCTURE_OPTIONS
    ):
        raise ValueError(f"invalid v12 envelope structure verdict: {source_id}")
    human_approved = vocal_envelope_v12_manual_verdict_is_approved(verdict)
    if bool(verdict.get("approved")) != human_approved:
        raise ValueError(f"v12 manual verdict approval flag mismatch: {source_id}")
    if bool(verdict.get("training_manifest_allowed")) != human_approved:
        raise ValueError(f"v12 manual verdict training flag mismatch: {source_id}")
    if not human_approved:
        raise ValueError(f"v12 manual verdict rejects canonical supervision: {source_id}")
    return True


def compile_canonical(
    *,
    manifest: Path,
    preaudit: Path,
    output_dir: Path,
    manual_verdicts: Path | None = None,
    calibration_manifest: Path | None = None,
    calibration_preaudit: Path | None = None,
    calibration_verdicts: Path | None = None,
    calibration_expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    preaudit = preaudit.resolve()
    sources = _index(_rows(manifest), "source_id", "v12 source manifest")
    labels = _index(_rows(preaudit), "source_id", "v12 preaudit")
    if set(sources) != set(labels):
        raise ValueError("v12 source manifest and preaudit IDs must match exactly")
    _validate_partition_and_core(sources)
    manifest_sha = _sha256(manifest)
    preaudit_sha = _sha256(preaudit)
    calibration_paths = (
        calibration_manifest,
        calibration_preaudit,
        calibration_verdicts,
    )
    if any(path is not None for path in calibration_paths) and not all(
        path is not None for path in calibration_paths
    ):
        raise ValueError(
            "v12 calibration requires manifest, preaudit and verdicts together"
        )
    calibration: dict[str, Any] | None = None
    calibration_ids: set[str] = set()
    calibration_heldout_ids: set[str] = set()
    if all(path is not None for path in calibration_paths):
        assert calibration_manifest is not None
        assert calibration_preaudit is not None
        assert calibration_verdicts is not None
        calibration = load_approved_calibration(
            manifest=calibration_manifest,
            preaudit=calibration_preaudit,
            verdicts=calibration_verdicts,
            expected_hashes=(
                calibration_expected_hashes or CALIBRATION_ARTIFACT_SHA256
            ),
        )
        calibration_ids = set(calibration["sources"])
        if not calibration_ids.issubset(sources):
            missing = sorted(calibration_ids - set(sources))
            raise ValueError(
                f"v12 full manifest omits calibrated pilot source IDs: {missing}"
            )
        calibration_heldout_ids = {
            source_id
            for source_id in calibration_ids
            if str(sources[source_id].get("partition") or "") in {"val", "test"}
        }
    verdict_rows: dict[str, dict[str, Any]] = {}
    verdict_sha = ""
    if manual_verdicts is not None:
        manual_verdicts = manual_verdicts.resolve()
        verdict_rows = _index(
            _rows(manual_verdicts), "source_id", "v12 manual verdicts"
        )
        verdict_sha = _sha256(manual_verdicts)
    if calibration is None:
        if verdict_rows and set(verdict_rows) != set(sources):
            raise ValueError("v12 manual verdicts must cover the exact source manifest")
    else:
        required_heldout_verdicts = {
            source_id
            for source_id, source in sources.items()
            if str(source.get("partition") or "") in {"val", "test"}
        } - calibration_heldout_ids
        if set(verdict_rows) != required_heldout_verdicts:
            raise ValueError(
                "v12 calibrated full canonical requires manual verdicts for exactly "
                "the non-pilot heldout sources; "
                f"required={sorted(required_heldout_verdicts)} "
                f"actual={sorted(verdict_rows)}"
            )
    compiled: list[dict[str, Any]] = []
    totals = Counter()
    partitions = Counter()
    provenance_counts = Counter()
    selected_profile = ""
    for source_id in sorted(sources):
        source = sources[source_id]
        evidence = labels[source_id]
        if evidence.get("schema") != VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA:
            raise ValueError(f"wrong v12 preaudit schema: {source_id}")
        if evidence.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError(f"wrong v12 central contract: {source_id}")
        if evidence.get("teacher_failed_closed") is True:
            raise ValueError(f"failed-closed v12 evidence cannot compile: {source_id}")
        profile = str(evidence.get("provider_profile") or "")
        provider_contract = PROVIDER_CONTRACTS.get(profile)
        if provider_contract is None:
            raise ValueError(f"unsupported v12 teacher profile: {source_id}")
        if selected_profile and profile != selected_profile:
            raise ValueError("v12 canonical cannot mix teacher provider profiles")
        selected_profile = profile
        if evidence.get("model") != provider_contract["model"]:
            raise ValueError(f"v12 teacher model/profile mismatch: {source_id}")
        for field, expected in (("env_file_name", profile), ("reasoning_effort", EXPECTED_REASONING), ("max_tokens", EXPECTED_MAX_TOKENS), ("task_semantics", VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS), ("teacher_timestamp_contract_id", EXPECTED_TIMESTAMP_CONTRACT), ("teacher_execution_contract_id", provider_contract["execution_contract"]), ("source_manifest_sha256", manifest_sha)):
            if evidence.get(field) != expected:
                raise ValueError(f"v12 teacher {field} mismatch: {source_id}")
        for field, expected in (
            ("prompt_profile", EXPECTED_PROMPT_PROFILE),
            ("prompt_version", EXPECTED_PROMPT_VERSION),
            *EXPECTED_TEACHER_CONTRACT_FINGERPRINTS.items(),
        ):
            if evidence.get(field) != expected:
                raise ValueError(f"v12 teacher {field} mismatch: {source_id}")
        if calibration is not None:
            for field, expected in calibration["teacher_contract"].items():
                if evidence.get(field) != expected:
                    raise ValueError(
                        f"v12 full teacher differs from calibrated {field}: "
                        f"{source_id}"
                    )
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
        sample_rate = int(source.get("sample_rate") or 0)
        sample_count = int(source.get("sample_count") or 0)
        if sample_rate != 16000 or sample_count <= 0:
            raise ValueError(f"v12 source sample geometry mismatch: {source_id}")
        if int(evidence.get("sample_rate") or 0) != sample_rate or int(
            evidence.get("sample_count") or 0
        ) != sample_count:
            raise ValueError(f"v12 teacher sample geometry mismatch: {source_id}")
        spans = _normalize_spans(evidence, frame_count=frame_count, source_id=source_id)
        partition = str(source["partition"])
        calibration_overlap = calibration is not None and source_id in calibration_ids
        if calibration_overlap:
            assert calibration is not None
            calibration_source = calibration["sources"][source_id]
            for field in (
                "video_id",
                "partition",
                "audio_sha256",
                "duration_s",
                "frame_count",
                "sample_rate",
                "sample_count",
            ):
                if source.get(field) != calibration_source.get(field):
                    raise ValueError(
                        f"v12 calibrated source {field} drift: {source_id}"
                    )
            source_cores = source.get("core_ids") or source.get("core_id") or []
            if isinstance(source_cores, str):
                source_cores = [source_cores]
            calibration_cores = (
                calibration_source.get("core_ids")
                or calibration_source.get("core_id")
                or []
            )
            if isinstance(calibration_cores, str):
                calibration_cores = [calibration_cores]
            if [str(value) for value in source_cores] != [
                str(value) for value in calibration_cores
            ]:
                raise ValueError(f"v12 calibrated source core drift: {source_id}")
            current_signature = evidence_span_signature(
                evidence,
                frame_count=frame_count,
                source_id=source_id,
            )
            if current_signature != calibration["signatures"][source_id]:
                raise ValueError(
                    f"v12 calibrated pilot evidence changed after approval: {source_id}"
                )
        verdict = verdict_rows.get(source_id)
        human_approved = calibration_overlap
        if verdict is not None:
            human_approved = _validate_manual_verdict(
                verdict,
                source_id=source_id,
                source=source,
                manifest_sha=manifest_sha,
                preaudit_sha=preaudit_sha,
            )
        calibrated_train = calibration is not None and partition == "train"
        training_allowed = human_approved or calibrated_train
        if human_approved and calibration_overlap:
            annotation_provenance = "human_approved_fixed_pilot_calibration_overlap_v1"
        elif human_approved:
            annotation_provenance = (
                f"human_approved_{profile}_gemini36_medium_single_pass_tristate_v2"
            )
        elif calibrated_train:
            annotation_provenance = (
                "pilot25_calibrated_gemini36_medium_train_supervision_v1"
            )
        else:
            annotation_provenance = (
                f"{profile}_gemini36_medium_single_pass_tristate_review_only_v2"
            )
        provenance_counts[annotation_provenance] += 1
        counts = Counter(str(span["label"]) for span in spans)
        for label, count in counts.items():
            totals[label] += sum(int(span["end_frame"]) - int(span["start_frame"]) for span in spans if span["label"] == label)
        partitions[partition] += 1
        compiled.append({
            "schema": VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA,
            "boundary_serialization_contract_id": CONTRACT_ID,
            "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
            **EXPECTED_TEACHER_CONTRACT_FINGERPRINTS,
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
            "sample_rate": sample_rate,
            "sample_count": sample_count,
            "frame_count": frame_count,
            "frame_hop_s": FRAME_HOP_S,
            "canonical_spans": spans,
            "labels": list(VOCAL_ENVELOPE_SCORER_V12_LABELS),
            "unsure_training_label": VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
            "annotation_provenance": annotation_provenance,
            "teacher_output_used_as_truth": human_approved,
            "teacher_output_used_as_calibrated_evidence": True,
            "training_manifest_allowed": training_allowed,
            "human_full_source_review_approved": human_approved,
            "manual_verdicts": str(manual_verdicts) if manual_verdicts else None,
            "manual_verdicts_sha256": verdict_sha or None,
            "calibration_id": calibration["calibration_id"] if calibration else None,
            "calibration_manifest_sha256": (
                calibration["hashes"]["manifest"] if calibration else None
            ),
            "calibration_preaudit_sha256": (
                calibration["hashes"]["preaudit"] if calibration else None
            ),
            "calibration_verdicts_sha256": (
                calibration["hashes"]["verdicts"] if calibration else None
            ),
            "calibration_overlap_human_approved": calibration_overlap,
            "calibrated_train_supervision": calibrated_train and not human_approved,
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
        "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
        **EXPECTED_TEACHER_CONTRACT_FINGERPRINTS,
        "canonical_label_schema": VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
        "dataset_contract": VOCAL_ENVELOPE_SCORER_V12_DATASET_CONTRACT,
        "manifest": str(manifest), "manifest_sha256": manifest_sha,
        "preaudit": str(preaudit), "preaudit_sha256": preaudit_sha,
        "output": str(output_path), "output_sha256": _sha256(output_path),
        "source_count": len(compiled), "partition_counts": dict(partitions),
        "frame_counts": dict(sorted(totals.items())),
        "provider_profile": selected_profile,
        "teacher_output_used_as_truth": all(
            bool(row["teacher_output_used_as_truth"]) for row in compiled
        ),
        "training_manifest_allowed": all(
            bool(row["training_manifest_allowed"]) for row in compiled
        ),
        "heldout_human_full_source_review_approved": all(
            bool(row["human_full_source_review_approved"])
            for row in compiled
            if row["partition"] in {"val", "test"}
        ),
        "human_full_source_review_approved": all(
            bool(row["human_full_source_review_approved"]) for row in compiled
        ),
        "manual_verdicts": str(manual_verdicts) if manual_verdicts else None,
        "manual_verdicts_sha256": verdict_sha or None,
        "calibration_id": calibration["calibration_id"] if calibration else None,
        "calibration_manifest": (
            str(calibration["manifest"]) if calibration else None
        ),
        "calibration_manifest_sha256": (
            calibration["hashes"]["manifest"] if calibration else None
        ),
        "calibration_preaudit": (
            str(calibration["preaudit"]) if calibration else None
        ),
        "calibration_preaudit_sha256": (
            calibration["hashes"]["preaudit"] if calibration else None
        ),
        "calibration_verdicts": (
            str(calibration["verdicts"]) if calibration else None
        ),
        "calibration_verdicts_sha256": (
            calibration["hashes"]["verdicts"] if calibration else None
        ),
        "calibration_source_count": len(calibration_ids),
        "calibration_heldout_source_count": len(calibration_heldout_ids),
        "annotation_provenance_counts": dict(sorted(provenance_counts.items())),
        "v11_complement_conversion": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preaudit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manual-verdicts")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--calibration-preaudit")
    parser.add_argument("--calibration-verdicts")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            compile_canonical(
                manifest=Path(args.manifest),
                preaudit=Path(args.preaudit),
                output_dir=Path(args.output_dir),
                manual_verdicts=(
                    Path(args.manual_verdicts) if args.manual_verdicts else None
                ),
                calibration_manifest=(
                    Path(args.calibration_manifest)
                    if args.calibration_manifest
                    else None
                ),
                calibration_preaudit=(
                    Path(args.calibration_preaudit)
                    if args.calibration_preaudit
                    else None
                ),
                calibration_verdicts=(
                    Path(args.calibration_verdicts)
                    if args.calibration_verdicts
                    else None
                ),
            ),
            ensure_ascii=False,
        )
    )
