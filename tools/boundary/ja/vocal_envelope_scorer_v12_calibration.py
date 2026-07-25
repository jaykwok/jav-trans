#!/usr/bin/env python3
"""Validate the fixed human-approved Scorer v12 Teacher calibration bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (
    VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
    vocal_envelope_v12_manual_verdict_is_approved,
)
from tools.omni.gemini_native import (
    GEMINI_NATIVE_EXECUTION_CONTRACT,
    GEMINI_NATIVE_MODEL,
)
from tools.omni.timestamp_contract import TIMESTAMP_CONTRACT_ID
from tools.boundary.ja.vocal_envelope_scorer_v12_teacher_contract import (
    teacher_contract_fingerprint_fields,
)


CONTRACT_ID = "boundary_acoustic_binary_v12"
CALIBRATION_ID = "vocal_envelope_scorer_v12_voice_only_pilot25_human_approved_v2"
# Frozen only after the voice-only pilot receives a new human approval.
CALIBRATION_ARTIFACT_SHA256: dict[str, str] = {}
CALIBRATION_TEACHER_CONTRACT = {
    "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
    "provider_profile": "gemini",
    "model": GEMINI_NATIVE_MODEL,
    "env_file_name": "gemini",
    "reasoning_effort": "medium",
    "max_tokens": 8192,
    "prompt_profile": "voice-envelope-single-pass-tristate-v4",
    "prompt_version": (
        "voice-envelope-single-pass-tristate-v4-voice-only-gemini36-medium-mmss"
    ),
    "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
    "teacher_execution_contract_id": GEMINI_NATIVE_EXECUTION_CONTRACT,
    **teacher_contract_fingerprint_fields(),
}


def _rows(path: Path) -> list[dict[str, Any]]:
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
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"{name} requires unique non-empty source_id")
        result[source_id] = dict(row)
    return result


def evidence_span_signature(
    evidence: Mapping[str, Any], *, frame_count: int, source_id: str
) -> tuple[tuple[str, int, int], ...]:
    spans: list[tuple[str, int, int]] = []
    for key, label in (
        ("vocal_spans", "vocal_candidate"),
        ("non_vocal_spans", "non_vocal_candidate"),
        ("unsure_spans", "unsure"),
    ):
        for item in evidence.get(key) or ():
            actual_label = str(item.get("label") or label)
            if actual_label != label:
                raise ValueError(
                    f"calibration span label mismatch: {source_id} {actual_label}"
                )
            spans.append(
                (
                    label,
                    int(item.get("start_frame") or 0),
                    int(item.get("end_frame") or 0),
                )
            )
    spans.sort(key=lambda value: (value[1], value[2], value[0]))
    cursor = 0
    for _label, start, end in spans:
        if start != cursor or end <= start or end > frame_count:
            raise ValueError(
                f"calibration evidence must cover contiguous frames: {source_id}"
            )
        cursor = end
    if cursor != frame_count:
        raise ValueError(f"calibration evidence misses source frames: {source_id}")
    return tuple(spans)


def load_approved_calibration(
    *,
    manifest: Path,
    preaudit: Path,
    verdicts: Path,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    preaudit = preaudit.resolve()
    verdicts = verdicts.resolve()
    hashes = {
        "manifest": _sha256(manifest),
        "preaudit": _sha256(preaudit),
        "verdicts": _sha256(verdicts),
    }
    required_hashes = dict(
        CALIBRATION_ARTIFACT_SHA256
        if expected_hashes is None
        else expected_hashes
    )
    if set(required_hashes) != {"manifest", "preaudit", "verdicts"} or any(
        not value for value in required_hashes.values()
    ):
        raise ValueError(
            "Scorer v12 voice-only calibration is not frozen; "
            "run and approve the new pilot first"
        )
    if hashes != required_hashes:
        raise ValueError(
            "Scorer v12 calibration artifact SHA mismatch; "
            f"expected={required_hashes} actual={hashes}"
        )

    sources = _index(_rows(manifest), name="calibration manifest")
    evidence = _index(_rows(preaudit), name="calibration preaudit")
    decisions = _index(_rows(verdicts), name="calibration verdicts")
    if set(sources) != set(evidence) or set(sources) != set(decisions):
        raise ValueError(
            "Scorer v12 calibration manifest/preaudit/verdict IDs must match exactly"
        )
    if not sources:
        raise ValueError("Scorer v12 calibration bundle is empty")

    signatures: dict[str, tuple[tuple[str, int, int], ...]] = {}
    for source_id in sorted(sources):
        source = sources[source_id]
        row = evidence[source_id]
        verdict = decisions[source_id]
        if row.get("schema") != VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA:
            raise ValueError(f"wrong calibration preaudit schema: {source_id}")
        if row.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError(f"wrong calibration central contract: {source_id}")
        for field, expected in CALIBRATION_TEACHER_CONTRACT.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Scorer v12 calibration teacher {field} mismatch: {source_id}"
                )
        if row.get("source_manifest_sha256") != hashes["manifest"]:
            raise ValueError(
                f"Scorer v12 calibration manifest binding mismatch: {source_id}"
            )
        for field in (
            "video_id",
            "partition",
            "audio_sha256",
            "duration_s",
            "frame_count",
            "sample_rate",
            "sample_count",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(
                    f"Scorer v12 calibration source {field} mismatch: {source_id}"
                )
        source_cores = source.get("core_ids") or source.get("core_id") or []
        if isinstance(source_cores, str):
            source_cores = [source_cores]
        if list(row.get("core_ids") or ()) != [
            str(value) for value in source_cores
        ]:
            raise ValueError(f"Scorer v12 calibration core mismatch: {source_id}")
        frame_count = int(source.get("frame_count") or 0)
        if frame_count <= 0:
            raise ValueError(f"invalid calibration frame count: {source_id}")
        signatures[source_id] = evidence_span_signature(
            row,
            frame_count=frame_count,
            source_id=source_id,
        )

        if verdict.get("schema") != VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA:
            raise ValueError(f"wrong calibration verdict schema: {source_id}")
        if verdict.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError(f"wrong calibration verdict contract: {source_id}")
        for field, expected in (
            ("source_manifest_sha256", hashes["manifest"]),
            ("preaudit_sha256", hashes["preaudit"]),
            ("video_id", str(source.get("video_id") or "")),
            ("partition", str(source.get("partition") or "")),
            ("audio_sha256", str(source.get("audio_sha256") or "")),
            ("frame_count", frame_count),
        ):
            if verdict.get(field) != expected:
                raise ValueError(
                    f"Scorer v12 calibration verdict {field} mismatch: {source_id}"
                )
        if abs(
            float(verdict.get("duration_s") or 0.0)
            - float(source.get("duration_s") or 0.0)
        ) > 1e-9:
            raise ValueError(
                f"Scorer v12 calibration verdict duration mismatch: {source_id}"
            )
        if not vocal_envelope_v12_manual_verdict_is_approved(verdict):
            raise ValueError(f"Scorer v12 calibration verdict is not approved: {source_id}")
        if verdict.get("approved") is not True:
            raise ValueError(f"Scorer v12 calibration approval flag mismatch: {source_id}")
        if verdict.get("training_manifest_allowed") is not True:
            raise ValueError(f"Scorer v12 calibration training flag mismatch: {source_id}")

    return {
        "calibration_id": CALIBRATION_ID,
        "manifest": manifest,
        "preaudit": preaudit,
        "verdicts": verdicts,
        "hashes": hashes,
        "sources": sources,
        "evidence": evidence,
        "verdict_rows": decisions,
        "signatures": signatures,
        "teacher_contract": dict(CALIBRATION_TEACHER_CONTRACT),
    }
