#!/usr/bin/env python3
"""Record an explicit full-page human approval for a Scorer v12 teacher audit.

This is intentionally an opt-in provenance tool, not an automatic quality gate.
It serializes the same all-positive verdict available in the audit UI only after
the source, Teacher evidence, audit manifest, and audit summary form one exact
hash- and identity-bound artifact chain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_AUDIT_ITEM_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_calibration import (  # noqa: E402
    CALIBRATION_TEACHER_CONTRACT,
    evidence_span_signature,
)

CONTRACT_ID = "boundary_acoustic_binary_v12"
AUDIT_SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_teacher_audit_summary_v2"
APPROVED_VOCAL_PURITY = "definite_vocal_excludes_separable_background"


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _index(
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"{name} requires unique non-empty source_id values")
        result[source_id] = dict(row)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _resolve_recorded_path(value: Any, *, relative_to: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"audit summary requires non-empty {field}")
    path = Path(value)
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _audio_path(row: Mapping[str, Any], *, relative_to: Path, name: str) -> Path:
    value = row.get("audio")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a non-empty audio path")
    path = Path(value)
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    expected = str(row.get("audio_sha256") or "")
    if not path.is_file() or not expected or _sha256(path) != expected:
        raise ValueError(f"{name} audio SHA mismatch")
    return path


def _same_number(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _core_ids(row: Mapping[str, Any]) -> list[str]:
    values = row.get("core_ids")
    if values is None:
        values = row.get("core_id") or []
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values]


def _signature_json(
    signature: tuple[tuple[str, int, int], ...]
) -> list[list[str | int]]:
    return [[label, start, end] for label, start, end in signature]


def _expected_audit_spans(
    evidence: Mapping[str, Any], *, source_id: str
) -> dict[str, list[dict[str, Any]]]:
    for field in ("vocal_spans", "non_vocal_spans", "unsure_spans"):
        if field not in evidence or not isinstance(evidence[field], list):
            raise ValueError(f"preaudit requires {field}: {source_id}")
    conflicts: set[tuple[int, int]] = set()
    for span in evidence.get("conflict_spans") or ():
        conflicts.add((int(span["start_frame"]), int(span["end_frame"])))
    unsure: list[dict[str, Any]] = []
    unsure_ranges: set[tuple[int, int]] = set()
    for span in evidence["unsure_spans"]:
        copied = dict(span)
        span_range = (int(span["start_frame"]), int(span["end_frame"]))
        unsure_ranges.add(span_range)
        copied["conflict"] = span_range in conflicts
        unsure.append(copied)
    if not conflicts.issubset(unsure_ranges):
        raise ValueError(f"preaudit conflict spans must also be unsure: {source_id}")
    return {
        "vocal_spans": [dict(span) for span in evidence["vocal_spans"]],
        "non_vocal_spans": [dict(span) for span in evidence["non_vocal_spans"]],
        "unsure_spans": unsure,
    }


def _load_and_validate_summary(
    *,
    path: Path,
    audit_manifest: Path,
    source_manifest: Path,
    preaudit: Path,
    audit_manifest_sha: str,
    source_manifest_sha: str,
    preaudit_sha: str,
    source_count: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"audit summary is required to approve an audit manifest: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("audit summary must be a JSON object")
    for field, expected in (
        ("schema", AUDIT_SUMMARY_SCHEMA),
        ("boundary_serialization_contract_id", CONTRACT_ID),
        ("task_semantics", VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS),
        ("source_manifest_sha256", source_manifest_sha),
        ("preaudit_sha256", preaudit_sha),
        ("audit_manifest_sha256", audit_manifest_sha),
        ("manual_verdict_schema", VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA),
        ("source_count", source_count),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"audit summary {field} mismatch")
    for field, expected in (
        ("source_manifest", source_manifest),
        ("preaudit", preaudit),
        ("audit_manifest", audit_manifest),
    ):
        recorded = _resolve_recorded_path(
            payload.get(field), relative_to=path.parent, field=field
        )
        if recorded != expected:
            raise ValueError(f"audit summary {field} path mismatch")
    if payload.get("manual_gate_status") != "pending":
        raise ValueError("audit summary must still have a pending manual gate")
    if payload.get("training_manifest_allowed") is not False:
        raise ValueError("audit summary must not already allow training")
    if int(payload.get("skipped_calibration_source_count") or 0) != 0 or list(
        payload.get("skipped_calibration_source_ids") or ()
    ):
        raise ValueError("blanket approval refuses an audit that skipped source IDs")
    return payload


def record_approval(
    *,
    audit_manifest: Path,
    source_manifest: Path,
    preaudit: Path,
    output: Path,
    note: str,
    approved_by: str,
    audit_summary: Path | None = None,
) -> dict[str, Any]:
    audit_manifest = audit_manifest.resolve()
    source_manifest = source_manifest.resolve()
    preaudit = preaudit.resolve()
    output = output.resolve()
    audit_summary = (
        audit_summary.resolve()
        if audit_summary is not None
        else (audit_manifest.parent / "summary.json").resolve()
    )
    if output in {audit_manifest, source_manifest, preaudit, audit_summary}:
        raise ValueError("approval output must not overwrite an input artifact")

    source_rows = _rows(source_manifest)
    preaudit_rows = _rows(preaudit)
    audit_rows = _rows(audit_manifest)
    sources = _index(source_rows, name="source manifest")
    evidence = _index(preaudit_rows, name="preaudit")
    audits = _index(audit_rows, name="audit manifest")
    if set(sources) != set(evidence) or set(sources) != set(audits):
        raise ValueError(
            "source, preaudit, and audit manifests must contain the exact same source IDs"
        )

    source_sha = _sha256(source_manifest)
    preaudit_sha = _sha256(preaudit)
    audit_sha = _sha256(audit_manifest)
    summary_payload = _load_and_validate_summary(
        path=audit_summary,
        audit_manifest=audit_manifest,
        source_manifest=source_manifest,
        preaudit=preaudit,
        audit_manifest_sha=audit_sha,
        source_manifest_sha=source_sha,
        preaudit_sha=preaudit_sha,
        source_count=len(sources),
    )
    summary_sha = _sha256(audit_summary)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    verdicts: list[dict[str, Any]] = []
    for source_id in sorted(sources):
        source = sources[source_id]
        teacher = evidence[source_id]
        audit = audits[source_id]
        if source.get("schema") != VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA:
            raise ValueError(f"wrong source schema: {source_id}")
        if teacher.get("schema") != VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA:
            raise ValueError(f"wrong preaudit schema: {source_id}")
        if audit.get("schema") != VOCAL_ENVELOPE_SCORER_V12_AUDIT_ITEM_SCHEMA:
            raise ValueError(f"wrong audit item schema: {source_id}")
        for name, row in (
            ("source", source),
            ("preaudit", teacher),
            ("audit", audit),
        ):
            if row.get("boundary_serialization_contract_id") != CONTRACT_ID:
                raise ValueError(f"wrong {name} central contract: {source_id}")
        for name, row in (("preaudit", teacher), ("audit", audit)):
            if row.get("task_semantics") != VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS:
                raise ValueError(f"wrong {name} task semantics: {source_id}")
        for field, expected in CALIBRATION_TEACHER_CONTRACT.items():
            if teacher.get(field) != expected:
                raise ValueError(f"preaudit Teacher {field} mismatch: {source_id}")
        if teacher.get("source_manifest_sha256") != source_sha:
            raise ValueError(f"preaudit source manifest binding mismatch: {source_id}")
        if audit.get("source_manifest_sha256") != source_sha:
            raise ValueError(f"audit source manifest binding mismatch: {source_id}")
        if audit.get("preaudit_sha256") != preaudit_sha:
            raise ValueError(f"audit preaudit binding mismatch: {source_id}")

        for field in (
            "video_id",
            "partition",
            "audio_sha256",
            "frame_count",
            "sample_rate",
            "sample_count",
        ):
            if teacher.get(field) != source.get(field):
                raise ValueError(f"preaudit/source {field} mismatch: {source_id}")
        for field in ("video_id", "partition", "audio_sha256", "frame_count"):
            if audit.get(field) != source.get(field):
                raise ValueError(f"audit/source {field} mismatch: {source_id}")
        if not _same_number(teacher.get("duration_s"), source.get("duration_s")):
            raise ValueError(f"preaudit/source duration mismatch: {source_id}")
        if not _same_number(audit.get("duration_s"), source.get("duration_s")):
            raise ValueError(f"audit/source duration mismatch: {source_id}")
        if _core_ids(teacher) != _core_ids(source):
            raise ValueError(f"preaudit/source core identity mismatch: {source_id}")
        if teacher.get("frame_hop_s") != source.get("frame_hop_s"):
            raise ValueError(f"preaudit/source frame hop mismatch: {source_id}")
        frame_count = int(source.get("frame_count") or 0)
        duration_s = float(source.get("duration_s") or 0.0)
        frame_hop_s = float(source.get("frame_hop_s") or 0.0)
        sample_rate = int(source.get("sample_rate") or 0)
        sample_count = int(source.get("sample_count") or 0)
        if (
            frame_count <= 0
            or duration_s <= 0
            or frame_hop_s <= 0
            or sample_rate <= 0
            or sample_count <= 0
        ):
            raise ValueError(f"invalid source geometry: {source_id}")
        if not _same_number(duration_s, sample_count / sample_rate):
            raise ValueError(f"source sample geometry mismatch: {source_id}")
        frame_duration_s = frame_count * frame_hop_s
        if duration_s > frame_duration_s + 1e-9 or frame_duration_s - duration_s > frame_hop_s + 1e-9:
            raise ValueError(f"source frame geometry mismatch: {source_id}")

        source_audio = _audio_path(
            source, relative_to=source_manifest.parent, name=f"source {source_id}"
        )
        teacher_audio = _audio_path(
            teacher, relative_to=preaudit.parent, name=f"preaudit {source_id}"
        )
        audit_audio = _audio_path(
            audit, relative_to=audit_manifest.parent, name=f"audit {source_id}"
        )
        expected_audio_sha = str(source["audio_sha256"])
        if any(
            _sha256(path) != expected_audio_sha
            for path in (source_audio, teacher_audio, audit_audio)
        ):
            raise ValueError(f"source/preaudit/audit audio identity mismatch: {source_id}")

        expected_spans = _expected_audit_spans(teacher, source_id=source_id)
        for field, expected in expected_spans.items():
            if audit.get(field) != expected:
                raise ValueError(f"audit/preaudit {field} mismatch: {source_id}")
        teacher_signature = evidence_span_signature(
            teacher, frame_count=frame_count, source_id=source_id
        )
        audit_signature = evidence_span_signature(
            audit, frame_count=frame_count, source_id=source_id
        )
        if audit_signature != teacher_signature:
            raise ValueError(f"audit/preaudit evidence signature mismatch: {source_id}")
        signature_json = _signature_json(teacher_signature)
        if audit.get("evidence_span_signature") != signature_json:
            raise ValueError(f"audit recorded evidence signature mismatch: {source_id}")

        verdicts.append(
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
                "source_id": source_id,
                "video_id": source["video_id"],
                "partition": source["partition"],
                "audio_sha256": source["audio_sha256"],
                "duration_s": source["duration_s"],
                "frame_count": source["frame_count"],
                "source_manifest_sha256": source_sha,
                "preaudit_sha256": preaudit_sha,
                "audit_manifest_sha256": audit_sha,
                "audit_summary_sha256": summary_sha,
                "evidence_span_signature": signature_json,
                "reviewed_full_source": True,
                "vocal_coverage": "definite_vocal_complete",
                "vocal_purity": APPROVED_VOCAL_PURITY,
                "non_vocal_safety": "definite_non_vocal_clean",
                "envelope_structure": "event_envelopes_continuous",
                "approved": True,
                "training_manifest_allowed": True,
                "notes": note,
                "approval_provenance": "explicit_user_blanket_approval",
                "approved_by": approved_by,
                "updated_at": now,
            }
        )
    _atomic_jsonl(output, verdicts)
    return {
        "schema": "vocal_envelope_scorer_v12_blanket_approval_summary_v2",
        "boundary_serialization_contract_id": CONTRACT_ID,
        "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
        "output": str(output),
        "output_sha256": _sha256(output),
        "verdict_count": len(verdicts),
        "approved_count": len(verdicts),
        "source_manifest_sha256": source_sha,
        "preaudit_sha256": preaudit_sha,
        "audit_manifest_sha256": audit_sha,
        "audit_summary_sha256": summary_sha,
        "audit_summary_schema": summary_payload["schema"],
        "approval_provenance": "explicit_user_blanket_approval",
        "approved_by": approved_by,
        "updated_at": now,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument(
        "--audit-summary",
        help="Defaults to summary.json beside --audit-manifest.",
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--preaudit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--approved-by", default="user")
    parser.add_argument(
        "--approve-all-reviewed",
        action="store_true",
        help="Required acknowledgement that the human reviewed and approved every source.",
    )
    args = parser.parse_args(argv)
    if not args.approve_all_reviewed:
        parser.error("--approve-all-reviewed is required")
    return args


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            record_approval(
                audit_manifest=Path(args.audit_manifest),
                audit_summary=(
                    Path(args.audit_summary) if args.audit_summary else None
                ),
                source_manifest=Path(args.source_manifest),
                preaudit=Path(args.preaudit),
                output=Path(args.output),
                note=args.note,
                approved_by=args.approved_by,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
