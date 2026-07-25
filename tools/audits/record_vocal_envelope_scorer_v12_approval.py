#!/usr/bin/env python3
"""Record an explicit full-page human approval for a Scorer v12 teacher audit.

This is intentionally an opt-in provenance tool, not an automatic quality gate.
It only serializes the same all-positive verdict available in the audit UI and
binds every row to the exact source manifest and preaudit hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_AUDIT_ITEM_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
)

CONTRACT_ID = "boundary_acoustic_binary_v12"


def _rows(path: Path) -> list[dict[str, Any]]:
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


def record_approval(
    *,
    audit_manifest: Path,
    source_manifest: Path,
    preaudit: Path,
    output: Path,
    note: str,
    approved_by: str,
) -> dict[str, Any]:
    audit_manifest = audit_manifest.resolve()
    source_manifest = source_manifest.resolve()
    preaudit = preaudit.resolve()
    output = output.resolve()
    sources = {str(row["source_id"]): row for row in _rows(source_manifest)}
    audit_rows = _rows(audit_manifest)
    if not audit_rows or len(sources) != len(audit_rows):
        raise ValueError("audit and source manifests must be non-empty and equally sized")
    if len(sources) != len(_rows(source_manifest)):
        raise ValueError("source manifest requires unique source_id values")
    audit_ids = [str(row.get("source_id") or "") for row in audit_rows]
    if len(set(audit_ids)) != len(audit_ids) or set(audit_ids) != set(sources):
        raise ValueError("audit and source manifests must contain the exact same source IDs")
    source_sha = _sha256(source_manifest)
    preaudit_sha = _sha256(preaudit)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    verdicts: list[dict[str, Any]] = []
    for audit in audit_rows:
        source_id = str(audit["source_id"])
        source = sources[source_id]
        if audit.get("schema") != VOCAL_ENVELOPE_SCORER_V12_AUDIT_ITEM_SCHEMA:
            raise ValueError(f"wrong audit item schema: {source_id}")
        for field in ("video_id", "partition", "audio_sha256", "frame_count"):
            if audit.get(field) != source.get(field):
                raise ValueError(f"audit/source {field} mismatch: {source_id}")
        if abs(float(audit.get("duration_s") or 0.0) - float(source.get("duration_s") or 0.0)) > 1e-9:
            raise ValueError(f"audit/source duration mismatch: {source_id}")
        verdicts.append(
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "source_id": source_id,
                "video_id": source["video_id"],
                "partition": source["partition"],
                "audio_sha256": source["audio_sha256"],
                "duration_s": source["duration_s"],
                "frame_count": source["frame_count"],
                "source_manifest_sha256": source_sha,
                "preaudit_sha256": preaudit_sha,
                "reviewed_full_source": True,
                "vocal_coverage": "definite_vocal_complete",
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
        "output": str(output),
        "output_sha256": _sha256(output),
        "verdict_count": len(verdicts),
        "approved_count": len(verdicts),
        "source_manifest_sha256": source_sha,
        "preaudit_sha256": preaudit_sha,
        "approval_provenance": "explicit_user_blanket_approval",
        "approved_by": approved_by,
        "updated_at": now,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", required=True)
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
