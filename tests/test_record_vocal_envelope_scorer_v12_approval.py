from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.audits.record_vocal_envelope_scorer_v12_approval import record_approval


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_records_hash_bound_explicit_blanket_approval(tmp_path: Path) -> None:
    source = {
        "source_id": "source-1",
        "video_id": "video-1",
        "partition": "train",
        "audio_sha256": "a" * 64,
        "duration_s": 1.0,
        "frame_count": 50,
    }
    audit = {
        "schema": "vocal_envelope_scorer_v12_teacher_audit_item_v1",
        **source,
    }
    source_manifest = tmp_path / "sources.jsonl"
    audit_manifest = tmp_path / "audit.jsonl"
    preaudit = tmp_path / "preaudit.jsonl"
    output = tmp_path / "manual_verdicts.jsonl"
    _write(source_manifest, [source])
    _write(audit_manifest, [audit])
    _write(preaudit, [{"source_id": "source-1"}])

    summary = record_approval(
        audit_manifest=audit_manifest,
        source_manifest=source_manifest,
        preaudit=preaudit,
        output=output,
        note="reviewed in app",
        approved_by="user",
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert summary["approved_count"] == 1
    assert row["approved"] is True
    assert row["training_manifest_allowed"] is True
    assert row["approval_provenance"] == "explicit_user_blanket_approval"
    assert row["source_manifest_sha256"] == hashlib.sha256(
        source_manifest.read_bytes()
    ).hexdigest()
    assert row["preaudit_sha256"] == hashlib.sha256(preaudit.read_bytes()).hexdigest()
