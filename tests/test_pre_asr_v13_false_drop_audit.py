from __future__ import annotations

import json
from pathlib import Path

from tools.asr.cueqc.evaluate_pre_asr_v13_false_drop_audit import evaluate


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_v13_false_drop_audit_requires_complete_zero_true_speech_gate(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "false_drops.jsonl"
    verdicts = tmp_path / "manual.jsonl"
    output = tmp_path / "summary.json"
    _write(manifest, [{"row_id": "a"}, {"row_id": "b"}])
    _write(
        verdicts,
        [
            {"row_id": "a", "verdict": "safe_drop"},
            {"row_id": "b", "verdict": "true_speech"},
        ],
    )

    summary = evaluate(
        false_drop_manifest=manifest,
        manual_verdicts=verdicts,
        output=output,
    )

    assert summary["complete"] is True
    assert summary["true_semantic_keep_deletion_count"] == 1
    assert summary["promote_allowed"] is False


def test_v13_empty_false_drop_manifest_is_a_complete_zero_target_gate(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "false_drops.jsonl"
    verdicts = tmp_path / "manual.jsonl"
    manifest.write_text("", encoding="utf-8")
    verdicts.write_text("", encoding="utf-8")

    summary = evaluate(
        false_drop_manifest=manifest,
        manual_verdicts=verdicts,
        output=tmp_path / "summary.json",
    )

    assert summary["complete"] is True
    assert summary["promote_allowed"] is True
    assert summary["target_manifest_count"] == 0
