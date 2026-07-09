from __future__ import annotations

import json
from pathlib import Path

from tools.audits.evaluate_pre_asr_v12_false_drop_audit import (
    evaluate_false_drop_audit,
    evaluate_paths,
)


def _manifest_row(candidate_id: str, *, duration_s: float = 1.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "audit_kind": "long_false_drop",
        "audio_id": "vid-w00",
        "chunk_index": int(candidate_id.rsplit("chunk", 1)[1]),
        "start": 0.0,
        "end": duration_s,
        "duration_s": duration_s,
        "truth": "keep",
        "v12_prediction": "drop",
        "v12_prob_drop": 0.7,
    }


def _verdict(candidate_id: str, verdict: str) -> dict:
    return {
        "schema": "pre_asr_v12_repair_manual_verdict_v1",
        "candidate_id": candidate_id,
        "verdict": verdict,
        "note": "",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_false_drop_audit_gate_passes_when_no_manual_keep() -> None:
    summary = evaluate_false_drop_audit(
        manifest_rows=[
            _manifest_row("preasr-vid-w00-chunk00001"),
            _manifest_row("preasr-vid-w00-chunk00002"),
        ],
        verdict_rows=[
            _verdict("preasr-vid-w00-chunk00001", "drop"),
            _verdict("preasr-vid-w00-chunk00002", "unsure"),
        ],
    )

    assert summary["complete"] is True
    assert summary["gate_pass"] is True
    assert summary["promote_allowed"] is True
    assert summary["manual_verdict_counts"] == {"drop": 1, "unsure": 1}
    assert summary["uncertain_count"] == 1


def test_false_drop_audit_gate_fails_on_manual_keep() -> None:
    summary = evaluate_false_drop_audit(
        manifest_rows=[_manifest_row("preasr-vid-w00-chunk00001")],
        verdict_rows=[_verdict("preasr-vid-w00-chunk00001", "keep")],
    )

    assert summary["complete"] is True
    assert summary["gate_pass"] is False
    assert summary["promote_allowed"] is False
    assert summary["true_semantic_keep_deletion_count"] == 1
    assert summary["true_semantic_keep_deletions"][0]["candidate_id"] == "preasr-vid-w00-chunk00001"


def test_false_drop_audit_gate_requires_exact_manifest_closure(tmp_path: Path) -> None:
    manifest = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            _manifest_row("preasr-vid-w00-chunk00001"),
            _manifest_row("preasr-vid-w00-chunk00002"),
        ],
    )
    verdicts = _write_jsonl(
        tmp_path / "manual_verdicts.jsonl",
        [
            _verdict("preasr-vid-w00-chunk00001", "drop"),
            _verdict("preasr-vid-w00-chunk99999", "drop"),
        ],
    )

    summary = evaluate_paths(
        manifest=manifest,
        verdicts=verdicts,
        output=tmp_path / "gate_summary.json",
    )

    stored = json.loads((tmp_path / "gate_summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is False
    assert summary["promote_allowed"] is False
    assert summary["missing_candidates"] == ["preasr-vid-w00-chunk00002"]
    assert summary["unexpected_candidates"] == ["preasr-vid-w00-chunk99999"]
    assert stored["schema"] == "pre_asr_v12_false_drop_audit_gate_summary_v1"


def test_false_drop_audit_gate_accepts_multiple_verdict_files(tmp_path: Path) -> None:
    manifest = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            _manifest_row("preasr-vid-w00-chunk00001"),
            _manifest_row("preasr-vid-w00-chunk00002"),
        ],
    )
    first_verdicts = _write_jsonl(
        tmp_path / "manual_verdicts_round1.jsonl",
        [_verdict("preasr-vid-w00-chunk00001", "drop")],
    )
    second_verdicts = _write_jsonl(
        tmp_path / "manual_verdicts_round2.jsonl",
        [_verdict("preasr-vid-w00-chunk00002", "unsure")],
    )

    summary = evaluate_paths(
        manifest=manifest,
        verdicts=[first_verdicts, second_verdicts],
        output=tmp_path / "gate_summary.json",
    )

    assert summary["complete"] is True
    assert summary["promote_allowed"] is True
    assert summary["manual_verdict_counts"] == {"drop": 1, "unsure": 1}
    assert len(summary["verdicts"]) == 2
    assert summary["verdicts"][0].endswith("manual_verdicts_round1.jsonl")
    assert summary["verdicts"][1].endswith("manual_verdicts_round2.jsonl")


def test_false_drop_audit_gate_can_allow_extra_verdicts() -> None:
    summary = evaluate_false_drop_audit(
        manifest_rows=[_manifest_row("preasr-vid-w00-chunk00001")],
        verdict_rows=[
            _verdict("preasr-vid-w00-chunk00001", "drop"),
            _verdict("preasr-vid-w00-chunk99999", "keep"),
        ],
        allow_extra_verdicts=True,
    )

    assert summary["complete"] is True
    assert summary["promote_allowed"] is True
    assert summary["unexpected_candidates"] == ["preasr-vid-w00-chunk99999"]
    assert summary["extra_verdicts_ignored_count"] == 1
