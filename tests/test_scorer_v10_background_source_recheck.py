from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audits.evaluate_scorer_v10_background_source_recheck import (
    OVERRIDE_SCHEMA,
    RESULT_SCHEMA,
    evaluate,
)
from tools.audits.generate_scorer_v10_prediction_audit_html import (
    VERDICT_SCHEMA,
    build_audit,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"RIFF-test-audio")
    audit_id = "background_false_keep:background-source"
    manifest = tmp_path / "original_manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "audit_id": audit_id,
                "source_id": "background-source",
                "partition": "test",
                "row_role": "all_background",
                "category": "background_false_keep",
                "audio": str(audio),
                "duration_s": 1.0,
                "frame_count": 50,
                "truth_spans": [
                    {
                        "label": "truth_background",
                        "start_frame": 0,
                        "end_frame": 50,
                        "start_s": 0.0,
                        "end_s": 1.0,
                    }
                ],
                "prediction_spans": [
                    {
                        "label": "model_speech",
                        "start_frame": 10,
                        "end_frame": 20,
                        "start_s": 0.2,
                        "end_s": 0.4,
                    }
                ],
            }
        ],
    )
    original_verdicts = tmp_path / "original_verdicts.jsonl"
    common = {
        "schema": VERDICT_SCHEMA,
        "audit_id": audit_id,
        "source_id": "background-source",
        "partition": "test",
        "row_role": "all_background",
        "category": "background_false_keep",
    }
    _write_jsonl(
        original_verdicts,
        [{**common, "verdict": "canonical_contains_target_speech"}],
    )
    recheck = tmp_path / "recheck"
    build_audit(
        selection=manifest,
        output_dir=recheck,
        source_ids={"background-source"},
    )
    _write_jsonl(
        recheck / "manual_verdicts.jsonl",
        [{**common, "verdict": "model_false_keep"}],
    )
    return manifest, original_verdicts, recheck


def test_background_source_recheck_binds_exact_override(tmp_path: Path) -> None:
    manifest, original_verdicts, recheck = _fixture(tmp_path)
    output = recheck / "manual_gate.json"
    gate = evaluate(
        original_audit_manifest=manifest,
        original_manual_verdicts=original_verdicts,
        recheck_summary=recheck / "summary.json",
        recheck_audit_manifest=recheck / "audit_manifest.jsonl",
        recheck_manual_verdicts=recheck / "manual_verdicts.jsonl",
        output=output,
    )
    overrides = [
        json.loads(line)
        for line in (recheck / "manual_gate.overrides.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert gate["schema"] == RESULT_SCHEMA
    assert gate["canonical_override_ready"] is True
    assert gate["overridden_source_ids"] == ["background-source"]
    assert overrides == [
        {
            "schema": OVERRIDE_SCHEMA,
            "source_id": "background-source",
            "audit_id": "background_false_keep:background-source",
            "partition": "test",
            "row_role": "all_background",
            "category": "background_false_keep",
            "original_verdict": "canonical_contains_target_speech",
            "replacement_verdict": "model_false_keep",
            "override_action": "withdraw_canonical_contains_target_speech",
            "canonical_action": "retain_all_background",
            "exclude_from_background_speech_repair": True,
        }
    ]


def test_background_source_recheck_rejects_changed_prediction(tmp_path: Path) -> None:
    manifest, original_verdicts, recheck = _fixture(tmp_path)
    row = json.loads((recheck / "audit_manifest.jsonl").read_text(encoding="utf-8"))
    row["prediction_spans"][0]["end_frame"] = 21
    _write_jsonl(recheck / "audit_manifest.jsonl", [row])

    with pytest.raises(ValueError, match="prediction evidence differs"):
        evaluate(
            original_audit_manifest=manifest,
            original_manual_verdicts=original_verdicts,
            recheck_summary=recheck / "summary.json",
            recheck_audit_manifest=recheck / "audit_manifest.jsonl",
            recheck_manual_verdicts=recheck / "manual_verdicts.jsonl",
            output=recheck / "manual_gate.json",
        )


def test_background_source_recheck_only_withdraws_old_target_verdict(
    tmp_path: Path,
) -> None:
    manifest, original_verdicts, recheck = _fixture(tmp_path)
    row = json.loads(original_verdicts.read_text(encoding="utf-8"))
    row["verdict"] = "model_false_keep"
    _write_jsonl(original_verdicts, [row])

    with pytest.raises(ValueError, match="can only withdraw"):
        evaluate(
            original_audit_manifest=manifest,
            original_manual_verdicts=original_verdicts,
            recheck_summary=recheck / "summary.json",
            recheck_audit_manifest=recheck / "audit_manifest.jsonl",
            recheck_manual_verdicts=recheck / "manual_verdicts.jsonl",
            output=recheck / "manual_gate.json",
        )
