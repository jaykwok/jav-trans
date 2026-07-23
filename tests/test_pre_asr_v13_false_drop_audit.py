from __future__ import annotations

import json
from pathlib import Path

import tools.audits.generate_pre_asr_v13_false_drop_audit_html as audit_page
from tools.audits.audit_prompt import resolve_audit_prompt
from tools.audits.review_page_core import validate_audit_option_contract
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


def test_v13_false_drop_page_is_a_complete_core_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    validate_audit_option_contract(
        axes=audit_page.CUEQC_FALSE_DROP_AXES,
        combination_results=audit_page.CUEQC_FALSE_DROP_RESULTS,
    )
    args = audit_page.parse_args(
        [
            "--false-drop-manifest",
            "manifest.jsonl",
            "--output-dir",
            "out",
            "--prompt",
            "custom CueQC review",
        ]
    )
    assert args.prompt == "custom CueQC review"

    audio = tmp_path / "source.wav"
    audio.write_bytes(b"RIFF")
    manifest = tmp_path / "manifest.jsonl"
    _write(
        manifest,
        [
            {
                "row_id": "drop-001",
                "audio": str(audio),
                "source_partition": "test",
                "start_s": 1.0,
                "end_s": 2.25,
                "prob_drop": 0.9,
                "teacher_label": "keep",
                "exact_core_label": "semantic_core",
            }
        ],
    )

    def fake_slice_audio_clip(*, output_path: Path, **_kwargs) -> None:
        output_path.write_bytes(b"ID3")

    monkeypatch.setattr(audit_page, "slice_audio_clip", fake_slice_audio_clip)
    monkeypatch.setattr(audit_page, "update_audit_entrypoints", lambda **_kwargs: None)
    prompt = resolve_audit_prompt(
        prompt="custom CueQC review",
        default_prompt=audit_page.DEFAULT_REVIEW_PROMPT,
    )
    output_dir = tmp_path / "out"
    summary = audit_page.build(
        false_drop_manifest=manifest,
        output_dir=output_dir,
        update_latest=False,
        review_prompt=prompt,
    )

    page = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "createAuditReviewCore" in page
    assert "直接播放完整 sub-island" in page
    assert "safe_drop" in page
    assert "true_speech" in page
    assert "unsure" in page
    assert "custom CueQC review" in page
    assert audit_page.MANUAL_VERDICT_SCHEMA in page
    assert summary["schema"] == audit_page.SUMMARY_SCHEMA
    assert summary["manual_verdict_schema"] == audit_page.MANUAL_VERDICT_SCHEMA
    assert summary["training_manifest_allowed"] is False
    assert summary["review_prompt_source"] == "cli:--prompt"
    assert len(summary["review_prompt_sha256"]) == 64
