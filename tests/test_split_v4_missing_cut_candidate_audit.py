from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.audits.generate_split_v4_missing_cut_candidate_audit_html as audit_page
from tools.audits.audit_prompt import resolve_audit_prompt
from tools.audits.review_page_core import validate_audit_option_contract


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_split_missing_cut_candidate_page_is_a_complete_core_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    validate_audit_option_contract(
        axes=audit_page.SPLIT_CANDIDATE_AXES,
        combination_results=audit_page.SPLIT_CANDIDATE_RESULTS,
    )
    args = audit_page.parse_args(
        [
            "--source-audit-dir",
            "source",
            "--verdicts",
            "manual.jsonl",
            "--output-dir",
            "out",
            "--prompt",
            "custom Split review",
        ]
    )
    assert args.prompt == "custom Split review"

    source_dir = tmp_path / "source"
    audio_dir = source_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "source.wav").write_bytes(b"RIFF")
    _write_jsonl(
        source_dir / "audit_manifest.jsonl",
        [
            {
                "audit_id": "long-residual-0001",
                "category": "long_residual",
                "audio_id": "source",
                "audio_src": "audio/source.wav",
                "core_start": 0.0,
                "core_end": 12.0,
                "start_s": 1.0,
                "end_s": 11.0,
                "duration_s": 10.0,
                "residual_candidates": [
                    {
                        "candidate_id": "candidate-a",
                        "time_s": 4.0,
                        "p_cut": 0.4,
                        "model_label": "continue",
                    },
                    {
                        "candidate_id": "candidate-b",
                        "time_s": 8.0,
                        "p_cut": 0.3,
                        "model_label": "continue",
                    },
                ],
            }
        ],
    )
    verdicts = tmp_path / "manual.jsonl"
    _write_jsonl(
        verdicts,
        [{"audit_id": "long-residual-0001", "verdict": "missing_cut"}],
    )
    monkeypatch.setattr(audit_page, "update_audit_entrypoints", lambda **_kwargs: None)
    prompt = resolve_audit_prompt(
        prompt="custom Split review",
        default_prompt=audit_page.DEFAULT_REVIEW_PROMPT,
    )
    output_dir = tmp_path / "out"
    summary = audit_page.build(
        source_dir=source_dir,
        verdict_paths=[verdicts],
        output_dir=output_dir,
        review_prompt=prompt,
        update_latest=False,
    )

    page = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "createAuditReviewCore" in page
    assert "左侧" in page
    assert "右侧" in page
    assert "左右合并" in page
    assert "cut：不同目标事件" in page
    assert "continue：同一目标事件" in page
    assert "unsure" in page
    assert "custom Split review" in page
    assert audit_page.MANUAL_VERDICT_SCHEMA in page
    assert summary["schema"] == audit_page.SUMMARY_SCHEMA
    assert summary["residual_count"] == 1
    assert summary["candidate_count"] == 2
    assert summary["training_manifest_allowed"] is False
    assert summary["review_prompt_source"] == "cli:--prompt"


def test_split_missing_cut_without_candidate_is_routed_to_proposal(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_jsonl(
        source_dir / "audit_manifest.jsonl",
        [
            {
                "audit_id": "long-residual-no-candidate",
                "audio_id": "source",
                "residual_candidates": [],
            }
        ],
    )
    verdicts = tmp_path / "manual.jsonl"
    _write_jsonl(
        verdicts,
        [
            {
                "audit_id": "long-residual-no-candidate",
                "verdict": "missing_cut",
            }
        ],
    )

    with pytest.raises(ValueError, match="Proposal candidate-coverage failure"):
        audit_page.build(
            source_dir=source_dir,
            verdict_paths=[verdicts],
            output_dir=tmp_path / "out",
            update_latest=False,
        )
