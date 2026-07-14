from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audits.generate_semantic_timeline_audit_html import build_audit
from tools.boundary.ja.compile_semantic_timeline_training_views import compile_views
from tools.boundary.ja import label_semantic_timeline_with_omni as teacher


def _sample() -> dict:
    return {
        "sample_id": "s1",
        "audio": "sample.wav",
        "duration_s": 8.0,
        "source": "test",
        "audit_focus": "multi-unit",
        "reference_text": "それと、もう一つ。彼は、たぶん甘えたくなかったんだと思う",
    }


def _response() -> dict:
    return {
        "sample_id": "s1",
        "text_units": [
            {
                "unit_id": "u00",
                "text": "それと、もう一つ。",
                "kind": "semantic",
                "confidence": 0.99,
                "reason": "独立补充",
            },
            {
                "unit_id": "u01",
                "text": "彼は、たぶん甘えたくなかったんだと思う",
                "kind": "semantic",
                "confidence": 0.98,
                "reason": "完整陈述",
            },
        ],
        "semantic_alignments": [
            {
                "unit_id": "u00",
                "status": "matched",
                "start_s": 0.1,
                "end_s": 1.7,
                "confidence": 0.95,
                "reason": "清楚",
            },
            {
                "unit_id": "u01",
                "status": "matched",
                "start_s": 2.0,
                "end_s": 7.5,
                "confidence": 0.94,
                "reason": "清楚",
            },
        ],
        "unsure_audio_spans": [],
        "reason": "two units",
    }


def test_prompt_builds_reusable_training_contract() -> None:
    assert "最小但完整" in teacher.SYSTEM_PROMPT
    assert "それと、もう一つ。" in teacher.SYSTEM_PROMPT
    assert "昨年の外部生の受け入れに続き、" in teacher.SYSTEM_PROMPT
    assert "不是最终 Split cut" in teacher.SYSTEM_PROMPT
    assert "不会直接当作 CueQC drop 或 Inner safe" in teacher.SYSTEM_PROMPT
    prompt = json.loads(teacher.build_prompt(_sample()))
    assert prompt["task_order"] == [
        "segment_reference_text_into_minimal_complete_semantic_units",
        "align_each_semantic_unit_to_full_audio",
    ]
    assert prompt["downstream_contract"]["inner_refiner"] == (
        "requires separate candidate safe-zone labels"
    )


def test_adjacent_semantic_units_derive_distinct_model_views() -> None:
    validated = teacher.validate_response(_response(), _sample())

    assert [row["unit_id"] for row in validated["semantic_timeline"]] == [
        "u00",
        "u01",
    ]
    assert validated["semantic_events"] == [
        {
            "event_id": "e00",
            "left_unit_id": "u00",
            "right_unit_id": "u01",
            "status": "matched",
            "interval_start_s": 1.7,
            "interval_end_s": 2.0,
            "overlap": False,
        }
    ]
    scorer = validated["scorer_view"]
    assert scorer["source_membership"] == {
        "status": "matched",
        "start_s": 0.1,
        "end_s": 7.5,
    }
    assert scorer["nonsemantic_complement_spans"] == [
        {"start_s": 0.0, "end_s": 0.1},
        {"start_s": 1.7, "end_s": 2.0},
        {"start_s": 7.5, "end_s": 8.0},
    ]
    assert validated["outer_refiner_view"] == {
        "status": "matched",
        "left_speech_start_s": 0.1,
        "right_speech_end_s": 7.5,
    }
    assert validated["inner_refiner_view"]["status"] == (
        "requires_candidate_safe_zone_teacher"
    )
    assert validated["cueqc_view"]["status"] == (
        "not_labeled_until_new_chunks_are_exported"
    )


def test_overlap_stays_semantic_event_but_routes_inner_to_abstain() -> None:
    response = _response()
    response["semantic_alignments"][1]["start_s"] = 1.5
    validated = teacher.validate_response(response, _sample())

    assert validated["semantic_events"][0]["overlap"] is True
    assert validated["semantic_events"][0]["interval_start_s"] == 1.7
    assert validated["semantic_events"][0]["interval_end_s"] == 1.5


def test_unsure_alignment_disables_complement_negative_labels() -> None:
    response = _response()
    response["semantic_alignments"][1].update(
        {"status": "unsure", "start_s": None, "end_s": None}
    )
    validated = teacher.validate_response(response, _sample())

    assert validated["scorer_view"]["source_membership"]["status"] == "unsure"
    assert validated["scorer_view"]["nonsemantic_complement_spans"] == []
    assert validated["semantic_events"][0]["status"] == "unsure"


def test_validation_preserves_reference_text_and_alignment_order() -> None:
    response = _response()
    response["text_units"][0]["text"] = "それともう一つ。"
    with pytest.raises(ValueError, match="exactly equal"):
        teacher.validate_response(response, _sample())

    response = _response()
    response["semantic_alignments"].reverse()
    with pytest.raises(ValueError, match="exactly match"):
        teacher.validate_response(response, _sample())


def test_audit_explains_model_specific_training_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "schema": teacher.SCHEMA,
                "prompt_version": teacher.PROMPT_VERSION,
                "model": teacher.DEFAULT_MODEL,
                "request_contract": "single_full_audio_plus_trusted_reference_text",
                **_sample(),
                "audio": str(audio),
                **teacher.validate_response(_response(), _sample()),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tools.audits.generate_semantic_timeline_audit_html.update_audit_entrypoints",
        lambda **_kwargs: None,
    )

    page = build_audit(
        labels=labels, output_dir=tmp_path / "audit", update_latest=True
    ).read_text(encoding="utf-8")

    assert "这不是 atomic-core 审计" in page
    assert "Scorer / Outer / Split / Inner" in page
    assert "CueQC 不在本页" in page
    assert "语义 event 的区间中点示意，不是最终 cut" in page
    assert "semantic_timeline_manual_verdict_v1" in page
    assert "播放此 semantic unit" in page


def test_compiler_emits_model_specific_manifests_without_fake_inner_labels(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "schema": teacher.SCHEMA,
                "prompt_version": teacher.PROMPT_VERSION,
                "model": teacher.DEFAULT_MODEL,
                "request_contract": "single_full_audio_plus_trusted_reference_text",
                **_sample(),
                **teacher.validate_response(_response(), _sample()),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = compile_views(labels=labels, output_dir=tmp_path / "compiled")

    assert summary["max_source_use_count"] == 1
    assert summary["training_ready"] is False
    assert summary["inner_training_ready"] is False
    assert summary["counts"] == {
        "scorer": 1,
        "outer_refiner": 1,
        "semantic_split": 1,
        "inner_refiner_event_seeds": 1,
    }
    scorer = json.loads(
        (tmp_path / "compiled" / "scorer_sources.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert scorer["nonsemantic_frame_spans"][1] == {
        "start_s": 1.7,
        "end_s": 2.0,
    }
    inner = json.loads(
        (tmp_path / "compiled" / "inner_refiner_event_seeds.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert inner["label_status"] == "requires_candidate_safe_zone_teacher"
    assert inner["training_eligible"] is False


def test_compiler_rejects_source_reuse(tmp_path: Path) -> None:
    base = {
        "schema": teacher.SCHEMA,
        "prompt_version": teacher.PROMPT_VERSION,
        "model": teacher.DEFAULT_MODEL,
        "request_contract": "single_full_audio_plus_trusted_reference_text",
        **_sample(),
        **teacher.validate_response(_response(), _sample()),
    }
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps({**base, "sample_id": sample_id})
            for sample_id in ("s1", "s2")
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source/core may appear at most once"):
        compile_views(labels=labels, output_dir=tmp_path / "compiled")
