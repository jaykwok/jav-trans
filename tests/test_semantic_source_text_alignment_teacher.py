from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audits.generate_semantic_source_text_alignment_audit_html import (
    build_audit,
)
from tools.boundary.ja import label_semantic_source_text_alignment_with_omni as teacher


def _sample() -> dict:
    return {
        "sample_id": "s1",
        "audio": "sample.wav",
        "duration_s": 3.0,
        "source": "test",
        "audit_focus": "mixed source",
        "reference_text": "んちゅぷ…好き",
    }


def _response() -> dict:
    return {
        "sample_id": "s1",
        "text_units": [
            {
                "unit_id": "u00",
                "text": "んちゅぷ…",
                "kind": "nonsemantic",
                "confidence": 0.95,
                "reason": "亲吻拟声",
            },
            {
                "unit_id": "u01",
                "text": "好き",
                "kind": "semantic",
                "confidence": 0.98,
                "reason": "明确词语",
            },
        ],
        "semantic_alignments": [
            {
                "unit_id": "u01",
                "status": "matched",
                "start_s": 2.1,
                "end_s": 2.6,
                "confidence": 0.94,
                "reason": "清楚可辨",
            }
        ],
        "unsure_audio_spans": [],
        "reason": "",
    }


def test_prompt_runs_two_stages_in_one_request() -> None:
    assert "一次请求" in teacher.SYSTEM_PROMPT
    assert "第一步" in teacher.SYSTEM_PROMPT
    assert "第二步" in teacher.SYSTEM_PROMPT
    assert "依次直接拼接后必须逐字符等于" in teacher.SYSTEM_PROMPT
    assert "不要把连续的正常词句按单词或句号切碎" in teacher.SYSTEM_PROMPT
    assert "禁止相邻两个 text_unit 使用相同 kind" in teacher.SYSTEM_PROMPT
    assert "不要返回 keep_span" in teacher.SYSTEM_PROMPT
    assert "只允许桥接多个 semantic spans 之间的内部空洞" in teacher.SYSTEM_PROMPT
    assert "绝不向两侧扩张吸收亲吻声" in teacher.SYSTEM_PROMPT
    assert "不得使用固定秒数 margin" in teacher.SYSTEM_PROMPT
    assert "不要输出最终 Split cut" in teacher.SYSTEM_PROMPT
    prompt = json.loads(teacher.build_prompt(_sample()))
    assert prompt["reference_text"] == "んちゅぷ…好き"
    assert prompt["task_order"] == [
        "split_reference_text_by_semantic_kind",
        "align_semantic_units_only",
    ]
    assert prompt["text_unit_contract"] == "maximal_contiguous_kind_runs"
    assert prompt["membership_contract"] == "derived_matched_semantic_envelope_v1"
    retry_prompt = json.loads(
        teacher.build_prompt(_sample(), validation_feedback="merge maximal kind runs")
    )
    assert retry_prompt["previous_response_validation_error"] == (
        "merge maximal kind runs"
    )


def test_validation_keeps_text_exact_and_contains_semantic_alignment() -> None:
    validated = teacher.validate_response(_response(), _sample())
    assert "".join(row["text"] for row in validated["text_units"]) == "んちゅぷ…好き"
    assert validated["semantic_alignments"][0]["unit_id"] == "u01"
    assert validated["keep_span"]["start_s"] == 2.1
    assert validated["keep_span"]["end_s"] == 2.6
    assert validated["keep_span"]["derivation"] == "matched_semantic_envelope_v1"
    assert validated["keep_span"]["bridged_internal_gap_count"] == 0


def test_validation_rejects_text_rewrite_and_alignment_id_drift() -> None:
    response = _response()
    response["text_units"][0]["text"] = "ちゅぷ…"
    with pytest.raises(ValueError, match="exactly equal"):
        teacher.validate_response(response, _sample())

    response = _response()
    response["semantic_alignments"][0]["unit_id"] = "u00"
    with pytest.raises(ValueError, match="exactly match"):
        teacher.validate_response(response, _sample())


def test_validation_rejects_adjacent_units_with_the_same_kind() -> None:
    response = _response()
    response["text_units"] = [
        {
            "unit_id": "u00",
            "text": "んちゅぷ",
            "kind": "nonsemantic",
            "confidence": 0.95,
            "reason": "kiss",
        },
        {
            "unit_id": "u01",
            "text": "…",
            "kind": "nonsemantic",
            "confidence": 0.95,
            "reason": "pause",
        },
        {
            "unit_id": "u02",
            "text": "好き",
            "kind": "semantic",
            "confidence": 0.95,
            "reason": "word",
        },
    ]
    response["semantic_alignments"][0]["unit_id"] = "u02"
    with pytest.raises(ValueError, match="merge maximal kind runs"):
        teacher.validate_response(response, _sample())


def test_validation_rejects_overlapping_semantic_alignments() -> None:
    sample = {**_sample(), "reference_text": "好き…平気"}
    response = _response()
    response["text_units"] = [
        {
            "unit_id": "u00",
            "text": "好き",
            "kind": "semantic",
            "confidence": 0.9,
            "reason": "word",
        },
        {
            "unit_id": "u01",
            "text": "…",
            "kind": "nonsemantic",
            "confidence": 0.9,
            "reason": "pause",
        },
        {
            "unit_id": "u02",
            "text": "平気",
            "kind": "semantic",
            "confidence": 0.9,
            "reason": "word",
        },
    ]
    response["semantic_alignments"] = [
        {
            "unit_id": "u00",
            "status": "matched",
            "start_s": 1.0,
            "end_s": 1.8,
            "confidence": 0.9,
            "reason": "word",
        },
        {
            "unit_id": "u02",
            "status": "matched",
            "start_s": 1.7,
            "end_s": 2.2,
            "confidence": 0.9,
            "reason": "word",
        },
    ]
    with pytest.raises(ValueError, match="must not overlap"):
        teacher.validate_response(response, sample)


def test_membership_envelope_only_bridges_internal_semantic_gaps() -> None:
    sample = {**_sample(), "reference_text": "好き…平気"}
    response = _response()
    response["text_units"] = [
        {
            "unit_id": "u00",
            "text": "好き",
            "kind": "semantic",
            "confidence": 0.9,
            "reason": "word",
        },
        {
            "unit_id": "u01",
            "text": "…",
            "kind": "nonsemantic",
            "confidence": 0.9,
            "reason": "pause",
        },
        {
            "unit_id": "u02",
            "text": "平気",
            "kind": "semantic",
            "confidence": 0.9,
            "reason": "word",
        },
    ]
    response["semantic_alignments"] = [
        {
            "unit_id": "u00",
            "status": "matched",
            "start_s": 1.0,
            "end_s": 1.8,
            "confidence": 0.9,
            "reason": "word",
        },
        {
            "unit_id": "u02",
            "status": "matched",
            "start_s": 2.0,
            "end_s": 2.2,
            "confidence": 0.8,
            "reason": "word",
        },
    ]
    validated = teacher.validate_response(response, sample)
    assert validated["keep_span"] == {
        "status": "matched",
        "start_s": 1.0,
        "end_s": 2.2,
        "derivation": "matched_semantic_envelope_v1",
        "bridged_internal_gap_count": 1,
        "confidence": 0.8,
        "reason": "由 matched semantic timeline 的最早 start 与最晚 end 确定性派生；只桥接内部空洞，不吸收两侧非语义声音。",
    }


def test_old_model_keep_span_cannot_expand_derived_membership() -> None:
    response = _response()
    response["keep_span"] = {
        "status": "matched",
        "start_s": 0.0,
        "end_s": 3.0,
        "leading_context": "attached_nonsemantic",
        "trailing_context": "attached_nonsemantic",
        "confidence": 1.0,
        "reason": "old contract",
    }
    validated = teacher.validate_response(response, _sample())
    assert validated["keep_span"]["start_s"] == 2.1
    assert validated["keep_span"]["end_s"] == 2.6


def test_nonsemantic_source_uses_empty_alignments_and_no_keep_span() -> None:
    sample = {**_sample(), "reference_text": "んんン～～～っ！"}
    response = {
        "sample_id": "s1",
        "text_units": [
            {
                "unit_id": "u00",
                "text": "んんン～～～っ！",
                "kind": "nonsemantic",
                "confidence": 0.98,
                "reason": "呻吟",
            }
        ],
        "semantic_alignments": [],
        "unsure_audio_spans": [],
        "reason": "",
    }
    validated = teacher.validate_response(response, sample)
    assert validated["semantic_alignments"] == []
    assert validated["keep_span"]["status"] == "none"


def test_unsure_semantic_alignment_abstains_membership() -> None:
    response = _response()
    response["semantic_alignments"][0].update(
        {
            "status": "unsure",
            "start_s": None,
            "end_s": None,
            "confidence": 0.6,
        }
    )
    validated = teacher.validate_response(response, _sample())
    assert validated["keep_span"] == {
        "status": "unsure",
        "start_s": None,
        "end_s": None,
        "derivation": "matched_semantic_envelope_v1",
        "bridged_internal_gap_count": 0,
        "confidence": 0.6,
        "reason": "存在 unsure semantic alignment，membership 整体 abstain。",
    }


def test_audit_separates_three_verdicts_and_explains_membership(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "schema": teacher.SCHEMA,
                "prompt_version": teacher.PROMPT_VERSION,
                "model": teacher.DEFAULT_MODEL,
                "request_contract": "single_full_audio_plus_reference_text",
                **_sample(),
                "audio": str(audio),
                **teacher.validate_response(_response(), _sample()),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    page_path = build_audit(labels=labels, output_dir=tmp_path / "audit")
    page = page_path.read_text(encoding="utf-8")
    assert "1 · 文本语义拆分" in page
    assert "2 · 语义音频对齐" in page
    assert "3 · membership envelope" in page
    assert "不决定 Split chunk" in page
    assert "不是 Omni 的额外判断" in page
    assert "只允许桥接语义区间之间的内部空洞" in page
    assert "绝不向两侧吸收亲吻声" in page
    assert "不显示旧切点" in page
    assert "text_verdict" in page
    assert "timeline_verdict" in page
    assert "membership_verdict" in page
    assert "keep_verdict" not in page
    assert "semantic_source_text_alignment_manual_verdict_v2" in page
    assert "播放该语义 span" in page
    assert "播放 membership envelope" in page
    assert "loadedmetadata" in page
    assert 'preload="metadata"' in page
    assert ").join('\\n')+'\\n'" in page
