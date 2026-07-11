from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audits.generate_timeline_teacher_audit_html import build_audit
from tools.datasets.fuse_timeline_teacher_labels import fuse, fuse_unit
from tools.datasets.label_timeline_with_forced_aligner import (
    _build_anchors,
    _prepare_batch_inputs,
)
from tools.datasets.label_timeline_with_omni import (
    DEFAULT_AUDIO_CONTENT_MODE,
    DEFAULT_MAX_TOKENS,
    _normalize_units,
    build_prompt,
)
from tools.datasets.prepare_timeline_teacher_dataset import (
    select_segments,
    validate_partition,
)


def _forced_unit(
    *,
    unit_id: str = "u0000",
    text: str = "帰ろう",
    start_s: float = 1.0,
    end_s: float = 1.4,
    score: float = 0.9,
) -> dict:
    return {
        "unit_id": unit_id,
        "text": text,
        "start_s": start_s,
        "end_s": end_s,
        "alignment_score": score,
    }


def _omni_unit(
    *,
    unit_id: str = "u0000",
    start_s: float = 1.08,
    end_s: float = 1.48,
    confidence: float = 0.95,
    status: str = "matched",
) -> dict:
    return {
        "unit_id": unit_id,
        "status": status,
        "start_s": start_s,
        "end_s": end_s,
        "confidence": confidence,
    }


def test_teacher_selection_prioritizes_long_segments() -> None:
    segments = [
        {"source_chunk_index": 1, "chunk_acoustic_duration": 2.0, "text": "a"},
        {"source_chunk_index": 2, "chunk_acoustic_duration": 8.0, "text": "b"},
        {"source_chunk_index": 3, "chunk_acoustic_duration": 6.0, "text": "c"},
        {"source_chunk_index": 4, "chunk_acoustic_duration": 10.0, "text": ""},
    ]

    selected = select_segments(segments, limit=2, long_threshold_s=5.0, seed=7)

    assert [row["source_chunk_index"] for row in selected] == [2, 3]


@pytest.mark.parametrize("source_id", ["FJIN-059", "FJIN-059-window-a", "NAMH-055"])
def test_heldout_sources_cannot_enter_training_partition(source_id: str) -> None:
    with pytest.raises(ValueError, match="held-out gate"):
        validate_partition(source_id=source_id, split="train")
    validate_partition(source_id=source_id, split="heldout")


def test_forced_aligner_processor_call_matches_hf_model_card() -> None:
    calls = []

    class Processor:
        def prepare_forced_aligner_inputs(self, **kwargs):
            calls.append(kwargs)
            return "inputs", "words"

    result = _prepare_batch_inputs(
        Processor(),
        audio=["audio-a"],
        transcripts=["text-a"],
    )

    assert result == ("inputs", "words")
    assert calls[0]["audio"] == ["audio-a"]
    assert calls[0]["transcript"] == ["text-a"]
    assert calls[0]["language"] == ["Japanese"]
    assert "processor_kwargs" not in calls[0]
    assert "sampling_rate" not in calls[0]
    assert "padding" not in calls[0]
    assert "return_tensors" not in calls[0]


def test_forced_aligner_anchor_contract_rejects_truncated_decode() -> None:
    with pytest.raises(RuntimeError, match="decoded 1 units for 2 expected"):
        _build_anchors(
            item={"absolute_start_s": 10.0},
            words=[{"start_time": 0.1, "end_time": 0.2}],
            word_list=["a", "b"],
            point_confidence=[0.9, 0.9, 0.8, 0.8],
        )


def test_omni_timeline_prompt_is_alignment_only() -> None:
    prompt = build_prompt(
        {
            "duration_s": 3.0,
            "word_units": [_forced_unit()],
        }
    )

    assert "本次唯一任务" in prompt
    assert "不做转录" in prompt
    assert "不要改写、纠错、补充或删除文本" in prompt
    assert "不判断切分点" in prompt
    assert "不判断字幕是否保留" in prompt
    assert '"unit_id": "u0000"' in prompt
    assert "cut|continue|unsure" not in prompt
    assert "keep|drop|unsure" not in prompt
    assert DEFAULT_AUDIO_CONTENT_MODE == "input_audio"
    assert DEFAULT_MAX_TOKENS == 4096


def test_omni_missing_duplicate_and_malformed_units_become_unmatched() -> None:
    expected = [
        _forced_unit(unit_id="u0000", text="a"),
        _forced_unit(unit_id="u0001", text="b"),
        _forced_unit(unit_id="u0002", text="c"),
    ]
    parsed = {
        "units": [
            _omni_unit(unit_id="u0000", start_s=0.1, end_s=0.2),
            _omni_unit(unit_id="u0000", start_s=0.2, end_s=0.3),
            _omni_unit(unit_id="u0002", start_s=0.4, end_s=0.5, confidence="bad"),
        ]
    }

    normalized = _normalize_units(parsed, expected, duration_s=1.0)

    assert [row["status"] for row in normalized] == [
        "unmatched",
        "unmatched",
        "unmatched",
    ]
    assert all(row["confidence"] == 0.0 for row in normalized)


def test_fusion_classifies_consensus_conflict_and_single_teacher_cases() -> None:
    kwargs = {
        "omni_min_confidence": 0.8,
        "agreement_tolerance_s": 0.32,
    }

    consensus = fuse_unit(_forced_unit(), _omni_unit(), **kwargs)
    conflict = fuse_unit(
        _forced_unit(),
        _omni_unit(start_s=2.0, end_s=2.4),
        **kwargs,
    )
    forced_only = fuse_unit(_forced_unit(), None, **kwargs)
    omni_only = fuse_unit(
        _forced_unit(start_s=0.0, end_s=0.0, score=0.2),
        _omni_unit(),
        **kwargs,
    )

    assert consensus["source"] == "forced_omni_consensus"
    assert consensus["trainable"] is True
    assert conflict["source"] == "forced_omni_conflict"
    assert forced_only["source"] == "forced_only_review"
    assert omni_only["source"] == "omni_only_review"
    assert not conflict["trainable"]
    assert not forced_only["trainable"]
    assert not omni_only["trainable"]


def test_fusion_summary_closes_source_and_trainable_counts(tmp_path: Path) -> None:
    forced_path = tmp_path / "forced.jsonl"
    omni_path = tmp_path / "omni.jsonl"
    forced_units = [
        {
            **_forced_unit(unit_id="u0000", text="a"),
            "absolute_start_s": 11.0,
            "absolute_end_s": 11.4,
        },
        {
            **_forced_unit(unit_id="u0001", text="b", start_s=1.5, end_s=1.8),
            "absolute_start_s": 11.5,
            "absolute_end_s": 11.8,
        },
    ]
    forced_path.write_text(
        json.dumps(
            {
                "item_id": "item-a",
                "source_id": "source-a",
                "source_chunk_index": 2,
                "duration_s": 3.0,
                "transcript": "ab",
                "word_units": forced_units,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    omni_path.write_text(
        json.dumps(
            {
                "item_id": "item-a",
                "units": [
                    _omni_unit(unit_id="u0000"),
                    _omni_unit(unit_id="u0001", start_s=1.55, end_s=1.85),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = fuse(
        forced_labels=forced_path,
        omni_labels=omni_path,
        output_dir=tmp_path / "out",
    )
    row = json.loads(
        (tmp_path / "out" / "fused_timeline_labels.jsonl").read_text(
            encoding="utf-8"
        )
    )

    assert summary["source_counts"] == {"forced_omni_consensus": 2}
    assert summary["trainable_item_count"] == 1
    assert row["trainable_coverage"] == 1.0
    assert row["absolute_display_start_s"] > 10.0
    assert row["absolute_display_end_s"] > row["absolute_display_start_s"]


def test_timeline_audit_contains_audio_and_three_teacher_lanes(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav")
    forced_path = tmp_path / "forced.jsonl"
    omni_path = tmp_path / "omni.jsonl"
    fused_path = tmp_path / "fused.jsonl"
    forced_path.write_text(
        json.dumps(
            {
                "item_id": "item-a",
                "source_chunk_index": 1,
                "duration_s": 2.0,
                "transcript": "帰ろう",
                "audio_path": str(audio_path),
                "word_units": [_forced_unit()],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    omni_path.write_text(
        json.dumps({"item_id": "item-a", "units": [_omni_unit()]}) + "\n",
        encoding="utf-8",
    )
    fused_path.write_text(
        json.dumps(
            {
                "item_id": "item-a",
                "trainable_coverage": 1.0,
                "item_trainable": True,
                "units": [
                    {
                        "unit_id": "u0000",
                        "text": "帰ろう",
                        "start_s": 1.04,
                        "end_s": 1.44,
                        "trainable": True,
                        "source": "forced_omni_consensus",
                        "forced_score": 0.9,
                        "omni_confidence": 0.95,
                        "start_delta_s": 0.08,
                        "end_delta_s": 0.08,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_audit(
        forced_labels=forced_path,
        omni_labels=omni_path,
        fused_labels=fused_path,
        output_dir=tmp_path / "audit",
        title="Timeline Audit",
        update_nav=False,
    )
    page = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")

    assert summary["trainable_unit_count"] == 1
    assert "<audio controls" in page
    assert 'class="caption-text"' in page
    assert "function activeUnit" in page
    assert 'data-track="forced"' in page
    assert 'data-track="omni"' in page
    assert 'data-track="fused"' in page
    assert 'class="lane-label">Forced' in page
    assert 'class="lane-label">Omni' in page
    assert 'class="lane-label">Fused' in page
