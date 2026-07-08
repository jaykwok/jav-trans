from __future__ import annotations

import json
from pathlib import Path

from tools.datasets import label_joint_boundary_preasr_with_omni as joint_omni
from tools.datasets.label_joint_boundary_preasr_with_omni import (
    PROMPT_VERSION,
    _build_pre_asr_prompt,
    _build_prompt,
    _build_split_prompt,
    _normalize_chunk_decision,
    _normalize_split_decision,
    _select_chunk_rows,
    _select_split_rows,
)
from tools.datasets.prepare_joint_boundary_omni_dataset import _window_starts


def test_v3_prompts_keep_omni_requests_single_task() -> None:
    split_prompt = _build_split_prompt(
        [
            {
                "index": 4,
                "time_s": 1.25,
                "label": "continue",
                "p_cut": 0.2,
            }
        ],
        duration_s=3.0,
    )
    pre_asr_prompt = _build_pre_asr_prompt(
        {"duration_s": 2.0},
        item_id="p000",
    )

    assert PROMPT_VERSION == "joint_boundary_preasr_omni_v3_separate"
    assert "split_candidates=" in split_prompt
    assert '"id":"s000"' in split_prompt
    assert "runtime_chunks=" not in split_prompt
    assert "chunk_decisions" not in split_prompt
    assert "missed_boundaries" not in split_prompt
    assert "任务 B" not in split_prompt

    assert 'chunk={"id":"p000"' in pre_asr_prompt
    assert "keep|drop|unsure" in pre_asr_prompt
    assert "split_candidates=" not in pre_asr_prompt
    assert "split_decisions" not in pre_asr_prompt
    assert "missed_boundaries" not in pre_asr_prompt
    assert "任务 A" not in pre_asr_prompt


def test_legacy_joint_prompt_still_documents_v2_shape() -> None:
    prompt = _build_prompt(
        [
            {
                "index": 4,
                "time_s": 1.25,
                "label": "continue",
                "p_cut": 0.2,
            }
        ],
        [
            {
                "chunk_index": 2,
                "start": 0.0,
                "end": 2.0,
            }
        ],
        duration_s=3.0,
    )
    assert "split_candidates=" in prompt
    assert "runtime_chunks=" in prompt
    assert "missed_boundaries" in prompt
    assert '"id":"s000"' in prompt
    assert '"id":"p000"' in prompt
    assert "重复但有明确词义" in prompt


def test_prepare_only_writes_separate_single_task_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_slice_audio_clip(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"wav")

    split_path = tmp_path / "split.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    audio_path = tmp_path / "window.wav"
    mp3_path = tmp_path / "window.mp3"
    audio_path.write_bytes(b"wav")
    mp3_path.write_bytes(b"mp3")
    split_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"index": 1, "time_s": 1.0, "label": "continue", "p_cut": 0.2},
                {"index": 2, "time_s": 2.0, "label": "cut", "p_cut": 0.9},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    chunk_path.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "candidate_id": "c1",
                "audio_id": "a1",
                "video_id": "v1",
                "chunk_index": 0,
                "start": 0.0,
                "end": 1.0,
                "duration_s": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(joint_omni, "slice_audio_clip", fake_slice_audio_clip)

    result = joint_omni._process_window(
        {
            "window_id": "w01",
            "duration_s": 3.0,
            "audio_wav": str(audio_path),
            "omni_mp3_32k": str(mp3_path),
            "semantic_split_metadata": str(split_path),
            "pre_asr_candidates": str(chunk_path),
        },
        output=tmp_path / "out",
        segments_root=tmp_path / "segments",
        split_limit=10,
        chunk_limit=10,
        split_confidence=0.8,
        keep_confidence=0.8,
        drop_confidence=0.9,
        model="qwen3.5-omni-flash",
        api_key="",
        base_url="",
        audio_content_mode="input_audio",
        timeout_s=1.0,
        max_tokens=256,
        prepare_only=True,
    )

    assert result["request_count"] == 2
    assert not list((tmp_path / "out" / "requests" / "split").glob("w01-s*.json"))
    split_request = json.loads(
        (tmp_path / "out" / "requests" / "split" / "w01.json").read_text(
            encoding="utf-8"
        )
    )
    pre_asr_request = json.loads(
        (tmp_path / "out" / "requests" / "pre_asr" / "c1.json").read_text(
            encoding="utf-8"
        )
    )

    assert split_request["request_kind"] == "split_candidates"
    assert len(split_request["split_candidates"]) == 2
    assert "runtime_chunks=" not in split_request["prompt"]
    assert "chunk_decisions" not in split_request["prompt"]
    assert "missed_boundaries" not in split_request["prompt"]

    assert pre_asr_request["request_kind"] == "pre_asr_chunk"
    assert pre_asr_request["audio_scope"] == "chunk"
    assert "split_candidates=" not in pre_asr_request["prompt"]
    assert "split_decisions" not in pre_asr_request["prompt"]


def test_invalid_cut_semantics_become_unsure() -> None:
    decision = _normalize_split_decision(
        {
            "label": "cut",
            "confidence": 0.99,
            "left_complete": True,
            "right_complete": False,
            "merged_better": False,
        },
        confidence_threshold=0.8,
    )
    assert decision["label"] == "unsure"


def test_chunk_confidence_thresholds_are_asymmetric() -> None:
    keep = _normalize_chunk_decision(
        {"label": "keep", "confidence": 0.85},
        keep_confidence=0.8,
        drop_confidence=0.9,
    )
    drop = _normalize_chunk_decision(
        {"label": "drop", "confidence": 0.85},
        keep_confidence=0.8,
        drop_confidence=0.9,
    )
    assert keep["label"] == "definite_keep"
    assert drop["label"] == "ambiguous_ignore"


def test_joint_selection_is_bounded_and_deterministic() -> None:
    split_rows = [
        {
            "index": index,
            "time_s": float(index),
            "p_cut": index / 100.0,
            "accepted": index % 9 == 0,
        }
        for index in range(100)
    ]
    chunk_rows = [
        {
            "chunk_index": index,
            "duration_s": float(index + 1),
        }
        for index in range(100)
    ]
    assert _select_split_rows(split_rows, limit=16, seed="x") == _select_split_rows(
        split_rows, limit=16, seed="x"
    )
    assert len(_select_split_rows(split_rows, limit=16, seed="x")) == 16
    assert _select_chunk_rows(chunk_rows, limit=20, seed="x") == _select_chunk_rows(
        chunk_rows, limit=20, seed="x"
    )
    assert len(_select_chunk_rows(chunk_rows, limit=20, seed="x")) == 20


def test_window_starts_are_reproducible_and_in_bounds() -> None:
    import random

    first = _window_starts(
        duration_s=1000.0,
        window_s=75.0,
        count=2,
        rng=random.Random(17),
    )
    second = _window_starts(
        duration_s=1000.0,
        window_s=75.0,
        count=2,
        rng=random.Random(17),
    )
    assert first == second
    assert len(first) == 2
    assert all(0.0 <= value <= 925.0 for value in first)
