from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools.audits.generate_omni_timeline_audit_html import build_audit
from tools.datasets import label_timeline_with_omni as omni_label
from tools.datasets.label_timeline_with_omni import (
    DEFAULT_AUDIO_CONTENT_MODE,
    DEFAULT_MAX_TOKENS,
    SYSTEM_PROMPT,
    _normalize_units,
    build_prompt,
)
from tools.datasets.prepare_timeline_teacher_dataset import (
    UNITIZER,
    select_segments,
    split_text_units,
    validate_partition,
)


def _text_units() -> list[dict[str, str]]:
    return [{"unit_id": "u0000", "text": "帰ろうか"}]


def _item(*, audio_path: str = "sample.wav") -> dict:
    return {
        "schema": "timeline_teacher_item_v2",
        "item_id": "item-a",
        "source_id": "source-a",
        "source_chunk_index": 2,
        "duration_s": 3.0,
        "transcript": "...帰ろうか?",
        "text_units": _text_units(),
        "unitizer": UNITIZER,
        "audio_path": audio_path,
    }


def _omni_unit(
    *,
    unit_id: str = "u0000",
    text: str = "帰ろう",
    start_s: float = 0.8,
    end_s: float = 1.4,
    confidence: float = 0.95,
    status: str = "matched",
) -> dict:
    return {
        "unit_id": unit_id,
        "text": text,
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


def test_zero_dependency_unitizer_splits_only_on_existing_punctuation() -> None:
    units = split_text_units("見なきゃいけない。父さん...母さん...俺と母さんの関係は?")

    assert [unit["text"] for unit in units] == [
        "見なきゃいけない",
        "父さん",
        "母さん",
        "俺と母さんの関係は",
    ]
    assert [unit["unit_id"] for unit in units] == [f"u{i:04d}" for i in range(4)]


@pytest.mark.parametrize("source_id", ["FJIN-059", "FJIN-059-window-a", "NAMH-055"])
def test_heldout_sources_cannot_enter_training_partition(source_id: str) -> None:
    with pytest.raises(ValueError, match="held-out gate"):
        validate_partition(source_id=source_id, split="train")
    validate_partition(source_id=source_id, split="heldout")


def test_omni_timeline_prompt_is_direct_alignment_only() -> None:
    prompt = build_prompt(_item())

    assert "唯一任务" in SYSTEM_PROMPT
    assert "不是转录" in SYSTEM_PROMPT
    assert "禁止改写、纠错、补充、删除" in SYSTEM_PROMPT
    assert "禁止判断 Split 切点" in SYSTEM_PROMPT
    assert "Pre-ASR keep/drop" in SYSTEM_PROMPT
    assert '"unit_id":"u0000"' in prompt
    assert json.loads(prompt)["duration_s"] == 3.0
    assert "Forced" not in SYSTEM_PROMPT
    assert "cut|continue|unsure" not in SYSTEM_PROMPT
    assert "keep|drop|unsure" not in SYSTEM_PROMPT
    assert DEFAULT_AUDIO_CONTENT_MODE == "input_audio"
    assert DEFAULT_MAX_TOKENS == 4096


def test_omni_missing_duplicate_and_malformed_units_become_unmatched() -> None:
    expected = [
        {"unit_id": "u0000", "text": "a"},
        {"unit_id": "u0001", "text": "b"},
        {"unit_id": "u0002", "text": "c"},
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


def test_omni_run_consumes_v2_items_without_forced_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav")
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(json.dumps(_item(audio_path=str(audio_path))) + "\n", encoding="utf-8")

    def fake_call_omni(**kwargs):
        assert "Forced" not in kwargs["prompt"]
        assert kwargs["system_prompt"] == SYSTEM_PROMPT
        assert kwargs["enable_thinking"] is True
        assert kwargs["thinking_budget"] == 1024
        return (
            {
                "units": [_omni_unit(text="帰ろうか")]
            },
            {"response": "ok"},
        )

    monkeypatch.setattr(omni_label, "call_omni", fake_call_omni)
    monkeypatch.setattr(omni_label, "load_env_file", lambda _path: None)
    monkeypatch.setattr(omni_label, "first_env_value", lambda _names: ("", ""))
    summary = omni_label.run(
        argparse.Namespace(
            items=str(items_path),
            output_dir=str(tmp_path / "out"),
            model="qwen3.5-omni-flash",
            env_file="",
            audio_content_mode="input_audio",
            timeout_s=10.0,
            max_tokens=4096,
            rpm=0.0,
            limit=0,
            max_attempts=6,
        )
    )
    row = json.loads(
        (tmp_path / "out" / "omni_timeline_labels.jsonl").read_text(encoding="utf-8")
    )

    assert summary["schema"] == "timeline_omni_alignment_summary_v3"
    assert row["schema"] == "timeline_omni_alignment_label_v3"
    assert row["matched_count"] == 1
    assert row["unitizer"] == UNITIZER
    assert row["attempts"] == 1
    assert summary["failed"] == 0


def test_omni_timeline_audit_has_one_caption_track(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav")
    items_path = tmp_path / "items.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    item = _item(audio_path=str(audio_path))
    unlabeled_item = {**item, "item_id": "item-b", "source_chunk_index": 3}
    items_path.write_text(
        json.dumps(item) + "\n" + json.dumps(unlabeled_item) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(
            {
                "item_id": "item-a",
                "units": [_omni_unit(text="帰ろうか")],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_audit(
        items_path=items_path,
        omni_labels=labels_path,
        output_dir=tmp_path / "audit",
        title="Omni Timeline Audit",
        update_nav=False,
    )
    page = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")

    assert summary["matched_unit_count"] == 1
    assert summary["item_count"] == 1
    assert "<audio controls" in page
    assert 'class="caption-text"' in page
    assert "function activeUnit" in page
    assert "audio.paused && audio.currentTime === 0" in page
    assert 'class="lane-label">Omni' in page
    assert "Forced" not in page
    assert "Selected" not in page


def test_omni_timeline_retries_incomplete_response(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav")
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(
        json.dumps(_item(audio_path=str(audio_path))) + "\n",
        encoding="utf-8",
    )
    calls = {"count": 0}

    def fake_call_omni(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"units": []}, {"usage": {}}
        return {"units": [_omni_unit(text="帰ろうか")]}, {"usage": {}}

    monkeypatch.setattr(omni_label, "call_omni", fake_call_omni)
    monkeypatch.setattr(omni_label, "load_env_file", lambda _path: None)
    monkeypatch.setattr(omni_label, "first_env_value", lambda _names: ("", ""))
    monkeypatch.setattr(omni_label.time, "sleep", lambda _seconds: None)
    summary = omni_label.run(
        argparse.Namespace(
            items=str(items_path),
            output_dir=str(tmp_path / "out"),
            model="",
            env_file="",
            audio_content_mode="input_audio",
            timeout_s=10.0,
            max_tokens=4096,
            enable_thinking=True,
            thinking_budget=1024,
            rpm=0.0,
            max_attempts=2,
            limit=0,
        )
    )
    row = json.loads(
        (tmp_path / "out" / "omni_timeline_labels.jsonl").read_text(encoding="utf-8")
    )

    assert calls["count"] == 2
    assert row["model"] == "qwen3.5-omni-plus"
    assert row["attempts"] == 2
    assert summary["request_attempts"] == 2


def test_omni_timeline_empty_audio_becomes_unmatched(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(
        json.dumps(_item(audio_path=str(audio_path))) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        omni_label,
        "call_omni",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("The audio is empty")),
    )
    monkeypatch.setattr(omni_label, "load_env_file", lambda _path: None)
    monkeypatch.setattr(omni_label, "first_env_value", lambda _names: ("", ""))
    summary = omni_label.run(
        argparse.Namespace(
            items=str(items_path),
            output_dir=str(tmp_path / "out"),
            model="qwen3.5-omni-plus",
            env_file="",
            audio_content_mode="input_audio",
            timeout_s=10.0,
            max_tokens=4096,
            enable_thinking=True,
            thinking_budget=1024,
            rpm=0.0,
            max_attempts=6,
            limit=0,
        )
    )
    row = json.loads(
        (tmp_path / "out" / "omni_timeline_labels.jsonl").read_text(encoding="utf-8")
    )

    assert row["units"][0]["status"] == "unmatched"
    assert row["unmatched_count"] == 1
    assert summary["failed"] == 0


def test_omni_timeline_persistent_error_writes_retry_manifest(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav")
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(
        json.dumps(_item(audio_path=str(audio_path))) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        omni_label,
        "call_omni",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("429 rate limit")),
    )
    monkeypatch.setattr(omni_label, "load_env_file", lambda _path: None)
    monkeypatch.setattr(omni_label, "first_env_value", lambda _names: ("", ""))
    monkeypatch.setattr(omni_label.time, "sleep", lambda _seconds: None)
    summary = omni_label.run(
        argparse.Namespace(
            items=str(items_path),
            output_dir=str(tmp_path / "out"),
            model="qwen3.5-omni-plus",
            env_file="",
            audio_content_mode="input_audio",
            timeout_s=10.0,
            max_tokens=4096,
            enable_thinking=True,
            thinking_budget=1024,
            rpm=0.0,
            max_attempts=2,
            limit=0,
        )
    )
    retries = [
        json.loads(line)
        for line in (tmp_path / "out" / "timeline_retry_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert summary["processed"] == 0
    assert summary["failed"] == 1
    assert retries[0]["item_id"] == "item-a"
    assert not (tmp_path / "out" / "omni_timeline_labels.jsonl").exists()
