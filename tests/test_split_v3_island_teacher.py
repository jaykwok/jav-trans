from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audits import generate_split_v3_island_audit_html as audit
from tools.boundary.ja import label_semantic_split_islands_with_omni as teacher


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_collect_islands_reconstructs_pre_split_outer_groups(tmp_path: Path) -> None:
    features = tmp_path / "features"
    audio = tmp_path / "window.wav"
    audio.write_bytes(b"wav")
    chunks = features / "pre_asr.jsonl"
    boundary = features / "boundary.jsonl"
    _write_jsonl(
        chunks,
        [
            {"chunk_index": 0, "raw_start": 0.0, "raw_end": 2.0, "acoustic_start": 0.1, "acoustic_end": 2.0},
            {"chunk_index": 1, "raw_start": 2.0, "raw_end": 4.0, "acoustic_start": 2.0, "acoustic_end": 3.9},
            {"chunk_index": 2, "raw_start": 5.0, "raw_end": 7.0, "acoustic_start": 5.1, "acoustic_end": 6.9},
        ],
    )
    _write_jsonl(
        boundary,
        [
            {"frame": 100, "time_s": 2.0, "kind": "primary", "accepted": True, "label": "cut", "p_cut": 0.9},
            {"frame": 100, "time_s": 2.0, "kind": "primary", "accepted": True, "label": "cut", "p_cut": 0.9},
            {"frame": 300, "time_s": 6.0, "kind": "weak", "accepted": False, "label": "continue", "p_cut": 0.2},
        ],
    )
    _write_jsonl(
        tmp_path / "source_windows.jsonl",
        [{"window_id": "w", "video_id": "v", "audio_wav": str(audio), "boundary_audit": str(boundary), "pre_asr_candidates": str(chunks)}],
    )

    islands = teacher.collect_islands(tmp_path)

    assert [(round(row["span_start_s"], 1), round(row["span_end_s"], 1)) for row in islands] == [(0.1, 3.9), (5.1, 6.9)]
    assert len(islands[0]["candidates"]) == 1
    assert islands[0]["candidates"][0]["relative_time_s"] == pytest.approx(1.9)


def test_island_response_requires_complete_ordered_search() -> None:
    island = {"island_id": "w#outer000", "duration_s": 5.0}
    teacher._validate_response(
        {"island_id": "w#outer000", "complete_search": True, "cuts": [{"time_s": 2.0, "confidence": 0.9}]},
        island,
    )
    with pytest.raises(ValueError, match="complete_search"):
        teacher._validate_response({"island_id": "w#outer000", "complete_search": False, "cuts": []}, island)
    with pytest.raises(ValueError, match="strictly ordered"):
        teacher._validate_response(
            {"island_id": "w#outer000", "complete_search": True, "cuts": [{"time_s": 3.0, "confidence": 0.9}, {"time_s": 2.0, "confidence": 0.9}]},
            island,
        )


def test_island_prompt_is_cut_eager_without_duration_rule() -> None:
    assert teacher.PROMPT_VERSION.endswith("cut_eager_complete_v3")
    assert "短暂停顿可以是切点" in teacher.SYSTEM_PROMPT
    assert "不要求左右都是语法上的完整长句" in teacher.SYSTEM_PROMPT
    assert "时长不是硬规则" in teacher.SYSTEM_PROMPT
    assert "再从尾到头复核一遍" in teacher.SYSTEM_PROMPT
    assert "短暂停顿、呼吸" not in teacher.SYSTEM_PROMPT


def test_island_audit_shows_omni_and_current_markers(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    request_audio = run / "request_audio" / "w__outer000.wav"
    request_audio.parent.mkdir(parents=True)
    request_audio.write_bytes(b"wav")
    selected = run / "selected.jsonl"
    labels = run / "labels.jsonl"
    _write_jsonl(selected, [{"island_id": "w#outer000", "candidates": [{"relative_time_s": 2.0, "accepted": True}]}])
    _write_jsonl(labels, [{"island_id": "w#outer000", "duration_s": 5.0, "cuts": [{"time_s": 3.0}], "reason": "test"}])
    monkeypatch.setattr(audit, "update_audit_entrypoints", lambda **_kwargs: None)
    monkeypatch.setattr(
        audit,
        "_build_segmented_audio",
        lambda **kwargs: kwargs["output"].write_bytes(b"segmented"),
    )

    summary = audit.build_audit(selected=selected, labels=labels, output_dir=tmp_path / "audit")
    page = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")

    assert summary["item_count"] == 1
    assert "Omni" in page
    assert "现役 accepted" in page
    assert "Omni 整体正确" in page
    assert "裁决对象：Omni current 红线" in page
    assert "现役 accepted（仅对照）" in page
    assert "原始 speech island" in page
    assert "块间静音 1 秒" in page
    assert "segmented_audio" in page
    assert "manual_verdicts.jsonl" in page
