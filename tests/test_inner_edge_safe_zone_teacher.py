from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from tools.boundary.ja import label_inner_edge_safe_zone_with_omni as teacher
from tools.audits import generate_inner_edge_safe_zone_audit_html as audit


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_select_boundaries_uses_adaptive_midpoint_region_and_seven_candidates(tmp_path: Path) -> None:
    islands = tmp_path / "islands.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write(islands, [{"island_id": "w#outer000", "window_id": "w", "duration_s": 10.0, "span_start_s": 1.0, "span_end_s": 11.0, "source_audio": "w.wav", "candidates": [{"feature_index": i, "relative_time_s": i / 2, "p_cut": i / 20} for i in range(1, 20)]}])
    _write(labels, [{"island_id": "w#outer000", "cuts": [{"time_s": 2.0}, {"time_s": 5.0}, {"time_s": 8.0}]}])

    selected = teacher.select_boundaries(islands, labels, ["w#outer000#b001"])

    assert len(selected[0]["candidates"]) == 7
    assert [row["relative_time_s"] for row in selected[0]["candidates"]] == [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]


def test_safe_zone_response_requires_exact_four_class_candidate_sequence() -> None:
    boundary = {"boundary_id": "w#outer000#b000", "candidates": [{"candidate_id": "c00"}, {"candidate_id": "c01"}]}
    rows = teacher._validate({"boundary_id": boundary["boundary_id"], "candidates": [{"candidate_id": "c00", "label": "left_clipped", "confidence": 0.9}, {"candidate_id": "c01", "label": "safe", "confidence": 0.8}]}, boundary)
    assert [row["label"] for row in rows] == ["left_clipped", "safe"]
    with pytest.raises(ValueError, match="candidate ids"):
        teacher._validate({"boundary_id": boundary["boundary_id"], "candidates": [{"candidate_id": "c01", "label": "safe", "confidence": 0.8}]}, boundary)
    with pytest.raises(ValueError, match="monotonic"):
        teacher._validate({"boundary_id": boundary["boundary_id"], "candidates": [{"candidate_id": "c00", "label": "right_clipped", "confidence": 0.9}, {"candidate_id": "c01", "label": "left_clipped", "confidence": 0.8}]}, boundary)


def test_prompt_has_no_transcript_or_timestamp_teacher() -> None:
    assert "不要转录" in teacher.SYSTEM_PROMPT
    assert "不要输出时间戳" in teacher.SYSTEM_PROMPT
    assert "left_clipped" in teacher.SYSTEM_PROMPT
    assert "句尾 mora 零容忍" in teacher.SYSTEM_PROMPT
    assert "left_clipped，随后" in teacher.SYSTEM_PROMPT


def test_audit_only_shows_current_candidate_plans(tmp_path: Path, monkeypatch) -> None:
    original = tmp_path / "original.wav"
    candidate = tmp_path / "c00.wav"
    original.write_bytes(b"wav")
    candidate.write_bytes(b"wav")
    selected = tmp_path / "selected.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write(selected, [{"boundary_id": "w#outer000#b000", "original_audio": str(original), "candidates": [{"candidate_id": "c00", "audio": str(candidate)}]}])
    _write(labels, [{"boundary_id": "w#outer000#b000", "candidates": [{"candidate_id": "c00", "label": "safe", "confidence": 0.9, "reason": "ok"}]}])
    monkeypatch.setattr(audit, "update_audit_entrypoints", lambda **_kwargs: None)
    summary = audit.build_audit(selected=selected, labels=labels, output_dir=tmp_path / "audit")
    page = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")
    assert summary["item_count"] == 1
    assert "你要做什么" in page
    assert "灰色长条只是浏览器音频播放器，不是评分进度" in page
    assert "播放插入点附近" in page
    assert "你的判断" in page
    assert "太早 / left_clipped" in page
    assert "安全 / safe" in page
    assert "旧切点、旧 Refiner 或 Timeline" in page
    assert "shared_absolute_cut" not in page
    script = re.search(r"<script>([\s\S]*?)</script>", page)
    assert script is not None
    script_path = tmp_path / "audit.js"
    script_path.write_text(script.group(1), encoding="utf-8")
    subprocess.run(["node", "--check", str(script_path)], check=True)
