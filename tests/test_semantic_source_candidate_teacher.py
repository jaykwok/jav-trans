from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from tools.audits.generate_semantic_source_candidate_audit_html import build_audit
from tools.boundary.ja import label_semantic_source_candidates_with_omni as teacher


def test_teacher_contract_is_candidate_classification_without_timestamps() -> None:
    assert teacher.LABELS == ("discardable", "semantic_target", "unsure")
    assert "纯喘息、呻吟、亲吻声、笑声" in teacher.SYSTEM_PROMPT
    assert "远处/嘈杂/不可辨" in teacher.SYSTEM_PROMPT
    assert "疑似包含词语" in teacher.SYSTEM_PROMPT
    assert "不要输出时间戳" in teacher.SYSTEM_PROMPT
    assert "不要转录、引用或改写具体台词" in teacher.SYSTEM_PROMPT
    assert "1 秒静音" in teacher.SYSTEM_PROMPT
    assert "可信数据集参考文本" in teacher.SYSTEM_PROMPT
    assert "不得根据文本猜测标记位置" in teacher.SYSTEM_PROMPT
    assert "start_s" not in teacher.SYSTEM_PROMPT
    assert "end_s" not in teacher.SYSTEM_PROMPT
    assert teacher.SOURCE_LABELS == ("discardable", "contains_semantic", "unsure")
    assert "整条只有喘息、呻吟、亲吻声、笑声" in teacher.SOURCE_GATE_PROMPT
    assert "不要输出时间戳" in teacher.SOURCE_GATE_PROMPT
    assert "start_s" not in teacher.SOURCE_GATE_PROMPT
    assert "end_s" not in teacher.SOURCE_GATE_PROMPT
    assert teacher.DEFAULT_PROJECTION.suffix == ".pt"
    assert "Qwen3-ASR-1.7B" in str(teacher.DEFAULT_PROJECTION)


def test_learned_farthest_medoid_selection_and_cells_are_deterministic() -> None:
    embeddings = np.asarray(
        [[float(index), float(index % 3), float(index % 5)] for index in range(30)],
        dtype=np.float32,
    )
    selected = teacher.select_candidate_frames(embeddings, 7)
    assert selected == teacher.select_candidate_frames(embeddings, 7)
    assert selected == sorted(set(selected))
    assert len(selected) == 7
    cells = teacher.candidate_cells(
        selected, frame_hop_s=0.02, duration_s=0.6
    )
    assert cells[0]["context_start_s"] == 0.0
    assert cells[-1]["context_end_s"] == 0.6
    assert all(
        left["context_end_s"] == right["context_start_s"]
        for left, right in zip(cells[:-1], cells[1:])
    )
    assert all(
        row["context_start_s"] < row["marker_s"] < row["context_end_s"]
        for row in cells
    )


def test_candidate_projection_uses_full_ptm_tail_not_front128() -> None:
    ptm = np.zeros((2, 2048), dtype=np.float32)
    ptm[1, 2047] = 3.0
    mean = np.zeros(2048, dtype=np.float32)
    components = np.zeros((2, 2048), dtype=np.float32)
    components[0, 2047] = 2.0

    projected = teacher.learned_frame_embeddings(
        ptm=ptm,
        projection_mean=mean,
        projection_components=components,
    )

    assert projected.shape == (2, 2)
    assert projected[0, 0] == 0.0
    assert projected[1, 0] == 6.0


def test_validation_requires_exact_candidate_order_and_three_classes() -> None:
    sample = {
        "sample_id": "s1",
        "candidates": [{"candidate_id": "c00"}, {"candidate_id": "c01"}],
    }
    candidates = teacher._validate(
        {
            "sample_id": "s1",
            "candidates": [
                {
                    "candidate_id": "c00",
                    "label": "discardable",
                    "confidence": 0.9,
                    "reason": "背景音乐",
                },
                {
                    "candidate_id": "c01",
                    "label": "semantic_target",
                    "confidence": 0.8,
                    "reason": "清楚前景语言",
                },
            ],
        },
        sample,
    )
    assert [row["label"] for row in candidates] == [
        "discardable",
        "semantic_target",
    ]


def test_source_gate_validation_uses_v2_hierarchical_labels() -> None:
    sample = {"sample_id": "s1"}
    validated = teacher._validate_source_gate(
        {
            "sample_id": "s1",
            "label": "contains_semantic",
            "confidence": 0.91,
            "reason": "可辨短词与非词声音并存",
        },
        sample,
    )
    assert validated == {
        "sample_id": "s1",
        "label": "contains_semantic",
        "confidence": 0.91,
        "reason": "可辨短词与非词声音并存",
    }


def test_audit_explains_marker_and_has_no_timing_editor(tmp_path: Path) -> None:
    full = tmp_path / "full.wav"
    original = tmp_path / "original.wav"
    marked = tmp_path / "marked.wav"
    for path in (full, original, marked):
        path.write_bytes(b"RIFF")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "audio": str(full),
                "duration_s": 2.0,
                "source": "test",
                "audit_focus": "focus",
                "reference_text": "固定参考文本",
                "source_gate": {
                    "sample_id": "s1",
                    "label": "contains_semantic",
                    "confidence": 0.9,
                    "reason": "可辨短词",
                },
                "selection_mode": "learned_full_ptm_projection_farthest_medoid_v2",
                "candidates": [
                    {
                        "candidate_id": "c00",
                        "feature_index": 4,
                        "marker_s": 0.09,
                        "context_start_s": 0.0,
                        "context_end_s": 2.0,
                        "original_audio": str(original),
                        "marked_audio": str(marked),
                        "label": "unsure",
                        "confidence": 0.5,
                        "reason": "unclear",
                        "label_source": "candidate_marker",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    expectations = tmp_path / "expectations.json"
    expectations.write_text(
        json.dumps({"s1": "只约束本例：最多两块"}, ensure_ascii=False),
        encoding="utf-8",
    )
    page_path = build_audit(
        labels=labels,
        output_dir=tmp_path / "audit",
        semantic_expectations=expectations,
    )
    page = page_path.read_text(encoding="utf-8")
    assert "原始候选邻域（非 chunk）" in page
    assert "1 秒静音标记版" in page
    assert "只判断静音标记紧邻位置" in page
    assert "固定参考文本" in page
    assert "不显示旧切点" in page
    assert "页面不允许修改时间" in page
    assert "source gate=" in page
    assert "learned full-PTM 2048→128 projection" in page
    assert "候选行不是 chunk" in page
    assert "membership 输出 1 个 coarse island" in page
    assert "Layer 1 · content 证据" in page
    assert "Layer 2 · source membership / Outer" in page
    assert "Layer 3 · 本样本 Split 期望（只约束本例）" in page
    assert "只约束本例：最多两块" in page
    assert "semantic-source-candidate-audit-v5" in page
    assert 'data-field="start_s"' not in page
    assert 'data-field="end_s"' not in page
    assert "semantic_source_candidate_manual_verdict_v5" in page
    assert re.search(r"<script>[\s\S]+</script>", page)
