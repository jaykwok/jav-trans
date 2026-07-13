from __future__ import annotations

import json
import re
from pathlib import Path

from tools.audits.generate_semantic_speech_outer_audit_html import build_audit
from tools.boundary.ja import label_semantic_speech_outer_with_omni as teacher


def test_teacher_contract_is_semantic_foreground_not_any_voice() -> None:
    assert teacher.LABELS == ("discardable", "semantic_target", "unsure")
    assert "纯喘息、呻吟、亲吻声、笑声" in teacher.SYSTEM_PROMPT
    assert "远处/嘈杂/不可辨" in teacher.SYSTEM_PROMPT
    assert "疑似包含词语" in teacher.SYSTEM_PROMPT
    assert "不要判断内部语义切分" in teacher.SYSTEM_PROMPT
    assert "不要转录、引用或改写任何具体台词" in teacher.SYSTEM_PROMPT


def test_teacher_validation_requires_contiguous_three_class_coverage() -> None:
    sample = {"sample_id": "s1", "duration_s": 2.0}
    segments = teacher._validate(
        {
            "sample_id": "s1",
            "segments": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "label": "discardable",
                    "confidence": 0.9,
                },
                {
                    "start_s": 1.0,
                    "end_s": 2.0,
                    "label": "semantic_target",
                    "confidence": 0.8,
                },
            ],
        },
        sample,
    )
    assert [row["label"] for row in segments] == [
        "discardable",
        "semantic_target",
    ]


def test_audit_explains_timeline_and_editable_labels(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "audio": str(audio),
                "duration_s": 2.0,
                "source": "test",
                "audit_focus": "focus",
                "segments": [
                    {
                        "start_s": 0.0,
                        "end_s": 2.0,
                        "label": "unsure",
                        "confidence": 0.5,
                        "reason": "unclear",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    page_path = build_audit(labels=labels, output_dir=tmp_path / "audit")
    page = page_path.read_text(encoding="utf-8")
    assert "彩色长条是时间轴，不是进度条" in page
    assert "喘息、呻吟、亲吻声、笑声" in page
    assert "通过（含我的修正）" in page
    assert "▶ 仅此区间" in page
    assert "Number(start)-0.8" not in page
    assert "Number(end)+0.8" not in page
    assert "semantic-speech-outer-smoke5-v1" in page
    assert re.search(r"<script>[\s\S]+</script>", page)
