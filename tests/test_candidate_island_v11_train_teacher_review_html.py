from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

from tools.audits.generate_candidate_island_v11_train_teacher_review_html import build


CONTRACT = "boundary_acoustic_binary_v12"


def _wav(path: Path, frames: int = 16000) -> str:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_train_teacher_review_selects_one_editable_source_per_video(tmp_path: Path) -> None:
    source_rows = []
    preaudit_rows = []
    ratios = {"a0": 0.1, "a1": 0.4, "b0": 0.9, "b1": 0.3}
    for source_id, video_id in (("a0", "video-a"), ("a1", "video-a"), ("b0", "video-b"), ("b1", "video-b")):
        audio = tmp_path / f"{source_id}.wav"
        sha = _wav(audio)
        source_rows.append(
            {
                "schema": "candidate_island_scorer_v11_train_teacher_source_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "frame_count": 50,
                "frame_hop_s": 0.02,
                "duration_s": 1.0,
                "audio": str(audio),
                "audio_sha256": sha,
            }
        )
        end = int(round(ratios[source_id] * 50))
        preaudit_rows.append(
            {
                "schema": "candidate_island_scorer_v11_omni_preaudit_v2",
                "source_id": source_id,
                "partition": "train",
                "frame_count": 50,
                "audio_sha256": sha,
                "model": "google/gemini-test",
                "overall_confidence": 0.9,
                "overall_reason": f"draft {source_id}",
                "islands": [] if end == 0 else [{"start_frame": 0, "end_frame": end}],
                "unsure_spans": [],
            }
        )
    sources = tmp_path / "sources.jsonl"
    preaudit = tmp_path / "preaudit.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    _write(sources, source_rows)
    _write(preaudit, preaudit_rows)
    _write(excluded, [{"source_id": "a1"}])

    output = tmp_path / "audit"
    summary = build(
        source_manifest=sources,
        preaudit=preaudit,
        exclude_sources=excluded,
        target_inside_ratio=0.35,
        output_dir=output,
    )

    assert summary["source_count"] == 2
    assert summary["video_count"] == 2
    assert summary["selected_source_ids"] == ["a0", "b1"]
    assert summary["teacher_output_used_as_truth"] is False
    assert summary["unselected_source_label_inheritance"] is False
    assert summary["training_manifest_allowed"] is False
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "candidate_island_scorer_v11_train_manual_verdict_v1" in page
    assert "Gemini 可编辑底稿（未确认）" in page
    assert "const initialAnn=" in page
    assert "function convertRange" in page
    assert "转 ${esc(alternate)}" in page
    assert "转 outside_candidate" in page
    assert "range.label==='inside_candidate'?'unsure':'inside_candidate'" in page
    assert 'class="span ${esc(span.label)} timeline-span"' in page
    assert "card.querySelectorAll('.timeline-span')" in page
    assert "黄色 outside_candidate 检查" in page
    assert "能独立于同一轮对白波形安全删除" in page
    assert "不是内部噪声分离器" in page
    assert "不要为了提前清理而牺牲 Scorer 的 inside recall" in page
    assert "draft a0" in page and "draft b1" in page
    manifest = [json.loads(line) for line in (output / "audit_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["video_id"] for row in manifest} == {"video-a", "video-b"}
    assert all((output / row["audio"]).is_file() for row in manifest)
