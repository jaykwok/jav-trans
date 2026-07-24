from __future__ import annotations

import hashlib
import json
from pathlib import Path

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.boundary.ja.select_candidate_island_scorer_v11_dual_evidence_train import (
    select_dual_evidence_train_sources,
)


CONTRACT = ACOUSTIC_BINARY_V12_CONTRACT.contract_id


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _span(label: str, start: int, end: int) -> dict:
    return {
        "label": label,
        "start_frame": start,
        "end_frame": end,
        "start_s": start * 0.02,
        "end_s": end * 0.02,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    specs = [
        ("video-a-w00", "video-a", 2, 4),
        ("video-a-w01", "video-a", 3, 3),
        ("video-b-w00", "video-b", 0, 6),
    ]
    sources: list[dict] = []
    outside: list[dict] = []
    evidence: list[dict] = []
    for source_id, video_id, inside, outside_count in specs:
        audio_sha = hashlib.sha256(source_id.encode()).hexdigest()
        common = {
            "boundary_serialization_contract_id": CONTRACT,
            "source_id": source_id,
            "video_id": video_id,
            "partition": "train",
            "audio": str(tmp_path / f"{source_id}.wav"),
            "audio_sha256": audio_sha,
            "duration_s": 0.16,
            "frame_count": 8,
            "frame_hop_s": 0.02,
        }
        sources.append(
            {
                **common,
                "schema": "candidate_island_scorer_v11_train_teacher_source_v1",
                "sample_rate": 16000,
                "sample_count": 2560,
                "teacher_only": True,
                "training_manifest_allowed": False,
            }
        )
        outside.append(
            {
                **common,
                "schema": "candidate_island_scorer_v11_real_train_outside_source_v1",
            }
        )
        unsure = 8 - inside - outside_count
        cursor = 0
        inside_spans = (
            [_span("inside_candidate", cursor, cursor + inside)] if inside else []
        )
        cursor += inside
        outside_spans = (
            [_span("outside_candidate", cursor, cursor + outside_count)]
            if outside_count
            else []
        )
        cursor += outside_count
        unsure_spans = [_span("unsure", cursor, cursor + unsure)] if unsure else []
        evidence.append(
            {
                **common,
                "schema": "candidate_island_scorer_v11_dual_evidence_preaudit_v1",
                "islands": inside_spans,
                "safe_outside_spans": outside_spans,
                "unsure_spans": unsure_spans,
                "conflict_spans": [],
                "protect_reasoning": {"reasoning_tokens": 10},
                "remove_reasoning": {"reasoning_tokens": 11},
            }
        )
    source_manifest = tmp_path / "sources.jsonl"
    outside_manifest = tmp_path / "outside.jsonl"
    preaudit = tmp_path / "preaudit.jsonl"
    _write(source_manifest, sources)
    _write(outside_manifest, outside)
    _write(preaudit, evidence)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "candidate_island_scorer_v11_dual_evidence_summary_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "manifest": str(outside_manifest),
                "manifest_sha256": _sha(outside_manifest),
                "labels": str(preaudit),
                "source_ids": [row["source_id"] for row in sources],
                "source_count": 3,
                "failed_closed_count": 0,
                "reasoning_contract_satisfied": True,
            }
        ),
        encoding="utf-8",
    )
    return source_manifest, outside_manifest, summary, preaudit


def test_select_dual_evidence_replaces_exactly_one_source_per_video(
    tmp_path: Path,
) -> None:
    source_manifest, outside, teacher_summary, preaudit = _fixture(tmp_path)
    result = select_dual_evidence_train_sources(
        source_manifest=source_manifest,
        outside_sources=outside,
        teacher_summary=teacher_summary,
        teacher_preaudit=preaudit,
        output_dir=tmp_path / "selected",
    )

    assert result["selected_source_ids"] == ["video-a-w01", "video-b-w00"]
    assert result["remaining_outside_source_ids"] == ["video-a-w00"]
    assert result["source_count"] == result["video_count"] == 2
    assert result["canonical_frame_counts"] == {
        "inside_candidate": 3,
        "outside_candidate": 9,
        "unsure": 4,
    }
    remaining = [
        json.loads(line)
        for line in Path(result["remaining_outside_sources"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["source_id"] for row in remaining] == ["video-a-w00"]
