from __future__ import annotations

import json
from pathlib import Path

from tools.boundary.export_cueqc_seed_drop_background_spans import (
    export_cueqc_seed_drop_background_spans,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _seed(sample_id: str) -> dict:
    return {
        "schema": "speech_boundary_hard_negative_candidate_from_cueqc_v1",
        "candidate_id": f"seed-{sample_id}",
        "sample_id": sample_id,
        "source_label_paths": ["agents/audits/test/cueqc_cluster_labels.jsonl"],
        "candidate_route": "speech_boundary_frame_negative_candidate",
        "source_audio_path": "agents/temp/test/audio.wav",
        "start": 0.0,
        "end": 0.5,
    }


def _candidate(
    sample_id: str,
    *,
    start: float,
    end: float,
    text: str = "...",
    char_count: int = 0,
    chunk_index: int = 0,
) -> dict:
    compact_text = "" if char_count <= 0 else text
    return {
        "schema": "cueqc_candidate_v4",
        "sample_id": sample_id,
        "video_id": "AAA",
        "audio_id": "AAA.audio",
        "chunk_index": chunk_index,
        "start": start,
        "end": end,
        "duration_s": end - start,
        "source_audio_path": "agents/temp/test/audio.wav",
        "compact_text": compact_text,
        "text": text,
        "text_features": {"char_count": char_count},
    }


def test_export_cueqc_seed_drop_background_spans_merges_empty_seeded_regions(tmp_path: Path):
    seeds = tmp_path / "seed.jsonl"
    all_candidates = tmp_path / "all.jsonl"
    _write_jsonl(seeds, [_seed("seed-1")])
    _write_jsonl(
        all_candidates,
        [
            _candidate("speech-before", start=0.0, end=1.0, text="こんにちは", char_count=5, chunk_index=0),
            _candidate("empty-a", start=1.05, end=2.0, chunk_index=1),
            _candidate("seed-1", start=2.05, end=3.0, chunk_index=2),
            _candidate("empty-b", start=3.1, end=4.0, chunk_index=3),
            _candidate("speech-after", start=4.2, end=5.0, text="はい", char_count=2, chunk_index=4),
            _candidate("empty-no-seed", start=8.0, end=10.0, chunk_index=5),
        ],
    )

    summary = export_cueqc_seed_drop_background_spans(
        seed_candidates_path=seeds,
        all_candidates_path=all_candidates,
        output_dir=tmp_path / "out",
        max_gap_s=0.2,
        guard_s=0.1,
        min_duration_s=1.0,
    )

    assert summary["counts"]["exported_candidates"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "cueqc_seed_drop_background_span_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["schema"] == "speech_boundary_hard_negative_candidate_from_cueqc_v1"
    assert rows[0]["candidate_route"] == "speech_boundary_frame_negative_candidate"
    assert rows[0]["start"] == 1.1
    assert rows[0]["end"] == 4.0
    assert rows[0]["seed_sample_ids"] == ["seed-1"]
    assert rows[0]["empty_chunk_count"] == 3


def test_export_cueqc_seed_drop_background_spans_rejects_text_overlap(tmp_path: Path):
    seeds = tmp_path / "seed.jsonl"
    all_candidates = tmp_path / "all.jsonl"
    _write_jsonl(seeds, [_seed("seed-1")])
    _write_jsonl(
        all_candidates,
        [
            _candidate("seed-1", start=1.0, end=2.0, chunk_index=1),
            _candidate("speech-overlap", start=1.5, end=1.8, text="だめ", char_count=2, chunk_index=2),
        ],
    )

    summary = export_cueqc_seed_drop_background_spans(
        seed_candidates_path=seeds,
        all_candidates_path=all_candidates,
        output_dir=tmp_path / "out",
        min_duration_s=0.1,
        require_nonempty=False,
    )

    assert summary["counts"]["exported_candidates"] == 0
    skipped = json.loads((tmp_path / "out" / "cueqc_seed_drop_background_span_skipped.json").read_text(encoding="utf-8"))
    assert any(row["reason"] == "overlaps_nonempty_text_candidate" for row in skipped)
