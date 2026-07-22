from __future__ import annotations

import json
from pathlib import Path

from tools.audits.compare_candidate_island_preaudits import compare


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_compare_candidate_island_preaudits_reports_frame_deltas(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"wav")
    manifest = tmp_path / "manifest.jsonl"
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    source = {
        "source_id": "s",
        "frame_count": 10,
        "duration_s": 0.2,
        "audio": str(audio),
        "audio_sha256": "sha",
    }
    _write(manifest, [source])
    _write(
        base,
        [
            {
                **source,
                "prompt_version": "v5",
                "islands": [{"start_frame": 2, "end_frame": 6}],
                "unsure_spans": [{"start_frame": 6, "end_frame": 8}],
            },
            {**source, "source_id": "unused", "prompt_version": "v5", "islands": [], "unsure_spans": []},
        ],
    )
    _write(
        candidate,
        [
            {
                **source,
                "prompt_version": "v6",
                "islands": [{"start_frame": 1, "end_frame": 7}],
                "unsure_spans": [],
            }
        ],
    )
    summary = compare(
        manifest=manifest,
        base=base,
        candidate=candidate,
        output_dir=tmp_path / "out",
    )
    assert summary["source_count"] == 1
    assert summary["base_extra_source_count"] == 1
    assert summary["sources_changed"] == 1
    assert summary["valid_source_count"] == 1
    assert summary["valid_candidate_unsure_ratio"] == 0.0
    assert summary["base_inside_frames"] == 4
    assert summary["candidate_inside_frames"] == 6
    assert summary["changed_frames"] == 3
    page = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert "同一冻结 1 source" in page
    assert "红色=两版逐帧标签不同" in page
    assert 'id="stop"' in page
    assert "document.getElementById('stop').onclick=stop" in page
