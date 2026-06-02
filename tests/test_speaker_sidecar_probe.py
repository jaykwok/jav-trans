from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.probe_speaker_sidecar import build_adjacent_speaker_change_rows


def test_adjacent_speaker_change_scores_from_embeddings():
    rows = [
        {"segment_id": "a", "start": 0.0, "end": 1.0, "embedding": [1.0, 0.0]},
        {"segment_id": "b", "start": 1.1, "end": 2.0, "embedding": [0.9, 0.1]},
        {"segment_id": "c", "start": 2.1, "end": 3.0, "embedding": [0.0, 1.0]},
    ]

    pairs = build_adjacent_speaker_change_rows(rows, threshold=0.35)

    assert len(pairs) == 2
    assert pairs[0]["speaker_change"] is False
    assert pairs[1]["speaker_change"] is True
    assert pairs[0]["left_segment_id"] == "a"
    assert pairs[1]["right_segment_id"] == "c"
    assert pairs[0]["gap_s"] == 0.1


def test_speaker_sidecar_cli_writes_pairs(monkeypatch, tmp_path: Path):
    from tools.fusionvad_ja import probe_speaker_sidecar

    segments = tmp_path / "segments.jsonl"
    output = tmp_path / "pairs.jsonl"
    rows = [
        {"segment_id": "a", "start": 0.0, "end": 1.0, "embedding": [1.0, 0.0]},
        {"segment_id": "b", "start": 1.1, "end": 2.0, "embedding": [0.0, 1.0]},
    ]
    segments.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "probe_speaker_sidecar",
            "--segments",
            str(segments),
            "--output",
            str(output),
            "--threshold",
            "0.35",
        ],
    )

    rc = probe_speaker_sidecar.main()

    assert rc == 0
    assert output.exists()
    pair = json.loads(output.read_text(encoding="utf-8").strip())
    assert pair["speaker_change"] is True
    assert output.with_suffix(".summary.json").exists()
