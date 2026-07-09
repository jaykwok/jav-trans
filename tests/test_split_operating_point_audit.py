from __future__ import annotations

import json
from pathlib import Path

from tools.audits.generate_split_operating_point_audit_html import build_audit


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_wav_header(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF$\x00\x00\x00WAVEfmt ")
    return path


def test_build_audit_writes_manifest_summary_and_audio_refs(tmp_path: Path) -> None:
    reexport = tmp_path / "reexport"
    feature_dir = reexport / "features" / "vid-w00"
    audio_path = _write_wav_header(reexport / "audio" / "vid-w00.wav")
    _write_jsonl(
        reexport / "source_windows.jsonl",
        [
            {
                "window_id": "vid-w00",
                "audio_wav": str(audio_path),
                "source_video": "source.mp4",
                "source_start_s": 100.0,
            }
        ],
    )
    _write_jsonl(
        feature_dir / "pre_asr_candidates.jsonl",
        [
            {
                "candidate_id": "preasr-vid-w00-chunk00000",
                "audio_id": "vid-w00",
                "chunk_index": 0,
                "start": 0.0,
                "end": 20.0,
                "duration_s": 20.0,
            }
        ],
    )
    _write_jsonl(
        feature_dir / "semantic_split_features.jsonl",
        [
            {"time_s": 4.0, "core_start": 0.0, "core_end": 20.0, "label": "continue", "p_cut": 0.10, "accepted": False},
            {"time_s": 10.0, "core_start": 0.0, "core_end": 20.0, "label": "cut", "p_cut": 0.72, "accepted": False},
            {"time_s": 18.9, "core_start": 0.0, "core_end": 20.0, "label": "cut", "p_cut": 0.30, "accepted": False},
        ],
    )
    adaptive_summary = tmp_path / "adaptive" / "summary.json"
    adaptive_summary.parent.mkdir(parents=True, exist_ok=True)
    adaptive_summary.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "policy": {
                            "abs_floor": 0.50,
                            "percentile_floor": 0.80,
                            "z_floor": 99.0,
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = build_audit(
        reexport_dir=reexport,
        adaptive_summary=adaptive_summary,
        output_dir=tmp_path / "audit",
        fixed_threshold=0.70,
        adaptive_policy_index=0,
        long_chunk_min_s=15.0,
        min_chunk_after_split_s=1.2,
        limit=5,
        cut_audio=False,
    )

    manifest_rows = [
        json.loads(line)
        for line in (tmp_path / "audit" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    index_html = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")
    stored = json.loads((tmp_path / "audit" / "summary.json").read_text(encoding="utf-8"))

    assert summary["item_count"] == 1
    assert stored["schema"] == "split_operating_point_audit_summary_v1"
    assert manifest_rows[0]["candidate_id"] == "preasr-vid-w00-chunk00000"
    assert manifest_rows[0]["fixed_cut_times_s"] == [10.0]
    assert manifest_rows[0]["adaptive_cut_times_s"] == [10.0]
    assert manifest_rows[0]["clips"]["cut_context"].endswith("_cut_context.wav")
    assert "preasr-vid-w00-chunk00000" in index_html
