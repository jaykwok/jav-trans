from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.fusionvad_ja.materialize_alignment_failure_audio import main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_materialize_alignment_failure_audio_slices_source_wav(tmp_path):
    source = tmp_path / "source.wav"
    samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    sf.write(str(source), samples, 16000)
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "case__sample__chunk0001__vad_coarse_alignment",
                "source_audio_path": str(source),
                "start": 0.25,
                "end": 0.5,
                "duration_s": 0.25,
                "review_type": "review_coarse_timing",
                "failure_bucket": "vad_coarse_alignment",
                "display_text": "ああ",
            }
        ],
    )
    output_dir = tmp_path / "out"

    assert main(["--manifest", str(manifest), "--output-dir", str(output_dir), "--pad-s", "0.1"]) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "alignment_failure_audio_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "alignment_failure_audio_summary.json").read_text(encoding="utf-8"))
    audio_path = Path(rows[0]["audio"])
    if not audio_path.is_absolute():
        audio_path = Path.cwd() / audio_path
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")

    assert summary["materialized_rows"] == 1
    assert summary["errors"] == 0
    assert sample_rate == 16000
    assert 0.44 <= len(audio) / sample_rate <= 0.46
    assert rows[0]["source_start_s"] == 0.15
    assert rows[0]["chunk_start_s"] == 0.1
    assert rows[0]["chunk_end_s"] == 0.35


def test_materialize_alignment_failure_audio_records_missing_audio(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "missing",
                "source_audio_path": str(tmp_path / "missing.wav"),
                "start": 0.0,
                "end": 1.0,
            }
        ],
    )
    output_dir = tmp_path / "out"

    assert main(["--manifest", str(manifest), "--output-dir", str(output_dir)]) == 0

    summary = json.loads((output_dir / "alignment_failure_audio_summary.json").read_text(encoding="utf-8"))
    errors = json.loads((output_dir / "alignment_failure_audio_errors.json").read_text(encoding="utf-8"))

    assert summary["materialized_rows"] == 0
    assert summary["errors"] == 1
    assert errors[0]["sample_id"] == "missing"
