from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.boundary.prepare_cueqc_drop_hard_negative_sources import (
    prepare_cueqc_drop_hard_negative_sources,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(sample_id: str, *, route: str, start: float, end: float) -> dict:
    return {
        "schema": "speech_boundary_hard_negative_candidate_from_cueqc_v1",
        "candidate_id": f"cueqc-drop-hardcase-{sample_id}",
        "sample_id": sample_id,
        "audit_id": sample_id,
        "video_id": "AAA",
        "video_label": "sample",
        "chunk_index": 1,
        "start": start,
        "end": end,
        "duration_s": end - start,
        "text": "...",
        "text_bucket": "nonlexical",
        "reason_tags": ["vocalization"],
        "display_prob_drop_mean": 0.91,
        "candidate_route": route,
        "route_reason": "test route",
        "source_label_paths": [],
    }


def test_prepare_cueqc_drop_hard_negative_sources_materializes_negative_audio(tmp_path: Path):
    source_audio = tmp_path / "source.wav"
    samples = np.linspace(-0.25, 0.25, 16000, dtype=np.float32)
    sf.write(source_audio, samples, 16000)

    candidates_path = tmp_path / "cueqc_confirmed_drop_candidates.jsonl"
    _write_jsonl(
        candidates_path,
        [
            _candidate(
                "cueqc-AAA-chunk00001",
                route="speech_boundary_frame_negative_candidate",
                start=0.10,
                end=0.30,
            ),
            _candidate(
                "cueqc-AAA-chunk00002",
                route="speech_boundary_frame_negative_candidate",
                start=0.50,
                end=0.90,
            ),
        ],
    )
    audit_items = tmp_path / "cueqc_prediction_audit_items.jsonl"
    _write_jsonl(
        audit_items,
        [
            {
                "sample_id": "cueqc-AAA-chunk00001",
                "audit_id": "cueqc-AAA-chunk00001",
                "context_start": 0.0,
                "context_end": 0.5,
                "media": {"audio_path": str(source_audio)},
                "chunk_subtitle_cues": [],
                "context_subtitle_cues": [],
                "aligned_segments": [],
            },
            {
                "sample_id": "cueqc-AAA-chunk00002",
                "audit_id": "cueqc-AAA-chunk00002",
                "context_start": 0.4,
                "context_end": 1.0,
                "media": {"audio_path": str(source_audio)},
                "chunk_subtitle_cues": [{"start": 0.5, "end": 0.9, "text": "..."}],
                "context_subtitle_cues": [],
                "aligned_segments": [],
            },
        ],
    )

    summary = prepare_cueqc_drop_hard_negative_sources(
        candidates_path=candidates_path,
        output_dir=tmp_path / "out",
        audit_item_paths=[audit_items],
    )

    assert summary["counts"]["input_candidates"] == 2
    assert summary["counts"]["frame_negative_materialized"] == 2
    assert summary["counts"]["speech_boundary_training_examples"] == 2
    assert summary["speech_boundary_hard_negative"]["direct_boundary_refiner_dataset_emitted"] is False

    labels = [
        json.loads(line)
        for line in (tmp_path / "out" / "speech_boundary_negative_labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert labels[0]["label_quality"] == "negative"
    assert sum(labels[0]["speech_frames"]) == 0
    assert labels[0]["boundary_metadata"]["source_start_s"] == 0.1
    manifest = json.loads((tmp_path / "out" / "speech_boundary_negative_manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest[0]["audio"]).exists()
    audio, sample_rate = sf.read(manifest[0]["audio"], dtype="float32")
    assert sample_rate == 16000
    assert 3000 <= len(audio) <= 3400

    output_names = {path.name for path in (tmp_path / "out").iterdir()}
    assert "speech_boundary_negative_labels.jsonl" in output_names
    assert "speech_boundary_negative_manifest.json" in output_names


def test_prepare_cueqc_drop_hard_negative_sources_rejects_unsupported_route(tmp_path: Path):
    candidates_path = tmp_path / "cueqc_confirmed_drop_candidates.jsonl"
    _write_jsonl(
        candidates_path,
        [
            _candidate(
                "cueqc-AAA-chunk00001",
                route="unsupported_legacy_route",
                start=0.10,
                end=0.30,
            )
        ],
    )

    with pytest.raises(ValueError, match="Regenerate candidates"):
        prepare_cueqc_drop_hard_negative_sources(
            candidates_path=candidates_path,
            output_dir=tmp_path / "out",
            audit_item_paths=[],
        )
