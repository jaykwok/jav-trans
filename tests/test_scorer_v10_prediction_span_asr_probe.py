from __future__ import annotations

import json
import wave
from array import array
from pathlib import Path

import pytest

from tools.audits.probe_scorer_v10_prediction_spans_with_asr import (
    PROBE_RESULT_SCHEMA,
    attach_probe_results,
    prepare_probe_inputs,
)


def _write_wav(path: Path, *, frame_count: int = 3200) -> None:
    samples = array("h", ((index % 200) - 100 for index in range(frame_count)))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())


def test_prediction_span_asr_probe_crops_exact_unmerged_islands_and_enriches(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "source.wav"
    _write_wav(audio, frame_count=1500)
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "source_id": "train-background",
                "audio": str(audio),
                "partition": "train",
                "row_role": "all_background",
                "category": "background_false_keep",
                "duration_s": 1500 / 16000,
                "prediction_spans": [
                    {
                        "label": "model_speech",
                        "start_frame": 1,
                        "end_frame": 3,
                        "start_s": 0.02,
                        "end_s": 0.06,
                    },
                    {
                        "label": "model_speech",
                        "start_frame": 4,
                        "end_frame": 5,
                        "start_s": 0.08,
                        "end_s": 0.1,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prepared = prepare_probe_inputs(
        selection_path=selection, output_dir=tmp_path / "probe"
    )
    inputs = prepared["probe_inputs"]
    assert len(inputs) == 2
    assert [row["sample_count"] for row in inputs] == [640, 220]
    assert [row["duration_s"] for row in inputs] == [0.04, 0.01375]
    assert [row["clamped_to_source_end"] for row in inputs] == [False, True]
    for row in inputs:
        with wave.open(str(row["audio"]), "rb") as reader:
            assert reader.getframerate() == 16000
            assert reader.getnframes() == row["sample_count"]
    results = [
        {
            "schema": PROBE_RESULT_SCHEMA,
            "probe_id": inputs[0]["probe_id"],
            "raw_text": "待って",
            "text": "待って",
            "normalized_text": "待って",
            "nonempty_text": True,
            "language": "Japanese",
            "error_kind": "",
            "error_detail": "",
            "elapsed_s": 0.1,
        },
        {
            "schema": PROBE_RESULT_SCHEMA,
            "probe_id": inputs[1]["probe_id"],
            "raw_text": "",
            "text": "",
            "normalized_text": "",
            "nonempty_text": False,
            "language": "Japanese",
            "error_kind": "",
            "error_detail": "",
            "elapsed_s": 0.1,
        },
    ]
    enriched = attach_probe_results(
        prepared_rows=prepared["prepared_rows"], results=results
    )
    assert enriched[0]["diagnostic_only"] is True
    assert enriched[0]["training_manifest_allowed"] is False
    assert enriched[0]["asr_probe_summary"] == {
        "span_count": 2,
        "nonempty_text_span_count": 1,
        "error_span_count": 0,
        "texts_in_workflow_order": ["待って", ""],
        "diagnostic_only": True,
        "automatic_label_change_allowed": False,
    }
    assert enriched[0]["prediction_spans"][0]["asr_probe"]["raw_text"] == "待って"
    assert enriched[0]["prediction_spans"][1]["asr_probe"]["nonempty_text"] is False


def test_prediction_span_asr_probe_rejects_foreign_result(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="foreign span identities"):
        attach_probe_results(
            prepared_rows=[],
            results=[
                {
                    "schema": PROBE_RESULT_SCHEMA,
                    "probe_id": "foreign",
                }
            ],
        )


def test_prediction_span_asr_probe_accepts_nonsemantic_probe_role(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    _write_wav(audio, frame_count=1500)
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "source_id": "real-source",
                "audio": str(audio),
                "prediction_spans": [
                    {
                        "label": "asr_probe_candidate",
                        "selection_role": "gemini_outside_complement_pending_asr",
                        "start_frame": 1,
                        "end_frame": 3,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prepared = prepare_probe_inputs(
        selection_path=selection, output_dir=tmp_path / "probe"
    )
    assert prepared["probe_inputs"][0]["probe_id"] == (
        "real-source::asr_probe_candidate::1-3"
    )
