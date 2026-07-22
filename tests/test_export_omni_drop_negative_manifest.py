from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.datasets.export_omni_drop_negative_manifest import (
    export_rows,
)


def test_export_rows_filters_to_strict_nonsemantic_drop(tmp_path: Path) -> None:
    audio = tmp_path / "drop.wav"
    audio.write_bytes(b"wav")
    labels = tmp_path / "labels.jsonl"
    rows = [
        {
            "candidate_id": "good",
            "audio": str(audio),
            "duration_s": 1.0,
            "label": "definite_drop",
            "training_label_included": True,
            "omni_semantic_speech_detected": False,
            "omni_confidence": 0.95,
            "video_id": "video-a",
            "source_id": "source-a",
            "source_partition": "val",
            "omni_flags": ["breathing"],
        },
        {
            "candidate_id": "semantic",
            "audio": str(audio),
            "duration_s": 1.0,
            "label": "definite_drop",
            "training_label_included": True,
            "omni_semantic_speech_detected": True,
            "omni_confidence": 0.99,
            "video_id": "video-b",
            "source_id": "source-b",
            "source_partition": "test",
        },
        {
            "candidate_id": "weak",
            "audio": str(audio),
            "duration_s": 1.0,
            "label": "definite_drop",
            "training_label_included": True,
            "omni_semantic_speech_detected": False,
            "omni_confidence": 0.70,
            "video_id": "video-c",
            "source_id": "source-c",
            "source_partition": "train",
        },
    ]
    labels.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    exported, counts = export_rows(
        [labels],
        min_confidence=0.90,
        min_duration_s=0.08,
        max_duration_s=12.0,
    )

    assert [row["audio_id"] for row in exported] == ["good"]
    assert exported[0]["source_id"] == "source-a"
    assert exported[0]["source_partition"] == "val"
    assert exported[0]["background_type"] == "breathing"
    assert counts["skip_semantic_speech"] == 1
    assert counts["skip_confidence"] == 1


def test_export_rows_rejects_missing_frozen_source_identity(tmp_path: Path) -> None:
    audio = tmp_path / "drop.wav"
    audio.write_bytes(b"wav")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "candidate_id": "missing-source",
                "audio": str(audio),
                "duration_s": 1.0,
                "label": "definite_drop",
                "training_label_included": True,
                "omni_semantic_speech_detected": False,
                "omni_confidence": 0.95,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen source_id"):
        export_rows(
            [labels],
            min_confidence=0.90,
            min_duration_s=0.08,
            max_duration_s=12.0,
        )
