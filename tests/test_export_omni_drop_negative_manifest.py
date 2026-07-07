from __future__ import annotations

import json
from pathlib import Path

from tools.datasets.export_omni_drop_negative_manifest import (
    export_rows,
    source_partition,
)


def test_source_partition_keeps_compiled_heldout_out_of_train() -> None:
    for index in range(500):
        video_id = f"video-{index}"
        import hashlib

        bucket = (
            int(hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        )
        partition = source_partition(video_id, heldout_percent=20)
        assert (bucket < 20) == (partition in {"val", "test"})


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
        heldout_percent=20,
    )

    assert [row["audio_id"] for row in exported] == ["good"]
    assert exported[0]["background_type"] == "breathing"
    assert counts["skip_semantic_speech"] == 1
    assert counts["skip_confidence"] == 1
