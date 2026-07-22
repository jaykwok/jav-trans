import json
from collections import Counter
from pathlib import Path

import pytest

from tools.boundary.ja.select_cueqc_v13_hardmix_sources import select


def _write_details(path: Path, audio_root: Path, *, include_source_id: bool) -> None:
    rows = []
    for partition, count in (("train", 17), ("val", 2), ("test", 1)):
        for index in range(count):
            audio_id = f"{partition}-{index}"
            (audio_root / f"{audio_id}.wav").write_bytes(b"wav")
            row = {
                "audio_id": audio_id,
                "source_partition": partition,
                "duration_s": 1.0 + index,
                "actual_speech_segments": [{}, {}, {}],
                "utterance_boundaries": [{}, {}],
                "augmentation": {"overlap_speech": {"enabled": False}},
            }
            if include_source_id:
                row["source_id"] = f"source-{audio_id}"
            rows.append(row)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_hardmix_selection_preserves_frozen_source_partitions(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    details = tmp_path / "details.jsonl"
    _write_details(details, audio_root, include_source_id=True)
    output = tmp_path / "selected.jsonl"

    summary = select(
        details=details,
        audio_root=audio_root,
        output=output,
        count=20,
        seed=3,
    )

    rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
    assert Counter(row["source_partition"] for row in rows) == {
        "train": 17,
        "val": 2,
        "test": 1,
    }
    assert all(row["source_id"] for row in rows)
    assert summary["source_count"] == 20


def test_hardmix_selection_rejects_missing_source_identity(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    details = tmp_path / "details.jsonl"
    _write_details(details, audio_root, include_source_id=False)

    with pytest.raises(ValueError, match="frozen source_id"):
        select(
            details=details,
            audio_root=audio_root,
            output=tmp_path / "selected.jsonl",
            count=20,
            seed=3,
        )
