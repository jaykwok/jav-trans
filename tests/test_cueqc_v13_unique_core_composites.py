from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools.boundary.ja import build_cueqc_v13_unique_core_composites as composites


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_builder_emits_every_partition_quota_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    quotas = {"train": 17, "val": 2, "test": 1}
    cores: list[dict] = []
    for partition, sample_count in quotas.items():
        for index in range(sample_count * 2):
            cores.append(
                {
                    "core_id": f"{partition}-core-{index}",
                    "source_id": f"{partition}-source-{index}",
                    "source_partition": partition,
                    "audio": str(tmp_path / f"{partition}-core-{index}.wav"),
                    "text": "semantic",
                }
            )
    semantic_cores = tmp_path / "cores.jsonl"
    _write_jsonl(semantic_cores, cores)

    negatives: list[dict] = []
    for partition in quotas:
        audio = tmp_path / f"{partition}-negative.wav"
        audio.write_bytes(b"placeholder")
        negatives.append(
            {
                "audio": str(audio),
                "audio_id": f"{partition}-negative",
                "source_id": f"{partition}-negative-source",
                "source_partition": partition,
                "background_type": "breath",
                "duration_s": 0.02,
            }
        )
    negative_manifest = tmp_path / "negatives.jsonl"
    _write_jsonl(negative_manifest, negatives)

    gap_durations = tmp_path / "gaps.json"
    gap_durations.write_text(json.dumps({"durations_s": [0.01, 0.02, 0.03]}))
    snr_reference = tmp_path / "snr.jsonl"
    _write_jsonl(
        snr_reference,
        [
            {"background_mix": {"enabled": True, "snr_db": value}}
            for value in (0, 5, 10, 15, 20)
        ],
    )

    monkeypatch.setattr(
        composites,
        "_load_audio",
        lambda _path: np.full(320, 0.05, dtype=np.float32),
    )

    def fake_write(path: str, *_args, **_kwargs) -> None:
        Path(path).write_bytes(b"wav")

    monkeypatch.setattr(composites.sf, "write", fake_write)
    output_dir = tmp_path / "output"
    summary = composites.build(
        argparse.Namespace(
            semantic_cores=str(semantic_cores),
            negative_manifest=str(negative_manifest),
            gap_durations=str(gap_durations),
            snr_reference=str(snr_reference),
            output_dir=str(output_dir),
            sample_count=20,
            seed=7,
        )
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "source_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 20
    assert Counter(row["source_partition"] for row in rows) == quotas
    assert summary["source_count"] == 20
    assert summary["unique_semantic_core_count"] == 40
    assert summary["max_core_use_count"] == 1
    assert summary["additive_overlay_count"] == 10
