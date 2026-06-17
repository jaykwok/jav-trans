from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.boundary.ja.build_positive_anchor_replay import build_positive_anchor_replay


def _write_audio(path: Path, duration_s: float = 0.2) -> None:
    samples = np.zeros(max(1, int(round(16000 * duration_s))), dtype=np.float32)
    sf.write(path, samples, 16000)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def test_build_positive_anchor_replay_uses_weighted_sources(tmp_path: Path):
    manifests: dict[str, Path] = {}
    for group in ("anime_nsfw", "anime_sfw", "galgame"):
        rows = []
        for index in range(12):
            audio = tmp_path / f"{group}_{index}.wav"
            _write_audio(audio)
            rows.append(
                {
                    "audio": str(audio),
                    "audio_id": f"{group}-{index}",
                    "duration_s": 0.2,
                    "source": group,
                    "text": "あ",
                }
            )
        manifest = tmp_path / f"{group}.json"
        _write_manifest(manifest, rows)
        manifests[group] = manifest

    summary = build_positive_anchor_replay(
        source_specs=[
            f"anime_nsfw=55={manifests['anime_nsfw']}",
            f"anime_sfw=20={manifests['anime_sfw']}",
            f"galgame=25={manifests['galgame']}",
        ],
        output_dir=tmp_path / "anchors",
        count=20,
        seed=7,
    )

    assert summary["count"] == 20
    assert summary["training_examples"] == 20
    assert summary["source_group_counts"] == {
        "anime_nsfw": 11,
        "anime_sfw": 4,
        "galgame": 5,
    }
    assert summary["speech_frame_ratio"] == 1.0

    labels = [
        json.loads(line)
        for line in (tmp_path / "anchors" / "positive_anchor_labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["label_quality"] for row in labels} == {"supervised"}
    assert all(sum(row["speech_frames"]) == len(row["speech_frames"]) for row in labels)
    manifest_rows = json.loads((tmp_path / "anchors" / "positive_anchor_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest_rows) == 20
