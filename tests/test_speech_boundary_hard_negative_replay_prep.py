from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from boundary.ja import build_negative_record, build_supervised_record, write_jsonl
from tools.boundary.prepare_speech_boundary_hard_negative_replay import (
    prepare_hard_negative_replay,
)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_audio(path: Path, duration_s: float = 0.2) -> None:
    samples = np.zeros(max(1, int(round(16000 * duration_s))), dtype=np.float32)
    sf.write(path, samples, 16000)


def test_hard_negative_replay_prep_blocks_negative_only_training(tmp_path: Path):
    audio = tmp_path / "neg.wav"
    _write_audio(audio)
    labels = tmp_path / "negative_labels.jsonl"
    manifest = tmp_path / "negative_manifest.json"
    write_jsonl(
        labels,
        [
            build_negative_record(
                audio_id="neg",
                source="cueqc_drop_frame_negative",
                duration_s=0.2,
                frame_hop_s=0.02,
            )
        ],
    )
    _write_manifest(manifest, [{"audio_id": "neg", "audio": str(audio), "duration_s": 0.2}])

    summary = prepare_hard_negative_replay(
        negative_labels=labels,
        negative_manifest=manifest,
        output_dir=tmp_path / "prep",
        min_negatives=1,
        min_positive_anchors=1,
    )

    assert summary["negative_source"]["trainable_examples"] == 1
    assert summary["gate"]["formal_training_ready"] is False
    assert summary["gate"]["checks"]["positive_anchor_examples_ge_min"] is False
    assert summary["gate"]["checks"]["negative_share_le_max"] is False
    assert (tmp_path / "prep" / "build_negative_feature_cache.ps1").exists()
    assert (tmp_path / "prep" / "tiny_negative_plumbing_smoke.ps1").exists()


def test_hard_negative_replay_prep_passes_with_anchor_examples(tmp_path: Path):
    neg_audio = tmp_path / "neg.wav"
    pos_audio = tmp_path / "pos.wav"
    _write_audio(neg_audio)
    _write_audio(pos_audio)
    neg_labels = tmp_path / "negative_labels.jsonl"
    neg_manifest = tmp_path / "negative_manifest.json"
    pos_labels = tmp_path / "positive_labels.jsonl"
    pos_manifest = tmp_path / "positive_manifest.json"
    write_jsonl(
        neg_labels,
        [
            build_negative_record(
                audio_id="neg",
                source="cueqc_drop_frame_negative",
                duration_s=0.2,
                frame_hop_s=0.02,
            )
        ],
    )
    write_jsonl(
        pos_labels,
        [
            build_supervised_record(
                audio_id="pos",
                source="anchor_positive",
                duration_s=0.2,
                speech_segments=[{"start": 0.02, "end": 0.18}],
                frame_hop_s=0.02,
            )
            for _ in range(3)
        ],
    )
    _write_manifest(neg_manifest, [{"audio_id": "neg", "audio": str(neg_audio), "duration_s": 0.2}])
    _write_manifest(pos_manifest, [{"audio_id": "pos", "audio": str(pos_audio), "duration_s": 0.2}])

    summary = prepare_hard_negative_replay(
        negative_labels=neg_labels,
        negative_manifest=neg_manifest,
        output_dir=tmp_path / "prep",
        anchor_labels=[pos_labels],
        anchor_manifests=[pos_manifest],
        min_negatives=1,
        min_positive_anchors=3,
        max_negative_share=0.5,
    )

    assert summary["gate"]["formal_training_ready"] is True
    assert summary["gate"]["checks"]["positive_anchor_has_speech_frames"] is True
    assert summary["gate"]["negative_share"] == 0.25
    assert summary["mixed_source"]["emitted"] is True
    assert summary["mixed_source"]["records"] == 4
    assert summary["mixed_source"]["trainable_examples"] == 4
    assert (tmp_path / "prep" / "speech_boundary_mixed_hard_negative_anchor_labels.jsonl").exists()
    assert (tmp_path / "prep" / "speech_boundary_mixed_hard_negative_anchor_manifest.json").exists()
    assert (tmp_path / "prep" / "build_mixed_feature_cache.ps1").exists()
    assert (tmp_path / "prep" / "tiny_mixed_plumbing_train.ps1").exists()
    assert not (tmp_path / "prep" / "train_mixed_feature_scorer.ps1").exists()
