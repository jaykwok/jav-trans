from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.fusionvad_ja.extract_speaker_sidecar_embeddings import build_summary


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_extract_speaker_sidecar_energy_mfcc_smoke(tmp_path: Path):
    sample_rate = 16000
    t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, audio, sample_rate)
    bilingual = _write_json(
        tmp_path / "sample.bilingual.json",
        {
            "blocks": [
                {"start": 0.0, "end": 0.8, "ja_text": "あ", "zh_text": "啊", "cue_id": 0},
                {"start": 0.9, "end": 1.7, "ja_text": "い", "zh_text": "咿", "cue_id": 1},
                {"start": 1.8, "end": 1.9, "ja_text": "短", "zh_text": "短", "cue_id": 2},
            ]
        },
    )

    summary = build_summary(
        bilingual_path=bilingual,
        audio_path=audio_path,
        output_dir=tmp_path / "out",
        backend="energy_mfcc",
        min_duration_s=0.3,
        speaker_threshold=0.35,
        max_segments=0,
    )

    assert summary["embedding_count"] == 2
    assert summary["pair_count"] == 1
    assert summary["skipped"]["short"] == 1
    embeddings_path = tmp_path / "out" / "speaker_embeddings.jsonl"
    pairs_path = tmp_path / "out" / "speaker_pairs.jsonl"
    assert embeddings_path.exists()
    assert pairs_path.exists()
    first = json.loads(embeddings_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["segment_id"] == "0"
    assert len(first["embedding"]) == 8
