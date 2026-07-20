from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.audits.build_scorer_v10_canonical_rescore_manifest import build_manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _source(source_id: str, partition: str, *, speech: bool) -> dict:
    identity = f"bg-{source_id}"
    return {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": "speech_scorer_canonical_frames_v1",
        "source_id": source_id,
        "audio": f"audio/{source_id}.wav",
        "row_role": "speech" if speech else "all_background",
        "partition": partition,
        "core_ids": [f"core-{source_id}"] if speech else [],
        "background_id": "" if speech else identity,
        "background_source_ids": [identity],
        "background_source_video_ids": [f"video-{identity}"],
        "sample_rate": 16000,
        "sample_count": 1600,
        "duration_s": 0.1,
        "input_distribution": "full_source_windows",
        "canonical_spans": [
            {
                "start_sample": 0,
                "end_sample": 1600,
                "label": "speech" if speech else "background",
                **(
                    {"core_id": f"core-{source_id}"}
                    if speech
                    else {"background_id": identity}
                ),
            }
        ],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sources = [
        _source(f"{partition}-speech", partition, speech=True)
        for partition in ("train", "val", "test")
    ] + [
        _source(f"{partition}-background", partition, speech=False)
        for partition in ("train", "val", "test")
    ]
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(canonical, sources)
    audio_rows = [
        {
            "audio_id": row["source_id"],
            "audio": row["audio"],
            "partition": row["partition"],
            "row_role": row["row_role"],
        }
        for row in sources
    ]
    audio_manifest = tmp_path / "audio_manifest.json"
    audio_manifest.write_text(json.dumps(audio_rows), encoding="utf-8")
    original_audio_manifest = tmp_path / "original_audio_manifest.json"
    original_audio_manifest.write_bytes(audio_manifest.read_bytes())

    feature_rows = []
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for row in sources:
        feature_path = feature_dir / f"{row['source_id']}.npz"
        np.savez(
            feature_path,
            ptm=np.zeros((5, 2048), dtype=np.float32),
            mfcc=np.zeros((5, 40), dtype=np.float32),
        )
        feature_rows.append(
            {
                "audio_id": row["source_id"],
                "audio_path": row["audio"],
                "cache_key": f"cache-{row['source_id']}",
                "feature_path": str(feature_path),
                "frame_count": 5,
                "frame_hop_s": 0.02,
                "ptm": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
                "ptm_dim": 2048,
                "mfcc_dim": 40,
            }
        )
    feature_manifest = tmp_path / "feature_manifest.jsonl"
    _write_jsonl(feature_manifest, feature_rows)
    feature_summary = tmp_path / "feature_summary.json"
    feature_summary.write_text(
        json.dumps(
            {
                "errors": 0,
                "skipped": 0,
                "ptm": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
                "source_manifest": str(original_audio_manifest),
            }
        ),
        encoding="utf-8",
    )
    return canonical, audio_manifest, feature_manifest, feature_summary


def test_build_canonical_rescore_manifest_is_diagnostic_only(tmp_path: Path) -> None:
    canonical, audio_manifest, feature_manifest, feature_summary = _fixture(tmp_path)
    output = tmp_path / "rescore"
    result = build_manifest(
        canonical_sources=canonical,
        audio_manifest=audio_manifest,
        feature_manifest=feature_manifest,
        feature_summary=feature_summary,
        output_dir=output,
    )
    rows = [
        json.loads(line)
        for line in (output / "diagnostic_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["row_count"] == 6
    assert result["audio_manifest_byte_identical"] is True
    assert result["checkpoint_rescore_allowed"] is True
    assert result["training_manifest_allowed"] is False
    assert {row["schema"] for row in rows} == {
        "speech_scorer_v10_binary_diagnostic_row_v1"
    }
    assert all(row["diagnostic_only"] is True for row in rows)
    assert len(list((output / "labels").glob("*.npz"))) == 6


def test_build_canonical_rescore_manifest_rejects_audio_manifest_change(
    tmp_path: Path,
) -> None:
    canonical, audio_manifest, feature_manifest, feature_summary = _fixture(tmp_path)
    audio_manifest.write_text(
        audio_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identical audio manifests"):
        build_manifest(
            canonical_sources=canonical,
            audio_manifest=audio_manifest,
            feature_manifest=feature_manifest,
            feature_summary=feature_summary,
            output_dir=tmp_path / "rejected",
        )
