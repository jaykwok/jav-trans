from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.boundary.ja.build_speech_island_scorer_v10_sparse_train_layout import (
    INPUT_SUMMARY_SCHEMA,
    SUMMARY_SCHEMA,
    build,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_audio(path: Path, pieces: list[np.ndarray]) -> int:
    audio = np.concatenate(pieces).astype(np.float32)
    sf.write(path, audio, 16000, subtype="PCM_16")
    return len(audio)


def _source(
    *,
    source_id: str,
    audio: Path,
    partition: str,
    row_role: str,
    spans: list[dict],
    core_ids: list[str],
    background_id: str = "",
) -> dict:
    sample_count = int(sf.info(audio).frames)
    background_ids = sorted(
        {
            str(span["background_id"])
            for span in spans
            if span.get("background_id")
        }
    )
    if background_id:
        background_ids = [background_id]
    return {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": "speech_scorer_canonical_frames_v1",
        "source_id": source_id,
        "audio": str(audio),
        "row_role": row_role,
        "partition": partition,
        "core_ids": core_ids,
        "background_id": background_id,
        "background_source_ids": background_ids,
        "background_source_video_ids": [f"video-{partition}-{source_id}"],
        "sample_rate": 16000,
        "sample_count": sample_count,
        "duration_s": sample_count / 16000,
        "input_distribution": "full_source_windows",
        "canonical_spans": spans,
        "additive_overlay": None,
    }


def _full_span(sample_count: int, label: str, identity: str) -> list[dict]:
    span = {
        "start_sample": 0,
        "end_sample": sample_count,
        "label": label,
        "label_source": "fixture",
    }
    span["core_id" if label == "speech" else "background_id"] = identity
    return [span]


def test_sparse_layout_changes_only_selected_train_audio(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    lengths = [3200, 1600, 2400, 1600, 4000]
    pieces = [
        rng.normal(0.0, scale, length).astype(np.float32)
        for scale, length in zip((0.12, 0.04, 0.05, 0.03, 0.10), lengths, strict=True)
    ]
    train_audio = tmp_path / "train.wav"
    train_count = _write_audio(train_audio, pieces)
    cursor = 0
    train_spans = []
    labels = ("speech", "background", "background", "background", "speech")
    for index, (label, length) in enumerate(zip(labels, lengths, strict=True)):
        span = {
            "start_sample": cursor,
            "end_sample": cursor + length,
            "label": label,
            "label_source": "teacher_fixture" if label == "speech" else "negative_fixture",
        }
        span["core_id" if label == "speech" else "background_id"] = (
            f"core-{index}" if label == "speech" else f"bg-{index}"
        )
        train_spans.append(span)
        cursor += length
    assert cursor == train_count

    sources = [
        _source(
            source_id="train-candidate",
            audio=train_audio,
            partition="train",
            row_role="speech",
            spans=train_spans,
            core_ids=["core-0", "core-4"],
        )
    ]
    for partition in ("train", "val", "test"):
        background_audio = tmp_path / f"{partition}-background.wav"
        background_count = _write_audio(
            background_audio, [rng.normal(0.0, 0.05, 3200).astype(np.float32)]
        )
        sources.append(
            _source(
                source_id=f"{partition}-background",
                audio=background_audio,
                partition=partition,
                row_role="all_background",
                spans=_full_span(background_count, "background", f"all-bg-{partition}"),
                core_ids=[],
                background_id=f"all-bg-{partition}",
            )
        )
        if partition != "train":
            speech_audio = tmp_path / f"{partition}-speech.wav"
            speech_count = _write_audio(
                speech_audio, [rng.normal(0.0, 0.10, 4000).astype(np.float32)]
            )
            sources.append(
                _source(
                    source_id=f"{partition}-speech",
                    audio=speech_audio,
                    partition=partition,
                    row_role="speech",
                    spans=_full_span(speech_count, "speech", f"core-{partition}"),
                    core_ids=[f"core-{partition}"],
                )
            )

    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sources),
        encoding="utf-8",
    )
    r5_summary = tmp_path / "r5-summary.json"
    r5_summary.write_text(
        json.dumps(
            {
                "schema": INPUT_SUMMARY_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "canonical_sources": str(canonical),
                "canonical_sources_sha256": _sha256(canonical),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = build(
        argparse.Namespace(
            r5_summary=str(r5_summary),
            output_dir=str(output),
            source_count=1,
            seed=7,
            target_db=[-8.0, 3.0],
        )
    )

    assert result["schema"] == SUMMARY_SCHEMA
    assert result["selected_source_count"] == 1
    assert result["selected_core_count"] == 2
    assert result["source_identity_changed"] is False
    assert result["core_identity_changed"] is False
    assert result["partition_identity_changed"] is False
    assert result["heldout_audio_identity_changed"] is False
    assert result["max_core_use_count"] == 1

    rebuilt = [
        json.loads(line)
        for line in (output / "canonical_sources.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["source_id"]: row for row in rebuilt}
    changed = by_id["train-candidate"]
    assert changed["core_ids"] == ["core-0", "core-4"]
    assert changed["partition"] == "train"
    assert changed["sample_count"] == train_count
    assert [span["label"] for span in changed["canonical_spans"]] == [
        "background",
        "speech",
        "background",
        "speech",
        "background",
    ]
    targets = changed["training_distribution_reconstruction"]["speech_targets"]
    assert [target["target_speech_to_adjacent_db"] for target in targets] == [-8.0, 3.0]
    assert all(
        abs(target["achieved_speech_to_adjacent_db"] - target["target_speech_to_adjacent_db"])
        <= 0.05
        for target in targets
    )
    for source in sources:
        if source["source_id"] != "train-candidate":
            assert by_id[source["source_id"]] == source

    changed_manifest = json.loads((output / "changed_audio_manifest.json").read_text())
    assert [row["audio_id"] for row in changed_manifest] == ["train-candidate"]
    assert (output / "changed_feature_cache_labels.jsonl").read_text().count("\n") == 1
