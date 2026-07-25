from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from boundary.ja.vocal_envelope_v12 import (
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_FEATURE_MANIFEST_SCHEMA,
)
from tools.boundary.ja.compile_vocal_envelope_scorer_v12_features import (
    _array_sha,
    _binding_sha,
    compile_features,
)
from tools.boundary.ja.extract_vocal_envelope_scorer_v12_raw_features import (
    _safe_id,
)
from tools.boundary.ja.rebind_vocal_envelope_scorer_v12_raw_features import (
    rebind,
)


CONTRACT_ID = "boundary_acoustic_binary_v12"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict], list[dict]]:
    canonical = tmp_path / "canonical_sources.jsonl"
    source_rows: list[dict] = []
    labels_by_partition = {
        "train": [0, 0, 1, 1, -100, 1],
        "val": [1, 1, 1, 1, 1, 1],
        "test": [0, 0, 0, 0, 0, 0],
    }
    for partition, labels in labels_by_partition.items():
        audio = tmp_path / f"{partition}.wav"
        audio.write_bytes(f"audio-{partition}".encode())
        spans: list[dict] = []
        start = 0
        current = labels[0]
        for index in range(1, len(labels) + 1):
            if index < len(labels) and labels[index] == current:
                continue
            spans.append(
                {
                    "label": {
                        0: "non_vocal_candidate",
                        1: "vocal_candidate",
                        -100: "unsure",
                    }[current],
                    "start_frame": start,
                    "end_frame": index,
                }
            )
            if index < len(labels):
                start, current = index, labels[index]
        source_rows.append(
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_CANONICAL_SOURCE_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "canonical_label_schema": VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
                "source_id": f"source::{partition}",
                "video_id": f"video-{partition}",
                "partition": partition,
                "core_ids": [f"core-{partition}"],
                "source_kind": "real_full_source",
                "synthetic_composite": False,
                "audio": str(audio),
                "audio_sha256": _sha256(audio),
                "duration_s": len(labels) * 0.02,
                "frame_count": len(labels),
                "frame_hop_s": 0.02,
                "canonical_spans": spans,
                "training_manifest_allowed": True,
            }
        )
    _write_jsonl(canonical, source_rows)
    canonical_sha = _sha256(canonical)

    raw_manifest = tmp_path / "raw_feature_manifest.jsonl"
    feature_rows: list[dict] = []
    for source in source_rows:
        frame_count = int(source["frame_count"])
        ptm = np.arange(frame_count * 2048, dtype=np.float32).reshape(frame_count, 2048)
        mfcc = np.arange(frame_count * 40, dtype=np.float32).reshape(frame_count, 40)
        feature_path = tmp_path / f"{source['partition']}.npz"
        np.savez(feature_path, ptm=ptm, mfcc=mfcc)
        ptm_sha, mfcc_sha = _array_sha(ptm), _array_sha(mfcc)
        feature_rows.append(
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_FEATURE_MANIFEST_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "canonical_sources_sha256": canonical_sha,
                "source_id": source["source_id"],
                "partition": source["partition"],
                "core_ids": source["core_ids"],
                "audio": source["audio"],
                "audio_sha256": source["audio_sha256"],
                "frame_count": frame_count,
                "frame_hop_s": 0.02,
                "ptm_dim": 2048,
                "mfcc_dim": 40,
                "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "feature_extractor_schema": VOCAL_ENVELOPE_SCORER_V12_FEATURE_EXTRACTOR_SCHEMA,
                "feature_path": str(feature_path),
                "feature_sha256": _sha256(feature_path),
                "ptm_sha256": ptm_sha,
                "mfcc_sha256": mfcc_sha,
                "source_audio_frame_binding_sha256": _binding_sha(
                    source_id=source["source_id"],
                    audio_sha256=source["audio_sha256"],
                    frame_count=frame_count,
                    ptm_sha256=ptm_sha,
                    mfcc_sha256=mfcc_sha,
                ),
            }
        )
    _write_jsonl(raw_manifest, feature_rows)
    return canonical, raw_manifest, source_rows, feature_rows


def test_v12_feature_compiler_binds_arrays_and_reports_real_controls(
    tmp_path: Path,
) -> None:
    canonical, raw_manifest, _, _ = _fixture(tmp_path)
    output = tmp_path / "compiled"
    summary = compile_features(
        canonical=canonical,
        raw_manifest=raw_manifest,
        output_dir=output,
    )
    assert summary["status"] == "approved_for_training"
    assert summary["heldout_strata"]["val:all_vocal"] == 1
    assert summary["heldout_strata"]["test:all_nonvocal"] == 1
    assert summary["heldout_strata"]["val:all_nonvocal"] == "n/a"
    windows = [
        json.loads(line)
        for line in (output / "training_windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert windows
    assert all(
        row["signed_feature_manifest_sha256"]
        == summary["signed_feature_manifest_sha256"]
        for row in windows
    )
    with pytest.raises(ValueError, match="final distribution"):
        compile_features(
            canonical=canonical,
            raw_manifest=raw_manifest,
            output_dir=tmp_path / "final-rejected",
            require_final_distribution=True,
        )


def test_v12_feature_compiler_rejects_array_or_binding_tampering(
    tmp_path: Path,
) -> None:
    canonical, raw_manifest, _, feature_rows = _fixture(tmp_path)
    feature_rows[0]["ptm_sha256"] = "0" * 64
    _write_jsonl(raw_manifest, feature_rows)
    with pytest.raises(ValueError, match="array SHA mismatch"):
        compile_features(
            canonical=canonical,
            raw_manifest=raw_manifest,
            output_dir=tmp_path / "tampered",
        )


def test_v12_raw_rebind_accepts_legacy_superset_but_not_projected_features(
    tmp_path: Path,
) -> None:
    canonical, raw_manifest, source_rows, feature_rows = _fixture(tmp_path)
    one_source = tmp_path / "one_source.jsonl"
    _write_jsonl(one_source, [source_rows[0]])
    feature_rows[0]["canonical_sources_sha256"] = _sha256(one_source)
    legacy_rows = [feature_rows[0], {"source_id": "unrelated-legacy-source"}]
    _write_jsonl(raw_manifest, legacy_rows)
    summary = rebind(
        canonical=one_source,
        raw_manifest=raw_manifest,
        output_dir=tmp_path / "rebound",
    )
    assert summary["source_count"] == 1
    assert summary["ignored_legacy_source_count"] == 1
    rebound = json.loads(
        (tmp_path / "rebound" / "raw_feature_manifest.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert rebound["legacy_label_reuse"] is False
    assert rebound["ptm_dim"] == 2048

    legacy_rows[0]["ptm_dim"] = 128
    _write_jsonl(raw_manifest, legacy_rows)
    with pytest.raises(ValueError, match="projected/legacy feature dimensions"):
        rebind(
            canonical=one_source,
            raw_manifest=raw_manifest,
            output_dir=tmp_path / "projected-rejected",
        )


def test_v12_windows_safe_feature_name_handles_colon_source_ids() -> None:
    value = _safe_id("source::event00::occ00")
    assert ":" not in value
    assert value.startswith("source-event00-occ00-")
