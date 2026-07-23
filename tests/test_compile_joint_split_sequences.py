from __future__ import annotations

from pathlib import Path

import numpy as np

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.sequence_features import SPLIT_CANDIDATE_SCALAR_NAMES
from boundary.split_model import SEMANTIC_SPLIT_FEATURE_SCHEMA
from tools.datasets import compile_joint_boundary_preasr_dataset as joint_compiler
from tools.datasets.compile_joint_boundary_preasr_dataset import _compile_split
from tools.boundary.ja.acoustic_split_teacher_contracts import (
    ACOUSTIC_SPLIT_TEACHER_PROMPT_VERSION,
    HISTORICAL_APPROVED_SPLIT_TEACHER_PROMPT_VERSIONS,
)
from tools.boundary.ja.acoustic_split_v4_dataset import (
    SPLIT_V4_DATASET_SUMMARY_SCHEMA,
    SPLIT_V4_INPUT_DISTRIBUTION,
    SPLIT_V4_MFCC_DIM,
    SPLIT_V4_PTM_DIM,
    SPLIT_V4_UPSTREAM_SHA_FIELDS,
    file_sha256,
)

import pytest

VALID_SPLIT_PROMPT_VERSION = ACOUSTIC_SPLIT_TEACHER_PROMPT_VERSION
UPSTREAM_SHA256 = {
    field: character * 64
    for field, character in zip(SPLIT_V4_UPSTREAM_SHA_FIELDS, "abc")
}


def _write_window_bundle(
    path: Path,
    *,
    window_id: str | None = None,
    source_id: str = "source-0",
    partition: str = "train",
) -> dict[str, object]:
    window_id = window_id or path.parent.name
    audio_path = path.parent / "audio.wav"
    audio_path.write_bytes(f"audio:{window_id}".encode("utf-8"))
    audio_sha256 = file_sha256(audio_path)
    frame_dim = SPLIT_V4_PTM_DIM + SPLIT_V4_MFCC_DIM
    scalar_dim = len(SPLIT_CANDIDATE_SCALAR_NAMES)
    np.savez(
        path,
        frame_features=np.arange(4 * 2 * frame_dim, dtype=np.float32).reshape(
            4, 2, frame_dim
        ),
        scalar_features=np.arange(4 * scalar_dim, dtype=np.float32).reshape(
            4, scalar_dim
        ),
        proposal_times_s=np.asarray([0.5, 1.0, 2.2, 2.6], dtype=np.float32),
        core_starts_s=np.asarray([0.0, 0.0, 2.0, 2.0], dtype=np.float32),
        core_ends_s=np.asarray([1.5, 1.5, 3.0, 3.0], dtype=np.float32),
        accepted=np.asarray([False, True, False, False]),
        p_cut=np.zeros(4, dtype=np.float32),
        p_continue=np.zeros(4, dtype=np.float32),
        p_unsure=np.zeros(4, dtype=np.float32),
        training_manifest_allowed=np.asarray([True]),
        feature_schema=np.asarray([SEMANTIC_SPLIT_FEATURE_SCHEMA]),
        input_distribution=np.asarray([SPLIT_V4_INPUT_DISTRIBUTION]),
        boundary_serialization_contract_id=np.asarray(
            [ACOUSTIC_BINARY_V12_CONTRACT.contract_id]
        ),
        ptm_repo_id=np.asarray([QWEN_ASR_17B_REPO_ID]),
        window_id=np.asarray([window_id]),
        source_id=np.asarray([source_id]),
        source_partition=np.asarray([partition]),
        audio_wav_sha256=np.asarray([audio_sha256]),
        **{field: np.asarray([value]) for field, value in UPSTREAM_SHA256.items()},
    )
    return {
        "semantic_split_training_manifest_allowed": True,
        "semantic_split_input_distribution": SPLIT_V4_INPUT_DISTRIBUTION,
        "semantic_split_feature_schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "audio_wav": str(audio_path),
        "audio_wav_sha256": audio_sha256,
        **UPSTREAM_SHA256,
    }


@pytest.mark.parametrize(
    "prompt_version",
    [
        VALID_SPLIT_PROMPT_VERSION,
        *sorted(HISTORICAL_APPROVED_SPLIT_TEACHER_PROMPT_VERSIONS),
    ],
)
def test_compile_split_emits_whole_islands_with_ignore_context(
    tmp_path: Path,
    prompt_version: str,
) -> None:
    bundle_path = tmp_path / "w0" / "semantic_split_features.npz"
    bundle_path.parent.mkdir(parents=True)
    provenance = _write_window_bundle(bundle_path)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "train",
            "source_start_s": 0.0,
            "semantic_split_features": str(bundle_path),
            **provenance,
        }
    ]
    labels = [
        {
            "window_id": "w0",
            "feature_index": 1,
            "label": "cut",
            "time_s": 1.0,
            "left_complete": True,
            "right_complete": True,
            "merged_better": False,
            "prompt_version": prompt_version,
        }
    ]

    summary = _compile_split(
        dataset=tmp_path,
        windows=windows,
        labels=labels,
    )

    output = np.load(tmp_path / "semantic_split" / "features.npz")
    # Only the labeled island is emitted, but with ALL of its candidates.
    assert output["labels"].tolist() == [-100, 0]
    assert output["feature_indexes"].tolist() == [0, 1]
    group_ids = output["group_ids"].astype(str).tolist()
    assert len(set(group_ids)) == 1
    assert group_ids[0].startswith("source-0|island|source-0:samples:")
    assert output["source_ids"].astype(str).tolist() == ["source-0", "source-0"]
    assert len(set(output["core_ids"].astype(str).tolist())) == 1
    omni = output["omni_aux"].tolist()
    assert omni[0] == [-1.0, -1.0, -1.0]
    assert omni[1] == [1.0, 1.0, 0.0]
    assert output["structural_roles"].tolist() == [-100, -100]
    assert output["pair_ids"].tolist() == [-1, -1]
    assert summary["labeled_count"] == 1
    assert summary["context_only_count"] == 1
    assert summary["group_count"] == 1
    assert summary["schema"] == SPLIT_V4_DATASET_SUMMARY_SCHEMA
    assert summary["training_manifest_allowed"] is True
    assert summary["frame_dim"] == SPLIT_V4_PTM_DIM + SPLIT_V4_MFCC_DIM
    assert summary["scalar_dim"] == len(SPLIT_CANDIDATE_SCALAR_NAMES)
    assert summary["dataset_sha256"] == file_sha256(
        tmp_path / "semantic_split" / "features.npz"
    )
    assert (tmp_path / "semantic_split" / "features.summary.json").exists()


def test_compile_split_reads_and_writes_named_variant(tmp_path: Path) -> None:
    canonical = tmp_path / "w0" / "semantic_split_features.npz"
    canonical.parent.mkdir(parents=True)
    provenance = _write_window_bundle(canonical)
    variant = canonical.with_name("semantic_split_features.06b.npz")
    _write_window_bundle(variant)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "train",
            "source_start_s": 0.0,
            "semantic_split_features": str(canonical),
            **provenance,
        }
    ]
    labels = [
        {
            "window_id": "w0",
            "feature_index": 1,
            "label": "cut",
            "time_s": 1.0,
            "prompt_version": VALID_SPLIT_PROMPT_VERSION,
        }
    ]

    summary = _compile_split(
        dataset=tmp_path,
        windows=windows,
        labels=labels,
        feature_variant="06b",
        output_variant="06b",
    )

    assert summary["output"].endswith("features.06b.npz")
    assert (tmp_path / "semantic_split" / "features.06b.npz").exists()
    assert (tmp_path / "semantic_split" / "summary.06b.json").exists()
    assert (tmp_path / "semantic_split" / "features.06b.summary.json").exists()


def test_compile_split_rejects_label_time_from_stale_candidate_export(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "w0" / "semantic_split_features.npz"
    bundle_path.parent.mkdir(parents=True)
    provenance = _write_window_bundle(bundle_path)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "train",
            "source_start_s": 0.0,
            "semantic_split_features": str(bundle_path),
            **provenance,
        }
    ]
    labels = [
        {
            "window_id": "w0",
            "feature_index": 1,
            "label": "cut",
            "time_s": 1.25,
            "prompt_version": VALID_SPLIT_PROMPT_VERSION,
        }
    ]

    with pytest.raises(ValueError, match="label time does not match"):
        _compile_split(dataset=tmp_path, windows=windows, labels=labels)


def test_compile_split_rejects_retired_teacher_labels(tmp_path: Path) -> None:
    bundle_path = tmp_path / "w0" / "semantic_split_features.npz"
    bundle_path.parent.mkdir(parents=True)
    provenance = _write_window_bundle(bundle_path)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "train",
            "source_start_s": 0.0,
            "semantic_split_features": str(bundle_path),
            **provenance,
        }
    ]
    labels = [
        {
            "window_id": "w0",
            "feature_index": 1,
            "label": "cut",
            "prompt_version": "joint_boundary_preasr_omni_v3_separate",
        }
    ]

    with pytest.raises(ValueError, match="retired teacher contract"):
        _compile_split(
            dataset=tmp_path,
            windows=windows,
            labels=labels,
            )


def test_compile_split_rejects_labels_without_prompt_version(tmp_path: Path) -> None:
    bundle_path = tmp_path / "w0" / "semantic_split_features.npz"
    bundle_path.parent.mkdir(parents=True)
    provenance = _write_window_bundle(bundle_path)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "train",
            "source_start_s": 0.0,
            "semantic_split_features": str(bundle_path),
            **provenance,
        }
    ]
    labels = [{"window_id": "w0", "feature_index": 1, "label": "cut"}]

    with pytest.raises(ValueError, match="retired teacher contract"):
        _compile_split(
            dataset=tmp_path,
            windows=windows,
            labels=labels,
            )


def test_compile_split_rejects_source_crossing_frozen_partitions(
    tmp_path: Path,
) -> None:
    paths = []
    for window_id in ("w0", "w1"):
        path = tmp_path / window_id / "semantic_split_features.npz"
        path.parent.mkdir(parents=True)
        _write_window_bundle(
            path,
            window_id=window_id,
            partition="train" if window_id == "w0" else "test",
        )
        paths.append(path)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "train",
            "semantic_split_training_manifest_allowed": True,
            "source_start_s": 0.0,
            "semantic_split_features": str(paths[0]),
            **_write_window_bundle(paths[0], window_id="w0", partition="train"),
        },
        {
            "window_id": "w1",
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "test",
            "semantic_split_training_manifest_allowed": True,
            "source_start_s": 10.0,
            "semantic_split_features": str(paths[1]),
            **_write_window_bundle(paths[1], window_id="w1", partition="test"),
        },
    ]
    with pytest.raises(ValueError, match="crosses frozen source partitions"):
        _compile_split(dataset=tmp_path, windows=windows, labels=[])


def test_compile_split_rejects_duplicate_core_from_overlapping_windows(
    tmp_path: Path,
) -> None:
    paths = []
    for window_id in ("w0", "w1"):
        path = tmp_path / window_id / "semantic_split_features.npz"
        path.parent.mkdir(parents=True)
        _write_window_bundle(path, window_id=window_id)
        paths.append(path)
    windows = []
    for window_id, path in zip(("w0", "w1"), paths):
        windows.append({
            "window_id": window_id,
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "train",
            "source_start_s": 0.0,
            "semantic_split_features": str(path),
            **_write_window_bundle(path, window_id=window_id),
        })
    labels = [
        {
            "window_id": window_id,
            "feature_index": 1,
            "label": "cut",
            "time_s": 1.0,
            "prompt_version": VALID_SPLIT_PROMPT_VERSION,
        }
        for window_id in ("w0", "w1")
    ]
    with pytest.raises(ValueError, match="duplicated by overlapping source windows"):
        _compile_split(dataset=tmp_path, windows=windows, labels=labels)


def test_compile_split_rejects_unverified_runtime_candidate_distribution(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "w0" / "semantic_split_features.npz"
    bundle_path.parent.mkdir(parents=True)
    provenance = _write_window_bundle(bundle_path)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "train",
            "source_start_s": 0.0,
            "semantic_split_features": str(bundle_path),
            **provenance,
            "semantic_split_training_manifest_allowed": False,
        }
    ]
    labels = [
        {
            "window_id": "w0",
            "feature_index": 1,
            "label": "cut",
            "time_s": 1.0,
            "prompt_version": VALID_SPLIT_PROMPT_VERSION,
        }
    ]
    with pytest.raises(ValueError, match="not approved for Split v4 training"):
        _compile_split(dataset=tmp_path, windows=windows, labels=labels)


def test_compile_split_rejects_audit_only_feature_bundle(tmp_path: Path) -> None:
    bundle_path = tmp_path / "w0" / "semantic_split_features.npz"
    bundle_path.parent.mkdir(parents=True)
    provenance = _write_window_bundle(bundle_path)
    with np.load(bundle_path) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    arrays["training_manifest_allowed"] = np.asarray([False])
    arrays["input_distribution"] = np.asarray(["retired_candidate_time_remap"])
    np.savez(bundle_path, **arrays)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "source_id": "source-0",
            "source_partition": "train",
            "source_start_s": 0.0,
            "semantic_split_features": str(bundle_path),
            **provenance,
        }
    ]
    labels = [
        {
            "window_id": "w0",
            "feature_index": 1,
            "label": "cut",
            "time_s": 1.0,
            "prompt_version": VALID_SPLIT_PROMPT_VERSION,
        }
    ]
    with pytest.raises(ValueError, match="audit-only, not training-ready"):
        _compile_split(dataset=tmp_path, windows=windows, labels=labels)


def test_compile_pre_asr_refreshes_sha_after_partition_rewrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import torch

    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("{}\n", encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}\n", encoding="utf-8")

    def fake_compile_features(*, output: Path, **_kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"groups": [{"audio_id": "w0", "dataset_role": "stale"}]},
            output,
        )
        return {
            "keep": 1,
            "drop": 0,
            "ambiguous_ignore": 0,
            "group_count": 1,
            "output_sha256": "0" * 64,
        }

    monkeypatch.setattr(joint_compiler, "compile_features", fake_compile_features)
    summary = joint_compiler._compile_pre_asr(
        dataset=tmp_path,
        windows=[
            {
                "window_id": "w0",
                "source_id": "source-0",
                "source_partition": "train",
                "pre_asr_candidates": str(candidates),
            }
        ],
        labels_path=labels,
        asr_repo_id=QWEN_ASR_17B_REPO_ID,
    )

    output = tmp_path / "pre_asr" / "features.pt"
    assert summary["output_sha256"] == file_sha256(output)
    assert summary["output_sha256"] != "0" * 64
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["groups"][0]["dataset_role"] == "train"
