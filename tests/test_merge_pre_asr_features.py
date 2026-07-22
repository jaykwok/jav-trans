from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools.asr.cueqc.merge_pre_asr_features import run


def _bundle(
    torch, *, source_id: str, partition: str, row_id: str, core_id: str | None = None
) -> dict:
    return {
        "schema": "cueqc_pre_asr_semantic_chunk_v13_features",
        "feature_schema": "features",
        "runtime_adapter": "adapter",
        "feature_names": ["a"],
        "all_feature_names": ["a"],
        "ptm_bin_count": 2,
        "ptm_dim": 3,
        "asr_repo_id": "repo",
        "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
        "training_manifest_allowed": True,
        "semantic_split_weights_sha256": "a" * 64,
        "inner_edge_refiner_weights_sha256": "b" * 64,
        "rows": [{"id": row_id}],
        "groups": [
            {
                "group_index": 0,
                "row_ids": [row_id],
                "source_id": source_id,
                "source_core_ids": [core_id or f"core-{row_id}"],
                "dataset_role": partition,
            }
        ],
        "source_files": [f"{row_id}.jsonl"],
        "label_files": [f"{row_id}.labels.jsonl"],
        "scalar_features": torch.zeros((1, 1, 1)),
        "ptm_bins": torch.zeros((1, 1, 2, 3)),
        "bin_mask": torch.ones((1, 1, 2)),
        "chunk_mask": torch.ones((1, 1)),
        "labels": torch.zeros((1, 1), dtype=torch.long),
    }


def test_merge_preserves_frozen_partitions_and_full_groups(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(_bundle(torch, source_id="s-train", partition="train", row_id="a"), first)
    torch.save(_bundle(torch, source_id="s-test", partition="test", row_id="b"), second)
    output = tmp_path / "merged.pt"

    run(argparse.Namespace(features=[str(first), str(second)], output=str(output)))

    merged = torch.load(output, map_location="cpu", weights_only=False)
    assert [group["dataset_role"] for group in merged["groups"]] == ["train", "test"]
    assert [group["source_id"] for group in merged["groups"]] == ["s-train", "s-test"]
    assert merged["chunk_mask"].shape == (2, 1)
    summary = json.loads(output.with_suffix(".summary.json").read_text("utf-8"))
    assert summary["context_preserved"] is True
    assert summary["partition_reassignment"] is False


def test_merge_rejects_duplicate_provisional_subisland(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(_bundle(torch, source_id="s", partition="train", row_id="same"), first)
    torch.save(_bundle(torch, source_id="s", partition="train", row_id="same"), second)

    with pytest.raises(ValueError, match="duplicated across inputs"):
        run(
            argparse.Namespace(
                features=[str(first), str(second)],
                output=str(tmp_path / "merged.pt"),
            )
        )


def test_merge_rejects_duplicate_semantic_core(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(
        _bundle(
            torch,
            source_id="s-a",
            partition="train",
            row_id="a",
            core_id="same-core",
        ),
        first,
    )
    torch.save(
        _bundle(
            torch,
            source_id="s-b",
            partition="train",
            row_id="b",
            core_id="same-core",
        ),
        second,
    )

    with pytest.raises(ValueError, match="semantic core is duplicated"):
        run(
            argparse.Namespace(
                features=[str(first), str(second)],
                output=str(tmp_path / "merged.pt"),
            )
        )
