from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from asr.backends import qwen
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT


REPO_IDS = (qwen.QWEN_ASR_06B_REPO_ID, qwen.QWEN_ASR_17B_REPO_ID)


@pytest.mark.parametrize("repo_id", REPO_IDS)
@pytest.mark.parametrize(
    ("mapping", "artifact_name", "stage", "repo_metadata_key"),
    [
        (
            qwen.DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO,
            "speech_island_scorer",
            1,
            "ptm_repo_id",
        ),
        (
            qwen.DEFAULT_SPEECH_BOUNDARY_PROPOSAL_CHECKPOINT_BY_REPO,
            "boundary_proposal_scorer",
            1,
            "ptm_repo_id",
        ),
        (
            qwen.DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
            "semantic_split_model",
            3,
            "ptm_repo_id",
        ),
        (
            qwen.DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO,
            "inner_edge_refiner",
            5,
            "ptm_repo_id",
        ),
        (
            qwen.DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO,
            "pre_asr_cueqc",
            5,
            "asr_repo_id",
        ),
    ],
)
def test_promoted_five_model_artifact_contract(
    repo_id: str,
    mapping: dict[str, str],
    artifact_name: str,
    stage: int,
    repo_metadata_key: str,
) -> None:
    torch = pytest.importorskip("torch")
    if not mapping.get(repo_id):
        pytest.skip("model is intentionally not active for this ASR repo")
    path = Path(mapping[repo_id])
    assert path.is_file()
    assert "agents/temp" not in path.as_posix()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    artifact = metadata["artifact"]

    assert artifact["name"] == artifact_name
    assert artifact["pipeline_stage"] == stage
    assert artifact["production_filename"] == path.name
    assert artifact["checkpoint_format_version"] == 1
    assert artifact["promoted"] is True
    assert artifact["self_contained"] is True
    assert metadata[repo_metadata_key] == repo_id
    assert metadata["boundary_serialization_contract_id"] == (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )


def test_split_v4_candidate_is_binary_argmax_and_excludes_unsure() -> None:
    torch = pytest.importorskip("torch")
    path = Path(
        "src/checkpoints/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/"
        "semantic_split_model_v4.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "semantic_split_model_v4"
    assert payload["model_config"]["num_classes"] == 2
    assert payload["decision_config"] == {"decision_mode": "binary_argmax_cut"}
    assert payload["metadata"]["training_labels"] == ["cut", "continue"]
    assert payload["metadata"]["excluded_training_labels"] == ["unsure"]
    assert payload["metadata"]["boundary_serialization_contract_id"] == (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    assert payload["metadata"]["excluded_training_label_count"] == 27388
    assert payload["metadata"]["manual_label_overrides"]["override_count"] == 33
    assert payload["metadata"]["manual_label_overrides"]["training_label_counts"] == {
        "cut": 8,
        "continue": 1,
        "ignore": 24,
    }


def test_17b_inner_v2_declares_binary_serialization_contract() -> None:
    torch = pytest.importorskip("torch")
    path = Path(
        qwen.DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO[qwen.QWEN_ASR_17B_REPO_ID]
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["schema"] == "inner_edge_refiner_v2"
    assert payload["feature_config"]["acoustic_refinement"] is True
    assert "display_only" not in payload["feature_config"]
    assert payload["metadata"]["acoustic_refinement"] is True
    assert payload["metadata"]["feeds_asr"] is True
    assert payload["metadata"]["metadata_correction"]["weights_changed"] is False
    assert payload["metadata"]["boundary_serialization_contract_id"] == (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )


def test_17b_cueqc_v13_is_binary_argmax_without_threshold_and_binds_split_inner() -> None:
    torch = pytest.importorskip("torch")
    path = Path(qwen.DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO[qwen.QWEN_ASR_17B_REPO_ID])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    decision = payload["decision_config"]
    metadata = payload["metadata"]
    split_path = Path(
        qwen.DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO[qwen.QWEN_ASR_17B_REPO_ID]
    )
    inner_path = Path(
        qwen.DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO[qwen.QWEN_ASR_17B_REPO_ID]
    )

    assert payload["schema"] == "cueqc_pre_asr_semantic_chunk_v13"
    assert payload["model_config"]["num_classes"] == 2
    assert decision["decision_mode"] == "argmax"
    assert "drop_threshold" not in decision
    assert decision["model_only"] is True
    assert decision["hard_keep_veto"] is False
    assert decision["hard_drop_rule"] is False
    assert decision["keep_veto"] is False
    assert metadata["training_labels"] == ["drop", "keep"]
    assert metadata["excluded_training_labels"] == ["unsure"]
    assert metadata["boundary_serialization_contract_id"] == (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    assert metadata["excluded_training_label_count"] >= 0
    assert payload["semantic_split_weights_sha256"] == hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest()
    assert payload["inner_edge_refiner_weights_sha256"] == hashlib.sha256(
        inner_path.read_bytes()
    ).hexdigest()


def test_06b_boundary_models_are_all_pending_retrain_placeholders() -> None:
    mappings = (
        qwen.DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO,
        qwen.DEFAULT_SPEECH_BOUNDARY_PROPOSAL_CHECKPOINT_BY_REPO,
        qwen.DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO,
        qwen.DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
        qwen.DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO,
        qwen.DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO,
    )
    assert all(mapping[qwen.QWEN_ASR_06B_REPO_ID] == "" for mapping in mappings)
    assert qwen.BOUNDARY_PIPELINE_STATUS_BY_REPO[qwen.QWEN_ASR_06B_REPO_ID] == (
        "pending_binary_retrain"
    )
    assert qwen.DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO[
        qwen.QWEN_ASR_17B_REPO_ID
    ] == ""
    assert qwen.BOUNDARY_PIPELINE_STATUS_BY_REPO[qwen.QWEN_ASR_17B_REPO_ID] == (
        "pending_outer_v3_audit"
    )
