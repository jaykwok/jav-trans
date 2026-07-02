from __future__ import annotations

from pathlib import Path

import pytest

from asr.backends import qwen


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
            qwen.DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO,
            "outer_edge_refiner",
            2,
            "ptm_repo_id",
        ),
        (
            qwen.DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
            "semantic_split_model",
            3,
            "ptm_repo_id",
        ),
        (
            qwen.DEFAULT_CUT_EDGE_REFINER_CHECKPOINT_BY_REPO,
            "cut_edge_refiner",
            4,
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


def test_windows_bundle_includes_all_five_model_checkpoint_roots() -> None:
    spec = Path("packaging/javtrans-web.spec").read_text(encoding="utf-8")
    assert '"src/boundary/ja/checkpoints"' in spec
    assert '"src/boundary/checkpoints"' in spec
    assert '"src/asr/checkpoints"' in spec
