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
    if repo_id not in mapping:
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


def test_windows_bundle_includes_all_five_model_checkpoint_roots() -> None:
    spec = Path("packaging/jav-trans-web.spec").read_text(encoding="utf-8")
    assert '"src/checkpoints"' in spec
    assert '"src/boundary/ja/checkpoints"' not in spec
    assert '"src/boundary/checkpoints"' not in spec
    assert '"src/asr/checkpoints"' not in spec
    assert "libtorchcodec_custom_ops*.dll" in spec
    assert "_torchcodec_binaries()" in spec


def test_windows_bundle_uses_jav_trans_package_name() -> None:
    spec = Path("packaging/jav-trans-web.spec").read_text(encoding="utf-8")
    build_script = Path("packaging/build_windows.ps1").read_text(encoding="utf-8")
    archive_script = Path("packaging/archive_release.ps1").read_text(encoding="utf-8")

    assert 'name="jav-trans"' in spec
    assert "JAV_TRANS_SKIP_MODELS" in spec
    assert "JAVTRANS_SKIP_MODELS" not in spec
    assert "dist/jav-trans/jav-trans.exe" in build_script
    assert "dist/JAVTrans/JAVTrans.exe" not in build_script
    assert "jav-trans-windows-x64.7z" in archive_script
