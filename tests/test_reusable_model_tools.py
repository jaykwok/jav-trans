from __future__ import annotations

from pathlib import Path

import pytest

from tools.audits.compare_pre_asr_route_coverage import (
    compare_coverage,
    read_srt,
)
from tools.workflows.promote_torch_checkpoint import promote_checkpoint


def test_promote_torch_checkpoint_completes_artifact_contract(tmp_path: Path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "trained.pt"
    output = tmp_path / "production.pt"
    torch.save(
        {
            "metadata": {
                "asr_repo_id": "example/repo",
                "artifact": {"name": "pre_asr_cueqc"},
            },
            "decision_config": {"inference_window_size": 128},
            "model_state_dict": {"weight": torch.tensor([1.0])},
        },
        source,
    )

    promote_checkpoint(
        input_path=source,
        output_path=output,
        artifact_name="pre_asr_cueqc",
        display_name="Pre-ASR CueQC",
        version="v11",
        pipeline_stage=5,
        pipeline_role="final_chunk_keep_drop_routing",
        source_training_run="agents/temp/example",
        selected_validation={"keep_recall": 0.9},
        drop_threshold=0.95,
        promotion_reason="test",
        promoted_at="2026-07-04T00:00:00+00:00",
    )

    payload = torch.load(output, map_location="cpu", weights_only=False)
    artifact = payload["metadata"]["artifact"]
    assert artifact["production_filename"] == "production.pt"
    assert artifact["checkpoint_format_version"] == 1
    assert artifact["promoted"] is True
    assert artifact["self_contained"] is True
    assert artifact["source_training_run"] == "agents/temp/example"
    assert payload["metadata"]["selected_validation"] == {"keep_recall": 0.9}
    assert payload["decision_config"] == {
        "inference_window_size": 128,
        "drop_threshold": 0.95,
    }
    assert payload["model_state_dict"]["weight"].tolist() == [1.0]


def test_compare_pre_asr_route_coverage_reports_uncovered_semantic_cues(
    tmp_path: Path,
):
    srt = tmp_path / "reference.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n台詞\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n...\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\n次の台詞\n",
        encoding="utf-8",
    )
    reference = read_srt(srt)
    routes = [
        {
            "start": 0.5,
            "end": 2.5,
            "route": "keep_for_asr",
        },
        {
            "start": 4.8,
            "end": 6.2,
            "route": "drop_before_asr",
        },
    ]

    result = compare_coverage(reference, routes)

    assert result["kept_chunks"] == 1
    assert result["uncovered_cues"] == 2
    assert result["semantic_uncovered_cues"] == 1
    assert result["semantic_uncovered"][0]["text"] == "次の台詞"


def test_promote_torch_checkpoint_rejects_invalid_threshold(tmp_path: Path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "trained.pt"
    torch.save({"metadata": {}, "model_state_dict": {}}, source)

    with pytest.raises(ValueError, match="between 0 and 1"):
        promote_checkpoint(
            input_path=source,
            output_path=tmp_path / "production.pt",
            artifact_name="model",
            display_name="Model",
            version="v1",
            pipeline_stage=1,
            pipeline_role="test",
            source_training_run="run",
            drop_threshold=1.1,
        )
