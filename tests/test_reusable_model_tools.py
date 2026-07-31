from __future__ import annotations

from pathlib import Path

import pytest

from tools.workflows.promote_torch_checkpoint import promote_checkpoint


def test_promote_torch_checkpoint_completes_artifact_contract(tmp_path: Path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "trained.pt"
    output = tmp_path / "production.pt"
    torch.save(
        {
            "metadata": {
                "asr_repo_id": "example/repo",
                "artifact": {"name": "ctc_alignment_head"},
            },
            "decision_config": {"inference_window_size": 128},
            "model_state_dict": {"weight": torch.tensor([1.0])},
        },
        source,
    )

    promote_checkpoint(
        input_path=source,
        output_path=output,
        artifact_name="ctc_alignment_head",
        display_name="CTC Alignment Head",
        version="v1",
        pipeline_stage=2,
        pipeline_role="frame_alignment_and_pause_gate",
        source_training_run="agents/temp/example",
        selected_validation={"median_onset_error_ms": 60.0},
        metadata_updates={"teacher_checkpoint_sha256": "teacher-sha"},
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
    assert payload["metadata"]["selected_validation"] == {
        "median_onset_error_ms": 60.0
    }
    assert payload["metadata"]["teacher_checkpoint_sha256"] == "teacher-sha"
    assert payload["decision_config"] == {"inference_window_size": 128}
    assert payload["model_state_dict"]["weight"].tolist() == [1.0]
