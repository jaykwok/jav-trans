from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from boundary.ja.vocal_envelope_training import (
    adjacency_loss,
    compute_vocal_envelope_losses,
    frame_cross_entropy,
    run_balanced_cross_entropy,
    source_metrics,
)
from boundary.ja.vocal_envelope_v12 import (
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_TRAINING_ROW_SCHEMA,
    VocalEnvelopeScorerV12Network,
    vocal_envelope_v12_model_config,
)
from tools.boundary.ja.train_vocal_envelope_scorer_v12 import (
    GATE_SCHEMA,
    _collate,
    _numeric_gate,
    _pack,
    _selection,
    _validate,
    _write_json,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v12_run_balanced_and_adjacency_losses_match_contract() -> None:
    import torch

    logits = torch.tensor(
        [
            [
                [2.0, -1.0],
                [1.0, -0.5],
                [0.5, -0.5],
                [-1.0, 2.0],
                [-0.5, 1.0],
            ]
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[0, 0, 0, 1, 1]])
    owner = torch.ones_like(labels, dtype=torch.bool)
    per_frame = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 2), labels.reshape(-1), reduction="none"
    )
    expected_run = torch.stack((per_frame[:3].mean(), per_frame[3:].mean())).mean()
    assert torch.allclose(
        run_balanced_cross_entropy(logits, labels, owner), expected_run
    )
    assert torch.allclose(
        frame_cross_entropy(logits, labels, owner), per_frame.mean()
    )
    losses_a = compute_vocal_envelope_losses(logits, labels, owner, arm="A")
    losses_b = compute_vocal_envelope_losses(logits, labels, owner, arm="B")
    losses_c = compute_vocal_envelope_losses(logits, labels, owner, arm="C")
    assert torch.allclose(losses_a["total_loss"], losses_a["main_loss"])
    assert torch.allclose(
        losses_b["total_loss"],
        losses_b["main_loss"] + 0.5 * losses_b["run_loss"],
    )
    assert torch.allclose(
        losses_c["total_loss"],
        losses_c["main_loss"]
        + 0.5 * losses_c["run_loss"]
        + 0.25 * losses_c["adjacency_loss"],
    )
    losses_c["total_loss"].backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())

    perfect = torch.tensor(
        [[[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0]]]
    )
    wrong = torch.tensor(
        [[[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0], [-8.0, 8.0]]]
    )
    adjacency_labels = torch.tensor([[0, 0, 1, 1]])
    adjacency_owner = torch.ones_like(adjacency_labels, dtype=torch.bool)
    assert adjacency_loss(perfect, adjacency_labels, adjacency_owner) < adjacency_loss(
        wrong, adjacency_labels, adjacency_owner
    )


def test_v12_source_metrics_count_internal_runs_not_frames() -> None:
    truth = np.asarray([1, 1, 1, 1, 1], dtype=np.int64)
    prediction = np.asarray([1, 0, 0, 1, 1], dtype=np.int64)
    metrics = source_metrics(truth, prediction)
    assert metrics.internal_hole_count == 1
    assert metrics.prediction_run_count == 2
    assert metrics.vocal_continuity == 0.0
    assert metrics.complete_vocal_run_deletion_count == 0
    assert metrics.all_vocal_keep is False

    deleted = source_metrics(truth, np.zeros_like(truth))
    assert deleted.complete_vocal_run_deletion_count == 1
    assert deleted.internal_hole_count == 0

    unsure = source_metrics(
        np.asarray([1, 1, -100], dtype=np.int64),
        np.asarray([1, 1, 1], dtype=np.int64),
    )
    assert unsure.all_vocal_keep is None


def test_v12_selection_rejects_degenerate_controls() -> None:
    good = {
        "vocal_recall": 0.98,
        "non_vocal_recall": 0.97,
        "vocal_continuity": 0.96,
        "complete_vocal_run_deletion_count": 0,
        "internal_hole_count": 2,
        "all_vocal_source_keep_recall": 1.0,
        "all_nonvocal_source_full_drop_recall": 1.0,
    }
    assert _numeric_gate(good) is True
    all_keep = {**good, "non_vocal_recall": 0.0, "all_nonvocal_source_full_drop_recall": 0.0}
    all_drop = {
        **good,
        "vocal_recall": 0.0,
        "vocal_continuity": 0.0,
        "complete_vocal_run_deletion_count": 1,
    }
    fragmented = {**good, "vocal_continuity": 0.5, "internal_hole_count": 30}
    assert _numeric_gate(all_keep) is False
    assert _numeric_gate(all_drop) is False
    assert _numeric_gate(fragmented) is False
    assert _selection(good) > _selection(all_keep)
    assert _selection(good) > _selection(fragmented)


def test_v12_cpu_forward_backward_and_batch_padding_equivalence() -> None:
    import torch

    torch.manual_seed(117)
    model = VocalEnvelopeScorerV12Network(**vocal_envelope_v12_model_config())
    model.eval()
    first = {
        "ptm": np.zeros((9, 2048), dtype=np.float32),
        "mfcc": np.zeros((9, 40), dtype=np.float32),
        "labels": np.asarray([0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int64),
        "owner_start": 0,
        "owner_end": 9,
    }
    second = {
        "ptm": np.ones((6, 2048), dtype=np.float32),
        "mfcc": np.ones((6, 40), dtype=np.float32),
        "labels": np.asarray([1, 1, 0, 0, 1, 1], dtype=np.int64),
        "owner_start": 0,
        "owner_end": 6,
    }
    singleton = _collate([first], torch, torch.device("cpu"))
    batched = _collate([first, second], torch, torch.device("cpu"))
    with torch.inference_mode():
        single_logits = model(
            singleton["ptm"], singleton["mfcc"], attention_mask=singleton["attention"]
        )
        batch_logits = model(
            batched["ptm"], batched["mfcc"], attention_mask=batched["attention"]
        )
    torch.testing.assert_close(single_logits[0], batch_logits[0], rtol=1e-5, atol=1e-5)

    model.train()
    logits = model(
        batched["ptm"], batched["mfcc"], attention_mask=batched["attention"]
    )
    losses = compute_vocal_envelope_losses(
        logits, batched["labels"], batched["owner"], arm="C"
    )
    losses["total_loss"].backward()
    assert model.ptm_projector.weight.grad is not None
    assert bool(torch.isfinite(model.ptm_projector.weight.grad).all())


def _training_rows() -> list[dict]:
    rows = []
    for partition in ("train", "val", "test"):
        rows.append(
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_TRAINING_ROW_SCHEMA,
                "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
                "canonical_label_schema": VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
                "canonical_sources_sha256": "canonical",
                "raw_feature_manifest_sha256": "raw",
                "signed_feature_manifest_sha256": "signed",
                "row_id": f"source-{partition}::window0000",
                "source_id": f"source-{partition}",
                "partition": partition,
                "core_ids": [f"core-{partition}"],
                "synthetic_composite": False,
                "feature_path": f"feature-{partition}.npz",
                "feature_sha256": f"feature-sha-{partition}",
                "label_path": f"label-{partition}.npz",
                "label_sha256": f"label-sha-{partition}",
                "source_frame_count": 4,
                "window_start_frame": 0,
                "window_end_frame": 4,
                "owner_start_frame": 0,
                "owner_end_frame": 4,
                "owner_local_start": 0,
                "owner_local_end": 4,
            }
        )
    return rows


def _write_gate(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    canonical = tmp_path / "canonical.jsonl"
    raw = tmp_path / "raw.jsonl"
    signed = tmp_path / "signed.jsonl"
    canonical.write_text("canonical\n", encoding="utf-8")
    raw.write_text("raw\n", encoding="utf-8")
    signed.write_text("signed\n", encoding="utf-8")
    bound_rows = []
    for row in rows:
        bound_rows.append(
            {
                **row,
                "canonical_sources_sha256": _sha256(canonical),
                "raw_feature_manifest_sha256": _sha256(raw),
                "signed_feature_manifest_sha256": _sha256(signed),
            }
        )
    rows_path = tmp_path / "training_windows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in bound_rows), encoding="utf-8"
    )
    gate_path = tmp_path / "feature_cache_gate.json"
    gate = {
        "schema": GATE_SCHEMA,
        "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
        "status": "approved_for_training",
        "training_allowed": True,
        "canonical_sources": str(canonical.resolve()),
        "canonical_sources_sha256": _sha256(canonical),
        "raw_feature_manifest": str(raw.resolve()),
        "raw_feature_manifest_sha256": _sha256(raw),
        "signed_feature_manifest": str(signed.resolve()),
        "signed_feature_manifest_sha256": _sha256(signed),
        "training_windows": str(rows_path.resolve()),
        "training_windows_sha256": _sha256(rows_path),
        "source_count": 3,
        "window_count": len(bound_rows),
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return gate_path, rows_path


def test_v12_training_manifest_enforces_owner_and_core_identity(tmp_path: Path) -> None:
    rows = _training_rows()
    gate_path, rows_path = _write_gate(tmp_path, rows)
    _, validated = _validate(gate_path, rows_path)
    assert len(validated) == 3

    reused_core = [dict(row) for row in rows]
    reused_core[1]["core_ids"] = reused_core[0]["core_ids"]
    gate_path, rows_path = _write_gate(tmp_path / "reused", reused_core)
    with pytest.raises(ValueError, match="core is reused"):
        _validate(gate_path, rows_path)

    gap = [dict(row) for row in rows]
    gap[0]["owner_start_frame"] = 1
    gap[0]["owner_local_start"] = 1
    gate_path, rows_path = _write_gate(tmp_path / "gap", gap)
    with pytest.raises(ValueError, match="cover each source exactly once"):
        _validate(gate_path, rows_path)


def test_v12_batch_budget_and_atomic_progress(tmp_path: Path) -> None:
    rows = _training_rows()
    assert len(_pack(rows, max_padded_frames=8, max_rows=2)) == 2
    with pytest.raises(ValueError, match="exceeds batch frame budget"):
        _pack(rows, max_padded_frames=3, max_rows=1)
    progress = tmp_path / "progress.json"
    _write_json(progress, {"step": 1, "total": 2})
    assert json.loads(progress.read_text(encoding="utf-8")) == {"step": 1, "total": 2}
    assert not list(tmp_path.glob(".progress.json.*.tmp"))
