from __future__ import annotations

import numpy as np

from tools.datasets.reexport_joint_split_features import remap_label_rows


def _label(window_id: str, index: int, time_s: float, label: str = "cut") -> dict:
    return {
        "window_id": window_id,
        "feature_index": index,
        "time_s": time_s,
        "label": label,
    }


def test_remap_moves_labels_to_nearest_new_candidate() -> None:
    labels = [
        _label("w0", 3, 1.00),
        _label("w0", 7, 4.02, label="continue"),
    ]
    times = {"w0": np.asarray([0.2, 1.01, 2.5, 4.00])}

    remapped, dropped = remap_label_rows(labels, times, time_match_s=0.05)

    assert not dropped
    assert [row["feature_index"] for row in remapped] == [1, 3]
    assert remapped[0]["time_s"] == 1.01
    assert remapped[0]["remap"]["source_feature_index"] == 3
    assert remapped[0]["remap"]["distance_s"] < 0.02


def test_remap_drops_labels_without_candidate_in_tolerance() -> None:
    labels = [_label("w0", 0, 1.0), _label("w1", 0, 2.0)]
    times = {"w0": np.asarray([1.2])}

    remapped, dropped = remap_label_rows(labels, times, time_match_s=0.05)

    assert not remapped
    reasons = {row["drop_reason"] for row in dropped}
    assert reasons == {"no_candidate_within_tolerance", "window_missing"}


def test_remap_never_collapses_two_labels_onto_one_candidate() -> None:
    labels = [
        _label("w0", 0, 1.00),
        _label("w0", 1, 1.02),
    ]
    times = {"w0": np.asarray([1.01])}

    remapped, dropped = remap_label_rows(labels, times, time_match_s=0.05)

    # The closer label wins the single candidate; the other is dropped.
    assert len(remapped) == 1
    assert len(dropped) == 1
    assert remapped[0]["remap"]["source_time_s"] == 1.00


def test_remap_cut_labels_use_wider_tolerance_than_continue() -> None:
    labels = [
        _label("w0", 0, 1.00, label="cut"),
        _label("w0", 1, 2.00, label="continue"),
    ]
    # Both new candidates sit 0.10s away from their old label positions.
    times = {"w0": np.asarray([1.10, 2.10])}

    remapped, dropped = remap_label_rows(
        labels, times, time_match_s=0.05, cut_time_match_s=0.15
    )

    assert [row["label"] for row in remapped] == ["cut"]
    assert remapped[0]["feature_index"] == 0
    assert [row["label"] for row in dropped] == ["continue"]


def test_remap_cut_label_wins_shared_candidate_over_closer_continue() -> None:
    labels = [
        _label("w0", 0, 1.08, label="continue"),
        _label("w0", 1, 1.00, label="cut"),
    ]
    times = {"w0": np.asarray([1.07])}

    remapped, dropped = remap_label_rows(
        labels, times, time_match_s=0.05, cut_time_match_s=0.15
    )

    # cut/unsure are matched first: the scarce cut anchor keeps the candidate
    # even though the continue label is nearer.
    assert [row["label"] for row in remapped] == ["cut"]
    assert [row["label"] for row in dropped] == ["continue"]
