from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.frame_teacher_supervision import (  # noqa: E402
    BLANK_LABEL,
    IGNORE_LABEL,
    SPEECH_LABEL,
    balanced_sparse_frame_loss,
    compile_sparse_frame_targets,
    load_accepted_frame_teachers,
    merge_intervals,
    summarize_sparse_frame_probabilities,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_interval_merge_closes_only_small_teacher_gaps() -> None:
    assert merge_intervals(
        [(0.1, 0.2), (0.25, 0.4), (0.8, 0.9)], maximum_gap_s=0.1
    ) == [(0.1, 0.4), (0.8, 0.9)]


def test_sparse_targets_keep_boundaries_and_short_gaps_ignored() -> None:
    labels = compile_sparse_frame_targets(
        {
            "duration_s": 3.0,
            "lexical_intervals": [(1.0, 1.2), (1.25, 1.5)],
        },
        output_frames=78,
        upsample=2,
        positive_merge_gap_s=0.1,
        boundary_ignore_s=0.1,
        negative_minimum_s=0.5,
    )

    # The 50 ms tokenizer gap is merged into speech.  The 100 ms guards stay
    # ignored, and only the long leading/trailing regions become hard blank.
    assert labels[32] == SPEECH_LABEL  # center ~= 1.25 s
    assert labels[23] == IGNORE_LABEL  # center ~= 0.90 s, left guard
    assert labels[10] == BLANK_LABEL
    assert labels[60] == BLANK_LABEL


def test_a_cropped_row_shifts_its_labels_by_its_own_start() -> None:
    """Word timestamps are absolute to the source clip; frame 0 is not.

    A crop row starting at 1.0 s sees the same speech at 0.0-0.5 s of *its own*
    frames. Without the offset the labels land a full second early and nothing
    raises - the auxiliary loss just teaches the head the wrong place.
    """
    teacher = {"duration_s": 3.0, "lexical_intervals": [(1.0, 1.5)]}

    whole = compile_sparse_frame_targets(teacher, output_frames=78, upsample=2)
    cropped = compile_sparse_frame_targets(
        teacher, output_frames=52, upsample=2, start_offset_s=1.0
    )

    frame_s = 3.0 / 78
    # Same speech, addressed in each row's own time base.
    assert whole[int(1.25 / frame_s)] == SPEECH_LABEL
    assert cropped[int(0.25 / frame_s)] == SPEECH_LABEL
    # And the crop must not still claim speech where the whole clip had it.
    assert cropped[int(1.25 / frame_s)] == BLANK_LABEL


def test_an_offset_past_the_clip_is_refused() -> None:
    with pytest.raises(ValueError, match="past the end"):
        compile_sparse_frame_targets(
            {"duration_s": 3.0, "lexical_intervals": [(1.0, 1.5)]},
            output_frames=26,
            upsample=2,
            start_offset_s=3.5,
        )


def test_sub_resolution_island_does_not_invent_a_positive() -> None:
    labels = compile_sparse_frame_targets(
        {"duration_s": 1.0, "lexical_intervals": [(0.4, 0.42)]},
        output_frames=26,
        upsample=2,
    )
    assert np.all(labels == IGNORE_LABEL)


def test_loader_uses_accepted_manifest_as_quality_gate(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    manifest = tmp_path / "accepted.jsonl"
    _write_jsonl(
        results,
        [
            {
                "source_id": "accepted",
                "audio_sha256": "sha-a",
                "source_duration_s": 2.0,
                "response": {
                    "words": [
                        {"text": "あ", "start_s": 0.2, "end_s": 0.4},
                        {"text": "、", "start_s": 0.4, "end_s": 0.8},
                    ]
                },
            },
            {
                "source_id": "rejected",
                "audio_sha256": "sha-r",
                "source_duration_s": 2.0,
                "response": {
                    "words": [{"text": "幻", "start_s": 0.1, "end_s": 0.2}]
                },
            },
        ],
    )
    _write_jsonl(
        manifest,
        [{"source_id": "accepted", "teacher_audio_sha256": "sha-a"}],
    )

    teachers, summary = load_accepted_frame_teachers(results, manifest)

    assert set(teachers) == {"accepted"}
    assert teachers["accepted"]["lexical_intervals"] == [(0.2, 0.4)]
    assert summary["lexical_units"] == 1


def test_balanced_frame_loss_ignores_unknown_and_balances_classes() -> None:
    torch = pytest.importorskip("torch")
    probabilities = torch.tensor(
        [[[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.8, 0.2]]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[BLANK_LABEL, SPEECH_LABEL, IGNORE_LABEL, BLANK_LABEL]],
        dtype=torch.int8,
    )

    loss, counts = balanced_sparse_frame_loss(probabilities.log(), labels, torch)

    expected_blank = (-np.log(0.9) - np.log(0.8)) / 2.0
    expected_speech = -np.log(0.9)
    assert float(loss) == pytest.approx((expected_blank + expected_speech) / 2.0)
    assert counts == {"blank_frames": 2, "speech_frames": 1}


def test_sparse_probability_summary_reports_separation() -> None:
    report = summarize_sparse_frame_probabilities(
        np.asarray([0.9, 0.8, 0.1, 0.2, 0.7]),
        np.asarray(
            [BLANK_LABEL, BLANK_LABEL, SPEECH_LABEL, SPEECH_LABEL, IGNORE_LABEL]
        ),
    )

    assert report["blank_probability_margin"] == pytest.approx(0.7)
    assert report["balanced_accuracy_at_0_5"] == 1.0
    assert report["blank_frames"] == 2
    assert report["speech_frames"] == 2
