from __future__ import annotations

from tools.boundary.ja.build_runtime_semantic_split_dataset import (
    boundary_noise_bucket,
    label_runtime_proposals,
    semantic_split_truth_boundaries,
)


def test_runtime_proposal_labeling_matches_one_candidate_per_truth() -> None:
    proposals = [
        {"time_s": 0.90},
        {"time_s": 1.03},
        {"time_s": 1.20},
        {"time_s": 2.00},
        {"time_s": 3.05},
    ]

    labels, cut_indexes, matched_truth, proposal_truth = label_runtime_proposals(
        proposals,
        truth_times_s=[1.0, 3.0],
        cut_match_s=0.10,
        unsure_radius_s=0.30,
    )

    assert labels == ["unsure", "cut", "unsure", "continue", "cut"]
    assert cut_indexes == {1, 4}
    assert matched_truth == {0, 1}
    assert proposal_truth == {1: 0, 4: 1}


def test_runtime_proposal_labeling_reports_missing_proposal() -> None:
    labels, cut_indexes, matched_truth, proposal_truth = label_runtime_proposals(
        [{"time_s": 2.0}],
        truth_times_s=[1.0],
        cut_match_s=0.10,
        unsure_radius_s=0.30,
    )

    assert labels == ["continue"]
    assert cut_indexes == set()
    assert matched_truth == set()
    assert proposal_truth == {}


def test_runtime_proposal_inside_long_nonsemantic_gap_is_a_cut() -> None:
    labels, cut_indexes, matched_truth, proposal_truth = label_runtime_proposals(
        [{"time_s": 1.65}, {"time_s": 2.70}],
        truth_times_s=[2.0],
        truth_regions_s=[(1.5, 2.5)],
        cut_match_s=0.10,
        unsure_radius_s=0.30,
    )

    assert labels == ["cut", "unsure"]
    assert cut_indexes == {0}
    assert matched_truth == {0}
    assert proposal_truth == {0: 0}


def test_boundary_noise_bucket_distinguishes_touch_single_and_multi() -> None:
    assert boundary_noise_bucket({"inter_noise_unit_count": 0}) == "touching_11"
    assert boundary_noise_bucket({"inter_noise_unit_count": 1}) == "single_noise_101"
    assert (
        boundary_noise_bucket({"inter_noise_unit_count": 3})
        == "multi_noise_1001_plus"
    )


def test_noise_run_expands_to_two_structural_split_boundaries() -> None:
    boundaries = semantic_split_truth_boundaries(
        {
            "utterance_boundaries": [
                {
                    "time_s": 2.0,
                    "previous_speech_end_s": 1.5,
                    "next_speech_start_s": 2.5,
                    "inter_noise_unit_count": 3,
                },
                {
                    "time_s": 4.0,
                    "previous_speech_end_s": 4.0,
                    "next_speech_start_s": 4.0,
                    "inter_noise_unit_count": 0,
                },
            ]
        }
    )

    assert [row["time_s"] for row in boundaries] == [1.5, 2.5, 4.0]
    assert [row["structural_role"] for row in boundaries] == [
        "speech_to_noise",
        "noise_to_speech",
        "speech_to_speech",
    ]
