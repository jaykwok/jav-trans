from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.build_ctc_ab_jav_predictions import fixed_context  # noqa: E402
from tools.audits.evaluate_ctc_alignment_ab_audit import build, combine_rounds  # noqa: E402
from tools.audits.generate_ctc_alignment_ab_audit import (  # noqa: E402
    balanced_arm1_assignments,
    galgame_candidates,
    render_page,
    select_trials,
)


def test_fixed_jav_context_holds_text_and_equal_context() -> None:
    assert fixed_context(line_start_s=4.0, line_duration_s=2.0, audio_duration_s=9.0) == (3.0, 7.0)
    assert fixed_context(line_start_s=0.4, line_duration_s=2.0, audio_duration_s=3.0) == (0.0, 3.0)


def test_galgame_geometry_offsets_become_absolute_boundaries() -> None:
    details_a = [{"sample_id": "s", "core_index": 0, "start_offset_edged_ms": 100.0, "end_offset_edged_ms": 200.0}]
    details_b = [{"sample_id": "s", "core_index": 0, "start_offset_edged_ms": 150.0, "end_offset_edged_ms": 125.0}]
    composites = [{"sample_id": "s", "audio": "x.wav", "duration_s": 8.0, "core_spans": [{"core_id": "c", "text": "あ", "start_s": 1.0, "end_s": 6.0}]}]
    row = galgame_candidates(details_a=details_a, details_b=details_b, composites=composites)[0]
    assert row["model_a_start_s"] == 1.1
    assert row["model_b_start_s"] == 1.15
    assert row["model_a_end_s"] == 5.8
    assert row["model_b_end_s"] == 5.875


def test_selection_covers_each_domain_and_boundary_without_identical_pairs() -> None:
    candidates = []
    for domain in ("galgame", "jav"):
        for index in range(8):
            candidates.append({"candidate_id": f"{domain}-{index}", "domain": domain, "source_id": str(index), "audio_duration_s": 10.0, "model_a_start_s": 1.0, "model_b_start_s": 1.04 + index * 0.01, "model_a_end_s": 7.0, "model_b_end_s": 7.04 + index * 0.01})
    selected = select_trials(candidates, per_boundary=4, clip_s=2.5, seed=7)
    assert len(selected) == 16
    assert {(row["domain"], row["boundary"]) for row in selected} == {
        ("galgame", "onset"), ("galgame", "end"), ("jav", "onset"), ("jav", "end")
    }
    assert all(row["delta_ms"] >= 20.0 for row in selected)


def test_selection_can_be_jav_only_and_exclude_prior_pairs() -> None:
    candidates = [
        {
            "candidate_id": f"jav-{index}",
            "domain": "jav",
            "source_id": str(index),
            "audio_duration_s": 10.0,
            "model_a_start_s": 1.0,
            "model_b_start_s": 1.04 + index * 0.01,
            "model_a_end_s": 7.0,
            "model_b_end_s": 7.04 + index * 0.01,
        }
        for index in range(8)
    ]
    selected = select_trials(
        candidates,
        per_boundary=4,
        clip_s=2.5,
        seed=7,
        domains=("jav",),
        exclude_pairs={("jav-7", "onset"), ("jav-7", "end")},
    )
    assert len(selected) == 8
    assert {row["domain"] for row in selected} == {"jav"}
    assert all(row["candidate_id"] != "jav-7" for row in selected)


def test_page_contains_no_model_mapping_or_checkpoint_identity() -> None:
    page = render_page([{"row_id": "r1", "domain": "真实 JAV", "boundary": "开头", "text": "固定文字", "reference_src": "ref.mp3", "arm_1_src": "1.mp3", "arm_2_src": "2.mp3"}])
    assert "固定文字" in page
    assert "model_a" not in page
    assert "model_b" not in page
    assert "ctc_aligner.pt" not in page
    assert "delta_ms" not in page


def test_arm_position_is_balanced_inside_each_stratum() -> None:
    trials = [
        {"domain": domain, "boundary": boundary}
        for domain in ("galgame", "jav")
        for boundary in ("onset", "end")
        for _ in range(12)
    ]
    assignments = balanced_arm1_assignments(trials, seed=17)
    for domain in ("galgame", "jav"):
        for boundary in ("onset", "end"):
            indices = [
                index
                for index, row in enumerate(trials)
                if row == {"domain": domain, "boundary": boundary}
            ]
            assert sum(assignments[index] == "model_b" for index in indices) == 6


def test_evaluator_reveals_randomized_arms() -> None:
    answers = [
        {"row_id": "r1", "domain": "galgame", "boundary": "onset", "arm_1": "model_b", "arm_2": "model_a"},
        {"row_id": "r2", "domain": "jav", "boundary": "end", "arm_1": "model_a", "arm_2": "model_b"},
    ]
    verdicts = [
        {"row_id": "r1", "verdict": "arm_1_better"},
        {"row_id": "r2", "verdict": "arm_2_better"},
    ]
    result = build(answers, verdicts)
    assert result["overall"]["candidate_better"] == 2
    assert result["overall"]["baseline_better"] == 0
    assert result["overall"]["sign_test_p_two_sided"] == 0.5


def test_evaluator_combines_rounds_with_reused_page_row_ids() -> None:
    answers, verdicts = combine_rounds(
        [
            [{"row_id": "r1", "domain": "jav", "boundary": "onset", "arm_1": "model_b", "arm_2": "model_a"}],
            [{"row_id": "r1", "domain": "jav", "boundary": "end", "arm_1": "model_a", "arm_2": "model_b"}],
        ],
        [
            [{"row_id": "r1", "verdict": "arm_1_better"}],
            [{"row_id": "r1", "verdict": "arm_2_better"}],
        ],
    )
    result = build(answers, verdicts)
    assert result["rows"] == 2
    assert result["overall"]["candidate_better"] == 2


def test_evaluator_marks_missing_verdicts_unreviewed() -> None:
    answers = [
        {"row_id": "r1", "domain": "jav", "boundary": "onset", "arm_1": "model_b", "arm_2": "model_a"},
        {"row_id": "r2", "domain": "jav", "boundary": "end", "arm_1": "model_a", "arm_2": "model_b"},
    ]
    result = build(answers, [{"row_id": "r1", "verdict": "equivalent_good"}])
    assert result["rows"] == 2
    assert result["overall"]["unreviewed"] == 1
    assert result["human_preference_gate"] == "incomplete"
