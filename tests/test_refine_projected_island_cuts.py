from tools.datasets.refine_projected_island_cuts import _feature_lookup_time_s


def test_primary_cut_uses_pre_refinement_proposal_time() -> None:
    cut = {"time_s": 6.025, "candidate_kind": "primary"}
    audit = [
        {
            "kind": "primary",
            "time_s": 16.025,
            "proposal_time_s": 16.063,
        }
    ]

    assert _feature_lookup_time_s(cut, span_start_s=10.0, audit_rows=audit) == 16.063


def test_weak_cut_uses_projected_candidate_time() -> None:
    cut = {"time_s": 2.5, "candidate_kind": "weak"}

    assert _feature_lookup_time_s(cut, span_start_s=10.0, audit_rows=[]) == 12.5
