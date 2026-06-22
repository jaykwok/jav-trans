from __future__ import annotations

from boundary.timing_formula import recommend_boundary_timing_params


GALGAME_100K_DURATION_STATS = {
    "percentiles_s": {
        "p5": 1.048678,
        "p50": 4.503656,
        "p80": 7.613963,
        "p90": 9.56,
    }
}


def test_galgame_100k_distribution_recommends_current_defaults():
    result = recommend_boundary_timing_params(GALGAME_100K_DURATION_STATS)

    assert result["env"] == {
        "BOUNDARY_PLANNER_TARGET_CHUNK_S": 3.0,
        "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S": 5.0,
        "BOUNDARY_PLANNER_MIN_CHUNK_S": 0.4,
        "SUBTITLE_SOFT_MAX_S": 5.5,
        "MAX_SUBTITLE_DURATION": 6.5,
        "ASR_SEGMENT_HARD_MAX_DURATION": 6.5,
    }
    assert result["rationale"]["target_core_source_equivalent_s"] == 4.5
    assert result["rationale"]["max_core_source_equivalent_s"] == 7.5


def test_faster_target_domain_tightens_core_and_context_formula():
    result = recommend_boundary_timing_params(
        GALGAME_100K_DURATION_STATS,
        target_domain_speedup=2.0,
    )

    assert result["env"]["BOUNDARY_PLANNER_TARGET_CHUNK_S"] == 2.3
    assert result["env"]["BOUNDARY_PLANNER_MAX_CORE_CHUNK_S"] == 3.5
