from tools.datasets.project_omni_island_cuts import project_island


def test_project_island_uses_distinct_nearest_candidates() -> None:
    island = {
        "island_id": "w#outer000",
        "window_id": "w",
        "video_id": "v",
        "span_index": 0,
        "span_start_s": 10.0,
        "span_end_s": 15.0,
        "duration_s": 5.0,
        "candidates": [
            {"feature_index": 10, "relative_time_s": 1.1, "kind": "weak", "p_cut": 0.2},
            {"feature_index": 20, "relative_time_s": 2.9, "kind": "weak", "p_cut": 0.8},
        ],
    }
    label = {"cuts": [{"time_s": 1.0}, {"time_s": 3.0}]}

    projected = project_island(island, label)

    assert [cut["time_s"] for cut in projected["cuts"]] == [1.1, 2.9]
    assert [cut["teacher_time_s"] for cut in projected["cuts"]] == [1.0, 3.0]
    assert projected["complete_search"] is True
