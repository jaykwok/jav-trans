from __future__ import annotations

from tools.boundary.ja.compile_semantic_source_dual_labels import compile_record


def _row(*, gate: str, candidates: list[tuple[float, float, str]]) -> dict:
    return {
        "sample_id": "sample",
        "source": "test",
        "duration_s": candidates[-1][1],
        "reference_text": "reference",
        "source_gate": {
            "sample_id": "sample",
            "label": gate,
            "confidence": 0.9,
            "reason": "test",
        },
        "candidates": [
            {
                "candidate_id": f"c{index:02d}",
                "context_start_s": start,
                "context_end_s": end,
                "label": label,
                "confidence": 0.8,
            }
            for index, (start, end, label) in enumerate(candidates)
        ],
    }


def test_contains_semantic_membership_bridges_discardable_content() -> None:
    record, summary = compile_record(
        _row(
            gate="contains_semantic",
            candidates=[
                (0.0, 0.04, "semantic_target"),
                (0.04, 0.08, "discardable"),
                (0.08, 0.12, "semantic_target"),
            ],
        ),
        frame_hop_s=0.02,
    )

    metadata = record.boundary_metadata or {}
    assert metadata["semantic_class_frames"] == [
        "semantic_target",
        "semantic_target",
        "discardable",
        "discardable",
        "semantic_target",
        "semantic_target",
    ]
    assert metadata["semantic_membership_frames"] == ["inside"] * 6
    assert summary["content_retained_run_count"] == 2
    assert summary["membership_island_count"] == 1


def test_discardable_source_compiles_to_outside_membership() -> None:
    record, summary = compile_record(
        _row(
            gate="discardable",
            candidates=[(0.0, 0.06, "discardable")],
        ),
        frame_hop_s=0.02,
        partition="validation",
    )

    metadata = record.boundary_metadata or {}
    assert metadata["semantic_membership_frames"] == ["outside"] * 3
    assert record.speech_frames == [0, 0, 0]
    assert metadata["source_partition"] == "validation"
    assert summary["membership_island_count"] == 0
