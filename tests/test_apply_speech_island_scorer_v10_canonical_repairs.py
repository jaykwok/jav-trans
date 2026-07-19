from __future__ import annotations

import hashlib
import json
from pathlib import Path

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.boundary.ja.apply_speech_island_scorer_v10_canonical_repairs import (
    apply_repairs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _source(
    source_id: str,
    partition: str,
    *,
    role: str,
    core_ids: list[str] | None = None,
    background_id: str = "",
    spans: list[dict] | None = None,
) -> dict:
    ids = [background_id] if background_id else [f"noise-{source_id}"]
    return {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": "speech_scorer_canonical_frames_v1",
        "source_id": source_id,
        "audio": f"{source_id}.wav",
        "row_role": role,
        "partition": partition,
        "core_ids": core_ids or [],
        "background_id": background_id,
        "background_source_ids": ids,
        "background_source_video_ids": [f"video-{ids[0]}"],
        "sample_rate": 16000,
        "sample_count": 1600,
        "duration_s": 0.1,
        "input_distribution": "full_source_windows",
        "canonical_spans": spans
        or [
            {
                "start_sample": 0,
                "end_sample": 1600,
                "label": "background" if role == "all_background" else "speech",
                "background_id": background_id if role == "all_background" else "",
                "core_id": (core_ids or [""])[0] if role == "speech" else "",
            }
        ],
    }


def test_apply_repairs_quarantines_assets_and_preserves_unsure(tmp_path: Path) -> None:
    sources = [
        _source(
            "train-bad-speech",
            "train",
            role="speech",
            core_ids=["train-bad-core"],
            spans=[
                {
                    "start_sample": 0,
                    "end_sample": 1600,
                    "label": "speech",
                    "core_id": "train-bad-core",
                }
            ],
        ),
        _source("train-clean-speech", "train", role="speech", core_ids=["train-core"]),
        _source(
            "train-bad-background",
            "train",
            role="all_background",
            background_id="noise-train-bad-speech",
        ),
        _source(
            "train-clean-background",
            "train",
            role="all_background",
            background_id="train-clean-bg",
        ),
        _source(
            "val-speech",
            "val",
            role="speech",
            core_ids=["val-core-a", "val-core-b"],
            spans=[
                {
                    "start_sample": 0,
                    "end_sample": 800,
                    "label": "speech",
                    "core_id": "val-core-a",
                },
                {
                    "start_sample": 800,
                    "end_sample": 1600,
                    "label": "speech",
                    "core_id": "val-core-b",
                },
            ],
        ),
        _source("val-background", "val", role="all_background", background_id="val-bg"),
        _source("test-speech", "test", role="speech", core_ids=["test-core"]),
        _source("test-background", "test", role="all_background", background_id="test-bg"),
    ]
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(canonical, sources)
    decisions = tmp_path / "decisions.jsonl"
    _write_jsonl(
        decisions,
        [
            {
                "schema": "speech_scorer_v10_canonical_span_repair_item_v1",
                "span_id": "val-speech::span00",
                "source_id": "val-speech",
                "original_label": "speech",
                "start_sample": 0,
                "end_sample": 800,
                "verdict": "unsure",
                "note": "",
            }
        ],
    )
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema": "speech_scorer_v10_canonical_span_repair_gate_v1",
                "canonical_sources_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
                "complete": True,
                "canonical_recompile_ready": True,
                "quarantined_background_ids": ["noise-train-bad-speech"],
                "decisions": str(decisions),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "corrected"
    previous_audit = tmp_path / "previous_audit.jsonl"
    _write_jsonl(
        previous_audit,
        [
            {"source_id": "train-bad-speech"},
            {"source_id": "train-bad-background"},
            {"source_id": "val-speech"},
        ],
    )
    summary = apply_repairs(
        canonical_sources=canonical,
        repair_gate=gate,
        output_dir=output,
        previous_audit_manifest=previous_audit,
    )
    corrected = [
        json.loads(line)
        for line in (output / "canonical_sources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert summary["dropped_source_count"] == 2
    assert summary["dataset"]["source_count"] == 6
    assert summary["dataset"]["core_count"] == 3
    assert summary["changed_span_count"] == 1
    assert summary["canonical_frame_counts"]["unsure"] > 0
    val = next(row for row in corrected if row["source_id"] == "val-speech")
    assert val["canonical_spans"][0]["label"] == "unsure"
    assert val["core_ids"] == ["val-core-b"]
    assert val["ignored_core_ids"] == [
        {
            "core_id": "val-core-a",
            "manual_label": "unsure",
            "span_id": "val-speech::span00",
        }
    ]
    assert summary["replacement_audit_required"] is True
    assert summary["repair_source_ids"] == ["val-speech"]
    assert len(summary["replacement_audit_source_ids"]) == 3
    assert "val-speech" in summary["replacement_audit_source_ids"]
    assert summary["feature_cache_ready"] is False
