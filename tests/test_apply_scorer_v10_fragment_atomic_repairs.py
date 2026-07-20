from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.audits.evaluate_scorer_v10_fragment_atomic_repair_audit import (
    DECISION_SCHEMA,
    RESULT_SCHEMA,
)
from tools.boundary.ja.apply_speech_island_scorer_v10_fragment_atomic_repairs import (
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
    core_id: str = "",
    background_id: str = "",
    sample_count: int = 3200,
) -> dict:
    identity = background_id or f"noise-{source_id}"
    return {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": "speech_scorer_canonical_frames_v1",
        "source_id": source_id,
        "audio": f"{source_id}.wav",
        "row_role": role,
        "partition": partition,
        "core_ids": [core_id] if core_id else [],
        "background_id": background_id,
        "background_source_ids": [identity],
        "background_source_video_ids": [f"video-{identity}"],
        "sample_rate": 16000,
        "sample_count": sample_count,
        "duration_s": sample_count / 16000,
        "input_distribution": "full_source_windows",
        "canonical_spans": [
            {
                "start_sample": 0,
                "end_sample": sample_count,
                "label": "speech" if core_id else "background",
                **({"core_id": core_id} if core_id else {"background_id": background_id}),
            }
        ],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    sources = [
        _source("train-speech", "train", role="speech", core_id="train-core"),
        _source(
            "train-background",
            "train",
            role="all_background",
            background_id="train-bg",
        ),
        _source("val-speech", "val", role="speech", core_id="val-core"),
        _source(
            "val-background", "val", role="all_background", background_id="val-bg"
        ),
        _source("test-speech", "test", role="speech", core_id="test-core"),
        _source(
            "test-background",
            "test",
            role="all_background",
            background_id="test-bg",
        ),
    ]
    canonical = tmp_path / "canonical_sources.jsonl"
    _write_jsonl(canonical, sources)

    decisions = tmp_path / "atomic_gate.decisions.jsonl"
    _write_jsonl(
        decisions,
        [
            {
                "schema": DECISION_SCHEMA,
                "atomic_id": "val-speech:truth0:model_speech:0-2",
                "source_id": "val-speech",
                "partition": "val",
                "start_frame": 0,
                "end_frame": 2,
                "sample_rate": 16000,
                "sample_count": 3200,
                "start_sample": 0,
                "end_sample": 640,
                "canonical_span_index": 0,
                "core_id": "val-core",
                "original_canonical_label": "speech",
                "label": "background",
                "label_source": "manual_fragment_atomic_repair_v1",
            },
            {
                "schema": DECISION_SCHEMA,
                "atomic_id": "val-speech:truth0:model_speech:2-10",
                "source_id": "val-speech",
                "partition": "val",
                "start_frame": 2,
                "end_frame": 10,
                "sample_rate": 16000,
                "sample_count": 3200,
                "start_sample": 640,
                "end_sample": 3200,
                "canonical_span_index": 0,
                "core_id": "val-core",
                "original_canonical_label": "speech",
                "label": "speech",
                "label_source": "manual_fragment_atomic_repair_v1",
            },
        ],
    )
    evidence = {}
    for name in ("audit_summary", "fragmentation_manifest", "fragmentation_verdicts"):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        evidence[name] = path
    gate = tmp_path / "atomic_gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "complete": True,
                "canonical_recompile_ready": True,
                "relation_violation_count": 0,
                "canonical_sources_sha256": hashlib.sha256(
                    canonical.read_bytes()
                ).hexdigest(),
                "audit_summary": str(evidence["audit_summary"]),
                "audit_summary_sha256": hashlib.sha256(
                    evidence["audit_summary"].read_bytes()
                ).hexdigest(),
                "fragmentation_audit_manifest": str(
                    evidence["fragmentation_manifest"]
                ),
                "fragmentation_audit_manifest_sha256": hashlib.sha256(
                    evidence["fragmentation_manifest"].read_bytes()
                ).hexdigest(),
                "fragmentation_manual_verdicts": str(
                    evidence["fragmentation_verdicts"]
                ),
                "fragmentation_manual_verdicts_sha256": hashlib.sha256(
                    evidence["fragmentation_verdicts"].read_bytes()
                ).hexdigest(),
                "decisions": str(decisions),
                "atomic_unit_count": 2,
            }
        ),
        encoding="utf-8",
    )
    return canonical, gate


def test_apply_fragment_atomic_repairs_splits_exact_spans_and_preserves_core(
    tmp_path: Path,
) -> None:
    canonical, gate = _fixture(tmp_path)
    output = tmp_path / "corrected"
    result = apply_repairs(
        canonical_sources=canonical,
        atomic_repair_gate=gate,
        output_dir=output,
    )
    corrected = [
        json.loads(line)
        for line in (output / "canonical_sources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    val = next(row for row in corrected if row["source_id"] == "val-speech")
    assert result["boundary_serialization_contract_id"] == (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    assert result["changed_atomic_span_count"] == 1
    assert result["verified_speech_atomic_span_count"] == 1
    assert result["dropped_source_count"] == 0
    assert result["removed_core_count"] == 0
    assert result["canonical_frame_count_delta"] == {
        "speech": -2,
        "background": 2,
        "unsure": 0,
    }
    assert val["core_ids"] == ["val-core"]
    assert [span["label"] for span in val["canonical_spans"]] == [
        "background",
        "speech",
    ]
    assert val["canonical_spans"][0]["origin_core_id"] == "val-core"
    assert "core_id" not in val["canonical_spans"][0]
    assert result["audio_bytes_changed"] is False
    assert result["existing_feature_cache_authorized"] is False
    assert result["checkpoint_promotion_authorized"] is False


def test_apply_fragment_atomic_repairs_rejects_another_canonical_manifest(
    tmp_path: Path,
) -> None:
    canonical, gate = _fixture(tmp_path)
    canonical.write_text(
        canonical.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="another canonical manifest"):
        apply_repairs(
            canonical_sources=canonical,
            atomic_repair_gate=gate,
            output_dir=tmp_path / "rejected",
        )
