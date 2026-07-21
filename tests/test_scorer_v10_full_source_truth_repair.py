from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.ja.model import (
    SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
)
from tools.audits.evaluate_scorer_v10_full_source_span_audit import evaluate
from tools.audits.generate_scorer_v10_full_source_span_audit_html import (
    ITEM_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
)
from tools.audits.rebind_scorer_v10_feature_cache_after_relabel import rebind
from tools.boundary.ja.apply_speech_island_scorer_v10_full_source_truth_repairs import (
    apply_repairs,
)
from tools.boundary.ja.apply_speech_island_scorer_v10_repair_event_unsure import (
    SUMMARY_SCHEMA as R7_SUMMARY_SCHEMA,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(source_id: str, partition: str, *, role: str, sample_count: int) -> dict:
    background_id = f"bg-{source_id}"
    if role == "speech":
        core_id = f"core-{source_id}"
        spans = [
            {
                "start_sample": 0,
                "end_sample": sample_count,
                "label": "speech",
                "label_source": "fixture",
                "core_id": core_id,
            }
        ]
        core_ids = [core_id]
        row_background_id = ""
    else:
        spans = [
            {
                "start_sample": 0,
                "end_sample": sample_count,
                "label": "background",
                "label_source": "fixture",
                "background_id": background_id,
            }
        ]
        core_ids = []
        row_background_id = background_id
    return {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "canonical_label_schema": "speech_scorer_canonical_frames_v1",
        "source_id": source_id,
        "audio": str(Path("fixture") / f"{source_id}.wav"),
        "row_role": role,
        "partition": partition,
        "core_ids": core_ids,
        "background_id": row_background_id,
        "background_source_ids": [background_id],
        "background_source_video_ids": [f"video-{source_id}"],
        "sample_rate": 16000,
        "sample_count": sample_count,
        "duration_s": sample_count / 16000,
        "input_distribution": "full_source_windows",
        "canonical_spans": spans,
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    target_id = "train-target"
    sources = [
        _source("train-speech", "train", role="speech", sample_count=3200),
        _source(target_id, "train", role="all_background", sample_count=3100),
        _source("train-background", "train", role="all_background", sample_count=3200),
        _source("val-speech", "val", role="speech", sample_count=3200),
        _source("val-background", "val", role="all_background", sample_count=3200),
        _source("test-speech", "test", role="speech", sample_count=3200),
        _source("test-background", "test", role="all_background", sample_count=3200),
    ]
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sources),
        encoding="utf-8",
    )
    audio_manifest = tmp_path / "audio_manifest.json"
    audio_manifest.write_text(
        json.dumps(
            [
                {
                    "audio_id": row["source_id"],
                    "audio": row["audio"],
                    "partition": row["partition"],
                    "row_role": row["row_role"],
                }
                for row in sources
            ]
        ),
        encoding="utf-8",
    )
    feature_labels = tmp_path / "feature_cache_labels.jsonl"
    feature_labels.write_text(
        "".join(
            json.dumps({"audio_id": row["source_id"], "fixture": True}) + "\n"
            for row in sources
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "r7-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": R7_SUMMARY_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "canonical_sources": str(canonical),
                "canonical_sources_sha256": _sha256(canonical),
                "audio_manifest": str(audio_manifest),
                "audio_manifest_sha256": _sha256(audio_manifest),
                "feature_cache_labels": str(feature_labels),
                "feature_cache_labels_sha256": _sha256(feature_labels),
                "audio_bytes_changed": False,
                "source_identity_changed": False,
                "partition_identity_changed": False,
                "training_manifest_ready": False,
                "checkpoint_promotion_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    audit_manifest = tmp_path / "audit_manifest.jsonl"
    audit_manifest.write_text(
        json.dumps(
            {
                "schema": ITEM_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": target_id,
                "partition": "train",
                "frame_count": 10,
                "frame_hop_s": 0.02,
                "duration_s": 0.19375,
                "audio": "audio/target.wav",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    verdicts = tmp_path / "manual_verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "schema": MANUAL_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": target_id,
                "partition": "train",
                "frame_count": 10,
                "frame_hop_s": 0.02,
                "reviewed_full_source": True,
                "verdict": "complete_with_target_speech",
                "spans": [
                    {
                        "label": "background",
                        "start_frame": 0,
                        "end_frame": 3,
                        "start_s": 0.0,
                        "end_s": 0.06,
                    },
                    {
                        "label": "speech",
                        "start_frame": 3,
                        "end_frame": 10,
                        "start_s": 0.06,
                        "end_s": 0.2,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gate_path = evaluate(
        audit_manifest=audit_manifest,
        manual_verdicts=verdicts,
        output_dir=tmp_path / "gate",
    )
    return summary, gate_path, canonical, target_id


def test_full_source_truth_repair_is_exact_deterministic_and_rebindable(
    tmp_path: Path,
) -> None:
    input_summary, gate_path, canonical, target_id = _write_fixture(tmp_path)
    original_rows = [
        json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines()
    ]
    output = tmp_path / "output"
    result = apply_repairs(
        input_summary_path=input_summary,
        manual_gate_path=gate_path,
        output_dir=output,
    )
    assert result["requested_label_frame_counts"] == {"background": 3, "speech": 7}
    assert result["canonical_frame_count_delta"] == {
        "background": -7,
        "speech": 7,
        "unsure": 0,
    }
    assert result["changed_frame_transition_counts"] == {
        "background_to_background": 3,
        "background_to_speech": 7,
    }
    assert result["changed_source_ids"] == [target_id]
    assert result["added_core_count"] == 1
    assert result["max_core_use_count"] == 1
    assert result["audio_bytes_changed"] is False
    assert result["source_identity_changed"] is False
    assert result["partition_identity_changed"] is False
    assert result["model_output_used_as_truth"] is False
    assert result["asr_output_used_as_truth"] is False

    corrected_rows = [
        json.loads(line)
        for line in (output / "canonical_sources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["source_id"] for row in corrected_rows] == [
        row["source_id"] for row in original_rows
    ]
    original_by_id = {row["source_id"]: row for row in original_rows}
    corrected_by_id = {row["source_id"]: row for row in corrected_rows}
    for source_id, original in original_by_id.items():
        if source_id != target_id:
            assert corrected_by_id[source_id] == original
    target = corrected_by_id[target_id]
    assert target["row_role"] == "speech"
    assert target["background_id"] == ""
    assert target["canonical_spans"][0]["start_sample"] == 0
    assert target["canonical_spans"][0]["end_sample"] == 960
    assert target["canonical_spans"][0]["label"] == "background"
    assert target["canonical_spans"][1]["start_sample"] == 960
    assert target["canonical_spans"][1]["end_sample"] == 3100
    assert target["canonical_spans"][1]["label"] == "speech"
    assert target["core_ids"] == [target["canonical_spans"][1]["core_id"]]

    second = apply_repairs(
        input_summary_path=input_summary,
        manual_gate_path=gate_path,
        output_dir=tmp_path / "second-output",
    )
    assert second["added_core_ids"] == result["added_core_ids"]
    assert second["canonical_sources_sha256"] == result["canonical_sources_sha256"]

    config_sha = "1" * 64
    signed_manifest = tmp_path / "signed.jsonl"
    signed_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "schema": SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
                    "boundary_serialization_contract_id": (
                        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                    ),
                    "feature_extractor_schema": (
                        SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA
                    ),
                    "feature_config_sha256": config_sha,
                    "source_id": row["source_id"],
                    "audio_path": row["audio"],
                    "audio_sample_count": row["sample_count"],
                    "audio_sample_rate": row["sample_rate"],
                }
            )
            + "\n"
            for row in corrected_rows
        ),
        encoding="utf-8",
    )
    base_gate = tmp_path / "base-feature-gate.json"
    base_gate.write_text(
        json.dumps(
            {
                "schema": SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "feature_extractor_schema": (
                    SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA
                ),
                "signed_feature_manifest": str(signed_manifest),
                "signed_feature_manifest_sha256": _sha256(signed_manifest),
                "feature_config": {"fixture": True},
                "feature_config_sha256": config_sha,
                "audio_content_signature": "a" * 64,
                "feature_content_signature": "b" * 64,
                "cache_binding_signature": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    rebound = rebind(
        relabel_summary_path=output / "summary.json",
        base_feature_gate_path=base_gate,
        output_dir=tmp_path / "rebound",
    )
    assert rebound["canonical_sources_sha256"] == result["canonical_sources_sha256"]
    assert rebound["label_only_changed_source_count"] == 1
    assert rebound["training_manifest_allowed"] is True


def test_full_source_truth_repair_rejects_changed_decisions(tmp_path: Path) -> None:
    input_summary, gate_path, _, _ = _write_fixture(tmp_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    decisions = Path(gate["decisions"])
    decisions.write_text(
        decisions.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="decisions SHA256 mismatch"):
        apply_repairs(
            input_summary_path=input_summary,
            manual_gate_path=gate_path,
            output_dir=tmp_path / "output",
        )


def test_full_source_truth_repair_rejects_non_background_target(
    tmp_path: Path,
) -> None:
    input_summary, gate_path, canonical, target_id = _write_fixture(tmp_path)
    rows = [
        json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines()
    ]
    target = next(row for row in rows if row["source_id"] == target_id)
    target["row_role"] = "speech"
    target["background_id"] = ""
    target["core_ids"] = ["preexisting-target-core"]
    target["canonical_spans"] = [
        {
            "start_sample": 0,
            "end_sample": target["sample_count"],
            "label": "speech",
            "label_source": "fixture",
            "core_id": "preexisting-target-core",
        }
    ]
    canonical.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = json.loads(input_summary.read_text(encoding="utf-8"))
    summary["canonical_sources_sha256"] = _sha256(canonical)
    audio_manifest = Path(summary["audio_manifest"])
    audio_rows = json.loads(audio_manifest.read_text(encoding="utf-8"))
    next(row for row in audio_rows if row["audio_id"] == target_id)[
        "row_role"
    ] = "speech"
    audio_manifest.write_text(json.dumps(audio_rows), encoding="utf-8")
    summary["audio_manifest_sha256"] = _sha256(audio_manifest)
    input_summary.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="exact all-background row"):
        apply_repairs(
            input_summary_path=input_summary,
            manual_gate_path=gate_path,
            output_dir=tmp_path / "output",
        )


def test_full_source_truth_repair_rejects_wrong_contract(tmp_path: Path) -> None:
    input_summary, gate_path, _, _ = _write_fixture(tmp_path)
    gate = copy.deepcopy(json.loads(gate_path.read_text(encoding="utf-8")))
    gate["boundary_serialization_contract_id"] = "legacy-generation"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="another Boundary contract"):
        apply_repairs(
            input_summary_path=input_summary,
            manual_gate_path=gate_path,
            output_dir=tmp_path / "output",
        )
