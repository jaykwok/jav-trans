from __future__ import annotations

import hashlib
import json
from pathlib import Path

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.ja.model import (
    SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
)
from tools.audits.rebind_scorer_v10_feature_cache_after_relabel import rebind
from tools.boundary.ja.apply_speech_island_scorer_v10_repair_event_unsure import (
    VERDICT_SCHEMA,
    apply_unsure,
)
from tools.boundary.ja.build_speech_island_scorer_v10_sparse_train_layout import (
    SUMMARY_SCHEMA as R6_SUMMARY_SCHEMA,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _span(start: int, end: int, label: str, identity: str, **extra) -> dict:
    result = {
        "start_sample": start,
        "end_sample": end,
        "label": label,
        "label_source": "fixture",
        **extra,
    }
    result["core_id" if label == "speech" else "background_id"] = identity
    return result


def _source(
    source_id: str,
    partition: str,
    spans: list[dict],
    core_ids: list[str],
    *,
    row_role: str = "speech",
    background_id: str = "",
) -> dict:
    return {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": "speech_scorer_canonical_frames_v1",
        "source_id": source_id,
        "audio": str(Path("fixture") / f"{source_id}.wav"),
        "row_role": row_role,
        "partition": partition,
        "core_ids": core_ids,
        "background_id": background_id,
        "background_source_ids": [f"bg-{partition}"],
        "background_source_video_ids": [f"video-{partition}"],
        "sample_rate": 16000,
        "sample_count": spans[-1]["end_sample"],
        "duration_s": spans[-1]["end_sample"] / 16000,
        "input_distribution": "full_source_windows",
        "canonical_spans": spans,
        "additive_overlay": None,
    }


def test_repair_event_unsure_propagates_and_rebinds_unchanged_cache(
    tmp_path: Path,
) -> None:
    event_id = "background::event00"
    mapped_repair = _span(
        1600,
        3200,
        "speech",
        "repair-mapped",
        repair_event_id=event_id,
        origin_background_id="bg-val",
    )
    control_repair = _span(
        800,
        2400,
        "speech",
        "repair-control",
        repair_event_id=event_id,
        origin_background_id="bg-val",
    )
    sources = [
        _source("train-speech", "train", [_span(0, 3200, "speech", "core-train")], ["core-train"]),
        _source(
            "train-bg",
            "train",
            [_span(0, 3200, "background", "bg-train")],
            [],
            row_role="all_background",
            background_id="bg-train",
        ),
        _source(
            "mapped",
            "val",
            [_span(0, 1600, "speech", "core-mapped"), mapped_repair, _span(3200, 4800, "background", "bg-val")],
            ["core-mapped", "repair-mapped"],
        ),
        {
            **_source(
                "control",
                "val",
                [_span(0, 800, "background", "bg-val"), control_repair, _span(2400, 4000, "background", "bg-val")],
                ["repair-control"],
            ),
            "repaired_background_id": "bg-val",
        },
        _source(
            "val-bg",
            "val",
            [_span(0, 3200, "background", "bg-val")],
            [],
            row_role="all_background",
            background_id="bg-val-extra",
        ),
        _source("test-speech", "test", [_span(0, 3200, "speech", "core-test")], ["core-test"]),
        _source(
            "test-bg",
            "test",
            [_span(0, 3200, "background", "bg-test")],
            [],
            row_role="all_background",
            background_id="bg-test",
        ),
    ]
    # Keep background identities partition-local and unique where row identity matters.
    sources[4]["background_source_ids"] = ["bg-val-extra"]
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(
        "".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8"
    )
    r6 = tmp_path / "r6.json"
    r6.write_text(
        json.dumps(
            {
                "schema": R6_SUMMARY_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "canonical_sources": str(canonical),
                "canonical_sources_sha256": _sha256(canonical),
            }
        ),
        encoding="utf-8",
    )
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "schema": VERDICT_SCHEMA,
                "repair_event_id": event_id,
                "verdict": "unsure",
                "reason": "unintelligible",
                "reviewed_occurrences": [
                    {"source_id": "mapped", "start_sample": 1600, "end_sample": 3200},
                    {"source_id": "control", "start_sample": 800, "end_sample": 2400},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "relabel"
    result = apply_unsure(
        input_summary_path=r6, verdicts_path=verdicts, output_dir=output
    )
    assert result["changed_source_ids"] == ["control", "mapped"]
    assert result["ignored_core_ids"] == ["repair-control", "repair-mapped"]
    # The mapped event starts on a frame boundary; the control event starts and
    # ends at half frames, so its two mixed cells were already unsure.
    assert result["canonical_frame_count_delta"]["speech"] == -9
    assert result["canonical_frame_count_delta"]["unsure"] == 9
    corrected = {
        row["source_id"]: row
        for row in (
            json.loads(line)
            for line in (output / "canonical_sources.jsonl").read_text().splitlines()
        )
    }
    assert corrected["mapped"]["core_ids"] == ["core-mapped"]
    assert corrected["mapped"]["canonical_spans"][1]["label"] == "unsure"
    assert corrected["control"]["core_ids"] == []
    assert corrected["control"]["row_role"] == "all_background"
    assert corrected["control"]["background_id"] == "bg-val"

    config_sha = "1" * 64
    signed_rows = []
    for source in corrected.values():
        signed_rows.append(
            {
                "schema": SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "feature_extractor_schema": SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
                "feature_config_sha256": config_sha,
                "source_id": source["source_id"],
                "audio_path": source["audio"],
                "audio_sample_count": source["sample_count"],
                "audio_sample_rate": source["sample_rate"],
            }
        )
    signed_manifest = tmp_path / "signed.jsonl"
    signed_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in signed_rows), encoding="utf-8"
    )
    base_gate = tmp_path / "base-gate.json"
    base_gate.write_text(
        json.dumps(
            {
                "schema": SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "feature_extractor_schema": SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
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
    assert rebound["signed_feature_manifest_sha256"] == _sha256(signed_manifest)
    assert rebound["canonical_sources_sha256"] == result["canonical_sources_sha256"]
    assert rebound["label_only_changed_source_count"] == 2
    assert rebound["training_manifest_allowed"] is True
