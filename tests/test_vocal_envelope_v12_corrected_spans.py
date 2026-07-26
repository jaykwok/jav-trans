from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from boundary.ja.vocal_envelope_v12 import (
    VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
    vocal_envelope_v12_corrected_span_signature,
    vocal_envelope_v12_corrected_spans_from_verdict,
    validate_vocal_envelope_v12_corrected_spans,
)
from tools.boundary.ja.compile_vocal_envelope_scorer_v12_canonical import (
    EXPECTED_MAX_TOKENS,
    EXPECTED_PROMPT_PROFILE,
    EXPECTED_PROMPT_VERSION,
    PROVIDER_CONTRACTS,
    compile_canonical,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_teacher_contract import (
    teacher_contract_fingerprint_fields,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_calibration import (
    CALIBRATION_TEACHER_CONTRACT,
    load_approved_calibration,
)
from tools.omni.timestamp_contract import TIMESTAMP_CONTRACT_ID


CONTRACT_ID = "boundary_acoustic_binary_v12"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_corrected_partition_is_complete_and_signature_matches_page_contract() -> None:
    spans = [
        {"label": "vocal_candidate", "start_frame": 0, "end_frame": 2},
        {"label": "non_vocal_candidate", "start_frame": 2, "end_frame": 4},
        {"label": "unsure", "start_frame": 4, "end_frame": 5},
    ]
    normalized = validate_vocal_envelope_v12_corrected_spans(
        spans, frame_count=5, source_id="source-1"
    )
    assert normalized[1]["start_s"] == 0.04
    assert vocal_envelope_v12_corrected_span_signature(
        spans, frame_count=5, source_id="source-1"
    ) == vocal_envelope_v12_corrected_span_signature(
        normalized, frame_count=5, source_id="source-1"
    )
    verdict = {
        "corrected_spans": spans,
        "corrected_span_signature": vocal_envelope_v12_corrected_span_signature(
            spans, frame_count=5, source_id="source-1"
        ),
        "audit_manifest_sha256": "a" * 64,
    }
    assert vocal_envelope_v12_corrected_spans_from_verdict(
        verdict, frame_count=5, source_id="source-1"
    )
    with pytest.raises(ValueError, match="signature mismatch"):
        vocal_envelope_v12_corrected_spans_from_verdict(
            {**verdict, "corrected_span_signature": "f" * 64},
            frame_count=5,
            source_id="source-1",
        )
    with pytest.raises(ValueError, match="audit_manifest_sha256"):
        vocal_envelope_v12_corrected_spans_from_verdict(
            {**verdict, "audit_manifest_sha256": "0" * 64},
            frame_count=5,
            source_id="source-1",
        )

    with pytest.raises(ValueError, match="contiguous"):
        validate_vocal_envelope_v12_corrected_spans(
            [
                {"label": "vocal_candidate", "start_frame": 0, "end_frame": 2},
                {"label": "non_vocal_candidate", "start_frame": 3, "end_frame": 5},
            ],
            frame_count=5,
            source_id="source-1",
        )
    with pytest.raises(ValueError, match="merge adjacent"):
        validate_vocal_envelope_v12_corrected_spans(
            [
                {"label": "vocal_candidate", "start_frame": 0, "end_frame": 2},
                {"label": "vocal_candidate", "start_frame": 2, "end_frame": 5},
            ],
            frame_count=5,
            source_id="source-1",
        )
    assert vocal_envelope_v12_corrected_spans_from_verdict(
        {"reviewed_full_source": True}, frame_count=5, source_id="source-1"
    ) is None


def test_canonical_prefers_valid_manual_corrected_partition(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"fixed-audio")
    source = {
        "schema": "vocal_envelope_scorer_v12_source_v1",
        "boundary_serialization_contract_id": CONTRACT_ID,
        "source_id": "source-1",
        "video_id": "video-1",
        "partition": "train",
        "core_ids": ["core-1"],
        "audio": str(audio),
        "audio_sha256": _sha(audio),
        "duration_s": 0.1,
        "frame_count": 5,
        "sample_rate": 16000,
        "sample_count": 1600,
    }
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [source])
    provider = PROVIDER_CONTRACTS["openrouter"]
    preaudit = tmp_path / "preaudit.jsonl"
    evidence = {
        "schema": VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
        "source_id": "source-1",
        "video_id": "video-1",
        "partition": "train",
        "core_ids": ["core-1"],
        "audio": str(audio),
        "audio_sha256": _sha(audio),
        "duration_s": 0.1,
        "frame_count": 5,
        "sample_rate": 16000,
        "sample_count": 1600,
        "model": provider["model"],
        "provider_profile": "openrouter",
        "env_file_name": "openrouter",
        "reasoning_effort": "medium",
        "max_tokens": EXPECTED_MAX_TOKENS,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "prompt_profile": EXPECTED_PROMPT_PROFILE,
        "prompt_version": EXPECTED_PROMPT_VERSION,
        **teacher_contract_fingerprint_fields(),
        "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
        "teacher_execution_contract_id": provider["execution_contract"],
        "source_manifest_sha256": _sha(manifest),
        "teacher_failed_closed": False,
        "vocal_spans": [{"label": "vocal_candidate", "start_frame": 0, "end_frame": 3}],
        "non_vocal_spans": [{"label": "non_vocal_candidate", "start_frame": 3, "end_frame": 5}],
        "unsure_spans": [],
    }
    _write_jsonl(preaudit, [evidence])
    corrected = [
        {"label": "vocal_candidate", "start_frame": 0, "end_frame": 2},
        {"label": "non_vocal_candidate", "start_frame": 2, "end_frame": 5},
    ]
    verdicts = tmp_path / "verdicts.jsonl"
    _write_jsonl(
        verdicts,
        [
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
                "source_id": "source-1",
                "video_id": "video-1",
                "partition": "train",
                "audio_sha256": source["audio_sha256"],
                "duration_s": 0.1,
                "frame_count": 5,
                "source_manifest_sha256": _sha(manifest),
                "preaudit_sha256": _sha(preaudit),
                "audit_manifest_sha256": "b" * 64,
                "corrected_spans": corrected,
                "corrected_span_signature": vocal_envelope_v12_corrected_span_signature(
                    corrected, frame_count=5, source_id="source-1"
                ),
                "reviewed_full_source": True,
                "vocal_coverage": "definite_vocal_complete",
                "vocal_purity": "definite_vocal_excludes_separable_background",
                "non_vocal_safety": "definite_non_vocal_clean",
                "envelope_structure": "event_envelopes_continuous",
                "approved": True,
                "training_manifest_allowed": True,
            }
        ],
    )
    summary = compile_canonical(
        manifest=manifest,
        preaudit=preaudit,
        manual_verdicts=verdicts,
        output_dir=tmp_path / "canonical",
    )
    assert summary["training_manifest_allowed"] is True
    row = json.loads(
        (tmp_path / "canonical" / "canonical_sources.jsonl").read_text().splitlines()[0]
    )
    assert row["canonical_spans_source"] == "manual_corrected"
    assert row["corrected_audit_manifest_sha256"] == "b" * 64
    assert [(s["label"], s["start_frame"], s["end_frame"]) for s in row["canonical_spans"]] == [
        ("vocal_candidate", 0, 2),
        ("non_vocal_candidate", 2, 5),
    ]


def test_calibration_exposes_frozen_manual_correction(tmp_path: Path) -> None:
    source = {
        "source_id": "source-1",
        "video_id": "video-1",
        "partition": "train",
        "core_ids": ["core-1"],
        "audio_sha256": "1" * 64,
        "duration_s": 0.1,
        "frame_count": 5,
        "sample_rate": 16000,
        "sample_count": 1600,
    }
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [source])
    evidence = {
        "schema": VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        **CALIBRATION_TEACHER_CONTRACT,
        **source,
        "source_manifest_sha256": _sha(manifest),
        "vocal_spans": [
            {"label": "vocal_candidate", "start_frame": 0, "end_frame": 3}
        ],
        "non_vocal_spans": [
            {"label": "non_vocal_candidate", "start_frame": 3, "end_frame": 5}
        ],
        "unsure_spans": [],
    }
    preaudit = tmp_path / "preaudit.jsonl"
    _write_jsonl(preaudit, [evidence])
    corrected = [
        {"label": "vocal_candidate", "start_frame": 0, "end_frame": 2},
        {"label": "non_vocal_candidate", "start_frame": 2, "end_frame": 5},
    ]
    verdicts = tmp_path / "verdicts.jsonl"
    signature = vocal_envelope_v12_corrected_span_signature(
        corrected, frame_count=5, source_id="source-1"
    )
    _write_jsonl(
        verdicts,
        [
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
                "source_id": "source-1",
                "video_id": "video-1",
                "partition": "train",
                "audio_sha256": source["audio_sha256"],
                "duration_s": 0.1,
                "frame_count": 5,
                "source_manifest_sha256": _sha(manifest),
                "preaudit_sha256": _sha(preaudit),
                "audit_manifest_sha256": "c" * 64,
                "corrected_spans": corrected,
                "corrected_span_signature": signature,
                "reviewed_full_source": True,
                "vocal_coverage": "definite_vocal_complete",
                "vocal_purity": "definite_vocal_excludes_separable_background",
                "non_vocal_safety": "definite_non_vocal_clean",
                "envelope_structure": "event_envelopes_continuous",
                "approved": True,
                "training_manifest_allowed": True,
            }
        ],
    )
    calibration = load_approved_calibration(
        manifest=manifest,
        preaudit=preaudit,
        verdicts=verdicts,
        expected_hashes={
            "manifest": _sha(manifest),
            "preaudit": _sha(preaudit),
            "verdicts": _sha(verdicts),
        },
    )
    assert calibration["corrected_signatures"] == {"source-1": signature}
    assert calibration["corrected_spans"]["source-1"][0]["end_frame"] == 2
