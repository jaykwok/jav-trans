from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from boundary.ja.vocal_envelope_v12 import (
    VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    VOCAL_ENVELOPE_SCORER_V12_LABELS,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
    VocalEnvelopeScorerV12Network,
    build_vocal_envelope_scorer_v12_checkpoint,
    load_vocal_envelope_scorer_v12_checkpoint,
    score_vocal_envelope_source,
    vocal_envelope_v12_model_config,
)
from tools.audits.generate_vocal_envelope_scorer_v12_teacher_audit_html import (
    build as build_teacher_audit,
)
from tools.boundary.ja.compile_vocal_envelope_scorer_v12_canonical import (
    EXPECTED_EXECUTION_CONTRACT,
    EXPECTED_NONVOCAL_PROMPT_VERSION,
    EXPECTED_PROMPT_PROFILE,
    EXPECTED_PROMPT_VERSION,
    EXPECTED_PROTECT_PROMPT_VERSION,
    compile_canonical,
)
from tools.boundary.ja.label_vocal_envelope_scorer_v12_with_omni import (
    EXPECTED_MAX_TOKENS,
    EXPECTED_MODEL,
    NONVOCAL_SYSTEM_PROMPT,
    PROTECT_SYSTEM_PROMPT,
    _normalize_spans,
    _request_prompt,
    _validate_manifest,
    merge_dual_evidence,
    parse_args,
)
from tools.omni.timestamp_contract import TIMESTAMP_CONTRACT_ID


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _teacher_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "sources.jsonl"
    source_rows: list[dict] = []
    for index, partition in enumerate(("train", "val", "test")):
        audio = tmp_path / f"{partition}.wav"
        audio.write_bytes(f"audio-{partition}".encode())
        source_rows.append(
            {
                "source_id": f"source-{partition}",
                "video_id": f"video-{partition}",
                "partition": partition,
                "core_ids": [f"core-{partition}"],
                "audio": str(audio),
                "audio_sha256": _sha256(audio),
                "duration_s": 0.1,
                "frame_count": 5,
                "sample_rate": 16000,
                "sample_count": 1600,
                "source_kind": "real_full_source",
                "synthetic_composite": False,
            }
        )
    _write_jsonl(manifest, source_rows)
    manifest_sha = _sha256(manifest)
    preaudit = tmp_path / "preaudit.jsonl"
    evidence_rows: list[dict] = []
    for source in source_rows:
        evidence_rows.append(
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
                "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
                "source_id": source["source_id"],
                "video_id": source["video_id"],
                "partition": source["partition"],
                "core_ids": source["core_ids"],
                "audio_sha256": source["audio_sha256"],
                "duration_s": source["duration_s"],
                "frame_count": source["frame_count"],
                "sample_rate": source["sample_rate"],
                "sample_count": source["sample_count"],
                "model": EXPECTED_MODEL,
                "provider_profile": "gemini",
                "reasoning_effort": "medium",
                "max_tokens": EXPECTED_MAX_TOKENS,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "prompt_profile": EXPECTED_PROMPT_PROFILE,
                "prompt_version": EXPECTED_PROMPT_VERSION,
                "protect_prompt_version": EXPECTED_PROTECT_PROMPT_VERSION,
                "nonvocal_prompt_version": EXPECTED_NONVOCAL_PROMPT_VERSION,
                "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
                "teacher_execution_contract_id": EXPECTED_EXECUTION_CONTRACT,
                "source_manifest_sha256": manifest_sha,
                "teacher_failed_closed": False,
                "vocal_spans": [
                    {
                        "label": "vocal_candidate",
                        "start_frame": 0,
                        "end_frame": 3,
                    }
                ],
                "non_vocal_spans": [
                    {
                        "label": "non_vocal_candidate",
                        "start_frame": 3,
                        "end_frame": 5,
                    }
                ],
                "unsure_spans": [],
            }
        )
    _write_jsonl(preaudit, evidence_rows)
    return manifest, preaudit


def _approved_verdicts(manifest: Path, preaudit: Path) -> Path:
    manifest_sha = _sha256(manifest)
    preaudit_sha = _sha256(preaudit)
    sources = [json.loads(line) for line in manifest.read_text().splitlines()]
    verdicts = manifest.parent / "manual_verdicts.jsonl"
    _write_jsonl(
        verdicts,
        [
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
                "source_id": source["source_id"],
                "video_id": source["video_id"],
                "partition": source["partition"],
                "audio_sha256": source["audio_sha256"],
                "duration_s": source["duration_s"],
                "frame_count": source["frame_count"],
                "source_manifest_sha256": manifest_sha,
                "preaudit_sha256": preaudit_sha,
                "reviewed_full_source": True,
                "vocal_coverage": "definite_vocal_complete",
                "non_vocal_safety": "definite_non_vocal_clean",
                "envelope_structure": "event_envelopes_continuous",
                "approved": True,
                "training_manifest_allowed": True,
            }
            for source in sources
        ],
    )
    return verdicts


def test_v12_contract_is_breaking_and_runtime_is_argmax(tmp_path: Path) -> None:
    import torch

    assert VOCAL_ENVELOPE_SCORER_V12_LABELS == (
        "non_vocal_candidate",
        "vocal_candidate",
    )
    assert VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX == -100
    config = vocal_envelope_v12_model_config()
    model = VocalEnvelopeScorerV12Network(**config)
    normalization = {
        "mfcc_mean": list(config["mfcc_mean"]),
        "mfcc_std": list(config["mfcc_std"]),
    }
    checkpoint = tmp_path / "v12.pt"
    torch.save(
        build_vocal_envelope_scorer_v12_checkpoint(
            model=model,
            model_config=config,
            normalization=normalization,
        ),
        checkpoint,
    )
    bundle = load_vocal_envelope_scorer_v12_checkpoint(checkpoint, device="cpu")
    ptm = np.zeros((7, 2048), dtype=np.float32)
    mfcc = np.zeros((7, 40), dtype=np.float32)
    outputs = score_vocal_envelope_source(bundle, ptm=ptm, mfcc=mfcc)
    assert outputs.probabilities.shape == (7, 2)
    assert outputs.labels.shape == (7,)
    np.testing.assert_array_equal(outputs.labels, outputs.probabilities.argmax(axis=-1))

    legacy = tmp_path / "legacy.pt"
    torch.save({"schema": "candidate_island_scorer_v11"}, legacy)
    with pytest.raises(ValueError, match="v11/v10 checkpoints are not compatible"):
        load_vocal_envelope_scorer_v12_checkpoint(legacy, device="cpu")


def test_v12_prompts_and_timestamp_quantization_are_task_specific() -> None:
    assert "连续发声事件包络" in PROTECT_SYSTEM_PROMPT
    assert "不要按音节" in PROTECT_SYSTEM_PROMPT
    assert all(value in PROTECT_SYSTEM_PROMPT for value in ("呻吟", "喘息", "吸气", "呼气"))
    assert "无论有无词义都属于 vocal" in PROTECT_SYSTEM_PROMPT
    assert "非语义呻吟和呼吸仍是人类发声" in NONVOCAL_SYSTEM_PROMPT
    assert "明确不含任何人类声道" in NONVOCAL_SYSTEM_PROMPT
    assert "MM:SS.mmm" in PROTECT_SYSTEM_PROMPT
    args = parse_args(["--manifest", "m", "--output-dir", "o"])
    assert args.env_file == "gemini"
    request = json.loads(
        _request_prompt(
            {"source_id": "source", "duration_s": 65.153},
            pass_name="protect",
        )
    )
    assert request["duration_ts"] == "01:05.153"
    assert "duration_s" not in request

    vocal = _normalize_spans(
        {
            "vocal_spans": [
                {"start_ts": "00:00.021", "end_ts": "00:00.081"}
            ]
        },
        field="vocal_spans",
        duration_s=0.1,
        frame_count=5,
    )
    assert (vocal[0]["start_frame"], vocal[0]["end_frame"]) == (1, 5)
    nonvocal = _normalize_spans(
        {
            "non_vocal_spans": [
                {
                    "start_ts": "00:00.021",
                    "end_ts": "00:00.081",
                    "category": "silence",
                }
            ]
        },
        field="non_vocal_spans",
        duration_s=0.1,
        frame_count=5,
    )
    assert (nonvocal[0]["start_frame"], nonvocal[0]["end_frame"]) == (2, 4)
    with pytest.raises(ValueError, match="numeric seconds are rejected"):
        _normalize_spans(
            {"vocal_spans": [{"start_ts": 0.0, "end_ts": "00:00.020"}]},
            field="vocal_spans",
            duration_s=0.1,
            frame_count=5,
        )
    with pytest.raises(ValueError, match="exceeds source duration"):
        _normalize_spans(
            {
                "vocal_spans": [
                    {"start_ts": "00:00.000", "end_ts": "00:00.120"}
                ]
            },
            field="vocal_spans",
            duration_s=0.1,
            frame_count=5,
        )


def test_v12_dual_evidence_keeps_conflict_distinct_from_no_evidence() -> None:
    merged = merge_dual_evidence(
        vocal_spans=[{"start_frame": 1, "end_frame": 4}],
        non_vocal_spans=[{"start_frame": 2, "end_frame": 3}],
        frame_count=5,
    )
    assert [
        (span["start_frame"], span["end_frame"])
        for span in merged["conflict_spans"]
    ] == [(2, 3)]
    assert [
        (span["start_frame"], span["end_frame"])
        for span in merged["unsure_spans"]
    ] == [(0, 1), (2, 3), (4, 5)]


def test_v12_manifest_freezes_core_and_video_partitions(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for index, partition in enumerate(("train", "val", "test")):
        rows.append(
            {
                "source_id": f"source-{index}",
                "video_id": "shared-video" if index < 2 else "video-test",
                "partition": partition,
                "core_ids": [f"core-{index}"],
                "audio": str(audio),
                "audio_sha256": _sha256(audio),
                "duration_s": 0.1,
                "frame_count": 5,
            }
        )
    with pytest.raises(ValueError, match="video crosses partitions"):
        _validate_manifest(rows, manifest=manifest)
    rows[1]["video_id"] = "video-val"
    rows[1]["core_ids"] = rows[0]["core_ids"]
    with pytest.raises(ValueError, match="core must be present once and unique"):
        _validate_manifest(rows, manifest=manifest)


def test_v12_canonical_refuses_legacy_and_is_review_only_by_default(
    tmp_path: Path,
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    output = tmp_path / "canonical"
    summary = compile_canonical(
        manifest=manifest,
        preaudit=preaudit,
        output_dir=output,
    )
    assert summary["training_manifest_allowed"] is False
    rows = [
        json.loads(line)
        for line in (output / "canonical_sources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["training_manifest_allowed"] is False for row in rows)
    assert all(row["v11_complement_conversion"] is False for row in rows)

    legacy_rows = [json.loads(line) for line in preaudit.read_text().splitlines()]
    legacy_rows[0]["schema"] = "candidate_island_scorer_v11_dual_evidence_preaudit_v1"
    _write_jsonl(preaudit, legacy_rows)
    with pytest.raises(ValueError, match="wrong v12 preaudit schema"):
        compile_canonical(
            manifest=manifest,
            preaudit=preaudit,
            output_dir=tmp_path / "legacy-output",
        )


def test_v12_canonical_can_be_explicitly_enabled_after_external_review(
    tmp_path: Path,
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    verdicts = _approved_verdicts(manifest, preaudit)
    summary = compile_canonical(
        manifest=manifest,
        preaudit=preaudit,
        output_dir=tmp_path / "approved",
        manual_verdicts=verdicts,
    )
    assert summary["training_manifest_allowed"] is True
    assert summary["frame_counts"] == {
        "non_vocal_candidate": 6,
        "vocal_candidate": 9,
    }
    assert summary["dataset_contract"]["label_unit"] == "human_vocal_event_envelope"
    assert summary["canonical_label_schema"] == "vocal_envelope_frames_v1"
    assert VOCAL_ENVELOPE_SCORER_V12_SCHEMA == "vocal_envelope_scorer_v12"


def test_v12_canonical_rejects_unapproved_or_unbound_manual_verdicts(
    tmp_path: Path,
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    verdicts = _approved_verdicts(manifest, preaudit)
    rows = [json.loads(line) for line in verdicts.read_text().splitlines()]
    rows[0]["non_vocal_safety"] = "definite_non_vocal_contains_vocal"
    rows[0]["approved"] = False
    rows[0]["training_manifest_allowed"] = False
    _write_jsonl(verdicts, rows)
    with pytest.raises(ValueError, match="rejects canonical supervision"):
        compile_canonical(
            manifest=manifest,
            preaudit=preaudit,
            output_dir=tmp_path / "rejected",
            manual_verdicts=verdicts,
        )

    verdicts = _approved_verdicts(manifest, preaudit)
    rows = [json.loads(line) for line in verdicts.read_text().splitlines()]
    rows[0]["preaudit_sha256"] = "0" * 64
    _write_jsonl(verdicts, rows)
    with pytest.raises(ValueError, match="preaudit_sha256 mismatch"):
        compile_canonical(
            manifest=manifest,
            preaudit=preaudit,
            output_dir=tmp_path / "unbound",
            manual_verdicts=verdicts,
        )


def test_v12_teacher_audit_uses_shared_core_and_independent_span_playback(
    tmp_path: Path,
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    output = tmp_path / "audit"
    summary = build_teacher_audit(
        source_manifest=manifest,
        preaudit=preaudit,
        output_dir=output,
    )
    assert summary["source_count"] == 3
    assert summary["training_manifest_allowed"] is False
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "createAuditReviewCore" in page
    assert "canonical vocal" in page
    assert "canonical non-vocal" in page
    assert "canonical unsure" in page
    assert "颜色条点击后只播放自身精确区间" in page
    assert "definite_non_vocal_contains_vocal" in page
    assert "both_fragmented_and_overmerged" in page
    assert VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA in page
    audit_rows = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl").read_text().splitlines()
    ]
    assert all((output / row["audio"]).is_file() for row in audit_rows)
