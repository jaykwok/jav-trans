from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from boundary.ja.vocal_envelope_v12 import (
    VOCAL_ENVELOPE_SCORER_V12_CRF_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_DENSE_SPAN_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    VOCAL_ENVELOPE_SCORER_V12_LABELS,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_QUERY_MASK_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
    VocalEnvelopeScorerV12DenseSpanNetwork,
    VocalEnvelopeScorerV12CrfNetwork,
    VocalEnvelopeScorerV12QueryMaskNetwork,
    VocalEnvelopeScorerV12Network,
    build_vocal_envelope_scorer_v12_checkpoint,
    load_vocal_envelope_scorer_v12_checkpoint,
    score_vocal_envelope_source,
    vocal_envelope_v12_crf_model_config,
    vocal_envelope_v12_dense_span_model_config,
    vocal_envelope_v12_model_config,
    vocal_envelope_v12_query_mask_model_config,
)
from tools.audits.generate_vocal_envelope_scorer_v12_teacher_audit_html import (
    build as build_teacher_audit,
)
from tools.boundary.ja.compile_vocal_envelope_scorer_v12_canonical import (
    EXPECTED_PROMPT_PROFILE,
    EXPECTED_PROMPT_VERSION,
    PROVIDER_CONTRACTS as COMPILER_PROVIDER_CONTRACTS,
    compile_canonical,
)
from tools.boundary.ja.label_vocal_envelope_scorer_v12_with_omni import (
    EXPECTED_MAX_TOKENS,
    PROVIDER_CONTRACTS,
    TRISTATE_RESPONSE_SCHEMA,
    TRISTATE_SYSTEM_PROMPT,
    _normalize_segments,
    _request_prompt,
    _validate_manifest,
    parse_args,
    run as run_teacher,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_teacher_contract import (
    RESPONSE_SCHEMA_SHA256,
    SYSTEM_PROMPT_SHA256,
    TEACHER_TASK_CONTRACT_ID,
    teacher_contract_fingerprint_fields,
    validate_teacher_contract_content,
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
    profile = "openrouter"
    provider_contract = PROVIDER_CONTRACTS[profile]
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
                "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
                "source_id": source["source_id"],
                "video_id": source["video_id"],
                "partition": source["partition"],
                "core_ids": source["core_ids"],
                "audio_sha256": source["audio_sha256"],
                "duration_s": source["duration_s"],
                "frame_count": source["frame_count"],
                "sample_rate": source["sample_rate"],
                "sample_count": source["sample_count"],
                "model": provider_contract["model"],
                "provider_profile": profile,
                "env_file_name": profile,
                "reasoning_effort": "medium",
                "max_tokens": EXPECTED_MAX_TOKENS,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "prompt_profile": EXPECTED_PROMPT_PROFILE,
                "prompt_version": EXPECTED_PROMPT_VERSION,
                **teacher_contract_fingerprint_fields(),
                "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
                "teacher_execution_contract_id": provider_contract[
                    "execution_contract"
                ],
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
                "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
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
                "vocal_purity": "definite_vocal_excludes_separable_nonvoice",
                "non_vocal_safety": "definite_non_vocal_clean",
                "envelope_structure": "event_envelopes_continuous",
                "approved": True,
                "training_manifest_allowed": True,
            }
            for source in sources
        ],
    )
    return verdicts


def _set_native_gemini_profile(preaudit: Path) -> None:
    native = COMPILER_PROVIDER_CONTRACTS["gemini"]
    rows = [json.loads(line) for line in preaudit.read_text().splitlines()]
    for row in rows:
        row["model"] = native["model"]
        row["provider_profile"] = "gemini"
        row["env_file_name"] = "gemini"
        row["teacher_execution_contract_id"] = native["execution_contract"]
    _write_jsonl(preaudit, rows)


def _approved_verdicts_for_ids(
    manifest: Path, preaudit: Path, source_ids: set[str], output: Path
) -> Path:
    manifest_sha = _sha256(manifest)
    preaudit_sha = _sha256(preaudit)
    sources = {
        row["source_id"]: row
        for row in (
            json.loads(line) for line in manifest.read_text().splitlines()
        )
    }
    _write_jsonl(
        output,
        [
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
                "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
                "source_id": source_id,
                "video_id": sources[source_id]["video_id"],
                "partition": sources[source_id]["partition"],
                "audio_sha256": sources[source_id]["audio_sha256"],
                "duration_s": sources[source_id]["duration_s"],
                "frame_count": sources[source_id]["frame_count"],
                "source_manifest_sha256": manifest_sha,
                "preaudit_sha256": preaudit_sha,
                "reviewed_full_source": True,
                "vocal_coverage": "definite_vocal_complete",
                "vocal_purity": "definite_vocal_excludes_separable_nonvoice",
                "non_vocal_safety": "definite_non_vocal_clean",
                "envelope_structure": "event_envelopes_continuous",
                "approved": True,
                "training_manifest_allowed": True,
            }
            for source_id in sorted(source_ids)
        ],
    )
    return output


def _calibration_subset(
    manifest: Path,
    preaudit: Path,
    *,
    source_ids: set[str],
    output_dir: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    output_dir.mkdir()
    source_rows = [
        json.loads(line) for line in manifest.read_text().splitlines()
    ]
    evidence_rows = [
        json.loads(line) for line in preaudit.read_text().splitlines()
    ]
    calibration_manifest = output_dir / "manifest.jsonl"
    _write_jsonl(
        calibration_manifest,
        [row for row in source_rows if row["source_id"] in source_ids],
    )
    calibration_manifest_sha = _sha256(calibration_manifest)
    selected_evidence = [
        row for row in evidence_rows if row["source_id"] in source_ids
    ]
    for row in selected_evidence:
        row["source_manifest_sha256"] = calibration_manifest_sha
    calibration_preaudit = output_dir / "preaudit.jsonl"
    _write_jsonl(calibration_preaudit, selected_evidence)
    calibration_verdicts = _approved_verdicts_for_ids(
        calibration_manifest,
        calibration_preaudit,
        source_ids,
        output_dir / "manual_verdicts.jsonl",
    )
    hashes = {
        "manifest": _sha256(calibration_manifest),
        "preaudit": _sha256(calibration_preaudit),
        "verdicts": _sha256(calibration_verdicts),
    }
    return (
        calibration_manifest,
        calibration_preaudit,
        calibration_verdicts,
        hashes,
    )


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

    retired_v12 = tmp_path / "retired-v12.pt"
    retired_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    retired_payload["metadata"]["dataset_contract"]["label_unit"] = (
        "human_vocal_event_envelope"
    )
    torch.save(retired_payload, retired_v12)
    with pytest.raises(ValueError, match="dataset contract mismatch"):
        load_vocal_envelope_scorer_v12_checkpoint(retired_v12, device="cpu")


def test_v12_crf_checkpoint_uses_exact_viterbi_runtime(tmp_path: Path) -> None:
    import torch

    config = vocal_envelope_v12_crf_model_config()
    model = VocalEnvelopeScorerV12CrfNetwork(**config)
    with torch.no_grad():
        model.crf.transitions.copy_(torch.tensor([[2.0, -2.0], [-2.0, 2.0]]))
    checkpoint = tmp_path / "v12-crf.pt"
    torch.save(
        build_vocal_envelope_scorer_v12_checkpoint(
            model=model,
            model_config=config,
            normalization={
                "mfcc_mean": list(config["mfcc_mean"]),
                "mfcc_std": list(config["mfcc_std"]),
            },
        ),
        checkpoint,
    )
    bundle = load_vocal_envelope_scorer_v12_checkpoint(checkpoint, device="cpu")
    outputs = score_vocal_envelope_source(
        bundle,
        ptm=np.zeros((7, 2048), dtype=np.float32),
        mfcc=np.zeros((7, 40), dtype=np.float32),
    )
    assert bundle.schema == VOCAL_ENVELOPE_SCORER_V12_CRF_SCHEMA
    assert bundle.metadata["decision_mode"] == (
        "learned_binary_sequence_viterbi_argmax"
    )
    assert outputs.probabilities.shape == (7, 2)
    assert outputs.labels.shape == (7,)


@pytest.mark.parametrize(
    ("schema", "config_factory", "network_factory", "decision_mode"),
    (
        (
            VOCAL_ENVELOPE_SCORER_V12_QUERY_MASK_SCHEMA,
            vocal_envelope_v12_query_mask_model_config,
            VocalEnvelopeScorerV12QueryMaskNetwork,
            "binary_frame_argmax_after_differentiable_query_fusion",
        ),
        (
            VOCAL_ENVELOPE_SCORER_V12_DENSE_SPAN_SCHEMA,
            vocal_envelope_v12_dense_span_model_config,
            VocalEnvelopeScorerV12DenseSpanNetwork,
            "learned_binary_dense_span_viterbi_argmax",
        ),
    ),
)
def test_v12_structured_checkpoint_schemas_are_runtime_separate(
    tmp_path: Path, schema, config_factory, network_factory, decision_mode
) -> None:
    import torch

    config = config_factory()
    model = network_factory(**config)
    checkpoint = tmp_path / f"{schema}.pt"
    torch.save(
        build_vocal_envelope_scorer_v12_checkpoint(
            model=model,
            model_config=config,
            normalization={
                "mfcc_mean": list(config["mfcc_mean"]),
                "mfcc_std": list(config["mfcc_std"]),
            },
        ),
        checkpoint,
    )
    bundle = load_vocal_envelope_scorer_v12_checkpoint(checkpoint, device="cpu")
    outputs = score_vocal_envelope_source(
        bundle,
        ptm=np.zeros((7, 2048), dtype=np.float32),
        mfcc=np.zeros((7, 40), dtype=np.float32),
    )
    assert bundle.schema == schema
    assert bundle.metadata["decision_mode"] == decision_mode
    assert outputs.probabilities.shape == (7, 2)
    assert outputs.labels.shape == (7,)


def test_v12_prompts_and_timestamp_quantization_are_task_specific() -> None:
    assert "连续的人类声音事件候选包络" in TRISTATE_SYSTEM_PROMPT
    assert "Proposal/Split" in TRISTATE_SYSTEM_PROMPT
    assert "CueQC" in TRISTATE_SYSTEM_PROMPT
    assert all(value in TRISTATE_SYSTEM_PROMPT for value in ("呻吟", "喘息", "吸气", "呼气"))
    assert "声音来自人体并不等于人声" in TRISTATE_SYSTEM_PROMPT
    assert "完整音频的 segments" in TRISTATE_SYSTEM_PROMPT
    assert "MM:SS.mmm" in TRISTATE_SYSTEM_PROMPT
    args = parse_args(["--manifest", "m", "--output-dir", "o"])
    assert args.env_file == "gemini"
    assert set(PROVIDER_CONTRACTS) == {"openrouter", "gemini"}
    for profile, contract in COMPILER_PROVIDER_CONTRACTS.items():
        assert PROVIDER_CONTRACTS[profile]["model"] == contract["model"]
        assert (
            PROVIDER_CONTRACTS[profile]["execution_contract"]
            == contract["execution_contract"]
        )
    request = json.loads(
        _request_prompt(
            {"source_id": "source", "duration_s": 65.153},
        )
    )
    assert request["duration_ts"] == "01:05.153"
    assert "duration_s" not in request
    assert request["task"] == TEACHER_TASK_CONTRACT_ID

    normalized = _normalize_segments(
        {
            "segments": [
                {
                    "start_ts": "00:00.000",
                    "end_ts": "00:00.021",
                    "label": "vocal_candidate",
                    "category": "speech",
                },
                {
                    "start_ts": "00:00.021",
                    "end_ts": "00:00.081",
                    "label": "non_vocal_candidate",
                    "category": "impact",
                },
                {
                    "start_ts": "00:00.081",
                    "end_ts": "00:00.100",
                    "label": "unsure",
                    "category": "uncertain",
                },
            ]
        },
        duration_s=0.1,
        frame_count=5,
    )
    assert (normalized["vocal_spans"][0]["start_frame"], normalized["vocal_spans"][0]["end_frame"]) == (0, 2)
    assert (normalized["non_vocal_spans"][0]["start_frame"], normalized["non_vocal_spans"][0]["end_frame"]) == (2, 4)
    assert (normalized["unsure_spans"][0]["start_frame"], normalized["unsure_spans"][0]["end_frame"]) == (4, 5)
    with pytest.raises(ValueError, match="numeric seconds are rejected"):
        _normalize_segments(
            {"segments": [{"start_ts": 0.0, "end_ts": "00:00.100", "label": "vocal_candidate", "category": "speech"}]},
            duration_s=0.1,
            frame_count=5,
        )
    with pytest.raises(ValueError, match="exceeds source duration"):
        _normalize_segments(
            {
                "segments": [
                    {"start_ts": "00:00.000", "end_ts": "00:00.120", "label": "vocal_candidate", "category": "speech"}
                ]
            },
            duration_s=0.1,
            frame_count=5,
        )


def test_v12_teacher_prompt_and_schema_content_are_fingerprint_bound() -> None:
    expected = {
        "teacher_task_contract_id": TEACHER_TASK_CONTRACT_ID,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
    }
    assert validate_teacher_contract_content(
        system_prompt=TRISTATE_SYSTEM_PROMPT,
        response_schema=TRISTATE_RESPONSE_SCHEMA,
    ) == expected

    with pytest.raises(ValueError, match="system prompt fingerprint mismatch"):
        validate_teacher_contract_content(
            system_prompt=TRISTATE_SYSTEM_PROMPT + "\n正文漂移",
            response_schema=TRISTATE_RESPONSE_SCHEMA,
        )

    changed_schema = json.loads(json.dumps(TRISTATE_RESPONSE_SCHEMA))
    changed_schema["properties"]["overall_reason"]["maxLength"] = 10
    with pytest.raises(ValueError, match="response schema fingerprint mismatch"):
        validate_teacher_contract_content(
            system_prompt=TRISTATE_SYSTEM_PROMPT,
            response_schema=changed_schema,
        )


@pytest.mark.parametrize("category", ("speech", "whisper_language", "moan", "voiced_vocalization"))
def test_v12_voice_categories_are_positive(category: str) -> None:
    normalized = _normalize_segments(
        {
            "segments": [
                {
                    "start_ts": "00:00.000",
                    "end_ts": "00:00.100",
                    "label": "vocal_candidate",
                    "category": category,
                }
            ]
        },
        duration_s=0.1,
        frame_count=5,
    )
    assert normalized["vocal_spans"][0]["start_frame"] == 0
    assert normalized["vocal_spans"][0]["end_frame"] == 5
    assert normalized["vocal_spans"][0]["category"] == category


@pytest.mark.parametrize(
    "category",
    ("breath_airflow", "pant_airflow", "kiss", "oral_action", "swallow", "cough"),
)
def test_v12_nonvoice_human_sounds_are_negative(category: str) -> None:
    normalized = _normalize_segments(
        {
            "segments": [
                {
                    "start_ts": "00:00.000",
                    "end_ts": "00:00.100",
                    "label": "non_vocal_candidate",
                    "category": category,
                }
            ]
        },
        duration_s=0.1,
        frame_count=5,
    )
    assert normalized["non_vocal_spans"][0]["start_frame"] == 0
    assert normalized["non_vocal_spans"][0]["end_frame"] == 5
    assert normalized["non_vocal_spans"][0]["category"] == category

    with pytest.raises(ValueError, match="unsupported vocal category"):
        _normalize_segments(
            {
                "segments": [
                    {
                        "start_ts": "00:00.000",
                        "end_ts": "00:00.100",
                        "label": "vocal_candidate",
                        "category": category,
                    }
                ]
            },
            duration_s=0.1,
            frame_count=5,
        )


def test_v12_single_pass_rejects_gaps_and_adjacent_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _normalize_segments(
            {
                "segments": [
                    {"start_ts": "00:00.000", "end_ts": "00:00.040", "label": "vocal_candidate", "category": "speech"},
                    {"start_ts": "00:00.060", "end_ts": "00:00.100", "label": "unsure", "category": "uncertain"},
                ]
            },
            duration_s=0.1,
            frame_count=5,
        )
    with pytest.raises(ValueError, match="same label"):
        _normalize_segments(
            {
                "segments": [
                    {"start_ts": "00:00.000", "end_ts": "00:00.040", "label": "vocal_candidate", "category": "speech"},
                    {"start_ts": "00:00.040", "end_ts": "00:00.100", "label": "vocal_candidate", "category": "speech"},
                ]
            },
            duration_s=0.1,
            frame_count=5,
        )


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


def test_v12_canonical_accepts_one_consistent_native_gemini_profile(
    tmp_path: Path,
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    native = COMPILER_PROVIDER_CONTRACTS["gemini"]
    rows = [json.loads(line) for line in preaudit.read_text().splitlines()]
    for row in rows:
        row["model"] = native["model"]
        row["provider_profile"] = "gemini"
        row["env_file_name"] = "gemini"
        row["teacher_execution_contract_id"] = native["execution_contract"]
    _write_jsonl(preaudit, rows)
    summary = compile_canonical(
        manifest=manifest,
        preaudit=preaudit,
        output_dir=tmp_path / "native-review-only",
    )
    assert summary["provider_profile"] == "gemini"
    assert summary["training_manifest_allowed"] is False


def test_v12_canonical_rejects_teacher_content_fingerprint_drift(
    tmp_path: Path,
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    rows = [json.loads(line) for line in preaudit.read_text().splitlines()]
    rows[0]["system_prompt_sha256"] = "0" * 64
    _write_jsonl(preaudit, rows)

    with pytest.raises(ValueError, match="system_prompt_sha256 mismatch"):
        compile_canonical(
            manifest=manifest,
            preaudit=preaudit,
            output_dir=tmp_path / "prompt-drift",
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
    assert summary["dataset_contract"]["label_unit"] == "human_voice_event_envelope"
    assert summary["canonical_label_schema"] == "vocal_envelope_frames_v2"
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


def test_v12_calibrated_full_canonical_allows_train_teacher_only(
    tmp_path: Path,
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    _set_native_gemini_profile(preaudit)
    (
        calibration_manifest,
        calibration_preaudit,
        calibration_verdicts,
        calibration_hashes,
    ) = _calibration_subset(
        manifest,
        preaudit,
        source_ids={"source-val"},
        output_dir=tmp_path / "calibration",
    )
    heldout_verdicts = _approved_verdicts_for_ids(
        manifest,
        preaudit,
        {"source-test"},
        tmp_path / "heldout-verdicts.jsonl",
    )

    summary = compile_canonical(
        manifest=manifest,
        preaudit=preaudit,
        output_dir=tmp_path / "calibrated-full",
        manual_verdicts=heldout_verdicts,
        calibration_manifest=calibration_manifest,
        calibration_preaudit=calibration_preaudit,
        calibration_verdicts=calibration_verdicts,
        calibration_expected_hashes=calibration_hashes,
    )

    assert summary["training_manifest_allowed"] is True
    assert summary["heldout_human_full_source_review_approved"] is True
    assert summary["human_full_source_review_approved"] is False
    assert summary["calibration_source_count"] == 1
    rows = {
        row["source_id"]: row
        for row in (
            json.loads(line)
            for line in (tmp_path / "calibrated-full" / "canonical_sources.jsonl")
            .read_text()
            .splitlines()
        )
    }
    assert rows["source-train"]["calibrated_train_supervision"] is True
    assert rows["source-train"]["teacher_output_used_as_truth"] is False
    assert rows["source-train"]["training_manifest_allowed"] is True
    assert rows["source-val"]["calibration_overlap_human_approved"] is True
    assert rows["source-test"]["human_full_source_review_approved"] is True


def test_v12_calibrated_full_canonical_requires_remaining_heldout_review(
    tmp_path: Path,
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    _set_native_gemini_profile(preaudit)
    (
        calibration_manifest,
        calibration_preaudit,
        calibration_verdicts,
        calibration_hashes,
    ) = _calibration_subset(
        manifest,
        preaudit,
        source_ids={"source-val"},
        output_dir=tmp_path / "calibration",
    )

    with pytest.raises(ValueError, match="non-pilot heldout sources"):
        compile_canonical(
            manifest=manifest,
            preaudit=preaudit,
            output_dir=tmp_path / "missing-heldout",
            calibration_manifest=calibration_manifest,
            calibration_preaudit=calibration_preaudit,
            calibration_verdicts=calibration_verdicts,
            calibration_expected_hashes=calibration_hashes,
        )


def test_v12_calibrated_full_canonical_rejects_changed_pilot_evidence(
    tmp_path: Path,
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    _set_native_gemini_profile(preaudit)
    (
        calibration_manifest,
        calibration_preaudit,
        calibration_verdicts,
        calibration_hashes,
    ) = _calibration_subset(
        manifest,
        preaudit,
        source_ids={"source-val"},
        output_dir=tmp_path / "calibration",
    )
    rows = [json.loads(line) for line in preaudit.read_text().splitlines()]
    val = next(row for row in rows if row["source_id"] == "source-val")
    val["vocal_spans"][0]["end_frame"] = 2
    val["non_vocal_spans"][0]["start_frame"] = 2
    _write_jsonl(preaudit, rows)
    heldout_verdicts = _approved_verdicts_for_ids(
        manifest,
        preaudit,
        {"source-test"},
        tmp_path / "heldout-verdicts.jsonl",
    )

    with pytest.raises(ValueError, match="evidence changed after approval"):
        compile_canonical(
            manifest=manifest,
            preaudit=preaudit,
            output_dir=tmp_path / "drifted-pilot",
            manual_verdicts=heldout_verdicts,
            calibration_manifest=calibration_manifest,
            calibration_preaudit=calibration_preaudit,
            calibration_verdicts=calibration_verdicts,
            calibration_expected_hashes=calibration_hashes,
        )


def test_v12_teacher_can_seed_approved_calibration_without_api_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    _set_native_gemini_profile(preaudit)
    (
        calibration_manifest,
        calibration_preaudit,
        calibration_verdicts,
        calibration_hashes,
    ) = _calibration_subset(
        manifest,
        preaudit,
        source_ids={"source-val"},
        output_dir=tmp_path / "calibration",
    )

    class FakeTransport:
        model = COMPILER_PROVIDER_CONTRACTS["gemini"]["model"]
        execution_contract = COMPILER_PROVIDER_CONTRACTS["gemini"][
            "execution_contract"
        ]
        transport_name = "google_ai_interactions_inline_audio"
        api_key_count = 1
        max_concurrency = 1

    monkeypatch.setattr(
        "tools.boundary.ja.label_vocal_envelope_scorer_v12_with_omni."
        "CALIBRATION_ARTIFACT_SHA256",
        calibration_hashes,
    )
    monkeypatch.setattr(
        "tools.boundary.ja.label_vocal_envelope_scorer_v12_with_omni."
        "create_audio_teacher_transport",
        lambda **_kwargs: FakeTransport(),
    )
    args = parse_args(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "seeded"),
            "--source-id",
            "source-val",
            "--calibration-manifest",
            str(calibration_manifest),
            "--calibration-preaudit",
            str(calibration_preaudit),
            "--calibration-verdicts",
            str(calibration_verdicts),
        ]
    )
    summary = run_teacher(args)

    assert summary["request_count"] == 0
    assert summary["calibration_seed_count"] == 1
    seeded = json.loads(
        (tmp_path / "seeded" / "preaudit.jsonl").read_text().strip()
    )
    assert seeded["source_manifest_sha256"] == _sha256(manifest)
    assert seeded["calibration_preaudit_sha256"] == calibration_hashes["preaudit"]
    progress = json.loads(
        (tmp_path / "seeded" / "progress.json").read_text(encoding="utf-8")
    )
    for field, expected in teacher_contract_fingerprint_fields().items():
        assert seeded[field] == expected
        assert summary[field] == expected
        assert progress[field] == expected


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
    assert summary["schema"] == "vocal_envelope_scorer_v12_teacher_audit_summary_v2"
    assert summary["task_semantics"] == VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "createAuditReviewCore" in page
    assert "Human Voice Envelope Teacher review" in page
    assert "canonical vocal" in page
    assert "canonical non-vocal" in page
    assert "canonical unsure" in page
    assert "颜色条点击后只播放自身精确区间" in page
    assert "纯呼吸气流、无声喘气、亲吻、吞咽" in page
    assert "带声呻吟" in page
    assert "definite_vocal_excludes_separable_nonvoice" in page
    assert "definite_vocal_contains_separable_nonvoice" in page
    assert "definite_non_vocal_contains_vocal" in page
    assert "both_fragmented_and_overmerged" in page
    assert "vocal-envelope-scorer-v12-teacher-audit-v2" in page
    assert "location.pathname" in page
    assert summary["source_manifest_sha256"] in page
    assert summary["preaudit_sha256"] in page
    assert summary["audit_manifest_sha256"] in page
    assert "vocal-envelope-scorer-v12-teacher-audit-v1" not in page
    assert VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA in page
    audit_rows = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl").read_text().splitlines()
    ]
    assert all((output / row["audio"]).is_file() for row in audit_rows)
    assert all(
        row["task_semantics"] == VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS
        and row["source_manifest_sha256"] == summary["source_manifest_sha256"]
        and row["preaudit_sha256"] == summary["preaudit_sha256"]
        and row["evidence_span_signature"]
        for row in audit_rows
    )


def test_v12_teacher_audit_rejects_wrong_task_semantics(tmp_path: Path) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    rows = [json.loads(line) for line in preaudit.read_text().splitlines()]
    rows[0]["task_semantics"] = "legacy_semantic_candidate"
    _write_jsonl(preaudit, rows)

    with pytest.raises(ValueError, match="wrong v12 task semantics"):
        build_teacher_audit(
            source_manifest=manifest,
            preaudit=preaudit,
            output_dir=tmp_path / "audit",
        )


def test_v12_teacher_audit_can_skip_exact_approved_calibration_heldout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, preaudit = _teacher_fixture(tmp_path)
    _set_native_gemini_profile(preaudit)
    (
        calibration_manifest,
        calibration_preaudit,
        calibration_verdicts,
        calibration_hashes,
    ) = _calibration_subset(
        manifest,
        preaudit,
        source_ids={"source-val"},
        output_dir=tmp_path / "calibration",
    )
    monkeypatch.setattr(
        "tools.audits.generate_vocal_envelope_scorer_v12_teacher_audit_html."
        "CALIBRATION_ARTIFACT_SHA256",
        calibration_hashes,
    )
    output = tmp_path / "heldout-audit"
    summary = build_teacher_audit(
        source_manifest=manifest,
        preaudit=preaudit,
        output_dir=output,
        partitions=("val", "test"),
        calibration_manifest=calibration_manifest,
        calibration_preaudit=calibration_preaudit,
        calibration_verdicts=calibration_verdicts,
    )

    assert summary["source_count"] == 1
    assert summary["skipped_calibration_source_ids"] == ["source-val"]
    audit_rows = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl").read_text().splitlines()
    ]
    assert [row["source_id"] for row in audit_rows] == ["source-test"]
    page = (output / "index.html").read_text(encoding="utf-8")
    assert _sha256(manifest) in page
    assert _sha256(preaudit) in page
