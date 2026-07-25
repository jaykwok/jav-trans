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
    TRISTATE_SYSTEM_PROMPT,
    _normalize_segments,
    _request_prompt,
    _validate_manifest,
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
    assert "连续的人类发声事件候选包络" in TRISTATE_SYSTEM_PROMPT
    assert "Proposal/Split" in TRISTATE_SYSTEM_PROMPT
    assert "CueQC" in TRISTATE_SYSTEM_PROMPT
    assert all(value in TRISTATE_SYSTEM_PROMPT for value in ("呻吟", "喘息", "吸气", "呼气"))
    assert "肉体撞击由人体动作产生也不等于人声" in TRISTATE_SYSTEM_PROMPT
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

    normalized = _normalize_segments(
        {
            "segments": [
                {
                    "start_ts": "00:00.000",
                    "end_ts": "00:00.021",
                    "label": "vocal_candidate",
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
            {"segments": [{"start_ts": 0.0, "end_ts": "00:00.100", "label": "vocal_candidate"}]},
            duration_s=0.1,
            frame_count=5,
        )
    with pytest.raises(ValueError, match="exceeds source duration"):
        _normalize_segments(
            {
                "segments": [
                    {"start_ts": "00:00.000", "end_ts": "00:00.120", "label": "vocal_candidate"}
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
                    {"start_ts": "00:00.000", "end_ts": "00:00.040", "label": "vocal_candidate"},
                    {"start_ts": "00:00.060", "end_ts": "00:00.100", "label": "unsure"},
                ]
            },
            duration_s=0.1,
            frame_count=5,
        )
    with pytest.raises(ValueError, match="same label"):
        _normalize_segments(
            {
                "segments": [
                    {"start_ts": "00:00.000", "end_ts": "00:00.040", "label": "vocal_candidate"},
                    {"start_ts": "00:00.040", "end_ts": "00:00.100", "label": "vocal_candidate"},
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
