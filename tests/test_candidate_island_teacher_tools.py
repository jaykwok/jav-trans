from __future__ import annotations

import json
import argparse
from pathlib import Path

from tools.audits.compare_candidate_island_teacher_to_human import compare
from tools.audits.generate_candidate_island_teacher_comparison_html import _audio_path
from tools.asr.cueqc.label_pre_asr_with_omni import normalize_openai_compat_base_url
from tools.boundary.ja.label_candidate_island_scorer_v11_with_omni import (
    ASSERTIVE_SAFE_OUTSIDE_PROMPT_PROFILE,
    ASSERTIVE_SAFE_OUTSIDE_PROMPT_VERSION,
    ASSERTIVE_SAFE_OUTSIDE_SYSTEM_PROMPT,
    BALANCED_V12_SAFE_OUTSIDE_PROMPT_PROFILE,
    BALANCED_V12_SAFE_OUTSIDE_PROMPT_VERSION,
    BALANCED_V12_SAFE_OUTSIDE_SYSTEM_PROMPT,
    CUSTOM_SAFE_OUTSIDE_PROMPT_PROFILE,
    FUNNEL_SAFE_OUTSIDE_PROMPT_PROFILE,
    FUNNEL_SAFE_OUTSIDE_PROMPT_VERSION,
    FUNNEL_SAFE_OUTSIDE_SYSTEM_PROMPT,
    GREENLIGHT_SAFE_OUTSIDE_PROMPT_PROFILE,
    GREENLIGHT_SAFE_OUTSIDE_PROMPT_VERSION,
    GREENLIGHT_SAFE_OUTSIDE_SYSTEM_PROMPT,
    SAFE_OUTSIDE_PROMPT_PROFILE,
    SAFE_OUTSIDE_PROMPT_VERSION,
    SAFE_OUTSIDE_SYSTEM_PROMPT,
    SIMPLE_SAFE_OUTSIDE_PROMPT_PROFILE,
    SIMPLE_SAFE_OUTSIDE_PROMPT_VERSION,
    SIMPLE_SAFE_OUTSIDE_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    _prompt,
    _resume_index,
    _safe_outside_complement,
    _spans,
    parse_args,
)
from tools.boundary.ja.build_candidate_island_scorer_v11_outside_consensus import (
    build as build_outside_consensus,
)
from tools.boundary.ja.build_candidate_island_scorer_v11_train_teacher_manifest import (
    build as build_train_teacher_manifest,
)
from tools.boundary.ja.build_candidate_island_scorer_v11_real_outside_selection import (
    build as build_real_outside_selection,
)
from tools.boundary.ja.compile_candidate_island_scorer_v11_real_train_outside import (
    build as compile_real_train_outside,
)
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.omni.run_audio_teacher import parse_args as parse_generic_args


def test_provider_base_url_normalization_and_known_profiles() -> None:
    assert normalize_openai_compat_base_url("https://openrouter.ai/api/v1/chat/completions") == "https://openrouter.ai/api/v1"
    assert parse_args(["--manifest", "x", "--output-dir", "y"]).env_file == "gemini"
    assert parse_args(["--manifest", "x", "--output-dir", "y", "--env-file", "qwen"]).env_file == "qwen"
    assert parse_generic_args(["--output-dir", "y", "--prompt", "x"]).env_file == "gemini"
    assert parse_generic_args(["--output-dir", "y", "--env-file", "qwen", "--prompt", "x"]).env_file == "qwen"


def test_candidate_island_teacher_prompt_matches_scorer_responsibility() -> None:
    assert PROMPT_VERSION.endswith("dialogue_islands_v5")
    assert "不是 Split" in SYSTEM_PROMPT
    assert "不是 CueQC" in SYSTEM_PROMPT
    assert "若整条 source 都是明确的纯非语义声音，必须允许 islands=[]" in SYSTEM_PROMPT
    assert "优先标为 unsure" in SYSTEM_PROMPT
    assert "同一场景、同一说话人、持续互动或声音连续，本身都不是合并理由" in SYSTEM_PROMPT
    assert "ASR 能否转录" in SYSTEM_PROMPT
    assert "固定时长" in SYSTEM_PROMPT
    payload = json.loads(_prompt({"source_id": "s", "duration_s": 75.0}))
    assert "target_language" not in payload
    assert "target_domain" not in payload
    assert "domain_rule" not in payload
    assert payload["anti_overmerge"].endswith("return islands=[]")
    assert "unsure" in payload["decision_order"][-1]


def test_safe_outside_prompt_is_precision_first_and_keeps_wordlike_moans() -> None:
    args = parse_args(
        [
            "--manifest",
            "x",
            "--output-dir",
            "y",
            "--prompt-profile",
            SAFE_OUTSIDE_PROMPT_PROFILE,
        ]
    )
    assert args.prompt_profile == SAFE_OUTSIDE_PROMPT_PROFILE
    assert SAFE_OUTSIDE_PROMPT_VERSION.endswith("safe_outside_complement_v1")
    assert "环境背景噪声" in SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "啊、嗯、哼、哈、诶" in SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "あ、あっ、うん、ん、ふん、え、はぁ" in SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "背景中的人声只要可能含词" in SAFE_OUTSIDE_SYSTEM_PROMPT
    payload = json.loads(
        _prompt(
            {"source_id": "s", "duration_s": 75.0},
            prompt_profile=SAFE_OUTSIDE_PROMPT_PROFILE,
        )
    )
    assert payload["uncertainty_policy"].startswith("omit")
    assert "environmental background noise" in payload["nonlexical_examples_to_consider"][1]
    assert "provisional keep" in payload["uncertainty_policy"]


def test_simple_safe_outside_prompt_keeps_the_request_compact() -> None:
    args = parse_args(
        [
            "--manifest",
            "x",
            "--output-dir",
            "y",
            "--prompt-profile",
            SIMPLE_SAFE_OUTSIDE_PROMPT_PROFILE,
        ]
    )
    assert args.prompt_profile == SIMPLE_SAFE_OUTSIDE_PROMPT_PROFILE
    assert SIMPLE_SAFE_OUTSIDE_PROMPT_VERSION.endswith(
        "safe_outside_complement_v2_simple"
    )
    assert "纯环境与物理音" in SIMPLE_SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "独立且连续的非语言生理音" in SIMPLE_SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "0.2～0.3 秒" in SIMPLE_SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "あ、あっ、うん、ん、ふん、え、はぁ" in SIMPLE_SAFE_OUTSIDE_SYSTEM_PROMPT
    payload = json.loads(
        _prompt(
            {"source_id": "s", "duration_s": 75.0},
            prompt_profile=SIMPLE_SAFE_OUTSIDE_PROMPT_PROFILE,
        )
    )
    assert payload == {
        "source_id": "s",
        "duration_s": 75.0,
        "coordinate_system": "0-based current full-source timeline in seconds",
    }


def test_greenlight_and_funnel_safe_outside_prompts_are_distinct_profiles() -> None:
    for profile, version, system_prompt, marker in (
        (
            GREENLIGHT_SAFE_OUTSIDE_PROMPT_PROFILE,
            GREENLIGHT_SAFE_OUTSIDE_PROMPT_VERSION,
            GREENLIGHT_SAFE_OUTSIDE_SYSTEM_PROMPT,
            "必须标记为 Outside 的情况（绿灯）",
        ),
        (
            FUNNEL_SAFE_OUTSIDE_PROMPT_PROFILE,
            FUNNEL_SAFE_OUTSIDE_PROMPT_VERSION,
            FUNNEL_SAFE_OUTSIDE_SYSTEM_PROMPT,
            "判定漏斗",
        ),
    ):
        args = parse_args(
            [
                "--manifest",
                "x",
                "--output-dir",
                "y",
                "--prompt-profile",
                profile,
            ]
        )
        assert args.prompt_profile == profile
        assert version.endswith(profile.replace("-", "_"))
        assert marker in system_prompt
        assert "safe_outside_spans" in system_prompt
        payload = json.loads(
            _prompt(
                {"source_id": "s", "duration_s": 75.0},
                prompt_profile=profile,
            )
        )
        assert set(payload) == {"source_id", "duration_s", "coordinate_system"}


def test_assertive_safe_outside_prompt_prioritizes_nonsemantic_vocal_cleanup() -> None:
    args = parse_args(
        [
            "--manifest",
            "x",
            "--output-dir",
            "y",
            "--prompt-profile",
            ASSERTIVE_SAFE_OUTSIDE_PROMPT_PROFILE,
        ]
    )
    assert args.prompt_profile == ASSERTIVE_SAFE_OUTSIDE_PROMPT_PROFILE
    assert ASSERTIVE_SAFE_OUTSIDE_PROMPT_VERSION.endswith(
        "safe_outside_complement_v5_assertive"
    )
    assert "必须积极标记为 outside" in ASSERTIVE_SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "至少 50ms" in ASSERTIVE_SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "うん" in ASSERTIVE_SAFE_OUTSIDE_SYSTEM_PROMPT
    payload = json.loads(
        _prompt(
            {"source_id": "s", "duration_s": 75.0},
            prompt_profile=ASSERTIVE_SAFE_OUTSIDE_PROMPT_PROFILE,
        )
    )
    assert set(payload) == {"source_id", "duration_s", "coordinate_system"}


def test_balanced_v12_teacher_prompt_allows_nonlexical_human_sound_without_forcing_it() -> None:
    args = parse_args(
        [
            "--manifest",
            "x",
            "--output-dir",
            "y",
            "--prompt-profile",
            BALANCED_V12_SAFE_OUTSIDE_PROMPT_PROFILE,
        ]
    )
    assert args.prompt_profile == BALANCED_V12_SAFE_OUTSIDE_PROMPT_PROFILE
    assert BALANCED_V12_SAFE_OUTSIDE_PROMPT_VERSION.endswith(
        "safe_outside_complement_v6_balanced_v12_teacher"
    )
    assert "不要求证明该区间“绝对不可能”包含语言" in BALANCED_V12_SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "非词化人声也可以标记" in BALANCED_V12_SAFE_OUTSIDE_SYSTEM_PROMPT
    assert "不代表 canonical inside truth" in BALANCED_V12_SAFE_OUTSIDE_SYSTEM_PROMPT
    payload = json.loads(
        _prompt(
            {"source_id": "s", "duration_s": 75.0},
            prompt_profile=BALANCED_V12_SAFE_OUTSIDE_PROMPT_PROFILE,
        )
    )
    assert set(payload) == {"source_id", "duration_s", "coordinate_system"}


def test_custom_safe_outside_prompt_profile_uses_compact_request_payload() -> None:
    args = parse_args(
        [
            "--manifest",
            "x",
            "--output-dir",
            "y",
            "--prompt-profile",
            CUSTOM_SAFE_OUTSIDE_PROMPT_PROFILE,
            "--system-prompt-file",
            "prompt.txt",
        ]
    )
    assert args.prompt_profile == CUSTOM_SAFE_OUTSIDE_PROMPT_PROFILE
    payload = json.loads(
        _prompt(
            {"source_id": "s", "duration_s": 75.0},
            prompt_profile=CUSTOM_SAFE_OUTSIDE_PROMPT_PROFILE,
        )
    )
    assert set(payload) == {"source_id", "duration_s", "coordinate_system"}


def test_safe_outside_prompt_materializes_only_the_provisional_keep_complement() -> None:
    islands, unsure, outside = _safe_outside_complement(
        {
            "safe_outside_spans": [
                {
                    "start_s": 0.04,
                    "end_s": 0.08,
                    "confidence": 0.9,
                    "reason": "room noise",
                },
                {
                    "start_s": 0.14,
                    "end_s": 0.2,
                    "confidence": 0.8,
                    "reason": "music",
                },
            ]
        },
        duration_s=0.2,
        frame_count=10,
    )
    assert unsure == []
    assert [(span["start_frame"], span["end_frame"]) for span in outside] == [
        (2, 4),
        (7, 10),
    ]
    assert [(span["start_frame"], span["end_frame"]) for span in islands] == [
        (0, 2),
        (4, 7),
    ]
    assert all(span["confidence"] == 0.0 for span in islands)


def test_teacher_span_error_names_local_clip_range() -> None:
    try:
        _spans(
            {
                "islands": [
                    {"start_s": 101.5, "end_s": 108.0, "confidence": 0.9}
                ]
            },
            duration_s=75.0,
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("out-of-range teacher coordinates must be rejected")
    assert "required_range=0..75.0" in message
    assert "0-based audio clip timeline" in message


def test_teacher_resume_rejects_rows_from_a_different_prompt(tmp_path: Path) -> None:
    rows = tmp_path / "preaudit.jsonl"
    rows.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "schema": "candidate_island_scorer_v11_omni_preaudit_v2",
                    "source_id": "old",
                    "model": "gemini",
                    "prompt_version": "candidate_island_scorer_v11_omni_preaudit_dialogue_islands_v6",
                },
                {
                    "schema": "candidate_island_scorer_v11_omni_preaudit_v2",
                    "source_id": "current",
                    "model": "gemini",
                    "prompt_version": PROMPT_VERSION,
                },
            )
        ),
        encoding="utf-8",
    )
    assert set(_resume_index(rows, model="gemini")) == {"current"}


def test_teacher_rejects_overlap_between_inside_and_unsure() -> None:
    try:
        _spans(
            {
                "islands": [
                    {"start_s": 1.0, "end_s": 4.0, "confidence": 0.9}
                ],
                "unsure_spans": [{"start_s": 3.5, "end_s": 5.0}],
            },
            duration_s=10.0,
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("inside/unsure overlap must be rejected")
    assert "mutually exclusive" in message


def test_teacher_comparison_uses_continuous_frame_membership(tmp_path: Path) -> None:
    human = tmp_path / "human.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    human.write_text(json.dumps({"source_id": "s", "frame_count": 10, "spans": [{"label": "outside_candidate", "start_frame": 0, "end_frame": 2}, {"label": "inside_candidate", "start_frame": 2, "end_frame": 8}, {"label": "outside_candidate", "start_frame": 8, "end_frame": 10}]}) + "\n", encoding="utf-8")
    qwen.write_text(json.dumps({"source_id": "s", "frame_count": 10, "islands": [{"start_frame": 2, "end_frame": 8}], "unsure_spans": []}) + "\n", encoding="utf-8")
    summary = compare(human_path=human, teacher_specs=[f"qwen={qwen}"], output_dir=tmp_path / "out")
    metrics = summary["aggregate"]["qwen"]
    assert metrics["inside_candidate_recall"] == 1.0
    assert metrics["outside_candidate_recall"] == 1.0
    assert metrics["sources_with_full_source_inside"] == 0


def test_teacher_comparison_audio_follows_source_audit_provenance(tmp_path: Path) -> None:
    source_audit = tmp_path / "source-audit"
    editable = tmp_path / "editable"
    audio = source_audit / "audio" / "source-000.wav"
    audio.parent.mkdir(parents=True)
    editable.mkdir()
    audio.write_bytes(b"wav")
    manifest = editable / "audit_manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    (editable / "summary.json").write_text(
        json.dumps({"source_audit_dir": str(source_audit)}), encoding="utf-8"
    )
    assert _audio_path("audio/source-000.wav", manifest=manifest) == audio.resolve()


def test_outside_consensus_requires_both_teachers_and_asr_silence(tmp_path: Path) -> None:
    def write(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    source_ids = ("clear", "teacher-inside", "asr-text")
    selection = tmp_path / "selection.jsonl"
    inventory = tmp_path / "inventory.jsonl"
    asr = tmp_path / "asr.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    gemini = tmp_path / "gemini.jsonl"
    write(
        selection,
        [
            {
                "schema": "candidate_island_scorer_v11_outside_asr_selection_v1",
                "source_id": source_id,
                "audio": f"{source_id}.wav",
                "audio_sha256": f"sha-{source_id}",
                "duration_s": 1.0,
            }
            for source_id in source_ids
        ],
    )
    write(
        inventory,
        [
            {
                "schema": "speech_scorer_v10_canonical_source_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "partition": "train",
                "row_role": "all_background",
                "background_type": "noise",
            }
            for source_id in source_ids
        ],
    )
    write(
        asr,
        [
            {
                "source_id": source_id,
                "audio_sha256": f"sha-{source_id}",
                "asr_probe_summary": {
                    "span_count": 1,
                    "nonempty_text_span_count": int(source_id == "asr-text"),
                    "error_span_count": 0,
                    "texts_in_workflow_order": ["待って"] if source_id == "asr-text" else ["…"],
                },
            }
            for source_id in source_ids
        ],
    )
    teacher_base = [
        {
            "schema": "candidate_island_scorer_v11_omni_preaudit_v2",
            "source_id": source_id,
            "audio_sha256": f"sha-{source_id}",
            "model": "teacher",
            "prompt_version": "v4",
            "islands": [],
            "unsure_spans": [],
        }
        for source_id in source_ids
    ]
    write(qwen, teacher_base)
    gemini_rows = [dict(row) for row in teacher_base]
    gemini_rows[1]["islands"] = [{"start_s": 0.0, "end_s": 1.0}]
    write(gemini, gemini_rows)
    summary = build_outside_consensus(
        argparse.Namespace(
            selection=str(selection),
            background_inventory=str(inventory),
            asr_enriched=str(asr),
            teacher=[f"qwen={qwen}", f"gemini={gemini}"],
            output_dir=str(tmp_path / "out"),
        )
    )
    assert summary["decision_counts"] == {"clear_outside": 1, "unsure": 2}
    rows = [
        json.loads(line)
        for line in Path(summary["outside_consensus"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_id = {row["source_id"]: row for row in rows}
    assert by_id["clear"]["training_label"] == 0
    assert by_id["teacher-inside"]["training_label"] == -100
    assert by_id["asr-text"]["decision_reasons"] == ["asr_text"]


def test_train_teacher_manifest_uses_only_frozen_train_sources(tmp_path: Path) -> None:
    import wave

    audio = tmp_path / "train.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 3200)
    source_windows = tmp_path / "sources.jsonl"
    source_windows.write_text(
        json.dumps(
            {
                "schema": "joint_boundary_omni_source_window_v1",
                "window_id": "train-w00",
                "video_id": "train",
                "audio_wav": str(audio),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    partition = tmp_path / "partition.jsonl"
    partition.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "schema": "candidate_island_scorer_v11_partition_manifest_v1",
                    "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                    "source_id": "train-w00",
                    "video_id": "train",
                    "partition": "train",
                },
                {
                    "schema": "candidate_island_scorer_v11_partition_manifest_v1",
                    "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                    "source_id": "heldout-w00",
                    "video_id": "heldout",
                    "partition": "test",
                },
            )
        ),
        encoding="utf-8",
    )
    summary = build_train_teacher_manifest(
        argparse.Namespace(
            source_windows=str(source_windows),
            partition_manifest=str(partition),
            output_dir=str(tmp_path / "out"),
        )
    )
    assert summary["source_count"] == 1
    row = json.loads(Path(summary["train_teacher_sources"]).read_text(encoding="utf-8"))
    assert row["source_id"] == "train-w00"
    assert row["partition"] == "train"
    assert row["frame_count"] == 10
    assert row["training_manifest_allowed"] is False


def test_real_outside_selection_uses_exact_gemini_complement(tmp_path: Path) -> None:
    sources = tmp_path / "sources.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    sources.write_text(
        json.dumps(
            {
                "schema": "candidate_island_scorer_v11_train_teacher_source_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": "s",
                "video_id": "v",
                "partition": "train",
                "audio": "s.wav",
                "audio_sha256": "a" * 64,
                "duration_s": 0.2,
                "frame_count": 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    teacher.write_text(
        json.dumps(
            {
                "schema": "candidate_island_scorer_v11_omni_preaudit_v2",
                "source_id": "s",
                "partition": "train",
                "frame_count": 10,
                "audio_sha256": "a" * 64,
                "model": "gemini",
                "prompt_version": "v4",
                "islands": [{"start_frame": 2, "end_frame": 5}],
                "unsure_spans": [{"start_frame": 7, "end_frame": 8}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = build_real_outside_selection(
        argparse.Namespace(
            train_teacher_sources=str(sources),
            gemini_preaudit=str(teacher),
            output_dir=str(tmp_path / "out"),
        )
    )
    assert summary["outside_frame_count"] == 6
    row = json.loads(
        Path(summary["real_outside_asr_selection"]).read_text(encoding="utf-8")
    )
    assert [
        (span["start_frame"], span["end_frame"])
        for span in row["prediction_spans"]
    ] == [(0, 2), (5, 7), (8, 10)]
    assert {span["label"] for span in row["prediction_spans"]} == {
        "asr_probe_candidate"
    }
    assert row["training_manifest_allowed"] is False


def test_real_train_outside_keeps_only_empty_asr_spans(tmp_path: Path) -> None:
    enriched = tmp_path / "enriched.jsonl"
    enriched.write_text(
        json.dumps(
            {
                "schema": "candidate_island_scorer_v11_real_outside_asr_selection_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": "s",
                "video_id": "v",
                "partition": "train",
                "audio": "s.wav",
                "audio_sha256": "a" * 64,
                "duration_s": 0.2,
                "frame_count": 10,
                "prediction_spans": [
                    {
                        "label": "asr_probe_candidate",
                        "start_frame": 0,
                        "end_frame": 3,
                        "asr_probe": {"nonempty_text": False, "error_kind": ""},
                    },
                    {
                        "label": "asr_probe_candidate",
                        "start_frame": 6,
                        "end_frame": 10,
                        "asr_probe": {"nonempty_text": True, "error_kind": ""},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = compile_real_train_outside(
        argparse.Namespace(
            asr_enriched_selection=str(enriched), output_dir=str(tmp_path / "out")
        )
    )
    assert summary["canonical_frame_counts"] == {
        "outside_candidate": 3,
        "unsure": 7,
    }
    assert summary["input_frame_counts"] == summary["canonical_frame_counts"]
    assert summary["skipped_no_outside_source_count"] == 0
    row = json.loads(
        Path(summary["real_train_outside_sources"]).read_text(encoding="utf-8")
    )
    assert row["canonical_spans"] == [
        {
            "end_frame": 3,
            "end_s": 0.06,
            "label": "outside_candidate",
            "start_frame": 0,
            "start_s": 0.0,
        },
        {
            "end_frame": 10,
            "end_s": 0.2,
            "label": "unsure",
            "start_frame": 3,
            "start_s": 0.06,
        },
    ]
