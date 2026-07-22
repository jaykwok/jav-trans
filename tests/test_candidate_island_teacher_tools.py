from __future__ import annotations

import json
import argparse
from pathlib import Path

from tools.audits.compare_candidate_island_teacher_to_human import compare
from tools.audits.generate_candidate_island_teacher_comparison_html import _audio_path
from tools.asr.cueqc.label_pre_asr_with_omni import normalize_openai_compat_base_url
from tools.boundary.ja.label_candidate_island_scorer_v11_with_omni import parse_args
from tools.boundary.ja.build_candidate_island_scorer_v11_outside_consensus import (
    build as build_outside_consensus,
)
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.omni.run_audio_teacher import parse_args as parse_generic_args


def test_provider_base_url_normalization_and_known_profiles() -> None:
    assert normalize_openai_compat_base_url("https://openrouter.ai/api/v1/chat/completions") == "https://openrouter.ai/api/v1"
    assert parse_args(["--manifest", "x", "--output-dir", "y"]).env_file == "gemini"
    assert parse_args(["--manifest", "x", "--output-dir", "y", "--env-file", "qwen"]).env_file == "qwen"
    assert parse_generic_args(["--output-dir", "y", "--prompt", "x"]).env_file == "gemini"
    assert parse_generic_args(["--output-dir", "y", "--env-file", "qwen", "--prompt", "x"]).env_file == "qwen"


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
