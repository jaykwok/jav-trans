from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tools.audits.generate_scorer_v10_fragmentation_gap_audit_html import (
    VERDICT_SCHEMA,
    build_audit,
    select_internal_truth_gaps,
)
from tools.audits.evaluate_scorer_v10_fragmentation_gap_audit import (
    ALLOWED_VERDICTS,
    evaluate,
)


def _prediction(*, audio: Path) -> dict[str, object]:
    return {
        "source_id": "speech-source",
        "audio": str(audio),
        "partition": "val",
        "row_role": "speech",
        "truth_spans": [
            {
                "label": "truth_speech",
                "start_frame": 10,
                "end_frame": 30,
                "start_s": 0.2,
                "end_s": 0.6,
            }
        ],
        "prediction_spans": [
            {
                "label": "model_speech",
                "start_frame": 8,
                "end_frame": 20,
                "start_s": 0.16,
                "end_s": 0.4,
            },
            {
                "label": "model_speech",
                "start_frame": 23,
                "end_frame": 32,
                "start_s": 0.46,
                "end_s": 0.64,
            },
        ],
    }


def test_fragmentation_gap_selection_excludes_all_background_and_clips_to_truth(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"RIFF-test")
    speech = _prediction(audio=audio)
    background = {**speech, "source_id": "background", "row_role": "all_background"}
    gaps = select_internal_truth_gaps([background, speech])
    assert len(gaps) == 1
    assert gaps[0]["gap_frames"] == 3
    assert gaps[0]["gap_ms"] == 60
    assert gaps[0]["left_span"]["start_frame"] == 10
    assert gaps[0]["gap_span"]["start_s"] == 0.4
    assert gaps[0]["gap_span"]["end_s"] == 0.46
    assert gaps[0]["right_span"]["end_frame"] == 30
    assert gaps[0]["cluster_id"] == "speech-source:truth0"
    assert gaps[0]["cluster_model_run_count"] == 2


def test_fragmentation_gap_page_uses_exact_no_context_playback_and_saves(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"RIFF-test")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(_prediction(audio=audio), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index = build_audit(predictions=predictions, output_dir=tmp_path / "audit")
    page = index.read_text(encoding="utf-8")
    summary = json.loads((index.parent / "summary.json").read_text(encoding="utf-8"))
    assert "playExact" in page
    assert "preload=\"none\"" in page
    assert "speech_scorer_v10_fragmentation_gap_manual_verdict_v3" in page
    assert "same_asr_unit_keep_continuous" in page
    assert "separate_drop_nonsemantic" in page
    assert "separate_keep_both_speech" in page
    assert "cluster_not_speech_core" in page
    assert "独立 Scorer island" in page
    assert "不合并、不补 gap" in page
    assert "整段 truth-run 条仅用于听感判断，不代表合并或送 ASR" in page
    assert "左侧 model_speech" in page
    assert "truth_speech / model_background" in page
    assert "右侧 model_speech" in page
    assert "完整 island 串审计播放（首个 island 至末个 island，含 gap；仅判断上下文，不送 ASR）" in page
    assert "context-playback" in page
    assert "整串均非 speech core（应用到全部断点）" in page
    assert "id=\"stop\"" in page
    assert r".join('\n')+'\n'" in page
    assert "scorer-v10-fragmentation-gap-audit-v1" not in page
    assert "useful_nonsemantic_separation" not in page
    assert "distinct_target_events" not in page
    assert "<textarea" not in page
    assert summary["review_item_count"] == 1
    assert summary["truth_run_count"] == 1
    assert summary["all_background_gaps_excluded"] is True
    assert summary["playback_context_s"] == 0.0
    assert summary["full_island_cluster_audit_playback"] is True
    assert summary["full_island_cluster_playback_runtime_effect"] == "none_audit_only"
    assert summary["merged_playback"] is False
    assert summary["runtime_gap_merge"] is False
    assert summary["workflow_view_contract"] == (
        "each_argmax_speech_run_is_an_independent_downstream_island"
    )
    assert (index.parent / "audio" / "item-000.wav").is_file()


def test_fragmentation_gap_evaluator_separates_model_and_canonical_failures(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "audit_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "audit_id": "gap-a",
                        "source_id": "source-a",
                        "partition": "val",
                    }
                ),
                json.dumps(
                    {
                        "audit_id": "gap-b",
                        "source_id": "source-b",
                        "partition": "test",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manual = tmp_path / "manual_verdicts.jsonl"
    manual.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema": VERDICT_SCHEMA,
                        "audit_id": "gap-a",
                        "verdict": "same_asr_unit_keep_continuous",
                    }
                ),
                json.dumps(
                    {
                        "schema": VERDICT_SCHEMA,
                        "audit_id": "gap-b",
                        "verdict": "cluster_not_speech_core",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = evaluate(
        audit_manifest=manifest,
        manual_verdicts=manual,
        output=tmp_path / "gate.json",
    )
    assert result["manual_review_complete"] is True
    assert result["model_wrong_fragmentation_count"] == 1
    assert result["model_behavior_pass"] is False
    assert result["canonical_repair_required"] is True
    assert result["cluster_not_speech_core_count"] == 1
    assert result["fragmentation_gate_pass"] is False
    assert result["checkpoint_promotion_authorized"] is False


def test_fragmentation_gap_verdicts_separate_topology_from_routing() -> None:
    assert ALLOWED_VERDICTS == {
        "same_asr_unit_keep_continuous",
        "separate_drop_nonsemantic",
        "separate_keep_both_speech",
        "cluster_not_speech_core",
        "unsure",
        "unreviewed",
    }


def test_fragmentation_gap_evaluator_runs_as_direct_cli(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "audit_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audit_id": "gap-a",
                "source_id": "source-a",
                "partition": "val",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manual = tmp_path / "manual_verdicts.jsonl"
    manual.write_text(
        json.dumps(
            {
                "schema": VERDICT_SCHEMA,
                "audit_id": "gap-a",
                "verdict": "same_asr_unit_keep_continuous",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manual_gate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "tools"
                / "audits"
                / "evaluate_scorer_v10_fragmentation_gap_audit.py"
            ),
            "--audit-manifest",
            str(manifest),
            "--manual-verdicts",
            str(manual),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))[
        "model_wrong_fragmentation_count"
    ] == 1
