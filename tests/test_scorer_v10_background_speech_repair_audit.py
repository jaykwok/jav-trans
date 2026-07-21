from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import soundfile as sf

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.audits.evaluate_scorer_v10_background_speech_repair_audit import (
    EVENT_SCHEMA,
    RESULT_SCHEMA,
    evaluate,
)
from tools.audits.evaluate_scorer_v10_background_source_recheck import (
    evaluate as evaluate_source_recheck,
)
from tools.audits.generate_scorer_v10_background_speech_repair_audit_html import (
    ISLAND_SCHEMA,
    LINK_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
    SUMMARY_SCHEMA,
    build_audit,
)
from tools.audits.generate_scorer_v10_prediction_audit_html import VERDICT_SCHEMA
from tools.audits.generate_scorer_v10_prediction_audit_html import (
    build_audit as build_prediction_audit,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _build_fixture(tmp_path: Path) -> Path:
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(
        canonical,
        [
            {
                "schema": "speech_scorer_v10_canonical_source_v1",
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "canonical_label_schema": "speech_scorer_canonical_frames_v1",
                "source_id": "background-source",
                "audio": str(audio),
                "row_role": "all_background",
                "partition": "test",
                "core_ids": [],
                "background_id": "background-a",
                "background_source_ids": ["background-a"],
                "background_source_video_ids": ["video-a"],
                "sample_rate": 16000,
                "sample_count": 1600,
                "duration_s": 0.1,
                "canonical_spans": [
                    {
                        "start_sample": 0,
                        "end_sample": 1600,
                        "label": "background",
                        "background_id": "background-a",
                    }
                ],
            }
        ],
    )
    prediction_manifest = tmp_path / "prediction_manifest.jsonl"
    audit_id = "background_false_keep:background-source"
    _write_jsonl(
        prediction_manifest,
        [
            {
                "audit_id": audit_id,
                "source_id": "background-source",
                "audio": str(audio),
                "partition": "test",
                "row_role": "all_background",
                "category": "background_false_keep",
                "frame_count": 5,
                "prediction_spans": [
                    {
                        "label": "model_speech",
                        "start_frame": 0,
                        "end_frame": 2,
                        "start_s": 0.0,
                        "end_s": 0.04,
                    },
                    {
                        "label": "model_speech",
                        "start_frame": 3,
                        "end_frame": 5,
                        "start_s": 0.06,
                        "end_s": 0.1,
                    },
                ],
            }
        ],
    )
    prediction_verdicts = tmp_path / "prediction_verdicts.jsonl"
    _write_jsonl(
        prediction_verdicts,
        [
            {
                "schema": VERDICT_SCHEMA,
                "audit_id": audit_id,
                "source_id": "background-source",
                "partition": "test",
                "row_role": "all_background",
                "category": "background_false_keep",
                "verdict": "canonical_contains_target_speech",
            }
        ],
    )
    output = tmp_path / "audit"
    build_audit(
        canonical_sources=canonical,
        prediction_audit_manifest=prediction_manifest,
        prediction_manual_verdicts=prediction_verdicts,
        output_dir=output,
    )
    return output


def test_background_speech_repair_audit_preserves_exact_islands_and_links(
    tmp_path: Path,
) -> None:
    output = _build_fixture(tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    page = (output / "index.html").read_text(encoding="utf-8")

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["source_count"] == 1
    assert summary["island_count"] == 2
    assert summary["link_count"] == 1
    assert [row["schema"] for row in manifest] == [
        ISLAND_SCHEMA,
        ISLAND_SCHEMA,
        LINK_SCHEMA,
    ]
    assert manifest[2]["start_sample"] == 640
    assert manifest[2]["end_sample"] == 960
    assert "蓝岛和黄色间隙都只播放自身区间" in page
    assert "页面不做 runtime merge 或时长规则" in page
    assert '<audio controls preload="none"' in page
    script = re.search(r"<script>([\s\S]*?)</script>", page)
    assert script is not None
    node = shutil.which("node")
    if node is not None:
        parsed = subprocess.run(
            [node, "--check", "-"],
            input=script.group(1),
            text=True,
            capture_output=True,
            check=False,
        )
        assert parsed.returncode == 0, parsed.stderr


def test_background_speech_repair_gate_groups_same_asr_unit(tmp_path: Path) -> None:
    output = _build_fixture(tmp_path)
    manifest = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manual = output / "manual_verdicts.jsonl"
    _write_jsonl(
        manual,
        [
            {
                "schema": MANUAL_VERDICT_SCHEMA,
                "item_id": row["item_id"],
                "item_type": row["item_type"],
                "source_id": row["source_id"],
                "verdict": (
                    "target_speech_span_ok"
                    if row["item_type"] == "island"
                    else "same_asr_unit"
                ),
            }
            for row in manifest
        ],
    )
    gate = evaluate(
        audit_summary=output / "summary.json",
        audit_manifest=output / "audit_manifest.jsonl",
        manual_verdicts=manual,
        output=output / "manual_gate.json",
    )
    events = [
        json.loads(line)
        for line in (output / "manual_gate.events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert gate["schema"] == RESULT_SCHEMA
    assert gate["manual_review_complete"] is True
    assert gate["canonical_repair_ready"] is True
    assert gate["target_island_count"] == 2
    assert gate["required_link_count"] == 1
    assert gate["repair_event_count"] == 1
    assert gate["repair_events_sha256"]
    assert gate["decisions_sha256"]
    assert events[0]["schema"] == EVENT_SCHEMA
    assert events[0]["start_sample"] == 0
    assert events[0]["end_sample"] == 1600
    assert events[0]["member_island_ids"] == [
        "background-source::island00",
        "background-source::island01",
    ]
    assert events[0]["core_id"].startswith("scorer-v10-repair-core-")


def test_background_speech_repair_gate_blocks_incomplete_boundaries(
    tmp_path: Path,
) -> None:
    output = _build_fixture(tmp_path)
    manifest = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manual = output / "manual_verdicts.incomplete.jsonl"
    _write_jsonl(
        manual,
        [
            {
                "schema": MANUAL_VERDICT_SCHEMA,
                "item_id": row["item_id"],
                "item_type": row["item_type"],
                "source_id": row["source_id"],
                "verdict": (
                    "target_speech_boundary_incomplete"
                    if row["item_type"] == "island"
                    else "unreviewed"
                ),
            }
            for row in manifest
        ],
    )
    gate = evaluate(
        audit_summary=output / "summary.json",
        audit_manifest=output / "audit_manifest.jsonl",
        manual_verdicts=manual,
        output=output / "manual_gate.incomplete.json",
    )

    assert gate["manual_review_complete"] is True
    assert gate["boundary_followup_count"] == 2
    assert gate["source_without_target_count"] == 1
    assert gate["canonical_repair_ready"] is False
    assert gate["repair_event_count"] == 0


def test_background_speech_repair_gate_accepts_strict_no_target_recheck(
    tmp_path: Path,
) -> None:
    output = _build_fixture(tmp_path)
    manifest = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manual = output / "manual_verdicts.background.jsonl"
    _write_jsonl(
        manual,
        [
            {
                "schema": MANUAL_VERDICT_SCHEMA,
                "item_id": row["item_id"],
                "item_type": row["item_type"],
                "source_id": row["source_id"],
                "verdict": (
                    "background_or_nonsemantic"
                    if row["item_type"] == "island"
                    else "unreviewed"
                ),
            }
            for row in manifest
        ],
    )
    original_manifest = tmp_path / "prediction_manifest.jsonl"
    original_verdicts = tmp_path / "prediction_verdicts.jsonl"
    recheck = tmp_path / "source_recheck"
    build_prediction_audit(
        selection=original_manifest,
        output_dir=recheck,
        source_ids={"background-source"},
    )
    original_verdict = json.loads(original_verdicts.read_text(encoding="utf-8"))
    _write_jsonl(
        recheck / "manual_verdicts.jsonl",
        [{**original_verdict, "verdict": "model_false_keep"}],
    )
    recheck_gate_path = recheck / "manual_gate.json"
    evaluate_source_recheck(
        original_audit_manifest=original_manifest,
        original_manual_verdicts=original_verdicts,
        recheck_summary=recheck / "summary.json",
        recheck_audit_manifest=recheck / "audit_manifest.jsonl",
        recheck_manual_verdicts=recheck / "manual_verdicts.jsonl",
        output=recheck_gate_path,
    )

    gate = evaluate(
        audit_summary=output / "summary.json",
        audit_manifest=output / "audit_manifest.jsonl",
        manual_verdicts=manual,
        source_recheck_gate=recheck_gate_path,
        output=output / "manual_gate.rechecked.json",
    )

    assert gate["canonical_repair_ready"] is True
    assert gate["source_without_target_count"] == 0
    assert gate["source_recheck_exclusion_ids"] == ["background-source"]
    assert gate["repair_source_count"] == 0
    assert gate["repair_event_count"] == 0
