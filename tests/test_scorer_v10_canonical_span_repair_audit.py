from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.audits.evaluate_scorer_v10_canonical_span_repair_audit import evaluate
from tools.audits.generate_scorer_v10_canonical_span_repair_audit_html import (
    build_audit,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_span_repair_audit_quarantines_failed_background_and_saves_exact_spans(
    tmp_path: Path,
) -> None:
    speech_audio = tmp_path / "speech.wav"
    background_audio = tmp_path / "background.wav"
    sf.write(speech_audio, np.zeros(1600, dtype=np.float32), 16000)
    sf.write(background_audio, np.zeros(1600, dtype=np.float32), 16000)
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(
        canonical,
        [
            {
                "source_id": "speech-source",
                "audio": str(speech_audio),
                "partition": "test",
                "row_role": "speech",
                "duration_s": 0.1,
                "sample_rate": 16000,
                "canonical_spans": [
                    {
                        "start_sample": 0,
                        "end_sample": 800,
                        "label": "speech",
                        "core_id": "core-a",
                    },
                    {
                        "start_sample": 800,
                        "end_sample": 1600,
                        "label": "background",
                        "background_id": "background-a",
                    },
                ],
            },
            {
                "source_id": "background-source",
                "audio": str(background_audio),
                "partition": "test",
                "row_role": "all_background",
                "background_id": "background-b",
                "duration_s": 0.1,
                "sample_rate": 16000,
                "canonical_spans": [
                    {"start_sample": 0, "end_sample": 1600, "label": "background"}
                ],
            },
        ],
    )
    source_verdicts = tmp_path / "source_verdicts.jsonl"
    _write_jsonl(
        source_verdicts,
        [
            {
                "schema": "speech_scorer_v10_canonical_manual_verdict_v1",
                "source_id": "speech-source",
                "verdict": "speech_in_background",
                "note": "second span",
            },
            {
                "schema": "speech_scorer_v10_canonical_manual_verdict_v1",
                "source_id": "background-source",
                "verdict": "contains_target_speech",
                "note": "",
            },
        ],
    )
    output = tmp_path / "audit"
    index = build_audit(
        canonical_sources=canonical,
        source_verdicts=source_verdicts,
        output_dir=output,
    )
    page = index.read_text(encoding="utf-8")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["repair_source_count"] == 1
    assert summary["review_item_count"] == 2
    assert summary["quarantined_background_ids"] == ["background-b"]
    assert "speech_scorer_v10_canonical_span_manual_verdict_v1" in page
    assert "if(audio.currentTime>=end)stopPlayback()" in page
    assert "只从该 span 起点播放到终点" in page

    manifest = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manual = output / "manual_verdicts.jsonl"
    _write_jsonl(
        manual,
        [
            {
                "schema": "speech_scorer_v10_canonical_span_manual_verdict_v1",
                "span_id": manifest[0]["span_id"],
                "verdict": "background",
            },
            {
                "schema": "speech_scorer_v10_canonical_span_manual_verdict_v1",
                "span_id": manifest[1]["span_id"],
                "verdict": "speech",
            },
        ],
    )
    gate = evaluate(
        audit_manifest=output / "audit_manifest.jsonl",
        audit_summary=output / "summary.json",
        manual_verdicts=manual,
        output=output / "manual_gate.json",
    )
    assert gate["complete"] is True
    assert gate["canonical_recompile_ready"] is True
    assert gate["changed_span_count"] == 2
    assert gate["quarantined_background_ids"] == ["background-a", "background-b"]
    assert gate["cores_relabelled_background"] == ["core-a"]
    assert gate["training_manifest_allowed"] is False
