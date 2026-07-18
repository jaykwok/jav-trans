from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.audits.generate_scorer_v10_canonical_data_audit_html import build_audit
from tools.audits.evaluate_scorer_v10_canonical_data_audit import evaluate


def test_scorer_v10_canonical_audit_is_playable_and_saveable(tmp_path: Path) -> None:
    rows = []
    for partition in ("train", "val", "test"):
        for role in ("speech", "all_background"):
            source_id = f"{partition}-{role}"
            audio = tmp_path / f"{source_id}.wav"
            sf.write(audio, np.zeros(1600, dtype=np.float32), 16000)
            rows.append(
                {
                    "source_id": source_id,
                    "audio": str(audio),
                    "partition": partition,
                    "row_role": role,
                    "duration_s": 0.1,
                    "sample_rate": 16000,
                    "core_ids": [f"core-{partition}"] if role == "speech" else [],
                    "canonical_spans": [
                        {
                            "start_sample": 0,
                            "end_sample": 1600,
                            "label": "speech" if role == "speech" else "background",
                            "label_source": "test",
                        }
                    ],
                    "background_type": "breathing",
                    "additive_overlay": None,
                }
            )
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "audit"
    index = build_audit(
        canonical_sources=canonical, output_dir=output, per_role_partition=1
    )
    page = index.read_text(encoding="utf-8")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["review_item_count"] == 6
    assert summary["canonical_sources_sha256"]
    assert len(list((output / "audio").glob("*.wav"))) == 6
    assert "speech_scorer_v10_canonical_manual_verdict_v1" in page
    assert "/__audit_api__/save-labels" in page
    assert "contains_target_speech" in page
    assert ".join('\\n')+'\\n'" in page

    verdicts = output / "manual_verdicts.jsonl"
    manifest = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    verdicts.write_text(
        "".join(
            json.dumps(
                {
                    "schema": "speech_scorer_v10_canonical_manual_verdict_v1",
                    "source_id": row["source_id"],
                    "verdict": "correct",
                }
            )
            + "\n"
            for row in manifest
        ),
        encoding="utf-8",
    )
    gate = evaluate(
        audit_manifest=output / "audit_manifest.jsonl",
        audit_summary=output / "summary.json",
        manual_verdicts=verdicts,
        output=output / "manual_gate.json",
    )
    assert gate["complete"] is True
    assert gate["manual_gate_pass"] is True

    saved = verdicts.read_text(encoding="utf-8").replace(
        '"verdict": "correct"', '"verdict": "contains_target_speech"', 1
    )
    verdicts.write_text(saved, encoding="utf-8")
    rejected = evaluate(
        audit_manifest=output / "audit_manifest.jsonl",
        audit_summary=output / "summary.json",
        manual_verdicts=verdicts,
        output=output / "manual_gate.rejected.json",
    )
    assert rejected["manual_gate_pass"] is False
    assert rejected["risk_count"] == 1
