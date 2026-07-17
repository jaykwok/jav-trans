from __future__ import annotations

import json
from pathlib import Path

from tools.audits import select_split_v4_binary_hard_audit as selector


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        "utf-8",
    )


def test_hard_audit_reuses_exact_prior_verdicts_and_selects_remaining_hard_rows(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    prior = tmp_path / "prior"
    output = tmp_path / "output"
    (source / "audio").mkdir(parents=True)
    prior.mkdir()
    for audio_id in ("reused", "hard", "easy", "residual"):
        (source / "audio" / f"{audio_id}.wav").write_bytes(b"RIFF")

    source_rows = [
        {
            "audit_id": "new-1",
            "category": "unmatched_predicted_cut_event",
            "audio_id": "reused",
            "time_s": 1.2344,
            "p_cut": 0.51,
            "audio_src": "audio/reused.wav",
        },
        {
            "audit_id": "new-2",
            "category": "unmatched_predicted_cut_event",
            "audio_id": "hard",
            "time_s": 2.0,
            "p_cut": 0.62,
            "audio_src": "audio/hard.wav",
        },
        {
            "audit_id": "new-3",
            "category": "unmatched_predicted_cut_event",
            "audio_id": "easy",
            "time_s": 3.0,
            "p_cut": 0.90,
            "audio_src": "audio/easy.wav",
        },
        {
            "audit_id": "new-4",
            "category": "long_residual",
            "audio_id": "residual",
            "start_s": 4.0,
            "end_s": 12.5,
            "duration_s": 8.5,
            "audio_src": "audio/residual.wav",
        },
    ]
    _write_jsonl(source / "audit_manifest.jsonl", source_rows)
    (source / "index.html").write_text(
        "<title>Split v4 binary gate audit</title><script>const rows=[]"
        ",key='split-v4-binary-gate-audit-v1:'+location.pathname</script>"
        "Acoustic Split v4 · 二分类晋升人工 gate",
        "utf-8",
    )
    prior_rows = [
        {
            "audit_id": "old-9",
            "category": "unmatched_predicted_cut_event",
            "audio_id": "reused",
            "time_s": 1.23449,
        }
    ]
    _write_jsonl(prior / "audit_manifest.jsonl", prior_rows)
    _write_jsonl(
        prior / "manual_verdicts.jsonl",
        [{"audit_id": "old-9", "verdict": "valid_cut", "note": "reviewed"}],
    )
    monkeypatch.setattr(selector, "update_audit_entrypoints", lambda **_kwargs: None)

    summary = selector.build(
        source_dir=source,
        output_dir=output,
        max_p_cut=0.65,
        prior_audit_dirs=(prior,),
    )

    selected = selector._rows(output / "audit_manifest.jsonl")
    reused = selector._rows(output / "reused_manual_verdicts.jsonl")
    assert [row["audit_id"] for row in selected] == ["new-2", "new-4"]
    assert reused[0]["audit_id"] == "new-1"
    assert reused[0]["source_audit_id"] == "old-9"
    assert reused[0]["verdict"] == "valid_cut"
    assert summary["reused_manual_verdict_count"] == 1
    assert summary["low_confidence_event_count"] == 1
    assert summary["long_residual_count"] == 1
