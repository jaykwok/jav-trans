from __future__ import annotations

import json
from pathlib import Path

from tools.audits import generate_pre_asr_v12_repair_audit_html as audit


def _paired(
    window_id: str,
    chunk_index: int,
    *,
    truth: str,
    prediction: str,
    prob_drop: float,
    start: float | None = None,
    duration_s: float = 1.0,
) -> dict:
    start_s = float(chunk_index) if start is None else start
    return {
        "schema": "pre_asr_cueqc_v12_gate_paired_decision_v1",
        "id": f"preasr-{window_id}-chunk{chunk_index:05d}",
        "audio_id": window_id,
        "group_index": 0,
        "chunk_index": chunk_index,
        "start": start_s,
        "end": start_s + duration_s,
        "duration_s": duration_s,
        "truth": truth,
        "v12_prediction": prediction,
        "v12_prob_drop": prob_drop,
        "v12_reason": "model_keep_default" if prediction == "keep" else "model_drop_threshold",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_select_repair_pools_full_a1_sampled_a2_and_deduped_b(tmp_path: Path) -> None:
    a1_rows = [
        _paired("vid-w00", 1, truth="drop", prediction="keep", prob_drop=0.49),
        _paired("vid-w00", 2, truth="drop", prediction="keep", prob_drop=0.20),
    ]
    t095_rows = [
        *a1_rows,
        _paired("vid-w00", 3, truth="drop", prediction="keep", prob_drop=0.55),
        _paired("vid-w00", 4, truth="drop", prediction="keep", prob_drop=0.65),
        _paired("vid-w00", 5, truth="drop", prediction="keep", prob_drop=0.75),
        _paired("vid-w00", 6, truth="keep", prediction="keep", prob_drop=0.01),
    ]
    long_row = _paired("vid-w00", 7, truth="keep", prediction="drop", prob_drop=0.72, duration_s=1.2)
    selected, summary = audit.select_repair_pools(
        t050_rows=a1_rows,
        t095_rows=t095_rows,
        long_false_drop_rows_by_path={
            str(tmp_path / "gate-t040" / "v12_false_drops_ge_0p8s.jsonl"): [long_row],
            str(tmp_path / "gate-t050" / "v12_false_drops_ge_0p8s.jsonl"): [long_row],
        },
        a2_limit=2,
        seed=7,
    )

    by_pool = {}
    for row in selected:
        by_pool.setdefault(row["source_pool"], []).append(row["candidate_id"])

    assert by_pool["A1_t050_residual_false_keep"] == [
        "preasr-vid-w00-chunk00001",
        "preasr-vid-w00-chunk00002",
    ]
    assert len(by_pool["A2_t095_exclusive_false_keep_sample"]) == 2
    assert not set(by_pool["A2_t095_exclusive_false_keep_sample"]) & set(
        by_pool["A1_t050_residual_false_keep"]
    )
    assert by_pool["B_low_threshold_long_false_drop"] == ["preasr-vid-w00-chunk00007"]
    assert summary["a2_t095_exclusive_false_keep_population"] == 3
    assert summary["b_low_threshold_long_false_drop_count"] == 1


def test_build_repair_audit_writes_manifest_summary_and_save_labels_page(tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"RIFF")
    source_windows = _write_jsonl(
        tmp_path / "source_windows.jsonl",
        [
            {
                "schema": "joint_pre_asr_chunk_reexport_v1",
                "window_id": "vid-w00",
                "video_id": "vid",
                "audio_wav": str(audio_path),
                "duration_s": 12.0,
                "source_start_s": 100.0,
                "source_end_s": 112.0,
            }
        ],
    )
    t050 = _write_jsonl(
        tmp_path / "t050" / "paired_decisions.jsonl",
        [
            _paired("vid-w00", 1, truth="drop", prediction="keep", prob_drop=0.49),
            _paired("vid-w00", 2, truth="drop", prediction="keep", prob_drop=0.20),
        ],
    )
    t095 = _write_jsonl(
        tmp_path / "t095" / "paired_decisions.jsonl",
        [
            _paired("vid-w00", 1, truth="drop", prediction="keep", prob_drop=0.49),
            _paired("vid-w00", 2, truth="drop", prediction="keep", prob_drop=0.20),
            _paired("vid-w00", 3, truth="drop", prediction="keep", prob_drop=0.55),
            _paired("vid-w00", 4, truth="drop", prediction="keep", prob_drop=0.85),
        ],
    )
    long_false_drop = _write_jsonl(
        tmp_path / "gate-t050" / "v12_false_drops_ge_0p8s.jsonl",
        [_paired("vid-w00", 5, truth="keep", prediction="drop", prob_drop=0.71, duration_s=1.1)],
    )

    summary = audit.build_audit(
        source_windows_jsonl=source_windows,
        t050_paired_jsonl=t050,
        t095_paired_jsonl=t095,
        long_false_drop_jsonls=[long_false_drop],
        output_dir=tmp_path / "audit",
        a2_limit=1,
        cut_audio=False,
    )

    manifest_rows = (tmp_path / "audit" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    html = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")
    stored_summary = json.loads((tmp_path / "audit" / "summary.json").read_text(encoding="utf-8"))

    assert summary["review_item_count"] == 4
    assert stored_summary["pool_counts"] == {
        "A1_t050_residual_false_keep": 2,
        "A2_t095_exclusive_false_keep_sample": 1,
        "B_low_threshold_long_false_drop": 1,
    }
    assert len(manifest_rows) == 4
    assert "manual_verdicts.jsonl" in html
    assert "/__audit_api__/save-labels" in html
    assert audit.LABEL_SCHEMA in html
    assert "chunk" in html
    assert "context" in html
    assert "physical dedicated VRAM * 0.95" in stored_summary["oom_discipline"]["gpu_oom_definition"]
