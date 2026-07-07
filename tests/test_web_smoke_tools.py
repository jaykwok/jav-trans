from __future__ import annotations

import json

from tools.web.smoke import submit_job
from tools.web.smoke import summarize_job


def test_web_smoke_summary_reads_pre_asr_cueqc_report() -> None:
    asr_details = {
        "transcript_chunks": [{"text": "keep"}],
        "pre_asr_cueqc": {
            "schema": "pre_asr_cueqc_report_v2",
            "enabled": True,
            "candidate_count": 3,
            "keep_count": 2,
            "drop_count": 1,
            "decisions": [],
        },
    }

    summary = summarize_job._pre_asr_cueqc_summary(asr_details)

    assert summary["source"] == "asr_details.pre_asr_cueqc"
    assert summary["enabled"] is True
    assert summary["candidate_count"] == 3
    assert summary["keep_count"] == 2
    assert summary["drop_count"] == 1
    assert summary["transcript_chunks"] == 1


def test_web_smoke_submit_does_not_emit_asr_after_cueqc_runtime_env(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_http_json(method: str, url: str, payload: dict):
        captured["method"] = method
        captured["url"] = url
        captured["payload"] = payload
        return {"ids": ["job-test"]}

    monkeypatch.setattr(submit_job, "http_json", fake_http_json)

    rc = submit_job.main(
        [
            "--video-path",
            "video/sample.mp4",
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert rc == 0
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert "CUEQC_SHADOW_ENABLED" not in payload["advanced"]
    assert "CUEQC_MODEL_PATH_BY_REPO" not in payload["advanced"]
    assert "CUEQC_INFERENCE_BATCH_SIZE" not in payload["advanced"]
    written = json.loads((tmp_path / "run" / "submit_payload.json").read_text(encoding="utf-8"))
    assert "CUEQC_SHADOW_ENABLED" not in written["advanced"]
