from __future__ import annotations

import json

from tools.web.smoke import submit_job
from tools.web.smoke import summarize_job


def test_web_smoke_summary_reads_postgate_report() -> None:
    asr_details = {
        "transcript_chunks": [{"text": "keep"}, {"text": "んっ" * 40}],
        "postgate": {
            "schema": "text_alignment_postgate_v1",
            "reviewed": 2,
            "flagged": 1,
            "flags": {"runaway_repetition": 1},
            "alignment_score_checked": 0,
        },
    }

    summary = summarize_job._postgate_summary(asr_details)

    assert summary["source"] == "asr_details.postgate"
    assert summary["reviewed"] == 2
    assert summary["flagged"] == 1
    assert summary["flags"] == {"runaway_repetition": 1}
    assert summary["transcript_chunks"] == 2


def test_web_smoke_summary_is_empty_without_a_postgate_report() -> None:
    # The old version read a key nothing writes any more and reported blanks,
    # which is indistinguishable from a clean run. Absent must stay absent.
    summary = summarize_job._postgate_summary({"transcript_chunks": []})
    assert summary["reviewed"] is None
    assert summary["flags"] == {}


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
    for retired in ("advanced", "keep_quality_report", "keep_temp_files"):
        assert retired not in payload
    written = json.loads((tmp_path / "run" / "submit_payload.json").read_text(encoding="utf-8"))
    for retired in ("advanced", "keep_quality_report", "keep_temp_files"):
        assert retired not in written


def test_web_smoke_summarize_reads_new_tmp_jobs_and_logs_jobs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # 1. Active/failed job timings under tmp/jobs/<id>/
    tmp_job = tmp_path / "tmp" / "jobs" / "job-active"
    tmp_job.mkdir(parents=True)
    (tmp_job / "sample-a.timings.json").write_text(
        json.dumps({"counts": {"segments": 12}, "stage_timings": {}, "asr_details": {}}),
        encoding="utf-8",
    )
    run_a = tmp_path / "run-a"
    assert summarize_job.main(["--job-id", "job-active", "--run-dir", str(run_a)]) == 0
    assert json.loads((run_a / "job_summary.json").read_text(encoding="utf-8"))["status"] == "artifacts_found"

    # 2. Completed job whose tmp/jobs/<id> was cleaned -> falls back to logs/jobs/<id>/
    log_job = tmp_path / "logs" / "jobs" / "job-done"
    log_job.mkdir(parents=True)
    (log_job / "sample-a.timings.json").write_text(
        json.dumps({"counts": {"segments": 24}, "stage_timings": {}, "asr_details": {}}),
        encoding="utf-8",
    )
    run_b = tmp_path / "run-b"
    assert summarize_job.main(["--job-id", "job-done", "--run-dir", str(run_b)]) == 0
    summary_b = json.loads((run_b / "job_summary.json").read_text(encoding="utf-8"))
    assert summary_b["status"] == "artifacts_found"
    assert summary_b["counts"] == {"segments": 24}
