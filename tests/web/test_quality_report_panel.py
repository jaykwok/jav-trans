"""The quality report has a reader now.

Every metric the pipeline learned to measure - cut provenance, layout break
types, display linger, the two post-gate layers - was written to
`<stem>.quality_report.json` and read by nobody: the file is not even registered
as an artifact (only its Markdown twin is). These tests pin the two halves that
make it reachable: the endpoint that finds the JSON beside an authorised
artifact, and the panel that labels what it finds.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from web.app import create_app  # noqa: E402
from web import pipeline_manager as pm  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"

_REPORT = {
    "spec_cue_count": 1700,
    "spec_zh_cps_over_9_count": 26,
    "spec_duration_under_min_share": 0.0218,
    "display_linger_applied_count": 1544,
    "display_linger_total_s": 607.939,
    "chunk_cut_policy": "latest_pause_midpoint",
    "chunk_cut_max_fallback_share": 0.0089,
    "layout_break_type_counts": {"sentence_punctuation": 626, "word_gap": 147},
    "postgate_chunks_flagged": 28,
    "postgate_flagged_cue_count": 128,
    "postgate_chunk_flag_counts": {"repeated_unit": 22},
    "postgate_cue_flag_counts": {"repeated_unit": 99},
    "warnings": ["spec_zh_cps_over_9_count=26 > QC_MAX_SPEC_CPS_OVER=0"],
}


async def _reset_pm_state() -> None:
    async with pm._state_lock:
        pm._jobs.clear()
        pm._cancel_events.clear()
    while not pm.gpu_queue.empty():
        pm.gpu_queue.get_nowait()
        pm.gpu_queue.task_done()


async def _job_with_artifacts(tmp_path, artifacts: list[str], output_dir: Path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    jobs = await pm.create_job(
        pm.JobSpec(video_paths=[str(video_path)], output_dir=str(output_dir))
    )
    job = jobs[0]
    async with pm._state_lock:
        job.status = "done"
        job.artifacts = artifacts
        pm._jobs[job.id] = job
    return job


def test_quality_endpoint_serves_the_json_beside_the_markdown(tmp_path, monkeypatch):
    asyncio.run(_test_serves_json_beside_markdown(tmp_path, monkeypatch))


async def _test_serves_json_beside_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "sample.quality_report.md").write_text("# report\n", encoding="utf-8")
    (output_dir / "sample.quality_report.json").write_text(
        json.dumps(_REPORT, ensure_ascii=False), encoding="utf-8"
    )

    try:
        job = await _job_with_artifacts(
            tmp_path, ["sample.srt", "sample.quality_report.md"], output_dir
        )
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/quality/{job.id}")
            missing = await client.get("/api/quality/does-not-exist")

        assert response.status_code == 200
        payload = response.json()
        assert payload["available"] is True
        assert payload["stem"] == "sample"
        assert payload["markdown_name"] == "sample.quality_report.md"
        # The whole report travels: the panel groups keys client-side, so the
        # endpoint must not curate them.
        assert payload["report"] == _REPORT
        assert missing.status_code == 404
    finally:
        await _reset_pm_state()


def test_quality_endpoint_reports_absence_without_failing(tmp_path, monkeypatch):
    asyncio.run(_test_absence_is_not_an_error(tmp_path, monkeypatch))


async def _test_absence_is_not_an_error(tmp_path, monkeypatch):
    """The report is opt-in, so "not generated" is the normal answer."""
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "sample.srt").write_text("1\n", encoding="utf-8")

    try:
        job = await _job_with_artifacts(tmp_path, ["sample.srt"], output_dir)
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/quality/{job.id}")

        assert response.status_code == 200
        assert response.json() == {"available": False, "reason": "not_generated"}
    finally:
        await _reset_pm_state()


def test_quality_endpoint_still_offers_the_markdown_when_json_is_gone(tmp_path, monkeypatch):
    asyncio.run(_test_markdown_only(tmp_path, monkeypatch))


async def _test_markdown_only(tmp_path, monkeypatch):
    """Older runs (and a deleted sidecar) leave the Markdown alone; the panel
    can still hand it to the system viewer instead of claiming nothing exists."""
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "sample.quality_report.md").write_text("# report\n", encoding="utf-8")

    try:
        job = await _job_with_artifacts(
            tmp_path, ["sample.quality_report.md"], output_dir
        )
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/quality/{job.id}")

        assert response.status_code == 200
        payload = response.json()
        assert payload["available"] is False
        assert payload["reason"] == "markdown_only"
        assert payload["markdown_name"] == "sample.quality_report.md"
    finally:
        await _reset_pm_state()


def test_quality_endpoint_refuses_a_report_outside_the_job_roots(tmp_path, monkeypatch):
    asyncio.run(_test_refuses_report_outside_roots(tmp_path, monkeypatch))


async def _test_refuses_report_outside_roots(tmp_path, monkeypatch):
    """A tampered jobs.json must not turn this route into a file reader."""
    from web.routes import files as files_routes

    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(files_routes, "PROJECT_ROOT", project_root)
    await _reset_pm_state()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.quality_report.md").write_text("# secret\n", encoding="utf-8")
    (elsewhere / "secret.quality_report.json").write_text('{"secret": 1}', encoding="utf-8")

    try:
        job = await _job_with_artifacts(
            tmp_path, [str(elsewhere / "secret.quality_report.md")], output_dir
        )
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/quality/{job.id}")

        assert response.status_code == 200
        assert response.json()["available"] is False
    finally:
        await _reset_pm_state()


def test_qc_panel_labels_the_metrics_that_had_no_reader() -> None:
    panel = (STATIC / "js" / "qcReport.js").read_text(encoding="utf-8")

    for key in (
        "layout_break_type_counts",
        "layout_word_gap_cut_count",
        "layout_word_gap_cut_under_0p2s",
        "layout_word_gap_median_s",
        "chunk_cut_policy",
        "chunk_cut_source",
        "chunk_cut_max_fallback_share",
        "chunk_duration_median_s",
        "display_linger_applied_count",
        "display_linger_total_s",
        "postgate_chunks_flagged",
        "postgate_chunks_flagged_share",
        "postgate_flagged_cue_count",
        "postgate_flagged_cue_share",
        "postgate_chunk_flag_counts",
        "postgate_cue_flag_counts",
        "spec_duration_under_min_share",
        "spec_gap_under_2frames_share",
        "spec_zh_cps_over_9_count",
        "vocalisation_cues_dropped",
        "cue_continues_from_previous_share",
    ):
        assert key in panel, f"质检面板没有 {key}"


def test_qc_panel_never_silently_drops_an_unknown_metric() -> None:
    """New metrics land in the report faster than they get labels here. The
    fallback group is what keeps an unlabelled one visible instead of gone."""
    panel = (STATIC / "js" / "qcReport.js").read_text(encoding="utf-8")

    assert "const OTHER_GROUP_TITLE = '其他指标';" in panel
    assert "!GROUPED_KEYS.has(key) && !STRUCTURED_KEYS.has(key)" in panel


def test_qc_panel_is_reachable_from_a_finished_job() -> None:
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    jobs_js = (STATIC / "js" / "jobsRender.js").read_text(encoding="utf-8")
    main_js = (STATIC / "js" / "main.js").read_text(encoding="utf-8")

    assert 'id="qc-overlay"' in index
    assert 'id="qc-body"' in index
    assert "installQcReport();" in main_js
    # The button only exists when the run actually wrote a report.
    assert "/\\.quality_report\\.md$/i.test(p)" in jobs_js
    assert "data-qc=" in jobs_js
    assert "openQcReport(jobId" in jobs_js
