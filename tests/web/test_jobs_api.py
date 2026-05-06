from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from web.app import create_app
from web import pipeline_manager as pm
from web.routes import config as config_routes


async def _drain_queue(queue: asyncio.Queue) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        queue.task_done()


async def _reset_pm_state() -> None:
    async with pm._state_lock:
        pm._jobs.clear()
        pm._cancel_events.clear()
    await _drain_queue(pm.gpu_queue)
    await _drain_queue(pm.trans_queue)


def test_jobs_api_crud(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_api_crud(tmp_path, monkeypatch))


def test_app_exposes_icon_assets(tmp_path, monkeypatch):
    asyncio.run(_test_app_exposes_icon_assets(tmp_path, monkeypatch))


def test_config_lists_recommended_asr_backend_first(monkeypatch):
    asyncio.run(_test_config_lists_recommended_asr_backend_first(monkeypatch))


def test_settings_hf_endpoint_updates_runtime_env(monkeypatch):
    asyncio.run(_test_settings_hf_endpoint_updates_runtime_env(monkeypatch))


def test_jobs_api_retry_cancelled_job(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_api_retry_cancelled_job(tmp_path, monkeypatch))


async def _test_app_exposes_icon_assets(tmp_path, monkeypatch):
    (tmp_path / "icon.ico").write_bytes(b"\x00\x00\x01\x00")
    (tmp_path / "icon.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIEND\xaeB`\x82"
    )

    import web.app as web_app

    monkeypatch.setattr(web_app, "resource_root", lambda: tmp_path)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        favicon = await client.get("/favicon.ico")
        icon = await client.get("/icon.png")

    assert favicon.status_code == 200
    assert icon.status_code == 200


async def _test_config_lists_recommended_asr_backend_first(monkeypatch):
    monkeypatch.delenv("ASR_BACKEND", raising=False)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backends"][0] == config_routes.RECOMMENDED_ASR_BACKEND
    assert payload["defaults"]["asr_backend"] == config_routes.RECOMMENDED_ASR_BACKEND
    assert set(payload["backends"]) == set(config_routes.BACKENDS)


async def _test_settings_hf_endpoint_updates_runtime_env(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(config_routes, "_update_env_file", lambda _changes: None)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={"hf_endpoint": "https://hf-mirror.com/"},
        )
        assert response.status_code == 200
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

        invalid = await client.post(
            "/api/settings",
            json={"hf_endpoint": "hf-mirror.com"},
        )
        assert invalid.status_code == 422
        assert "HF_ENDPOINT must be empty or a full URL" in invalid.text
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

        cleared = await client.post(
            "/api/settings",
            json={"hf_endpoint": ""},
        )

    assert cleared.status_code == 200
    assert "HF_ENDPOINT" not in os.environ


async def _test_jobs_api_crud(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    original_create_job = pm.create_job

    async def fake_create_job(spec):
        jobs = await original_create_job(spec)
        assert not pm.gpu_queue.empty()
        return jobs

    monkeypatch.setattr(pm, "create_job", fake_create_job)

    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/jobs",
                json={"video_paths": ["sample.mp4"], "skip_translation": True},
            )
            assert response.status_code == 201
            payload = response.json()
            assert payload["ids"]
            job_id = payload["ids"][0]

            response = await client.get("/api/jobs")
            assert response.status_code == 200
            jobs = response.json()
            assert jobs
            assert jobs[0]["status"] in {
                "queued",
                "asr",
                "translating",
                "done",
                "failed",
            }

            response = await client.get(f"/api/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json()["id"] == job_id

            response = await client.delete(f"/api/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json() == {"ok": True}
    finally:
        await _reset_pm_state()


async def _test_jobs_api_retry_cancelled_job(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()

    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/api/jobs",
                json={"video_paths": ["sample.mp4"]},
            )
            assert created.status_code == 201
            job_id = created.json()["ids"][0]

            cancelled = await client.delete(f"/api/jobs/{job_id}")
            assert cancelled.status_code == 200

            retried = await client.post(f"/api/jobs/{job_id}/retry")
            assert retried.status_code == 200
            payload = retried.json()
            assert payload["id"] == job_id
            assert payload["status"] == "queued"

            missing = await client.post("/api/jobs/missing/retry")
            assert missing.status_code == 404

            conflict = await client.post(f"/api/jobs/{job_id}/retry")
            assert conflict.status_code == 409
    finally:
        await _reset_pm_state()
