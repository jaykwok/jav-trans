from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from llm import backends as llm_backends
from web.app import create_app
from web import pipeline_manager as pm
from web.models import JobSpec
from web.routes import config as config_routes
from web.routes import files as files_routes


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
        pm._tombstones.clear()
    await _drain_queue(pm.gpu_queue)
    await _drain_queue(pm.trans_queue)
    # A module-level asyncio.Queue binds to the first loop that waits on it, and
    # every test here runs its own asyncio.run(): reuse would kill the next
    # test's workers with "bound to a different event loop" as soon as they
    # await an empty get(). Hand each test fresh queues instead.
    pm.gpu_queue = asyncio.Queue()
    pm.trans_queue = asyncio.Queue()


def test_jobs_api_crud(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_api_crud(tmp_path, monkeypatch))


def test_gpu_state_api_reports_and_retries_cleanup(tmp_path, monkeypatch):
    asyncio.run(_test_gpu_state_api_reports_and_retries_cleanup(tmp_path, monkeypatch))


def test_gpu_state_and_cancel_stay_responsive_during_a_model_load(tmp_path, monkeypatch):
    asyncio.run(
        _test_gpu_state_and_cancel_stay_responsive_during_a_model_load(
            tmp_path, monkeypatch
        )
    )


def test_app_exposes_icon_assets(tmp_path, monkeypatch):
    asyncio.run(_test_app_exposes_icon_assets(tmp_path, monkeypatch))


def test_config_payload_has_no_backend_select(monkeypatch):
    asyncio.run(_test_config_payload_has_no_backend_select(monkeypatch))


def test_model_requirements_single_asr_model_present(tmp_path, monkeypatch):
    asyncio.run(_test_model_requirements_single_asr_model_present(tmp_path, monkeypatch))


def test_model_requirements_missing_model_needs_download(tmp_path, monkeypatch):
    asyncio.run(_test_model_requirements_missing_model_needs_download(tmp_path, monkeypatch))


def test_model_requirements_includes_cuda_driver_warning(tmp_path, monkeypatch):
    asyncio.run(_test_model_requirements_includes_cuda_driver_warning(tmp_path, monkeypatch))


def test_model_requirements_can_skip_the_cuda_probe(tmp_path, monkeypatch):
    asyncio.run(_test_model_requirements_can_skip_the_cuda_probe(tmp_path, monkeypatch))


def test_the_cuda_probe_never_stalls_the_event_loop(tmp_path, monkeypatch):
    asyncio.run(_test_the_cuda_probe_never_stalls_the_event_loop(tmp_path, monkeypatch))


def test_a_restart_is_a_new_clock_not_an_older_one(tmp_path, monkeypatch):
    asyncio.run(_test_a_restart_is_a_new_clock_not_an_older_one(tmp_path, monkeypatch))


def test_a_restart_inside_one_millisecond_is_still_a_new_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(
        pm, "_time", SimpleNamespace(time=lambda: 1_700_000_000.0, sleep=lambda _s: None)
    )
    monkeypatch.setattr(pm, "_epoch", 0)

    first = pm.state_epoch()
    pm._epoch = 0  # a fresh interpreter, before the clock has moved
    second = pm.state_epoch()

    assert second == first + 1


def test_every_published_change_dates_itself(tmp_path, monkeypatch):
    asyncio.run(_test_every_published_change_dates_itself(tmp_path, monkeypatch))


def test_cuda_environment_status_reads_frozen_probe_file(tmp_path, monkeypatch):
    config_routes._cuda_environment_status.cache_clear()
    monkeypatch.setattr(config_routes, "is_frozen", lambda: True)
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)

    def fake_run(command, *, env, **kwargs):
        Path(env["JAV_TRANS_CUDA_PROBE_OUTPUT"]).write_text(
            '{"status":"ok","ok":true,"code":"ok","message":"file result"}',
            encoding="utf-8",
        )
        return config_routes.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(config_routes.subprocess, "run", fake_run)

    try:
        payload = config_routes._cuda_environment_status()
    finally:
        config_routes._cuda_environment_status.cache_clear()

    assert payload["ok"] is True
    assert payload["code"] == "ok"
    assert payload["message"] == "file result"
    assert not list((tmp_path / "tmp").glob("cuda-probe-*.json"))


def test_windows_picker_subprocesses_hide_console(monkeypatch):
    calls: list[dict] = []

    monkeypatch.setattr(
        files_routes,
        "no_window_subprocess_kwargs",
        lambda: {"creationflags": 0x08000000},
    )

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return files_routes.subprocess.CompletedProcess(
            command,
            0,
            stdout="D:\\Videos\\sample.mp4\nD:\\Videos\\sample.mkv\n",
            stderr="",
        )

    monkeypatch.setattr(files_routes.subprocess, "run", fake_run)

    assert files_routes._pick_files_windows() == [
        "D:\\Videos\\sample.mp4",
        "D:\\Videos\\sample.mkv",
    ]
    assert files_routes._pick_folder_windows() == "D:\\Videos\\sample.mp4\nD:\\Videos\\sample.mkv"

    assert len(calls) == 2
    for call in calls:
        assert call["command"][0] == "powershell"
        assert call["creationflags"] == 0x08000000
        assert call["check"] is False
        assert call["capture_output"] is True


def test_pick_folder_windows_passes_through_a_custom_description(monkeypatch):
    calls: list[dict] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return files_routes.subprocess.CompletedProcess(
            command, 0, stdout="D:\\llama.cpp\n", stderr=""
        )

    monkeypatch.setattr(files_routes.subprocess, "run", fake_run)

    assert files_routes._pick_folder_windows("选择 llama-server 所在文件夹") == "D:\\llama.cpp"
    assert "选择 llama-server 所在文件夹" in calls[0]["command"][-1]


def test_pick_directory_api_returns_the_picked_path(monkeypatch):
    asyncio.run(_test_pick_directory_api_returns_the_picked_path(monkeypatch))


async def _test_pick_directory_api_returns_the_picked_path(monkeypatch):
    seen_descriptions: list[str] = []

    def fake_pick_folder_blocking(description="选择视频文件夹"):
        seen_descriptions.append(description)
        return "D:\\llama.cpp"

    monkeypatch.setattr(files_routes, "_pick_folder_blocking", fake_pick_folder_blocking)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/pick-directory",
            params={"description": "选择 llama-server 所在文件夹"},
        )

    assert r.status_code == 200
    assert r.json() == {"path": "D:\\llama.cpp"}
    assert seen_descriptions == ["选择 llama-server 所在文件夹"]


def test_settings_proxy_updates_runtime_env(monkeypatch):
    asyncio.run(_test_settings_proxy_updates_runtime_env(monkeypatch))


def test_proxy_test_reports_unconfigured_when_no_proxy(monkeypatch):
    asyncio.run(_test_proxy_test_reports_unconfigured(monkeypatch))


def test_proxy_test_ok_when_reachable(monkeypatch):
    asyncio.run(_test_proxy_test_ok_when_reachable(monkeypatch))


def test_settings_translation_fields_update_runtime_env(monkeypatch):
    asyncio.run(_test_settings_translation_fields_update_runtime_env(monkeypatch))


def test_settings_rejects_unknown_fields(monkeypatch):
    asyncio.run(_test_settings_rejects_unknown_fields(monkeypatch))


def test_settings_env_file_quotes_multiline_values(tmp_path, monkeypatch):
    asyncio.run(_test_settings_env_file_quotes_multiline_values(tmp_path, monkeypatch))


def test_settings_creates_env_file_on_first_save(tmp_path, monkeypatch):
    asyncio.run(_test_settings_creates_env_file_on_first_save(tmp_path, monkeypatch))


def test_settings_seeds_template_into_an_installer_written_env(tmp_path, monkeypatch):
    """The installer writes `.env` before the web page can exist, so keying the
    template off "file missing" gave the examples only to users who declined a
    proxy - the opposite of who needs them. Seed off the marker instead, and
    never clobber the proxy the installer just wrote."""
    from dotenv import dotenv_values

    env_path = tmp_path / ".env"
    env_path.write_text(
        "# --- network proxy (written by the first-run installer) ---\n"
        "PROXY_PROTOCOL=http\nPROXY_HOST=127.0.0.1\nPROXY_PORT=7890\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)

    config_routes._update_env_file({"TARGET_LANG": "简体中文"})
    text = env_path.read_text(encoding="utf-8")
    assert config_routes._ENV_TEMPLATE_MARKER in text
    assert "# ASR_BATCH_SIZE_BY_REPO=" in text
    values = dotenv_values(env_path)
    assert values["PROXY_HOST"] == "127.0.0.1"
    assert values["PROXY_PORT"] == "7890"
    assert values["TARGET_LANG"] == "简体中文"

    config_routes._update_env_file({"TARGET_LANG": "繁體中文"})
    second = env_path.read_text(encoding="utf-8")
    assert second.count(config_routes._ENV_TEMPLATE_MARKER) == 1
    assert dotenv_values(env_path)["TARGET_LANG"] == "繁體中文"


def test_settings_respects_a_user_who_deleted_the_examples(tmp_path, monkeypatch):
    """Seeding is one-shot. A user who stripped the commented block does not get
    it pushed back on every save."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# jav-trans local overrides.\n"
        "# Defaults live in src/core/config.py. Keep these examples commented unless\n"
        'TARGET_LANG="简体中文"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)

    config_routes._update_env_file({"TARGET_LANG": "English"})
    text = env_path.read_text(encoding="utf-8")
    assert "# ASR_BATCH_SIZE_BY_REPO=" not in text
    assert text.count(config_routes._ENV_TEMPLATE_MARKER) == 1


def test_models_api_falls_back_to_v1_models(monkeypatch):
    asyncio.run(_test_models_api_falls_back_to_v1_models(monkeypatch))


def test_models_api_accepts_models_slug_shape(monkeypatch):
    asyncio.run(_test_models_api_accepts_models_slug_shape(monkeypatch))


def test_jobs_snapshot_saved_translation_settings(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_snapshot_saved_translation_settings(tmp_path, monkeypatch))


def test_jobs_api_retry_cancelled_job(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_api_retry_cancelled_job(tmp_path, monkeypatch))


def test_jobs_api_rejects_a_job_that_cannot_translate(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_api_rejects_a_job_that_cannot_translate(tmp_path, monkeypatch))


def test_retry_rejects_and_then_rereads_translation_settings(tmp_path, monkeypatch):
    asyncio.run(
        _test_retry_rejects_and_then_rereads_translation_settings(tmp_path, monkeypatch)
    )


def test_public_error_message_prefers_exception_detail():
    class DetailError(RuntimeError):
        detail = "GPU 显存不足：只有 1.7B 一档"

        def __str__(self) -> str:
            return "oom: internal wrapper"

    assert pm._public_error_message(DetailError()) == "GPU 显存不足：只有 1.7B 一档"


def test_jobs_api_rejects_invalid_job_spec(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_api_rejects_invalid_job_spec(tmp_path, monkeypatch))


def test_open_routes_are_limited_to_job_paths(tmp_path, monkeypatch):
    asyncio.run(_test_open_routes_are_limited_to_job_paths(tmp_path, monkeypatch))


def test_open_folder_allows_default_video_directory(tmp_path, monkeypatch):
    asyncio.run(_test_open_folder_allows_default_video_directory(tmp_path, monkeypatch))


def test_downloading_an_artifact_follows_the_same_rule_as_opening_it(
    tmp_path, monkeypatch
):
    asyncio.run(
        _test_downloading_an_artifact_follows_the_same_rule_as_opening_it(
            tmp_path, monkeypatch
        )
    )


async def _test_app_exposes_icon_assets(tmp_path, monkeypatch):
    image_dir = tmp_path / "src" / "assets" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "icon.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIEND\xaeB`\x82"
    )
    (image_dir / "icon.ico").write_bytes(b"\x00\x00\x01\x00")

    import web.app as web_app

    monkeypatch.setattr(web_app, "resource_root", lambda: tmp_path)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        icon = await client.get("/assets/images/icon.png")
        app_icon = await client.get("/assets/images/icon.ico")

    assert icon.status_code == 200
    assert app_icon.status_code == 200
    assert icon.headers["content-type"] == "image/png"
    index_html = (_SRC_WEB / "static" / "index.html").read_text(encoding="utf-8")
    assert "/assets/images/icon.png" in index_html
    assert 'href="/icon.png"' not in index_html
    assert 'src="/icon.png"' not in index_html
    assert "favicon.ico" not in index_html


async def _test_config_payload_has_no_backend_select(monkeypatch):
    monkeypatch.delenv("ASR_BACKEND", raising=False)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    # Only one ASR model ships; a backend choice must not resurface anywhere
    # the browser could render a select from.
    assert "backends" not in payload
    assert "recommended_asr_backend" not in payload
    assert "engine_defaults" not in payload
    assert "asr_backend" not in payload["defaults"]
    assert "translation_batch_size" not in payload["defaults"]
    assert payload["defaults"]["translation_max_workers"] == 4
    assert "show_speaker" not in payload["defaults"]
    assert payload["subtitle_modes"] == ["zh", "bilingual"]


def _isolate_model_requirement_env(
    tmp_path,
    monkeypatch,
    *,
    cuda_status: dict | None = None,
) -> None:
    models_root = tmp_path / "models"
    resource_root = tmp_path / "resource"
    monkeypatch.setattr(config_routes.model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_routes.model_paths, "MODELS_ROOT", models_root)
    monkeypatch.setattr(config_routes.model_paths, "RESOURCE_ROOT", resource_root)
    monkeypatch.setattr(config_routes.model_paths, "BUNDLED_MODELS_ROOT", resource_root / "models")
    monkeypatch.setattr(config_routes.model_paths, "is_frozen", lambda: False)
    for key in ("ASR_BACKEND", "ASR_MODEL_ID", "ASR_MODEL_PATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(
        config_routes,
        "_cuda_environment_status",
        lambda: cuda_status
        or {
            "status": "ok",
            "ok": True,
            "code": "ok",
            "message": "CUDA 可用。",
            "torch_cuda_version": "12.8",
        },
    )


async def _test_model_requirements_single_asr_model_present(tmp_path, monkeypatch):
    _isolate_model_requirement_env(tmp_path, monkeypatch)
    local_model = tmp_path / "models" / "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (local_model / "model.safetensors").write_bytes(b"weights")

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/model-requirements")

    assert response.status_code == 200
    payload = response.json()
    assert "asr_backend" not in payload
    assert "required_checkpoints" not in payload
    assert "checkpoint_missing_count" not in payload
    assert len(payload["required_models"]) == 1
    only = payload["required_models"][0]
    assert only["repo_id"] == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    assert only["present"] is True
    assert payload["missing_count"] == 0
    assert payload["needs_download"] is False
    assert payload["all_present"] is True
    assert payload["pipeline_ready"] is True


async def _test_model_requirements_missing_model_needs_download(tmp_path, monkeypatch):
    _isolate_model_requirement_env(tmp_path, monkeypatch)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/model-requirements")

    assert response.status_code == 200
    payload = response.json()
    assert payload["missing_count"] == 1
    assert payload["needs_download"] is True
    assert payload["download_disabled"] is False
    assert payload["all_present"] is False
    assert payload["pipeline_ready"] is False


async def _test_model_requirements_can_skip_the_cuda_probe(tmp_path, monkeypatch):
    """The readiness notice polls this while a model downloads; the probe spawns
    a torch-importing child, so the poll must be able to leave it out."""
    _isolate_model_requirement_env(tmp_path, monkeypatch)
    probes = 0

    def counting_probe():
        nonlocal probes
        probes += 1
        return {"status": "ok", "ok": True, "code": "ok", "message": "CUDA 可用。"}

    monkeypatch.setattr(config_routes, "_cuda_environment_status", counting_probe)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        skipped = await client.get("/api/model-requirements?include_cuda=0")
        included = await client.get("/api/model-requirements")

    assert skipped.status_code == 200
    assert skipped.json()["cuda"] is None
    assert skipped.json()["missing_count"] == 1
    assert included.json()["cuda"]["ok"] is True
    assert probes == 1


async def _test_every_published_change_dates_itself(tmp_path, monkeypatch):
    """Ordering the page can act on has to come from the server.

    `run_id` distinguishes two executions; it says nothing about two states of
    the same one, which is what arrives out of order. So every published change
    carries a monotonic revision, and a list response carries the reading it was
    taken at - the only way an empty list can say "everything was deleted"
    rather than "this is from before anything existed".
    """
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()

    try:
        job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
        created = job.revision
        await pm.update_job_progress(
            job.id, {"stage": "asr", "pct": 10}, run_id=job.run_id
        )
        progressed = (await pm.get_job(job.id)).revision
        await pm._set_job(job, status="done", expected_run_id=job.run_id)
        finished = (await pm.get_job(job.id)).revision

        assert 0 < created < progressed < finished

        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            listed = await client.get("/api/jobs")
            watermark = int(listed.headers["X-Jobs-Revision"])
            assert [item["revision"] for item in listed.json()] == [finished]
            assert watermark >= finished

            cleared = await client.delete("/api/jobs")
            emptied = await client.get("/api/jobs")

        assert cleared.json()["removed"] == 1
        assert emptied.json() == []
        # The removal itself moved the clock: a list response taken before it
        # is now datably older, which is what lets the page drop it.
        assert int(emptied.headers["X-Jobs-Revision"]) > watermark
    finally:
        await _reset_pm_state()


async def _test_a_restart_is_a_new_clock_not_an_older_one(tmp_path, monkeypatch):
    """The revision counter is resumed from what survived, so it can rewind.

    Delete every job and restart and there is nothing left to resume from: the
    clock starts at 0, below the reading a page left open has already applied.
    That page then reads everything the new server says as older news and drops
    it - including the empty list that would have explained the deletion, and
    every job created afterwards. The epoch is what makes the rewind legible as
    a different clock rather than an older one.
    """
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_epoch", 0)
    await _reset_pm_state()

    try:
        job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
        await pm._set_job(job, status="done", expected_run_id=job.run_id)

        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            listed = await client.get("/api/jobs")
            first_epoch = int(listed.headers["X-Jobs-Epoch"])
            single = await client.get(f"/api/jobs/{job.id}")
            # A single-job response has a revision in its body and nowhere but a
            # header to say which clock issued it.
            assert int(single.headers["X-Jobs-Epoch"]) == first_epoch

            await client.delete("/api/jobs")
            emptied = await client.get("/api/jobs")
            high_water = int(emptied.headers["X-Jobs-Revision"])
            assert int(emptied.headers["X-Jobs-Epoch"]) == first_epoch

        assert high_water > 0

        # The restart: a fresh interpreter starts the counter at zero and
        # load_jobs can only raise it to the highest revision that survived.
        pm._epoch = 0
        pm._revision_counter = 0
        pm._tombstones.clear()
        await pm.load_jobs()
        _jobs, revision = await pm.jobs_snapshot()

        assert revision < high_water  # the rewind is real, not designed away
        assert pm.state_epoch() > first_epoch  # and it arrives labelled as one
    finally:
        await _reset_pm_state()


async def _test_the_cuda_probe_never_stalls_the_event_loop(tmp_path, monkeypatch):
    """A 20-second child process is not something to await on the event loop.

    The probe spawns a torch-importing child and waits for it. Run inline in an
    async route, everything else the server owes the page - progress, 取消,
    another tab - waits behind it, because a coroutine only yields where it
    awaits. It is cached, so this is the first load: exactly when the page is
    trying to render.
    """
    _isolate_model_requirement_env(tmp_path, monkeypatch)
    release = threading.Event()
    ticks = 0

    def slow_probe():
        # Stands in for the torch-importing child: it ends only once the loop
        # has proved it is still running, so a loop that stops ticking never
        # gets its answer back.
        release.wait(10.0)
        return {"status": "ok", "ok": True, "code": "ok", "message": "CUDA 可用。"}

    monkeypatch.setattr(config_routes, "_cuda_environment_status", slow_probe)

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1
            if ticks >= 30:
                release.set()

    beating = asyncio.create_task(heartbeat())
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            probed = await asyncio.wait_for(
                client.get("/api/model-requirements"), timeout=5.0
            )
    finally:
        release.set()
        beating.cancel()
        try:
            await beating
        except asyncio.CancelledError:
            pass

    assert probed.status_code == 200
    assert probed.json()["cuda"]["ok"] is True
    assert ticks >= 30


async def _test_model_requirements_includes_cuda_driver_warning(tmp_path, monkeypatch):
    _isolate_model_requirement_env(
        tmp_path,
        monkeypatch,
        cuda_status={
            "status": "error",
            "ok": False,
            "code": "driver_too_old",
            "message": "NVIDIA 驱动最高支持 CUDA 12.1，但当前打包的 PyTorch 需要 CUDA 12.8。",
            "torch_cuda_version": "12.8",
            "nvidia_smi": {
                "cuda_version": "12.1",
                "driver_version": "531.79",
            },
        },
    )
    local_model = tmp_path / "models" / "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (local_model / "model.safetensors").write_bytes(b"weights")

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/model-requirements")

    assert response.status_code == 200
    payload = response.json()
    assert payload["missing_count"] == 0
    assert payload["gpu_ready"] is False
    assert payload["pipeline_ready"] is False
    assert payload["cuda"]["code"] == "driver_too_old"
    assert "CUDA 12.8" in payload["cuda"]["message"]


async def _test_settings_proxy_updates_runtime_env(monkeypatch):
    for key in (
        "PROXY_PROTOCOL",
        "PROXY_HOST",
        "PROXY_PORT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setattr(config_routes, "_update_env_file", lambda _changes: None)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={
                "proxy_protocol": "http",
                "proxy_host": "127.0.0.1",
                "proxy_port": 7890,
            },
        )
        assert response.status_code == 200
        assert os.environ["PROXY_PROTOCOL"] == "http"
        assert os.environ["PROXY_HOST"] == "127.0.0.1"
        assert os.environ["PROXY_PORT"] == "7890"
        assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
        assert os.environ["ALL_PROXY"] == "http://127.0.0.1:7890"
        assert "HF_ENDPOINT" not in os.environ

        current = await client.get("/api/settings")
        assert current.status_code == 200
        payload = current.json()
        assert payload["proxy_protocol"] == "http"
        assert payload["proxy_host"] == "127.0.0.1"
        assert payload["proxy_port"] == 7890

        cleared = await client.post(
            "/api/settings",
            json={"proxy_host": "", "proxy_port": None},
        )

    assert cleared.status_code == 200
    assert os.environ["PROXY_HOST"] == ""
    assert os.environ["PROXY_PORT"] == ""
    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert "ALL_PROXY" not in os.environ


async def _test_proxy_test_reports_unconfigured(monkeypatch):
    for key in ("PROXY_HOST", "PROXY_PORT", "PROXY_PROTOCOL"):
        monkeypatch.delenv(key, raising=False)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/proxy-test")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["proxy_url"] == ""
    assert "未启用" in payload["error"]


class _FakeProxyResponse:
    status_code = 200


class _FakeProxyClient:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        self.requested_url = url
        return _FakeProxyResponse()


async def _test_proxy_test_ok_when_reachable(monkeypatch):
    monkeypatch.setenv("PROXY_PROTOCOL", "http")
    monkeypatch.setenv("PROXY_HOST", "127.0.0.1")
    monkeypatch.setenv("PROXY_PORT", "7890")
    # Capture the real client BEFORE patching so the ASGI test client below is
    # unaffected; the route uses config_routes.httpx.AsyncClient (patched).
    original_async_client = httpx.AsyncClient
    monkeypatch.setattr(config_routes.httpx, "AsyncClient", _FakeProxyClient)

    transport = httpx.ASGITransport(app=create_app())
    async with original_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/proxy-test")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status_code"] == 200
    assert payload["proxy_url"] == "http://127.0.0.1:7890"
    assert isinstance(payload["elapsed_ms"], int)


async def _test_settings_translation_fields_update_runtime_env(monkeypatch):
    for key in ("TRANSLATION_GLOSSARY", "TARGET_LANG", "LLM_REASONING_EFFORT"):
        monkeypatch.delenv(key, raising=False)
    stale_env = {
        "TRANSLATION_GLOSSARY": "old glossary",
        "TARGET_LANG": "简体中文",
        "LLM_REASONING_EFFORT": "max",
    }
    monkeypatch.setattr(
        config_routes,
        "_read_env_entry",
        lambda key: (key in stale_env, stale_env.get(key, "")),
    )
    monkeypatch.setattr(config_routes, "_update_env_file", lambda _changes: None)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={
                "translation_glossary": "",
                "target_lang": "繁體中文",
                "llm_reasoning_effort": "low",
            },
        )
        assert response.status_code == 200

        settings = await client.get("/api/settings")

    assert settings.status_code == 200
    payload = settings.json()
    assert payload["translation_glossary"] == ""
    assert payload["target_lang"] == "繁體中文"
    assert payload["llm_reasoning_effort"] == "low"


def test_reasoning_effort_frontend_and_models_default_to_low():
    from web.models import SettingsRead, normalize_llm_reasoning_effort

    project_root = Path(__file__).resolve().parents[2]
    index = (project_root / "src" / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    settings_js = (
        project_root / "src" / "web" / "static" / "js" / "settings.js"
    ).read_text(encoding="utf-8")

    assert '<option value="low" selected>' in index
    assert "s.llm_reasoning_effort || 'low'" in settings_js
    assert normalize_llm_reasoning_effort(None) == "low"
    assert SettingsRead(
        api_key_set=False,
        api_key_preview="",
        base_url="",
    ).llm_reasoning_effort == "low"


async def _test_settings_rejects_unknown_fields(monkeypatch):
    monkeypatch.setattr(config_routes, "_update_env_file", lambda _changes: None)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={"unexpected_setting": "value"},
        )

    assert response.status_code == 422


async def _test_settings_env_file_quotes_multiline_values(tmp_path, monkeypatch):
    from dotenv import dotenv_values

    env_path = tmp_path / ".env"
    env_path.write_text("API_KEY=old\nTARGET_LANG=简体中文\n", encoding="utf-8")
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    malicious_value = "safe\nOPENAI_COMPATIBILITY_BASE_URL=https://evil.example"

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={"api_key": malicious_value},
        )

    assert response.status_code == 200
    values = dotenv_values(env_path)
    assert values["API_KEY"] == malicious_value
    assert values.get("OPENAI_COMPATIBILITY_BASE_URL") is None
    assert os.environ["API_KEY"] == malicious_value


async def _test_settings_creates_env_file_on_first_save(tmp_path, monkeypatch):
    from dotenv import dotenv_values

    env_path = tmp_path / ".env"
    assert not env_path.exists()
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)
    for key in (
        "API_KEY",
        "OPENAI_COMPATIBILITY_BASE_URL",
        "LLM_MODEL_NAME",
        "TARGET_LANG",
        "LLM_REASONING_EFFORT",
    ):
        monkeypatch.delenv(key, raising=False)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={
                "api_key": "first-run-key",
                "base_url": "https://api.example.test/v1",
                "model": "translator-test",
                "target_lang": "繁體中文",
                "llm_reasoning_effort": "low",
            },
        )
        settings = await client.get("/api/settings")

    assert response.status_code == 200
    assert env_path.exists()
    text = env_path.read_text(encoding="utf-8")
    assert "# Defaults live in src/core/config.py" in text
    assert "# ASR_STAGE_WORKER_VRAM_BUDGET_MB=auto" in text
    assert "# ASR_STAGE_WORKER_VRAM_RATIO=0.95" in text
    assert "# ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO=" in text
    assert "# ASR_STAGE_WORKER_MODE=subprocess" not in text
    assert "# ASR_BATCH_SIZE_BY_REPO=" in text
    values = dotenv_values(env_path)
    assert values["API_KEY"] == "first-run-key"
    assert values["OPENAI_COMPATIBILITY_BASE_URL"] == "https://api.example.test/v1"
    assert values["LLM_MODEL_NAME"] == "translator-test"
    assert values["TARGET_LANG"] == "繁體中文"
    assert "LLM_API_FORMAT" not in values
    assert values["LLM_REASONING_EFFORT"] == "low"
    assert "ASR_BACKEND" not in values
    assert "ASR_BATCH_SIZE_BY_REPO" not in values
    assert settings.status_code == 200
    payload = settings.json()
    assert payload["api_key_set"] is True
    assert payload["base_url"] == "https://api.example.test/v1"
    assert payload["model"] == "translator-test"
    assert payload["target_lang"] == "繁體中文"


async def _test_models_api_falls_back_to_v1_models(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.example.test")

    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if str(request.url) == "https://api.example.test/models":
            return httpx.Response(
                200,
                text="<html></html>",
                headers={"content-type": "text/html"},
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-5.4-mini"},
                    {"id": "gpt-5.5"},
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mock_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(config_routes.httpx, "AsyncClient", mock_async_client)

    app_transport = httpx.ASGITransport(app=create_app())
    async with original_async_client(
        transport=app_transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == {"models": ["gpt-5.4-mini", "gpt-5.5"]}
    assert requests == [
        "https://api.example.test/models",
        "https://api.example.test/v1/models",
    ]


async def _test_models_api_accepts_models_slug_shape(monkeypatch):
    """bigmodel.cn's /api/v1/models (Codex-CLI gateway) answers `models[].slug`.

    Confirmed live against the vendor on 2026-09-01: HTTP 200 with
    `{"models": [{"slug": "glm-5.3", ...}]}`, not OpenAI's `data[].id`.
    """
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://open.bigmodel.cn/api/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "glm-5.3", "display_name": "glm-5.3"},
                    {"slug": "glm-5.3-flash", "display_name": "glm-5.3-flash"},
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mock_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(config_routes.httpx, "AsyncClient", mock_async_client)

    app_transport = httpx.ASGITransport(app=create_app())
    async with original_async_client(
        transport=app_transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == {"models": ["glm-5.3", "glm-5.3-flash"]}


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


async def _test_gpu_state_api_reports_and_retries_cleanup(tmp_path, monkeypatch):
    # "清理失败" is a state the user has to be able to see and act on, not just a
    # sentence buried in one job's error text. One status shape covers every
    # owner of the card, so the page cannot call the GPU free on the strength of
    # the ASR worker alone.
    from core import resources

    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    attempts: list[int] = []

    class _Immortal:
        pid = 4321

        def poll(self):
            return None

    def stubborn_stopper(_process, job_handle):
        attempts.append(1)
        return resources.StopOutcome(
            process_state="alive",
            job_handle=job_handle,
            detail="child process is alive after terminate and kill",
        )

    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    resources.register(
        resource_id="asr-gpu-worker:test",
        generation=1,
        kind="asr_worker",
        description="asr-gpu-worker",
        process=_Immortal(),
        job_handle=7,
        stopper=stubborn_stopper,
        holds_gpu=True,
        state="stopping",
        process_state="alive",
        detail="child process is alive after terminate and kill",
    )
    monkeypatch.setattr(
        pm.subprocess_tools,
        "unreleased_children",
        lambda: [{"name": "ffmpeg.exe", "pid": 99, "state": "unknown"}],
    )

    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get("/api/gpu-state")
            assert response.status_code == 200
            payload = response.json()
            assert payload["state"] == "cleanup_failed"
            assert payload["gpu_blocked"] is True
            assert payload["asr"]["pid"] == 4321
            assert payload["pending_resources"][0]["kind"] == "asr_worker"
            assert payload["handle_open"] is True
            assert payload["orphan_media_children"][0]["name"] == "ffmpeg.exe"

            response = await client.post("/api/gpu-state/retry-cleanup")
            assert response.status_code == 200
            assert attempts == [1]  # the one entry point retried the record
            assert response.json()["state"] == "cleanup_failed"
    finally:
        resources.forget_all()
        await _reset_pm_state()


async def _test_gpu_state_and_cancel_stay_responsive_during_a_model_load(
    tmp_path,
    monkeypatch,
):
    """The status poll must never wait on a resource operation.

    The local backend's lifecycle lock is held across model load and health
    checks - up to the 300s startup timeout. The page polls /api/gpu-state every
    5 seconds *on the event loop*, so one blocking read there stalled every
    other request, 取消 included.
    """
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    holding = threading.Event()
    release = threading.Event()
    backend = LlamaCppServerBackend()
    backend._proc = SimpleNamespace(poll=lambda: None, kill=lambda: None)
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", backend)

    def hold_the_lock():
        # Stands in for a model load: `_ensure_server` holds this across the
        # spawn and the health-check poll, up to the 300s startup timeout.
        with backend._lock:
            holding.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold_the_lock, daemon=True)
    holder.start()
    try:
        assert holding.wait(2.0)
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
            started = time.perf_counter()
            state, cancelled = await asyncio.wait_for(
                asyncio.gather(
                    client.get("/api/gpu-state"),
                    client.delete(f"/api/jobs/{job.id}"),
                ),
                timeout=3.0,
            )
            elapsed = time.perf_counter() - started

        assert state.status_code == 200
        assert cancelled.status_code == 200
        assert elapsed < 1.0
    finally:
        release.set()
        holder.join(5.0)
        await _reset_pm_state()


async def _test_jobs_api_rejects_invalid_job_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()

    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            invalid_payloads = [
                {"video_paths": []},
                {"video_paths": ["sample.mp4"], "vad_threshold": 1.5},
                {"video_paths": ["sample.mp4"], "translation_max_workers": 0},
                {"video_paths": ["sample.mp4"], "translation_max_workers": 65},
                {"video_paths": ["sample.mp4"], "translation_max_workers": 99},
            ]
            for payload in invalid_payloads:
                response = await client.post("/api/jobs", json=payload)
                assert response.status_code == 422

        assert await pm.list_jobs() == []
    finally:
        await _reset_pm_state()


async def _test_jobs_snapshot_saved_translation_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    # POST /api/jobs preflights the translation config; pin a key so the test
    # exercises snapshotting rather than the machine's .env.
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("ASR_CONTEXT", "小那海")
    monkeypatch.setenv("TRANSLATION_GLOSSARY", "ねこ-猫")
    monkeypatch.setenv("TARGET_LANG", "繁體中文")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
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

            response = await client.get(f"/api/jobs/{job_id}")
            assert response.status_code == 200
            spec = response.json()["spec"]

        assert "asr_context" not in spec
        assert spec["translation_glossary"] == "ねこ-猫"
        assert spec["target_lang"] == "繁體中文"
        assert spec["llm_reasoning_effort"] == "low"
    finally:
        await _reset_pm_state()


async def _test_jobs_api_retry_cancelled_job(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setenv("API_KEY", "test-key")
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


async def _test_open_routes_are_limited_to_job_paths(tmp_path, monkeypatch):
    from web.routes import files as files_routes

    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    other_video = tmp_path / "other.mp4"
    other_video.write_bytes(b"other")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    artifact_path = output_dir / "sample.srt"
    artifact_path.write_text("1\n", encoding="utf-8")
    unrelated_path = tmp_path / "secret.txt"
    unrelated_path.write_text("secret", encoding="utf-8")
    opened: list[tuple[str, str]] = []
    app = create_app()

    class DummyPopen:
        def __init__(self, args):
            opened.append(("popen", str(args[-1])))

    if os.name == "nt":
        monkeypatch.setattr(
            files_routes.os,
            "startfile",
            lambda path: opened.append(("startfile", str(path))),
            raising=False,
        )
    else:
        monkeypatch.setattr(files_routes.subprocess, "Popen", DummyPopen)

    try:
        jobs = await pm.create_job(
            pm.JobSpec(
                video_paths=[str(video_path)],
                output_dir=str(output_dir),
            )
        )
        job = jobs[0]
        async with pm._state_lock:
            job.status = "done"
            job.output_generation = 0  # legacy artifact authorization fixture
            job.output_target = ""
            job.artifacts = ["sample.srt"]
            pm._jobs[job.id] = job

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            allowed_video = await client.post(
                "/api/open-video",
                params={"job_id": job.id, "path": str(video_path)},
            )
            blocked_video = await client.post(
                "/api/open-video",
                params={"job_id": job.id, "path": str(other_video)},
            )
            allowed_folder = await client.post(
                "/api/open-folder",
                params={"job_id": job.id, "path": "sample.srt"},
            )
            allowed_artifact = await client.post(
                "/api/open-artifact",
                params={"job_id": job.id, "path": "sample.srt"},
            )
            blocked_artifact = await client.post(
                "/api/open-artifact",
                params={"job_id": job.id, "path": str(unrelated_path)},
            )
            blocked_folder = await client.post(
                "/api/open-folder",
                params={"job_id": job.id, "path": str(unrelated_path)},
            )

        assert allowed_video.status_code == 200
        assert blocked_video.status_code == 403
        assert allowed_folder.status_code == 200
        assert allowed_artifact.status_code == 200
        assert blocked_artifact.status_code == 403
        assert blocked_folder.status_code == 403
        open_kind = "startfile" if os.name == "nt" else "popen"
        assert (open_kind, str(video_path.resolve())) in opened
        assert (open_kind, str(artifact_path.resolve())) in opened
        assert (open_kind, str(output_dir.resolve())) in opened
    finally:
        await _reset_pm_state()


async def _test_open_folder_allows_default_video_directory(tmp_path, monkeypatch):
    """With no output_dir, completed subtitles live beside the source video."""
    from web.routes import files as files_routes

    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(files_routes, "PROJECT_ROOT", project_root)
    await _reset_pm_state()
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "sample.mp4"
    video_path.write_bytes(b"video")
    artifact_path = video_dir / "sample.srt"
    artifact_path.write_text("1\n", encoding="utf-8")
    outside_artifact = tmp_path / "outside.srt"
    outside_artifact.write_text("secret\n", encoding="utf-8")
    opened: list[str] = []

    if os.name == "nt":
        monkeypatch.setattr(
            files_routes.os,
            "startfile",
            lambda path: opened.append(str(path)),
            raising=False,
        )
    else:
        monkeypatch.setattr(
            files_routes.subprocess,
            "Popen",
            lambda args: opened.append(str(args[-1])),
        )

    try:
        jobs = await pm.create_job(
            pm.JobSpec(video_paths=[str(video_path)], output_dir=None)
        )
        job = jobs[0]
        async with pm._state_lock:
            job.status = "done"
            job.output_generation = 0  # legacy paths, including a tampered entry
            job.output_target = ""
            job.artifacts = [str(artifact_path), str(outside_artifact)]
            pm._jobs[job.id] = job

        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            allowed = await client.post(
                "/api/open-folder",
                params={"job_id": job.id, "path": str(artifact_path)},
            )
            blocked = await client.post(
                "/api/open-folder",
                params={"job_id": job.id, "path": str(outside_artifact)},
            )

        assert allowed.status_code == 200
        assert blocked.status_code == 403
        assert opened == [str(video_dir.resolve())]
    finally:
        await _reset_pm_state()


async def _test_downloading_an_artifact_follows_the_same_rule_as_opening_it(
    tmp_path, monkeypatch
):
    """Opening and downloading ask the same question, so they share a resolver.

    The download route kept its own pair of roots - the job's output_dir and
    the project - and omitted the directory the pipeline actually writes to
    when no output_dir is set: the one beside the source video. A film outside
    the project produced subtitles the page would happily open and then refuse
    to download, which reads as data loss to the person who ran the job.
    """
    from web.routes import files as files_routes

    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(files_routes, "PROJECT_ROOT", project_root)
    await _reset_pm_state()
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "sample.mp4"
    video_path.write_bytes(b"video")
    artifact_path = video_dir / "sample.srt"
    artifact_path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n", encoding="utf-8"
    )
    outside_artifact = tmp_path / "outside.srt"
    outside_artifact.write_text("secret\n", encoding="utf-8")

    try:
        jobs = await pm.create_job(
            pm.JobSpec(video_paths=[str(video_path)], output_dir=None)
        )
        job = jobs[0]
        async with pm._state_lock:
            job.status = "done"
            job.output_generation = 0  # legacy paths, including a tampered entry
            job.output_target = ""
            job.artifacts = [str(artifact_path), str(outside_artifact)]
            pm._jobs[job.id] = job

        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            downloaded = await client.get(f"/api/output/{job.id}/sample.srt")
            unauthorized = await client.get(f"/api/output/{job.id}/outside.srt")
            missing = await client.get(f"/api/output/{job.id}/never-written.srt")

        assert downloaded.status_code == 200
        assert "こんにちは" in downloaded.content.decode("utf-8")
        # Recorded as an artifact, but outside every root this job may write to:
        # the traversal guard still refuses it.
        assert unauthorized.status_code == 404
        assert missing.status_code == 404
    finally:
        await _reset_pm_state()


async def _test_jobs_api_rejects_a_job_that_cannot_translate(tmp_path, monkeypatch):
    """A forgotten API key is worth 400 at submit, not ten minutes of ASR."""
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("API_KEY", "")
    await _reset_pm_state()

    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            rejected = await client.post(
                "/api/jobs",
                json={"video_paths": ["sample.mp4"]},
            )
            assert rejected.status_code == 400
            detail = rejected.json()["detail"]
            assert "API Key" in detail and "翻译设置" in detail
            assert await pm.list_jobs() == []

            # The same job with translation off has nothing to preflight.
            accepted = await client.post(
                "/api/jobs",
                json={"video_paths": ["sample.mp4"], "skip_translation": True},
            )
            assert accepted.status_code == 201
    finally:
        await _reset_pm_state()


async def _test_retry_rejects_and_then_rereads_translation_settings(tmp_path, monkeypatch):
    """Retry runs with the settings in effect now, not the ones that failed."""
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
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
            assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 200

            queued_spec = (await client.get(f"/api/jobs/{job_id}")).json()["spec"]
            assert queued_spec["llm_reasoning_effort"] == "low"

            # Clearing the key blocks the retry with the same actionable message.
            monkeypatch.setenv("API_KEY", "")
            blocked = await client.post(f"/api/jobs/{job_id}/retry")
            assert blocked.status_code == 400
            assert "API Key" in blocked.json()["detail"]

            monkeypatch.setenv("API_KEY", "test-key")
            monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
            retried = await client.post(f"/api/jobs/{job_id}/retry")
            assert retried.status_code == 200
            spec = retried.json()["spec"]
            assert spec["llm_reasoning_effort"] == "high"
    finally:
        await _reset_pm_state()
