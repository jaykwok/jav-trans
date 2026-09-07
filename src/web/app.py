from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse, Response

from core import events
from core.config import DEFAULT_SETTINGS, load_config
from core.typed_config import env_int
from utils.runtime_paths import resource_root
from web import broadcaster, pipeline_manager
from web.job_store import JobPersistenceError
from web.routes import config, events as event_routes, files, jobs


log = logging.getLogger(__name__)


def _events_port() -> int:
    return env_int(
        "JAV_TRANS_EVENTS_PORT",
        int(DEFAULT_SETTINGS.get("JAV_TRANS_EVENTS_PORT", "2234")),
        minimum=1,
        maximum=65535,
    )


def _warn_translation_budget() -> None:
    """Say it once at startup, where a config mistake is still cheap to fix.

    Import is local because this is the only thing app startup needs from the
    translation stack, and pulling the profiles in at module scope would put
    them on the import path of every web test.
    """
    try:
        from llm.preflight import translation_budget_warnings

        for warning in translation_budget_warnings():
            print(f"[WARN] {warning}", flush=True)
    except Exception as exc:  # never let a warning stop the server
        log.debug("translation budget warning skipped: %r", exc)


def _warn_configuration_problems() -> None:
    """Scan common inexpensive settings and list all problems read so far.

    Settings owned by lazily loaded stages report their problems when first
    read. Startup does not import every model or exhaust the configuration.
    """
    try:
        from core.typed_config import configuration_problems
        from pipeline import batch_profile
        from subtitles.options import SubtitleOptions

        try:
            SubtitleOptions.from_env()
        except ValueError as exc:
            # A version stamp, which is refused rather than corrected.
            print(f"[WARN] {exc}", flush=True)
        from asr import decode_guard

        decode_guard.loop_guard_enabled()
        decode_guard.tokens_per_second_ceiling()
        decode_guard.loop_budget_fraction()
        decode_guard.explicit_token_cap()
        batch_profile.enabled()
        batch_profile.growth_threshold()
        batch_profile.max_entries()

        from core.typed_config import FALL_BACK, env_float
        from llm.backends.llamacpp_server import server_limits

        server_limits()
        env_int("JAV_TRANS_PORT", 2233, minimum=1, maximum=65535)
        _events_port()
        for field, default in (
            ("LOCAL_RETIRE_WAIT_S", 600.0),
            ("LOCAL_BACKEND_WAIT_TIMEOUT_S", 600.0),
            ("LOCAL_BACKEND_CLOSE_WAIT_S", 15.0),
        ):
            env_float(field, default, minimum=0.001, out_of_range=FALL_BACK)

        for problem in configuration_problems():
            print(f"[WARN] 配置项 {problem}", flush=True)
    except Exception as exc:  # never let a warning stop the server
        log.debug("configuration problem scan skipped: %r", exc)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    load_config()
    _warn_configuration_problems()
    _warn_translation_budget()
    listener_task: asyncio.Task | None = None
    worker_tasks: list[asyncio.Task] = []
    try:
        port = _events_port()
        listener_task = asyncio.create_task(
            broadcaster.tcp_listener("127.0.0.1", port)
        )

        def _on_listener_done(task: asyncio.Task) -> None:
            # Surface bind/port-in-use failures that would otherwise vanish
            # as an unretrieved exception on a background task.
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                log.error(
                    "SSE TCP listener on port %s failed",
                    port,
                    exc_info=exc,
                )

        listener_task.add_done_callback(_on_listener_done)
        await asyncio.sleep(0.05)
        events.configure_sink(f"tcp:127.0.0.1:{port}")
        await pipeline_manager.load_jobs()
        await pipeline_manager.reclaim_stale_audio_partials()
        worker_tasks = await pipeline_manager.start_workers()
        yield
    finally:
        events.configure_sink("")
        for task in worker_tasks:
            task.cancel()
        await pipeline_manager.shutdown_executor()
        if listener_task is not None:
            listener_task.cancel()
        tasks = [*worker_tasks]
        if listener_task is not None:
            tasks.append(listener_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def create_app() -> FastAPI:
    app = FastAPI(title="jav-trans Web", lifespan=_lifespan)

    @app.exception_handler(JobPersistenceError)
    async def persistence_error(_request, exc: JobPersistenceError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(pipeline_manager.artifact_store.OutputsSupersededError)
    async def superseded_output(_request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(pipeline_manager.artifact_store.OutputExportError)
    @app.exception_handler(pipeline_manager.artifact_store.OutputCommitError)
    async def export_error(_request, exc):
        conflict = isinstance(exc, pipeline_manager.artifact_store.OutputExportError) and exc.superseded
        return JSONResponse(status_code=409 if conflict else 503, content={"detail": str(exc)})

    @app.exception_handler(pipeline_manager.artifact_store.OutputIntegrityError)
    async def invalid_output(_request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    app.include_router(jobs.router, prefix="/api")
    app.include_router(event_routes.router, prefix="/api")
    app.include_router(files.router, prefix="/api")
    app.include_router(config.router, prefix="/api")

    assets_dir = resource_root() / "src" / "assets"
    if not assets_dir.exists():
        assets_dir = Path(__file__).resolve().parents[1] / "assets"

    def _asset_response(relative_path: str, media_type: str) -> Response:
        return Response((assets_dir / relative_path).read_bytes(), media_type=media_type)

    @app.get("/assets/images/icon.png", include_in_schema=False)
    async def app_icon_png() -> Response:
        return _asset_response("images/icon.png", "image/png")

    @app.get("/assets/images/icon.ico", include_in_schema=False)
    async def app_icon_ico() -> Response:
        return _asset_response("images/icon.ico", "image/x-icon")

    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    static_dir = resource_root() / "src" / "web" / "static"
    if not static_dir.exists():
        static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
