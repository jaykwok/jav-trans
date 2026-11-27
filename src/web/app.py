from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse, Response

from core import events, updater
from core.config import DEFAULT_SETTINGS, config_load_error, load_config
from core.diagnostics import (
    attach_asyncio_exception_handler,
    configure_process_logging,
    current_process_log_hint,
    format_exception_chain,
)
from core.typed_config import env_int
from utils.runtime_paths import resource_root
from web import broadcaster, pipeline_manager
from web.job_store import JobPersistenceError
from web.routes import config, events as event_routes, files, jobs, update as update_routes
from web.security import LocalSecurityMiddleware


log = logging.getLogger(__name__)


def _events_port() -> int:
    return env_int(
        "JAV_SUB_EVENTS_PORT",
        int(DEFAULT_SETTINGS.get("JAV_SUB_EVENTS_PORT", "2234")),
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
            log.warning("%s", warning)
            print(f"[WARN] {warning}", flush=True)
    except Exception as exc:  # never let a warning stop the server
        log.warning("translation budget warning skipped: %r", exc)


def _warn_configuration_problems() -> None:
    """Scan common inexpensive settings and list all problems read so far.

    Settings owned by lazily loaded stages report their problems when first
    read. Startup does not import every model or exhaust the configuration.
    """
    try:
        from core.typed_config import configuration_problems
        from subtitles.options import SubtitleOptions

        try:
            SubtitleOptions.from_env()
        except ValueError as exc:
            # A version stamp, which is refused rather than corrected.
            log.warning("%s", exc)
            print(f"[WARN] {exc}", flush=True)
        from asr import decode_guard

        decode_guard.loop_guard_enabled()
        decode_guard.tokens_per_second_ceiling()
        decode_guard.loop_budget_fraction()
        decode_guard.explicit_token_cap()

        from core.typed_config import FALL_BACK, env_float
        from llm.backends.llamacpp_server import server_limits

        server_limits()
        env_int("JAV_SUB_PORT", 2233, minimum=1, maximum=65535)
        _events_port()
        for field, default in (
            ("LOCAL_RETIRE_WAIT_S", 600.0),
            ("LOCAL_BACKEND_WAIT_TIMEOUT_S", 600.0),
            ("LOCAL_BACKEND_CLOSE_WAIT_S", 15.0),
        ):
            env_float(field, default, minimum=0.001, out_of_range=FALL_BACK)

        config_err = config_load_error()
        if config_err:
            log.warning("%s", config_err)
            print(f"[WARN] {config_err}", flush=True)
        for problem in configuration_problems():
            log.warning("配置项 %s", problem)
            print(f"[WARN] 配置项 {problem}", flush=True)
    except Exception as exc:  # never let a warning stop the server
        log.warning("configuration problem scan skipped: %r", exc)


async def _cancel_and_gather(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if not tasks:
        return
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for task, result in zip(tasks, results):
        if isinstance(result, BaseException) and not isinstance(
            result, asyncio.CancelledError
        ):
            log.error(
                "Background task %s ended with an error during shutdown",
                task.get_name(),
                exc_info=result,
            )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_process_logging("app.log")
    attach_asyncio_exception_handler(asyncio.get_running_loop())
    await asyncio.to_thread(load_config)
    _warn_configuration_problems()
    _warn_translation_budget()
    listener_task: asyncio.Task | None = None
    worker_tasks: list[asyncio.Task] = []
    try:
        port = _events_port()
        try:
            _server, listener_task = await broadcaster.start_tcp_listener("127.0.0.1", port)
        except OSError:
            log.exception("Failed to bind SSE TCP listener on 127.0.0.1:%s", port)
        else:

            def _on_listener_done(task: asyncio.Task) -> None:
                # Surface unexpected runtime failures on the listener task.
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
            events.configure_sink(f"tcp:127.0.0.1:{port}")
        await pipeline_manager.load_jobs()
        await pipeline_manager.reclaim_stale_audio_partials()
        worker_tasks = await pipeline_manager.start_workers()
        if str(os.getenv("TRANSLATION_BACKEND", "") or "").strip().lower() == "llamacpp":
            from llm.backends import llamacpp_runtime

            llamacpp_runtime.start_managed_llama_server_prefetch()
        if updater.should_auto_check_on_startup():
            updater.get_update_manager().start_background_check()
        yield
    finally:
        # 1. Stop accepting new pipeline events before tearing down the listener.
        events.configure_sink("")
        if listener_task is not None:
            await _cancel_and_gather([listener_task])
        # 2. Signal all in-flight work, then stop worker & progress tasks cleanly.
        pipeline_manager.signal_all_cancel_events()
        pending_progress = list(broadcaster._pending_progress_tasks)
        await _cancel_and_gather([*worker_tasks, *pending_progress])
        # 3. Drain queued loggers, flush durable job state, and shut down executor.
        await pipeline_manager.shutdown_executor()


def create_app() -> FastAPI:
    app = FastAPI(title="jav-sub Web", lifespan=_lifespan)
    app.add_middleware(LocalSecurityMiddleware)

    @app.exception_handler(JobPersistenceError)
    async def persistence_error(request: Request, exc: JobPersistenceError):
        log.error(
            "Job persistence error on %s %s: %s",
            request.method,
            request.url.path,
            format_exception_chain(exc),
            exc_info=exc,
        )
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(pipeline_manager.artifact_store.OutputsSupersededError)
    async def superseded_output(request: Request, exc):
        log.warning("Superseded output on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(pipeline_manager.JobOutputConflict)
    async def output_conflict(_request: Request, exc: pipeline_manager.JobOutputConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc), "job_id": exc.job_id})

    @app.exception_handler(pipeline_manager.JobStateConflictError)
    async def state_conflict(_request: Request, exc: pipeline_manager.JobStateConflictError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(pipeline_manager.UpdateInProgressError)
    async def update_in_progress(_request: Request, exc: pipeline_manager.UpdateInProgressError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(pipeline_manager.artifact_store.OutputExportError)
    @app.exception_handler(pipeline_manager.artifact_store.OutputCommitError)
    async def export_error(request: Request, exc):
        log.error(
            "Output delivery error on %s %s: %s",
            request.method,
            request.url.path,
            format_exception_chain(exc),
            exc_info=exc,
        )
        conflict = isinstance(exc, pipeline_manager.artifact_store.OutputExportError) and exc.superseded
        return JSONResponse(status_code=409 if conflict else 503, content={"detail": str(exc)})

    @app.exception_handler(pipeline_manager.artifact_store.OutputIntegrityError)
    async def invalid_output(request: Request, exc):
        log.error(
            "Output integrity error on %s %s: %s",
            request.method,
            request.url.path,
            format_exception_chain(exc),
            exc_info=exc,
        )
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    f"服务器内部错误（{type(exc).__name__}）：{exc}"
                    f"（详细日志见 {current_process_log_hint()}）"
                )
            },
        )

    app.include_router(jobs.router, prefix="/api")
    app.include_router(event_routes.router, prefix="/api")
    app.include_router(files.router, prefix="/api")
    app.include_router(config.router, prefix="/api")
    app.include_router(update_routes.router, prefix="/api")

    assets_dir = resource_root() / "src" / "assets"
    if not assets_dir.exists():
        assets_dir = Path(__file__).resolve().parents[1] / "assets"

    def _asset_response(relative_path: str, media_type: str) -> Response:
        target = assets_dir / relative_path
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(
            target,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )

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
