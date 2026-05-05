from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core import events
from core.config import DEFAULT_SETTINGS, load_config
from web import broadcaster, pipeline_manager
from web.routes import config, events as event_routes, files, jobs


def _events_port() -> int:
    try:
        return int(
            os.getenv(
                "JAVTRANS_EVENTS_PORT",
                DEFAULT_SETTINGS.get("JAVTRANS_EVENTS_PORT", "17322"),
            )
        )
    except ValueError:
        return 17322


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    load_config()
    listener_task: asyncio.Task | None = None
    worker_tasks: list[asyncio.Task] = []
    try:
        port = _events_port()
        listener_task = asyncio.create_task(
            broadcaster.tcp_listener("127.0.0.1", port)
        )
        await asyncio.sleep(0.05)
        events.configure_sink(f"tcp:127.0.0.1:{port}")
        await pipeline_manager.load_jobs()
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
    app = FastAPI(title="JAVTrans Web", lifespan=_lifespan)
    app.include_router(jobs.router, prefix="/api")
    app.include_router(event_routes.router, prefix="/api")
    app.include_router(files.router, prefix="/api")
    app.include_router(config.router, prefix="/api")

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
