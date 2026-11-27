from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from core import updater
from core.config import update_config_file
from web import pipeline_manager
from web.routes import config as config_routes

router = APIRouter()


class AutoCheckPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


@router.get("/update/status")
async def get_update_status() -> dict:
    return updater.get_update_manager().snapshot()


@router.post("/update/check")
async def check_for_update() -> dict:
    return await asyncio.to_thread(updater.get_update_manager().check_now)


@router.post("/update/start")
async def start_update() -> dict:
    try:
        return await pipeline_manager.run_under_update_gate(
            updater.get_update_manager().start_update
        )
    except updater.UpdateError as exc:
        detail = str(exc)
        status_code = 409 if ("任务" in detail or "窗口" in detail) else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc


@router.post("/update/cancel")
async def cancel_update() -> dict:
    return updater.get_update_manager().cancel_update()


@router.post("/update/auto-check")
async def set_auto_update_check(body: AutoCheckPreference) -> dict:
    value = "0" if body.enabled else "1"
    try:
        await asyncio.to_thread(
            update_config_file,
            {"JAV_SUB_DISABLE_UPDATE_CHECK": value},
            path=config_routes._config_file_path(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail="保存自动更新设置失败，请检查程序目录的写入权限和磁盘空间。",
        ) from exc
    os.environ["JAV_SUB_DISABLE_UPDATE_CHECK"] = value
    return updater.get_update_manager().snapshot()
