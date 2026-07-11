from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Any


class MemoryMonitorError(RuntimeError):
    pass


_SHARED_VRAM_BASELINE_BY_PID: dict[int, float] = {}


def physical_ram_snapshot(ratio: float = 0.95) -> dict[str, float]:
    import psutil

    memory = psutil.virtual_memory()
    scale = 1024 * 1024
    total_mb = float(memory.total) / scale
    available_mb = float(memory.available) / scale
    used_mb = max(0.0, total_mb - available_mb)
    effective_ratio = min(1.0, max(0.1, float(ratio)))
    return {
        "physical_ram_total_mb": round(total_mb, 1),
        "physical_ram_available_mb": round(available_mb, 1),
        "physical_ram_used_mb": round(used_mb, 1),
        "physical_ram_budget_mb": round(total_mb * effective_ratio, 1),
        "physical_ram_ratio": round(effective_ratio, 4),
    }


def _windows_shared_vram_bytes(pid: int) -> int:
    pdh = ctypes.WinDLL("pdh.dll")
    query = wintypes.HANDLE()
    counter = wintypes.HANDLE()

    class _ValueUnion(ctypes.Union):
        _fields_ = [
            ("long_value", wintypes.LONG),
            ("double_value", ctypes.c_double),
            ("large_value", ctypes.c_longlong),
            ("wide_string_value", wintypes.LPWSTR),
        ]

    class _CounterValue(ctypes.Structure):
        _anonymous_ = ("value",)
        _fields_ = [("status", wintypes.DWORD), ("value", _ValueUnion)]

    class _CounterValueItem(ctypes.Structure):
        _fields_ = [("name", wintypes.LPWSTR), ("value", _CounterValue)]

    pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
    pdh.PdhOpenQueryW.restype = wintypes.LONG
    pdh.PdhAddEnglishCounterW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCWSTR,
        ctypes.c_size_t,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    pdh.PdhAddEnglishCounterW.restype = wintypes.LONG
    pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
    pdh.PdhCollectQueryData.restype = wintypes.LONG
    pdh.PdhGetFormattedCounterArrayW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    pdh.PdhGetFormattedCounterArrayW.restype = wintypes.LONG
    pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
    pdh.PdhCloseQuery.restype = wintypes.LONG

    def require_ok(status: int, operation: str) -> None:
        if int(status) != 0:
            raise MemoryMonitorError(f"Windows shared VRAM monitor failed: {operation}=0x{int(status) & 0xFFFFFFFF:08x}")

    require_ok(pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)), "open_query")
    try:
        require_ok(
            pdh.PdhAddEnglishCounterW(
                query,
                r"\GPU Process Memory(*)\Shared Usage",
                0,
                ctypes.byref(counter),
            ),
            "add_counter",
        )
        require_ok(pdh.PdhCollectQueryData(query), "collect")
        buffer_size = wintypes.DWORD(0)
        item_count = wintypes.DWORD(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter,
            0x00000400,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            None,
        )
        if int(status) != -2147481646 and (int(status) & 0xFFFFFFFF) != 0x800007D2:
            require_ok(status, "size_counter_array")
        if buffer_size.value == 0 or item_count.value == 0:
            return 0
        buffer = ctypes.create_string_buffer(buffer_size.value)
        require_ok(
            pdh.PdhGetFormattedCounterArrayW(
                counter,
                0x00000400,
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                buffer,
            ),
            "read_counter_array",
        )
        items = ctypes.cast(buffer, ctypes.POINTER(_CounterValueItem))
        prefix = f"pid_{int(pid)}_".lower()
        return sum(
            max(0, int(items[index].value.large_value))
            for index in range(int(item_count.value))
            if str(items[index].name or "").lower().startswith(prefix)
            and int(items[index].value.status) == 0
        )
    finally:
        pdh.PdhCloseQuery(query)


def shared_vram_snapshot(*, pid: int | None = None, required: bool = False) -> dict[str, Any]:
    if os.name != "nt":
        return {"shared_vram_mb": 0.0, "shared_vram_monitor": "not_applicable"}
    try:
        used_bytes = _windows_shared_vram_bytes(int(pid or os.getpid()))
    except Exception as exc:
        if required:
            if isinstance(exc, MemoryMonitorError):
                raise
            raise MemoryMonitorError(f"Windows shared VRAM monitor failed: {exc}") from exc
        return {"shared_vram_mb": None, "shared_vram_monitor": "unavailable"}
    return {
        "shared_vram_mb": round(used_bytes / (1024 * 1024), 3),
        "shared_vram_monitor": "windows_pdh_gpu_process_memory",
    }


def runtime_memory_snapshot(*, require_shared_vram: bool = False) -> dict[str, Any]:
    try:
        ratio = float(os.getenv("ASR_STAGE_WORKER_RAM_RATIO", "0.95"))
    except ValueError:
        ratio = 0.95
    pid = os.getpid()
    shared = shared_vram_snapshot(pid=pid, required=require_shared_vram)
    raw_shared_mb = shared.get("shared_vram_mb")
    if isinstance(raw_shared_mb, (int, float)):
        baseline_mb = _SHARED_VRAM_BASELINE_BY_PID.setdefault(pid, float(raw_shared_mb))
        shared = {
            **shared,
            "shared_vram_raw_mb": round(float(raw_shared_mb), 3),
            "shared_vram_baseline_mb": round(baseline_mb, 3),
            "shared_vram_mb": round(max(0.0, float(raw_shared_mb) - baseline_mb), 3),
        }
    return {**physical_ram_snapshot(ratio), **shared}
