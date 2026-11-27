from __future__ import annotations

import atexit
import ctypes
import os
import threading
from typing import Any

from core.typed_config import env_float


class MemoryMonitorError(RuntimeError):
    pass


_SHARED_VRAM_BASELINE_BY_PID: dict[int, float] = {}

_PDH_MORE_DATA = 0x800007D2
_PDH_FMT_LARGE = 0x00000400
_PDH_MAX_ARRAY_RETRIES = 3

if os.name == "nt":
    from ctypes import wintypes

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

    def _init_pdh_dll():
        pdh = ctypes.WinDLL("pdh.dll")
        pdh.PdhOpenQueryW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.c_size_t,
            ctypes.POINTER(wintypes.HANDLE),
        ]
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
        return pdh

    _PDH_DLL = _init_pdh_dll()
else:  # pragma: no cover - project targets Windows
    _PDH_DLL = None

_PDH_LOCK = threading.Lock()
_PDH_QUERY: Any = None
_PDH_COUNTER: Any = None
_PDH_QUERY_PID: int | None = None


def _pdh_require_ok(status: int, operation: str) -> None:
    if int(status) != 0:
        raise MemoryMonitorError(
            f"Windows shared VRAM monitor failed: {operation}=0x{int(status) & 0xFFFFFFFF:08x}"
        )


def _close_cached_pdh_query() -> None:
    global _PDH_QUERY, _PDH_COUNTER, _PDH_QUERY_PID
    with _PDH_LOCK:
        query = _PDH_QUERY
        _PDH_QUERY = None
        _PDH_COUNTER = None
        _PDH_QUERY_PID = None
        if query is not None and _PDH_DLL is not None:
            try:
                _PDH_DLL.PdhCloseQuery(query)
            except Exception:
                pass


atexit.register(_close_cached_pdh_query)


def _ensure_pdh_query_locked():
    global _PDH_QUERY, _PDH_COUNTER, _PDH_QUERY_PID
    current_pid = os.getpid()
    if _PDH_QUERY is not None and _PDH_COUNTER is not None and _PDH_QUERY_PID == current_pid:
        return _PDH_QUERY, _PDH_COUNTER
    if _PDH_QUERY is not None:
        try:
            _PDH_DLL.PdhCloseQuery(_PDH_QUERY)
        except Exception:
            pass
        _PDH_QUERY = None
        _PDH_COUNTER = None
    query = wintypes.HANDLE()
    counter = wintypes.HANDLE()
    _pdh_require_ok(_PDH_DLL.PdhOpenQueryW(None, 0, ctypes.byref(query)), "open_query")
    try:
        _pdh_require_ok(
            _PDH_DLL.PdhAddEnglishCounterW(
                query,
                r"\GPU Process Memory(*)\Shared Usage",
                0,
                ctypes.byref(counter),
            ),
            "add_counter",
        )
    except BaseException:
        try:
            _PDH_DLL.PdhCloseQuery(query)
        except Exception:
            pass
        raise
    _PDH_QUERY = query
    _PDH_COUNTER = counter
    _PDH_QUERY_PID = current_pid
    return query, counter


def _read_pdh_counter_array(counter, pid: int) -> int:
    buffer_size = wintypes.DWORD(0)
    item_count = wintypes.DWORD(0)
    status = _PDH_DLL.PdhGetFormattedCounterArrayW(
        counter,
        _PDH_FMT_LARGE,
        ctypes.byref(buffer_size),
        ctypes.byref(item_count),
        None,
    )
    if (int(status) & 0xFFFFFFFF) != _PDH_MORE_DATA:
        _pdh_require_ok(status, "size_counter_array")
    if buffer_size.value == 0 or item_count.value == 0:
        return 0

    for attempt in range(_PDH_MAX_ARRAY_RETRIES):
        buffer = ctypes.create_string_buffer(buffer_size.value)
        status = _PDH_DLL.PdhGetFormattedCounterArrayW(
            counter,
            _PDH_FMT_LARGE,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            buffer,
        )
        if int(status) == 0:
            items = ctypes.cast(buffer, ctypes.POINTER(_CounterValueItem))
            prefix = f"pid_{int(pid)}_".lower()
            return sum(
                max(0, int(items[index].value.large_value))
                for index in range(int(item_count.value))
                if str(items[index].name or "").lower().startswith(prefix)
                and int(items[index].value.status) == 0
            )
        if (int(status) & 0xFFFFFFFF) == _PDH_MORE_DATA and attempt + 1 < _PDH_MAX_ARRAY_RETRIES:
            if buffer_size.value == 0:
                return 0
            continue
        _pdh_require_ok(status, "read_counter_array")
    return 0


def _windows_shared_vram_bytes(pid: int) -> int:
    global _PDH_QUERY, _PDH_COUNTER, _PDH_QUERY_PID
    if _PDH_DLL is None:
        raise MemoryMonitorError("Windows PDH library unavailable")
    with _PDH_LOCK:
        for attempt in range(2):
            query, counter = _ensure_pdh_query_locked()
            status = _PDH_DLL.PdhCollectQueryData(query)
            if int(status) != 0:
                try:
                    _PDH_DLL.PdhCloseQuery(query)
                except Exception:
                    pass
                _PDH_QUERY = None
                _PDH_COUNTER = None
                _PDH_QUERY_PID = None
                if attempt == 0:
                    continue
                _pdh_require_ok(status, "collect")
            return _read_pdh_counter_array(counter, pid)
    return 0


def reset_shared_vram_baseline(pid: int | None = None) -> None:
    """Reset the shared-VRAM growth baseline so a persistent worker starts a new job fresh."""
    target_pid = os.getpid() if pid is None else int(pid)
    _SHARED_VRAM_BASELINE_BY_PID.pop(target_pid, None)


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
    ratio = env_float("ASR_STAGE_WORKER_RAM_RATIO", 0.95, minimum=0.05, maximum=1.0)
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
