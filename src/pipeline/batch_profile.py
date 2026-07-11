from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from utils.runtime_paths import runtime_path


PROFILE_SCHEMA = "gpu_inference_batch_profiles_v2"
PROFILE_VERSION = 2
_LOCK = threading.RLock()


def enabled() -> bool:
    return os.getenv("GPU_BATCH_PROFILE_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def growth_threshold() -> float:
    try:
        value = float(os.getenv("GPU_BATCH_PROFILE_GROWTH_THRESHOLD", "0.80"))
    except (TypeError, ValueError):
        value = 0.80
    return min(0.95, max(0.10, value))


def profile_path() -> Path:
    raw = os.getenv(
        "GPU_BATCH_PROFILE_PATH",
        "tmp/cache/gpu_batch_profiles.json",
    ).strip()
    return runtime_path(raw or "tmp/cache/gpu_batch_profiles.json")


def identity_key(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _empty_payload() -> dict[str, Any]:
    return {
        "schema": PROFILE_SCHEMA,
        "version": PROFILE_VERSION,
        "profiles": {},
    }


def _load_payload(path: Path | None = None) -> dict[str, Any]:
    target = path or profile_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return _empty_payload()
    if not isinstance(payload, dict) or payload.get("schema") != PROFILE_SCHEMA:
        return _empty_payload()
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        payload["profiles"] = {}
    return payload


def _write_payload(payload: dict[str, Any], path: Path | None = None) -> None:
    target = path or profile_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(target)


def recommendation(
    identity: Mapping[str, Any],
    *,
    heuristic_batch: int,
    max_batch: int,
) -> tuple[int, dict[str, Any]]:
    heuristic = max(1, int(heuristic_batch))
    maximum = max(heuristic, int(max_batch))
    if not enabled():
        return heuristic, {}
    with _LOCK:
        payload = _load_payload()
        entry = payload["profiles"].get(identity_key(identity))
    if not isinstance(entry, dict):
        return heuristic, {}
    try:
        recommended = int(entry.get("recommended_batch") or heuristic)
    except (TypeError, ValueError):
        recommended = heuristic
    return max(1, min(maximum, recommended)), dict(entry)


def record_success(
    identity: Mapping[str, Any],
    *,
    batch_size: int,
    peak_allocated_mb: float,
    budget_mb: float,
    max_batch: int,
) -> dict[str, Any]:
    if not enabled():
        return {}
    batch = max(1, int(batch_size))
    maximum = max(batch, int(max_batch))
    peak = max(0.0, float(peak_allocated_mb))
    budget = max(0.0, float(budget_mb))
    utilization = peak / budget if budget > 0.0 else 1.0
    key = identity_key(identity)
    with _LOCK:
        payload = _load_payload()
        old = payload["profiles"].get(key)
        entry = dict(old) if isinstance(old, dict) else {}
        previous_safe = max(0, int(entry.get("safe_batch") or 0))
        safe_batch = max(previous_safe, batch)
        unsafe_raw = entry.get("unsafe_batch")
        try:
            unsafe_batch = max(1, int(unsafe_raw)) if unsafe_raw is not None else None
        except (TypeError, ValueError):
            unsafe_batch = None
        if unsafe_batch is not None and batch >= unsafe_batch:
            unsafe_batch = None

        recommended = safe_batch
        if utilization < growth_threshold():
            upper = unsafe_batch if unsafe_batch is not None else maximum + 1
            if upper - safe_batch > 1:
                recommended = (safe_batch + upper) // 2
        if unsafe_batch is not None:
            recommended = min(recommended, max(1, unsafe_batch - 1))
        recommended = max(1, min(maximum, recommended))
        entry.update(
            {
                "identity": dict(identity),
                "safe_batch": safe_batch,
                "unsafe_batch": unsafe_batch,
                "recommended_batch": recommended,
                "last_batch": batch,
                "last_peak_allocated_mb": round(peak, 1),
                "last_budget_mb": round(budget, 1),
                "last_utilization": round(utilization, 4),
                "last_result": "success",
                "updated_ts": round(time.time(), 3),
            }
        )
        payload["profiles"][key] = entry
        _write_payload(payload)
    return dict(entry)


def record_oom(
    identity: Mapping[str, Any],
    *,
    batch_size: int,
    max_batch: int,
) -> dict[str, Any]:
    if not enabled():
        return {}
    batch = max(1, int(batch_size))
    maximum = max(batch, int(max_batch))
    key = identity_key(identity)
    with _LOCK:
        payload = _load_payload()
        old = payload["profiles"].get(key)
        entry = dict(old) if isinstance(old, dict) else {}
        previous_oom = entry.get("unsafe_batch")
        try:
            unsafe_batch = (
                min(batch, int(previous_oom)) if previous_oom is not None else batch
            )
        except (TypeError, ValueError):
            unsafe_batch = batch
        try:
            safe_batch = max(0, int(entry.get("safe_batch") or 0))
        except (TypeError, ValueError):
            safe_batch = 0
        if safe_batch > 0 and unsafe_batch - safe_batch > 1:
            recommended = (safe_batch + unsafe_batch) // 2
        elif safe_batch > 0:
            recommended = safe_batch
        else:
            recommended = max(1, batch // 2)
        recommended = min(recommended, max(1, unsafe_batch - 1), maximum)
        entry.update(
            {
                "identity": dict(identity),
                "safe_batch": safe_batch,
                "unsafe_batch": unsafe_batch,
                "recommended_batch": max(1, recommended),
                "last_batch": batch,
                "last_result": "oom",
                "updated_ts": round(time.time(), 3),
            }
        )
        payload["profiles"][key] = entry
        _write_payload(payload)
    return dict(entry)
