from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from utils.runtime_paths import runtime_path


PROFILE_SCHEMA = "gpu_inference_batch_profiles_v1"
PROFILE_VERSION = 1
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
        previous_safe = max(0, int(entry.get("max_safe_batch") or 0))
        max_safe = max(previous_safe, batch)
        unsafe_raw = entry.get("min_oom_batch")
        try:
            min_oom = max(1, int(unsafe_raw)) if unsafe_raw is not None else None
        except (TypeError, ValueError):
            min_oom = None
        if min_oom is not None and batch >= min_oom:
            min_oom = None

        recommended = max_safe
        if utilization < growth_threshold():
            recommended = max_safe + 1
        if min_oom is not None:
            recommended = min(recommended, max(1, min_oom - 1))
        recommended = max(1, min(maximum, recommended))
        entry.update(
            {
                "identity": dict(identity),
                "max_safe_batch": max_safe,
                "min_oom_batch": min_oom,
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
        previous_oom = entry.get("min_oom_batch")
        try:
            min_oom = min(batch, int(previous_oom)) if previous_oom is not None else batch
        except (TypeError, ValueError):
            min_oom = batch
        try:
            max_safe = max(0, int(entry.get("max_safe_batch") or 0))
        except (TypeError, ValueError):
            max_safe = 0
        recommended = max_safe if max_safe > 0 else max(1, batch - 1)
        recommended = min(recommended, max(1, min_oom - 1), maximum)
        entry.update(
            {
                "identity": dict(identity),
                "max_safe_batch": max_safe,
                "min_oom_batch": min_oom,
                "recommended_batch": max(1, recommended),
                "last_batch": batch,
                "last_result": "oom",
                "updated_ts": round(time.time(), 3),
            }
        )
        payload["profiles"][key] = entry
        _write_payload(payload)
    return dict(entry)
