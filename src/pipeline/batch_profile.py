from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from utils.runtime_paths import runtime_path


PROFILE_SCHEMA = "gpu_inference_batch_profiles_v3"
# v3 adds chunk geometry to the identity (see gpu_worker._profile_identity).
# Bumped rather than migrated: a v2 entry's safe_batch was measured under an
# unknown chunk length, so it is not a claim about any v3 identity.
#
# v4 drops `chunk_target_s` from that identity, and the bump is load-bearing for
# the same reason: every v3 entry was learned while chunks averaged the 20s
# target, and chunks now run to the 30s ceiling. Carrying a v3 `safe_batch` over
# would recommend a batch measured against two thirds of the real activation
# footprint - the exact OOM-per-job the geometry keys were added to stop.
#
# v5 is the decode budget: it stopped being a flat 128 and became
# `duration x ASR_DECODE_TOKENS_PER_SECOND`, which at 30s chunks is 316 rather
# than 128 tokens of KV cache per sequence. A v4 `safe_batch` was measured
# against a quarter of that and would OOM.
#
# v6 separates physical hardware identity from workload identity and keys the
# former by a hashed CUDA UUID (with PCI/fallback identities when unavailable).
# Old profiles are intentionally not migrated: they cannot distinguish two
# cards with the same model and VRAM.
#
# v7 requires evidence that the configured batch was actually exercised. A
# short/cache-heavy task previously recorded the configured ceiling as safe and
# used the artificially low peak to recommend an untested larger batch.
PROFILE_VERSION = 7
_LOCK = threading.RLock()
_DEFAULT_MAX_ENTRIES = 16


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


def max_entries() -> int:
    try:
        value = int(float(os.getenv("GPU_BATCH_PROFILE_MAX_ENTRIES", "16")))
    except (TypeError, ValueError):
        value = _DEFAULT_MAX_ENTRIES
    return max(1, value)


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
    if payload.get("version") != PROFILE_VERSION:
        # The bumps above are load-bearing, so the version has to actually be
        # read. It was written and never checked, which meant every "bumped
        # rather than migrated" claim in this file was decorative: entries stayed
        # in place and only a change in the identity keys retired any of them.
        # Found with a live file still stamped `version: 2` holding a
        # `safe_batch: 16` learned under an unknown chunk length.
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


def _prune_profiles(payload: dict[str, Any]) -> None:
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or len(profiles) <= max_entries():
        return

    def recency(item: tuple[str, Any]) -> tuple[float, str]:
        key, entry = item
        if not isinstance(entry, dict):
            return (0.0, key)
        try:
            updated = float(
                entry.get("last_used_ts") or entry.get("updated_ts") or 0.0
            )
        except (TypeError, ValueError):
            updated = 0.0
        return (updated, key)

    keep = {
        key
        for key, _entry in sorted(
            profiles.items(),
            key=recency,
            reverse=True,
        )[: max_entries()]
    }
    payload["profiles"] = {
        key: entry for key, entry in profiles.items() if key in keep
    }


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
        key = identity_key(identity)
        entry = payload["profiles"].get(key)
        if isinstance(entry, dict):
            entry = dict(entry)
            entry["last_used_ts"] = round(time.time(), 3)
            payload["profiles"][key] = entry
            _write_payload(payload)
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
        now = round(time.time(), 3)
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
                "updated_ts": now,
                "last_used_ts": now,
            }
        )
        payload["profiles"][key] = entry
        _prune_profiles(payload)
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
        # A batch that OOM'd retires every "safe" claim at or above it. Without
        # this a corrupt or hand-edited profile could keep proposing a value the
        # card has already refused.
        safe_batch = min(safe_batch, max(0, unsafe_batch - 1))
        if safe_batch > 0 and unsafe_batch - safe_batch > 1:
            recommended = (safe_batch + unsafe_batch) // 2
        elif safe_batch > 0:
            recommended = safe_batch
        else:
            recommended = max(1, batch // 2)
        recommended = min(recommended, max(1, unsafe_batch - 1), maximum)
        now = round(time.time(), 3)
        entry.update(
            {
                "identity": dict(identity),
                "safe_batch": safe_batch,
                "unsafe_batch": unsafe_batch,
                "recommended_batch": max(1, recommended),
                "last_batch": batch,
                "last_result": "oom",
                "updated_ts": now,
                "last_used_ts": now,
            }
        )
        payload["profiles"][key] = entry
        _prune_profiles(payload)
        _write_payload(payload)
    return dict(entry)
