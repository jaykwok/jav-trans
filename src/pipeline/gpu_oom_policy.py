"""CUDA VRAM sizing, allocator cap, and per-stage OOM downshift policies for the ASR GPU worker."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from asr import vram_sizing as asr_vram_sizing
from core.diagnostics import is_fatal_cuda_error
from core.typed_config import env_float

_env_float = env_float

# Track the most recent per-process CUDA allocator cap applied inside this
# worker process so a later job requesting unlimited budget (budget_mb <= 0)
# explicitly resets the cap back to 1.0 instead of inheriting a stale cap.
_LAST_ALLOCATOR_FRACTION: float | None = None

_STAGE_BATCH_SETTINGS: dict[str, tuple[str, str]] = {
    "audio_chunking": ("ASR_FEATURE_BATCH_SIZE", "asr_feature_batch_size"),
    "subtitle_timing": ("ASR_ALIGN_BATCH_SIZE", "asr_align_batch_size"),
    "asr_text_transcribe": ("ASR_BATCH_SIZE", "asr_batch_size"),
}


def _is_oom_error(exc: BaseException, torch_module: Any | None) -> bool:
    if torch_module is not None:
        try:
            if isinstance(exc, torch_module.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass

    detail = repr(exc).lower()
    return (
        ("out of memory" in detail and ("cuda" in detail or "gpu" in detail))
        or "gpu vram budget exceeded" in detail
        or "vram budget exceeded" in detail
        or "cumemalloc" in detail
        or "shared vram spill" in detail
    )


def _is_ram_oom_error(exc: BaseException) -> bool:
    return "physical ram budget exceeded" in repr(exc).lower()


def _effective_stage_batch_size(env: dict[str, str], key: str) -> int | None:
    raw = str(env.get(key) or os.getenv(key, "auto")).strip()
    if raw and raw.lower() != "auto":
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return None
    return None


def _effective_asr_batch_size(env: dict[str, str]) -> int | None:
    return _effective_stage_batch_size(env, "ASR_BATCH_SIZE")


def _physical_vram_mb(torch_module: Any) -> float:
    try:
        if not torch_module.cuda.is_available():
            return 0.0
        device_index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(device_index)
        return float(properties.total_memory) / (1024 * 1024)
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return 0.0


def _gpu_device_name(torch_module: Any) -> str:
    try:
        if not torch_module.cuda.is_available():
            return ""
        return str(torch_module.cuda.get_device_name(torch_module.cuda.current_device()))
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return ""


def _gpu_hardware_identity(
    torch_module: Any,
    *,
    device_name: str,
    total_mb: float,
) -> dict[str, Any]:
    """Return a privacy-preserving identity for the physical CUDA device.

    CUDA ordinals are deliberately excluded: ``cuda:0`` changes when device
    visibility or enumeration order changes. A CUDA UUID identifies the card
    itself, while PCI coordinates are the best identity older torch builds can
    expose. Only a truncated hash is persisted, never the raw UUID or PCI
    address.
    """
    properties = None
    try:
        if torch_module.cuda.is_available():
            device_index = torch_module.cuda.current_device()
            properties = torch_module.cuda.get_device_properties(device_index)
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        properties = None

    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)
    compute_capability = (
        f"{int(major)}.{int(minor)}"
        if major is not None and minor is not None
        else ""
    )

    raw_uuid = str(getattr(properties, "uuid", "") or "").strip().lower()
    if raw_uuid and raw_uuid not in {"none", "unknown"}:
        fingerprint_source = "cuda_uuid"
        fingerprint_material = f"nvidia:uuid:{raw_uuid}"
    else:
        pci_parts = (
            getattr(properties, "pci_domain_id", None),
            getattr(properties, "pci_bus_id", None),
            getattr(properties, "pci_device_id", None),
        )
        if all(part is not None for part in pci_parts):
            fingerprint_source = "pci_location"
            fingerprint_material = "nvidia:pci:" + ":".join(
                str(int(part)) for part in pci_parts
            )
        else:
            fingerprint_source = "model_vram_compute"
            fingerprint_material = (
                "nvidia:fallback:"
                f"{str(device_name).strip().lower()}:"
                f"{round(float(total_mb))}:{compute_capability}"
            )

    digest = hashlib.sha256(fingerprint_material.encode("utf-8")).hexdigest()[:24]
    return {
        "vendor": "nvidia",
        "fingerprint": f"sha256:{digest}",
        "fingerprint_source": fingerprint_source,
        "name": str(device_name or "").strip(),
        "physical_vram_mb": round(float(total_mb)),
        "compute_capability": compute_capability,
    }


def _enforce_min_physical_vram(
    *,
    total_mb: float,
    env: dict[str, str],
) -> dict[str, Any]:
    from asr.backends.qwen import qwen_asr_min_physical_vram_mb, qwen_asr_repo_id

    backend = str(env.get("ASR_BACKEND") or os.getenv("ASR_BACKEND", "")).strip()
    repo_id = qwen_asr_repo_id(backend or None)
    minimum_mb = float(qwen_asr_min_physical_vram_mb(repo_id))
    if total_mb < minimum_mb:
        raise RuntimeError(
            "Physical dedicated VRAM below supported minimum: "
            f"repo={repo_id} detected_mb={total_mb:.0f} required_mb={minimum_mb:.0f}. "
            "Shared VRAM and an explicit worker budget do not count; CPU fallback is disabled."
        )
    return {"repo_id": repo_id, "minimum_physical_vram_mb": round(minimum_mb, 1)}


def _live_vram_snapshot_mb(torch_module: Any) -> tuple[float | None, float]:
    """Return ``(free_mb, own_reserved_mb)`` when the CUDA module exposes them."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return None, 0.0
    try:
        if not cuda.is_available():
            return None, 0.0
        mem_info = getattr(cuda, "mem_get_info", None)
        if not callable(mem_info):
            return None, 0.0
        device_index = cuda.current_device()
        free_bytes, _total_bytes = mem_info(device_index)
        reserved_fn = getattr(cuda, "memory_reserved", None)
        own_reserved_bytes = reserved_fn(device_index) if callable(reserved_fn) else 0
        scale = 1024 * 1024
        return float(free_bytes) / scale, float(own_reserved_bytes) / scale
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return None, 0.0


def _cuda_device_indexes(cuda: Any) -> tuple[int, ...] | range:
    device_count_fn = getattr(cuda, "device_count", None)
    if callable(device_count_fn):
        return range(max(1, int(device_count_fn())))
    current_device_fn = getattr(cuda, "current_device", None)
    current = int(current_device_fn()) if callable(current_device_fn) else 0
    return (current,)


def _apply_worker_vram_allocator_cap(
    torch_module: Any,
    *,
    total_mb: float,
    budget_mb: float,
) -> float | None:
    """Keep the CUDA worker below the physical-VRAM soft budget.

    Windows WDDM can satisfy CUDA allocations from shared system memory after
    dedicated VRAM is exhausted. The stage snapshot budget catches that after a
    stage, but the allocator fraction makes the offending allocation fail at
    the boundary instead of running slowly in shared memory. When a subsequent
    job disables the budget (``budget_mb <= 0``) after a cap ``< 1.0`` was
    previously applied, explicitly restore ``1.0``.
    """
    global _LAST_ALLOCATOR_FRACTION
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return None
    try:
        if not cuda.is_available():
            return None
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return None
    setter = getattr(cuda, "set_per_process_memory_fraction", None)
    if not callable(setter):
        return None

    if total_mb <= 0.0 or budget_mb <= 0.0:
        if _LAST_ALLOCATOR_FRACTION is not None and _LAST_ALLOCATOR_FRACTION < 1.0:
            try:
                for device_index in _cuda_device_indexes(cuda):
                    setter(1.0, device_index)
                _LAST_ALLOCATOR_FRACTION = 1.0
            except Exception as exc:
                if is_fatal_cuda_error(exc):
                    raise
        return None

    fraction = min(1.0, max(0.1, float(budget_mb) / float(total_mb)))
    try:
        for device_index in _cuda_device_indexes(cuda):
            setter(fraction, device_index)
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return None
    _LAST_ALLOCATOR_FRACTION = fraction
    return fraction


def _resolve_batch_env_setting(
    env: dict[str, str],
    *,
    key: str,
    auto_batch: int,
    auto_source: str,
) -> tuple[int, str]:
    raw = str(env.get(key) or os.getenv(key, "auto")).strip().lower()
    if raw in {"", "auto"}:
        effective = max(1, int(auto_batch))
        source = auto_source
    else:
        try:
            effective = max(1, int(float(raw)))
            source = "explicit"
        except (TypeError, ValueError):
            effective = max(1, int(auto_batch))
            source = auto_source
    os.environ[key] = str(effective)
    return effective, source


def _adaptive_runtime_tuning(
    torch_module: Any,
    env: dict[str, str],
    *,
    enforce_minimum: bool = True,
) -> dict[str, Any]:
    """Resolve auto VRAM budget and per-stage ASR batch sizes inside the CUDA owner process."""
    if enforce_minimum:
        cuda = getattr(torch_module, "cuda", None)
        cuda_available = False
        if cuda is not None:
            try:
                cuda_available = bool(cuda.is_available())
            except Exception as exc:
                if is_fatal_cuda_error(exc):
                    raise
                cuda_available = False
        if not cuda_available:
            raise RuntimeError(
                "CUDA 不可用，无法在 GPU 上运行 ASR（已禁用 CPU 回退）。"
                "请检查 NVIDIA 驱动、显卡状态与 CUDA 环境。"
            )

    total_mb = _physical_vram_mb(torch_module)
    device_name = _gpu_device_name(torch_module)
    hardware = _gpu_hardware_identity(
        torch_module,
        device_name=device_name,
        total_mb=total_mb,
    )
    minimum_contract = (
        _enforce_min_physical_vram(total_mb=total_mb, env=env)
        if enforce_minimum
        else {}
    )
    ratio = min(1.0, max(0.1, _env_float("ASR_STAGE_WORKER_VRAM_RATIO", 0.95)))
    raw_budget = str(
        env.get("ASR_STAGE_WORKER_VRAM_BUDGET_MB")
        or os.getenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "auto")
    ).strip().lower()
    if raw_budget in {"", "auto", "physical"}:
        budget_mb = total_mb * ratio if total_mb > 0.0 else 0.0
        os.environ["ASR_STAGE_WORKER_VRAM_BUDGET_MB"] = (
            f"{budget_mb:.1f}" if budget_mb > 0.0 else "0"
        )
        budget_source = "physical_vram_ratio"
    else:
        try:
            budget_mb = max(0.0, float(raw_budget))
        except (TypeError, ValueError):
            budget_mb = 0.0
        budget_source = "explicit"
    allocator_fraction = _apply_worker_vram_allocator_cap(
        torch_module,
        total_mb=total_mb,
        budget_mb=budget_mb,
    )

    free_mb, own_reserved_mb = _live_vram_snapshot_mb(torch_module)

    def _read_float_opt(name: str, default: float) -> float:
        raw = str(env.get(name) or os.getenv(name, "")).strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    raw_max_new = str(
        env.get("ASR_MAX_NEW_TOKENS") or os.getenv("ASR_MAX_NEW_TOKENS", "")
    ).strip()
    try:
        parsed_max_new = int(float(raw_max_new)) if raw_max_new else None
    except ValueError:
        parsed_max_new = None

    plan = asr_vram_sizing.select_asr_batch_plan(
        total_vram_mb=total_mb,
        budget_mb=budget_mb,
        free_vram_mb=free_mb,
        own_reserved_mb=own_reserved_mb,
        explicit_budget=(budget_source == "explicit"),
        dtype=str(env.get("ASR_DTYPE") or os.getenv("ASR_DTYPE", "auto")),
        chunk_max_s=_read_float_opt("ASR_CHUNK_MAX_S", 30.0),
        attention=str(env.get("ASR_ATTENTION") or os.getenv("ASR_ATTENTION", "auto")),
        max_new_tokens=parsed_max_new,
        tokens_per_second=_read_float_opt("ASR_DECODE_TOKENS_PER_SECOND", 10.0),
    )

    effective_batch, batch_source = _resolve_batch_env_setting(
        env,
        key="ASR_BATCH_SIZE",
        auto_batch=plan.batch_size,
        auto_source=plan.source,
    )
    effective_feature_batch, feature_batch_source = _resolve_batch_env_setting(
        env,
        key="ASR_FEATURE_BATCH_SIZE",
        auto_batch=plan.feature_batch_size,
        auto_source=plan.source,
    )
    effective_align_batch, align_batch_source = _resolve_batch_env_setting(
        env,
        key="ASR_ALIGN_BATCH_SIZE",
        auto_batch=plan.align_batch_size,
        auto_source=plan.source,
    )

    return {
        "device_name": device_name,
        "hardware": hardware,
        "physical_vram_mb": round(total_mb, 1),
        "minimum_physical_vram_mb": minimum_contract.get("minimum_physical_vram_mb"),
        "vram_ratio": round(ratio, 4),
        "vram_budget_mb": round(budget_mb, 1),
        "vram_budget_source": budget_source,
        "vram_allocator_fraction": (
            round(float(allocator_fraction), 4)
            if allocator_fraction is not None
            else None
        ),
        "asr_batch_size": effective_batch,
        "asr_batch_source": batch_source,
        "asr_feature_batch_size": effective_feature_batch,
        "asr_feature_batch_source": feature_batch_source,
        "asr_align_batch_size": effective_align_batch,
        "asr_align_batch_source": align_batch_source,
        "asr_vram_plan": {
            "batch_size": plan.batch_size,
            "feature_batch_size": plan.feature_batch_size,
            "align_batch_size": plan.align_batch_size,
            "estimated_allocated_mb": plan.estimated_allocated_mb,
            "estimated_reserved_mb": plan.estimated_reserved_mb,
            "weights_mb": plan.weights_mb,
            "per_sample_mb": plan.per_sample_mb,
            "usable_vram_mb": plan.usable_vram_mb,
            "free_vram_mb": plan.free_vram_mb,
            "own_reserved_mb": plan.own_reserved_mb,
        },
    }


def _downshift_stage_batch_env(
    env: dict[str, str],
    *,
    key: str,
    current_override: int | None = None,
) -> tuple[dict[str, str], int, int] | None:
    current = current_override or _effective_stage_batch_size(env, key)
    if current is None or current <= 1:
        return None
    lowered = max(1, current // 2)
    if lowered >= current:
        return None
    updated = dict(env)
    updated[key] = str(lowered)
    return updated, current, lowered


def _downshift_asr_batch_env(
    env: dict[str, str],
    *,
    current_override: int | None = None,
) -> tuple[dict[str, str], int, int] | None:
    return _downshift_stage_batch_env(
        env,
        key="ASR_BATCH_SIZE",
        current_override=current_override,
    )


def _oom_stage_from_message(message: str) -> str:
    text = str(message or "")
    if (
        "ASR 文本转写" in text
        or "asr_text_transcribe" in text
        or "GPU model manager asr" in text
        or "加载本地 ASR 模型" in text
    ):
        return "asr_text_transcribe"
    if "字幕时间轴" in text or "subtitle_timing" in text:
        return "subtitle_timing"
    if "切分" in text or "音频切块" in text:
        return "audio_chunking"
    return ""


def _oom_setting_for_error(stage: str, detail: str = "") -> tuple[str, str]:
    normalized_stage = str(stage or "")
    normalized_detail = str(detail or "").lower()
    if normalized_stage == "audio_chunking" or "stage=split_done" in normalized_detail:
        return _STAGE_BATCH_SETTINGS["audio_chunking"]
    if normalized_stage == "subtitle_timing":
        return _STAGE_BATCH_SETTINGS["subtitle_timing"]
    return _STAGE_BATCH_SETTINGS["asr_text_transcribe"]


def _oom_downshift(
    env: dict[str, str],
    exc: Any,
) -> tuple[dict[str, str], str, float, float] | None:
    stage = str(getattr(exc, "stage", "") or "")
    detail = str(getattr(exc, "detail", "") or "")
    tuning = getattr(exc, "runtime_tuning", None) or {}
    env_key, tuning_key = _oom_setting_for_error(stage, detail)
    override = int(tuning.get(tuning_key) or 0) or None

    lowered = _downshift_stage_batch_env(
        env,
        key=env_key,
        current_override=override,
    )
    if lowered is None:
        return None
    updated, previous, next_value = lowered
    return updated, env_key, float(previous), float(next_value)


def _terminal_oom_detail(
    *,
    env: dict[str, str],
    detail: str,
    batch_size: int | None,
    retry_records: list[dict[str, Any]],
    retry_limit_exhausted: bool = False,
    failed_setting: str = "ASR_BATCH_SIZE",
) -> str:
    budget_mb = _vram_budget_for_env(env)
    retry_summary = ""
    if retry_records:
        steps = [
            "{setting} {previous}->{next}".format(
                setting=record.get("setting") or "ASR_BATCH_SIZE",
                previous=record.get("previous_value", record.get("previous_batch_size")),
                next=record.get("next_value", record.get("next_batch_size")),
            )
            for record in retry_records
        ]
        retry_summary = f" 已尝试降 batch：{', '.join(steps)}。"
    budget_text = f" 当前显存预算为 {budget_mb:.0f}MB。" if budget_mb > 0 else ""

    stage_label = {
        "ASR_FEATURE_BATCH_SIZE": "音频切分编码（ASR_FEATURE_BATCH_SIZE）",
        "ASR_ALIGN_BATCH_SIZE": "字幕时间轴对齐（ASR_ALIGN_BATCH_SIZE）",
        "ASR_BATCH_SIZE": "ASR_BATCH_SIZE",
    }.get(failed_setting, failed_setting)

    if failed_setting == "ASR_CHUNK_ENCODER":
        headline = (
            "GPU 显存不足：OOM 出现在切分用的 encoder 前向；"
            "该阶段没有可继续降低的 batch，任务已停止。"
        )
    elif batch_size is not None and batch_size <= 1:
        headline = f"GPU 显存不足：{stage_label} 已降到 1 后仍然 OOM，任务已停止。"
    elif retry_limit_exhausted:
        headline = f"GPU 显存不足：{stage_label} OOM 自动降 batch 重试次数已用尽，任务已停止。"
    else:
        headline = f"GPU 显存不足：无法继续降低 {stage_label}，任务已停止。"

    # The 0.6B tier was dropped on 2026-07-31, so there is no smaller model to
    # suggest. Pointing at one that no longer ships would be worse than saying
    # plainly that this hardware cannot run the job.
    action = (
        "只有 1.7B 一档，没有更小的模型可切换；请关闭其他占用 GPU 的程序后重试。"
        "在当前硬件/可用显存下仍然不行时任务只能停止，"
        "本程序不会改用 CPU 或缩短时序窗口继续推理。"
    )
    return f"{headline}{retry_summary}{budget_text}{action} 原始错误：{detail}"


def _vram_budget_for_env(env: dict[str, str]) -> float:
    raw = str(
        env.get("ASR_STAGE_WORKER_VRAM_BUDGET_MB")
        or os.getenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "0")
    ).strip().lower()
    if raw in {"", "0", "false", "no", "off", "none"}:
        return 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _cuda_peak_mb_from_details(
    asr_details: dict[str, Any],
    *,
    metric: str,
) -> float | None:
    """Peak VRAM (MB) across worker-reported snapshots for one metric.

    metric="reserved"  -> max of max_reserved_mb / reserved_mb (allocator pool).
    metric="allocated" -> max of max_allocated_mb / allocated_mb (working set).
    """
    keys = {
        "reserved": ("max_reserved_mb", "reserved_mb"),
        "allocated": ("max_allocated_mb", "allocated_mb"),
    }[metric]
    values: list[float] = []
    snapshots = asr_details.get("cuda_memory")
    if isinstance(snapshots, list):
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            for key in keys:
                try:
                    values.append(float(snapshot.get(key)))
                except (TypeError, ValueError):
                    continue
    return round(max(values), 1) if values else None


def _budget_exceeded_warning(
    asr_details: dict[str, Any],
    env: dict[str, str],
) -> str:
    """Non-fatal note when the allocator's *reserved* pool exceeded the budget.

    A job that completed successfully is never retroactively turned into an OOM:
    the reserved pool routinely fills dedicated VRAM on a 6GB card without any
    spill to shared memory, and reserved does not respond to batch downshift.
    Real enforcement is allocated-based and happens mid-pipeline
    (asr/pipeline.py). This only surfaces the peak so it can be logged.
    """
    budget_mb = _vram_budget_for_env(env)
    if budget_mb <= 0.0:
        stage_worker = asr_details.get("stage_worker")
        if isinstance(stage_worker, dict):
            tuning = stage_worker.get("runtime_tuning")
            if isinstance(tuning, dict):
                try:
                    budget_mb = float(tuning.get("vram_budget_mb") or 0.0)
                except (TypeError, ValueError):
                    budget_mb = 0.0
    if budget_mb <= 0.0:
        return ""
    peak_reserved = _cuda_peak_mb_from_details(asr_details, metric="reserved")
    if peak_reserved is None or peak_reserved <= budget_mb:
        return ""
    peak_allocated = _cuda_peak_mb_from_details(asr_details, metric="allocated")
    alloc_text = f"{peak_allocated:.0f}MB" if peak_allocated is not None else "n/a"
    return (
        f"GPU reserved VRAM peaked at {peak_reserved:.0f}MB "
        f"(budget {budget_mb:.0f}MB, allocated peak {alloc_text}); "
        "job completed successfully, no retry triggered."
    )
