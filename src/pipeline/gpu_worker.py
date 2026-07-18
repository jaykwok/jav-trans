from __future__ import annotations

import atexit
import gc
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable

from pipeline import batch_profile
from utils.ffmpeg_runtime import configure_ffmpeg_shared_runtime


class GpuWorkerError(RuntimeError):
    def __init__(
        self,
        kind: str,
        detail: str,
        *,
        stage: str = "",
        runtime_tuning: dict[str, Any] | None = None,
    ):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.stage = str(stage or "")
        self.runtime_tuning = dict(runtime_tuning or {})


class GpuWorkerTimeoutError(GpuWorkerError):
    def __init__(self, detail: str):
        super().__init__("timeout", detail)


LOW_VRAM_ASR_BACKEND = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
_PROFILE_MARKER_SUFFIX = "__PROFILE_ACTIVE"
_PROFILE_STAGE_BY_SETTING = {
    "ASR_BATCH_SIZE": "asr_text_transcribe",
    "ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES": "semantic_split_model",
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _safe_send(conn: Connection, message: dict[str, Any]) -> bool:
    try:
        conn.send(message)
        return True
    except Exception:
        return False


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


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _env_payload(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if str(key).strip() and value is not None
    }


def _resolve_device(device: str) -> str:
    requested = str(device or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _clear_worker_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_asr_pipeline_for_request():
    # No per-request reload. The settings that vary per job (ASR_BACKEND,
    # ASR_BATCH_SIZE, dtype/attention, boundary/pre-asr-cueqc flags) are all read
    # from env at CALL time -- current_asr_backend(), _resolve_asr_batch_size(),
    # _detect_dtype(), _boundary_config(), pre_asr_cueqc.enabled(), ... -- so a
    # persistent worker picks up each job's _temporary_env without re-importing.
    # The only env frozen at CUDA init (PYTORCH_CUDA_ALLOC_CONF,
    # CUDA_VISIBLE_DEVICES) is handled by the client's restart-on-change.
    from asr import pipeline as asr_pipeline

    return asr_pipeline


def _ensure_torchcodec_runtime(shared_directories: tuple[str, ...]) -> None:
    try:
        import torchcodec  # noqa: F401
    except Exception as exc:
        directories = ", ".join(shared_directories) or "未找到 FFmpeg Shared 目录"
        raise RuntimeError(
            "TorchCodec 无法加载 FFmpeg 共享运行库，ASR 任务已在边界分析前停止。"
            "Windows 源码环境请卸载 Gyan.FFmpeg 静态版并安装 "
            f"Gyan.FFmpeg.Shared。已注册目录：{directories}。原始错误：{exc}"
        ) from exc


class GpuModelManager:
    """Owns GPU lifecycle boundaries inside the unified worker process."""

    def __init__(
        self,
        *,
        pid: int,
        on_stage: Callable[[str], None] | None = None,
        runtime_tuning: dict[str, Any] | None = None,
    ) -> None:
        self.pid = int(pid)
        self.on_stage = on_stage
        self.runtime_tuning = dict(runtime_tuning or {})
        self.events: list[dict[str, Any]] = []

    def _record(self, *, stage: str, action: str, **extra: Any) -> None:
        event = {
            "ts": round(time.time(), 3),
            "stage": str(stage),
            "action": str(action),
        }
        event.update(extra)
        self.events.append(event)

    def reset_cuda_state(self, *, reason: str) -> dict[str, Any]:
        gc.collect()
        snapshot: dict[str, Any] = {"reason": reason, "cuda_available": False}
        try:
            import torch

            snapshot["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                device_index = torch.cuda.current_device()
                scale = 1024 * 1024
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                snapshot.update(
                    {
                        "device_index": int(device_index),
                        "allocated_mb": round(
                            torch.cuda.memory_allocated(device_index) / scale,
                            1,
                        ),
                        "reserved_mb": round(
                            torch.cuda.memory_reserved(device_index) / scale,
                            1,
                        ),
                        "free_mb": round(free_bytes / scale, 1),
                        "total_mb": round(total_bytes / scale, 1),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            snapshot["error"] = f"{type(exc).__name__}: {exc}"
        self._record(stage="cuda", action="reset", **snapshot)
        return snapshot

    def lifecycle_event(self, *, stage: str, action: str) -> None:
        stage = str(stage)
        action = str(action)
        if stage == "asr" and action == "load_exclusive":
            self.reset_cuda_state(reason="before_asr_exclusive_load")
        self._record(stage=stage, action=action)
        if action == "unload":
            self.reset_cuda_state(reason=f"after_{stage}_unload")

    def run_transcribe_and_align(
        self,
        *,
        audio_path: str,
        requested_device: str,
        mock: bool,
    ) -> tuple[list[dict], list[str], dict]:
        self.reset_cuda_state(reason="before_request")
        device = _resolve_device(requested_device)
        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        if mock:
            if self.on_stage is not None:
                self.on_stage("ASR stage worker mock")
            segments, asr_log, asr_details = _mock_result(
                audio_path,
                device,
                pid=self.pid,
            )
        else:
            asr_pipeline = _load_asr_pipeline_for_request()
            segments, asr_log, asr_details = asr_pipeline.transcribe_and_align(
                audio_path,
                device,
                on_stage=self.on_stage,
                include_details=True,
                model_manager=self,
            )
            asr_details = dict(asr_details or {})

        self.reset_cuda_state(reason="after_request")
        asr_details.setdefault("device", device)
        asr_details["stage_worker"] = {
            "mode": "gpu_worker",
            "process_model": "persistent_subprocess",
            "pid": self.pid,
            "mock": bool(mock),
            "gpu_owner": True,
            "runtime_tuning": dict(self.runtime_tuning),
            "model_manager": {
                "policy": "sequential_exclusive",
                "events": list(self.events),
            },
        }
        return segments, asr_log, asr_details


def _mock_result(audio_path: str, device: str, *, pid: int) -> tuple[list[dict], list[str], dict]:
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "mock",
            "words": [{"start": 0.0, "end": 1.0, "word": "mock"}],
        }
    ]
    log = ["ASR stage worker mock result"]
    details = {
        "backend": "mock",
        "audio_path": audio_path,
        "device": device,
        "chunk_count": 1,
        "transcript_chunks": [{"text": "mock"}],
        "stage_timings": {},
        "cuda_memory": [],
        "word_count": 1,
        "segment_count": 1,
        "stage_worker": {
            "mode": "gpu_worker",
            "process_model": "persistent_subprocess",
            "pid": pid,
            "mock": True,
        },
    }
    return segments, log, details


def _parse_batch_mapping(raw: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        repo_id, value = item.rsplit("=", 1)
        try:
            mapping[repo_id.strip()] = max(1, int(float(value.strip())))
        except (TypeError, ValueError):
            continue
    return mapping


def _effective_asr_batch_size(env: dict[str, str]) -> int | None:
    raw = str(env.get("ASR_BATCH_SIZE") or os.getenv("ASR_BATCH_SIZE", "auto")).strip()
    if raw and raw.lower() != "auto":
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return None
    try:
        from asr.backends.qwen import (
            DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO,
            qwen_asr_repo_id,
        )

        backend = str(env.get("ASR_BACKEND") or os.getenv("ASR_BACKEND", "")).strip()
        repo_id = qwen_asr_repo_id(backend or None)
        mapping = dict(DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO)
        mapping.update(
            _parse_batch_mapping(
                str(
                    env.get("ASR_BATCH_SIZE_BY_REPO")
                    or os.getenv("ASR_BATCH_SIZE_BY_REPO", "")
                )
            )
        )
        return max(1, int(mapping[repo_id]))
    except Exception:
        return None


def _physical_vram_mb(torch_module: Any) -> float:
    try:
        if not torch_module.cuda.is_available():
            return 0.0
        device_index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(device_index)
        return float(properties.total_memory) / (1024 * 1024)
    except Exception:
        return 0.0


def _gpu_device_name(torch_module: Any) -> str:
    try:
        if not torch_module.cuda.is_available():
            return ""
        return str(torch_module.cuda.get_device_name(torch_module.cuda.current_device()))
    except Exception:
        return ""


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


class _CudaStageMemoryTracker:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.current_stage = ""
        self.peaks_mb: dict[str, float] = {}

    def _cuda_ready(self) -> bool:
        try:
            return bool(self.torch is not None and self.torch.cuda.is_available())
        except Exception:
            return False

    def _finish_current(self) -> None:
        if not self.current_stage or not self._cuda_ready():
            return
        try:
            device_index = self.torch.cuda.current_device()
            peak_bytes = max(
                self.torch.cuda.max_memory_allocated(device_index),
                self.torch.cuda.memory_allocated(device_index),
            )
            self.peaks_mb[self.current_stage] = max(
                self.peaks_mb.get(self.current_stage, 0.0),
                float(peak_bytes) / (1024 * 1024),
            )
        except Exception:
            pass

    def observe(self, message: str) -> str:
        stage = _oom_stage_from_message(message)
        if not stage or stage == self.current_stage:
            return self.current_stage
        self._finish_current()
        self.current_stage = stage
        if self._cuda_ready():
            try:
                self.torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        return self.current_stage

    def finish(self) -> dict[str, float]:
        self._finish_current()
        return {
            stage: round(value, 1)
            for stage, value in self.peaks_mb.items()
        }


def _profile_identity(
    *,
    stage: str,
    device_name: str,
    total_mb: float,
    budget_mb: float,
    env: dict[str, str],
) -> dict[str, Any]:
    return {
        "profile_version": batch_profile.PROFILE_VERSION,
        "stage": stage,
        "gpu_name": device_name,
        "physical_vram_mb": round(total_mb),
        "budget_mb": round(budget_mb),
        "backend": str(
            env.get("ASR_BACKEND") or os.getenv("ASR_BACKEND", "")
        ).strip(),
        "model_id": str(
            env.get("ASR_MODEL_ID") or os.getenv("ASR_MODEL_ID", "")
        ).strip(),
        "model_path": str(
            env.get("ASR_MODEL_PATH") or os.getenv("ASR_MODEL_PATH", "")
        ).strip(),
        "dtype": str(env.get("ASR_DTYPE") or os.getenv("ASR_DTYPE", "")).strip(),
        "attention": str(
            env.get("ASR_ATTENTION") or os.getenv("ASR_ATTENTION", "")
        ).strip(),
        "max_new_tokens": str(
            env.get("ASR_MAX_NEW_TOKENS")
            or os.getenv("ASR_MAX_NEW_TOKENS", "")
        ).strip(),
    }


def _auto_batch_setting(
    *,
    setting: str,
    stage: str,
    raw_value: str,
    base_batch: int,
    scale: float,
    identity: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    heuristic = max(1, min(base_batch * 4, int(base_batch * scale)))
    profile_active = raw_value in {"", "auto"} or os.getenv(
        f"{setting}{_PROFILE_MARKER_SUFFIX}",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if raw_value not in {"", "auto"}:
        try:
            explicit = max(1, int(float(raw_value)))
        except (TypeError, ValueError):
            explicit = heuristic
        return explicit, {
            "active": profile_active,
            "stage": stage,
            "identity": identity,
            "batch_size": explicit,
            "heuristic_batch": heuristic,
            "max_batch": base_batch * 4,
            "profile_entry": {},
        }

    effective, entry = batch_profile.recommendation(
        identity,
        heuristic_batch=heuristic,
        max_batch=base_batch * 4,
    )
    return effective, {
        "active": profile_active,
        "stage": stage,
        "identity": identity,
        "batch_size": effective,
        "heuristic_batch": heuristic,
        "max_batch": base_batch * 4,
        "profile_entry": entry,
    }


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
    the boundary instead of running slowly in shared memory.
    """

    if total_mb <= 0.0 or budget_mb <= 0.0:
        return None
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return None
    try:
        if not cuda.is_available():
            return None
    except Exception:
        return None
    setter = getattr(cuda, "set_per_process_memory_fraction", None)
    if not callable(setter):
        return None
    fraction = min(1.0, max(0.1, float(budget_mb) / float(total_mb)))
    try:
        device_count_fn = getattr(cuda, "device_count", None)
        if callable(device_count_fn):
            device_indexes = range(max(1, int(device_count_fn())))
        else:
            current_device_fn = getattr(cuda, "current_device", None)
            current = int(current_device_fn()) if callable(current_device_fn) else 0
            device_indexes = (current,)
        for device_index in device_indexes:
            setter(fraction, device_index)
    except Exception:
        return None
    return fraction


def _adaptive_runtime_tuning(
    torch_module: Any,
    env: dict[str, str],
    *,
    enforce_minimum: bool = True,
) -> dict[str, Any]:
    """Resolve auto VRAM budget and ASR batch inside the CUDA owner process."""
    total_mb = _physical_vram_mb(torch_module)
    device_name = _gpu_device_name(torch_module)
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

    raw_batch = str(
        env.get("ASR_BATCH_SIZE") or os.getenv("ASR_BATCH_SIZE", "auto")
    ).strip().lower()
    base_batch = _effective_asr_batch_size(env)
    effective_batch = base_batch
    batch_source = "explicit"
    scale = max(0.25, budget_mb / 5600.0) if budget_mb > 0.0 else 1.0
    batch_profiles: dict[str, dict[str, Any]] = {}
    if base_batch is not None:
        asr_identity = _profile_identity(
            stage="asr_text_transcribe",
            device_name=device_name,
            total_mb=total_mb,
            budget_mb=budget_mb,
            env=env,
        )
        effective_batch, asr_profile = _auto_batch_setting(
            setting="ASR_BATCH_SIZE",
            stage="asr_text_transcribe",
            raw_value=raw_batch,
            base_batch=base_batch,
            scale=scale,
            identity=asr_identity,
        )
        batch_profiles["ASR_BATCH_SIZE"] = asr_profile
        os.environ["ASR_BATCH_SIZE"] = str(effective_batch)
        if raw_batch in {"", "auto"}:
            batch_source = (
                "learned_profile"
                if asr_profile.get("profile_entry")
                else "auto_scaled_from_vram"
            )

    semantic_raw = str(
        env.get("ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES")
        or os.getenv("ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES", "auto")
    ).strip().lower()
    semantic_identity = _profile_identity(
        stage="semantic_split_model",
        device_name=device_name,
        total_mb=total_mb,
        budget_mb=budget_mb,
        env=env,
    )
    semantic_batch, semantic_profile = _auto_batch_setting(
        setting="ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES",
        stage="semantic_split_model",
        raw_value=semantic_raw,
        base_batch=128,
        scale=scale,
        identity=semantic_identity,
    )
    batch_profiles["ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES"] = semantic_profile
    os.environ["ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES"] = str(semantic_batch)

    return {
        "device_name": device_name,
        "physical_vram_mb": round(total_mb, 1),
        "minimum_physical_vram_mb": minimum_contract.get(
            "minimum_physical_vram_mb"
        ),
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
        "acoustic_split_max_batch_candidates": semantic_batch,
        "batch_profiles": batch_profiles,
    }


def _profile_for_stage(
    runtime_tuning: dict[str, Any],
    stage: str,
) -> tuple[str, dict[str, Any]] | None:
    profiles = runtime_tuning.get("batch_profiles")
    if not isinstance(profiles, dict):
        return None
    for setting, raw_profile in profiles.items():
        if not isinstance(raw_profile, dict):
            continue
        if raw_profile.get("active") and str(raw_profile.get("stage") or "") == stage:
            return str(setting), raw_profile
    return None


def _record_profile_oom(runtime_tuning: dict[str, Any], stage: str) -> None:
    selected = _profile_for_stage(runtime_tuning, stage)
    if selected is None:
        return
    _setting, profile = selected
    try:
        batch_profile.record_oom(
            profile["identity"],
            batch_size=int(profile["batch_size"]),
            max_batch=int(profile["max_batch"]),
        )
    except Exception:
        pass


def _record_profile_successes(
    runtime_tuning: dict[str, Any],
    stage_peaks_mb: dict[str, float],
) -> dict[str, Any]:
    learned: dict[str, Any] = {}
    profiles = runtime_tuning.get("batch_profiles")
    if not isinstance(profiles, dict):
        return learned
    budget_mb = float(runtime_tuning.get("vram_budget_mb") or 0.0)
    for setting, raw_profile in profiles.items():
        if not isinstance(raw_profile, dict) or not raw_profile.get("active"):
            continue
        stage = str(raw_profile.get("stage") or "")
        peak_mb = stage_peaks_mb.get(stage)
        if peak_mb is None:
            continue
        try:
            entry = batch_profile.record_success(
                raw_profile["identity"],
                batch_size=int(raw_profile["batch_size"]),
                peak_allocated_mb=float(peak_mb),
                budget_mb=budget_mb,
                max_batch=int(raw_profile["max_batch"]),
            )
        except Exception:
            continue
        learned[str(setting)] = entry
    return learned


def _downshift_asr_batch_env(
    env: dict[str, str],
    *,
    current_override: int | None = None,
) -> tuple[dict[str, str], int, int] | None:
    current = current_override or _effective_asr_batch_size(env)
    if current is None or current <= 1:
        return None
    lowered = max(1, current // 2)
    if lowered >= current:
        return None
    updated = dict(env)
    updated["ASR_BATCH_SIZE"] = str(lowered)
    return updated, current, lowered


def _oom_stage_from_message(message: str) -> str:
    text = str(message or "")
    if "语音岛检测" in text or "speech_island" in text:
        return "speech_island_scorer"
    if (
        "ASR 文本转写" in text
        or "asr_text_transcribe" in text
        or "GPU model manager asr" in text
        or "加载本地 ASR 模型" in text
    ):
        return "asr_text_transcribe"
    if "字幕时间轴" in text or "subtitle_timing" in text:
        return "subtitle_timing"
    if "Pre-ASR" in text:
        return "pre_asr_cueqc"
    if "语义切分" in text:
        return "semantic_split_model"
    if "外边界" in text:
        return "outer_edge_refiner"
    return ""


def _oom_downshift(
    env: dict[str, str],
    exc: GpuWorkerError,
) -> tuple[dict[str, str], str, float, float] | None:
    stage = str(exc.stage or "")
    detail = str(exc.detail or "").lower()
    tuning = exc.runtime_tuning
    selected_profile = _profile_for_stage(tuning, stage)
    if selected_profile is not None and stage == "semantic_split_model":
        setting, profile = selected_profile
        current = max(1, int(profile.get("batch_size") or 1))
        if current <= 1:
            return None
        lowered = max(1, current // 2)
        updated = dict(env)
        updated[setting] = str(lowered)
        updated[f"{setting}{_PROFILE_MARKER_SUFFIX}"] = "1"
        return updated, setting, float(current), float(lowered)
    if stage in {
        "speech_island_scorer",
        "outer_edge_refiner",
        "pre_asr_cueqc",
    } or "stage=split_done" in detail or "stage=pre_asr_boundary" in detail:
        # Boundary/PTM inputs are temporal. Shrinking the 20-second window would
        # change embeddings and overlap averaging, so it is not a batch knob.
        return None

    lowered = _downshift_asr_batch_env(
        env,
        current_override=int(tuning.get("asr_batch_size") or 0) or None,
    )
    if lowered is None:
        return None
    updated, previous, next_value = lowered
    updated[f"ASR_BATCH_SIZE{_PROFILE_MARKER_SUFFIX}"] = "1"
    return updated, "ASR_BATCH_SIZE", float(previous), float(next_value)


def _terminal_oom_detail(
    *,
    env: dict[str, str],
    detail: str,
    batch_size: int | None,
    retry_records: list[dict[str, Any]],
    retry_limit_exhausted: bool = False,
    failed_setting: str = "ASR_BATCH_SIZE",
) -> str:
    backend = str(env.get("ASR_BACKEND") or os.getenv("ASR_BACKEND", "")).strip()
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
    if failed_setting == "SPEECH_BOUNDARY_JA_WINDOW_S":
        headline = (
            "GPU 显存不足：OOM 出现在 SpeechBoundary/PTM 时序推理；"
            "为避免缩短窗口改变推理结果，任务已停止。"
        )
    elif failed_setting == "ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES":
        headline = (
            "GPU 显存不足：Semantic Split batch 已降到 1 后仍然 OOM，"
            "任务已停止。"
        )
    elif batch_size is not None and batch_size <= 1:
        headline = "GPU 显存不足：ASR_BATCH_SIZE 已降到 1 后仍然 OOM，任务已停止。"
    elif retry_limit_exhausted:
        headline = "GPU 显存不足：OOM 自动降 batch 重试次数已用尽，任务已停止。"
    else:
        headline = "GPU 显存不足：无法继续降低 ASR_BATCH_SIZE，任务已停止。"

    if LOW_VRAM_ASR_BACKEND not in backend:
        action = (
            "建议在前端将 ASR 后端切换为 0.6B 低显存档 "
            f"({LOW_VRAM_ASR_BACKEND}) 后重试；也可以关闭其他占用 GPU 的程序。"
        )
    else:
        action = (
            "当前已经是 0.6B 最低显存档；在当前硬件/可用显存下无法运行，"
            "本程序不会改用 CPU 或缩短时序窗口继续推理。"
        )
    return (
        f"{headline}{retry_summary}{budget_text}{action} "
        f"原始错误：{detail}"
    )


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


def worker_main(parent_conn: Connection) -> None:
    sys.stdout = sys.stderr
    pid = os.getpid()
    ffmpeg_shared_directories = configure_ffmpeg_shared_runtime()
    if not _safe_send(parent_conn, {"op": "ready", "pid": pid}):
        return

    # Idle self-exit (read from the inherited process env, not per-job
    # overrides -- this is a process-lifecycle knob, see config.py). When the
    # worker has had no inbound request for MAX_IDLE_S seconds, exit so the next
    # job starts from a clean CUDA state. The client detects the exit via
    # is_alive() and restarts transparently on the next request. A per-job
    # restart cadence is intentionally NOT provided: every job already does
    # gc.collect() + empty_cache() on completion, so VRAM does not accumulate
    # across jobs and a job counter would only add cold-start cost.
    idle_timeout_s = _env_float("ASR_STAGE_WORKER_MAX_IDLE_S", 300.0)

    while True:
        try:
            if idle_timeout_s > 0.0 and not parent_conn.poll(idle_timeout_s):
                _clear_worker_cuda()
                return
            msg = parent_conn.recv()
        except (EOFError, OSError):
            return

        if not isinstance(msg, dict):
            if not _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": "",
                    "kind": "protocol_error",
                    "detail": "message must be an object",
                },
            ):
                return
            continue

        op = msg.get("op")
        if op == "shutdown":
            _clear_worker_cuda()
            raise SystemExit(0)

        job_id = str(msg.get("job_id") or "")
        if op != "transcribe_and_align":
            _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": job_id,
                    "kind": "protocol_error",
                    "detail": f"unknown op: {op}",
                },
            )
            continue

        audio_path = str(msg.get("audio_path") or "")
        if not audio_path:
            _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": job_id,
                    "kind": "protocol_error",
                    "detail": "missing audio_path",
                },
            )
            continue

        env = _env_payload(msg.get("env"))
        requested_device = str(msg.get("device") or "auto")
        torch_module = None
        runtime_tuning: dict[str, Any] = {}
        active_stage = ""
        stage_memory: _CudaStageMemoryTracker | None = None
        try:
            with _temporary_env(env):
                mock = os.getenv("ASR_STAGE_WORKER_MOCK", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if not mock:
                    _ensure_torchcodec_runtime(ffmpeg_shared_directories)
                try:
                    import torch as imported_torch
                except ImportError:
                    imported_torch = None
                if imported_torch is not None:
                    torch_module = imported_torch
                    runtime_tuning = _adaptive_runtime_tuning(
                        torch_module,
                        env,
                        enforce_minimum=not mock,
                    )
                    stage_memory = _CudaStageMemoryTracker(torch_module)

                def _on_stage(message: str) -> None:
                    nonlocal active_stage
                    stage_hint = (
                        stage_memory.observe(message)
                        if stage_memory is not None
                        else _oom_stage_from_message(message)
                    )
                    if stage_hint:
                        active_stage = stage_hint
                    _safe_send(
                        parent_conn,
                        {
                            "op": "stage",
                            "job_id": job_id,
                            "message": str(message),
                        },
                    )

                _on_stage(
                    "GPU batch tuning "
                    f"asr={runtime_tuning.get('asr_batch_size')} "
                    f"asr_source={runtime_tuning.get('asr_batch_source')} "
                    "acoustic_split_max_batch_candidates="
                    f"{runtime_tuning.get('acoustic_split_max_batch_candidates')} "
                    f"budget_mb={runtime_tuning.get('vram_budget_mb')}"
                )
                model_manager = GpuModelManager(
                    pid=pid,
                    on_stage=_on_stage,
                    runtime_tuning=runtime_tuning,
                )
                segments, asr_log, asr_details = model_manager.run_transcribe_and_align(
                    audio_path=audio_path,
                    requested_device=requested_device,
                    mock=mock,
                )
                stage_peaks = stage_memory.finish() if stage_memory is not None else {}
                learned_profiles = _record_profile_successes(
                    runtime_tuning,
                    stage_peaks,
                )
                stage_worker = asr_details.get("stage_worker")
                if isinstance(stage_worker, dict):
                    stage_worker["stage_peak_allocated_mb"] = stage_peaks
                    stage_worker["learned_batch_profiles"] = learned_profiles
                if learned_profiles:
                    next_values = ", ".join(
                        f"{setting}={entry.get('recommended_batch')}"
                        for setting, entry in learned_profiles.items()
                    )
                    _on_stage(f"GPU batch profile next task: {next_values}")

                _clear_worker_cuda()
                if not _safe_send(
                    parent_conn,
                    {
                        "op": "result",
                        "job_id": job_id,
                        "segments": segments,
                        "asr_log": asr_log,
                        "asr_details": asr_details,
                        "device": str(asr_details.get("device") or "auto"),
                    },
                ):
                    return
        except Exception as exc:
            kind = (
                "ram_oom"
                if _is_ram_oom_error(exc)
                else "oom"
                if _is_oom_error(exc, torch_module)
                else "crash"
            )
            if stage_memory is not None:
                stage_memory.finish()
            if kind == "oom" and active_stage:
                _record_profile_oom(runtime_tuning, active_stage)
            _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": job_id,
                    "kind": kind,
                    "detail": repr(exc),
                    "stage": active_stage,
                    "runtime_tuning": runtime_tuning,
                },
            )
            _clear_worker_cuda()
            raise SystemExit(0)


# Env read by CUDA only at runtime/driver init, so changing it on a *running*
# persistent worker has no effect. The client restarts the worker when one of
# these differs from the values it was started under.
_CUDA_INIT_ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")


class _GpuWorkerClient:
    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._conn = None
        self._job_handle = None
        self.kill_grace_s = _env_float("ASR_STAGE_WORKER_KILL_GRACE_S", 5.0)
        # Snapshot of the CUDA-init-time env the live worker was started under,
        # so we can restart it when a later job needs different alloc/device.
        self._cuda_init_env: dict[str, str] | None = None

    def is_alive(self) -> bool:
        return (
            self._process is not None
            and self._process.is_alive()
            and self._conn is not None
        )

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def _kill_child(self) -> None:
        process = self._process
        self._process = None
        self._job_handle = None
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(self.kill_grace_s)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                if process.is_alive():
                    process.join(1)
            except Exception:
                pass
        self._close_conn()

    def close(self) -> None:
        conn = self._conn
        process = self._process
        try:
            if conn is not None and process is not None and process.is_alive():
                conn.send({"op": "shutdown"})
                process.join(5)
        except Exception:
            pass
        finally:
            if process is not None and process.is_alive():
                self._kill_child()
            else:
                self._process = None
                self._job_handle = None
                self._close_conn()

    def _effective_cuda_init_env(
        self,
        env_overrides: dict[str, str] | None,
    ) -> dict[str, str]:
        env_overrides = env_overrides or {}
        result: dict[str, str] = {}
        for key in _CUDA_INIT_ENV_KEYS:
            if key in env_overrides:
                result[key] = str(env_overrides[key])
            else:
                result[key] = os.environ.get(key, "")
        return result

    def _start_worker(self) -> None:
        self._kill_child()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        # daemon=True off-Windows so a SIGKILL'd/orphaned parent still reaps the
        # GPU child (no Job Object there). Windows keeps daemon=False and relies
        # on the kill-on-close Job Object assigned below.
        process = self._ctx.Process(
            target=worker_main,
            args=(child_conn,),
            daemon=(os.name != "nt"),
        )
        process.start()
        self._job_handle = None
        if os.name == "nt":
            try:
                from asr.local_backend import (
                    _assign_process_to_job_object,
                    _create_kill_on_close_job_object,
                )

                job = _create_kill_on_close_job_object()
                _assign_process_to_job_object(job, process)
                self._job_handle = job
            except Exception:
                self._job_handle = None
        child_conn.close()
        self._process = process
        self._conn = parent_conn

        ready_timeout_s = _env_float("ASR_STAGE_WORKER_READY_TIMEOUT_S", 60.0)
        deadline = time.monotonic() + ready_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_child()
                raise GpuWorkerTimeoutError(
                    f"ASR stage worker ready timeout after {ready_timeout_s:.1f}s"
                )
            if parent_conn.poll(min(0.5, remaining)):
                break
            if process.exitcode is not None:
                exitcode = process.exitcode
                self._kill_child()
                raise GpuWorkerError(
                    "crash",
                    f"ASR stage worker exited before ready: {exitcode}",
                )

        try:
            message = parent_conn.recv()
        except EOFError as exc:
            exitcode = process.exitcode
            self._kill_child()
            raise GpuWorkerError(
                "crash",
                f"ASR stage worker pipe closed before ready exitcode={exitcode}",
            ) from exc

        if not isinstance(message, dict):
            self._kill_child()
            raise GpuWorkerError("protocol_error", "ready message is not a dict")
        if message.get("op") != "ready":
            kind = str(message.get("kind") or "protocol_error")
            detail = str(message.get("detail") or f"unexpected ready message: {message!r}")
            self._kill_child()
            raise GpuWorkerError(kind, detail)

    def _ensure_worker(self) -> None:
        if self.is_alive():
            return
        self._start_worker()

    def _send_request(self, payload: dict[str, Any], env_overrides: dict[str, str] | None) -> None:
        """Send a request, transparently restarting once if the pipe is dead.

        A persistent worker can self-exit between jobs (idle timeout / max-jobs
        cadence) or be killed by an OOM retry; rather than surfacing that as a
        crash, restart and resend once before giving up. Also records the
        CUDA-init-time env the (re)started worker is running under.
        """
        for attempt in range(2):
            self._ensure_worker()
            conn = self._conn
            if conn is None:
                continue
            try:
                conn.send(payload)
                self._cuda_init_env = self._effective_cuda_init_env(env_overrides)
                return
            except (BrokenPipeError, EOFError, OSError) as exc:
                exitcode = self._process.exitcode if self._process is not None else None
                self._kill_child()
                if attempt == 1:
                    raise GpuWorkerError(
                        "crash",
                        f"ASR stage worker send failed exitcode={exitcode}: {exc!r}",
                    ) from exc
                # Loop: _ensure_worker starts a fresh worker and we resend.

    def _transcribe_and_align_once(
        self,
        audio_path: str,
        *,
        device: str = "auto",
        env_overrides: dict[str, str] | None = None,
        job_id: str = "",
        on_stage: Callable[[str], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> tuple[list[dict], list[str], dict]:
        # If CUDA-init-time env changed since the worker started (e.g. a new
        # PYTORCH_CUDA_ALLOC_CONF), restart it: those keys are only read at CUDA
        # runtime init and would otherwise be frozen for the worker's lifetime.
        if self.is_alive() and self._cuda_init_env is not None:
            if self._effective_cuda_init_env(env_overrides) != self._cuda_init_env:
                self._kill_child()

        request_id = str(job_id or uuid.uuid4().hex[:8])
        payload = {
            "op": "transcribe_and_align",
            "job_id": request_id,
            "audio_path": str(Path(audio_path).resolve()),
            "device": str(device or "auto"),
            "env": dict(env_overrides or {}),
        }
        self._send_request(payload, env_overrides)
        assert self._conn is not None

        timeout_s = _env_float("ASR_STAGE_WORKER_TIMEOUT_S", 0.0)
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        heartbeat_s = max(0.0, _env_float("ASR_STAGE_WORKER_HEARTBEAT_S", 10.0))
        request_started = time.monotonic()
        last_stage_at = request_started
        last_heartbeat_at = request_started
        last_stage_message = "GPU worker request accepted"

        while True:
            if cancel_requested is not None and cancel_requested():
                self._kill_child()
                raise GpuWorkerError("cancelled", "ASR stage worker cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                self._kill_child()
                raise GpuWorkerTimeoutError(
                    f"ASR stage worker timeout after {timeout_s:.1f}s"
                )

            wait_s = 0.25
            if deadline is not None:
                wait_s = max(0.05, min(wait_s, deadline - time.monotonic()))
            if not self._conn.poll(wait_s):
                exitcode = self._process.exitcode if self._process is not None else None
                if exitcode is not None:
                    self._kill_child()
                    raise GpuWorkerError(
                        "crash",
                        f"ASR stage worker exited before result exitcode={exitcode}",
                    )
                now = time.monotonic()
                if (
                    heartbeat_s > 0.0
                    and on_stage is not None
                    and now - last_heartbeat_at >= heartbeat_s
                ):
                    on_stage(
                        "阶段心跳 "
                        f"current={last_stage_message} "
                        f"elapsed={now - request_started:.1f}s "
                        f"idle={now - last_stage_at:.1f}s"
                    )
                    last_heartbeat_at = now
                continue

            try:
                message = self._conn.recv()
            except EOFError as exc:
                exitcode = self._process.exitcode if self._process is not None else None
                self._kill_child()
                raise GpuWorkerError(
                    "crash",
                    f"ASR stage worker pipe closed exitcode={exitcode}",
                ) from exc

            if not isinstance(message, dict):
                self._kill_child()
                raise GpuWorkerError("protocol_error", "worker message is not a dict")
            if str(message.get("job_id") or "") != request_id:
                continue

            op = message.get("op")
            if op == "stage":
                last_stage_message = str(message.get("message") or "")
                last_stage_at = time.monotonic()
                last_heartbeat_at = last_stage_at
                if on_stage is not None:
                    try:
                        on_stage(last_stage_message)
                    except BaseException:
                        self._kill_child()
                        raise
                continue

            if op == "result":
                segments = message.get("segments")
                asr_log = message.get("asr_log")
                asr_details = message.get("asr_details")
                if not isinstance(segments, list):
                    self._kill_child()
                    raise GpuWorkerError("protocol_error", "segments must be a list")
                if not isinstance(asr_log, list):
                    self._kill_child()
                    raise GpuWorkerError("protocol_error", "asr_log must be a list")
                if not isinstance(asr_details, dict):
                    self._kill_child()
                    raise GpuWorkerError("protocol_error", "asr_details must be a dict")
                return segments, [str(item) for item in asr_log], dict(asr_details)

            if op == "error":
                kind = str(message.get("kind") or "crash")
                detail = str(message.get("detail") or "ASR stage worker error")
                stage = str(message.get("stage") or "")
                runtime_tuning = message.get("runtime_tuning")
                self._kill_child()
                raise GpuWorkerError(
                    kind,
                    detail,
                    stage=stage,
                    runtime_tuning=(
                        runtime_tuning if isinstance(runtime_tuning, dict) else {}
                    ),
                )

            self._kill_child()
            raise GpuWorkerError("protocol_error", f"unexpected worker op: {op}")

    def transcribe_and_align(
        self,
        audio_path: str,
        *,
        device: str = "auto",
        env_overrides: dict[str, str] | None = None,
        job_id: str = "",
        on_stage: Callable[[str], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> tuple[list[dict], list[str], dict]:
        retry_limit = max(0, _env_int("ASR_STAGE_WORKER_OOM_RETRY_LIMIT", 6))
        attempts_by_setting: dict[str, int] = {}
        retry_records: list[dict[str, Any]] = []
        current_env = dict(env_overrides or {})
        while True:
            try:
                segments, asr_log, asr_details = self._transcribe_and_align_once(
                    audio_path,
                    device=device,
                    env_overrides=current_env,
                    job_id=job_id,
                    on_stage=on_stage,
                    cancel_requested=cancel_requested,
                )
                # A successful result is final -- never discard it for a VRAM
                # budget. Surface a non-fatal note if the reserved pool spiked,
                # but keep the segments. Hard OOM is caught below / mid-pipeline.
                warning = _budget_exceeded_warning(asr_details, current_env)
                if warning and on_stage is not None:
                    try:
                        on_stage(warning)
                    except BaseException:
                        pass
                if retry_records:
                    stage_worker = asr_details.setdefault("stage_worker", {})
                    if isinstance(stage_worker, dict):
                        stage_worker["oom_retries"] = list(retry_records)
                return segments, asr_log, asr_details
            except GpuWorkerError as exc:
                if str(getattr(exc, "kind", "")) != "oom":
                    raise
                lowered = _oom_downshift(current_env, exc)
                if lowered is None:
                    batch_size = int(exc.runtime_tuning.get("asr_batch_size") or 0) or (
                        _effective_asr_batch_size(current_env)
                    )
                    failed_setting = (
                        "ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES"
                        if exc.stage == "semantic_split_model"
                        else "SPEECH_BOUNDARY_JA_WINDOW_S"
                        if exc.stage in {
                            "speech_island_scorer",
                            "outer_edge_refiner",
                            "pre_asr_cueqc",
                        }
                        or "stage=split_done" in str(exc.detail or "").lower()
                        else "ASR_BATCH_SIZE"
                    )
                    raise GpuWorkerError(
                        "oom",
                        _terminal_oom_detail(
                            env=current_env,
                            detail=str(getattr(exc, "detail", exc)),
                            batch_size=batch_size,
                            retry_records=retry_records,
                            failed_setting=failed_setting,
                        ),
                    ) from exc
                next_env, setting, previous_value, next_value = lowered
                setting_attempt = attempts_by_setting.get(setting, 0)
                if setting_attempt >= retry_limit:
                    batch_size = int(exc.runtime_tuning.get("asr_batch_size") or 0) or (
                        _effective_asr_batch_size(current_env)
                    )
                    raise GpuWorkerError(
                        "oom",
                        _terminal_oom_detail(
                            env=current_env,
                            detail=str(getattr(exc, "detail", exc)),
                            batch_size=batch_size,
                            retry_records=retry_records,
                            retry_limit_exhausted=True,
                            failed_setting=setting,
                        ),
                    ) from exc
                setting_attempt += 1
                attempts_by_setting[setting] = setting_attempt
                record = {
                    "attempt": len(retry_records) + 1,
                    "stage": str(exc.stage or ""),
                    "setting": setting,
                    "previous_value": previous_value,
                    "next_value": next_value,
                    "detail": str(getattr(exc, "detail", exc)),
                }
                if setting == "ASR_BATCH_SIZE":
                    record["previous_batch_size"] = int(previous_value)
                    record["next_batch_size"] = int(next_value)
                retry_records.append(record)
                current_env = next_env
                if on_stage is not None:
                    on_stage(
                        "GPU worker OOM; restarting worker and resuming from "
                        f"cache/checkpoint with {setting} "
                        f"{previous_value:g}->{next_value:g}"
                    )
                continue


_GLOBAL_WORKER: _GpuWorkerClient | None = None
_GLOBAL_WORKER_LOCK = threading.RLock()


def _get_global_worker() -> _GpuWorkerClient:
    global _GLOBAL_WORKER
    if _GLOBAL_WORKER is None or not _GLOBAL_WORKER.is_alive():
        _GLOBAL_WORKER = _GpuWorkerClient()
    return _GLOBAL_WORKER


def transcribe_and_align(
    audio_path: str,
    *,
    device: str = "auto",
    env_overrides: dict[str, str] | None = None,
    job_id: str = "",
    on_stage: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[list[dict], list[str], dict]:
    global _GLOBAL_WORKER
    with _GLOBAL_WORKER_LOCK:
        worker = _get_global_worker()
        try:
            return worker.transcribe_and_align(
                audio_path,
                device=device,
                env_overrides=env_overrides,
                job_id=job_id,
                on_stage=on_stage,
                cancel_requested=cancel_requested,
            )
        except Exception:
            if not worker.is_alive():
                _GLOBAL_WORKER = None
            raise


def shutdown_global_worker() -> None:
    global _GLOBAL_WORKER
    with _GLOBAL_WORKER_LOCK:
        if _GLOBAL_WORKER is not None:
            _GLOBAL_WORKER.close()
        _GLOBAL_WORKER = None


atexit.register(shutdown_global_worker)
