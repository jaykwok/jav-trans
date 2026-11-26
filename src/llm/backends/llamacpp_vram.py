"""Formula-driven VRAM estimation and adaptive sizing for local ``llama-server``.

Reads model geometry directly from the GGUF header (layer count, KV head count,
head dimensions, file size) and combines it with measured GPU VRAM to choose
``ctx_per_slot``, ``parallel`` slots, ``--kv-unified``, and local focus radius
without spilling into Windows WDDM shared memory.

Measured on RTX 4060 Ti 8 GB + ``Hy-MT3-7B-Q4_K_M.gguf`` (4,907 MiB weights,
32 layers, 8 KV heads, head_dim 128, ``q8_0`` KV = 68 MiB / 1,024 tokens):
  - ``4096 x 1`` (``-c 4096``):  +5,311 MiB        (formula: 5,459 MiB)
  - ``4096 x 2`` (``-c 8192``):  +5,709 MiB        (formula: 5,763 MiB)
  - ``6144 x 2`` (``-c 12288``): +6,021 MiB        (formula: 6,035 MiB)
  - ``4096 x 3`` (``-c 12288``): +5,981 MiB        (formula: 6,067 MiB)
  - ``8192 x 2`` (``-c 16384``): +6,253 MiB        (formula: 6,307 MiB)

Batch size itself stays capped at the empirically validated 7B quality sweet
spot (32 cues / 2400 source chars) rather than scaling with VRAM: benchmarked
2026-11-26, raising batch size to 48 caused severe source-echo collapse even
with 6144 ctx/slot, whereas scaling slots (``parallel``) and per-slot context
improves throughput and neighbour window without hurting accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import struct
import subprocess
from typing import Sequence

from utils.subprocess_tools import no_window_subprocess_kwargs

logger = logging.getLogger(__name__)

_GGUF_MAGIC = b"GGUF"
_KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q4_0": 18.0 / 32.0,
    "q4_1": 20.0 / 32.0,
}

# Fallback geometry matching official Hy-MT3-7B-Q4_K_M.gguf when no readable
# GGUF exists on disk (e.g. unit tests with synthetic paths or pre-download).
DEFAULT_GGUF_SPEC = None  # populated below after dataclass definition

# CUDA compute-graph + runtime overhead calibrated against RTX 4060 Ti
# measurements (observed ~258–298 MiB across 2–3 slots and 8K–16K KV).
_BASE_COMPUTE_OVERHEAD_MIB = 280.0
_PER_EXTRA_SLOT_OVERHEAD_MIB = 32.0

# Single-layer free-VRAM and total-VRAM safety reserves to prevent Windows WDDM
# shared-RAM spill without double-counting overhead.
_MIN_FREE_RESERVE_MIB = 512.0
_FREE_RESERVE_FRACTION = 0.06
_MIN_TOTAL_RESERVE_MIB = 2176.0
_TOTAL_RESERVE_FRACTION = 0.12

# Ordered from safest minimal footprint up to high-VRAM throughput/context tier.
# Parallel caps at 4 (matches default `translation_max_workers=4`); ctx caps at
# 8192/slot (ample for 32-cue batches + radius=6 + multi-cue repair).
_ADAPTIVE_LADDER: tuple[tuple[int, int], ...] = (
    (1, 4096),
    (2, 4096),
    (3, 4096),
    (3, 6144),
    (4, 6144),
    (4, 8192),
)

_DEVICE_VRAM_RE = re.compile(
    r"\(\s*(\d+)\s*MiB(?:\s*,\s*(\d+)\s*MiB\s*free)?\s*\)",
    re.I,
)


@dataclass(frozen=True)
class GgufVramSpec:
    weights_mib: float
    n_layer: int
    n_head_kv: int
    head_dim_k: int
    head_dim_v: int
    n_ctx_train: int = 32768
    from_file: bool = False


DEFAULT_GGUF_SPEC = GgufVramSpec(
    weights_mib=4907.2,
    n_layer=32,
    n_head_kv=8,
    head_dim_k=128,
    head_dim_v=128,
    n_ctx_train=262144,
    from_file=False,
)


@dataclass(frozen=True)
class LlamaAdaptivePlan:
    ctx_per_slot: int
    parallel: int
    kv_unified: bool
    focus_radius: int
    focus_context_chars: int
    estimated_vram_mib: float
    kv_cache_mib: float
    weights_mib: float
    total_vram_mb: int | None
    free_vram_mb: int | None
    budget_mib: float | None
    adaptive: bool


_SCALAR_FMTS: dict[int, str] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("unexpected EOF reading GGUF header")
    return data


def _read_gguf_string(handle) -> str:
    length = struct.unpack("<Q", _read_exact(handle, 8))[0]
    if length > 16 * 1024 * 1024:
        raise ValueError(f"implausible GGUF string length: {length}")
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _skip_or_read_gguf_value(handle, vtype: int, *, keep: bool):
    fmt = _SCALAR_FMTS.get(vtype)
    if fmt is not None:
        size = struct.calcsize(fmt)
        raw = _read_exact(handle, size)
        return struct.unpack(fmt, raw)[0] if keep else None
    if vtype == 8:
        if keep:
            return _read_gguf_string(handle)
        length = struct.unpack("<Q", _read_exact(handle, 8))[0]
        if length > 64 * 1024 * 1024:
            raise ValueError(f"implausible GGUF string length: {length}")
        handle.seek(length, 1)
        return None
    if vtype == 9:
        elem_type, count = struct.unpack("<IQ", _read_exact(handle, 12))
        if elem_type == 8:
            for _ in range(count):
                length = struct.unpack("<Q", _read_exact(handle, 8))[0]
                handle.seek(length, 1)
            return None
        elem_fmt = _SCALAR_FMTS.get(elem_type)
        if elem_fmt is None:
            raise ValueError(f"unsupported GGUF array element type: {elem_type}")
        handle.seek(count * struct.calcsize(elem_fmt), 1)
        return None
    raise ValueError(f"unsupported GGUF value type: {vtype}")


def read_gguf_vram_spec(model_path: str | Path | None) -> GgufVramSpec:
    """Read KV-relevant geometry and weight size from a GGUF file header.

    Falls back to ``DEFAULT_GGUF_SPEC`` when ``model_path`` is missing, unreadable,
    or not a valid GGUF binary (e.g. unit-test stubs).
    """
    if not model_path:
        return DEFAULT_GGUF_SPEC
    path = Path(model_path)
    try:
        if not path.is_file():
            return DEFAULT_GGUF_SPEC
        size_bytes = path.stat().st_size
        if size_bytes < 24:
            return DEFAULT_GGUF_SPEC
        weights_mib = size_bytes / (1024.0 * 1024.0)
        wanted_suffixes = (
            "general.architecture",
            ".block_count",
            ".embedding_length",
            ".context_length",
            ".attention.head_count",
            ".attention.head_count_kv",
            ".attention.key_length",
            ".attention.value_length",
        )
        meta: dict[str, object] = {}
        with path.open("rb") as handle:
            if _read_exact(handle, 4) != _GGUF_MAGIC:
                return DEFAULT_GGUF_SPEC
            version, _n_tensors, n_kv = struct.unpack("<IQQ", _read_exact(handle, 20))
            if version < 2 or n_kv > 100_000:
                return DEFAULT_GGUF_SPEC
            for _ in range(n_kv):
                key = _read_gguf_string(handle)
                vtype = struct.unpack("<I", _read_exact(handle, 4))[0]
                keep = key == "general.architecture" or any(
                    key.endswith(suffix) for suffix in wanted_suffixes[1:]
                )
                value = _skip_or_read_gguf_value(handle, vtype, keep=keep)
                if keep and value is not None:
                    meta[key] = value
    except (OSError, EOFError, ValueError, struct.error) as exc:
        logger.debug("Could not parse GGUF header from %s (%s); using default spec", path, exc)
        return DEFAULT_GGUF_SPEC

    arch = str(meta.get("general.architecture") or "").strip()
    if not arch:
        return DEFAULT_GGUF_SPEC

    def _int(key: str, default: int = 0) -> int:
        raw = meta.get(key)
        try:
            return int(raw) if raw is not None else default
        except (TypeError, ValueError):
            return default

    n_layer = _int(f"{arch}.block_count")
    n_head = _int(f"{arch}.attention.head_count")
    n_head_kv = _int(f"{arch}.attention.head_count_kv", n_head)
    embd = _int(f"{arch}.embedding_length")
    default_head_dim = (embd // n_head) if (embd > 0 and n_head > 0) else 128
    head_dim_k = _int(f"{arch}.attention.key_length", default_head_dim)
    head_dim_v = _int(f"{arch}.attention.value_length", head_dim_k)
    n_ctx_train = _int(f"{arch}.context_length", DEFAULT_GGUF_SPEC.n_ctx_train)

    if n_layer <= 0 or n_head_kv <= 0 or head_dim_k <= 0 or head_dim_v <= 0:
        return DEFAULT_GGUF_SPEC

    return GgufVramSpec(
        weights_mib=round(weights_mib, 1),
        n_layer=n_layer,
        n_head_kv=n_head_kv,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        n_ctx_train=max(1024, n_ctx_train),
        from_file=True,
    )


def kv_cache_mib(
    spec: GgufVramSpec,
    total_ctx: int,
    *,
    cache_type: str = "q8_0",
) -> float:
    """KV cache footprint in MiB for ``total_ctx`` tokens across all layers."""
    bpe = _KV_BYTES_PER_ELEMENT.get(str(cache_type or "q8_0").strip().lower(), 34.0 / 32.0)
    elements_per_token = spec.n_layer * spec.n_head_kv * (spec.head_dim_k + spec.head_dim_v)
    return round((max(0, int(total_ctx)) * elements_per_token * bpe) / (1024.0 * 1024.0), 1)


def estimate_server_vram_mib(
    spec: GgufVramSpec,
    *,
    ctx_per_slot: int,
    parallel: int,
    cache_type: str = "q8_0",
) -> float:
    """Estimated dedicated VRAM (MiB) for llama-server weights + KV + compute buffers."""
    slots = max(1, int(parallel))
    total_ctx = max(1024, int(ctx_per_slot)) * slots
    kv = kv_cache_mib(spec, total_ctx, cache_type=cache_type)
    overhead = _BASE_COMPUTE_OVERHEAD_MIB + _PER_EXTRA_SLOT_OVERHEAD_MIB * (slots - 1)
    return round(spec.weights_mib + kv + overhead, 1)


def parse_device_vram_mb(devices: Sequence[str]) -> tuple[int, int | None] | None:
    """Extract ``(total_mib, free_mib)`` from ``llama-server --list-devices`` lines."""
    best: tuple[int, int | None] | None = None
    for entry in devices or ():
        match = _DEVICE_VRAM_RE.search(str(entry or ""))
        if not match:
            continue
        total = int(match.group(1))
        free = int(match.group(2)) if match.group(2) is not None else None
        if total <= 0:
            continue
        if best is None or total > best[0]:
            best = (total, free)
    return best


def query_nvidia_smi_vram_mb(*, timeout_s: float = 5.0) -> tuple[int, int] | None:
    """Fast, torch-free ``(total_mib, free_mib)`` probe via ``nvidia-smi``."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            **no_window_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    best: tuple[int, int] | None = None
    for line in (result.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            total = int(float(parts[0]))
            free = int(float(parts[1]))
        except ValueError:
            continue
        if total <= 0 or free < 0:
            continue
        if best is None or total > best[0]:
            best = (total, free)
    return best


def focus_window_for_ctx(ctx_per_slot: int) -> tuple[int, int]:
    """Local ``SourceContext.focus`` ``(radius, context_chars)`` for a per-slot context."""
    if int(ctx_per_slot) >= 6144:
        return (6, 5000)
    return (4, 4000)


class InsufficientLlamaVramError(RuntimeError):
    """Raised when live GPU VRAM budget cannot accommodate even the minimum tier."""


def _vram_budget_mib(
    total_vram_mb: int | None,
    free_vram_mb: int | None,
) -> float | None:
    if total_vram_mb is None or total_vram_mb <= 0:
        return None
    total_cap = total_vram_mb - max(_MIN_TOTAL_RESERVE_MIB, total_vram_mb * _TOTAL_RESERVE_FRACTION)
    if free_vram_mb is not None and free_vram_mb >= 0:
        free_cap = free_vram_mb - max(_MIN_FREE_RESERVE_MIB, total_vram_mb * _FREE_RESERVE_FRACTION)
        return round(max(0.0, min(total_cap, free_cap)), 1)
    return round(max(0.0, total_cap), 1)


def select_adaptive_plan(
    spec: GgufVramSpec | None = None,
    *,
    total_vram_mb: int | None = None,
    free_vram_mb: int | None = None,
    explicit_ctx: int | None = None,
    explicit_parallel: int | None = None,
    cache_type: str = "q8_0",
) -> LlamaAdaptivePlan:
    """Choose ``(ctx_per_slot, parallel)`` from the GGUF VRAM model and GPU headroom.

    Respects explicit ``LLAMACPP_CTX_SIZE`` / ``LLAMACPP_PARALLEL`` overrides.
    When neither GPU total nor free VRAM is available, falls back to the verified
    8 GB baseline (``ctx=4096, parallel=2``).
    """
    model_spec = spec or DEFAULT_GGUF_SPEC
    budget = _vram_budget_mib(total_vram_mb, free_vram_mb)

    def _finish(ctx: int, parallel: int, *, adaptive: bool) -> LlamaAdaptivePlan:
        clamped_ctx = max(1024, min(int(ctx), model_spec.n_ctx_train, 131072))
        clamped_parallel = max(1, min(int(parallel), 16))
        kv = kv_cache_mib(model_spec, clamped_ctx * clamped_parallel, cache_type=cache_type)
        est = estimate_server_vram_mib(
            model_spec,
            ctx_per_slot=clamped_ctx,
            parallel=clamped_parallel,
            cache_type=cache_type,
        )
        radius, ctx_chars = focus_window_for_ctx(clamped_ctx)
        return LlamaAdaptivePlan(
            ctx_per_slot=clamped_ctx,
            parallel=clamped_parallel,
            kv_unified=True,
            focus_radius=radius,
            focus_context_chars=ctx_chars,
            estimated_vram_mib=est,
            kv_cache_mib=kv,
            weights_mib=model_spec.weights_mib,
            total_vram_mb=total_vram_mb,
            free_vram_mb=free_vram_mb,
            budget_mib=budget,
            adaptive=adaptive,
        )

    if explicit_ctx is not None and explicit_parallel is not None:
        return _finish(explicit_ctx, explicit_parallel, adaptive=False)

    if budget is None:
        # No GPU VRAM measurement available: stick to the verified 8 GB baseline.
        return _finish(
            explicit_ctx if explicit_ctx is not None else 4096,
            explicit_parallel if explicit_parallel is not None else 2,
            adaptive=False,
        )

    def _raise_insufficient(min_ctx: int, min_parallel: int, min_est: float) -> None:
        free_note = f"空闲 {free_vram_mb} MiB / " if free_vram_mb is not None else ""
        raise InsufficientLlamaVramError(
            f"本地翻译显存不足：当前可用显存预算 {budget:.0f} MiB"
            f"（{free_note}总计 {total_vram_mb} MiB），"
            f"不足以容纳最小档位（ctx={min_ctx}, parallel={min_parallel}，"
            f"预计需要 {min_est:.0f} MiB）。"
            "请先关闭占用显存的程序（游戏、浏览器硬件加速等），或改用 API 翻译。"
        )

    # Partial override: fix the explicit axis and pick the highest safe tier on the other.
    if explicit_ctx is not None:
        chosen_parallel: int | None = None
        min_est = estimate_server_vram_mib(
            model_spec, ctx_per_slot=explicit_ctx, parallel=1, cache_type=cache_type
        )
        for slots in (1, 2, 3, 4):
            est = estimate_server_vram_mib(
                model_spec, ctx_per_slot=explicit_ctx, parallel=slots, cache_type=cache_type
            )
            if est <= budget:
                chosen_parallel = slots
        if chosen_parallel is None:
            _raise_insufficient(explicit_ctx, 1, min_est)
        return _finish(explicit_ctx, chosen_parallel, adaptive=True)

    if explicit_parallel is not None:
        chosen_ctx: int | None = None
        candidates_ctx = [ctx for ctx in (4096, 6144, 8192) if ctx <= model_spec.n_ctx_train] or [
            min(4096, model_spec.n_ctx_train)
        ]
        min_est = estimate_server_vram_mib(
            model_spec,
            ctx_per_slot=candidates_ctx[0],
            parallel=explicit_parallel,
            cache_type=cache_type,
        )
        for ctx in candidates_ctx:
            est = estimate_server_vram_mib(
                model_spec, ctx_per_slot=ctx, parallel=explicit_parallel, cache_type=cache_type
            )
            if est <= budget:
                chosen_ctx = ctx
        if chosen_ctx is None:
            _raise_insufficient(candidates_ctx[0], explicit_parallel, min_est)
        return _finish(chosen_ctx, explicit_parallel, adaptive=True)

    # Full auto: climb the empirically ordered ladder while staying inside budget.
    chosen_tier: tuple[int, int] | None = None
    first_tier: tuple[int, int, float] | None = None
    for slots, ctx in _ADAPTIVE_LADDER:
        if ctx > model_spec.n_ctx_train:
            continue
        est = estimate_server_vram_mib(
            model_spec, ctx_per_slot=ctx, parallel=slots, cache_type=cache_type
        )
        if first_tier is None:
            first_tier = (slots, ctx, est)
        if est <= budget:
            chosen_tier = (slots, ctx)
    if chosen_tier is None:
        min_slots, min_ctx, min_est = first_tier or (
            _ADAPTIVE_LADDER[0][0],
            _ADAPTIVE_LADDER[0][1],
            estimate_server_vram_mib(
                model_spec,
                ctx_per_slot=_ADAPTIVE_LADDER[0][1],
                parallel=_ADAPTIVE_LADDER[0][0],
                cache_type=cache_type,
            ),
        )
        _raise_insufficient(min_ctx, min_slots, min_est)
    return _finish(chosen_tier[1], chosen_tier[0], adaptive=True)
