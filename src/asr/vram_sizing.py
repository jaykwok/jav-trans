"""Formula-driven VRAM estimation and direct batch sizing for Qwen3-ASR 1.7B.

Replaces the cross-job binary-search ``gpu_batch_profiles.json`` learner with a
stateless, physics-calibrated estimator modeled after
``llm.backends.llamacpp_vram``:

Measured on RTX 4060 Ti 8 GB + ``Qwen3-ASR-1.7B-JA-Anime-Galgame-hf``
(bf16, sdpa, ``chunk_max_s=30.0``, dynamic ~300-token decode budget):
  - Model weights loaded: ``3,905.6 MiB`` allocated
  - ``batch = 5``:  ``5,023.6 MiB`` allocated (``+223.6 MiB / sample``)
  - ``batch = 11``: ``6,360.4 MiB`` allocated (``+223.2 MiB / sample``),
    ``7,096.0 MiB`` max reserved (``1.116x`` PyTorch caching-allocator factor)

Runtime CUDA OOM downshift (halving ``ASR_BATCH_SIZE`` inside a fresh worker)
and ``set_per_process_memory_fraction`` remain as dynamic safety backstops.
"""

from __future__ import annotations

from dataclasses import dataclass

# Calibrated on Qwen3-ASR-1.7B-JA-Anime-Galgame-hf (bf16/fp16, 30s chunks, sdpa).
_BASE_WEIGHTS_BF16_MIB = 3906.0
_PER_SAMPLE_ALLOC_30S_BF16_MIB = 224.0
_BASE_CHUNK_MAX_S = 30.0
_BASE_DECODE_TOKENS = 300
# Qwen3-1.7B KV cache: 28 layers * 2(K+V) * 8 kv_heads * 128 dim * 2 B ≈ 0.109 MiB/tok.
_KV_MIB_PER_EXTRA_TOKEN_BF16 = 0.11

# PyTorch CUDA caching allocator fragmentation/reserved multiplier over allocated
# peak (measured 7096 / 6360 = 1.116 at batch=11).
_ALLOCATOR_RESERVED_FACTOR = 1.12

# Safety reserves to prevent Windows WDDM shared-memory spill.
# When live free VRAM is known, desktop usage is already excluded from
# `free + own_reserved`, so only a modest transient headroom reserve is needed.
# When live free VRAM is unavailable (e.g. unit-test stubs or pre-init probes),
# reserve enough from physical total VRAM for typical Windows desktop overhead.
_FREE_HEADROOM_MIB = 448.0
_FREE_HEADROOM_FRACTION = 0.05
_TOTAL_DESKTOP_RESERVE_MIB = 1280.0
_TOTAL_DESKTOP_RESERVE_FRACTION = 0.10

# Upper bound on auto batch size: ASR batches run in chronological order without
# duration bucketing, so batches larger than 24 add padding waste with minimal
# GPU compute gain even on 16-24 GB cards.
MAX_AUTO_ASR_BATCH_SIZE = 24


@dataclass(frozen=True)
class AsrBatchPlan:
    batch_size: int
    estimated_allocated_mb: float
    estimated_reserved_mb: float
    weights_mb: float
    per_sample_mb: float
    usable_vram_mb: float
    total_vram_mb: float
    free_vram_mb: float | None
    own_reserved_mb: float
    budget_mb: float
    dtype: str
    attention: str
    chunk_max_s: float
    source: str


def _normalize_dtype(dtype: str) -> str:
    raw = str(dtype or "").strip().lower()
    if raw in {"float32", "fp32"}:
        return "float32"
    if raw in {"float16", "fp16"}:
        return "float16"
    return "bfloat16"


def _normalize_attention(attention: str) -> str:
    raw = str(attention or "").strip().lower()
    if raw == "eager":
        return "eager"
    if raw in {"flash_attention_2", "flash2", "fa2"}:
        return "flash_attention_2"
    return "sdpa"


def _effective_decode_tokens(
    *,
    chunk_max_s: float,
    max_new_tokens: int | None = None,
    tokens_per_second: float = 10.0,
) -> int:
    if max_new_tokens is not None and max_new_tokens > 0:
        return int(max_new_tokens)
    rate = max(1.0, float(tokens_per_second or 10.0))
    return max(32, int(round(max(1.0, float(chunk_max_s)) * rate)))


def estimate_asr_vram_mb(
    batch_size: int,
    *,
    dtype: str = "bfloat16",
    chunk_max_s: float = _BASE_CHUNK_MAX_S,
    attention: str = "sdpa",
    max_new_tokens: int | None = None,
    tokens_per_second: float = 10.0,
) -> tuple[float, float, float, float]:
    """Return ``(allocated_mb, reserved_mb, weights_mb, per_sample_mb)`` for ``batch_size``."""
    norm_dtype = _normalize_dtype(dtype)
    norm_attn = _normalize_attention(attention)
    dtype_scale = 2.0 if norm_dtype == "float32" else 1.0
    attn_scale = 1.45 if norm_attn == "eager" else 1.0

    clamped_chunk_s = min(120.0, max(5.0, float(chunk_max_s or _BASE_CHUNK_MAX_S)))
    # Fixed prompt/decoder framing means tiny chunks do not drop linearly to zero.
    chunk_scale = max(0.35, clamped_chunk_s / _BASE_CHUNK_MAX_S)

    decode_tokens = _effective_decode_tokens(
        chunk_max_s=clamped_chunk_s,
        max_new_tokens=max_new_tokens,
        tokens_per_second=tokens_per_second,
    )
    extra_decode_tokens = max(0, decode_tokens - _BASE_DECODE_TOKENS)
    extra_kv_mb = extra_decode_tokens * _KV_MIB_PER_EXTRA_TOKEN_BF16 * dtype_scale

    weights_mb = round(_BASE_WEIGHTS_BF16_MIB * dtype_scale, 1)
    per_sample_mb = round(
        _PER_SAMPLE_ALLOC_30S_BF16_MIB * chunk_scale * dtype_scale * attn_scale + extra_kv_mb,
        1,
    )
    allocated_mb = round(weights_mb + max(1, int(batch_size)) * per_sample_mb, 1)
    reserved_mb = round(allocated_mb * _ALLOCATOR_RESERVED_FACTOR, 1)
    return allocated_mb, reserved_mb, weights_mb, per_sample_mb


def select_asr_batch_plan(
    *,
    total_vram_mb: float,
    budget_mb: float,
    free_vram_mb: float | None = None,
    own_reserved_mb: float = 0.0,
    explicit_budget: bool = False,
    dtype: str = "bfloat16",
    chunk_max_s: float = _BASE_CHUNK_MAX_S,
    attention: str = "sdpa",
    max_new_tokens: int | None = None,
    tokens_per_second: float = 10.0,
    max_batch: int = MAX_AUTO_ASR_BATCH_SIZE,
) -> AsrBatchPlan:
    """Compute the optimal ASR batch size directly from VRAM physics.

    When ``free_vram_mb`` is supplied inside a persistent GPU worker,
    ``own_reserved_mb`` (memory already held by this worker's own resident ASR
    model) is added back so subsequent jobs do not treat their own resident
    weights as external desktop load.
    """
    norm_dtype = _normalize_dtype(dtype)
    norm_attn = _normalize_attention(attention)
    clamped_chunk_s = min(120.0, max(5.0, float(chunk_max_s or _BASE_CHUNK_MAX_S)))
    cap = max(1, int(max_batch))

    total = max(0.0, float(total_vram_mb or 0.0))
    budget = max(0.0, float(budget_mb or 0.0))
    own_reserved = max(0.0, float(own_reserved_mb or 0.0))
    live_free = (
        max(0.0, float(free_vram_mb))
        if free_vram_mb is not None
        else None
    )

    # Determine usable reserved VRAM ceiling.
    caps: list[float] = []
    if budget > 0.0:
        caps.append(budget)
    if total > 0.0 and not explicit_budget:
        desktop_reserve = max(_TOTAL_DESKTOP_RESERVE_MIB, total * _TOTAL_DESKTOP_RESERVE_FRACTION)
        caps.append(max(0.0, total - desktop_reserve))
    if live_free is not None:
        process_available = live_free + own_reserved
        headroom = max(_FREE_HEADROOM_MIB, total * _FREE_HEADROOM_FRACTION)
        caps.append(max(0.0, process_available - headroom))

    usable_vram_mb = round(min(caps), 1) if caps else 0.0

    chosen = 1
    chosen_alloc, chosen_res, weights_mb, per_sample_mb = estimate_asr_vram_mb(
        1,
        dtype=norm_dtype,
        chunk_max_s=clamped_chunk_s,
        attention=norm_attn,
        max_new_tokens=max_new_tokens,
        tokens_per_second=tokens_per_second,
    )
    if usable_vram_mb > 0.0:
        for candidate in range(1, cap + 1):
            alloc_mb, res_mb, w_mb, s_mb = estimate_asr_vram_mb(
                candidate,
                dtype=norm_dtype,
                chunk_max_s=clamped_chunk_s,
                attention=norm_attn,
                max_new_tokens=max_new_tokens,
                tokens_per_second=tokens_per_second,
            )
            if res_mb <= usable_vram_mb:
                chosen = candidate
                chosen_alloc, chosen_res, weights_mb, per_sample_mb = (
                    alloc_mb,
                    res_mb,
                    w_mb,
                    s_mb,
                )
            else:
                break

    return AsrBatchPlan(
        batch_size=chosen,
        estimated_allocated_mb=chosen_alloc,
        estimated_reserved_mb=chosen_res,
        weights_mb=weights_mb,
        per_sample_mb=per_sample_mb,
        usable_vram_mb=usable_vram_mb,
        total_vram_mb=round(total, 1),
        free_vram_mb=round(live_free, 1) if live_free is not None else None,
        own_reserved_mb=round(own_reserved, 1),
        budget_mb=round(budget, 1),
        dtype=norm_dtype,
        attention=norm_attn,
        chunk_max_s=clamped_chunk_s,
        source="formula_vram",
    )
