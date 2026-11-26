"""Tests for formula-driven Qwen3-ASR 1.7B VRAM estimation and batch sizing."""

from __future__ import annotations

from asr.vram_sizing import (
    MAX_AUTO_ASR_BATCH_SIZE,
    estimate_asr_vram_mb,
    select_asr_batch_plan,
)


def test_estimate_matches_measured_rtx_4060_ti_points():
    # Measured on RTX 4060 Ti 8 GB (bf16, sdpa, 30s):
    # batch=5 -> 5023.6 MiB alloc; batch=11 -> 6360.4 MiB alloc, 7096 MiB reserved.
    alloc5, _res5, weights, per_sample = estimate_asr_vram_mb(5)
    alloc11, res11, _, _ = estimate_asr_vram_mb(11)
    assert 3850.0 <= weights <= 3950.0
    assert 215.0 <= per_sample <= 235.0
    assert abs(alloc5 - 5023.6) <= 40.0
    assert abs(alloc11 - 6360.4) <= 50.0
    assert abs(res11 - 7096.0) <= 100.0


def test_select_plan_scales_across_vram_tiers_and_caps_at_24():
    # 8 GB clean desktop -> jumps directly to 8-9 (not old conservative 5).
    p8_clean = select_asr_batch_plan(
        total_vram_mb=8188.0,
        budget_mb=8188.0 * 0.95,
        free_vram_mb=7075.0,
    )
    assert 8 <= p8_clean.batch_size <= 10
    assert p8_clean.estimated_reserved_mb <= p8_clean.usable_vram_mb

    # 8 GB heavily loaded desktop -> automatically steps down to avoid WDDM spill.
    p8_busy = select_asr_batch_plan(
        total_vram_mb=8188.0,
        budget_mb=8188.0 * 0.95,
        free_vram_mb=5500.0,
    )
    assert 1 <= p8_busy.batch_size < p8_clean.batch_size

    # 12 GB card -> scales well past the old hardcoded 16 ceiling.
    p12 = select_asr_batch_plan(
        total_vram_mb=12288.0,
        budget_mb=12288.0 * 0.95,
        free_vram_mb=10800.0,
    )
    assert 18 <= p12.batch_size <= MAX_AUTO_ASR_BATCH_SIZE

    # 16 GB / 24 GB card -> caps cleanly at MAX_AUTO_ASR_BATCH_SIZE (24).
    p24 = select_asr_batch_plan(
        total_vram_mb=24576.0,
        budget_mb=24576.0 * 0.95,
        free_vram_mb=22500.0,
    )
    assert p24.batch_size == MAX_AUTO_ASR_BATCH_SIZE


def test_resident_own_reserved_vram_does_not_penalize_subsequent_jobs():
    fresh = select_asr_batch_plan(
        total_vram_mb=8188.0,
        budget_mb=7778.0,
        free_vram_mb=6800.0,
        own_reserved_mb=0.0,
    )
    resident = select_asr_batch_plan(
        total_vram_mb=8188.0,
        budget_mb=7778.0,
        free_vram_mb=1932.0,
        own_reserved_mb=4868.0,
    )
    assert resident.batch_size == fresh.batch_size
    assert resident.usable_vram_mb == fresh.usable_vram_mb


def test_dtype_attention_and_chunk_length_reduce_or_raise_batch_sensibly():
    bf16_sdpa = select_asr_batch_plan(
        total_vram_mb=12288.0,
        budget_mb=11673.0,
        free_vram_mb=10500.0,
        dtype="bfloat16",
        attention="sdpa",
        chunk_max_s=30.0,
    )
    eager = select_asr_batch_plan(
        total_vram_mb=12288.0,
        budget_mb=11673.0,
        free_vram_mb=10500.0,
        dtype="bfloat16",
        attention="eager",
        chunk_max_s=30.0,
    )
    fp32 = select_asr_batch_plan(
        total_vram_mb=12288.0,
        budget_mb=11673.0,
        free_vram_mb=10500.0,
        dtype="float32",
        attention="sdpa",
        chunk_max_s=30.0,
    )
    short_chunk = select_asr_batch_plan(
        total_vram_mb=8188.0,
        budget_mb=7778.0,
        free_vram_mb=6800.0,
        chunk_max_s=15.0,
    )
    long_chunk = select_asr_batch_plan(
        total_vram_mb=8188.0,
        budget_mb=7778.0,
        free_vram_mb=6800.0,
        chunk_max_s=45.0,
    )
    assert eager.batch_size < bf16_sdpa.batch_size
    assert fp32.batch_size < bf16_sdpa.batch_size
    assert short_chunk.batch_size > long_chunk.batch_size


def test_explicit_budget_is_honored_directly():
    tight = select_asr_batch_plan(
        total_vram_mb=16384.0,
        budget_mb=5600.0,
        free_vram_mb=14000.0,
        explicit_budget=True,
    )
    assert tight.usable_vram_mb == 5600.0
    assert tight.batch_size <= 5
