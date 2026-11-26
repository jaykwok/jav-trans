"""Tests for GGUF header parsing and formula-driven llama-server VRAM sizing."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from core import typed_config
from llm.backends import llamacpp_server
from llm.backends.llamacpp_vram import (
    DEFAULT_GGUF_SPEC,
    GgufVramSpec,
    InsufficientLlamaVramError,
    estimate_server_vram_mib,
    kv_cache_mib,
    parse_device_vram_mb,
    read_gguf_vram_spec,
    select_adaptive_plan,
)


def _write_synthetic_gguf(path: Path, kv_pairs: list[tuple[str, int, object]], *, pad_bytes: int = 0) -> None:
    def _pack_str(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, 0, len(kv_pairs)))
    for key, vtype, val in kv_pairs:
        buf.extend(_pack_str(key))
        buf.extend(struct.pack("<I", vtype))
        if vtype == 8:
            buf.extend(_pack_str(str(val)))
        elif vtype == 4:
            buf.extend(struct.pack("<I", int(val)))
        else:
            raise ValueError(vtype)
    if pad_bytes > 0:
        buf.extend(b"\x00" * pad_bytes)
    path.write_bytes(bytes(buf))


def test_read_gguf_vram_spec_parses_real_header(tmp_path):
    gguf = tmp_path / "model.gguf"
    _write_synthetic_gguf(
        gguf,
        [
            ("general.architecture", 8, "hunyuan-dense"),
            ("hunyuan-dense.block_count", 4, 32),
            ("hunyuan-dense.context_length", 4, 262144),
            ("hunyuan-dense.embedding_length", 4, 4096),
            ("hunyuan-dense.attention.head_count", 4, 32),
            ("hunyuan-dense.attention.head_count_kv", 4, 8),
            ("hunyuan-dense.attention.key_length", 4, 128),
            ("hunyuan-dense.attention.value_length", 4, 128),
        ],
        pad_bytes=2 * 1024 * 1024,
    )
    spec = read_gguf_vram_spec(gguf)
    assert spec.from_file is True
    assert spec.n_layer == 32
    assert spec.n_head_kv == 8
    assert spec.head_dim_k == 128
    assert spec.head_dim_v == 128
    assert spec.n_ctx_train == 262144
    assert 2.0 <= spec.weights_mib <= 2.2


def test_read_gguf_vram_spec_falls_back_on_missing_or_non_gguf(tmp_path):
    assert read_gguf_vram_spec(None) == DEFAULT_GGUF_SPEC
    assert read_gguf_vram_spec(tmp_path / "missing.gguf") == DEFAULT_GGUF_SPEC
    fake = tmp_path / "stub.gguf"
    fake.write_bytes(b"not-a-gguf")
    assert read_gguf_vram_spec(fake) == DEFAULT_GGUF_SPEC


def test_kv_cache_and_vram_estimate_match_measured_hymt3_q4():
    # Hy-MT3-7B-Q4_K_M: 32 layers, 8 kv heads, 128+128 dim -> 68 MiB per 1024 q8_0 tokens.
    assert kv_cache_mib(DEFAULT_GGUF_SPEC, 8192, cache_type="q8_0") == 544.0
    assert kv_cache_mib(DEFAULT_GGUF_SPEC, 16384, cache_type="q8_0") == 1088.0
    est_8k = estimate_server_vram_mib(DEFAULT_GGUF_SPEC, ctx_per_slot=4096, parallel=2)
    # Measured RTX 4060 Ti delta was ~5709 MiB; calibrated formula stays just above it.
    assert 5700.0 <= est_8k <= 5850.0


def test_parse_device_vram_mb_extracts_total_and_free():
    devices = ["CUDA0: NVIDIA GeForce RTX 4060 Ti (8187 MiB, 7075 MiB free)"]
    assert parse_device_vram_mb(devices) == (8187, 7075)
    assert parse_device_vram_mb(["CUDA0: Mock GPU (12288 MiB)"]) == (12288, None)
    assert parse_device_vram_mb(["gpu0"]) is None


def test_select_adaptive_plan_scales_across_vram_tiers():
    # Standard 8 GB card (RTX 4060 Ti, typical ~1.5 GiB desktop) -> 4096 x 2 sweet spot.
    p8 = select_adaptive_plan(DEFAULT_GGUF_SPEC, total_vram_mb=8188, free_vram_mb=6430)
    assert (p8.parallel, p8.ctx_per_slot, p8.focus_radius, p8.kv_unified) == (2, 4096, 4, True)

    # Tight 8 GB desktop (~2.0 GiB used -> 5588 MiB budget) -> safely steps down to 1 slot.
    p8_tight = select_adaptive_plan(DEFAULT_GGUF_SPEC, total_vram_mb=8188, free_vram_mb=6100)
    assert (p8_tight.parallel, p8_tight.ctx_per_slot) == (1, 4096)

    # 12 GB card -> scales up to 4 parallel slots and 8192 ctx/slot with wider radius=6.
    p12 = select_adaptive_plan(DEFAULT_GGUF_SPEC, total_vram_mb=12288, free_vram_mb=10800)
    assert p12.parallel == 4
    assert p12.ctx_per_slot == 8192
    assert p12.focus_radius == 6

    # 16 GB card -> caps cleanly at 4 slots x 8192 ctx/slot.
    p16 = select_adaptive_plan(DEFAULT_GGUF_SPEC, total_vram_mb=16384, free_vram_mb=14800)
    assert (p16.parallel, p16.ctx_per_slot, p16.focus_radius) == (4, 8192, 6)

    # Missing GPU VRAM info -> safe 8 GB baseline.
    p_none = select_adaptive_plan(DEFAULT_GGUF_SPEC, total_vram_mb=None, free_vram_mb=None)
    assert (p_none.parallel, p_none.ctx_per_slot, p_none.adaptive) == (2, 4096, False)


def test_select_adaptive_plan_respects_explicit_and_partial_overrides():
    # Both explicit -> used verbatim, adaptive=False.
    both = select_adaptive_plan(
        DEFAULT_GGUF_SPEC,
        total_vram_mb=16384,
        free_vram_mb=15000,
        explicit_ctx=4096,
        explicit_parallel=1,
    )
    assert (both.parallel, both.ctx_per_slot, both.adaptive) == (1, 4096, False)

    # Partial override: pin ctx=6144, auto-pick parallel on 12 GB card.
    partial = select_adaptive_plan(
        DEFAULT_GGUF_SPEC,
        total_vram_mb=12288,
        free_vram_mb=10800,
        explicit_ctx=6144,
        explicit_parallel=None,
    )
    assert partial.ctx_per_slot == 6144
    assert partial.parallel == 4
    assert partial.adaptive is True


def test_server_limits_handles_auto_and_reports_invalid_env(monkeypatch):
    typed_config.clear_configuration_problems()
    monkeypatch.setenv("LLAMACPP_CTX_SIZE", "auto")
    monkeypatch.setenv("LLAMACPP_PARALLEL", "auto")
    try:
        limits = llamacpp_server.server_limits(
            devices=["CUDA0: RTX 4060 Ti (8187 MiB, 7075 MiB free)"]
        )
        assert limits["ctx"] == 4096
        assert limits["parallel"] == 2
        assert typed_config.configuration_problems() == []

        # Non-auto garbage still surfaces as a startup config problem.
        monkeypatch.setenv("LLAMACPP_CTX_SIZE", "nan")
        llamacpp_server.server_limits()
        problems = typed_config.configuration_problems()
        assert any("LLAMACPP_CTX_SIZE" in str(item) for item in problems)
    finally:
        typed_config.clear_configuration_problems()


def test_smaller_model_on_8gb_unlocks_wider_slots_via_formula():
    small_spec = GgufVramSpec(
        weights_mib=2200.0,
        n_layer=24,
        n_head_kv=4,
        head_dim_k=128,
        head_dim_v=128,
        n_ctx_train=32768,
        from_file=True,
    )
    plan = select_adaptive_plan(small_spec, total_vram_mb=8188, free_vram_mb=7000)
    assert plan.parallel == 4
    assert plan.ctx_per_slot == 8192


def test_select_adaptive_plan_rejects_insufficient_and_zero_free_vram():
    # 8 GiB card with only 4096 MiB free -> budget 3584 MiB < 5459 MiB minimum -> fail fast.
    with pytest.raises(InsufficientLlamaVramError, match="显存不足"):
        select_adaptive_plan(DEFAULT_GGUF_SPEC, total_vram_mb=8188, free_vram_mb=4096)

    # Exactly 0 MiB free must yield 0 MiB budget, never fall back to total_cap / 2 slots.
    with pytest.raises(InsufficientLlamaVramError, match="显存不足"):
        select_adaptive_plan(DEFAULT_GGUF_SPEC, total_vram_mb=8188, free_vram_mb=0)

    # Partial override that cannot fit budget also fails fast instead of oversubscribing.
    with pytest.raises(InsufficientLlamaVramError, match="显存不足"):
        select_adaptive_plan(
            DEFAULT_GGUF_SPEC,
            total_vram_mb=8188,
            free_vram_mb=4096,
            explicit_ctx=8192,
        )
