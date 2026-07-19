from __future__ import annotations

import pytest

import numpy as np

from tools.boundary.ja.build_feature_cache import (
    _extract_ptm_window_features,
    _validate_cache_memory_snapshot,
)


def test_feature_cache_memory_snapshot_accepts_physical_budget_without_spill() -> None:
    _validate_cache_memory_snapshot(
        {
            "physical_ram_used_mb": 100.0,
            "physical_ram_budget_mb": 200.0,
            "shared_vram_mb": 0.0,
        },
        require_shared_vram=True,
    )


def test_feature_cache_memory_snapshot_rejects_any_shared_vram_spill() -> None:
    with pytest.raises(MemoryError, match="shared VRAM spill"):
        _validate_cache_memory_snapshot(
            {
                "physical_ram_used_mb": 100.0,
                "physical_ram_budget_mb": 200.0,
                "shared_vram_mb": 0.001,
            },
            require_shared_vram=True,
        )


def test_feature_cache_memory_snapshot_rejects_physical_ram_over_budget() -> None:
    with pytest.raises(MemoryError, match="physical RAM budget"):
        _validate_cache_memory_snapshot(
            {
                "physical_ram_used_mb": 201.0,
                "physical_ram_budget_mb": 200.0,
                "shared_vram_mb": 0.0,
            },
            require_shared_vram=True,
        )


def test_singleton_ptm_window_batches_preserve_input_order() -> None:
    class FakeExtractor:
        def __init__(self) -> None:
            self.calls: list[list[float]] = []

        def extract_batch(
            self, audios: list[np.ndarray], *, sample_rate: int
        ) -> list[np.ndarray]:
            assert sample_rate == 16000
            self.calls.append([float(audio[0]) for audio in audios])
            return [np.asarray([[audio[0]]], dtype=np.float32) for audio in audios]

    extractor = FakeExtractor()
    features, batch_count = _extract_ptm_window_features(
        ptm_extractor=extractor,
        window_audios=[
            np.asarray([1.0], dtype=np.float32),
            np.asarray([2.0], dtype=np.float32),
        ],
        sample_rate=16000,
        ptm_window_batch_size=1,
    )
    assert extractor.calls == [[1.0], [2.0]]
    assert batch_count == 2
    assert [float(feature[0, 0]) for feature in features] == [1.0, 2.0]
