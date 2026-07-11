from __future__ import annotations

import json

from pipeline import batch_profile


def test_profile_v2_binary_searches_safe_and_unsafe_bounds(monkeypatch, tmp_path):
    profile_path = tmp_path / "gpu_batch_profiles.json"
    monkeypatch.setenv("GPU_BATCH_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("GPU_BATCH_PROFILE_ENABLED", "1")
    monkeypatch.setenv("GPU_BATCH_PROFILE_GROWTH_THRESHOLD", "0.80")
    identity = {"stage": "asr_text_transcribe", "gpu": "test"}

    first = batch_profile.record_success(
        identity,
        batch_size=4,
        peak_allocated_mb=2000,
        budget_mb=6000,
        max_batch=16,
    )
    assert first["safe_batch"] == 4
    assert first["unsafe_batch"] is None
    assert first["recommended_batch"] == 10

    failed = batch_profile.record_oom(identity, batch_size=10, max_batch=16)
    assert failed["safe_batch"] == 4
    assert failed["unsafe_batch"] == 10
    assert failed["recommended_batch"] == 7

    recovered = batch_profile.record_success(
        identity,
        batch_size=7,
        peak_allocated_mb=3000,
        budget_mb=6000,
        max_batch=16,
    )
    assert recovered["safe_batch"] == 7
    assert recovered["unsafe_batch"] == 10
    assert recovered["recommended_batch"] == 8

    recommendation, entry = batch_profile.recommendation(
        identity,
        heuristic_batch=4,
        max_batch=16,
    )
    assert recommendation == 8
    assert entry["safe_batch"] == 7


def test_profile_v2_does_not_probe_when_peak_is_high(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_BATCH_PROFILE_PATH", str(tmp_path / "profiles.json"))
    identity = {"stage": "semantic_split_model", "gpu": "test"}

    entry = batch_profile.record_success(
        identity,
        batch_size=64,
        peak_allocated_mb=5400,
        budget_mb=6000,
        max_batch=512,
    )

    assert entry["safe_batch"] == 64
    assert entry["recommended_batch"] == 64


def test_profile_v1_is_not_silently_migrated(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "gpu_inference_batch_profiles_v1",
                "version": 1,
                "profiles": {"legacy": {"recommended_batch": 99}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GPU_BATCH_PROFILE_PATH", str(profile_path))

    recommended, entry = batch_profile.recommendation(
        {"stage": "asr_text_transcribe"},
        heuristic_batch=5,
        max_batch=16,
    )

    assert recommended == 5
    assert entry == {}
