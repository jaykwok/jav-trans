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


def test_a_stale_version_is_not_silently_migrated(monkeypatch, tmp_path):
    """The version bumps in `batch_profile` exist because a `safe_batch` learned
    under a different chunk geometry or decode budget is not a claim about the
    current one. That only holds if the version is read - it was written and
    never checked, and a live file stamped `version: 2` was still handing out a
    `safe_batch: 16` measured against a quarter of the current KV cache."""
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": batch_profile.PROFILE_SCHEMA,
                "version": batch_profile.PROFILE_VERSION - 1,
                "profiles": {"legacy": {"safe_batch": 16, "recommended_batch": 16}},
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


def test_profiles_are_pruned_to_most_recent_identities(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    monkeypatch.setenv("GPU_BATCH_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("GPU_BATCH_PROFILE_MAX_ENTRIES", "3")

    timestamps = iter(float(index) for index in range(20))
    monkeypatch.setattr(batch_profile.time, "time", lambda: next(timestamps))
    for index in range(5):
        batch_profile.record_success(
            {"hardware": {"fingerprint": f"gpu-{index}"}, "workload": {}},
            batch_size=index + 1,
            peak_allocated_mb=1000,
            budget_mb=6000,
            max_batch=16,
        )

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert len(payload["profiles"]) == 3
    kept = {
        entry["identity"]["hardware"]["fingerprint"]
        for entry in payload["profiles"].values()
    }
    assert kept == {"gpu-2", "gpu-3", "gpu-4"}


def test_recommendation_refreshes_lru_recency(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    monkeypatch.setenv("GPU_BATCH_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("GPU_BATCH_PROFILE_MAX_ENTRIES", "2")
    clock = iter(float(index) for index in range(1, 20))
    monkeypatch.setattr(batch_profile.time, "time", lambda: next(clock))
    identities = [
        {"hardware": {"fingerprint": f"gpu-{index}"}, "workload": {}}
        for index in range(3)
    ]
    for identity in identities[:2]:
        batch_profile.record_success(
            identity,
            batch_size=4,
            peak_allocated_mb=3000,
            budget_mb=6000,
            max_batch=16,
        )

    batch_profile.recommendation(
        identities[0],
        heuristic_batch=4,
        max_batch=16,
    )
    batch_profile.record_success(
        identities[2],
        batch_size=4,
        peak_allocated_mb=3000,
        budget_mb=6000,
        max_batch=16,
    )

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    kept = {
        entry["identity"]["hardware"]["fingerprint"]
        for entry in payload["profiles"].values()
    }
    assert kept == {"gpu-0", "gpu-2"}


def test_recommendation_touch_prunes_an_overfull_existing_file(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    monkeypatch.setenv("GPU_BATCH_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("GPU_BATCH_PROFILE_MAX_ENTRIES", "2")
    identities = [
        {"hardware": {"fingerprint": f"gpu-{index}"}, "workload": {}}
        for index in range(3)
    ]
    profiles = {}
    for index, identity in enumerate(identities):
        profiles[batch_profile.identity_key(identity)] = {
            "identity": identity,
            "recommended_batch": 4,
            "updated_ts": float(index + 1),
            "last_used_ts": float(index + 1),
        }
    profile_path.write_text(
        json.dumps(
            {
                "schema": batch_profile.PROFILE_SCHEMA,
                "version": batch_profile.PROFILE_VERSION,
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(batch_profile.time, "time", lambda: 10.0)

    recommended, _entry = batch_profile.recommendation(
        identities[0],
        heuristic_batch=2,
        max_batch=16,
    )

    assert recommended == 4
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    kept = {
        entry["identity"]["hardware"]["fingerprint"]
        for entry in payload["profiles"].values()
    }
    assert kept == {"gpu-0", "gpu-2"}
