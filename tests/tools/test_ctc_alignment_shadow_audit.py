from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment_shadow import SHADOW_RUN_SCHEMA  # noqa: E402
from tools.audits.generate_ctc_alignment_shadow_audit import (  # noqa: E402
    candidates_from_observations,
    observation_files,
)
from tools.audits.generate_ctc_alignment_ab_audit import select_trials  # noqa: E402


def _payload(video: Path) -> dict:
    return {
        "schema": SHADOW_RUN_SCHEMA,
        "job_id": "job-1",
        "audio_cache_key": "audio-key",
        "source_video_path": str(video),
        "source_video_duration_s": 20.0,
        "minimum_disagreement_ms": 20.0,
        "comparisons": [
            {
                "status": "ok",
                "chunk_index": 3,
                "text": "テスト",
                "primary_start_abs_s": 3.0,
                "primary_end_abs_s": 8.0,
                "shadow_start_abs_s": 3.1,
                "shadow_end_abs_s": 7.9,
            },
            {"status": "declined", "chunk_index": 4, "text": "除外"},
        ],
    }


def test_shadow_observations_become_deduplicated_jav_candidates(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    encoded = json.dumps(_payload(video), ensure_ascii=False)
    first.write_text(encoded, encoding="utf-8")
    second.write_text(encoded, encoding="utf-8")

    paths = observation_files([str(tmp_path)])
    candidates = candidates_from_observations(paths)
    assert len(paths) == 2
    assert len(candidates) == 1
    row = candidates[0]
    assert row["domain"] == "jav"
    assert row["model_a_start_s"] == 3.0
    assert row["model_b_end_s"] == 7.9
    assert row["minimum_delta_ms"] == 20.0
    assert row["candidate_id"].startswith("jav-shadow:")


def test_shadow_selection_honors_threshold_stored_with_observation(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    payload = _payload(video)
    payload["minimum_disagreement_ms"] = 120.0
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    selected = select_trials(
        candidates_from_observations([path]),
        per_boundary=2,
        clip_s=2.5,
        seed=7,
        domains=("jav",),
        minimum_delta_ms=0.0,
    )
    assert selected == []


def test_missing_source_video_is_not_offered_for_audit(tmp_path) -> None:
    path = tmp_path / "missing.json"
    path.write_text(
        json.dumps(_payload(tmp_path / "missing.mp4"), ensure_ascii=False),
        encoding="utf-8",
    )
    assert candidates_from_observations([path]) == []
