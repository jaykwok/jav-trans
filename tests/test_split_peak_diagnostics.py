from __future__ import annotations

import numpy as np
import pytest

from boundary.ja.backend import SpeechBoundaryJaConfig
from tools.boundary.ja.diagnose_split_peaks import diagnose_split_peaks


def _base_config() -> SpeechBoundaryJaConfig:
    return SpeechBoundaryJaConfig(
        threshold=0.5,
        speech_on_threshold=0.5,
        speech_off_threshold=0.5,
        frame_hop_s=0.1,
        frame_dilation_s=0.0,
        min_segment_s=0.0,
        split_smooth_s=0.0,
        split_nms_s=0.2,
        split_snap_s=0.0,
        min_split_segment_s=0.1,
        split_score_quantile=0.5,
        split_prominence_quantile=0.5,
    )


def test_split_peak_diagnostic_reports_selected_adaptive_peaks():
    frame_count = 120
    split_scores = np.full(frame_count, 0.02, dtype=np.float32)
    for peak, value in ((30, 0.18), (65, 0.24), (95, 0.16)):
        split_scores[peak - 1 : peak + 2] = [0.08, value, 0.08]
    payload = {
        "duration_s": 12.0,
        "frame_hop_s": 0.1,
        "scores": [0.9] * frame_count,
        "split_boundary_scores": split_scores.tolist(),
    }

    diagnostic = diagnose_split_peaks(payload, config=_base_config(), top_candidates=3)

    assert diagnostic["summary"]["coarse_segment_count"] == 1
    assert diagnostic["summary"]["islands_with_candidates"] == 1
    assert diagnostic["summary"]["islands_with_selected_splits"] == 1
    island = diagnostic["islands"][0]
    assert island["decision"] == "split_selected"
    assert len(island["selected_frames"]) == 2
    assert len(island["parts"]) == 3
    assert island["top_candidates"][0]["frame"] == pytest.approx(65, abs=1)


def test_split_peak_diagnostic_distinguishes_missing_split_signal():
    payload = {
        "duration_s": 10.0,
        "frame_hop_s": 0.1,
        "scores": [0.9] * 100,
        "split_boundary_scores": [0.05] * 100,
    }

    diagnostic = diagnose_split_peaks(payload, config=_base_config())

    island = diagnostic["islands"][0]
    assert island["decision"] == "no_local_prominent_peaks"
    assert island["candidate_count"] == 0
    assert island["selected_frames"] == []
    assert diagnostic["summary"]["redecoded_segment_count"] == 1
