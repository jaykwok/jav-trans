from __future__ import annotations

from tools.web.smoke import summarize_job


def test_web_smoke_summary_prefers_full_cueqc_decisions() -> None:
    asr_details = {
        "transcript_chunks": [
            {
                "text": "keep",
                "cueqc_shadow": {
                    "mode": "cueqc_mamba_v4_binary",
                    "display_hint": "keep",
                },
            }
        ],
        "stage_timings": {"cueqc_drop_count": 1},
        "cueqc_shadow": {
            "candidate_count": 3,
            "counts": {"display_hint": {"keep": 2, "drop": 1}},
            "fallback_summary": {"count": 1, "stages": {"capture": 1}},
            "decisions": [
                {"mode": "cueqc_mamba_v4_binary", "display_hint": "keep"},
                {"mode": "cueqc_mamba_v4_binary", "display_hint": "drop"},
                {
                    "mode": "fallback_keep",
                    "display_hint": "keep",
                    "fallback_stage": "capture",
                    "fallback_detail": "worker pipe closed",
                    "reasons": ["cueqc_capture_error"],
                },
            ],
        },
    }

    summary = summarize_job._cueqc_summary(asr_details)

    assert summary["source"] == "cueqc_shadow.decisions"
    assert summary["candidate_count"] == 3
    assert summary["transcript_chunks"] == 1
    assert summary["model_count"] == 2
    assert summary["drop_count"] == 1
    assert summary["fallback_count"] == 1
    assert summary["stage_cueqc_drop_count"] == 1
    assert summary["display_hint"] == {"keep": 2, "drop": 1}
    assert summary["fallback_stage"] == {"capture": 1}
    assert summary["fallback_reason"] == {"cueqc_capture_error": 1}
    assert summary["fallback_detail_top"] == {"worker pipe closed": 1}
