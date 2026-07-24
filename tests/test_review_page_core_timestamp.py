from __future__ import annotations

from pathlib import Path

import pytest

from tools.audits.review_page_core import (
    AUDIO_SPAN_PLAYER_JS,
    AUDIT_REVIEW_CORE_JS,
    format_audit_timestamp,
)


def test_audit_core_uses_shared_mmss_mmm_labels() -> None:
    assert format_audit_timestamp(65.153) == "01:05.153"
    assert "function formatAuditTimestamp" in AUDIO_SPAN_PLAYER_JS
    assert "function formatAuditSpan" in AUDIO_SPAN_PLAYER_JS
    assert "formatAuditSpan(start,end)" in AUDIT_REVIEW_CORE_JS
    assert "start.toFixed(2)" not in AUDIO_SPAN_PLAYER_JS
    assert "safeStart.toFixed(2)" not in AUDIO_SPAN_PLAYER_JS


@pytest.mark.parametrize(
    ("relative_path", "required", "forbidden"),
    (
        (
            "tools/audits/generate_candidate_island_dual_evidence_review.py",
            "formatAuditSpan(start,end)",
            "start.toFixed(2)",
        ),
        (
            "tools/audits/generate_candidate_island_dual_evidence_review.py",
            "formatAuditSpan(gap.start_s,gap.end_s)",
            "Number(gap.start_s).toFixed(2)",
        ),
        (
            "tools/audits/generate_acoustic_split_canonical_candidate_audit_html.py",
            "formatAuditSpan(start,end)",
            "Number(start).toFixed(3)",
        ),
        (
            "tools/audits/generate_split_v4_missing_cut_candidate_audit_html.py",
            "formatAuditSpan(start,end)",
            "Number(start).toFixed(3)",
        ),
        (
            "tools/audits/generate_pre_asr_v13_false_drop_audit_html.py",
            "formatAuditSpan(row.start_s,row.end_s)",
            "Number(row.start_s).toFixed(3)",
        ),
    ),
)
def test_current_audit_adapters_use_core_timestamp_formatters(
    relative_path: str,
    required: str,
    forbidden: str,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / relative_path).read_text(encoding="utf-8")
    assert required in source
    assert forbidden not in source
