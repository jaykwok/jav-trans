from __future__ import annotations

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


def test_audit_core_clamps_exact_span_end_with_high_frequency_fallbacks() -> None:
    assert "audio.addEventListener('ended',stopFn)" in AUDIO_SPAN_PLAYER_JS
    assert "requestAnimationFrame(watch)" in AUDIO_SPAN_PLAYER_JS
    assert "setTimeout(finishAtEnd,remainingMilliseconds)" in AUDIO_SPAN_PLAYER_JS
    assert "stop(safeEnd)" in AUDIO_SPAN_PLAYER_JS
    assert "activeAudio.currentTime=finalTime" in AUDIO_SPAN_PLAYER_JS


def test_audit_core_downloads_labels_when_opened_as_a_local_file() -> None:
    assert "location.protocol==='file:'" in AUDIT_REVIEW_CORE_JS
    assert "new Blob([content]" in AUDIT_REVIEW_CORE_JS
    assert "link.download=filename" in AUDIT_REVIEW_CORE_JS
    assert "link.click()" in AUDIT_REVIEW_CORE_JS
