from __future__ import annotations

import pytest

from tools.asr.cueqc import label_semantic_pre_asr_with_omni as labeler


def test_omni_retry_only_retries_rate_limits(monkeypatch):
    calls = 0

    def fake_call(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("HTTP 429 limit_requests")
        return {"decision": "keep"}, "raw"

    delays: list[int] = []
    monkeypatch.setattr(labeler, "call_omni", fake_call)
    monkeypatch.setattr(labeler.time, "sleep", delays.append)

    assert labeler._call_omni_with_rate_limit_retry() == (
        {"decision": "keep"},
        "raw",
    )
    assert calls == 2
    assert delays == [5]

    monkeypatch.setattr(
        labeler,
        "call_omni",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bad response")),
    )
    with pytest.raises(RuntimeError, match="bad response"):
        labeler._call_omni_with_rate_limit_retry()
