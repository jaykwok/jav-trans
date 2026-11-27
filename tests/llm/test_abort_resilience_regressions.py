"""Regression tests for catastrophic-abort resilience across translation engine,
local llama-server, OpenAI Responses pre-stream retries, and lease admission waits."""

from __future__ import annotations

import json
import logging
import re
import threading
from types import SimpleNamespace

import httpx
import pytest

from llm import backends, cache as translation_cache, engine as engine_module
from llm.backends import llamacpp_server, openai_compat
from llm.errors import (
    BackendUnavailableError,
    ResponseTruncatedError,
    RetryableTranslationFormatError,
)
from llm.profiles import get_profile


def _segments(count: int, *, prefix: str = "台詞") -> list[dict]:
    return [
        {"start": float(i), "end": float(i + 1), "text": f"{prefix}{i}です。"}
        for i in range(count)
    ]


def _requested_ids(messages: list[dict]) -> list[int]:
    content = messages[-1]["content"]
    match = re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", content)
    assert match is not None, content
    return json.loads(match.group(1))


def _run_batched_common(segments, chat, **overrides):
    profile = get_profile("json")
    kwargs = dict(
        profile=profile,
        backend_name="llamacpp",
        chat=chat,
        backoff_sleep=lambda *_a, **_kw: None,
        crash_probe=lambda: 0,
        batch_size=len(segments),
        max_workers=1,
        api_retries=4,
        batch_repair_retries=2,
        batch_max_requests=24,
        prefix_warmup=False,
        extra_glossary="",
        full_context="",
        full_source_payload="",
        use_full_json_prefix=False,
        cache_path="",
        cache_lock=threading.Lock(),
        target_lang="简体中文",
        glossary="",
        character_reference="",
        prompt_version=profile.cache_signature(),
        model_identity="llamacpp:test",
        compact_system_prompt=True,
        reasoning_effort="",
        focus_radius=4,
        focus_context_chars=4000,
    )
    kwargs.update(overrides)
    return engine_module.run_batched(segments, **kwargs)


# ---------------------------------------------------------------------------
# #1: Single-cue empty translation degrades to warning + blank instead of aborting
# ---------------------------------------------------------------------------


def test_engine_blanks_single_empty_cue_after_parsed_retries_and_preserves_batch(
    tmp_path, caplog
):
    segments = _segments(4)
    cache_path = str(tmp_path / "empty_cue.jsonl")
    seen_ids: list[list[int]] = []

    def _chat(messages, **_kw):
        ids = _requested_ids(messages)
        seen_ids.append(ids)
        return json.dumps(
            {
                "translations": [
                    {"id": idx, "text": "" if idx == 2 else f"译文{idx}"}
                    for idx in ids
                ]
            },
            ensure_ascii=False,
        )

    with caplog.at_level(logging.WARNING, logger="llm.engine"):
        texts, timings, _ = _run_batched_common(
            segments, _chat, cache_path=cache_path
        )

    assert texts == ["译文0", "译文1", "", "译文3"]
    # Initial full batch + 2 single-cue repair attempts for cue 2, then blanked.
    assert seen_ids == [[0, 1, 2, 3], [2], [2]]
    batch_timing = next(t for t in timings if t.get("mode") == "batched_full_context")
    assert batch_timing["empty_translation_count"] == 1
    assert batch_timing["empty_translation_ids"] == [2]
    assert any("blanking cue" in rec.message for rec in caplog.records)
    # Exact batch cache entry was NOT poisoned with "", while memory saved the 3 good cues.
    assert translation_cache._load_translation_cache(cache_path) == {}
    mem = translation_cache._load_translation_memory(cache_path)
    assert sorted(mem.values()) == ["译文0", "译文1", "译文3"]


def test_engine_narrows_multiple_empty_cues_from_actual_pending_length(caplog):
    segments = _segments(6)
    seen_ids: list[list[int]] = []

    def _chat(messages, **_kw):
        ids = _requested_ids(messages)
        seen_ids.append(ids)
        return json.dumps(
            {
                "translations": [
                    {"id": idx, "text": "" if idx in {1, 4} else f"译文{idx}"}
                    for idx in ids
                ]
            },
            ensure_ascii=False,
        )

    with caplog.at_level(logging.WARNING, logger="llm.engine"):
        texts, timings, _ = _run_batched_common(
            segments, _chat, batch_size=6, batch_repair_retries=2
        )

    assert texts == ["译文0", "", "译文2", "译文3", "", "译文5"]
    # Pending [1, 4] (len 2) narrows to [1] instead of staying stuck at stale span=6 -> 3.
    assert [1] in seen_ids and [4] in seen_ids
    batch_timing = next(t for t in timings if t.get("mode") == "batched_full_context")
    assert batch_timing["empty_translation_ids"] == [1, 4]


def test_engine_still_fails_when_single_cue_never_parses_valid_json():
    segments = _segments(1)

    def _broken(_messages, **_kw):
        return "not valid json"

    with pytest.raises(RuntimeError, match="no usable reply"):
        _run_batched_common(segments, _broken, api_retries=2)


# ---------------------------------------------------------------------------
# #2: ResponseTruncatedError narrows multi-cue batches instead of aborting film
# ---------------------------------------------------------------------------


def test_engine_narrows_batch_on_response_truncated_error():
    segments = _segments(8)
    seen_sizes: list[int] = []

    def _chat(messages, **_kw):
        ids = _requested_ids(messages)
        seen_sizes.append(len(ids))
        if len(ids) > 2:
            raise ResponseTruncatedError("cut off at context limit", limit=1024)
        return json.dumps(
            {"translations": [{"id": idx, "text": f"译文{idx}"} for idx in ids]},
            ensure_ascii=False,
        )

    texts, _, _ = _run_batched_common(segments, _chat, batch_size=8, api_retries=2)
    assert texts == [f"译文{i}" for i in range(8)]
    assert seen_sizes == [8, 4, 2, 2, 2, 2]


def test_engine_propagates_response_truncated_error_on_single_cue_without_context():
    segments = _segments(1)

    def _always_trunc(_messages, **_kw):
        raise ResponseTruncatedError("single cue runaway", limit=512)

    with pytest.raises(ResponseTruncatedError, match="single cue runaway"):
        _run_batched_common(
            segments,
            _always_trunc,
            batch_size=1,
            focus_radius=0,
            focus_context_chars=0,
        )


# ---------------------------------------------------------------------------
# #4: Context-limit HTTP 400 descends to span=1 and sheds neighbour context
# ---------------------------------------------------------------------------


def test_engine_context_limit_descends_past_api_retries_and_drops_focus_context():
    segments = _segments(32, prefix="長い日本語の台詞")
    observed: list[tuple[int, int]] = []

    def _chat(messages, **_kw):
        ids = _requested_ids(messages)
        user_msg = messages[-1]["content"]
        context_only_count = user_msg.count('"role":"context_only"')
        observed.append((len(ids), context_only_count))
        # Fail until both span reaches 1 AND neighbour context is stripped to 0.
        if len(ids) > 1 or context_only_count > 0:
            raise RetryableTranslationFormatError(
                "llama-server HTTP 400: request exceeds the available context size"
            )
        return json.dumps(
            {"translations": [{"id": idx, "text": f"译文{idx}"} for idx in ids]},
            ensure_ascii=False,
        )

    texts, _, _ = _run_batched_common(
        segments,
        _chat,
        batch_size=32,
        api_retries=4,
        batch_max_requests=48,
        focus_radius=4,
        focus_context_chars=4000,
    )

    assert texts == [f"译文{i}" for i in range(32)]
    # Descended 32 -> 16 -> 8 -> 4 -> 2 -> 1 (past api_retries=4!) and shed context_only to 0.
    assert [size for size, _ in observed[:6]] == [32, 16, 8, 4, 2, 1]
    assert observed[5][1] == 0


# ---------------------------------------------------------------------------
# #3: Local llama-server empty choices & alive-process transient read errors
# ---------------------------------------------------------------------------


class _AliveProc:
    pid = 99001

    def poll(self):
        return None

    def wait(self, timeout=None):
        import subprocess

        raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout or 0.1)


def test_llamacpp_empty_choices_raises_retryable_format_error(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    backend._proc = _AliveProc()
    backend._port = 18099
    monkeypatch.setattr(backend, "_ensure_server", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        backend,
        "_request_completion",
        lambda *_a, **_kw: SimpleNamespace(choices=[], usage=None),
    )

    with pytest.raises(RetryableTranslationFormatError, match="空的 choices"):
        backend.chat_completion([{"role": "user", "content": "hi"}])


def test_llamacpp_alive_process_transient_protocol_error_is_retryable(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    backend._proc = _AliveProc()
    backend._port = 18098

    class _DropCompletions:
        async def create(self, **_kw):
            raise httpx.RemoteProtocolError("peer closed connection")

    monkeypatch.setattr(
        backend,
        "_make_async_client",
        lambda _p: SimpleNamespace(
            chat=SimpleNamespace(completions=_DropCompletions()),
            close=lambda: None,
        ),
    )

    with pytest.raises(RetryableTranslationFormatError, match="请求失败"):
        backend._request_completion({"model": "m"}, cancel_event=None)


# ---------------------------------------------------------------------------
# #5: OpenAI Responses pre-stream exhausted retryable errors wrap for engine retry
# ---------------------------------------------------------------------------


def test_openai_compat_prestream_retryable_error_wraps_as_retryable_format_error(
    monkeypatch,
):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5")
    monkeypatch.setattr(openai_compat, "_make_async_client", lambda: SimpleNamespace(close=lambda: None))

    async def _fail_timeout(_req, cancel_event=None):
        del cancel_event
        raise httpx.ReadTimeout("upstream 504 timeout")

    monkeypatch.setattr(openai_compat, "_call_create_response", _fail_timeout)

    with pytest.raises(RetryableTranslationFormatError, match="ReadTimeout"):
        openai_compat._chat_responses([{"role": "user", "content": "hi"}], expected_count=1)


def test_openai_compat_prestream_non_retryable_error_is_not_wrapped(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5")
    monkeypatch.setattr(openai_compat, "_make_async_client", lambda: SimpleNamespace(close=lambda: None))

    req = httpx.Request("POST", "https://api.example.test/v1/responses")
    auth_err = httpx.HTTPStatusError(
        "unauthorized", request=req, response=httpx.Response(401, request=req)
    )

    async def _fail_auth(_req, cancel_event=None):
        del cancel_event
        raise auth_err

    monkeypatch.setattr(openai_compat, "_call_create_response", _fail_auth)

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        openai_compat._chat_responses([{"role": "user", "content": "hi"}], expected_count=1)
    assert not isinstance(excinfo.value, RetryableTranslationFormatError)


# ---------------------------------------------------------------------------
# #6: Healthy retiring holder wait exceeds default 600s without false failure
# ---------------------------------------------------------------------------


def test_retiring_holder_wait_does_not_fail_at_600s_when_no_explicit_timeout(
    monkeypatch,
):
    monkeypatch.delenv("LOCAL_BACKEND_WAIT_TIMEOUT_S", raising=False)

    class _RetiringStub:
        def __init__(self):
            self.retiring = True

        def invalidate(self):
            pass

        def is_invalidated(self):
            return False

        def cleanup_status(self):
            return {"stuck": False}

        def generation_retiring(self):
            return self.retiring

        def close(self):
            return True

    old_inst = _RetiringStub()
    new_inst = _RetiringStub()
    new_inst.retiring = False
    monkeypatch.setattr(backends, "_BACKEND_INSTANCES", {"retire-long-test": old_inst})
    monkeypatch.setattr(backends, "_BACKEND_LEASES", {"retire-long-test": [old_inst]})
    monkeypatch.setattr(backends, "_CLOSING", {})
    monkeypatch.setattr(backends, "_CLOSE_OPERATIONS", {})
    monkeypatch.setitem(backends._BACKEND_REGISTRY, "retire-long-test", lambda: new_inst)

    now = {"t": 0.0}

    def _fake_monotonic():
        return now["t"]

    def _fake_wait(_cancel, _timeout=0.25):
        # Advance clock past the 600s cleanup timeout while sibling holds lease,
        # then release sibling lease at t=900s (15 min) so retirement finishes.
        now["t"] += 350.0
        if now["t"] >= 700.0 and backends._BACKEND_LEASES.get("retire-long-test"):
            backends._BACKEND_LEASES["retire-long-test"].clear()

    monkeypatch.setattr(backends.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(backends, "_wait_briefly", _fake_wait)

    with backends.backend_lease("retire-long-test", threading.Event()) as leased:
        assert leased is new_inst
    assert now["t"] >= 700.0


def test_stuck_cleanup_without_holder_still_times_out_at_default_600s(monkeypatch):
    monkeypatch.delenv("LOCAL_BACKEND_WAIT_TIMEOUT_S", raising=False)

    class _StuckStub:
        def invalidate(self):
            pass

        def is_invalidated(self):
            return True

        def cleanup_status(self):
            return {"stuck": True}

        def generation_retiring(self):
            return False

        def close(self):
            return False

    stuck = _StuckStub()
    monkeypatch.setattr(backends, "_BACKEND_INSTANCES", {"stuck-600-test": stuck})
    monkeypatch.setattr(backends, "_BACKEND_LEASES", {})
    monkeypatch.setattr(backends, "_CLOSING", {})

    now = {"t": 0.0}
    monkeypatch.setattr(backends.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(
        backends, "_wait_briefly", lambda _c, _s=0.25: now.__setitem__("t", now["t"] + 350.0)
    )

    with pytest.raises(BackendUnavailableError, match="600s"):
        with backends.backend_lease("stuck-600-test", threading.Event()):
            pytest.fail("must not claim stuck instance")
