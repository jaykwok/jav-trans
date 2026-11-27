"""Regression tests for P0-3, P0-5, P1-6..P1-11, and P2 engine/repair/transport fixes."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
from types import SimpleNamespace

import httpx
import pytest

from core.cancellation import PipelineCancelledError
from llm import (
    async_transport,
    backends,
    cache as translation_cache,
    engine as engine_module,
    max_tokens_limits,
    preflight,
    repair,
    settings as llm_settings,
    translator,
    transport_util,
    zh_variant,
)
from llm.backends import openai_compat
from llm.context import SourceContext
from llm.errors import (
    BackendLeaseInvalidatedError,
    ContentPolicyRefusalError,
    MaxTokensRejectedError,
    RetryableTranslationFormatError,
    TranslationCancelledError,
)
from llm.profiles import get_profile, json_v3
from llm.session import TranslationSession


# ---------------------------------------------------------------------------
# P0-3 & P1-9: llamacpp repair cache key alignment + incremental rewrite + review preservation
# ---------------------------------------------------------------------------


def test_llamacpp_repair_cache_key_matches_engine_and_hits_on_second_run(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("LLAMACPP_MODEL_REPO", "test/repo")
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "test.gguf")
    monkeypatch.setenv("LLAMACPP_GGUF_PATH", "")
    monkeypatch.setattr(translator, "COMPACT_SYSTEM_PROMPT", False)
    monkeypatch.setattr(translator, "LLAMACPP_TRANSLATION_BATCH_SIZE", 2)
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_MAX_IDS", 50)

    cache_path = str(tmp_path / "local_repair.jsonl")
    segments = [
        {"text": "今日はいい天気ですね。", "start": 0.0, "end": 2.0},
        {"text": "明日も晴れそうです。", "start": 2.0, "end": 4.0},
    ]

    calls = {"count": 0}

    def _fake_chat(messages, **kwargs):
        del kwargs
        calls["count"] += 1
        user_msg = messages[-1]["content"]
        if "翻译修复任务" in user_msg:
            return json.dumps(
                {"translations": [{"id": 0, "text": "今天天气真好。"}]},
                ensure_ascii=False,
            )
        # First pass returns an echo on id 0, clean translation on id 1.
        return json.dumps(
            {
                "translations": [
                    {"id": 0, "text": "今日はいい天気ですね。"},
                    {"id": 1, "text": "明天好像也是晴天。"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", _fake_chat)

    first, _timings1, _ = translator.translate_segments(
        segments,
        cache_path=cache_path,
        target_lang="简体中文",
    )
    assert first == ["今天天气真好。", "明天好像也是晴天。"]
    assert calls["count"] == 2  # 1 batch + 1 repair

    # Second run must hit the repaired batch cache with ZERO LLM calls.
    second, timings2, _ = translator.translate_segments(
        segments,
        cache_path=cache_path,
        target_lang="简体中文",
    )
    assert second == ["今天天气真好。", "明天好像也是晴天。"]
    assert calls["count"] == 2
    assert timings2[-1]["cache_hit_count"] == 1


def test_repair_cache_write_is_incremental_and_preserves_untouched_reviews(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("LLAMACPP_MODEL_REPO", "test/repo")
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "test.gguf")
    monkeypatch.setenv("LLAMACPP_GGUF_PATH", "")
    monkeypatch.setattr(translator, "LLAMACPP_TRANSLATION_BATCH_SIZE", 2)
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_MAX_IDS", 50)

    cache_path = tmp_path / "incremental.jsonl"
    segments = [
        {"text": "今日はいい天気ですね。", "start": 0.0, "end": 1.0},  # batch 0, off 0 (repaired)
        {"text": "学校に行きます。", "start": 1.0, "end": 2.0},        # batch 0, off 1 (untouched, has review)
        {"text": "お元気ですか。", "start": 2.0, "end": 3.0},          # batch 1, off 0 (untouched batch, has review)
        {"text": "ありがとうございます。", "start": 3.0, "end": 4.0},   # batch 1, off 1 (untouched batch)
    ]

    # Pre-seed batch 0 and batch 1 cache entries with review records.
    profile = get_profile("json")
    source_context = SourceContext.build(segments, external_context="")
    model_identity = "llamacpp:test/repo/test.gguf"
    context_sig = engine_module.cache_scope(
        source_context, profile, model_identity, backend_name="llamacpp"
    )
    session = TranslationSession(
        backend_name="llamacpp",
        profile=profile,
        batch_size=2,
        max_workers=1,
        cache_path=str(cache_path),
        target_lang="简体中文",
        glossary="",
        character_reference="",
        reasoning_effort="",
    )
    with session.bound():
        key0 = translator._translation_cache_key(
            0,
            segments[0:2],
            target_lang="简体中文",
            compact_system_prompt=True,
            reasoning_effort="",
            prefix_mode="windowed_context",
            context_signature=context_sig,
        )
        key1 = translator._translation_cache_key(
            1,
            segments[2:4],
            target_lang="简体中文",
            compact_system_prompt=True,
            reasoning_effort="",
            prefix_mode="windowed_context",
            context_signature=context_sig,
        )

    lock = threading.Lock()
    translation_cache._save_cache_entry(
        cache_path,
        key0,
        ["今日はいい天気ですね。", "去学校。"],
        lock,
        review=[
            {"offset": 0, "reason": "constrained_retry"},
            {"offset": 1, "reason": "constrained_retry"},
        ],
    )
    translation_cache._save_cache_entry(
        cache_path,
        key1,
        ["你好吗？", "谢谢你。"],
        lock,
        review=[{"offset": 0, "reason": "constrained_retry"}],
    )

    def _fake_chat(messages, **_kw):
        assert "翻译修复任务" in messages[-1]["content"]
        return json.dumps(
            {"translations": [{"id": 0, "text": "今天天气真好。"}]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", _fake_chat)

    out, _, _ = translator.translate_segments(
        segments,
        cache_path=str(cache_path),
        target_lang="简体中文",
    )
    assert out == ["今天天气真好。", "去学校。", "你好吗？", "谢谢你。"]

    # Check raw JSONL lines: batch 1 must NOT have been appended a second time!
    lines = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [entry["key"] for entry in lines].count(key1) == 1
    assert [entry["key"] for entry in lines].count(key0) == 2

    # Check loaded reviews: batch 0 kept offset=1 review (dropped repaired offset=0);
    # batch 1 kept its offset=0 review untouched.
    reviews: dict[str, list] = {}
    loaded = translation_cache._load_translation_cache(cache_path, reviews)
    assert loaded[key0] == ["今天天气真好。", "去学校。"]
    assert reviews[key0] == [{"offset": 1, "reason": "constrained_retry"}]
    assert reviews[key1] == [{"offset": 0, "reason": "constrained_retry"}]

    # Check memory file: ONLY cue 0 (repaired) was written to memory, not cues 1/2/3.
    mem_path = translation_cache._translation_memory_jsonl_path(cache_path)
    mem_lines = [
        json.loads(line)
        for line in mem_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(mem_lines) == 1
    assert mem_lines[0]["value"] == "今天天气真好。"


# ---------------------------------------------------------------------------
# P0-5 & P1-9 & P2-6: repair.py variant conversion, persist before final_echoes,
# single-item no infinite recursion, dynamic request_cap, dynamic target_lang
# ---------------------------------------------------------------------------


def _repair_session(**overrides) -> TranslationSession:
    fields = {
        "backend_name": "openai",
        "profile": get_profile("json"),
        "batch_size": 1,
        "max_workers": 1,
        "cache_path": "",
        "target_lang": "简体中文",
        "glossary": "",
        "character_reference": "",
        "reasoning_effort": "low",
    }
    fields.update(overrides)
    return TranslationSession(**fields)


def test_repair_applies_zh_variant_and_preserves_echo_detection(monkeypatch):
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_MAX_IDS", 100)
    monkeypatch.setattr(llm_settings, "TRANSLATION_BATCH_MAX_REQUESTS", 20)
    # Stub converter: turns 繁->简 (and simulates OpenCC converting 発->发 on echo!)
    monkeypatch.setattr(
        zh_variant,
        "converter_for",
        lambda _lang: (lambda s: s.replace("發現", "发见").replace("天氣", "天气")),
    )

    segments = [
        {"text": "今日はいい天気ですね。", "start": 0.0, "end": 1.0},
        {"text": "大きな発見があった。", "start": 1.0, "end": 2.0},
    ]
    originals = ["今日はいい天気ですね。", "大きな発見があった。"]

    def _chat(_messages, **_kw):
        # Cue 0 fixed (in Traditional Chinese), cue 1 still raw Japanese source echo!
        return json.dumps(
            {
                "translations": [
                    {"id": 0, "text": "今天天氣真好。"},
                    {"id": 1, "text": "大きな発見があった。"},
                ]
            },
            ensure_ascii=False,
        )

    persisted: list[tuple[list[str], set[int]]] = []
    with pytest.raises(RuntimeError, match="still echoes the Japanese source"):
        repair.apply_repair_pass(
            segments,
            originals,
            chat=_chat,
            session=_repair_session(batch_size=2),
            cache_writer=lambda texts, ids: persisted.append((list(texts), set(ids))),
        )

    # P0-5a: Already-repaired cue 0 was converted to Simplified Chinese AND persisted BEFORE raising!
    assert len(persisted) == 1
    saved_texts, saved_ids = persisted[0]
    assert saved_texts[0] == "今天天气真好。"
    assert saved_ids == {0}


def test_repair_single_missing_cue_does_not_recurse_infinitely(monkeypatch):
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_MAX_IDS", 100)
    monkeypatch.setattr(llm_settings, "TRANSLATION_BATCH_MAX_REQUESTS", 20)

    segments = [{"text": "テストです。", "start": 0.0, "end": 1.0}]
    originals = ["测试です。"]  # kana remnant (not source echo)
    calls = {"count": 0}

    def _empty_reply(_messages, **_kw):
        calls["count"] += 1
        return json.dumps({"translations": []}, ensure_ascii=False)

    _out, timing = repair.apply_repair_pass(
        segments,
        originals,
        chat=_empty_reply,
        session=_repair_session(),
    )

    # Must NOT loop to request_cap (20)! At most 1 stage-1 + 1 stage-2 call.
    assert calls["count"] == 2
    assert timing["request_count"] == 2


def test_repair_dynamic_request_cap_handles_large_repair_sets(monkeypatch):
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_MAX_IDS", 100)
    # Intentionally tiny batch_max_requests (2) while repairing 30 cues (span=10 -> 3 chunks)
    monkeypatch.setattr(llm_settings, "TRANSLATION_BATCH_MAX_REQUESTS", 2)

    segments = [
        {"text": f"文{i}のテストです。", "start": float(i), "end": float(i + 1)}
        for i in range(30)
    ]
    originals = [f"句子{i}です。" for i in range(30)]

    def _fix_requested(messages, **_kw):
        import re

        user = messages[-1]["content"]
        ids = [int(m) for m in re.findall(r'"id":(\d+),"start":', user)]
        repair_ids = [
            item["id"]
            for item in json.loads(
                user.split("【局部上下文 JSON】\n\n")[1].split("\n\n【完整语义单元】")[0]
            )
            if item["role"] == "repair"
        ]
        del ids
        return json.dumps(
            {"translations": [{"id": idx, "text": f"已修复{idx}"} for idx in repair_ids]},
            ensure_ascii=False,
        )

    out, timing = repair.apply_repair_pass(
        segments,
        originals,
        chat=_fix_requested,
        session=_repair_session(batch_size=10),
    )

    assert all(text.startswith("已修复") for text in out)
    assert timing["request_count"] == 3


def test_repair_messages_dynamic_target_lang_compact_json_and_reused_context(monkeypatch):
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_CONTEXT_RADIUS", 1)
    segments = [
        {"text": f"台詞{i}", "start": float(i), "end": float(i + 1)}
        for i in range(6)
    ]
    zh_texts = [f"台词{i}" for i in range(6)]
    ctx = SourceContext.build(segments)

    def _forbid_rebuild(*_a, **_kw):
        raise AssertionError("SourceContext.build should not be called when source_context is supplied")

    monkeypatch.setattr(SourceContext, "build", classmethod(_forbid_rebuild))

    messages = repair._build_repair_messages(
        segments,
        zh_texts,
        [3],
        {3: ["japanese_remaining"]},
        target_lang="繁体中文",
        glossary="",
        character_reference="",
        source_context=ctx,
    )
    user_msg = messages[-1]["content"]
    assert "每个 text 只能是修复后的繁体中文字幕；不要解释原因。" in user_msg
    # Compact JSON (no `": "` spacing after keys)
    ctx_json_block = user_msg.split("【局部上下文 JSON】\n\n")[1].split("\n\n【完整语义单元】")[0]
    assert '": ' not in ctx_json_block
    items = json.loads(ctx_json_block)
    # Radius=1 around id=3 -> only {2, 3, 4}
    assert [it["id"] for it in items] == [2, 3, 4]
    assert repair._public_repair_reasons(["source_echo", "unknown_reason"]) == [
        "source_echo",
        "translation_quality",
    ]


# ---------------------------------------------------------------------------
# P1-6 & P1-7 & P1-10: engine narrowed no-sleep, cancel propagation, lease check
# ---------------------------------------------------------------------------


def test_engine_narrowed_partial_progress_skips_backoff_sleep():
    profile = get_profile("json")
    segments = [
        {"text": f"セリフ{i}", "start": float(i), "end": float(i + 1)}
        for i in range(4)
    ]
    sleeps: list[tuple[int, str]] = []
    calls = {"count": 0}

    def _chat(messages, **kwargs):
        del messages, kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            # First call fails format -> narrows span from 4 to 2
            raise RetryableTranslationFormatError("bad json")
        if calls["count"] == 2:
            # Narrowed half 1 succeeds -> returns ids 0, 1
            return json.dumps(
                {"translations": [{"id": 0, "text": "译0"}, {"id": 1, "text": "译1"}]},
                ensure_ascii=False,
            )
        # Narrowed half 2 succeeds -> returns ids 2, 3
        return json.dumps(
            {"translations": [{"id": 2, "text": "译2"}, {"id": 3, "text": "译3"}]},
            ensure_ascii=False,
        )

    out, _, _ = engine_module.run_batched(
        segments,
        profile=profile,
        backend_name="openai",
        chat=_chat,
        backoff_sleep=lambda attempt, exc, **_kw: sleeps.append((attempt, str(exc))),
        crash_probe=lambda: 0,
        batch_size=4,
        max_workers=1,
        api_retries=3,
        batch_repair_retries=3,
        batch_max_requests=10,
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
        model_identity="test-model",
        compact_system_prompt=False,
        reasoning_effort="none",
    )

    assert out == ["译0", "译1", "译2", "译3"]
    # Sleep happened ONLY after the format failure (call 1), NOT after narrowed partial progress (call 2)!
    assert len(sleeps) == 1
    assert "bad json" in sleeps[0][1]


def test_progress_callbacks_propagate_cancellation_errors():
    def _raise_translation_cancel(_payload):
        raise TranslationCancelledError("cancelled in callback")

    def _raise_pipeline_cancel(_payload):
        raise PipelineCancelledError("pipeline cancelled in callback")

    callbacks, _ = engine_module._make_aggregated_progress_callback(1, 2, _raise_translation_cancel)
    with pytest.raises(TranslationCancelledError):
        callbacks[0]({"phase": "translating", "translated": 1, "expected": 2})

    callbacks2, _ = engine_module._make_aggregated_progress_callback(1, 2, _raise_pipeline_cancel)
    with pytest.raises(PipelineCancelledError):
        callbacks2[0]({"phase": "translating", "translated": 1, "expected": 2})

    backend_inst = openai_compat.OpenAICompatBackend()
    with pytest.raises(TranslationCancelledError):
        backend_inst._emit_progress(_raise_translation_cancel, {"phase": "done"})
    with pytest.raises(PipelineCancelledError):
        backend_inst._emit_progress(_raise_pipeline_cancel, {"phase": "done"})


def test_translator_chat_enforces_openai_backend_lease(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setattr(
        translator,
        "task_backend",
        lambda: (_ for _ in ()).throw(BackendLeaseInvalidatedError("lease gone")),
    )
    with pytest.raises(BackendLeaseInvalidatedError, match="lease gone"):
        translator._chat([{"role": "user", "content": "hi"}], expected_count=1)


# ---------------------------------------------------------------------------
# P1-8 & P1-11: openai_compat usage on incomplete/failed, tightened regex,
# content_policy codes, error redaction
# ---------------------------------------------------------------------------


def test_openai_compat_emits_usage_on_incomplete_and_failed(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5")

    usage_obj = SimpleNamespace(
        input_tokens=500,
        output_tokens=200,
        total_tokens=700,
        input_tokens_details=SimpleNamespace(cached_tokens=100),
        output_tokens_details=SimpleNamespace(reasoning_tokens=50),
    )

    for event_type, expected_exc in (
        ("response.incomplete", Exception),
        ("response.failed", RetryableTranslationFormatError),
    ):

        async def _stream(et=event_type):
            yield SimpleNamespace(
                type=et,
                response=SimpleNamespace(
                    usage=usage_obj,
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    error=SimpleNamespace(code="server_error", message="boom"),
                ),
            )

        async def _create(stream_fn=_stream, **_kw):
            return stream_fn()

        fake_client = SimpleNamespace(
            responses=SimpleNamespace(create=_create),
            close=lambda: None,
        )
        monkeypatch.setattr(
            openai_compat, "_make_async_client", lambda _c=fake_client: _c
        )

        usages: list[dict] = []
        with pytest.raises(expected_exc):
            openai_compat._chat_responses(
                [{"role": "user", "content": "hi"}],
                expected_count=1,
                on_usage=usages.append,
                max_tokens=200,
            )
        assert len(usages) == 1
        assert usages[0]["prompt_tokens"] == 500
        assert usages[0]["completion_tokens"] == 200
        assert usages[0]["cached_tokens"] == 100
        assert usages[0]["reasoning_tokens"] == 50


def test_openai_compat_tightened_combined_budget_regex_and_policy_codes():
    # Standalone "Invalid input:" or "输入参数" MUST NOT misclassify output ceiling as request/combined!
    assert openai_compat._max_tokens_refusal_kind("Invalid input: max_tokens must be <= 131072") == "output"
    assert openai_compat._max_tokens_refusal_kind("输入参数 max_tokens 超出上限 65536") == "output"
    # Actual combined prompt+output budget errors MUST still classify as "request"!
    assert (
        openai_compat._max_tokens_refusal_kind(
            "This model's maximum context length is 128000 tokens. However, you requested 130000 tokens (input + output)."
        )
        == "request"
    )
    assert (
        openai_compat._max_tokens_refusal_kind(
            "max_tokens + input_tokens exceeds context window"
        )
        == "request"
    )

    # Verify policy codes
    for code in ("content_policy_violation", "content_filter"):
        err = openai_compat._provider_error_reason(code, "blocked", 4096)
        assert isinstance(err, ContentPolicyRefusalError)

    # Verify learnable=True for tightened output refusal
    err_out = openai_compat._provider_error_reason(
        "invalid_request_error",
        "Invalid input: max_tokens must be <= 32768",
        65536,
    )
    assert isinstance(err_out, MaxTokensRejectedError)
    assert err_out.learnable is True
    assert err_out.limit == 32768


def test_error_sanitization_redacts_url_credentials_and_bearer_tokens():
    raw = (
        "Failed connecting via http://alice:super-secret@proxy.local:8080 "
        "with Authorization: Bearer sk-live-secret-99999"
    )
    cleaned = transport_util._sanitize_error_text(raw)
    assert "super-secret" not in cleaned
    assert "sk-live-secret-99999" not in cleaned
    assert "Bearer ***" in cleaned


# ---------------------------------------------------------------------------
# P1-10 & P2: async_transport loop reuse & close_quietly, transport_util backoff & markers
# ---------------------------------------------------------------------------


def test_async_transport_reuses_thread_loop_and_close_quietly_handles_aclose():
    loops_seen: list[int] = []

    async def _op():
        loops_seen.append(id(asyncio.get_running_loop()))
        return "ok"

    assert async_transport.run_cancellable_request(_op, cancel_event=None) == "ok"
    assert async_transport.run_cancellable_request(_op, cancel_event=None) == "ok"
    assert loops_seen[0] == loops_seen[1]

    # Different thread gets its own loop
    other_loop_ids: list[int] = []
    t = threading.Thread(
        target=lambda: other_loop_ids.append(
            async_transport.run_cancellable_request(
                lambda: asyncio.sleep(0, result=id(asyncio.get_running_loop())),
                cancel_event=None,
            )
        )
    )
    t.start()
    t.join(timeout=5.0)
    assert other_loop_ids and other_loop_ids[0] != loops_seen[0]

    # close_quietly supports aclose() and async close() and bounds stuck closers
    closed_flags: list[str] = []

    class _HasAclose:
        async def aclose(self):
            closed_flags.append("aclose")

    class _HasAsyncClose:
        async def close(self):
            closed_flags.append("async_close")

    class _StuckCloser:
        async def aclose(self):
            await asyncio.sleep(10.0)

    async def _test_close():
        await async_transport.close_quietly(_HasAclose())
        await async_transport.close_quietly(_HasAsyncClose())
        t0 = time.monotonic()
        await async_transport.close_quietly(_StuckCloser(), timeout_s=0.05)
        assert time.monotonic() - t0 < 1.0

    async_transport.run_cancellable_request(_test_close, cancel_event=None)
    assert closed_flags == ["aclose", "async_close"]


def test_transport_util_retry_after_jitter_and_marker_counting():
    req = httpx.Request("POST", "https://api.example.test/v1/responses")
    # Numeric Retry-After
    exc_num = httpx.HTTPStatusError(
        "429",
        request=req,
        response=httpx.Response(429, headers={"Retry-After": "7"}, request=req),
    )
    assert transport_util._request_backoff_delay(1, exc_num, jitter=False) == 7.0

    # HTTP-date Retry-After
    future_http_date = formatdate(time.time() + 10, usegmt=True)
    exc_date = httpx.HTTPStatusError(
        "429",
        request=req,
        response=httpx.Response(429, headers={"Retry-After": future_http_date}, request=req),
    )
    delay_date = transport_util._request_backoff_delay(1, exc_date, jitter=False)
    assert 5.0 <= delay_date <= 15.0

    # Jitter stays within [0.5 * base, 1.0 * base]
    det = transport_util._request_backoff_delay(2, jitter=False)
    samples = [transport_util._request_backoff_delay(2, jitter=True) for _ in range(30)]
    assert all(0.5 * det <= s <= det for s in samples)

    # Marker counter across chunk boundaries
    count, tail = transport_util._count_translation_markers(
        piece='{"id":0,"id":1,"i', id_scan_tail="", id_marker='"id":'
    )
    assert count == 2
    assert tail == '1,"i'
    count2, _ = transport_util._count_translation_markers(
        piece='d":2}', id_scan_tail=tail, id_marker='"id":'
    )
    assert count2 == 1


# ---------------------------------------------------------------------------
# P2 small modules: json_v3, preflight, zh_variant, max_tokens_limits, backends
# ---------------------------------------------------------------------------


def test_json_v3_coerce_int_quotes_and_live_reasoning_effort(monkeypatch, tmp_path):
    assert json_v3._coerce_int(True) is None
    assert json_v3._coerce_int(False) is None
    assert json_v3._coerce_int(3) == 3
    assert json_v3._coerce_int("42") == 42

    # Paired outer quotes stripped; inner quote + continuation preserved
    assert json_v3._normalize_translation_text("“你好”") == "你好"
    assert json_v3._normalize_translation_text("“你好”，他说") == "“你好”，他说"
    assert json_v3._normalize_translation_text('"hello"') == "hello"
    assert json_v3._normalize_translation_text('"hello", he said') == '"hello", he said'

    # Live LLM_REASONING_EFFORT read in json_v3 and preflight
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    assert json_v3._reasoning_token_allowance(None) == 0
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    assert json_v3._reasoning_token_allowance(None) > 0

    store = tmp_path / "limits.json"
    monkeypatch.setattr(max_tokens_limits, "limits_path", lambda: store)
    monkeypatch.setattr(max_tokens_limits, "_CACHE_KEY", None)
    monkeypatch.setattr(max_tokens_limits, "_CACHE_PAYLOAD", None)
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.example.test/v1")
    max_tokens_limits.record_exact_ceiling("https://api.example.test/v1", "gpt-5", 16000)
    monkeypatch.setattr(llm_settings, "TRANSLATION_BATCH_SIZE", 200)
    monkeypatch.setattr(llm_settings, "TRANSLATION_REASONING_TOKEN_ALLOWANCE", 32000)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    warnings_none = preflight.translation_budget_warnings()
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    warnings_high = preflight.translation_budget_warnings()
    assert warnings_none == []
    assert len(warnings_high) == 1
    assert "推理档=high" in warnings_high[0]


def test_zh_variant_converter_thread_safe():
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: zh_variant.converter_for("简体中文"), range(16)))
    assert len({id(fn) for fn in results}) == 1


def test_max_tokens_limits_mtime_cache_and_tmp_cleanup(monkeypatch, tmp_path):
    store = tmp_path / "limits.json"
    monkeypatch.setattr(max_tokens_limits, "limits_path", lambda: store)
    monkeypatch.setattr(max_tokens_limits, "_CACHE_KEY", None)
    monkeypatch.setattr(max_tokens_limits, "_CACHE_PAYLOAD", None)

    base_url, model = "https://api.example.test/v1", "model-x"
    max_tokens_limits.record_exact_ceiling(base_url, model, 32768)
    assert max_tokens_limits.load_limits(base_url, model).exact_ceiling == 32768

    # Second load hits mtime cache without re-reading disk
    orig_read_text = max_tokens_limits.Path.read_text
    read_calls = {"count": 0}

    def _counting_read(self, *args, **kwargs):
        read_calls["count"] += 1
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(max_tokens_limits.Path, "read_text", _counting_read)
    assert max_tokens_limits.load_limits(base_url, model).exact_ceiling == 32768
    assert read_calls["count"] == 0

    # Simulate Path.replace failure -> .tmp file must be cleaned up!
    monkeypatch.setattr(
        max_tokens_limits.Path,
        "replace",
        lambda _self, _dst: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        max_tokens_limits._write_payload({"schema": max_tokens_limits.LIMITS_SCHEMA, "limits": {}})
    assert not list(tmp_path.glob("*.tmp"))


def test_backends_perform_close_has_no_none_visibility_window_when_unconfirmed():
    class _StubbornBackend:
        def close(self) -> bool:
            return False

    name = "_test_stubborn_backend"
    inst = _StubbornBackend()
    op = backends._CloseOperation(instance=inst)
    try:
        with backends._REGISTRY_LOCK:
            backends._BACKEND_INSTANCES.pop(name, None)
            backends._CLOSING[name] = inst
            backends._CLOSE_OPERATIONS[name] = op

        seen_none = False
        stop_reader = threading.Event()

        def _reader():
            nonlocal seen_none
            while not stop_reader.is_set():
                if backends._visible_instance(name) is None:
                    seen_none = True
                    return

        t = threading.Thread(target=_reader)
        t.start()
        try:
            backends._perform_close(name, op)
            assert op.confirmed is False
        finally:
            stop_reader.set()
            t.join(timeout=2.0)

        assert not seen_none
        assert backends._visible_instance(name) is inst
    finally:
        with backends._REGISTRY_LOCK:
            backends._BACKEND_INSTANCES.pop(name, None)
            backends._CLOSING.pop(name, None)
            backends._CLOSE_OPERATIONS.pop(name, None)
