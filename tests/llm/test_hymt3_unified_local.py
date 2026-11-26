"""Tests for the unified Hy-MT3-7B local + API JSON batch translation path."""

from __future__ import annotations

import json
import re

import httpx
from openai import AsyncOpenAI

from llm import backends, preflight, repair, settings as llm_settings, translator
from llm.backends.llamacpp_server import LlamaCppServerBackend
from llm.context import SourceContext, cue_item
from llm.profiles import get_profile, select_profile
from llm.profiles.base import ProfileContext


def _segments(count: int) -> list[dict]:
    return [
        {"start": float(i * 2), "end": float(i * 2 + 1.2), "text": f"セリフ{i}です。"}
        for i in range(count)
    ]


def test_lean_source_json_omits_timings_while_preserving_context_breaks():
    cues = [
        {"start": 0.0, "end": 1.0, "text": "前の一言"},
        {"start": 2.0, "end": 3.0, "text": "対象", "continues_into_next": True},
        {"start": 3.0, "end": 4.0, "text": "続き", "continues_from_previous": True},
        {"start": 30.0, "end": 31.0, "text": "別シーン"},
    ]
    assert set(cue_item(cues[0], 0)) == {"id", "ja"}
    assert cue_item(cues[1], 1) == {"id": 1, "ja": "対象", "cont_next": True}
    assert cue_item(cues[2], 2) == {"id": 2, "ja": "続き", "cont_prev": True}

    ctx = SourceContext.build(cues)
    focus = ctx.focus([1], radius=2)
    by_id = {row["id"]: row for row in focus["items"]}
    # Cue 3 is 26s after cue 2 (>12s break), so timing-based scene break still works
    # even though start/end/duration_sec are no longer emitted into prompt rows.
    assert set(by_id) == {0, 1, 2}
    for row in focus["items"]:
        assert "start" not in row
        assert "end" not in row
        assert "duration_sec" not in row
    assert focus["semantic_units"] == [{"ids": [1, 2], "ja": "対象続き"}]


def test_windowed_batch_messages_send_focus_once_without_full_prefix_or_summary():
    cues = _segments(10)
    context = SourceContext.build(cues)
    profile = get_profile("json")
    messages = profile.build_messages(
        cues[4:6],
        ids=[4, 5],
        ctx=ProfileContext(
            global_context="",
            full_source_payload=None,
            total_count=len(cues),
            compact_system_prompt=True,
            batch_index=0,
            source_context=context,
        ),
    )
    system_text, user_text = messages[0]["content"], messages[1]["content"]
    assert "全片字幕概览" not in system_text
    assert "风格示例" not in system_text  # compact system prompt used from batch 0
    assert "【全片字幕 JSON】" not in user_text
    assert "【当前批次字幕 JSON】" not in user_text
    assert "【聚焦上下文" in user_text
    assert "duration_sec" not in user_text
    assert '"start"' not in user_text
    # Each cue appears once in focus JSON (no duplicate batch JSON).
    assert user_text.count("セリフ4です。") == 1
    assert "セリフ2です。" in user_text and "セリフ7です。" in user_text
    assert "セリフ0です。" not in user_text


def test_llamacpp_runs_unified_json_profile_with_small_batches_and_hymt3_sampling(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("LLAMACPP_MODEL_REPO", "tencent/Hy-MT3-7B-GGUF")
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "Hy-MT3-7B-Q4_K_M.gguf")
    monkeypatch.setenv("LLAMACPP_PARALLEL", "2")
    monkeypatch.setattr(translator, "LLAMACPP_TRANSLATION_BATCH_SIZE", 16)
    monkeypatch.setattr(translator, "TRANSLATION_BATCH_SIZE", 200)
    monkeypatch.setattr(llm_settings, "TRANSLATION_REASONING_TOKEN_ALLOWANCE", 32000)

    seen_requests: list[dict] = []

    def handler(request):
        body = json.loads(request.content)
        seen_requests.append(body)
        user_content = body["messages"][-1]["content"]
        match = re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", user_content)
        assert match is not None, user_content
        ids = json.loads(match.group(1))
        reply = json.dumps(
            {"translations": [{"id": idx, "text": f"台词{idx}"} for idx in ids]},
            ensure_ascii=False,
        )
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "object": "chat.completion",
                "created": 0,
                "model": "Hy-MT3-7B-Q4_K_M.gguf",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    backend = LlamaCppServerBackend()
    monkeypatch.setattr(backend, "_ensure_server", lambda *_a: None)
    monkeypatch.setattr(
        backend,
        "_make_async_client",
        lambda _port: AsyncOpenAI(
            api_key="test-only",
            base_url="http://localhost.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )
    monkeypatch.setitem(backends._BACKEND_REGISTRY, "llamacpp", lambda: backend)

    assert select_profile(backend="llamacpp").id == "json"

    cues = _segments(20)
    with backends.backend_lease("llamacpp"):
        texts, timings, _ = translator.translate_segments(
            cues,
            max_workers=4,
            target_lang="简体中文",
            glossary="ちんぽ-肉棒",
        )

    assert texts == [f"台词{i}" for i in range(20)]
    # 20 cues at local batch_size=16 -> 2 batches (not 1 batch of 200, not 20 single cues).
    assert len(seen_requests) == 2
    for req in seen_requests:
        # Official Hy-MT3-7B sampling reaches llama-server wire payload.
        assert req["temperature"] == 0.7
        assert req["top_p"] == 0.6
        assert req["top_k"] == 20
        assert req["repeat_penalty"] == 1.05
        # Bounded JSON schema is sent; no 32k reasoning allowance leaked into max_tokens.
        assert req["response_format"]["type"] == "json_schema"
        assert req["max_tokens"] < 4096
        # Windowed lean JSON only: no full prefix, no timestamps, glossary reaches prompt.
        system_msg, user_msg = req["messages"][0]["content"], req["messages"][1]["content"]
        assert "ちんぽ-肉棒" in system_msg
        assert "全片字幕概览" not in system_msg
        assert "【全片字幕 JSON】" not in user_msg
        assert "duration_sec" not in user_msg

    batch_modes = [t.get("prefix_mode") for t in timings if t.get("prefix_mode")]
    assert batch_modes and all(mode == "windowed_context" for mode in batch_modes)


def test_repair_catches_halfwidth_kana_and_ignores_middle_dot_and_prolonged_mark():
    cues = [
        {"text": "こんにちは"},
        {"text": "メアリー・スミス"},
        {"text": "待ってー"},
    ]
    ids, reasons = repair._select_translation_repair_ids(
        cues,
        ["ｺﾝﾆﾁﾊ", "玛丽・史密斯", "等一下ー"],
    )
    assert ids == [0]
    assert "japanese_remaining" in reasons[0]


def test_preflight_budget_warning_sizes_llamacpp_without_reasoning_allowance(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("TRANSLATION_MAX_TOKENS_CACHE_PATH", str(tmp_path / "limits.json"))
    monkeypatch.setattr(llm_settings, "TRANSLATION_REASONING_TOKEN_ALLOWANCE", 32000)
    monkeypatch.setattr(llm_settings, "LLM_REASONING_EFFORT", "high")
    # Even with a small 4096 ceiling, 16-cue local budget needs ~480 tokens, so no false warning.
    from llm import token_budget

    monkeypatch.setattr(token_budget, "CONFIGURED_MAX_TOKENS", 4096)
    token_budget._clamp_warned.clear()
    assert preflight.translation_budget_warnings() == []


def test_local_repair_guards_echo_regression_and_forbids_kana_on_stage2(monkeypatch):
    from llm.session import TranslationSession

    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_MAX_IDS", 100)
    monkeypatch.setattr(llm_settings, "TRANSLATION_BATCH_MAX_REQUESTS", 10)

    cues = [{"start": 0.0, "end": 2.0, "text": "らまご、まごつうきもひゅるしゅ..."}]
    # First-pass output had residual katakana (NOT a full source echo).
    initial = ["拉马戈，马戈茨乌基莫ヒュルシュ..."]
    session = TranslationSession(
        backend_name="llamacpp",
        profile=get_profile("json"),
        batch_size=16,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
        character_reference="",
        reasoning_effort="",
    )

    schemas_seen: list[dict | None] = []

    def fake_chat(_messages, **kwargs):
        schema = kwargs.get("bounded_response_schema")
        schemas_seen.append(schema)
        if len(schemas_seen) == 1:
            # Stage 1 unconstrained repair regresses into echoing the raw Japanese source.
            return json.dumps(
                {"translations": [{"id": 0, "text": "らまご、まごつうきもひゅるしゅ..."}]},
                ensure_ascii=False,
            )
        # Stage 2 receives kana-forbidding pattern in bounded_response_schema and resolves cleanly.
        return json.dumps(
            {"translations": [{"id": 0, "text": "拉玛戈，马戈茨乌基莫休鲁舒..."}]},
            ensure_ascii=False,
        )

    repaired, timing = repair.apply_repair_pass(
        cues,
        initial,
        chat=fake_chat,
        session=session,
    )

    assert len(schemas_seen) == 2
    stage1_text = schemas_seen[0]["properties"]["translations"]["items"]["properties"]["text"]
    stage2_text = schemas_seen[1]["properties"]["translations"]["items"]["properties"]["text"]
    assert "pattern" not in stage1_text
    assert "pattern" in stage2_text
    assert repaired == ["拉玛戈，马戈茨乌基莫休鲁舒..."]
    assert timing is not None
    assert timing["mode"] == "translation_repair_pass"


def test_repair_echo_regression_guard_keeps_prior_non_echo_if_stage2_also_echoes(monkeypatch):
    from llm.session import TranslationSession

    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_MAX_IDS", 100)
    monkeypatch.setattr(llm_settings, "TRANSLATION_BATCH_MAX_REQUESTS", 10)

    cues = [{"start": 0.0, "end": 2.0, "text": "らまご、まごつうきもひゅるしゅ..."}]
    initial = ["拉马戈，马戈茨乌基莫ヒュルシュ..."]
    session = TranslationSession(
        backend_name="openai",
        profile=get_profile("json"),
        batch_size=16,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
        character_reference="",
        reasoning_effort="low",
    )

    schemas_seen: list[dict | None] = []

    def always_echo(_messages, **kwargs):
        schemas_seen.append(kwargs.get("bounded_response_schema"))
        return json.dumps(
            {"translations": [{"id": 0, "text": "らまご、まごつうきもひゅるしゅ..."}]},
            ensure_ascii=False,
        )

    # Because initial text was NOT a source echo, ignoring regressive echo replies
    # preserves the non-echo partial translation instead of crashing the whole film.
    repaired, timing = repair.apply_repair_pass(
        cues,
        initial,
        chat=always_echo,
        session=session,
    )
    assert repaired == initial
    assert timing is not None
    # OpenAI repair never attaches kana regex pattern to bounded schema.
    for schema in schemas_seen:
        assert "pattern" not in schema["properties"]["translations"]["items"]["properties"]["text"]
