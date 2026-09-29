from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm.context import SourceContext
from llm.profiles.base import ProfileContext
from llm.profiles.hymt2 import HyMt2Profile
from llm.errors import (
    ResponseTruncatedError, RetryableTranslationFormatError, UntranslatedOutputError,
)
from llm.output_checks import NO_KANA_GRAMMAR


def test_native_context_keeps_one_target_and_only_relevant_terms():
    cues = [{"text": "お手伝いしましょうか"}, {"text": "大丈夫です"}, {"text": "では失礼します"}]
    profile = HyMt2Profile()
    messages = profile.build_messages(cues[1:2], ids=[1], ctx=ProfileContext(
        source_context=SourceContext.build(cues), glossary="大丈夫-不用了, 東京-东京",
    ))
    assert len(messages) == 1 and messages[0]["role"] == "user"
    prompt = messages[0]["content"]
    assert "お手伝いしましょうか" in prompt and "では失礼します" in prompt
    assert prompt.endswith("【待翻译文本】\n大丈夫です")
    assert "大丈夫 翻译成 不用了" in prompt and "东京" not in prompt
    assert "requested_ids" not in prompt


@pytest.mark.parametrize(("label", "card_name"), [
    ("简体中文", "中文"), ("繁體中文", "繁体中文"), ("English", "英语"), ("中文", "中文"),
])
def test_prompts_name_the_target_as_the_model_card_does(label, card_name):
    cues = [{"text": "こんにちは。"}, {"text": "大丈夫です"}]
    profile = HyMt2Profile()
    bare = profile.build_messages(cues[:1], ids=[0], ctx=ProfileContext(target_lang=label))
    assert bare[0]["content"].startswith(f"将以下文本翻译为{card_name}，")
    with_context = profile.build_messages(cues[:1], ids=[0], ctx=ProfileContext(
        target_lang=label, source_context=SourceContext.build(cues)))
    assert f"请结合背景信息将以下文本翻译为{card_name}。" in with_context[0]["content"]


def test_output_validation_is_independent_of_schema_and_allows_shared_han():
    profile = HyMt2Profile()
    assert profile.schema is None
    for target in ("", "大丈夫です", "没事です"):
        with pytest.raises(RetryableTranslationFormatError):
            profile.validate_translation("大丈夫です", target, "简体中文")
    profile.validate_translation("先生", "先生", "简体中文")
    profile.validate_translation("okay", "大丈夫です", "日语")


@pytest.mark.parametrize("target", ["玛丽・史密斯", "玛丽･史密斯", "等一下ー", "等一下ｰ"])
def test_shared_punctuation_does_not_reject_a_chinese_translation(target):
    HyMt2Profile().validate_translation("待って。", target, "简体中文")


@pytest.mark.parametrize("label", ["简体中文", "繁體中文", "zh-CN", "zh-Hant", "正體", "simplified"])
@pytest.mark.parametrize("target", ["ｺﾝﾆﾁﾊ", "你好ﾃﾞｽ", "你好です"])
def test_kana_including_halfwidth_is_rejected_for_supported_chinese_labels(label, target):
    profile = HyMt2Profile()
    with pytest.raises(UntranslatedOutputError) as caught:
        profile.validate_translation("こんにちは。", target, label)
    assert profile.retry_grammar(caught.value, label) == NO_KANA_GRAMMAR


def test_llama_receives_native_sampling_and_reports_truncation(monkeypatch):
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    monkeypatch.setattr(backend, "_ensure_server", lambda *_a: None)
    seen = []

    def request(payload, cancel_event):
        seen.append(payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="尚未结束"), finish_reason="length")])

    monkeypatch.setattr(backend, "_request_completion", request)
    with pytest.raises(ResponseTruncatedError) as caught:
        backend.chat_completion([{"role": "user", "content": "test"}], max_tokens=64, sampling_parameters=HyMt2Profile().sampling_parameters())
    assert caught.value.limit == 64
    assert seen[0]["extra_body"] == {"top_k": 20, "repeat_penalty": 1.05}


def test_native_sampling_reaches_wire_through_lease_and_real_sdk(monkeypatch):
    import httpx
    from openai import AsyncOpenAI
    from llm import backends
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "synthetic", "object": "chat.completion", "created": 0, "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "不用了"}, "finish_reason": "stop"}],
        })

    backend = LlamaCppServerBackend()
    monkeypatch.setattr(backend, "_ensure_server", lambda *_a: None)
    monkeypatch.setattr(backend, "cache_identity", lambda: "test-model")
    monkeypatch.setattr(backend, "_make_async_client", lambda _port: AsyncOpenAI(
        api_key="test-only", base_url="http://localhost.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ))
    monkeypatch.setitem(backends._BACKEND_REGISTRY, "llamacpp", lambda: backend)
    sampling = HyMt2Profile().sampling_parameters()
    with backends.backend_lease("llamacpp") as leased:
        result = leased.chat_completion(
            [{"role": "user", "content": "大丈夫です"}],
            temperature=sampling["temperature"], top_p=sampling["top_p"],
            max_tokens=64, sampling_parameters=sampling,
        )
    assert result == "不用了"
    assert len(seen) == 1
    assert seen[0]["top_k"] == 20 and seen[0]["repeat_penalty"] == 1.05
    assert seen[0]["temperature"] == 0.7 and seen[0]["top_p"] == 0.6
    assert "extra_body" not in seen[0]


def test_line_engine_retries_nonempty_japanese_output(monkeypatch):
    from llm import translator

    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "hymt2")
    monkeypatch.setattr(translator, "_call_request_backoff_sleep", lambda *_a, **_k: None)
    replies = iter(["大丈夫です", "不用了"])
    calls = []

    def chat(*_a, **kwargs):
        calls.append(kwargs)
        return next(replies)

    monkeypatch.setattr(translator, "_chat_with_reasoning", chat)
    texts, _, _ = translator.translate_segments([{"text": "大丈夫です"}], max_workers=1)
    assert texts == ["不用了"]
    # Grammars are a llama-server feature; an API retry stays unconstrained.
    assert [call.get("output_grammar") for call in calls] == [None, None]


def test_only_kana_rejections_for_a_chinese_target_name_a_grammar():
    profile = HyMt2Profile()
    for source, target in (("こんにちは。", "こんにちは。"), ("大丈夫です", "没事です")):
        with pytest.raises(UntranslatedOutputError) as caught:
            profile.validate_translation(source, target, "简体中文")
        assert profile.retry_grammar(caught.value, "简体中文") == NO_KANA_GRAMMAR
    with pytest.raises(UntranslatedOutputError) as empty:
        profile.validate_translation("大丈夫です", "", "简体中文")
    assert profile.retry_grammar(empty.value, "简体中文") is None
    echo = UntranslatedOutputError("echo", reason="source_echo")
    assert profile.retry_grammar(echo, "English") is None
    assert profile.retry_grammar(RetryableTranslationFormatError("shape"), "简体中文") is None


@pytest.mark.parametrize(("echo", "reason"), [
    ("こんにちは。", "source_echo"), ("ｺﾝﾆﾁﾊ", "japanese_remaining"),
])
def test_llama_retry_after_an_echo_cannot_echo_again(monkeypatch, echo, reason):
    """Issue #2: `こんにちは。` came back verbatim four times and failed the film.
    The first request stays native; the retry excludes what was rejected."""
    from llm import translator

    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "hymt2")
    monkeypatch.setattr(translator, "_call_request_backoff_sleep", lambda *_a, **_k: None)
    events = []
    grammars = []

    def chat(*_a, **kwargs):
        grammars.append(kwargs.get("output_grammar"))
        return "你好。" if kwargs.get("output_grammar") else echo

    monkeypatch.setattr(translator, "_chat_with_reasoning", chat)
    texts, _, _ = translator.translate_segments(
        [{"text": "こんにちは。"}], max_workers=1, on_progress=events.append
    )
    assert texts == ["你好。"]
    assert grammars == [None, NO_KANA_GRAMMAR]
    assert [evt["error"] for evt in events if evt.get("phase") == "batch_output_constrained"] == [
        f"Hy-MT2 reply is not a 简体中文 translation: {reason}"
    ]


def test_an_exhausted_line_cue_names_its_reason_not_json(monkeypatch):
    from llm import translator

    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "hymt2")
    monkeypatch.setattr(translator, "_call_request_backoff_sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(translator, "_chat_with_reasoning", lambda *_a, **_k: "こんにちは。")
    with pytest.raises(RuntimeError) as caught:
        translator.translate_segments([{"text": "こんにちは。"}], max_workers=1)
    assert "no usable reply" in str(caught.value) and "source_echo" in str(caught.value)
    assert "JSON" not in str(caught.value)


def test_the_grammar_reaches_llama_server_and_excludes_a_schema(monkeypatch, tmp_path):
    from llm import request_config, translator
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    monkeypatch.setattr(backend, "_ensure_server", lambda *_a: None)
    seen = []

    def request(payload, cancel_event):
        seen.append(payload)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="你好。"), finish_reason="stop")])

    monkeypatch.setattr(backend, "_request_completion", request)
    monkeypatch.setenv("TRANSLATION_MAX_TOKENS_CACHE_PATH", str(tmp_path / "limits.json"))
    monkeypatch.setattr(translator, "task_backend_name", lambda *_a, **_k: "llamacpp")
    monkeypatch.setattr(translator, "task_backend", lambda *_a, **_k: backend)
    monkeypatch.setattr(request_config, "current_profile", lambda: HyMt2Profile())
    reply = translator._chat(
        [{"role": "user", "content": "こんにちは。"}], expected_count=1, max_tokens=64,
        response_schema=None, output_grammar=NO_KANA_GRAMMAR,
    )
    assert reply == "你好。"
    assert seen[0]["extra_body"] == {"top_k": 20, "repeat_penalty": 1.05, "grammar": NO_KANA_GRAMMAR}
    assert "response_format" not in seen[0]
    with pytest.raises(ValueError, match="not both"):
        backend.chat_completion(
            [{"role": "user", "content": "x"}], response_format={"type": "object"}, grammar=NO_KANA_GRAMMAR
        )


def _continued_cues():
    return [
        {"text": "行きたくない", "start": 0.0, "end": 2.0, "continues_into_next": True},
        {"text": "わけじゃない。", "start": 2.0, "end": 4.0, "continues_from_previous": True},
    ]


def test_source_sentence_is_translated_once_without_target_splitting(monkeypatch):
    from copy import deepcopy
    from llm import translator
    from subtitles.writer import prepare_srt_blocks
    from subtitles.options import SubtitleOptions

    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "hymt2")
    source = "行きたくないわけじゃない。"
    words = [
        {"word": char, "start": i * 0.5, "end": i * 0.5 + 0.3,
         "timestamp_kind": "ctc_forced_alignment"}
        for i, char in enumerate(source)
    ]
    cues = prepare_srt_blocks(
        [{"text": source, "start": 0.0, "end": words[-1]["end"], "words": words}],
        options=SubtitleOptions(max_source_chars=4),
    )
    assert len(cues) == 1 and cues[0]["text"] == source
    original = deepcopy(cues)
    calls = []

    def chat(messages, **kwargs):
        calls.append(kwargs)
        assert messages[0]["content"].endswith("行きたくないわけじゃない。")
        assert kwargs["response_schema"] is None
        return "并不是不想去。"

    monkeypatch.setattr(translator, "_chat_with_reasoning", chat)
    texts, _, _ = translator.translate_segments(cues, max_workers=2)
    assert texts == ["并不是不想去。"]
    assert cues == original
    assert len(calls) == 1 and calls[0]["expected_count"] == 1


def test_each_preplanned_clause_remains_an_independent_translation(monkeypatch):
    from llm import translator

    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "hymt2")
    cues = [
        {"text": "明日の仕事が早く終わったら、", "continues_into_next": True},
        {"text": "そちらに寄るよ。", "continues_from_previous": True},
    ]
    calls = []

    def chat(messages, **kwargs):
        calls.append(kwargs["expected_count"])
        target = messages[0]["content"].split("【待翻译文本】\n")[-1]
        return {cues[0]["text"]: "如果明天工作结束得早", cues[1]["text"]: "我就过去一趟"}[target]

    monkeypatch.setattr(translator, "_chat_with_reasoning", chat)
    texts, _, _ = translator.translate_segments(cues, max_workers=1)
    assert texts == ["如果明天工作结束得早", "我就过去一趟"]
    assert calls == [1, 1]


def test_line_profile_rejects_multiple_targets_instead_of_distributing_text():
    with pytest.raises(ValueError, match="one source cue"):
        HyMt2Profile().build_messages(_continued_cues(), ids=[0, 1], ctx=ProfileContext())
    with pytest.raises(RetryableTranslationFormatError, match="one source cue"):
        HyMt2Profile().parse_response("并不是不想去。", ids=[0, 1])


def test_a_line_that_passed_only_under_the_grammar_is_marked_and_the_cache_keeps_it(
    monkeypatch, tmp_path
):
    """Finished is not verified: a constrained retry is how a job survives an
    echo, so the line completes and says so. A restored run must say it too."""
    from llm import translator

    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "hymt2")
    monkeypatch.setattr(translator, "_call_request_backoff_sleep", lambda *_a, **_k: None)

    def fresh(messages, **kwargs):
        # Only the greeting echoes; the other line is fine on the first try.
        if messages[0]["content"].rstrip().endswith("ありがとう。"):
            return "谢谢。"
        return "你好。" if kwargs.get("output_grammar") else "こんにちは。"

    monkeypatch.setattr(translator, "_chat_with_reasoning", fresh)
    cache_path = str(tmp_path / "translation_cache.jsonl")
    cues = [{"text": "こんにちは。", "start": 0.0, "end": 1.0},
            {"text": "ありがとう。", "start": 2.0, "end": 3.0}]
    texts, timings, _ = translator.translate_segments(cues, max_workers=1, cache_path=cache_path)
    assert texts == ["你好。", "谢谢。"]
    reviews = [entry for timing in timings for entry in timing.get("translation_review", [])]
    assert reviews == [
        {"id": 0, "reason": "constrained_retry", "rejected": "source_echo", "requests": 2}
    ]

    def no_requests(*_a, **_k):
        raise AssertionError("a restored run must not translate again")

    monkeypatch.setattr(translator, "_chat_with_reasoning", no_requests)
    texts, timings, _ = translator.translate_segments(cues, max_workers=1, cache_path=cache_path)
    assert texts == ["你好。", "谢谢。"]
    assert [timing.get("cache_hit_type") for timing in timings[:2]] == ["exact_batch", "exact_batch"]
    restored = [entry for timing in timings for entry in timing.get("translation_review", [])]
    assert restored == reviews
