from __future__ import annotations

import json
import re
import threading
from collections import defaultdict

import pytest

from llm import translator
def _segments(count: int) -> list[dict]:
    return [
        {"start": float(index), "end": float(index) + 1.0, "text": f"ja-{index}"}
        for index in range(count)
    ]


def _mock_json(start: int, count: int) -> str:
    items = [
        {"id": index, "text": f"zh-{index}"}
        for index in range(start, start + count)
    ]
    import json

    return json.dumps({"translations": items}, ensure_ascii=False)


def _requested_ids_from_messages(messages) -> list[int]:
    content = messages[1]["content"]
    match = re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", content)
    assert match is not None, content
    return json.loads(match.group(1))


def _fixed_prefix_from_messages(messages) -> str:
    content = messages[1]["content"]
    marker = "【本次任务】"
    assert marker in content
    return content.split(marker, 1)[0]


def test_split_into_batches():
    assert len(translator._split_into_batches(_segments(0), 200)) == 0
    assert len(translator._split_into_batches(_segments(199), 200)) == 1
    assert len(translator._split_into_batches(_segments(200), 200)) == 1
    assert len(translator._split_into_batches(_segments(201), 200)) == 2
    assert len(translator._split_into_batches(_segments(450), 200)) == 3


def test_translate_segments_single_request_when_below_threshold(monkeypatch):
    calls: list[int] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        calls.append(expected_count)
        return _mock_json(0, expected_count)

    monkeypatch.setattr(translator, "_chat", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(100),
        batch_size=200,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert calls == [100]
    assert retry_events == []
    assert len(zh_texts) == 100
    assert zh_texts[0] == "zh-0"
    assert zh_texts[-1] == "zh-99"
    assert timings[0]["mode"] == "single_request_full_context"


def test_translate_segments_uses_task_character_reference(monkeypatch):
    system_prompts: list[str] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        system_prompts.append(messages[0]["content"])
        return _mock_json(0, expected_count)

    monkeypatch.setattr(translator, "CHARACTER_FULL_NAME_REFERENCE", "Env Name")
    monkeypatch.setattr(translator, "_chat", fake_chat)

    translator.translate_segments(
        _segments(1),
        batch_size=10,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
        character_reference="Task Name",
    )

    assert system_prompts
    assert "Task Name" in system_prompts[0]
    assert "Env Name" not in system_prompts[0]


def test_character_name_guidance_is_conservative_for_unrelated_surnames():
    prompt = translator._build_system_prompt(
        1,
        "小那海あや",
        target_lang="简体中文",
        glossary="",
    )

    assert "不要为了统一人物而把不同汉字姓氏或不同读音的称呼强行合并" in prompt
    assert "按日语读音罗马音化" in prompt
    assert "高橋、高岡、高野" not in prompt
    assert "Takahashi/Takaoka/Takano" not in prompt


def test_repair_prompt_keeps_asr_special_cases_out_of_static_prompt():
    messages = translator._build_repair_messages(
        [
            {"start": 0.0, "end": 1.0, "text": "おまけ、さらけないでください!"},
            {"start": 1.0, "end": 2.0, "text": "きゅう、きゅうしてください"},
        ],
        ["不要露出来！", "请吸，请用力吸。"],
        [0, 1],
        {
            0: ["suspicious_omake_asr"],
            1: ["suspicious_kyuu_asr"],
        },
        target_lang="简体中文",
        glossary="",
        character_reference="",
    )

    system_prompt = messages[0]["content"]
    assert "おまけ" not in system_prompt
    assert "きゅうしてください" not in system_prompt
    assert "asr_homophone_or_context_drift" in messages[1]["content"]
    assert "suspicious_omake_asr" not in messages[1]["content"]


def test_translate_segments_batched(monkeypatch):
    calls: list[tuple[int, int]] = []
    lock = threading.Lock()

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        requested_ids = _requested_ids_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        start = min(requested_ids)
        with lock:
            calls.append((start, expected_count))
        return _mock_json(start, expected_count)

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(450),
        batch_size=200,
        max_workers=3,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert sorted(calls) == [(0, 200), (200, 200), (400, 50)]
    assert retry_events == []
    assert len(zh_texts) == 450
    assert zh_texts[:3] == ["zh-0", "zh-1", "zh-2"]
    assert zh_texts[199:202] == ["zh-199", "zh-200", "zh-201"]
    assert zh_texts[-1] == "zh-449"
    batch_timings = [item for item in timings if item.get("mode") == "batched_full_context"]
    assert [item["segment_count"] for item in batch_timings] == [200, 200, 50]
    assert timings[-1]["mode"] == "batched_full_context_total"


def test_batched_translation_emits_worker_timeline_diagnostics(monkeypatch):
    events: list[dict] = []
    done_timings: list[dict] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        requested_ids = _requested_ids_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        if on_progress:
            on_progress({"phase": "thinking", "reasoning_chars": len(requested_ids)})
        return json.dumps(
            {
                "translations": [
                    {"id": idx, "text": f"zh-{idx}"}
                    for idx in requested_ids
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(4),
        batch_size=2,
        max_workers=2,
        cache_path="",
        target_lang="简体中文",
        glossary="",
        on_batch_done=done_timings.append,
        on_progress=events.append,
    )

    assert retry_events == []
    assert zh_texts == ["zh-0", "zh-1", "zh-2", "zh-3"]

    starts = [event for event in events if event.get("phase") == "batch_start"]
    first_tokens = [
        event for event in events if event.get("phase") == "batch_first_token"
    ]
    finishes = [event for event in events if event.get("phase") == "batch_finish"]
    assert {event["batch_index"] for event in starts} == {0, 1}
    assert {event["batch_index"] for event in first_tokens} == {0, 1}
    assert {event["batch_index"] for event in finishes} == {0, 1}

    for event in starts:
        assert event["diagnostic"] is True
        assert event["started_ts"] > 0
        assert event["thread_id"] > 0
        assert event["thread_name"]
        assert event["requested_ids"] in ([0, 1], [2, 3])
    for event in first_tokens:
        assert event["diagnostic"] is True
        assert event["first_token_ts"] >= event["started_ts"]
        assert event["thread_id"] > 0
    for event in finishes:
        assert event["diagnostic"] is True
        assert event["finished_ts"] >= event["started_ts"]
        assert event["elapsed_s"] >= 0
        assert event["request_count"] == 1

    batch_timings = [
        item for item in timings if item.get("mode") == "batched_full_context"
    ]
    assert done_timings == batch_timings
    for timing in batch_timings:
        assert timing["started_ts"] > 0
        assert timing["finished_ts"] >= timing["started_ts"]
        assert timing["first_token_ts"] >= timing["started_ts"]
        assert timing["worker_thread_id"] > 0
        assert timing["worker_thread_name"]


def test_translate_segments_uses_task_api_format(monkeypatch):
    calls: list[str] = []

    def fake_chat(
        messages,
        expected_count=0,
        on_progress=None,
        reasoning_effort=None,
        api_format=None,
        **_kwargs,
    ):
        calls.append(api_format)
        return _mock_json(0, expected_count)

    monkeypatch.setenv("LLM_API_FORMAT", "chat")
    monkeypatch.setattr(translator, "_chat", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(1),
        batch_size=10,
        max_workers=1,
        cache_path="",
        target_lang="簡体中文",
        glossary="",
        api_format="responses",
    )

    assert calls == ["responses"]
    assert retry_events == []
    assert zh_texts == ["zh-0"]
    assert timings[0]["mode"] == "single_request_full_context"


def test_aggregated_progress_callback(monkeypatch):
    events: list[dict] = []
    current = {"value": 100.0}

    def fake_monotonic():
        current["value"] += 0.3
        return current["value"]

    monkeypatch.setattr(translator.time, "monotonic", fake_monotonic)
    callbacks, _ = translator._make_aggregated_progress_callback(4, 450, events.append)

    callbacks[0]({"phase": "thinking", "reasoning_chars": 10})
    callbacks[1]({"phase": "thinking", "reasoning_chars": 40})
    callbacks[2]({"phase": "thinking", "reasoning_chars": 25})
    callbacks[3]({"phase": "thinking", "reasoning_chars": 30})

    thinking_events = [event for event in events if event["phase"] == "thinking"]
    assert thinking_events[-1]["reasoning_chars"] == 40

    callbacks[0]({"phase": "done", "translated": 100, "expected": 100})
    callbacks[1]({"phase": "done", "translated": 100, "expected": 100})
    callbacks[2]({"phase": "done", "translated": 100, "expected": 100})
    callbacks[3]({"phase": "done", "translated": 150, "expected": 150})

    done_events = [event for event in events if event["phase"] == "done"]
    assert done_events == [{"phase": "done", "translated": 450, "expected": 450}]


def test_batch_retry_isolation(monkeypatch):
    attempts: defaultdict[int, int] = defaultdict(int)

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        requested_ids = _requested_ids_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        start = min(requested_ids)
        attempts[start] += 1
        if start == 200 and attempts[start] < 2:
            raise translator.RetryableTranslationFormatError("batch failed once")
        return _mock_json(start, expected_count)

    monkeypatch.setattr(translator, "TRANSLATION_API_RETRIES", 2)
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda attempt, exc: None)
    monkeypatch.setattr(translator, "_chat", fake_chat)

    zh_texts, _timings, retry_events = translator.translate_segments(
        _segments(450),
        batch_size=200,
        max_workers=3,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert attempts[0] == 1
    assert attempts[200] == 2
    assert attempts[400] == 1
    assert retry_events == []
    assert zh_texts[200] == "zh-200"
    assert zh_texts[-1] == "zh-449"


def test_batch_retry_only_requests_missing_ids(monkeypatch):
    calls: list[tuple[list[int], int]] = []

    def ids_from_messages(messages) -> list[int]:
        return _requested_ids_from_messages(messages)

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        ids = ids_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        calls.append((ids, expected_count))
        if ids == [0, 1, 2, 3, 4]:
            returned_ids = [0, 1, 3]
        else:
            returned_ids = ids
        import json

        return json.dumps(
            {
                "translations": [
                    {"id": idx, "text": f"zh-{idx}"}
                    for idx in returned_ids
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "TRANSLATION_API_RETRIES", 2)
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda attempt, exc: None)
    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(6),
        batch_size=5,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert retry_events == []
    assert zh_texts == [f"zh-{idx}" for idx in range(6)]
    assert calls == [
        ([0, 1, 2, 3, 4], 5),
        ([2, 4], 2),
        ([5], 1),
    ]
    assert next(item for item in timings if item.get("batch_index") == 0)["request_count"] == 2


def test_batch_retry_gets_fresh_budget_after_missing_set_shrinks(monkeypatch):
    calls: list[list[int]] = []

    def ids_from_messages(messages) -> list[int]:
        return _requested_ids_from_messages(messages)

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        ids = ids_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        calls.append(ids)
        returned_ids = ids[:-1] if len(ids) > 1 else ids
        import json

        return json.dumps(
            {
                "translations": [
                    {"id": idx, "text": f"zh-{idx}"}
                    for idx in returned_ids
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "TRANSLATION_API_RETRIES", 2)
    monkeypatch.setattr(translator, "TRANSLATION_BATCH_REPAIR_RETRIES", 2)
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda attempt, exc: None)
    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(5),
        batch_size=4,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert retry_events == []
    assert zh_texts == [f"zh-{idx}" for idx in range(5)]
    assert calls == [
        [0, 1, 2, 3],
        [3],
        [4],
    ]
    assert next(item for item in timings if item.get("batch_index") == 0)["request_count"] == 2


def test_batched_translation_uses_stable_full_json_prefix_and_requested_ids(monkeypatch):
    calls: list[dict] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        requested_ids = _requested_ids_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        calls.append(
            {
                "expected_count": expected_count,
                "requested_ids": requested_ids,
                "system": messages[0]["content"],
                "fixed_prefix": _fixed_prefix_from_messages(messages),
                "user": messages[1]["content"],
            }
        )
        return json.dumps(
            {
                "translations": [
                    {"id": idx, "text": f"zh-{idx}"}
                    for idx in requested_ids
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(6),
        batch_size=3,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert retry_events == []
    assert zh_texts == [f"zh-{idx}" for idx in range(6)]
    assert [call["requested_ids"] for call in calls] == [[0, 1, 2], [3, 4, 5]]
    assert [call["expected_count"] for call in calls] == [3, 3]
    assert calls[0]["system"] == calls[1]["system"]
    assert calls[0]["fixed_prefix"] == calls[1]["fixed_prefix"]
    assert '"id":0' in calls[0]["fixed_prefix"]
    assert '"id":5' in calls[0]["fixed_prefix"]
    assert "requested_ids = [0, 1, 2]" in calls[0]["user"]
    assert "requested_ids = [3, 4, 5]" in calls[1]["user"]
    batch_timings = [item for item in timings if item.get("mode") == "batched_full_context"]
    assert batch_timings[0]["requested_ids"] == [0, 1, 2]
    assert batch_timings[1]["requested_ids"] == [3, 4, 5]


def test_batch_warmup_runs_before_pending_batches(monkeypatch):
    calls: list[tuple[int, list[int]]] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        requested_ids = _requested_ids_from_messages(messages)
        calls.append((expected_count, requested_ids))
        return json.dumps(
            {
                "translations": [
                    {"id": idx, "text": f"zh-{idx}"}
                    for idx in requested_ids
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(4),
        batch_size=2,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert retry_events == []
    assert zh_texts == [f"zh-{idx}" for idx in range(4)]
    assert calls == [(0, []), (2, [0, 1]), (2, [2, 3])]
    assert timings[0]["is_warmup"] is True
    assert timings[0]["requested_ids"] == []
    assert timings[0]["mode"] == "translation_prefix_warmup"


def test_translation_repair_pass_uses_neighbor_context_for_asr_fragment(monkeypatch):
    calls: list[dict] = []
    segments = [
        {"start": 0.0, "end": 1.0, "text": "半分出ちゃった外に半分外出した"},
        {"start": 1.0, "end": 2.0, "text": "これ指でさマンゴーに精子さ"},
        {"start": 2.0, "end": 3.0, "text": "入れてもらう。"},
    ]

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        requested_ids = _requested_ids_from_messages(messages)
        content = messages[1]["content"]
        calls.append(
            {
                "repair": "【翻译修复任务】" in content,
                "requested_ids": requested_ids,
                "content": content,
            }
        )
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        if "【翻译修复任务】" in content:
            return json.dumps(
                {
                    "translations": [
                        {"id": 1, "text": "用手指把精液弄进小穴里"}
                    ]
                },
                ensure_ascii=False,
            )
        initial_texts = {
            0: "一半射出来，一半射外面了",
            1: "用这个手指，精液",
            2: "让人塞进去。",
        }
        return json.dumps(
            {
                "translations": [
                    {"id": idx, "text": initial_texts[idx]}
                    for idx in requested_ids
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        segments,
        batch_size=2,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert retry_events == []
    assert zh_texts == [
        "一半射出来，一半射外面了",
        "用手指把精液弄进小穴里",
        "让人塞进去。",
    ]
    repair_calls = [call for call in calls if call["repair"]]
    assert len(repair_calls) == 1
    assert repair_calls[0]["requested_ids"] == [1]
    assert "入れてもらう。" in repair_calls[0]["content"]
    assert "用这个手指，精液" in repair_calls[0]["content"]
    assert any(item.get("mode") == "translation_repair_pass" for item in timings)


def test_translation_repair_pass_fixes_forbidden_genital_term(monkeypatch):
    calls: list[dict] = []
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "やばい、3人の選手がまんこ入っちゃった。",
        },
        {"start": 1.0, "end": 2.0, "text": "あっ、違う、違う。"},
    ]

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        requested_ids = _requested_ids_from_messages(messages)
        content = messages[1]["content"]
        calls.append({"repair": "【翻译修复任务】" in content, "content": content})
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        if "【翻译修复任务】" in content:
            return json.dumps(
                {
                    "translations": [
                        {"id": 0, "text": "不得了，三个选手都插进小穴了。"}
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "translations": [
                    {"id": idx, "text": text}
                    for idx, text in zip(
                        requested_ids,
                        ["不得了，三个选手都插进阴道了。", "啊，不对，不对。"],
                    )
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        segments,
        batch_size=1,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert retry_events == []
    assert zh_texts[0] == "不得了，三个选手都插进小穴了。"
    repair_call = next(call for call in calls if call["repair"])
    assert "まんこ" in repair_call["content"]
    assert "阴道" in repair_call["content"]
    assert any(item.get("mode") == "translation_repair_pass" for item in timings)


def test_translation_repair_selects_suspicious_asr_homophones_in_sexual_context():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "あっ、気持ちいい"},
        {"start": 1.0, "end": 2.0, "text": "おまけ、さらけないでください!"},
        {"start": 2.0, "end": 3.0, "text": "声が出てるぞ"},
        {"start": 3.0, "end": 4.0, "text": "イッちゃう"},
        {"start": 4.0, "end": 5.0, "text": "あんな、私の国にできたことなんか、大癖に……"},
        {"start": 5.0, "end": 6.0, "text": "なんでお前は言う!?"},
        {"start": 6.0, "end": 7.0, "text": "あっ、気持ちいいっ!"},
        {"start": 7.0, "end": 8.0, "text": "こ、こよく言う……"},
        {"start": 8.0, "end": 9.0, "text": "そっちは止まんないじゃないか。"},
        {"start": 9.0, "end": 10.0, "text": "あっ、気持ちいいっ!"},
        {"start": 10.0, "end": 11.0, "text": "きゅう、きゅうしてください"},
    ]
    zh_texts = [
        "啊，好舒服。",
        "不要露出来！",
        "声音都出来了。",
        "要去了。",
        "那种事，在我的国家，明明是大变态……",
        "为什么你要说？",
        "啊，舒服！",
        "你倒会说……",
        "你那边不是停不下来吗？",
        "啊，舒服！",
        "请，请吸吮。",
    ]

    repair_ids, reasons = translator._select_translation_repair_ids(segments, zh_texts)

    assert repair_ids == [1, 4, 7, 10]
    assert reasons[1] == ["suspicious_omake_asr"]
    assert reasons[4] == ["suspicious_kuni_asr"]
    assert reasons[7] == ["suspicious_koyoku_asr"]
    assert reasons[10] == ["suspicious_kyuu_asr"]


def test_translation_repair_does_not_flag_literal_country_outside_sex_context():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "私の国にできた会社です。"},
        {"start": 1.0, "end": 2.0, "text": "来月から働きます。"},
    ]
    zh_texts = ["这是在我的国家成立的公司。", "下个月开始工作。"]

    repair_ids, reasons = translator._select_translation_repair_ids(segments, zh_texts)

    assert repair_ids == []
    assert reasons == {}
