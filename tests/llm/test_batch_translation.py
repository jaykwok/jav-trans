from __future__ import annotations

import json
import re
import threading
from collections import defaultdict

import pytest

from llm import engine as engine_module
from llm import repair as repair_module
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


def test_the_repair_write_back_partitions_with_the_engine_not_a_copy():
    """One partition function, shared - the same rule as the reasoning tiers.

    `_persist_repaired_translation_cache` rebuilds the batches to write repaired
    text under `_translation_cache_key(b_index, ...)`, the key the engine will
    read. A second implementation of the same five lines is one edit away from
    splitting differently, and every repaired translation would then be stored
    under a key nothing ever looks up - silently, since writing succeeds.
    """
    from llm import engine as engine_module

    assert translator._split_into_batches is engine_module._split_into_batches


def test_batch_size_ignores_the_worker_count(monkeypatch):
    """Decoupled 2026-08-24. The old rule sized batches as `count / (workers x 2)`
    to balance the pool, which made concurrency a billing control: reasoning is
    charged per request and barely scales with the batch, so asking for more
    workers manufactured more requests and multiplied the thinking bill for the
    same work. Cost belongs to `TRANSLATION_BATCH_SIZE`, parallelism to workers."""
    monkeypatch.setattr(translator, "TRANSLATION_BATCH_SIZE", 200)
    assert translator._auto_translation_batch_size(0) == 0
    assert translator._auto_translation_batch_size(1384) == 200
    assert translator._auto_translation_batch_size(60) == 60


def test_more_workers_no_longer_buys_more_requests(monkeypatch):
    """The regression this closes: on 1,396 cues the same film cost 8 requests at
    4 workers and 32 at 16, purely because the pool size chose the batch size."""
    monkeypatch.setattr(translator, "TRANSLATION_BATCH_SIZE", 200)
    cues = 1396
    size = translator._auto_translation_batch_size(cues)
    requests = len(translator._split_into_batches(_segments(cues), size))

    assert requests == 7
    for workers in (1, 4, 16, 64):
        assert translator._auto_translation_workers(requests, workers) == min(
            workers, requests
        )


def test_workers_never_exceed_the_batches_there_are():
    """Spawning idle workers only ever looked useful because more workers used to
    manufacture more batches."""
    assert translator._auto_translation_workers(3, 16) == 3
    assert translator._auto_translation_workers(40, 4) == 4
    assert translator._auto_translation_workers(0, 8) == 1


def test_auto_translation_batch_size_keeps_a_large_request_safety_cap(monkeypatch):
    cap = 400
    monkeypatch.setattr(translator, "TRANSLATION_BATCH_SIZE", cap)
    assert translator._auto_translation_batch_size(cap + 100) == cap
    assert translator._auto_translation_batch_size(100_000) == cap


def test_default_translation_batch_size_is_the_measured_one():
    """200 is now the whole rule, so it is the only number deciding a film's
    request count - and the request count is what the thinking bill tracks.
    Measured at this size on sample-v: 8 requests per run, with 7 of 32 coming
    back with a dropped tail and needing a reissue."""
    from core.config import DEFAULT_SETTINGS

    assert DEFAULT_SETTINGS["TRANSLATION_BATCH_SIZE"] == "200"


def test_env_float_falls_back_on_bad_value(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.8")
    assert translator._env_float("LLM_TEMPERATURE", 0.6) == 0.8
    monkeypatch.setenv("LLM_TEMPERATURE", "not-a-number")
    assert translator._env_float("LLM_TEMPERATURE", 0.6) == 0.6
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    assert translator._env_float("LLM_TEMPERATURE", 0.6) == 0.6


def test_env_int_clamped_bounds_and_fallback(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BATCH_SIZE", "32")
    assert translator._env_int_clamped("TRANSLATION_BATCH_SIZE", 64, 8, 400) == 32
    monkeypatch.setenv("TRANSLATION_BATCH_SIZE", "5000")
    assert translator._env_int_clamped("TRANSLATION_BATCH_SIZE", 64, 8, 400) == 400
    monkeypatch.setenv("TRANSLATION_BATCH_SIZE", "1")
    assert translator._env_int_clamped("TRANSLATION_BATCH_SIZE", 64, 8, 400) == 8
    monkeypatch.setenv("TRANSLATION_BATCH_SIZE", "garbage")
    assert translator._env_int_clamped("TRANSLATION_BATCH_SIZE", 64, 8, 400) == 64


def test_translate_segments_single_batch_below_threshold(monkeypatch):
    calls: list[int] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        calls.append(expected_count)
        return _mock_json(0, expected_count)

    monkeypatch.setattr(translator, "_chat", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(60),
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert calls == [60]
    assert retry_events == []
    assert len(zh_texts) == 60
    assert zh_texts[0] == "zh-0"
    assert zh_texts[-1] == "zh-59"
    assert timings[0]["mode"] == "batched_full_context"


def test_translate_segments_uses_task_character_reference(monkeypatch):
    system_prompts: list[str] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        system_prompts.append(messages[0]["content"])
        return _mock_json(0, expected_count)

    monkeypatch.setattr(translator, "_chat", fake_chat)

    translator.translate_segments(
        _segments(1),
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
        character_reference="Task Name",
    )

    assert system_prompts
    assert "Task Name" in system_prompts[0]


def test_character_name_guidance_is_conservative_for_unrelated_surnames():
    prompt = translator._build_system_prompt(
        "小那海あや",
        target_lang="简体中文",
        glossary="",
    )

    assert "不要为了统一人物而把不同汉字姓氏或不同读音的称呼强行合并" in prompt
    assert "按日语读音罗马音化" in prompt
    assert "高橋、高岡、高野" not in prompt
    assert "Takahashi/Takaoka/Takano" not in prompt


def test_repair_prompt_does_not_authorize_asr_or_context_rewrites():
    messages = translator._build_repair_messages(
        [
            {"start": 0.0, "end": 1.0, "text": "おまけ、さらけないでください!"},
            {"start": 1.0, "end": 2.0, "text": "きゅう、きゅうしてください"},
        ],
        ["不要露出来！", "请吸，请用力吸。"],
        [0, 1],
        {
            0: ["length_mismatch"],
            1: ["length_mismatch"],
        },
        target_lang="简体中文",
        glossary="",
        character_reference="",
    )

    system_prompt = messages[0]["content"]
    assert "おまけ" not in system_prompt
    assert "きゅうしてください" not in system_prompt
    combined = messages[0]["content"] + messages[1]["content"]
    assert "明显 ASR 同音误听" not in combined
    assert "上下文漂移" not in combined
    assert "术语漂移" not in combined
    assert "被切断的半句" not in combined
    assert "asr_homophone_or_context_drift" not in combined
    assert "suspicious_omake_asr" not in combined


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
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 200)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(450),
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
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(4),
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


def test_aggregated_progress_callback(monkeypatch):
    events: list[dict] = []
    current = {"value": 100.0}

    def fake_monotonic():
        current["value"] += 0.3
        return current["value"]

    # Patch the clock where it is read. `translator.time` used to work only
    # because both modules held the same stdlib module object, so this was
    # really patching `engine`'s clock through an unrelated name.
    monkeypatch.setattr(engine_module.time, "monotonic", fake_monotonic)
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
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda attempt, exc, **_kw: None)
    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 200)

    zh_texts, _timings, retry_events = translator.translate_segments(
        _segments(450),
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
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda attempt, exc, **_kw: None)
    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 5)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(6),
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
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda attempt, exc, **_kw: None)
    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 4)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(5),
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
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 3)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(6),
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
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(4),
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


def test_batch_warmup_also_runs_for_summary_fallback(monkeypatch):
    calls: list[dict] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        requested_ids = _requested_ids_from_messages(messages)
        calls.append(
            {
                "expected_count": expected_count,
                "requested_ids": requested_ids,
                "system": messages[0]["content"],
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
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)
    monkeypatch.setattr(translator, "TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS", 1)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(4),
        max_workers=2,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert retry_events == []
    assert zh_texts == ["zh-0", "zh-1", "zh-2", "zh-3"]
    assert calls[0]["expected_count"] == 0
    assert calls[0]["requested_ids"] == []
    assert calls[0]["system"] == calls[1]["system"] == calls[2]["system"]
    assert timings[0]["mode"] == "translation_prefix_warmup"
    assert timings[0]["prefix_mode"] == "summary_fallback"


def test_translation_repair_pass_does_not_fix_asr_fragments(monkeypatch):
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
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    zh_texts, timings, retry_events = translator.translate_segments(
        segments,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert retry_events == []
    assert zh_texts == [
        "一半射出来，一半射外面了",
        "用这个手指，精液",
        "让人塞进去。",
    ]
    repair_calls = [call for call in calls if call["repair"]]
    assert repair_calls == []
    assert not any(item.get("mode") == "translation_repair_pass" for item in timings)


def test_translation_repair_pass_does_not_fix_term_drift(monkeypatch):
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
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 1)

    zh_texts, timings, retry_events = translator.translate_segments(
        segments,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert retry_events == []
    assert zh_texts[0] == "不得了，三个选手都插进阴道了。"
    assert [call for call in calls if call["repair"]] == []
    assert not any(item.get("mode") == "translation_repair_pass" for item in timings)


def test_translation_repair_does_not_select_suspicious_asr_homophones():
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

    assert repair_ids == []
    assert reasons == {}


def test_translation_repair_selects_length_mismatch_candidates():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "これは普通の文です。"},
        {"start": 1.0, "end": 2.0, "text": "短い"},
        {"start": 2.0, "end": 3.0, "text": "これはかなり長い日本語の原文です。"},
    ]
    zh_texts = [
        "这是普通句子。",
        "这是一个明显被过度展开的中文翻译，长度远远超过原文。",
        "嗯",
    ]

    repair_ids, reasons = translator._select_translation_repair_ids(segments, zh_texts)

    assert repair_ids == [1, 2]
    assert reasons[1] == ["length_mismatch"]
    assert reasons[2] == ["length_mismatch"]


def test_the_repair_gate_selects_echo_kana_and_length_anomalies():
    """One detector set, not two. There used to be a second selector for the
    cascade's escalation path with these same three checks plus a copy of the
    length one, which is the arrangement where they drift apart."""
    segments = [
        {"text": "これは翻訳されるべきです。"},
        {"text": "こんにちは。"},
        {"text": "これはかなり長い日本語の字幕です。"},
        {"text": "大丈夫ですか。"},
    ]
    zh_texts = [
        " これは翻訳されるべきです ",
        "你好，太郎さん。",
        "嗯",
        "没事吧？",
    ]

    repair_ids, reasons = translator._select_translation_repair_ids(segments, zh_texts)

    assert repair_ids == [0, 1, 2]
    assert reasons[0] == ["source_echo", "japanese_remaining"]
    assert reasons[1] == ["japanese_remaining"]
    assert reasons[2] == ["length_mismatch"]


def test_the_repair_gate_catches_a_glossary_term_the_translation_dropped():
    """Measured on sample-v 2026-08-24: at effort=low the base pass rendered 6
    of 37 ちんぽ cues as 鸡巴 instead of the configured 肉棒, and the other three
    detectors saw none of them - a substituted term is not an echo, carries no
    kana, and is the same length. Thinking paraphrases away from an injected
    term list, so the cheap tier is only safe if the gate can see that."""
    segments = [
        {"text": "新吉のチンポが、新吉のチンポが..."},
        {"text": "ちんぽ、気持ちいい。"},
        {"text": "今日はいい天気ですね。"},
    ]
    zh_texts = [
        "新吉的鸡巴 新吉的鸡巴",
        "肉棒，好舒服。",
        "今天天气真好呢。",
    ]
    glossary = "ちんぽ-肉棒, チンポ-肉棒"

    repair_ids, reasons = translator._select_translation_repair_ids(
        segments, zh_texts, glossary
    )

    assert repair_ids == [0]
    assert reasons[0] == ["glossary_violation"]


def test_the_repair_gate_ignores_the_glossary_when_none_is_configured():
    """No glossary means no opinion about wording; the rule must not invent one."""
    segments = [{"text": "新吉のチンポが、新吉のチンポが..."}]
    zh_texts = ["新吉的鸡巴 新吉的鸡巴"]

    assert translator._select_translation_repair_ids(segments, zh_texts) == ([], {})
    assert translator._select_translation_repair_ids(segments, zh_texts, "") == ([], {})


def test_the_repair_gate_catches_a_rendering_that_drifted_from_the_settled_index():
    """`settled_pairs` comes from `global_glossary.derive_settled_glossary` -
    this same film's own dominant rendering for a line it translated more than
    once. A cue whose exact source line has a settled entry but whose current
    text does not match it is exactly the drift the index exists to catch."""
    segments = [
        {"text": "気持ちいい…"},
        {"text": "気持ちいい…"},
        {"text": "今日はいい天気ですね。"},
    ]
    zh_texts = ["好舒服…", "爽死了…", "今天天气真好呢。"]
    settled_pairs = {"気持ちいい…": "好舒服…"}

    repair_ids, reasons = translator._select_translation_repair_ids(
        segments, zh_texts, "", settled_pairs
    )

    assert repair_ids == [1]
    assert reasons[1] == ["inconsistent_rendering"]


def test_the_repair_gate_ignores_settled_pairs_when_none_are_derived():
    segments = [{"text": "気持ちいい…"}]
    zh_texts = ["爽死了…"]

    assert translator._select_translation_repair_ids(segments, zh_texts, "", {}) == (
        [],
        {},
    )
    assert translator._select_translation_repair_ids(segments, zh_texts, "", None) == (
        [],
        {},
    )


def test_the_repair_prompt_names_the_glossary_reason():
    """The reason reaches the model as a category it was told how to act on;
    an unmapped reason degrades to the generic translation_quality label."""
    segments = [{"text": "新吉のチンポが..."}]
    zh_texts = ["新吉的鸡巴"]
    repair_ids, reasons = translator._select_translation_repair_ids(
        segments, zh_texts, "チンポ-肉棒"
    )

    items = repair_module._build_repair_context_items(
        segments, zh_texts, repair_ids, reasons
    )

    assert items[0]["reason"] == ["glossary_violation"]


def test_a_cheap_first_pass_escalates_only_the_flagged_ids(monkeypatch):
    """The cost cascade, end to end: the whole film goes out at the job's tier,
    then only the ids a local detector flagged are reissued - cheap (`none`)
    first, and only what is still wrong after that pays for a tier up. The
    saving is that the escalation is proportional to the failures, not to the
    film - reasoning is charged per request and does not scale with the batch.

    The fake repair backend deliberately does not fix anything at `none`
    (returns the same broken text), so this exercises the real cascade: a
    none-tier repair attempt for every flagged id, then an escalated attempt
    for whatever the local detectors still flag afterwards.
    """
    segments = [
        {"start": 0.0, "end": 1.0, "text": "これは翻訳されるべきです。"},
        {"start": 1.0, "end": 2.0, "text": "こんにちは。"},
        {"start": 2.0, "end": 3.0, "text": "これはかなり長い日本語の字幕です。"},
        {"start": 3.0, "end": 4.0, "text": "大丈夫ですか。"},
    ]
    initial = {
        0: "これは翻訳されるべきです。",
        1: "你好，太郎さん。",
        2: "嗯",
        3: "没事吧？",
    }
    repaired = {0: "这句话应该被翻译。", 1: "你好，太郎。", 2: "这是一条很长的字幕。"}
    calls: list[dict] = []

    def fake_chat(messages, expected_count=0, reasoning_effort=None, **_kwargs):
        ids = _requested_ids_from_messages(messages)
        is_repair = "【翻译修复任务】" in messages[1]["content"]
        calls.append(
            {"ids": ids, "repair": is_repair, "reasoning_effort": reasoning_effort}
        )
        if is_repair:
            # Only the escalated tier actually fixes these - the none-tier
            # repair attempt returns the same broken text on purpose.
            values = repaired if reasoning_effort == "low" else initial
        else:
            values = initial
        return json.dumps(
            {"translations": [{"id": idx, "text": values[idx]} for idx in ids]},
            ensure_ascii=False,
        )

    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)
    monkeypatch.setattr(translator, "TRANSLATION_PREFIX_WARMUP", False)

    zh_texts, timings, retry_events = translator.translate_segments(
        segments,
        max_workers=2,
        cache_path="",
        reasoning_effort="none",
    )

    assert retry_events == []
    assert zh_texts == [repaired[0], repaired[1], repaired[2], initial[3]]
    first_pass = [call for call in calls if not call["repair"]]
    repair = [call for call in calls if call["repair"]]
    assert {idx for call in first_pass for idx in call["ids"]} == {0, 1, 2, 3}
    assert all(call["reasoning_effort"] == "none" for call in first_pass)
    # Every flagged id gets a cheap none-tier repair attempt first...
    none_tier_repair = [call for call in repair if call["reasoning_effort"] == "none"]
    assert {idx for call in none_tier_repair for idx in call["ids"]} == {0, 1, 2}
    # ...and only the three flagged ids ever reach the escalated tier.
    escalated = [call for call in repair if call["reasoning_effort"] == "low"]
    assert {idx for call in escalated for idx in call["ids"]} == {0, 1, 2}
    timing = next(
        item for item in timings if item.get("mode") == "translation_repair_pass"
    )
    assert timing["reasoning_effort"] == "low"
    assert timing["none_tier_reasoning_effort"] == "none"
    assert timing["escalated_count"] == 3


def test_the_settled_index_from_the_base_pass_repairs_a_drifted_repeat(monkeypatch):
    """End to end: the base pass renders the same line two ways, the settled
    index built from its own output (see `global_glossary`) picks the majority
    rendering, and the repair pass both sees that rendering in its prompt and
    fixes the cue that drifted from it - with no model call involved in
    building the index itself."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "気持ちいい…"},
        {"start": 1.0, "end": 2.0, "text": "気持ちいい…"},
        {"start": 2.0, "end": 3.0, "text": "気持ちいい…"},
        {"start": 3.0, "end": 4.0, "text": "こんにちは。"},
        {"start": 4.0, "end": 5.0, "text": "こんにちは。"},
    ]
    base = {0: "好舒服…", 1: "好舒服…", 2: "很舒服…", 3: "你好。", 4: "你好。"}
    repair_prompts: list[str] = []

    def fake_chat(messages, expected_count=0, reasoning_effort=None, **_kwargs):
        ids = _requested_ids_from_messages(messages)
        is_repair = "【翻译修复任务】" in messages[1]["content"]
        if is_repair:
            repair_prompts.append(messages[1]["content"])
            values = {2: "好舒服…"}
        else:
            values = base
        return json.dumps(
            {"translations": [{"id": idx, "text": values[idx]} for idx in ids]},
            ensure_ascii=False,
        )

    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 5)
    monkeypatch.setattr(translator, "TRANSLATION_PREFIX_WARMUP", False)

    zh_texts, _timings, _retry_events = translator.translate_segments(
        segments,
        max_workers=1,
        cache_path="",
        reasoning_effort="none",
    )

    assert zh_texts == ["好舒服…", "好舒服…", "好舒服…", "你好。", "你好。"]
    assert repair_prompts, "the drifted cue should have triggered a repair request"
    assert "気持ちいい…-好舒服…" in repair_prompts[0]


def test_the_repair_pass_splits_an_invalid_large_reply_instead_of_repeating_it(
    monkeypatch,
):
    segments = [
        {"start": float(idx), "end": float(idx + 1), "text": f"日本語です{idx}"}
        for idx in range(4)
    ]
    repair_request_sizes: list[int] = []

    def fake_chat(messages, expected_count=0, reasoning_effort=None, **_kwargs):
        ids = _requested_ids_from_messages(messages)
        is_repair = "【翻译修复任务】" in messages[1]["content"]
        if not is_repair:
            assert reasoning_effort == "none"
            return json.dumps(
                {
                    "translations": [
                        {"id": idx, "text": segments[idx]["text"]} for idx in ids
                    ]
                },
                ensure_ascii=False,
            )
        repair_request_sizes.append(len(ids))
        if len(ids) > 1:
            return "not json"
        return json.dumps(
            {"translations": [{"id": ids[0], "text": f"中文{ids[0]}"}]},
            ensure_ascii=False,
        )

    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 4)
    monkeypatch.setattr(translator, "TRANSLATION_PREFIX_WARMUP", False)

    zh_texts, timings, retry_events = translator.translate_segments(
        segments,
        max_workers=1,
        cache_path="",
        reasoning_effort="none",
    )

    assert retry_events == []
    assert zh_texts == ["中文0", "中文1", "中文2", "中文3"]
    assert repair_request_sizes == [4, 2, 1, 1, 2, 1, 1]
    timing = next(
        item for item in timings if item.get("mode") == "translation_repair_pass"
    )
    assert timing["format_split_count"] == 3


def test_translation_repair_length_mismatch_uses_source_translation_fields():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "source": "短い",
            "translation": "这是一个明显被过度展开的中文翻译，长度远远超过原文。",
        }
    ]

    repair_ids, reasons = translator._select_translation_repair_ids(segments, [])
    context_items = translator._build_repair_context_items(
        segments,
        [],
        repair_ids,
        reasons,
    )

    assert repair_ids == [0]
    assert reasons[0] == ["length_mismatch"]
    assert context_items[0]["reason"] == ["length_mismatch"]
    assert context_items[0]["ja"] == "短い"
    assert context_items[0]["current_zh"] == "这是一个明显被过度展开的中文翻译，长度远远超过原文。"


def test_translation_repair_selects_only_length_mismatch_candidates():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "これは普通の文です。"},
        {"start": 1.0, "end": 2.0, "text": "これも普通の文です。"},
        {"start": 2.0, "end": 3.0, "text": "短い"},
    ]
    zh_texts = [
        "这是普通句子。",
        "这也是普通句子。",
        "这是一个明显被过度展开的中文翻译，长度远远超过原文。",
    ]

    repair_ids, reasons = translator._select_translation_repair_ids(segments, zh_texts)

    assert repair_ids == [2]
    assert reasons[2] == ["length_mismatch"]


def test_translation_repair_does_not_flag_literal_country_outside_sex_context():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "私の国にできた会社です。"},
        {"start": 1.0, "end": 2.0, "text": "来月から働きます。"},
    ]
    zh_texts = ["这是在我的国家成立的公司。", "下个月开始工作。"]

    repair_ids, reasons = translator._select_translation_repair_ids(segments, zh_texts)

    assert repair_ids == []
    assert reasons == {}
