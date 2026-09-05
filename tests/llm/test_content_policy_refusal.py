"""A provider content filter is a verdict on the input, not a bad reply.

`cyber_policy` used to arrive as `RetryableTranslationFormatError`, so the
batch loop reissued the same text four times and halved the request span twice
on the way. It cannot work: the filter reads what was sent, and narrowing the
span sends a subset of the same thing. On 2026-09-04 that cost 21 minutes and
four batches' worth of reasoning tokens before the film failed anyway.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm import translator
from llm.backends import openai_compat
from llm.errors import ContentPolicyRefusalError, RetryableTranslationFormatError


def _failed_event(code: str, message: str = "refused"):
    return SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(
            error=SimpleNamespace(code=code, message=message),
        ),
    )


def _segments(count: int) -> list[dict]:
    return [
        {"start": float(index), "end": float(index) + 1.0, "text": f"ja-{index}"}
        for index in range(count)
    ]


def test_cyber_policy_is_terminal_not_retryable(monkeypatch):
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: iter([_failed_event("cyber_policy", "内容不合规")]),
    )

    with pytest.raises(ContentPolicyRefusalError) as raised:
        translator._chat(
            [
                {"role": "system", "content": "json"},
                {"role": "user", "content": "translate"},
            ],
            expected_count=1,
        )

    assert not isinstance(raised.value, RetryableTranslationFormatError)
    assert "cyber_policy" in str(raised.value)
    assert "内容不合规" in str(raised.value)


def test_other_failure_codes_stay_retryable(monkeypatch):
    # The narrow list is the point: a transport hiccup misfiled as terminal
    # would kill films that the repair loop recovers today.
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: iter([_failed_event("server_error", "upstream blip")]),
    )

    with pytest.raises(RetryableTranslationFormatError):
        translator._chat(
            [
                {"role": "system", "content": "json"},
                {"role": "user", "content": "translate"},
            ],
            expected_count=1,
        )


def test_refused_batch_fails_once_and_names_its_cues(monkeypatch):
    calls: list[int] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        calls.append(expected_count)
        raise ContentPolicyRefusalError("refused (code=cyber_policy)")

    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    with pytest.raises(ContentPolicyRefusalError) as raised:
        translator.translate_segments(
            _segments(2),
            max_workers=1,
            cache_path="",
            target_lang="简体中文",
            glossary="",
        )

    # One request, no span narrowing, no backoff.
    assert calls == [2]
    message = str(raised.value)
    assert "cyber_policy" in message
    assert "batch=0" in message
    assert "requested_ids=[0, 1]" in message
