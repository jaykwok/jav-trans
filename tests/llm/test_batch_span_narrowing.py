"""When a batch shape fails, ask for less - do not ask for the same thing again.

sample-b (2026-08-13) died on this. Batch 24 asked for ids 1296-1349 and got 54
objects back numbered 1297-1350: the right count, the wrong ids, i.e. the model
lost the id sequence rather than the content. The parser refuses that, and it is
right to - accepting a shifted set attaches every line to the neighbouring cue,
silently. But the retry budget then bought four more requests of exactly the
same size and shape, all of which failed the same way, and the film was lost
with 1,310 of 1,701 cues already translated.

An id shift over 54 items is a capacity symptom, so the budget now buys smaller
requests instead, which is the move the ASR stage already makes on OOM.
"""
from __future__ import annotations

import json
import threading

from llm import translator


def _segments(count: int) -> list[dict]:
    return [
        {"start": float(index), "end": float(index) + 1.0, "text": f"ja-{index}"}
        for index in range(count)
    ]


def _requested_ids(messages) -> list[int]:
    import re

    content = messages[1]["content"]
    match = re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", content)
    assert match is not None, content
    return json.loads(match.group(1))


def _reply(ids: list[int], *, shift: int = 0) -> str:
    return json.dumps(
        {"translations": [{"id": idx + shift, "text": f"zh-{idx}"} for idx in ids]},
        ensure_ascii=False,
    )


def _translate(fake_chat, monkeypatch, *, count: int, batch_size: int, **kwargs):
    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(
        translator, "_auto_translation_batch_size", lambda *_args: batch_size
    )
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda *_a, **_kw: None)
    return translator.translate_segments(
        _segments(count),
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
        **kwargs,
    )


class _ShiftsAboveSize:
    """A model that loses the id sequence once the request gets big enough."""

    def __init__(self, breaks_above: int) -> None:
        self.breaks_above = breaks_above
        self.sizes: list[int] = []
        self.lock = threading.Lock()

    def __call__(self, messages, expected_count=0, on_progress=None, **_kwargs) -> str:
        ids = _requested_ids(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        with self.lock:
            self.sizes.append(len(ids))
        return _reply(ids, shift=1 if len(ids) > self.breaks_above else 0)


class TestNarrowing:
    def test_a_shifted_id_set_halves_the_next_request(self, monkeypatch):
        model = _ShiftsAboveSize(breaks_above=4)
        zh_texts, _, _ = _translate(model, monkeypatch, count=8, batch_size=8)

        assert model.sizes == [8, 4, 4]
        assert zh_texts == [f"zh-{index}" for index in range(8)]

    def test_covering_the_rest_at_the_narrowed_span_is_progress_not_a_failure(
        self, monkeypatch
    ):
        """The second half answers only 4 of the 8 ids the batch still wants.

        Scored against the request it is complete; scored against the batch it
        looks like 4 missing. The loop has to use the batch reading, or every
        narrowed request would burn a retry and the descent could never finish.
        """
        model = _ShiftsAboveSize(breaks_above=2)
        zh_texts, _, _ = _translate(model, monkeypatch, count=8, batch_size=8)

        # 8 fails, 4 fails, then 2 at a time covers the batch.
        assert model.sizes == [8, 4, 2, 2, 2, 2]
        assert zh_texts == [f"zh-{index}" for index in range(8)]

    def test_the_span_never_grows_back_within_a_batch(self, monkeypatch):
        """Whatever made the model lose the sequence is still true.

        Growing back would re-enter the size that just failed, which is the
        behaviour this replaced.
        """
        model = _ShiftsAboveSize(breaks_above=4)
        zh_texts, _, _ = _translate(model, monkeypatch, count=16, batch_size=16)

        assert zh_texts == [f"zh-{index}" for index in range(16)]
        # 16 fails, 8 fails, then 4 covers the batch four times over - and the
        # span stays at 4 even though three of those requests succeed.
        assert model.sizes == [16, 8, 4, 4, 4, 4]

    def test_a_clean_batch_still_costs_exactly_one_request(self, monkeypatch):
        model = _ShiftsAboveSize(breaks_above=10_000)
        zh_texts, _, _ = _translate(model, monkeypatch, count=8, batch_size=8)

        assert model.sizes == [8]
        assert zh_texts == [f"zh-{index}" for index in range(8)]

    def test_the_narrowing_is_reported(self, monkeypatch):
        events: list[dict] = []
        model = _ShiftsAboveSize(breaks_above=4)
        _translate(
            model, monkeypatch, count=8, batch_size=8, on_progress=events.append
        )

        narrowings = [
            item for item in events if item.get("phase") == "batch_span_narrowed"
        ]
        assert len(narrowings) == 1
        assert narrowings[0]["from_span"] == 8
        assert narrowings[0]["to_span"] == 4
        assert "invalid batch translation id" in narrowings[0]["error"]


class TestStillFails:
    def test_a_model_that_fails_at_every_size_still_fails(self, monkeypatch):
        """Narrowing buys smaller requests, not unlimited ones."""
        import pytest

        model = _ShiftsAboveSize(breaks_above=0)
        with pytest.raises(RuntimeError):
            _translate(model, monkeypatch, count=8, batch_size=8)
        assert model.sizes, "it has to have tried"
        assert len(model.sizes) <= translator.llm_settings.TRANSLATION_BATCH_MAX_REQUESTS
