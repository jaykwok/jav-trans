"""The runaway guard: an arithmetic bound on how long a reply may get.

`TRANSLATION_MAX_TOKENS` is 384000 - a ceiling sized for API models, which
locally is no bound at all. Measured 2026-08-04 on the 300-cue benchmark, three
local GGUF models all produced repetition loops that ran to that ceiling: the
echo-mtp fine-tune hit exactly 4096 generated tokens on 21 of 25 twelve-line
batches (`truncated=0`, so context was not the limit), and even the clean
official Qwen3.5-9B did it on 5 of 25. The bound comes from the data: over 1098
clean translations the output/source character ratio topped out at 1.27.
"""
from __future__ import annotations

import json

import pytest

from llm import settings as llm_settings
from llm.profiles import get_profile
from llm.profiles.base import ProfileContext, TranslationProfile


def _segments(*texts: str) -> list[dict]:
    return [
        {"text": text, "start": float(i), "end": float(i) + 1.0}
        for i, text in enumerate(texts)
    ]


def test_a_profile_opts_out_by_default():
    class Bare(TranslationProfile):
        id = "bare"

        def build_messages(self, segments, *, ids, ctx):
            return []

        def parse_response(self, text, *, ids):
            return {}

    assert Bare().response_token_budget(_segments("あ")) is None


def test_the_budget_grows_with_the_source():
    profile = get_profile("json")
    small = profile.response_token_budget(_segments("こんばんは"))
    large = profile.response_token_budget(_segments("こんばんは" * 20))
    assert large > small


def test_the_budget_covers_a_real_batch():
    """Regression for the property that matters: the bound must never truncate
    a legitimate answer. These are twelve real cues from the benchmark, whose
    actual JSON reply was 612 characters."""
    profile = get_profile("json")
    cues = [
        "んー...ん、んっ...ちゅ、ちゅ、ちゅぅ...んー...んー...んー...んー...",
        "あっ…",
        "ごめん、ちょっと待って",
        "先輩、それ本気で言ってます？",
        "だって、そんなの聞いてないもん",
        "もう、知らない！",
        "……ずっと、こうしていたかった",
        "えっ、今なんて言った？",
        "やだ、恥ずかしいってば",
        "ありがとう。本当に、ありがとう",
        "しばらくはここにいて。ご飯は僕が持ってくるから",
        "誰があなたの相手をするかわからないわ。もしかして、お母さんかもしれない",
    ]
    budget = profile.response_token_budget(_segments(*cues))
    source_chars = sum(len(c) for c in cues)
    # The reply is a JSON object wrapping translations that are themselves
    # shorter than the source, so the budget has to clear the source by a
    # comfortable margin without approaching a repetition loop.
    assert budget > source_chars
    assert budget < source_chars * 4


def test_an_empty_batch_has_no_budget():
    assert get_profile("json").response_token_budget([]) is None


def test_the_ratio_is_the_knob(monkeypatch):
    profile = get_profile("json")
    segments = _segments("こんばんは" * 10)
    monkeypatch.setattr(llm_settings, "TRANSLATION_OUTPUT_CHAR_RATIO", 1.5)
    tight = profile.response_token_budget(segments)
    monkeypatch.setattr(llm_settings, "TRANSLATION_OUTPUT_CHAR_RATIO", 3.0)
    loose = profile.response_token_budget(segments)
    assert loose > tight


# --- the budget actually reaching a request ------------------------------------


def test_the_budget_reaches_the_backend(monkeypatch):
    """The wiring this file exists for: `engine._request_kwargs` and
    `profile.sampling()` were both dead code, so nothing the profile said about
    a request had ever reached one."""
    from llm import translator

    sent: dict = {}

    class _Backend:
        def chat_completion(self, messages, **kwargs):
            sent.update(kwargs)
            return '{"translations": []}'

    monkeypatch.setattr(translator, "selected_backend_name", lambda: "llamacpp")
    monkeypatch.setattr(translator, "get_backend", lambda name: _Backend())
    translator._chat([{"role": "user", "content": "x"}], max_tokens=777)
    assert sent["max_tokens"] == 777


def test_no_budget_falls_back_to_the_configured_ceiling(monkeypatch):
    from llm import translator

    sent: dict = {}

    class _Backend:
        def chat_completion(self, messages, **kwargs):
            sent.update(kwargs)
            return '{"translations": []}'

    monkeypatch.setattr(translator, "selected_backend_name", lambda: "llamacpp")
    monkeypatch.setattr(translator, "get_backend", lambda name: _Backend())
    translator._chat([{"role": "user", "content": "x"}])
    assert sent["max_tokens"] == translator.TRANSLATION_MAX_TOKENS


def test_a_budget_can_only_lower_the_ceiling(monkeypatch):
    """A profile asking for more than the configured ceiling does not get it -
    the setting stays the ceiling."""
    from llm import translator

    sent: dict = {}

    class _Backend:
        def chat_completion(self, messages, **kwargs):
            sent.update(kwargs)
            return '{"translations": []}'

    monkeypatch.setattr(translator, "selected_backend_name", lambda: "llamacpp")
    monkeypatch.setattr(translator, "get_backend", lambda name: _Backend())
    translator._chat(
        [{"role": "user", "content": "x"}],
        max_tokens=translator.TRANSLATION_MAX_TOKENS * 10,
    )
    assert sent["max_tokens"] == translator.TRANSLATION_MAX_TOKENS


def test_the_engine_sizes_each_request_by_its_own_segments(monkeypatch):
    """A partial reissue asks for only the missing ids, so it must not carry
    the whole batch's budget."""
    from llm import engine as engine_module

    seen: list[int] = []
    segments = _segments(*(f"これはテスト用の台詞です{i}" for i in range(6)))

    def fake_chat(messages, **kwargs):
        seen.append(kwargs["max_tokens"])
        ids = list(range(len(segments)))
        return {"translations": ids}

    profile = get_profile("json")

    def parse(text, *, ids):
        return {i: f"译文{i}" for i in ids}

    monkeypatch.setattr(profile, "parse_response", parse)

    engine_module.run_batched(
        segments,
        profile=profile,
        backend_name="llamacpp",
        chat=lambda messages, **kwargs: (seen.append(kwargs["max_tokens"]) or "{}"),
        backoff_sleep=lambda *a, **k: None,
        crash_probe=lambda: 0,
        batch_size=6,
        max_workers=1,
        api_retries=2,
        batch_repair_retries=2,
        batch_max_requests=4,
        prefix_warmup=False,
        extra_glossary="",
        full_context="",
        full_source_payload="",
        use_full_json_prefix=False,
        cache_path="",
        cache_lock=__import__("threading").Lock(),
        target_lang="简体中文",
        glossary="",
        character_reference="",
        prompt_version="test",
        model_identity="test",
        compact_system_prompt=False,
    )

    assert seen, "the engine never passed a token budget"
    expected = profile.response_token_budget(segments)
    assert seen[0] == expected


# --- the same bound, moved into the grammar --------------------------------


class TestBoundedSchema:
    """`minItems`/`maxItems` pin the count, not the size.

    That is why a runaway still satisfies the grammar: the model returns the
    right twelve objects and writes into one of them until `max_tokens` stops
    it, which leaves the reply unparseable. A `maxLength` on `text` makes the
    overflow unrepresentable instead - verified against llama-server b10256,
    where `maxLength=8` returned exactly 8 characters and still-valid JSON.
    """

    def test_a_profile_opts_out_by_default(self):
        class Bare(TranslationProfile):
            id = "bare"

            def build_messages(self, segments, *, ids, ctx):
                return []

            def parse_response(self, text, *, ids):
                return {}

        assert Bare().bounded_schema(_segments("あ")) is None

    def test_an_empty_batch_has_no_bound(self):
        assert get_profile("json").bounded_schema([]) is None

    def _limit(self, segments) -> int:
        schema = get_profile("json").bounded_schema(segments)
        return schema["properties"]["translations"]["items"]["properties"]["text"][
            "maxLength"
        ]

    def test_the_bound_follows_the_longest_line_not_the_total(self):
        """One `items` schema applies to every element, so the bound has to
        clear the longest line. Summing would scale with batch size and stop
        bounding anything."""
        long_line = "これはかなり長い台詞で、字数を稼ぐために書かれています" * 2
        one_long = self._limit(_segments("あっ", "んっ", long_line))
        many_short = self._limit(_segments(*(["あっ"] * 12)))
        assert one_long > many_short
        assert one_long == self._limit(_segments(long_line))

    def test_a_batch_of_tiny_cues_still_gets_room_to_expand(self):
        """`ん` legitimately renders as `嗯嗯嗯…`, so a bound of 1x2 characters
        would cut real output. The floor is what prevents that."""
        assert self._limit(_segments("ん", "あ")) >= 32

    def test_the_bound_clears_a_real_batch(self):
        """The property that matters: never cut a legitimate translation. Over
        1362 clean translations no output exceeded its source by more than 7
        characters, and the worst per-batch
        `max(len(translation))/max(len(source))` was 1.025."""
        cues = [
            "あっ…",
            "もう、知らない！",
            "しばらくはここにいて。ご飯は僕が持ってくるから",
            "誰があなたの相手をするかわからないわ。もしかして、お母さんかもしれない",
        ]
        limit = self._limit(_segments(*cues))
        assert limit > max(len(c) for c in cues)

    def test_the_static_schema_is_not_mutated(self):
        """The bound is per request; writing it into the shared dict would pin
        one batch's longest line onto every later batch."""
        from llm.profiles import json_v3

        before = json.dumps(json_v3.TRANSLATION_OUTPUT_SCHEMA, sort_keys=True)
        get_profile("json").bounded_schema(_segments("こんばんは" * 30))
        assert json.dumps(json_v3.TRANSLATION_OUTPUT_SCHEMA, sort_keys=True) == before

    def test_the_bound_reaches_a_local_backend(self, monkeypatch):
        from llm import translator

        sent: dict = {}

        class _Backend:
            def chat_completion(self, messages, **kwargs):
                sent.update(kwargs)
                return '{"translations": []}'

        monkeypatch.setattr(translator, "selected_backend_name", lambda: "llamacpp")
        monkeypatch.setattr(translator, "get_backend", lambda name: _Backend())
        bounded = get_profile("json").bounded_schema(_segments("こんばんは"))
        translator._chat(
            [{"role": "user", "content": "x"}], bounded_response_schema=bounded
        )
        assert sent["response_format"] == bounded

    def test_the_bound_is_withheld_from_the_openai_transports(self, monkeypatch):
        """OpenAI's strict structured-output mode validates against a fixed
        keyword allowlist with no `maxLength` for strings, so forwarding it is a
        400 at request time. The runaway it guards was only ever measured on
        local GGUF models."""
        from llm import translator

        seen: dict = {}

        def _fake_completions(messages, **kwargs):
            seen.update(kwargs)
            return '{"translations": []}'

        monkeypatch.setattr(translator, "selected_backend_name", lambda: "openai")
        monkeypatch.setattr(translator, "_llm_api_format", lambda fmt: "chat")
        monkeypatch.setattr(translator, "_chat_completions", _fake_completions)
        bounded = get_profile("json").bounded_schema(_segments("こんばんは"))
        translator._chat(
            [{"role": "user", "content": "x"}], bounded_response_schema=bounded
        )
        # The unbounded contract schema still goes out - that is how structured
        # output works on this transport. What must not go out is the length
        # bound bolted onto it.
        assert "maxLength" in json.dumps(bounded)
        assert "maxLength" not in json.dumps(seen)

    def test_the_engine_sends_the_bound_with_each_request(self, monkeypatch):
        from llm import engine as engine_module

        seen: list[dict] = []
        segments = _segments(*(f"これはテスト用の台詞です{i}" for i in range(6)))
        profile = get_profile("json")
        monkeypatch.setattr(
            profile, "parse_response", lambda text, *, ids: {i: f"译文{i}" for i in ids}
        )

        engine_module.run_batched(
            segments,
            profile=profile,
            backend_name="llamacpp",
            chat=lambda messages, **kwargs: (
                seen.append(kwargs.get("bounded_response_schema")) or "{}"
            ),
            backoff_sleep=lambda *a, **k: None,
            crash_probe=lambda: 0,
            batch_size=6,
            max_workers=1,
            api_retries=2,
            batch_repair_retries=2,
            batch_max_requests=4,
            prefix_warmup=False,
            extra_glossary="",
            full_context="",
            full_source_payload="",
            use_full_json_prefix=False,
            cache_path="",
            cache_lock=__import__("threading").Lock(),
            target_lang="简体中文",
            glossary="",
            character_reference="",
            prompt_version="test",
            model_identity="test",
            compact_system_prompt=False,
        )
        assert seen and seen[0] == profile.bounded_schema(segments)
