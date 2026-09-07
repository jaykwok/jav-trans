from __future__ import annotations

import json
import threading

import pytest

from llm import semantic_review as review, settings
from llm.context import SourceContext
from llm.profiles.json_v3 import JsonProfile
from llm.session import TranslationSession
from llm.errors import TranslationCancelledError, ContentPolicyRefusalError


def _session(cache_path=""):
    return TranslationSession(
        backend_name="openai", profile=JsonProfile(), batch_size=200, max_workers=1,
        cache_path=cache_path, target_lang="简体中文", glossary="",
        character_reference="", reasoning_effort="low",
    )


def _proposal(index=0, **changes):
    return {"id": index, "issue": "negation", "source_quote": "行かない", "translation_quote": "去",
            "reason": "原文否定出行", "suggestion": "明天不去", **changes}


def _run(chat, *, texts=None, session=None, segments=None, **kwargs):
    cues = segments or [{"text": "明日は行かない", "start": 0.0, "end": 3.0}]
    return review.apply_semantic_review(
        cues, texts or ["明天去"], chat=chat, session=session or _session(),
        source_context=SourceContext.build(cues), model_identity="model-a", **kwargs,
    )


@pytest.fixture(autouse=True)
def enable(monkeypatch):
    monkeypatch.setattr(settings, "TRANSLATION_SEMANTIC_REVIEW_ENABLED", True)


def test_disabled_review_never_calls_a_model(monkeypatch):
    monkeypatch.setattr(settings, "TRANSLATION_SEMANTIC_REVIEW_ENABLED", False)
    assert _run(lambda *_a, **_k: pytest.fail("not authorised")) == (["明天去"], None)


def test_candidate_budget_spans_the_film_and_includes_controls():
    cues = [{"text": "行かない" if i % 3 else "こんにちは"} for i in range(300)]
    ids = review.select_candidates(cues, 20, sample_every=10)
    assert len(ids) == 20 and ids[-1] > 250
    assert any(cues[i]["text"] == "こんにちは" for i in ids)
    assert review.select_candidates(cues, 0) == []


def test_length_anomalies_are_candidates_for_evidence_review_only():
    cues = [{"text": "ありがとうございます。"}, {"text": "先生"}]
    assert review.select_candidates(cues, 2, translations=["谢谢", "先生"]) == [0]
    from llm.repair import _select_translation_repair_ids

    assert _select_translation_repair_ids(cues, ["谢谢", "先生"]) == ([], {})


@pytest.mark.parametrize("changes", [
    {"source_quote": "存在しない証拠"}, {"source_quote": ""},
    {"translation_quote": "没有这句话"}, {"suggestion": "行かない"},
    {"issue": "style"}, {"suggestion": "明天去"}, {"reason": ""},
])
def test_unsupported_proposals_never_reach_verification(changes):
    calls = []

    def chat(*_args, **kwargs):
        calls.append(kwargs)
        return json.dumps({"reviews": [_proposal(**changes)]}, ensure_ascii=False)

    texts, timing = _run(chat)
    assert texts == ["明天去"] and len(calls) == 1
    assert timing["accepted_count"] == 0


@pytest.mark.parametrize("verdict,quote,accepted", [
    ("b", "行かない", True), ("a", "行かない", False),
    ("equivalent", "", False), ("uncertain", "", False),
    ("b", "原文没有的内容", False),
])
def test_only_evidenced_candidate_preference_changes_the_translation(verdict, quote, accepted):
    calls = []

    def chat(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return json.dumps({"reviews": [_proposal()]}, ensure_ascii=False)
        assert "原文否定出行" not in messages[1]["content"]  # no first judge's rationale
        return json.dumps({"verdicts": [{"id": 0, "verdict": verdict, "source_quote": quote}]}, ensure_ascii=False)

    texts, timing = _run(chat)
    assert texts == (["明天不去"] if accepted else ["明天去"])
    assert timing["accepted_count"] == int(accepted)


def test_continuation_rewrite_is_committed_as_a_unit():
    cues = [
        {"text": "行きたくない", "continues_into_next": True},
        {"text": "わけじゃない", "continues_from_previous": True},
    ]
    calls = []

    def chat(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return json.dumps({"reviews": [
                _proposal(0, source_quote="行きたくない", translation_quote="不想去", suggestion="并不是"),
                _proposal(1, source_quote="わけじゃない", translation_quote="不是", suggestion="不想去"),
            ]}, ensure_ascii=False)
        return json.dumps({"verdicts": [
            {"id": 0, "verdict": "b", "source_quote": "わけじゃない"},
            {"id": 1, "verdict": "uncertain", "source_quote": ""},
        ]}, ensure_ascii=False)

    texts, timing = _run(chat, segments=cues, texts=["不想去", "不是"])
    assert texts == ["不想去", "不是"] and timing["accepted_count"] == 0


def test_corrupt_review_preserves_first_pass_and_is_reported():
    texts, timing = _run(lambda *_a, **_k: '{"reviews":[]}')
    assert texts == ["明天去"] and timing["failures"]


@pytest.mark.parametrize("exception", [TranslationCancelledError("cancel"), ContentPolicyRefusalError("refused")])
def test_terminal_errors_propagate_without_retry(exception):
    calls = []

    def chat(*_args, **_kwargs):
        calls.append(1)
        raise exception

    with pytest.raises(type(exception)):
        _run(chat)
    assert len(calls) == 1


def test_cancel_before_review_has_no_external_side_effect():
    event = threading.Event()
    event.set()
    with pytest.raises(TranslationCancelledError):
        _run(lambda *_a, **_k: pytest.fail("cancelled request"), cancel_event=event)


def test_resume_reuses_completed_review_of_the_returned_text(tmp_path):
    session = _session(str(tmp_path / "translation_cache.jsonl"))
    replies = iter([
        json.dumps({"reviews": [_proposal()]}, ensure_ascii=False),
        json.dumps({"verdicts": [{"id": 0, "verdict": "b", "source_quote": "行かない"}]}, ensure_ascii=False),
    ])
    writes = []
    texts, _ = _run(lambda *_a, **_k: next(replies), session=session, cache_writer=lambda value: writes.append(list(value)))
    assert writes[-1] == texts == ["明天不去"]
    again, timing = _run(lambda *_a, **_k: pytest.fail("review billed twice"), session=session, texts=texts)
    assert again == texts and timing["mode"] == "semantic_review_cache_hit"


@pytest.mark.parametrize("bad_row", [
    {"id": 0}, _proposal(issue="style"), _proposal(reason=[]),
    _proposal(id=True), _proposal(unexpected="field"),
    _proposal(issue="none"),
])
def test_invalid_schema_is_a_failure_and_cannot_cache_completion(tmp_path, bad_row):
    session = _session(str(tmp_path / "translation_cache.jsonl"))
    calls = []

    def chat(*_a, **_k):
        calls.append(1)
        return json.dumps({"reviews": [bad_row]}, ensure_ascii=False)

    for _ in range(2):
        texts, timing = _run(chat, session=session)
        assert texts == ["明天去"]
        assert timing["failures"] == ["RetryableTranslationFormatError"]
    assert len(calls) == 2
    assert not (tmp_path / "translation_cache.review.json").exists()


def test_group_boundary_and_budget_never_split_continuation():
    cues = [{"text": "行かない"} for _ in range(18)]
    cues[15]["continues_into_next"] = True
    cues[16]["continues_from_previous"] = True
    context = SourceContext.build(cues)
    assert review._review_groups(context, list(range(18)), 18) == [list(range(15)), [15, 16, 17]]
    assert review._review_groups(context, [16], 2) == [[15, 16]]
    assert review._review_groups(context, [16], 1) == []


def test_cache_error_during_termination_preserves_original_exception():
    cues = [{"text": "明日は行かない"} for _ in range(17)]
    calls, writes = [], []

    def chat(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            rows = [_proposal()] + [
                {"id": i, "issue": "none", "source_quote": "", "translation_quote": "", "reason": "", "suggestion": ""}
                for i in range(1, 16)
            ]
            return json.dumps({"reviews": rows}, ensure_ascii=False)
        if len(calls) == 2:
            return json.dumps({"verdicts": [{"id": 0, "verdict": "b", "source_quote": "行かない"}]}, ensure_ascii=False)
        raise TranslationCancelledError("original cancellation")

    def persist(texts):
        writes.append(list(texts))
        if len(writes) > 1:
            raise OSError("disk unavailable")

    with pytest.raises(TranslationCancelledError, match="original cancellation"):
        _run(chat, segments=cues, texts=["明天去"] * 17, cache_writer=persist)
    assert len(calls) == 3 and len(writes) == 2
    assert writes[0][0] == "明天不去"


def test_cancellation_after_response_prevents_proposal_adoption():
    event = threading.Event()

    def chat(*_a, **_k):
        event.set()
        return json.dumps({"reviews": [_proposal()]}, ensure_ascii=False)

    with pytest.raises(TranslationCancelledError):
        _run(chat, cancel_event=event)


def test_malformed_review_recovers_in_smaller_complete_groups():
    cues = [{"text": "明日は行かない"} for _ in range(4)]
    calls = []

    def chat(messages, **kwargs):
        ids = json.loads(messages[1]["content"])["requested_ids"]
        calls.append(ids)
        schema = kwargs["response_schema"]["properties"]["reviews"]
        assert schema["minItems"] == schema["maxItems"] == len(ids)
        assert schema["items"]["properties"]["id"]["enum"] == ids
        if len(calls) == 1:
            return '{"reviews":[]}'
        return json.dumps({"reviews": [
            {"id": i, "issue": "none", "source_quote": "", "translation_quote": "", "reason": "", "suggestion": ""}
            for i in ids
        ]})

    texts, timing = _run(chat, segments=cues, texts=["明天不去"] * 4)
    assert calls == [[0, 1, 2, 3], [0, 1], [2, 3]]
    assert texts == ["明天不去"] * 4
    assert timing["failures"] == [] and timing["reviewed_ids"] == list(range(4))
    assert timing["format_retries"][0]["reason"].endswith("missing id")


def test_format_recovery_has_a_shared_run_budget():
    cues = [{"text": "明日は行かない"} for _ in range(80)]
    calls = []

    def chat(*_a, **_k):
        calls.append(1)
        return '{"reviews":[]}'

    texts, timing = _run(chat, segments=cues, texts=["明天不去"] * 80)
    assert texts == ["明天不去"] * 80
    assert len(calls) <= 5 + 4
    assert timing["failures"] and timing["failure_details"]
    assert timing["reviewed_ids"] == []


def test_format_recovery_never_splits_one_continuation_unit():
    cues = [
        {"text": "明日は行かないが、", "continues_into_next": True},
        {"text": "今日は行く。", "continues_from_previous": True},
    ]
    calls = []

    def chat(*_a, **_k):
        calls.append(1)
        return '{"reviews":[]}'

    texts, timing = _run(chat, segments=cues, texts=["明天不去", "今天去"])
    assert texts == ["明天不去", "今天去"] and len(calls) == 1
    assert timing["format_retries"] == []
