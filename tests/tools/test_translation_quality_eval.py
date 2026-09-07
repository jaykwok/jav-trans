from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def evaluator(monkeypatch):
    # Importing the CLI for tests must not read or change local credentials.
    from core import config
    monkeypatch.setattr(config, "load_config", lambda: None)
    return importlib.import_module("tools.audits.run_translation_quality_eval")


def test_paid_budget_is_reserved_before_concurrent_requests(evaluator):
    budget = evaluator.Budget(maximum=0.01, input_price=1.0, output_price=1.0, requests=20)

    def reserve(_index):
        try:
            return budget.reserve([{"role": "user", "content": "synthetic"}], 128, paid=True)
        except evaluator.SpendLimit:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(20)))
    assert [value for value in results if value is not None] == [1]
    assert budget.requests == 1 and budget.reserved <= budget.maximum


def test_request_limit_also_applies_to_free_local_experiments(evaluator):
    budget = evaluator.Budget(maximum=0.0, input_price=0.0, output_price=0.0, requests=1)
    budget.reserve([], 128, paid=False)
    with pytest.raises(evaluator.SpendLimit):
        budget.reserve([], 128, paid=False)


def test_review_can_reuse_a_synthetic_translation_without_buying_a_first_pass(evaluator, monkeypatch, tmp_path):
    from llm import engine, repair, settings
    from llm.backends import openai_compat
    from llm.profiles.hymt2 import HyMt2Profile
    from llm.profiles.json_v3 import JsonProfile

    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    monkeypatch.setattr(settings, "TRANSLATION_SEMANTIC_REVIEW_ENABLED", False)
    monkeypatch.setattr(settings, "TRANSLATION_SEMANTIC_REVIEW_MAX_IDS", 80)
    # Track CLI changes to the client factory so pytest restores them.
    monkeypatch.setattr(openai_compat, "_make_async_client", openai_compat._make_async_client)
    monkeypatch.setattr(evaluator, "baseline_modules", lambda _path: (JsonProfile(), HyMt2Profile(), repair, engine))
    monkeypatch.setattr(engine, "run_batched", lambda *_a, **_k: pytest.fail("a reused first pass was billed"))
    calls = []

    class Backend:
        def cache_identity(self):
            return "test-model"

        def chat_completion(self, messages, **_kwargs):
            calls.append(messages)
            return json.dumps({"reviews": [{
                "id": 0, "issue": "none", "source_quote": "", "translation_quote": "", "reason": "", "suggestion": "",
            }]})

    @contextmanager
    def lease(_name):
        yield Backend()

    monkeypatch.setattr(evaluator, "backend_lease", lease)
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "synthetic_only": True, "cases": [{"text": "明日は行かない", "start": 0.0, "end": 2.0}],
        "arms": {"candidate": {"after_repair": ["明天不去"]}},
    }), encoding="utf-8")
    args = SimpleNamespace(
        output=str(tmp_path / "review"), baseline="unused", backend="api", priced_model="test-model",
        max_usd=0.02, input_price=0.1, output_price=0.2, max_requests=2, max_output_tokens=8192,
        cues=4, workers=1, batch_size=40, review_limit=8, glossary="", server="",
        arms=["candidate_review"], review_fixture=False, review_from=str(previous), reuse_arm="candidate",
    )
    assert evaluator.run(args) == 0
    assert len(calls) == 1
    result = json.loads((tmp_path / "review/report.json").read_text(encoding="utf-8"))
    assert result["arms"]["candidate_review"]["texts"] == ["明天不去"]
    assert "first_pass" not in result["arms"]["candidate_review"]
    assert result["requests"] == 1


def test_synthetic_source_fixture_keeps_complete_negation_before_translation():
    from tools.audits.translation_quality_cases import build_cases, plan_source_cases

    raw = build_cases()
    planned = plan_source_cases(raw)
    assert len(raw) == 200 and len(planned) == 210
    targets = [case["text"] for case in planned if case["case_id"] == "split-negation"]
    assert targets == ["一緒に来るのは嫌なの？", "行きたくないわけじゃない。", "今日は少し疲れているだけ。"]
