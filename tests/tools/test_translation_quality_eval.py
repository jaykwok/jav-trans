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


def test_synthetic_source_fixture_keeps_complete_negation_before_translation():
    from tools.audits.translation_quality_cases import build_cases, plan_source_cases

    raw = build_cases()
    planned = plan_source_cases(raw)
    assert len(raw) == 200 and len(planned) == 210
    targets = [case["text"] for case in planned if case["case_id"] == "split-negation"]
    assert targets == ["一緒に来るのは嫌なの？", "行きたくないわけじゃない。", "今日は少し疲れているだけ。"]


def test_local_evaluation_uses_unified_json_profile_and_native_sampling(evaluator, monkeypatch, tmp_path):
    from llm import engine, repair
    from llm.backends import llamacpp_server
    from llm.profiles.json_v3 import JsonProfile

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "LLAMACPP_GGUF_PATH", "TRANSLATION_BACKEND"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(llamacpp_server, "resolve_gguf_model_path", lambda **_kw: "synthetic.gguf")
    monkeypatch.setattr(evaluator, "baseline_modules", lambda _p: (JsonProfile(), repair, engine))
    cues = [{"text": text, "start": i * 2, "end": i * 2 + 1} for i, text in enumerate(
        ["こんにちは。", "ありがとう。", "待って。", "さようなら。"])]
    monkeypatch.setattr(evaluator, "build_cases", lambda: cues)
    monkeypatch.setattr(evaluator, "plan_source_cases", lambda rows: rows)
    calls = []

    class Backend:
        def cache_identity(self):
            return "synthetic"

        def chat_completion(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return json.dumps({
                "translations": [
                    {"id": 0, "text": "你好。"},
                    {"id": 1, "text": "谢谢。"},
                    {"id": 2, "text": "等等。"},
                    {"id": 3, "text": "再见。"},
                ]
            }, ensure_ascii=False)

    @contextmanager
    def lease(_name):
        yield Backend()

    monkeypatch.setattr(evaluator, "backend_lease", lease)
    args = SimpleNamespace(
        output=str(tmp_path / "run"), baseline="unused", backend="llamacpp", server="",
        max_usd=None, input_price=None, output_price=None, max_requests=10, max_output_tokens=1024,
        cues=4, workers=1, batch_size=16, glossary="",
        arms=["candidate"],
    )
    assert evaluator.run(args) == 0
    assert len(calls) == 1
    sent_messages, sent_kwargs = calls[0]
    assert sent_kwargs["sampling_parameters"] == {
        "temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05
    }
    assert sent_kwargs["response_format"] is not None
    assert "duration_sec" not in sent_messages[-1]["content"]
    assert "【全片字幕 JSON】" not in sent_messages[-1]["content"]
