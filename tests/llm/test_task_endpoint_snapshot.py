"""A task translates against one endpoint, and caches under that same one.

The backend lease froze the *instance*, which was never what varied. Model,
endpoint and credential were re-read from `os.environ` on every request, and the
settings page writes to `os.environ` while jobs are running - so this chain was
open end to end:

    the engine takes the cache identity for model A
    -> the settings panel switches to model B
    -> the worker builds and sends its request with model B
    -> B's reply is written under A's cache key
    -> the settings panel switches back to A
    -> the next run "hits cache" for A and never sends a request at all.

Every step there is ordinary use. The fix is one frozen configuration per task
(`llm.request_config`), read by request construction, the client factories, the
preflight checks, the learned `max_tokens` limits and the cache identity alike.
"""

from __future__ import annotations

import re
import threading
from types import SimpleNamespace

import pytest

from llm import backends, engine, profiles, request_config, translator
from llm import settings as llm_settings
from llm.backends import openai_compat

_BASE_URL = "https://sample.invalid/v1"
_MODEL_A = "sample-model-a"
_MODEL_B = "sample-model-b"

_MESSAGES = [
    {"role": "system", "content": "json"},
    {"role": "user", "content": '[{"id":0,"text":"あ"}]'},
]


@pytest.fixture(autouse=True)
def _task_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "auto")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", _BASE_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", _MODEL_A)
    monkeypatch.setenv("API_KEY", "sample-key")
    monkeypatch.setenv(
        "TRANSLATION_MAX_TOKENS_CACHE_PATH", str(tmp_path / "limits.json")
    )
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_MAX_IDS", 0)
    yield
    request_config.bind(None)
    request_config.bind_profile(None)
    backends._BACKEND_INSTANCES.pop("openai", None)


def _stream_for(request: dict):
    """Answer exactly the ids the request asked for, so the engine is satisfied.

    `requested_ids`, not every id in the prompt: a batch request carries the
    whole film's source as a prefix, and replying for all of it is a contract
    violation the engine rightly rejects.
    """
    text = "".join(
        part.get("text", "")
        for message in request["input"]
        for part in message["content"]
    )
    requested = re.findall(r"requested_ids\s*=\s*\[([^\]]*)\]", text)
    ids = (
        sorted({int(found) for found in re.findall(r"\d+", requested[-1])})
        if requested
        else [0]
    )
    payload = (
        '{"translations":['
        + ",".join(f'{{"id":{index},"text":"译{index}"}}' for index in ids)
        + "]}"
    )
    return iter(
        [
            SimpleNamespace(type="response.output_text.delta", delta=payload),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(output=[])),
        ]
    )


def _record_requests(monkeypatch) -> list[str]:
    sent: list[str] = []

    def fake_create_response(request):
        sent.append(request["model"])
        return _stream_for(request)

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    return sent


def _segments(count: int) -> list[dict]:
    return [
        {"start": index, "end": index + 1, "text": f"日本語 {index}"}
        for index in range(count)
    ]


def test_a_settings_change_mid_task_cannot_move_the_endpoint(monkeypatch):
    sent = _record_requests(monkeypatch)

    with backends.backend_lease("openai"):
        before = backends.task_backend().cache_identity()
        monkeypatch.setenv("LLM_MODEL_NAME", _MODEL_B)
        after = backends.task_backend().cache_identity()
        translator._chat(_MESSAGES, expected_count=1)

    assert before == after == f"openai:{_BASE_URL}:{_MODEL_A}"
    # The identity the engine keyed on and the model that answered agree. They
    # are the pair the cache is only correct for.
    assert sent == [_MODEL_A]


def test_every_batch_worker_uses_the_endpoint_the_task_started_with(monkeypatch):
    # The pool threads are the ones that actually send, and they hold nothing of
    # their own: they used to fall back to reading the live environment.
    sent = _record_requests(monkeypatch)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_a: 1)

    with backends.backend_lease("openai"):
        identity = translator._translation_model_identity()
        monkeypatch.setenv("LLM_MODEL_NAME", _MODEL_B)
        zh_texts, _timings, _retries = translator.translate_segments(
            _segments(4),
            max_workers=4,
            cache_path="",
            target_lang="简体中文",
            glossary="",
        )

    assert len(zh_texts) == 4
    assert identity == f"openai:{_BASE_URL}:{_MODEL_A}"
    assert len(sent) >= 4
    assert set(sent) == {_MODEL_A}


def test_a_learned_token_ceiling_is_filed_under_the_endpoint_that_refused_it(
    monkeypatch,
):
    # The same disagreement, in the other direction: a ceiling read out of a
    # refusal is a fact about the endpoint that refused it, and filing it under
    # whatever the panel names next teaches one endpoint's cap to another.
    with backends.backend_lease("openai"):
        monkeypatch.setenv("LLM_MODEL_NAME", _MODEL_B)
        monkeypatch.setattr(translator, "task_backend_name", lambda *_a, **_k: "openai")
        assert translator._endpoint_capability().identity == (_BASE_URL, _MODEL_A)


def test_the_cache_key_names_the_contract_that_produced_the_text(monkeypatch):
    """The prompt profile was derived twice, and the two could disagree.

    Once for the run and once, later, for the cache key - and detection reads
    configuration the settings panel rewrites. Switching to a Hy-MT2 model name
    mid-run gave `json@...` output (correctly, from the frozen endpoint) filed
    under `hymt2@hymt2-line-v1`, so the key named a contract that had never
    produced a word of it. Resolved once, from the leased backend and the
    snapshot, and passed on.
    """
    sent = _record_requests(monkeypatch)
    observed: dict[str, str] = {}
    real_run = engine.run_batched

    def capture(*args, **kwargs):
        observed["profile"] = kwargs["profile"].cache_signature()
        observed["prompt_version"] = kwargs["prompt_version"]
        return real_run(*args, **kwargs)

    monkeypatch.setattr(engine, "run_batched", capture)

    with backends.backend_lease("openai"):
        # A model name the auto-detection would route to the other contract.
        monkeypatch.setenv("LLM_MODEL_NAME", "sample-hy-mt2")
        translator.translate_segments(
            _segments(1), max_workers=1, cache_path="", target_lang="简体中文"
        )

    assert observed["profile"] == observed["prompt_version"]
    assert observed["profile"].startswith("json@")
    assert sent == [_MODEL_A]


def test_the_repair_pass_writes_under_the_base_passs_contract(monkeypatch):
    # The repair pass reaches the cache-key helpers without carrying the
    # profile, so the binding - not an argument - is what keeps its writes on
    # the keys the base pass used.
    profile = profiles.select_profile(backend="openai", model_name=_MODEL_A)

    with backends.backend_lease("openai"), request_config.bound_profile(profile):
        monkeypatch.setenv("LLM_MODEL_NAME", "sample-hy-mt2")
        assert translator._effective_prompt_version() == profile.cache_signature()

    # Outside a task the live environment is still what decides.
    assert translator._effective_prompt_version().startswith("hymt2@")


def test_a_failed_task_puts_the_binding_back(monkeypatch):
    # The binding is per-thread state that outlives the call unless the task
    # restores it, and both the request thread and the pool threads are reused.
    # A task that raised while bound would leave the *next* piece of work on
    # that thread keying its cache under a contract it never ran against - the
    # same disagreement this module exists to prevent, arriving one job later.
    outer = profiles.select_profile(backend="openai", model_name=_MODEL_A)

    def explode(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(engine, "run_batched", explode)

    with backends.backend_lease("openai"), request_config.bound_profile(outer):
        with pytest.raises(RuntimeError, match="synthetic failure"):
            translator.translate_segments(
                _segments(1), max_workers=1, cache_path="", target_lang="简体中文"
            )
        assert request_config.current_profile() is outer


def test_a_key_cleared_mid_task_does_not_fail_the_running_job(monkeypatch):
    # The preflight checks run inside the client factory as well as at queue
    # time, so they have to ask the same frozen configuration the request is
    # built from - or clearing a field in the panel fails a job that captured a
    # perfectly good one.
    sent = _record_requests(monkeypatch)

    with backends.backend_lease("openai"):
        monkeypatch.delenv("API_KEY")
        translator._chat(_MESSAGES, expected_count=1)

    assert sent == [_MODEL_A]


def test_a_worker_thread_without_the_snapshot_is_the_bug_this_prevents():
    # Pinning the propagation itself: binding is what a pool thread needs, and
    # an unbound thread falls back to the live environment on purpose - that is
    # what keeps the CLI, the queue-time preflight and direct transport tests
    # reading configuration the way they always have.
    with backends.backend_lease("openai"):
        config = request_config.current()
        assert config is not None
        seen: list[str] = []

        def worker(bind: bool) -> None:
            if bind:
                request_config.bind(config)
            seen.append(request_config.getenv("LLM_MODEL_NAME", ""))

        for bind in (True, False):
            thread = threading.Thread(target=worker, args=(bind,))
            thread.start()
            thread.join(3.0)

    assert seen == [_MODEL_A, _MODEL_A]


def test_only_the_snapshotted_keys_may_be_read_through_it():
    # A silent fallback to the live environment for a key nobody froze is
    # exactly the bug this module exists for, so asking is an error.
    with pytest.raises(ValueError):
        request_config.capture().get("TRANSLATION_BATCH_SIZE")


def test_outside_a_task_it_is_plain_getenv(monkeypatch):
    assert request_config.current() is None
    assert request_config.getenv("LLM_MODEL_NAME", "") == _MODEL_A
    monkeypatch.setenv("LLM_MODEL_NAME", _MODEL_B)
    assert request_config.getenv("LLM_MODEL_NAME", "") == _MODEL_B
