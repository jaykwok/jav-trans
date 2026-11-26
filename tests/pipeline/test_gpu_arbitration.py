"""ASR and the local translation model, arbitrated at the real entry points.

Two failures met here, and they were the same failure seen from either end.

Asking for the card used to *be* stopping the ASR worker:
``_release_asr_worker_vram()`` calls ``shutdown_global_worker()``, which took
the ASR module lock - and that lock was held for the whole of another job's
transcription. So a local model start blocked for as long as the other job ran,
in a wait no cancel event could reach.

From the other end, nothing stopped ASR from starting beside a *healthy* local
server. GPU admission read the cleanup ledger, which only knows about owners
that were asked to stop, so a resident multi-GB model looked exactly like a
free card.

Both are answered by the permit in `core.gpu_admission`, and these drive it
through the shipped entry points rather than through the permit itself.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import gpu_admission, resources
from llm.backends import llamacpp_server as local
from llm.errors import TranslationCancelledError
from pipeline import gpu_worker
from web import pipeline_manager


@pytest.fixture(autouse=True)
def _clean_gpu_state(monkeypatch):
    monkeypatch.setattr(resources, "_RECORDS", {})
    yield
    gpu_admission._reset_for_tests()


class _BlockingAsrWorker:
    """A transcription that runs until the test lets it finish."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = 0

    def transcribe_and_align(self, *args, **kwargs):
        self.entered.set()
        assert self.release.wait(5.0)
        return [], [], {}

    def is_alive(self) -> bool:
        return True

    def has_unreleased_child(self) -> bool:
        return False

    def close(self) -> bool:
        self.closed += 1
        return True


def _start_asr(worker: _BlockingAsrWorker, monkeypatch) -> threading.Thread:
    monkeypatch.setattr(gpu_worker, "_GLOBAL_WORKER", worker)
    monkeypatch.setattr(gpu_worker, "_get_global_worker", lambda: worker)
    thread = threading.Thread(
        target=lambda: gpu_worker.transcribe_and_align("sample-a.wav"),
        daemon=True,
    )
    thread.start()
    assert worker.entered.wait(3.0)
    return thread


def test_cancel_reaches_a_local_start_waiting_for_a_running_asr(monkeypatch):
    worker = _BlockingAsrWorker()
    asr_thread = _start_asr(worker, monkeypatch)
    monkeypatch.setattr(local, "resolve_server_executable", lambda: Path("sample-server.exe"))
    monkeypatch.setattr(
        local, "resolve_gguf_model_path", lambda **_kwargs: Path("sample-model.gguf")
    )
    monkeypatch.setattr(local, "_pick_free_port", lambda: 12345)

    backend = local.LlamaCppServerBackend()
    cancel = threading.Event()
    outcome: list[str] = []
    started = threading.Event()

    def start_local() -> None:
        started.set()
        try:
            backend._ensure_server(cancel)
        except TranslationCancelledError:
            outcome.append("cancelled")
        except BaseException as exc:  # pragma: no cover - would be a real failure
            outcome.append(repr(exc))

    local_thread = threading.Thread(target=start_local, daemon=True)
    local_thread.start()
    assert started.wait(3.0)

    cancel.set()
    local_thread.join(3.0)

    assert local_thread.is_alive() is False
    assert outcome == ["cancelled"]
    # The healthy ASR worker was never touched: cancelling a wait is not a
    # licence to stop what is running, and the transcription is still going.
    assert worker.closed == 0
    assert worker.release.is_set() is False

    worker.release.set()
    asr_thread.join(5.0)


def test_asr_does_not_start_beside_a_healthy_local_server(monkeypatch):
    resources.register(
        resource_id=resources.new_resource_id("llamacpp-server"),
        generation=1,
        kind=gpu_admission.LOCAL_LLM,
        description="llama-server",
        process=SimpleNamespace(pid=4242),
        stopper=lambda process, handle: resources.StopOutcome(process_state="gone"),
        holds_gpu=True,
    )

    # Nothing was asked to stop, so the cleanup ledger is clear - and that is
    # exactly the state a job used to be dispatched in.
    assert resources.gpu_blocked() is False
    assert pipeline_manager.gpu_admission_blocked() is True

    worker = _BlockingAsrWorker()
    monkeypatch.setattr(gpu_worker, "_GLOBAL_WORKER", worker)
    monkeypatch.setattr(gpu_worker, "_get_global_worker", lambda: worker)
    cancel = threading.Event()
    outcome: list[str] = []

    def run_asr() -> None:
        try:
            gpu_worker.transcribe_and_align(
                "sample-a.wav", cancel_requested=cancel.is_set
            )
        except gpu_worker.GpuWorkerError as exc:
            outcome.append(exc.kind)

    thread = threading.Thread(target=run_asr, daemon=True)
    thread.start()
    assert worker.entered.wait(0.4) is False  # held at the door, not transcribing

    cancel.set()
    thread.join(3.0)
    assert outcome == ["cancelled"]
    assert worker.entered.is_set() is False
