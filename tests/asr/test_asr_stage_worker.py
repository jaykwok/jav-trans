from pathlib import Path
from types import SimpleNamespace

import main
import pytest
from asr import pipeline as asr_pipeline
from helpers import make_job_context
from pipeline import audio as pipeline_audio
from pipeline import gpu_worker


@pytest.fixture(autouse=True)
def _isolate_project_root(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)


def test_asr_details_cuda_skip_reason_always_skips_in_main():
    # The main/web process must never own CUDA -- not even is_available() --
    # including for asr_details loaded from a stale cache that lacks the
    # stage_worker tag (which would otherwise re-split the CUDA context).
    assert main._asr_details_cuda_skip_reason(None)
    assert main._asr_details_cuda_skip_reason({})
    assert main._asr_details_cuda_skip_reason({"stage_worker": {"mode": "gpu_worker"}})
    assert main._asr_details_cuda_skip_reason({"stage_worker": {"mode": "subprocess"}})
    assert main._asr_details_cuda_skip_reason({"stage_worker": {"mode": "unknown"}})


def test_vram_budget_enforced_on_allocated_not_reserved(monkeypatch):
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "5600")
    # Reserved over budget but allocated under it: must NOT raise. The caching
    # allocator's reserved pool routinely fills dedicated VRAM on a 6GB card
    # without spilling, so reserved must not trip the budget.
    asr_pipeline._enforce_vram_budget_from_snapshot(
        {"stage": "x", "max_reserved_mb": 6200.0, "max_allocated_mb": 4800.0}
    )
    # Allocated over budget: raises (classified as OOM upstream for retry).
    with pytest.raises(RuntimeError, match="GPU VRAM budget exceeded"):
        asr_pipeline._enforce_vram_budget_from_snapshot(
            {"stage": "x", "max_reserved_mb": 6200.0, "max_allocated_mb": 5800.0}
        )


def test_shared_vram_jitter_below_tolerance_is_not_spill(monkeypatch):
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "5600")
    # The PDH counter jitters by a few MB from driver staging buffers; a 4MB
    # delta once killed a fully finished transcription pass three times in a
    # row. Growth below the tolerance must NOT raise.
    asr_pipeline._enforce_vram_budget_from_snapshot(
        {
            "stage": "asr_text_transcribe_done",
            "shared_vram_mb": 4.0,
            "physical_ram_used_mb": 1000.0,
            "physical_ram_budget_mb": 15000.0,
            "max_allocated_mb": 4800.0,
        }
    )


def test_shared_vram_growth_above_tolerance_is_soft_oom(monkeypatch):
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "5600")
    base = {
        "stage": "split_done",
        "total_mb": 8188.0,
        "physical_ram_used_mb": 1000.0,
        "physical_ram_budget_mb": 15000.0,
        "max_allocated_mb": 4800.0,
    }
    with pytest.raises(RuntimeError, match="shared VRAM spill"):
        asr_pipeline._enforce_vram_budget_from_snapshot(
            {**base, "shared_vram_mb": 65.0}
        )


def test_shared_vram_spill_tolerance_env_override(monkeypatch):
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "5600")
    monkeypatch.setenv("ASR_SHARED_VRAM_SPILL_TOLERANCE_MB", "0")
    with pytest.raises(RuntimeError, match="shared VRAM spill"):
        asr_pipeline._enforce_vram_budget_from_snapshot(
            {
                "stage": "split_done",
                "shared_vram_mb": 0.001,
                "physical_ram_used_mb": 1000.0,
                "physical_ram_budget_mb": 15000.0,
                "max_allocated_mb": 4800.0,
            }
        )


def test_enforce_vram_budget_fails_fast_on_shared_vram_spill(monkeypatch):
    from types import SimpleNamespace
    from asr import transcribe
    from pipeline import memory_safety

    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "7778")
    monkeypatch.setenv("ASR_SHARED_VRAM_SPILL_TOLERANCE_MB", "64")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            max_memory_allocated=lambda _idx: 5600 * 1024 * 1024,
            memory_allocated=lambda _idx: 3910 * 1024 * 1024,
            max_memory_reserved=lambda _idx: 6400 * 1024 * 1024,
            memory_reserved=lambda _idx: 4200 * 1024 * 1024,
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(
        memory_safety,
        "runtime_memory_snapshot",
        lambda **_kwargs: {
            "shared_vram_mb": 86.0,
            "shared_vram_raw_mb": 162.0,
            "shared_vram_baseline_mb": 76.0,
        },
    )
    warnings: list[str] = []
    with pytest.raises(RuntimeError, match="GPU shared VRAM spill detected"):
        transcribe._enforce_vram_budget("ASR 文本转写_batch_1", warnings.append)
    assert any("shared VRAM spill" in msg for msg in warnings)


def test_stage_worker_emits_heartbeat_while_provider_is_silent(monkeypatch, tmp_path):
    import time
    from types import SimpleNamespace

    class FakeConnection:
        def __init__(self):
            self.poll_count = 0

        def poll(self, _wait_s):
            self.poll_count += 1
            if self.poll_count == 1:
                time.sleep(0.02)
                return False
            return True

        def recv(self):
            return {
                "op": "result",
                "job_id": "heartbeat-job",
                "segments": [],
                "asr_log": [],
                "asr_details": {},
            }

    client = gpu_worker._GpuWorkerClient()
    client._conn = FakeConnection()
    client._process = SimpleNamespace(is_alive=lambda: True, exitcode=None)
    monkeypatch.setattr(client, "_send_request", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("ASR_STAGE_WORKER_HEARTBEAT_S", "0.01")
    messages: list[str] = []

    client._transcribe_and_align_once(
        str(tmp_path / "audio.wav"),
        job_id="heartbeat-job",
        on_stage=messages.append,
    )

    assert any(message.startswith("阶段心跳 ") for message in messages)


class _FakeChild:
    """A child that answers is_alive()/terminate()/kill() like mp.Process."""

    def __init__(self, *, refuse_terminate=False, immortal=False):
        self.alive = True
        self.pid = 4321
        self.exitcode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.joins: list[float | None] = []
        self._refuse_terminate = refuse_terminate
        self._immortal = immortal

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_calls += 1
        if self._refuse_terminate:
            raise PermissionError("Access is denied")
        if not self._immortal:
            self.alive = False

    def kill(self):
        self.kill_calls += 1
        if not self._immortal:
            self.alive = False

    def join(self, timeout=None):
        self.joins.append(timeout)


def test_pipe_eof_waits_for_exit_before_cleaning_up(monkeypatch, tmp_path):
    class ExitingChild(_FakeChild):
        def join(self, timeout=None):
            super().join(timeout)
            self.exitcode = 23
            self.alive = False

    class BrokenConnection:
        stage_sent = False

        def poll(self, _wait_s):
            return True

        def recv(self):
            if self.stage_sent:
                raise EOFError
            self.stage_sent = True
            return {"op": "stage", "job_id": "job", "message": "ASR batch started"}

        def close(self):
            pass

    child = ExitingChild()
    client = gpu_worker._GpuWorkerClient()
    client._process = child
    client._conn = BrokenConnection()
    monkeypatch.setattr(client, "_send_request", lambda *_args: None)

    with pytest.raises(gpu_worker.GpuWorkerError) as error:
        client._transcribe_and_align_once(str(tmp_path / "sample-a.wav"), job_id="job")

    assert error.value.kind == "crash"
    assert "pid=4321 exitcode=23" in error.value.detail
    assert "last_step='ASR batch started'" in error.value.detail
    assert "worker_log=logs/asr-worker-4321.log" in error.value.detail
    assert child.joins == [gpu_worker._EOF_EXIT_GRACE_S]
    assert child.terminate_calls == 0
    assert client._process is None


@pytest.mark.parametrize("disconnected_again", [False, True])
def test_disconnected_worker_is_restarted_only_once(monkeypatch, disconnected_again):
    client = gpu_worker._GpuWorkerClient()
    messages = []
    attempts = []

    def run(*_args, **kwargs):
        attempts.append(kwargs["env_overrides"])
        if len(attempts) == 1 or disconnected_again:
            raise gpu_worker._GpuWorkerDisconnectedError("pipe closed")
        return [], [], {}

    monkeypatch.setattr(client, "_transcribe_and_align_once", run)
    if disconnected_again:
        with pytest.raises(gpu_worker.GpuWorkerError, match="pipe closed"):
            client.transcribe_and_align("sample-a.wav", on_stage=messages.append)
    else:
        assert client.transcribe_and_align("sample-a.wav", on_stage=messages.append) == ([], [], {})
    assert len(attempts) == 2
    assert len(messages) == 1 and "pipe closed" in messages[0]


@pytest.mark.parametrize("fail_result_send", [False, True])
def test_worker_publishes_before_chunk_cleanup_and_reports_send_failure(
    monkeypatch, caplog, fail_result_send,
):
    events = []

    class ScriptedConnection:
        def poll(self, _wait_s):
            return not events or events[-1] != "cleanup"

        def recv(self):
            return {
                "op": "transcribe_and_align",
                "job_id": "job",
                "audio_path": "sample-a.wav",
                "env": {"ASR_STAGE_WORKER_MOCK": "1"},
            }

        def send(self, message):
            if message["op"] == "result":
                events.append("result")
                if fail_result_send:
                    raise OSError("injected result send failure")
            if message["op"] == "error":
                events.append(message["kind"])
                assert message["stage"] == "result_send"
                assert "injected result send failure" in message["detail"]

    def run(_manager, *, defer_cleanup, **_kwargs):
        defer_cleanup(lambda: events.append("cleanup"))
        return [], [], {}

    monkeypatch.setattr(gpu_worker, "configure_ffmpeg_shared_runtime", lambda: ())
    monkeypatch.setattr(gpu_worker, "_adaptive_runtime_tuning", lambda *_a, **_kw: {})
    monkeypatch.setattr(gpu_worker, "_clear_worker_cuda", lambda: None)
    monkeypatch.setattr(gpu_worker.GpuModelManager, "run_transcribe_and_align", run)
    if fail_result_send:
        with pytest.raises(SystemExit) as error:
            gpu_worker._serve_worker(ScriptedConnection())
        assert error.value.code == 0
        assert events == ["result", "result_send_failed", "cleanup"]
        assert "injected result send failure" in caplog.text
    else:
        gpu_worker._serve_worker(ScriptedConnection())
        assert events == ["result", "cleanup"]


def test_cancel_raised_by_the_heartbeat_callback_still_kills_the_child(
    monkeypatch,
    tmp_path,
):
    # Cancellation is checked at the top of the wait loop, but the heartbeat
    # callback re-checks it too (main._on_stage raises). A cancel landing
    # between the two used to leave the loop through the heartbeat, where -
    # unlike every other branch - nothing killed the GPU child: the job was
    # reported cancelled while the worker kept transcribing on the card.
    import time as _time

    class FakeConnection:
        def poll(self, _wait_s):
            _time.sleep(0.02)
            return False

        def recv(self):  # pragma: no cover - never reached
            raise AssertionError("no message should be read")

    client = gpu_worker._GpuWorkerClient()
    client._conn = FakeConnection()
    client._process = SimpleNamespace(is_alive=lambda: True, exitcode=None)
    monkeypatch.setattr(client, "_send_request", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("ASR_STAGE_WORKER_HEARTBEAT_S", "0.01")
    killed = {"count": 0}

    def fake_kill(self):
        del self
        killed["count"] += 1
        return True

    monkeypatch.setattr(gpu_worker._GpuWorkerClient, "_kill_child", fake_kill)

    def cancelling_on_stage(_message):
        raise main.PipelineCancelledError("任务已取消")

    with pytest.raises(main.PipelineCancelledError):
        client._transcribe_and_align_once(
            str(tmp_path / "audio.wav"),
            job_id="heartbeat-cancel",
            on_stage=cancelling_on_stage,
        )

    assert killed["count"] == 1


def test_kill_child_escalates_when_terminate_is_refused():
    # terminate() raising used to skip kill() entirely (one try block, one
    # swallowed exception) *after* the handles had already been cleared, so the
    # caller believed the GPU was free while the child was still running.
    client = gpu_worker._GpuWorkerClient()
    child = _FakeChild(refuse_terminate=True)
    client._process = child

    assert client._kill_child() is True
    assert child.terminate_calls == 1
    assert child.kill_calls == 1
    assert client._process is None


def test_kill_child_closes_the_job_object_handle(monkeypatch):
    # The Job Object handle is a raw integer; setting the attribute to None
    # leaked it, and on Windows the kill-on-close job is the only thing that
    # reaches a child's own grandchildren.
    closed: list[int] = []
    monkeypatch.setattr(
        gpu_worker,
        "close_kill_on_close_job",
        lambda handle: bool(closed.append(handle)) or True,
    )
    client = gpu_worker._GpuWorkerClient()
    client._process = _FakeChild()
    client._job_handle = 987654

    assert client._kill_child() is True
    assert closed == [987654]
    assert client._job_handle is None


def test_child_that_survives_the_kill_blocks_a_replacement_worker(monkeypatch):
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: True)
    client = gpu_worker._GpuWorkerClient()
    child = _FakeChild(immortal=True)
    client._process = child

    assert client._kill_child() is False
    # The client keeps the only reference to a live GPU process...
    assert client._process is child
    assert client.has_unreleased_child()
    # ...so nothing may start a second worker beside it.
    with pytest.raises(gpu_worker.GpuWorkerError) as exc_info:
        client._start_worker()
    assert exc_info.value.kind == "cleanup_failed"


def test_global_worker_is_not_replaced_while_a_child_survives(monkeypatch):
    client = gpu_worker._GpuWorkerClient()
    client._process = _FakeChild(immortal=True)
    monkeypatch.setattr(gpu_worker, "_GLOBAL_WORKER", client)

    assert gpu_worker._get_global_worker() is client


def test_worker_startup_wait_is_cancellable(monkeypatch):
    # Model load takes tens of seconds; without a cancel check here the user's
    # 取消 was only noticed after the worker came up and the job was sent.
    client = gpu_worker._GpuWorkerClient()
    monkeypatch.setattr(client, "_kill_child", lambda: True)

    class FakeConnection:
        def poll(self, _wait_s):
            return False

        def close(self):
            return None

    def fake_process_factory(*_args, **_kwargs):
        return SimpleNamespace(
            start=lambda: None,
            is_alive=lambda: True,
            exitcode=None,
            pid=1234,
        )

    monkeypatch.setattr(
        client,
        "_ctx",
        SimpleNamespace(
            Pipe=lambda duplex=True: (FakeConnection(), FakeConnection()),
            Process=fake_process_factory,
        ),
    )

    with pytest.raises(gpu_worker.GpuWorkerError) as exc_info:
        client._start_worker(lambda: True)

    assert exc_info.value.kind == "cancelled"


def test_physical_ram_ratio_is_hard_oom(monkeypatch):
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "5600")
    with pytest.raises(RuntimeError, match="Physical RAM budget exceeded"):
        asr_pipeline._enforce_vram_budget_from_snapshot(
            {
                "stage": "asr_batch",
                "shared_vram_mb": 0.0,
                "physical_ram_used_mb": 15201.0,
                "physical_ram_budget_mb": 15200.0,
                "max_allocated_mb": 4800.0,
            }
        )


def test_ram_oom_is_not_gpu_batch_retry():
    error = RuntimeError("Physical RAM budget exceeded: used_mb=10 budget_mb=9")
    assert gpu_worker._is_ram_oom_error(error)
    assert not gpu_worker._is_oom_error(error, None)


def test_auto_vram_budget_and_batch_scale_from_physical_memory():
    allocator_calls: list[tuple[float, int]] = []

    class _Properties:
        total_memory = 8 * 1024 * 1024 * 1024

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(_index):
            return _Properties()

        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            allocator_calls.append((fraction, device))

    class _Torch:
        cuda = _Cuda()

    env = {
        "ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        "ASR_BATCH_SIZE": "auto",
        "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "auto",
        "ASR_STAGE_WORKER_VRAM_RATIO": "0.95",
    }
    with gpu_worker._temporary_env(env):
        tuning = gpu_worker._adaptive_runtime_tuning(_Torch(), env)
        assert float(gpu_worker.os.environ["ASR_STAGE_WORKER_VRAM_BUDGET_MB"]) == pytest.approx(
            8192 * 0.95,
            abs=0.1,
        )
        # Formula directly selects optimal 8 GB batch (10) on Job 1 without
        # starting at a conservative heuristic of 5 and climbing across jobs.
        assert gpu_worker.os.environ["ASR_BATCH_SIZE"] == "10"

    assert tuning["physical_vram_mb"] == pytest.approx(8192.0)
    assert tuning["vram_budget_source"] == "physical_vram_ratio"
    assert tuning["vram_allocator_fraction"] == pytest.approx(0.95)
    assert len(allocator_calls) == 1
    assert allocator_calls[0][0] == pytest.approx(0.95)
    assert allocator_calls[0][1] == 0
    assert tuning["asr_batch_source"] == "formula_vram"
    assert tuning["asr_vram_plan"]["batch_size"] == 10


def test_gpu_profile_identity_distinguishes_same_model_cards_by_uuid():
    class _Cuda:
        uuid = "71339454-9251-d059-72b3-118778b30f38"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @classmethod
        def get_device_properties(cls, _index):
            return SimpleNamespace(
                uuid=cls.uuid,
                major=8,
                minor=9,
                pci_domain_id=0,
                pci_bus_id=43,
                pci_device_id=0,
            )

    class _Torch:
        cuda = _Cuda()

    first = gpu_worker._gpu_hardware_identity(
        _Torch(),
        device_name="NVIDIA GeForce RTX 4060 Ti",
        total_mb=8187.5,
    )
    _Cuda.uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    second = gpu_worker._gpu_hardware_identity(
        _Torch(),
        device_name="NVIDIA GeForce RTX 4060 Ti",
        total_mb=8187.5,
    )

    assert first["fingerprint_source"] == "cuda_uuid"
    assert first["fingerprint"].startswith("sha256:")
    assert first["fingerprint"] != second["fingerprint"]
    assert "71339454" not in str(first)


def test_gpu_profile_identity_falls_back_to_pci_without_persisting_location():
    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(_index):
            return SimpleNamespace(
                major=8,
                minor=9,
                pci_domain_id=0,
                pci_bus_id=43,
                pci_device_id=0,
            )

    class _Torch:
        cuda = _Cuda()

    identity = gpu_worker._gpu_hardware_identity(
        _Torch(),
        device_name="NVIDIA GeForce RTX 4060 Ti",
        total_mb=8187.5,
    )

    assert identity["fingerprint_source"] == "pci_location"
    assert identity["compute_capability"] == "8.9"
    assert "43" not in identity["fingerprint"]


def test_adaptive_runtime_tuning_accounts_for_live_free_and_resident_vram():
    class _Properties:
        total_memory = 8188 * 1024 * 1024

    class _Cuda:
        free_mb = 6800
        reserved_mb = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(_index):
            return _Properties()

        @classmethod
        def mem_get_info(cls, _index):
            return (cls.free_mb * 1024 * 1024, 8188 * 1024 * 1024)

        @classmethod
        def memory_reserved(cls, _index):
            return cls.reserved_mb * 1024 * 1024

        @staticmethod
        def set_per_process_memory_fraction(_fraction, _device):
            pass

    class _Torch:
        cuda = _Cuda()

    env = {
        "ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        "ASR_BATCH_SIZE": "auto",
        "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "auto",
    }
    # Fresh worker with 6800 MiB free -> picks batch 5..6 (leaves CUDA/cuDNN
    # workspace + WDDM headroom so it does not spill to shared VRAM).
    with gpu_worker._temporary_env(env):
        fresh = gpu_worker._adaptive_runtime_tuning(_Torch(), env)
    # Resident worker holding 4868 MiB of its own weights -> free drops to 1932 MiB,
    # but adding back own_reserved yields the exact same process-available VRAM & batch!
    _Cuda.free_mb = 1932
    _Cuda.reserved_mb = 4868
    with gpu_worker._temporary_env(env):
        resident = gpu_worker._adaptive_runtime_tuning(_Torch(), env)
    assert resident["asr_batch_size"] == fresh["asr_batch_size"]
    assert 5 <= fresh["asr_batch_size"] <= 7

    # Heavy external desktop load (only 5400 MiB free, 0 own reserved) -> steps down safely.
    _Cuda.free_mb = 5400
    _Cuda.reserved_mb = 0
    with gpu_worker._temporary_env(env):
        loaded = gpu_worker._adaptive_runtime_tuning(_Torch(), env)
    assert 1 <= loaded["asr_batch_size"] < fresh["asr_batch_size"]


@pytest.mark.parametrize(
    ("backend", "minimum_mb"),
    [
        ("jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf", 7168),
    ],
)
def test_repo_physical_vram_floor_accepts_exact_minimum(backend, minimum_mb):
    result = gpu_worker._enforce_min_physical_vram(
        total_mb=minimum_mb,
        env={"ASR_BACKEND": backend},
    )

    assert result["repo_id"] == backend
    assert result["minimum_physical_vram_mb"] == minimum_mb


@pytest.mark.parametrize(
    ("backend", "minimum_mb"),
    [
        ("jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf", 7168),
    ],
)
def test_repo_physical_vram_floor_rejects_lower_card_without_fallback(
    backend,
    minimum_mb,
):
    with pytest.raises(RuntimeError, match="CPU fallback is disabled") as exc_info:
        gpu_worker._enforce_min_physical_vram(
            total_mb=minimum_mb - 1,
            env={
                "ASR_BACKEND": backend,
                "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "99999",
            },
        )

    detail = str(exc_info.value)
    assert "Shared VRAM" in detail
    assert f"required_mb={minimum_mb}" in detail


@pytest.mark.parametrize(("total_mb", "accepted"), [(8187.5, True), (8163.0, True), (6144.0, False)])
def test_the_eight_gigabyte_floor_admits_what_8gb_cards_report(total_mb, accepted):
    """8GB cards report a little under 8192 MiB; 6GB cards must still stop."""
    env = {"ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"}
    if accepted:
        gpu_worker._enforce_min_physical_vram(total_mb=total_mb, env=env)
    else:
        with pytest.raises(RuntimeError, match="below supported minimum"):
            gpu_worker._enforce_min_physical_vram(total_mb=total_mb, env=env)


def test_explicit_vram_budget_sets_worker_allocator_fraction():
    allocator_calls: list[tuple[float, int]] = []

    class _Properties:
        total_memory = 8 * 1024 * 1024 * 1024

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(_index):
            return _Properties()

        @staticmethod
        def get_device_name(_index):
            return "Fake GPU"

        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            allocator_calls.append((fraction, device))

    class _Torch:
        cuda = _Cuda()

    env = {
        "ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        "ASR_BATCH_SIZE": "auto",
        "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "4096",
        "ASR_STAGE_WORKER_VRAM_RATIO": "0.95",
    }
    with gpu_worker._temporary_env(env):
        tuning = gpu_worker._adaptive_runtime_tuning(_Torch(), env)

    assert tuning["vram_budget_source"] == "explicit"
    assert tuning["vram_budget_mb"] == pytest.approx(4096.0)
    assert tuning["vram_allocator_fraction"] == pytest.approx(0.5)
    assert len(allocator_calls) == 1
    assert allocator_calls[0][0] == pytest.approx(0.5)
    assert allocator_calls[0][1] == 0


def test_chunking_oom_is_not_answered_by_lowering_the_asr_batch():
    """OOM in chunking must stop, not shrink an unrelated knob.

    Chunking encodes fixed 30-second windows one at a time; halving
    `ASR_BATCH_SIZE` would change nothing about the stage that ran out of
    memory, so the retry would fail identically while looking like progress.
    """
    env = {"ASR_BATCH_SIZE": "4"}
    exc = gpu_worker.GpuWorkerError(
        "oom",
        "CUDA out of memory",
        stage="audio_chunking",
        runtime_tuning={"asr_batch_size": 4},
    )

    assert gpu_worker._oom_downshift(env, exc) is None
    assert env["ASR_BATCH_SIZE"] == "4"


def test_chunking_oom_is_recognised_from_the_stage_message():
    assert gpu_worker._oom_stage_from_message("切分 0/1") == "audio_chunking"
    assert gpu_worker._oom_stage_from_message("音频切块 3/9...") == "audio_chunking"
    for retired in ("语音岛检测 1/1", "Pre-ASR CueQC 1/1", "外边界精修 1/1"):
        assert gpu_worker._oom_stage_from_message(retired) == ""


def test_terminal_oom_guidance_offers_no_smaller_model():
    # The 0.6B tier was dropped on 2026-07-31. Telling the user to switch
    # to a model that no longer ships would send them chasing a fix that
    # does not exist.
    detail = gpu_worker._terminal_oom_detail(
        env={
            "ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
            "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "6144",
        },
        detail="CUDA out of memory",
        batch_size=1,
        retry_records=[],
    )

    assert "0.6B" not in detail
    assert "只有 1.7B 一档" in detail
    assert "当前硬件/可用显存下仍然不行" in detail
    assert "不会改用 CPU 或缩短时序窗口" in detail


class _CudaTrap:
    def __getattr__(self, name):
        raise AssertionError(f"main process touched torch.cuda.{name}")


class _TorchTrap:
    cuda = _CudaTrap()


def test_unified_asr_stage_worker_does_not_touch_cuda_in_main(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        skip_translation=True,
    )

    monkeypatch.setattr(main, "torch", _TorchTrap())
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(
        main.asr_module,
        "_get_asr_runtime_signature",
        lambda last_boundary_signature=None: {"asr": "sig"},
    )

    def fake_extract_audio(_video_path: str, out_path: str, **_kwargs) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_worker_transcribe(
        _audio_path,
        *,
        device,
        env_overrides,
        job_id,
        on_stage=None,
        cancel_requested=None,
        **_run_kwargs,
    ):
        assert device == "auto"
        assert "ASR_STAGE_WORKER_MODE" not in env_overrides
        assert env_overrides["ASR_STAGE_WORKER_VRAM_BUDGET_MB"] == "auto"
        assert env_overrides["ASR_STAGE_WORKER_VRAM_RATIO"] == "0.95"
        assert job_id == ctx.job_id
        assert cancel_requested is not None
        if on_stage is not None:
            on_stage("ASR 文本转写 1/1")
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {
                "device": "cuda:0",
                "stage_worker": {
                    "mode": "gpu_worker",
                    "process_model": "persistent_subprocess",
                    "pid": 1234,
                },
                "transcript_chunks": [],
                "stage_timings": {},
            },
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        fake_worker_transcribe,
    )

    artifacts = main.run_asr_alignment(
        str(video_path),
        ctx=ctx,
        job_id=ctx.job_id,
    )

    assert artifacts.device == "cuda:0"
    assert artifacts.asr_details["stage_worker"]["mode"] == "gpu_worker"
    assert all(
        snapshot.get("skipped")
        for snapshot in artifacts.asr_details["pipeline_cuda_memory"]
    )


def test_stage_worker_retries_with_lower_batch_on_hard_oom(monkeypatch, tmp_path):
    # The OOM-retry/downshift path is driven by a HARD OOM reported by the
    # worker (torch OOM or the allocated-based VRAM budget tripped mid-pipeline),
    # not by retroactively discarding a successful result. On a hard OOM the
    # client kills the worker, halves ASR_BATCH_SIZE, restarts and retries.
    monkeypatch.setenv("ASR_STAGE_WORKER_OOM_RETRY_LIMIT", "1")
    calls: list[dict] = []
    killed = {"count": 0}

    def fake_once(
        self,
        audio_path,
        *,
        device,
        env_overrides,
        job_id,
        on_stage=None,
        cancel_requested=None,
        **_run_kwargs,
    ):
        del audio_path, device, job_id, on_stage, cancel_requested
        calls.append(dict(env_overrides))
        if len(calls) == 1:
            self._kill_child()
            raise gpu_worker.GpuWorkerError("oom", "hard oom from worker")
        assert env_overrides["ASR_BATCH_SIZE"] == "6"
        return (
            [{"start": 0.0, "end": 1.0, "text": "ok"}],
            ["ok"],
            {
                "cuda_memory": [
                    {
                        "stage": "asr_text_transcribe_done",
                        "max_allocated_mb": 4800.0,
                    }
                ],
                "stage_worker": {"mode": "gpu_worker"},
            },
        )

    def fake_kill(self):
        del self
        killed["count"] += 1

    monkeypatch.setattr(gpu_worker._GpuWorkerClient, "_transcribe_and_align_once", fake_once)
    monkeypatch.setattr(gpu_worker._GpuWorkerClient, "_kill_child", fake_kill)

    client = gpu_worker._GpuWorkerClient()
    segments, _log, details = client.transcribe_and_align(
        str(tmp_path / "audio.wav"),
        env_overrides={
            "ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
            "ASR_BATCH_SIZE": "12",
            "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "5600",
        },
    )

    assert segments[0]["text"] == "ok"
    assert killed["count"] == 1
    assert len(calls) == 2
    assert details["stage_worker"]["oom_retries"][0]["previous_batch_size"] == 12
    assert details["stage_worker"]["oom_retries"][0]["next_batch_size"] == 6


def test_stage_worker_keeps_successful_result_when_reserved_exceeds_budget(
    monkeypatch, tmp_path
):
    # Regression guard for the VRAM-budget fix: a job that completes successfully
    # must NOT be discarded just because the allocator's *reserved* pool exceeded
    # the budget. Reserved routinely fills dedicated VRAM on a 6GB card without
    # spilling and does not respond to batch downshift. Enforcement is
    # allocated-based and happens mid-pipeline, never retroactively on success.
    warned: list[str] = []

    def fake_once(
        self,
        audio_path,
        *,
        device,
        env_overrides,
        job_id,
        on_stage=None,
        cancel_requested=None,
        **_run_kwargs,
    ):
        del self, audio_path, device, job_id, cancel_requested
        return (
            [{"start": 0.0, "end": 1.0, "text": "ok"}],
            ["ok"],
            {
                "cuda_memory": [
                    {
                        "stage": "asr_text_transcribe_done",
                        "max_reserved_mb": 6200.0,
                        "max_allocated_mb": 4800.0,
                    }
                ],
                "stage_worker": {"mode": "gpu_worker"},
            },
        )

    def fake_kill(self):
        raise AssertionError("successful result must not kill/retry the worker")

    monkeypatch.setattr(gpu_worker._GpuWorkerClient, "_transcribe_and_align_once", fake_once)
    monkeypatch.setattr(gpu_worker._GpuWorkerClient, "_kill_child", fake_kill)

    client = gpu_worker._GpuWorkerClient()
    segments, _log, details = client.transcribe_and_align(
        str(tmp_path / "audio.wav"),
        env_overrides={
            "ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
            "ASR_BATCH_SIZE": "12",
            "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "5600",
        },
        on_stage=lambda message: warned.append(message),
    )

    assert segments[0]["text"] == "ok"
    assert "oom_retries" not in details.get("stage_worker", {})
    # A non-fatal warning surfaces the reserved spike (6200 > 5600) for logs.
    assert any("reserved VRAM peaked" in message for message in warned)


def test_stage_worker_stops_with_low_vram_guidance_when_batch_one_oom(
    monkeypatch,
    tmp_path,
):
    # Persistent hard OOM down to batch=1 surfaces the low-VRAM guidance and
    # stops the job instead of looping forever.
    monkeypatch.setenv("ASR_STAGE_WORKER_OOM_RETRY_LIMIT", "3")
    calls: list[dict] = []
    killed = {"count": 0}

    def fake_once(
        self,
        audio_path,
        *,
        device,
        env_overrides,
        job_id,
        on_stage=None,
        cancel_requested=None,
        **_run_kwargs,
    ):
        del audio_path, device, job_id, on_stage, cancel_requested
        calls.append(dict(env_overrides))
        self._kill_child()
        raise gpu_worker.GpuWorkerError("oom", "hard oom from worker")

    def fake_kill(self):
        del self
        killed["count"] += 1

    monkeypatch.setattr(gpu_worker._GpuWorkerClient, "_transcribe_and_align_once", fake_once)
    monkeypatch.setattr(gpu_worker._GpuWorkerClient, "_kill_child", fake_kill)

    client = gpu_worker._GpuWorkerClient()
    with pytest.raises(gpu_worker.GpuWorkerError) as exc_info:
        client.transcribe_and_align(
            str(tmp_path / "audio.wav"),
            env_overrides={
                "ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
                "ASR_BATCH_SIZE": "4",
                "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "5600",
            },
        )

    assert exc_info.value.kind == "oom"
    assert [call["ASR_BATCH_SIZE"] for call in calls] == ["4", "2", "1"]
    assert killed["count"] == 3
    assert "ASR_BATCH_SIZE 已降到 1" in exc_info.value.detail
    assert "任务已停止" in exc_info.value.detail
    assert "0.6B" not in exc_info.value.detail
    assert "只有 1.7B 一档" in exc_info.value.detail
