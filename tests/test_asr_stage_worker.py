from pathlib import Path

import main
import pytest
from asr import pipeline as asr_pipeline
from helpers import make_job_context
from pipeline import audio as pipeline_audio
from pipeline import gpu_worker


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


def test_shared_vram_spill_is_soft_oom(monkeypatch):
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "5600")
    monkeypatch.setenv("ASR_STAGE_WORKER_SHARED_VRAM_TOLERANCE_MB", "0")
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


def test_shared_vram_auto_deadband_ignores_pdh_granularity_but_catches_spill(
    monkeypatch,
):
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "5600")
    monkeypatch.setenv("ASR_STAGE_WORKER_SHARED_VRAM_TOLERANCE_MB", "auto")
    base = {
        "stage": "split_done",
        "total_mb": 8188.0,
        "physical_ram_used_mb": 1000.0,
        "physical_ram_budget_mb": 15000.0,
        "max_allocated_mb": 4800.0,
    }
    asr_pipeline._enforce_vram_budget_from_snapshot(
        {**base, "shared_vram_mb": 4.0}
    )
    with pytest.raises(RuntimeError, match="measurement_deadband_mb"):
        asr_pipeline._enforce_vram_budget_from_snapshot(
            {**base, "shared_vram_mb": 20.0}
        )


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


def test_physical_ram_ratio_is_hard_oom(monkeypatch):
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "5600")
    with pytest.raises(RuntimeError, match="Physical RAM budget exceeded"):
        asr_pipeline._enforce_vram_budget_from_snapshot(
            {
                "stage": "pre_asr_cueqc",
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


def test_auto_vram_budget_and_batch_scale_from_physical_memory(monkeypatch):
    monkeypatch.setenv("GPU_BATCH_PROFILE_ENABLED", "0")
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
        assert gpu_worker.os.environ["ASR_BATCH_SIZE"] == "5"

    assert tuning["physical_vram_mb"] == pytest.approx(8192.0)
    assert tuning["vram_budget_source"] == "physical_vram_ratio"
    assert tuning["vram_allocator_fraction"] == pytest.approx(0.95)
    assert len(allocator_calls) == 1
    assert allocator_calls[0][0] == pytest.approx(0.95)
    assert allocator_calls[0][1] == 0
    assert tuning["asr_batch_source"] == "auto_scaled_from_vram"


@pytest.mark.parametrize(
    ("backend", "minimum_mb"),
    [
        ("jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf", 4096),
        ("jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf", 6144),
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
        ("jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf", 4096),
        ("jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf", 6144),
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


def test_explicit_vram_budget_sets_worker_allocator_fraction(monkeypatch):
    monkeypatch.setenv("GPU_BATCH_PROFILE_ENABLED", "0")
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


def test_boundary_oom_does_not_change_temporal_window():
    env = {
        "ASR_BATCH_SIZE": "4",
        "SPEECH_BOUNDARY_JA_WINDOW_S": "20",
    }
    exc = gpu_worker.GpuWorkerError(
        "oom",
        "CUDA out of memory",
        stage="speech_island_scorer",
        runtime_tuning={"asr_batch_size": 4},
    )

    assert gpu_worker._oom_downshift(env, exc) is None
    assert env["SPEECH_BOUNDARY_JA_WINDOW_S"] == "20"


def test_terminal_oom_guidance_marks_06b_as_unsupported():
    detail = gpu_worker._terminal_oom_detail(
        env={
            "ASR_BACKEND": "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf",
            "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "4096",
        },
        detail="CUDA out of memory",
        batch_size=1,
        retry_records=[],
    )

    assert "0.6B 最低显存档" in detail
    assert "当前硬件/可用显存下无法运行" in detail
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
        keep_temp_files=True,
    )

    monkeypatch.setattr(main, "torch", _TorchTrap())
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(
        main.asr_module,
        "_get_asr_runtime_signature",
        lambda last_boundary_signature=None: {"asr": "sig"},
    )

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
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
    assert "0.6B" in exc_info.value.detail
    assert "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf" in exc_info.value.detail
