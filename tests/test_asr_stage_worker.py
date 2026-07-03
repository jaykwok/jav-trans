from pathlib import Path

import main
from helpers import make_job_context
from pipeline import audio as pipeline_audio
from pipeline import gpu_worker


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
        assert env_overrides["ASR_STAGE_WORKER_VRAM_BUDGET_MB"] == "5600"
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


def test_stage_worker_treats_vram_budget_excess_as_oom_retry(monkeypatch, tmp_path):
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
        del self, audio_path, device, job_id, on_stage, cancel_requested
        calls.append(dict(env_overrides))
        if len(calls) == 1:
            return (
                [],
                [],
                {
                    "cuda_memory": [
                        {
                            "stage": "asr_text_transcribe_done",
                            "max_reserved_mb": 6200.0,
                        }
                    ],
                    "stage_worker": {"mode": "gpu_worker"},
                },
            )
        assert env_overrides["ASR_BATCH_SIZE"] == "6"
        return (
            [{"start": 0.0, "end": 1.0, "text": "ok"}],
            ["ok"],
            {
                "cuda_memory": [
                    {
                        "stage": "asr_text_transcribe_done",
                        "max_reserved_mb": 5200.0,
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
