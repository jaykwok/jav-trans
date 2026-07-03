from pathlib import Path

import main
from helpers import make_job_context
from pipeline import audio as pipeline_audio


class _CudaTrap:
    def __getattr__(self, name):
        raise AssertionError(f"main process touched torch.cuda.{name}")


class _TorchTrap:
    cuda = _CudaTrap()


def test_asr_stage_worker_mode_does_not_touch_cuda_in_main(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        skip_translation=True,
        keep_temp_files=True,
        advanced={"ASR_STAGE_WORKER_MODE": "subprocess"},
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
        assert env_overrides["ASR_STAGE_WORKER_MODE"] == "subprocess"
        assert job_id == ctx.job_id
        assert cancel_requested is not None
        if on_stage is not None:
            on_stage("ASR 文本转写 1/1")
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {
                "device": "cuda:0",
                "stage_worker": {"mode": "subprocess", "pid": 1234},
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
    assert artifacts.asr_details["stage_worker"]["mode"] == "subprocess"
    assert all(
        snapshot.get("skipped")
        for snapshot in artifacts.asr_details["pipeline_cuda_memory"]
    )
