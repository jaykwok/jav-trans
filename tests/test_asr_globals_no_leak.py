from pathlib import Path

import pytest

import main
from helpers import make_job_context
from pipeline import audio as pipeline_audio


def test_asr_stage_env_restored_when_transcribe_raises(monkeypatch, tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        asr_backend="qwen3-asr-1.7b",
        asr_context="task context",
        skip_translation=True,
        keep_temp_files=True,
    )
    ctx.asr_recovery = True

    monkeypatch.setenv("ASR_BACKEND", "anime-whisper")
    monkeypatch.setenv("ASR_CONTEXT", "process context")
    monkeypatch.delenv("ASR_RECOVERY_ENABLED", raising=False)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        main.asr_module,
        "get_backend_label",
        lambda: f"backend:{main.os.environ['ASR_BACKEND']}",
    )

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        assert main.os.environ["ASR_BACKEND"] == "qwen3-asr-1.7b"
        assert main.os.environ["ASR_CONTEXT"] == "task context"
        assert main.os.environ["ASR_RECOVERY_ENABLED"] == "1"
        raise RuntimeError("forced ASR failure")

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)

    with pytest.raises(RuntimeError, match="forced ASR failure"):
        main.run_asr_alignment_f0(
            str(video_path),
            ctx=ctx,
            job_id=ctx.job_id,
        )

    assert main.os.environ["ASR_BACKEND"] == "anime-whisper"
    assert main.os.environ["ASR_CONTEXT"] == "process context"
    assert "ASR_RECOVERY_ENABLED" not in main.os.environ
