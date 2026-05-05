import json
from pathlib import Path

import pytest

import main
from helpers import make_job_context, run_pipeline
def _segments(count: int = 5) -> list[dict]:
    return [
        {"start": float(index), "end": float(index) + 0.8, "text": f"ja-{index}"}
        for index in range(count)
    ]


def _batch_start_from_messages(messages) -> int:
    content = messages[1]["content"]
    ids = []
    for line in content.splitlines():
        if '"id": ' in line:
            ids.append(int(line.split('"id": ')[1].split(",", 1)[0]))
    return min(ids)


def _mock_translation_json(start: int, count: int) -> str:
    return json.dumps(
        {
            "translations": [
                {"id": index, "text": f"zh-{index}"}
                for index in range(start, start + count)
            ]
        },
        ensure_ascii=False,
    )


def _load_cache(path: Path) -> dict:
    return main.translator_module._load_translation_cache(path)


def _patch_pipeline(
    monkeypatch,
    *,
    tmp_path: Path,
    video_path: Path,
    output_dir: Path,
    job_temp_root: Path,
    cache_path: Path | None = None,
    segments: list[dict] | None = None,
):
    ctx = make_job_context(
        video_path,
        output_dir,
        job_temp_root,
        subtitle_mode="zh",
        translation_batch_size=2,
        translation_max_workers=1,
        translation_cache_path=cache_path,
        keep_temp_files=True,
    )
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("QC_IGNORE_EMPTY", raising=False)
    monkeypatch.setattr(
        main.asr_module,
        "_ASR_CHUNK_ROOT",
        tmp_path / "asr_root" / "chunks",
        raising=False,
    )
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")

    monkeypatch.setattr(main.translator_module, "_request_backoff_sleep", lambda *_args: None)

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"fake-wav")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        assert include_details is True
        if on_stage:
            on_stage("ASR mock")
        return (
            list(segments if segments is not None else _segments()),
            ["mock asr"],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(main, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)
    return ctx


def test_cleanup_removes_translation_cache_and_matching_asr_checkpoint(monkeypatch, tmp_path):
    job_dir = tmp_path / "jobs" / "clip"
    job_dir.mkdir(parents=True)
    (job_dir / "clip.transcript.json").write_text("{}", encoding="utf-8")
    audio_path = job_dir / "audio" / "clip.hash.wav"
    audio_path.parent.mkdir()
    audio_path.write_bytes(b"wav")

    cache_path = tmp_path / "translation_cache.json"
    cache_path.write_text('{"0": ["zh-0"]}', encoding="utf-8")
    checkpoint_root = tmp_path / "asr_root"
    checkpoint_root.mkdir()
    matching_checkpoint = checkpoint_root / "asr_checkpoint_match.json"
    matching_checkpoint.write_text(
        json.dumps({"audio_path": f"{audio_path}|mock_text_stage"}),
        encoding="utf-8",
    )
    unrelated_checkpoint = checkpoint_root / "asr_checkpoint_other.json"
    unrelated_checkpoint.write_text(
        json.dumps({"audio_path": "other/job/audio.wav|mock_text_stage"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("TRANSLATION_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(
        main.asr_module,
        "_ASR_CHUNK_ROOT",
        checkpoint_root / "chunks",
        raising=False,
    )

    main._cleanup_job_temp(str(job_dir))

    assert not cache_path.exists()
    assert not matching_checkpoint.exists()
    assert unrelated_checkpoint.exists()
    assert not audio_path.exists()


def test_translation_crash_resume_and_success_cleanup(monkeypatch, tmp_path, capsys):
    video_path = tmp_path / "nmsl036_60s.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    job_temp_root = tmp_path / "jobs"
    cache_path = tmp_path / "translation_cache.jsonl"
    chat_calls: list[int] = []

    def fake_chat(messages, expected_count=0, on_progress=None):
        start = _batch_start_from_messages(messages)
        chat_calls.append(start)
        return _mock_translation_json(start, expected_count)

    monkeypatch.setattr(main.translator_module, "_chat", fake_chat)
    ctx = _patch_pipeline(
        monkeypatch,
        tmp_path=tmp_path,
        video_path=video_path,
        output_dir=output_dir,
        job_temp_root=job_temp_root,
        cache_path=cache_path,
    )
    monkeypatch.setenv("_TEST_CRASH_TRANSLATION_BATCH", "2")

    with pytest.raises(SystemExit) as first_exit:
        run_pipeline(video_path, ctx)

    assert first_exit.value.code == 1
    cache_payload = _load_cache(cache_path)
    assert sorted(key.split("::")[1] for key in cache_payload) == ["0", "1"]
    job_dir = job_temp_root / "nmsl036_60s"
    wav_files = list(job_dir.rglob("*.wav"))
    assert wav_files
    checkpoint_path = tmp_path / "asr_root" / "asr_checkpoint_resume.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps({"audio_path": f"{wav_files[0]}|mock_text_stage"}),
        encoding="utf-8",
    )

    chat_calls.clear()
    monkeypatch.delenv("_TEST_CRASH_TRANSLATION_BATCH", raising=False)
    ctx.keep_temp_files = False

    run_pipeline(video_path, ctx)
    stdout = capsys.readouterr().out

    assert "translation-cache" in stdout
    assert chat_calls == [4]
    srt_path = output_dir / "nmsl036_60s.srt"
    assert srt_path.exists()
    assert srt_path.read_text(encoding="utf-8").count("-->") == 5
    assert not cache_path.exists()
    assert not checkpoint_path.exists()
    assert not list(job_dir.rglob("*.wav")) if job_dir.exists() else True


def test_qc_gate_blocks_headless_before_translation(monkeypatch, tmp_path, capsys):
    video_path = tmp_path / "silence_10s.mp4"
    video_path.write_bytes(b"fake-silence")
    output_dir = tmp_path / "out"
    job_temp_root = tmp_path / "jobs"

    ctx = _patch_pipeline(
        monkeypatch,
        tmp_path=tmp_path,
        video_path=video_path,
        output_dir=output_dir,
        job_temp_root=job_temp_root,
        segments=[
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": "   "},
        ],
    )

    def fail_translate_segments(*_args, **_kwargs):
        raise AssertionError("translation should not run after QC gate blocks")

    monkeypatch.setattr(main.translator_module, "translate_segments", fail_translate_segments)

    with pytest.raises(
        RuntimeError,
        match="ASR empty text rate too high; set QC_IGNORE_EMPTY=1 to skip",
    ):
        run_pipeline(video_path, ctx)

    stdout = capsys.readouterr().out
    assert "ASR 空文本率过高" in stdout or "empty_text_ratio" in stdout
    assert not (output_dir / "silence_10s.srt").exists()

