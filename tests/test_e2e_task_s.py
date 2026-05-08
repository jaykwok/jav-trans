from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import main
from helpers import make_job_context, run_pipeline
def _segments(count: int, *, empty: bool = False) -> list[dict]:
    return [
        {
            "start": float(index),
            "end": float(index) + 0.8,
            "text": "" if empty else f"ja-{index}",
        }
        for index in range(count)
    ]


def _configure_headless(monkeypatch) -> None:
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)


def _mock_audio_and_asr(monkeypatch, segments: list[dict], *, checkpoint: bool = False) -> None:
    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake wav")

    def fake_transcribe_and_align(audio_path, _device, on_stage=None, include_details=False):
        assert include_details is True
        if on_stage:
            on_stage(f"ASR 文本转写 {len(segments)}/{len(segments)}")
        if checkpoint:
            job_dir = Path(audio_path).parents[1]
            (job_dir / "asr_checkpoint_fake.json").write_text("{}", encoding="utf-8")
        return (
            segments,
            ["mock asr"],
            {
                "transcript_chunks": [
                    {"index": index, "text": segment.get("text", "")}
                    for index, segment in enumerate(segments)
                ],
                "stage_timings": {},
            },
        )

    monkeypatch.setattr(main, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)


def _fake_translation_json(messages, expected_count=0, on_progress=None, **_kwargs):
    content = messages[1]["content"]
    ids = [int(value) for value in re.findall(r'"id":\s*(\d+)', content)]
    ids = ids[:expected_count]
    if on_progress:
        on_progress({"phase": "translating", "translated": len(ids), "expected": expected_count})
        on_progress({"phase": "done", "translated": len(ids), "expected": expected_count})
    return json.dumps(
        {"translations": [{"id": index, "text": f"zh-{index}"} for index in ids]},
        ensure_ascii=False,
    )


def _load_cache(path: Path) -> dict:
    return main.translator_module._load_translation_cache(path)


def test_s1_skip_translation_writes_japanese_srt_without_translator(monkeypatch, tmp_path):
    video_path = Path("temp/nmsl036_60s.mp4")
    if not video_path.exists():
        video_path = tmp_path / "test_clip_60s.mp4"
        video_path.write_bytes(b"fake mp4")

    output_dir = tmp_path / "out"
    _configure_headless(monkeypatch)
    ctx = make_job_context(
        video_path,
        output_dir,
        tmp_path / "jobs",
        skip_translation=True,
        keep_temp_files=True,
    )
    _mock_audio_and_asr(monkeypatch, _segments(3))
    monkeypatch.setattr(
        main.translator_module,
        "translate_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("translation should be skipped")
        ),
    )

    run_pipeline(video_path, ctx)

    srt_path = output_dir / f"{video_path.stem}.ja.srt"
    assert srt_path.is_file()
    assert "ja-0" in srt_path.read_text(encoding="utf-8")


def test_s2_cleanup_removes_entire_job_temp_dir(tmp_path):
    job_dir = tmp_path / "job"
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "clip.wav").write_bytes(b"wav")
    (job_dir / "translation_cache.jsonl").write_text("", encoding="utf-8")
    (job_dir / "asr_checkpoint_fake.json").write_text("{}", encoding="utf-8")
    (job_dir / "clip.srt").write_text("1\n", encoding="utf-8")

    main._cleanup_job_temp(str(job_dir))

    assert not (audio_dir / "clip.wav").exists()
    assert not (job_dir / "translation_cache.jsonl").exists()
    assert not (job_dir / "asr_checkpoint_fake.json").exists()
    assert not (job_dir / "clip.srt").exists()
    assert not job_dir.exists()


def test_s3_s4_translation_resume_then_auto_cleanup(monkeypatch, tmp_path, capsys):
    video_path = tmp_path / "resume.mp4"
    video_path.write_bytes(b"fake mp4 resume")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    job_dir = temp_root / "resume"

    _configure_headless(monkeypatch)
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        subtitle_mode="zh",
        translation_batch_size=2,
        translation_max_workers=1,
        keep_temp_files=False,
    )
    monkeypatch.setattr(main.translator_module, "_chat", _fake_translation_json)
    _mock_audio_and_asr(monkeypatch, _segments(5), checkpoint=True)

    monkeypatch.setenv("_TEST_CRASH_TRANSLATION_BATCH", "2")
    with pytest.raises(SystemExit) as first_exit:
        run_pipeline(video_path, ctx)
    assert first_exit.value.code == 1

    cache_path = job_dir / "translation_cache.jsonl"
    wav_files = list((job_dir / "audio").glob("*.wav"))
    assert cache_path.is_file()
    assert _load_cache(cache_path)
    assert wav_files
    assert (job_dir / "asr_checkpoint_fake.json").is_file()

    calls: list[int] = []

    def fake_chat_counting(messages, expected_count=0, on_progress=None, **_kwargs):
        calls.append(expected_count)
        return _fake_translation_json(messages, expected_count, on_progress, **_kwargs)

    monkeypatch.delenv("_TEST_CRASH_TRANSLATION_BATCH", raising=False)
    monkeypatch.setattr(main.translator_module, "_chat", fake_chat_counting)

    run_pipeline(video_path, ctx)
    stdout = capsys.readouterr().out

    srt_path = output_dir / "resume.srt"
    assert srt_path.is_file()
    assert "zh-4" in srt_path.read_text(encoding="utf-8")
    assert "translation-cache" in stdout or "cache" in stdout.lower()
    assert len(calls) < 3
    assert not cache_path.exists()
    assert not list((job_dir / "audio").glob("*.wav"))
    assert not (job_dir / "asr_checkpoint_fake.json").exists()


def test_s5_qc_gate_blocks_headless_before_translation(monkeypatch, tmp_path, capsys):
    video_path = Path("temp/silence_10s.mp4")
    if not video_path.exists():
        video_path = tmp_path / "silence_10s.mp4"
        video_path.write_bytes(b"fake silent mp4")

    _configure_headless(monkeypatch)
    monkeypatch.delenv("QC_IGNORE_EMPTY", raising=False)
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        keep_temp_files=True,
    )
    _mock_audio_and_asr(monkeypatch, _segments(3, empty=True))
    monkeypatch.setattr(
        main.translator_module,
        "translate_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("translation should be blocked by QC")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="ASR empty text rate too high; set QC_IGNORE_EMPTY=1 to skip",
    ):
        run_pipeline(video_path, ctx)

    stdout = capsys.readouterr().out
    assert "ASR 空文本率过高" in stdout or "empty_text_ratio" in stdout

