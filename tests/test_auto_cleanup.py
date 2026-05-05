from pathlib import Path

import main
from whisper import pipeline as asr
def test_cleanup_job_temp_removes_wav_json_and_keeps_srt(tmp_path):
    job_dir = tmp_path / "job"
    nested = job_dir / "audio"
    nested.mkdir(parents=True)

    wav_path = nested / "sample.wav"
    json_path = job_dir / "sample.timings.json"
    srt_path = job_dir / "sample.srt"
    txt_path = job_dir / "notes.txt"

    wav_path.write_bytes(b"wav")
    json_path.write_text("{}", encoding="utf-8")
    srt_path.write_text("1\n", encoding="utf-8")
    txt_path.write_text("keep", encoding="utf-8")

    main._cleanup_job_temp(str(job_dir))

    assert not wav_path.exists()
    assert not json_path.exists()
    assert not nested.exists()
    assert not srt_path.exists()
    assert not txt_path.exists()
    assert not job_dir.exists()


def test_keep_temp_context_skips_cleanup_call(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    wav_path = job_dir / "sample.wav"
    json_path = job_dir / "sample.json"
    wav_path.write_bytes(b"wav")
    json_path.write_text("{}", encoding="utf-8")

    keep_temp_files = True
    if not keep_temp_files:
        main._cleanup_job_temp(str(job_dir))

    assert wav_path.exists()
    assert json_path.exists()


def test_runtime_ephemeral_cleanup_keeps_reusable_caches(tmp_path, monkeypatch):
    temp_root = tmp_path / "temp"
    jobs = temp_root / "jobs"
    chunks = temp_root / "chunks"
    recovery = temp_root / "recovery"
    hf_cache = temp_root / "hf-cache"

    for path in [jobs, chunks, recovery, hf_cache]:
        path.mkdir(parents=True)
    (hf_cache / "cache.bin").write_bytes(b"cache")

    monkeypatch.setenv("JOB_TEMP_DIR", str(jobs))
    monkeypatch.setenv("ASR_RECOVERY_OUTPUT_ROOT", str(recovery))
    monkeypatch.setattr(asr, "_ASR_CHUNK_ROOT", chunks)

    main._cleanup_runtime_ephemeral_temp()

    assert not jobs.exists()
    assert not chunks.exists()
    assert not recovery.exists()
    assert (hf_cache / "cache.bin").exists()

