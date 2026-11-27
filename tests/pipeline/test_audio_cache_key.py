from pipeline import audio as pipeline_audio
from pipeline.audio import get_audio_cache_key


def test_audio_cache_key_includes_video_content_for_same_filename(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()

    first = left_dir / "sample.mp4"
    second = right_dir / "sample.mp4"
    first.write_bytes(b"A" * 1024)
    second.write_bytes(b"B" * 1024)

    assert first.name == second.name
    assert get_audio_cache_key(str(first)) != get_audio_cache_key(
        str(second)
    )


def test_audio_cache_key_changes_with_timeline_filter(monkeypatch, tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"same-video")
    old_key = get_audio_cache_key(str(video))

    monkeypatch.setattr(
        pipeline_audio,
        "audio_timeline_filter",
        lambda *, sample_rate=None: "aresample=16000:async=999:first_pts=0",
    )

    assert get_audio_cache_key(str(video)) != old_key


def test_audio_config_reads_env_at_call_time(monkeypatch, tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"same-video")
    default_key = get_audio_cache_key(str(video))

    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "24000")
    assert pipeline_audio.audio_sample_rate() == 24000
    assert "aresample=24000:" in pipeline_audio.build_audio_filter_chain()
    assert get_audio_cache_key(str(video)) != default_key

    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "16000")
    monkeypatch.setenv("AUDIO_FILTER", "highpass=f=120")
    assert "highpass=f=120" in pipeline_audio.build_audio_filter_chain()
    assert get_audio_cache_key(str(video)) != default_key
