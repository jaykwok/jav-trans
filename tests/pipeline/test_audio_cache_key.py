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
        "_AUDIO_TIMELINE_FILTER",
        "aresample=16000:async=999:first_pts=0",
    )

    assert get_audio_cache_key(str(video)) != old_key

