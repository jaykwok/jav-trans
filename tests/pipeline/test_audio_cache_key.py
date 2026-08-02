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

