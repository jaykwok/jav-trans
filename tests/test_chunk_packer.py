from audio.chunk_packer import PackedChunk
from boundary.base import SpeechSegment


def test_packed_chunk_keeps_absolute_source_timeline() -> None:
    segment = SpeechSegment(12.0, 14.5, 0.9)
    chunk = PackedChunk(
        start=12.0,
        end=14.5,
        source_abs_start=12.0,
        source_abs_end=14.5,
        speech_segments=[segment],
        duration=2.5,
        split_reason="semantic_split",
        boundary_source="shared_absolute_cut",
    )

    assert chunk.start == chunk.source_abs_start
    assert chunk.end == chunk.source_abs_end
    assert chunk.speech_segments == [segment]
