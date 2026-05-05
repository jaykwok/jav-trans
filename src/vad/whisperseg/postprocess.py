# Vendored from WhisperJAV (https://github.com/meizhong986/WhisperJAV) — MIT License
# Original: whisperjav/modules/speech_segmentation/backends/ten.py @ v1.8.12
# Local modifications: import paths, env-driven config, project logger

from __future__ import annotations

from vad.base import SpeechSegment


def group_segments(
    segments: list[SpeechSegment],
    max_group_duration_s: float = 29.0,
    chunk_threshold_s: float = 1.0,
) -> list[list[SpeechSegment]]:
    """Group speech segments by time gaps and max duration."""
    if not segments:
        return []

    groups: list[list[SpeechSegment]] = [[]]

    for i, segment in enumerate(segments):
        if i > 0:
            prev_end = segments[i - 1].end
            gap = segment.start - prev_end

            would_exceed_max = False
            if groups[-1]:
                group_start = groups[-1][0].start
                potential_duration = segment.end - group_start
                would_exceed_max = potential_duration > max_group_duration_s

            if gap > chunk_threshold_s or would_exceed_max:
                groups.append([])

        groups[-1].append(segment)

    return groups

