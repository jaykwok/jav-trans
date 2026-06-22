from __future__ import annotations

from boundary.base import SpeechSegment


def group_segments(
    segments: list[SpeechSegment],
    max_group_duration_s: float = 29.0,
    chunk_threshold_s: float = 1.0,
) -> list[list[SpeechSegment]]:
    """Group speech segments by gap and maximum grouped duration."""
    if not segments:
        return []

    groups: list[list[SpeechSegment]] = [[]]
    for index, segment in enumerate(segments):
        if index > 0:
            previous_end = segments[index - 1].end
            gap = segment.start - previous_end
            would_exceed_max = False
            if groups[-1]:
                group_start = groups[-1][0].start
                would_exceed_max = segment.end - group_start > max_group_duration_s
            if gap > chunk_threshold_s or would_exceed_max:
                groups.append([])
        groups[-1].append(segment)
    return groups
