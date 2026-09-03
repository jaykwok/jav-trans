"""The frame-class track has to survive from the head to the cue filter.

Three hops, each of which has silently dropped something before in this project:
the chunking pass produces it, a JSON cache carries it across a rerun, and the
layout stage asks it about spans that did not exist when it was made. A break
anywhere shows up as the filter quietly falling back to the text rule - the
shipped behaviour - so nothing fails and the acoustics simply stop being used.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import (  # noqa: E402
    ENCODER_FRAME_S,
    FRAME_CLASS_SILENCE,
    FRAME_CLASS_SPEECH,
    FRAME_CLASS_VOCALISATION,
)
from asr.pipeline import _encode_frame_class_track, decode_frame_class_track  # noqa: E402


def _track(classes: list[int], *, upsample: int = 2, confidence: float = 0.94):
    spread = (1.0 - confidence) / 2.0
    posteriors = np.full((len(classes), 3), spread, dtype=np.float32)
    for frame, klass in enumerate(classes):
        posteriors[frame, klass] = confidence
    return _encode_frame_class_track(
        posteriors, frame_s=ENCODER_FRAME_S / upsample
    )


class TestTheTrackSurvivesJson:
    def test_quantisation_keeps_far_more_precision_than_any_threshold_needs(self):
        """A byte per class is 1/255; the thresholds it feeds are 0.05 and 0.10,
        so the rounding is two orders of magnitude below anything that decides."""
        track = _track([FRAME_CLASS_VOCALISATION] * 4)
        posteriors, frame_s = decode_frame_class_track(track)

        assert posteriors.shape == (4, 3)
        assert posteriors[:, FRAME_CLASS_VOCALISATION] == pytest.approx(0.94, abs=0.004)
        assert frame_s == pytest.approx(ENCODER_FRAME_S / 2)

    def test_it_is_json_serialisable(self):
        import json

        track = _track([FRAME_CLASS_SPEECH] * 10)
        restored = json.loads(json.dumps(track))

        assert decode_frame_class_track(restored) is not None

    def test_a_missing_track_decodes_to_none_rather_than_raising(self):
        """Every reader must treat absence as "v1 head" and fall back, because a
        promoted head outlives the code that trained it."""
        assert decode_frame_class_track(None) is None
        assert decode_frame_class_track({}) is None
        assert decode_frame_class_track({"schema": "something_else"}) is None

    def test_reordered_classes_are_refused(self):
        track = _track([FRAME_CLASS_SPEECH])
        track["classes"] = ["speech", "silence", "vocalisation"]

        assert decode_frame_class_track(track) is None

    def test_a_truncated_payload_is_refused_rather_than_reshaped(self):
        track = _track([FRAME_CLASS_SPEECH] * 8)
        track["frames"] = 9

        assert decode_frame_class_track(track) is None


class TestItReachesTheCue:
    def test_a_cue_is_dropped_on_evidence_the_text_rule_cannot_see(self):
        """The whole chain, at the seam that matters: one isolated `あっ` that
        the run rule keeps, and a track saying those seconds are moaning."""
        from main import _frame_class_reader
        from subtitles import writer
        from subtitles.options import SubtitleOptions

        # 13 frames of speech, then 26 of moaning, at 26 fps: 0.5 s then 1.0 s.
        track = _track(
            [FRAME_CLASS_SPEECH] * 13
            + [FRAME_CLASS_VOCALISATION] * 26
            + [FRAME_CLASS_SILENCE] * 13
        )
        reader = _frame_class_reader({"frame_class_track": track})
        assert reader is not None

        moaning = reader(0.5, 1.5)
        assert moaning["vocalisation"] > 0.9
        assert moaning["speech"] < 0.05
        assert moaning["speech_max_run_s"] == 0.0

        talking = reader(0.0, 0.5)
        assert talking["speech"] > 0.9

        blocks = [
            {"ja_text": "そうなんだ", "start": 0.0, "end": 0.5,
             "acoustic_start": 0.0, "acoustic_end": 0.5},
            {"ja_text": "あっ", "start": 0.5, "end": 1.5,
             "acoustic_start": 0.5, "acoustic_end": 1.5},
            {"ja_text": "本当に？", "start": 1.5, "end": 2.0,
             "acoustic_start": 1.5, "acoustic_end": 2.0},
        ]
        diagnostics: dict = {}
        writer.prepare_srt_blocks(
            blocks,
            options=SubtitleOptions(),
            diagnostics=diagnostics,
            acoustic_classes=reader,
        )

        assert diagnostics["vocalisation_cues_dropped_by_acoustics"] == 1

    def test_without_a_track_the_filter_is_exactly_the_text_rule(self):
        from main import _frame_class_reader
        from subtitles import writer
        from subtitles.options import SubtitleOptions

        assert _frame_class_reader({}) is None

        blocks = [
            {"ja_text": "そうなんだ", "start": 0.0, "end": 0.5},
            {"ja_text": "あっ", "start": 0.5, "end": 1.5},
            {"ja_text": "本当に？", "start": 1.5, "end": 2.0},
        ]
        diagnostics: dict = {}
        cues = writer.prepare_srt_blocks(
            blocks, options=SubtitleOptions(), diagnostics=diagnostics
        )

        assert diagnostics["vocalisation_cues_dropped"] == 0
        assert len(cues) == 3
