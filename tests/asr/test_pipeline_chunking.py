"""Chunking after the boundary chain was retired.

What replaced Scorer/Outer/Split/CueQC/Inner on 2026-07-31 is deliberately much
weaker: it only chooses where the cuts fall. The property that makes that
acceptable is that it cannot decide anything irreversible, so these tests are
about coverage rather than placement quality - if the chunks tile the file, a
badly placed cut costs a worse boundary and never a lost line.

The old chain's failure mode was the opposite one, which is why it is worth
pinning explicitly: it dropped audio on acoustic evidence alone, before any text
existed, and 55 real sources went with it.
"""

from __future__ import annotations

from pathlib import Path
import struct
import sys
import wave

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr import pipeline as asr  # noqa: E402


def _wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<%dh" % frames, *([0] * frames)))
    return path


@pytest.fixture()
def audio(tmp_path: Path) -> Path:
    return _wav(tmp_path / "clip.wav", 95.0)


class TestCoverage:
    def test_the_chunks_tile_the_whole_file(self, audio: Path) -> None:
        spans = asr._build_processing_spans(str(audio))
        assert spans
        assert spans[0][0] == pytest.approx(0.0)
        assert spans[-1][1] == pytest.approx(95.0, abs=0.05)
        for earlier, later in zip(spans, spans[1:]):
            assert earlier[1] == pytest.approx(later[0])

    def test_no_second_of_audio_is_dropped(self, audio: Path) -> None:
        """The one guarantee the retired chain did not make."""
        spans = asr._build_processing_spans(str(audio))
        covered = sum(end - begin for begin, end in spans)
        assert covered == pytest.approx(95.0, abs=0.05)

    def test_chunks_respect_the_maximum_length(
        self, audio: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ASR_CHUNK_MAX_S", "20.0")
        monkeypatch.setenv("ASR_CHUNK_TARGET_S", "15.0")
        spans = asr._build_processing_spans(str(audio))
        assert all(end - begin <= 20.0 + 1e-6 for begin, end in spans)
        assert sum(end - begin for begin, end in spans) == pytest.approx(95.0, abs=0.05)


class TestDegradation:
    def test_no_head_configured_still_produces_chunks(
        self, audio: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ASR_ALIGNMENT_HEAD_PATH", raising=False)
        spans = asr._build_processing_spans(str(audio))
        assert spans
        signature = asr._LAST_BOUNDARY_SIGNATURE["chunking"]
        assert signature["source"] == "fixed_length_no_head_configured"
        assert signature["pause_count"] == 0

    def test_a_broken_head_degrades_instead_of_raising(
        self, audio: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunking must never take transcription down.

        Fixed-length cuts are worse placed, not wrong, because nothing is
        dropped either way - so falling back beats failing the job.
        """
        monkeypatch.setenv("ASR_ALIGNMENT_HEAD_PATH", str(audio / "missing.pt"))
        spans = asr._build_processing_spans(str(audio))
        assert spans
        assert sum(end - begin for begin, end in spans) == pytest.approx(95.0, abs=0.05)
        assert asr._LAST_BOUNDARY_SIGNATURE["chunking"]["source"].startswith(
            "fixed_length"
        )

    def test_pauses_from_the_head_are_used_when_available(
        self, audio: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            asr,
            "_blank_runs_for_audio",
            lambda _path: ([(19.0, 20.0), (41.0, 42.0)], 95.0, "alignment_head_blank_runs"),
        )
        spans = asr._build_processing_spans(str(audio))
        assert asr._LAST_BOUNDARY_SIGNATURE["chunking"]["source"] == (
            "alignment_head_blank_runs"
        )
        assert spans[0][1] == pytest.approx(19.5)
        assert sum(end - begin for begin, end in spans) == pytest.approx(95.0)


class TestWindowSeams:
    """The head must see the same context in production that it saw in training.

    Audio longer than one encoder window is windowed, and the head used to be
    run on each window independently: every frame within ~1.15 s of a seam was
    convolved against zeros standing in for audio that exists, once per 30 s.
    The fix overlaps the AUDIO by the head's own receptive field and drops the
    overlap afterwards. It has to be the audio - the encoder runs per window
    too, so concatenating features from butt-jointed windows would leave the
    same hole one layer down.

    The head here returns its input frame indices verbatim, so the tensor that
    reaches `blank_runs` says exactly which window each output frame came from.
    """

    SECONDS = 95.0

    @pytest.fixture()
    def wired(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        torch = pytest.importorskip("torch")
        import numpy as np

        from asr import encoder_features, qwen_native
        from audio import loading

        path = _wav(tmp_path / "long.wav", self.SECONDS)
        total = int(self.SECONDS * 16000)
        # Sample i holds the value i, so a window reveals where it was cut from.
        monkeypatch.setattr(
            loading,
            "load_audio_16k_mono",
            lambda _p: (np.arange(total, dtype=np.float32), 16000),
        )

        seen: list[np.ndarray] = []

        def _prepare(_processor, *, audio, language=None):
            seen.extend(audio)
            return {"audio": list(audio)}

        def _move(inputs, *, device=None, dtype=None):
            group = inputs["audio"]
            return {
                "input_features": group,
                "input_features_mask": torch.tensor(
                    [[len(piece)] for piece in group], dtype=torch.long
                ),
            }

        def _lengths(samples):
            return [int(round(int(v) * 13 / 16000)) for v in samples.tolist()]

        class _Features:
            def __init__(self, pooler_output) -> None:
                self.pooler_output = pooler_output

        class _Model:
            device = "cpu"
            dtype = None

            def get_audio_features(self, *, input_features, input_features_mask):
                rows = []
                for piece in input_features:
                    base = int(round(float(piece[0]) * 13 / 16000))
                    frames = int(round(len(piece) * 13 / 16000))
                    rows.append(
                        np.arange(base, base + frames, dtype=np.float32)[:, None]
                    )
                return _Features(torch.from_numpy(np.concatenate(rows, axis=0)))

        class _Head:
            upsample = 2
            context_frames = 15
            # Matches `AlignmentHead.silent_classes`; empty is what an
            # acoustic-only vocabulary reports, so the reading is unchanged.
            silent_classes = frozenset()

            @classmethod
            def from_env(cls):
                return cls()

            def log_probs(self, features):
                return torch.from_numpy(np.asarray(features)).repeat_interleave(
                    2, dim=0
                )

        captured: dict = {}

        def _blank_runs(log_probs, *, upsample, min_seconds, silent_classes=None):
            captured["log_probs"] = log_probs
            captured["silent_classes"] = silent_classes
            return [(1.0, 2.0)]

        monkeypatch.setattr(qwen_native, "prepare_transcription_inputs", _prepare)
        monkeypatch.setattr(qwen_native, "move_processor_inputs", _move)
        monkeypatch.setattr(
            encoder_features, "qwen3_asr_audio_output_lengths", _lengths
        )
        monkeypatch.setattr(asr, "AlignmentHead", _Head)
        monkeypatch.setattr(asr, "blank_runs", _blank_runs)
        monkeypatch.setattr(
            asr, "_load_asr_model_for_features", lambda: (_Model(), object())
        )
        return path, seen, captured

    def test_the_windows_handed_to_the_encoder_overlap(self, wired) -> None:
        path, seen, _ = wired
        runs, _duration, source = asr._blank_runs_for_audio(str(path))
        assert source == "alignment_head_blank_runs"
        assert runs == [(1.0, 2.0)]
        assert len(seen) > 1
        starts = [int(piece[0]) for piece in seen]
        # 15 encoder frames a side, so consecutive windows share 30 frames of
        # audio: hop = (390 - 30) frames, not the full 390.
        hop_samples = int(round((390 - 30) * 16000 / 13))
        for earlier, later in zip(starts, starts[1:]):
            assert later - earlier == hop_samples

    def test_no_frame_is_dropped_or_counted_twice(self, wired) -> None:
        """What the seam fix must not break while fixing context: the timeline.

        `blank_runs` reads frame indices as seconds, so a duplicated or missing
        frame at a seam does not look like an error - it shifts every cut after
        it by 38.5 ms per occurrence.
        """
        path, _seen, captured = wired
        asr._blank_runs_for_audio(str(path))
        values = [int(v) for v in captured["log_probs"][:, 0].tolist()]
        expected_frames = int(round(self.SECONDS * 13))
        assert values[::2] == list(range(expected_frames))
        assert values[1::2] == list(range(expected_frames))

    def test_every_kept_frame_came_from_a_window_that_had_its_context(
        self, wired
    ) -> None:
        """The point of the exercise, stated as the property it buys."""
        path, seen, captured = wired
        asr._blank_runs_for_audio(str(path))
        kept = [int(v) for v in captured["log_probs"][::2, 0].tolist()]
        spans = []
        for piece in seen:
            base = int(round(float(piece[0]) * 13 / 16000))
            spans.append((base, base + int(round(len(piece) * 13 / 16000))))
        last_frame = kept[-1]
        for frame in kept:
            best = max(
                min(frame - begin, end - 1 - frame)
                for begin, end in spans
                if begin <= frame < end
            )
            # 15 frames on each side, except where the file itself ends.
            assert best >= min(15, frame, last_frame - frame)


class TestSpansDigest:
    """Counts alone do not identify a set of cuts.

    Caches keyed on the boundary signature hold word timings measured against
    the exact chunk boundaries. Encoding the blank-run pass at a different batch
    size was measured to move the run count by ±3 out of ~215 - so different
    geometry can and sometimes does carry the same counts, and reusing timings
    across it would put words on the wrong side of a cut.
    """

    def test_different_cuts_produce_different_digests(self) -> None:
        assert asr._spans_digest([(0.0, 10.0), (10.0, 20.0)]) != asr._spans_digest(
            [(0.0, 11.0), (11.0, 20.0)]
        )

    def test_the_same_cuts_produce_the_same_digest(self) -> None:
        spans = [(0.0, 10.0), (10.0, 20.5)]
        assert asr._spans_digest(spans) == asr._spans_digest(list(spans))

    def test_the_same_count_with_different_geometry_still_differs(self) -> None:
        """The exact case the count-only signature could not see."""
        assert asr._spans_digest([(0.0, 5.0), (5.0, 9.0)]) != asr._spans_digest(
            [(0.0, 4.0), (4.0, 9.0)]
        )

    def test_sub_millisecond_noise_does_not_invalidate_a_cache(self) -> None:
        """Float noise below a millisecond cannot move a subtitle, and an exact
        digest would throw away good cached timings over it."""
        assert asr._spans_digest([(0.0, 10.0)]) == asr._spans_digest([(0.0, 10.00004)])

    def test_the_signature_carries_it(
        self, audio: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ASR_ALIGNMENT_HEAD_PATH", raising=False)
        spans = asr._build_processing_spans(str(audio))
        signature = asr._LAST_BOUNDARY_SIGNATURE["chunking"]
        assert signature["spans_sha256"] == asr._spans_digest(spans)

    def test_the_runtime_signature_version_moved_with_it(self) -> None:
        """Adding a key changes cache identity; the version says so out loud."""
        assert asr._get_asr_runtime_signature()["version"] == 10


class TestRetirement:
    def test_the_dead_chain_is_gone_from_the_transcription_path(self) -> None:
        """Named individually so a partial revert cannot pass silently."""
        for name in (
            "_boundary_config",
            "_apply_pre_asr_cueqc",
            "_run_inner_after_pre_asr_cueqc",
            "_pre_asr_candidates_for_spans",
            "_annotate_scorer_stats_on_packed_chunks",
            "_BoundaryProcessingContext",
        ):
            assert not hasattr(asr, name), f"{name} should have been retired"

    def test_chunking_never_calls_a_boundary_stage_model(self) -> None:
        source = (PROJECT_ROOT / "src" / "asr" / "pipeline.py").read_text(
            encoding="utf-8"
        )
        for symbol in (
            "load_outer_edge_refiner_v3",
            "load_acoustic_split_v4_planner",
            "load_inner_edge_refiner_v2",
            "require_boundary_pipeline_ready",
        ):
            assert symbol not in source, f"{symbol} still referenced"
