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
