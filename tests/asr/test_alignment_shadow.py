from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from asr import alignment, alignment_shadow
from asr.local_backend import LocalAsrBackend
from asr.subtitle_timing import build_aligned_word_timestamps


class _FakeShadowHead:
    def __init__(self, spans, extent):
        self.spans = spans
        self.extent = extent

    def align_extent(self, _features, _text):
        return self.spans, self.extent[0], self.extent[1]


def _span(start: float, end: float, char: str = "あ") -> alignment.CharSpan:
    return alignment.CharSpan(
        char=char,
        index=0,
        start_frame=0,
        end_frame=1,
        start_s=start,
        end_s=end,
        score=-0.1,
    )


def test_shadow_comparison_is_boundary_only_and_signed() -> None:
    primary_words = [
        {"start": 1.0, "end": 2.0, "word": "あ", "timestamp_kind": "ctc_forced_alignment"}
    ]
    original = [dict(primary_words[0])]
    result = alignment_shadow.compare_alignment_heads(
        primary_words=primary_words,
        primary_timing_meta={"alignment_score": -0.2},
        shadow_head=_FakeShadowHead([_span(1.1, 1.9)], (1.1, 1.9)),
        features=np.zeros((3, 4), dtype=np.float32),
        text="あ",
        window_start=0.0,
        window_end=3.0,
    )
    assert result["status"] == "ok"
    assert result["onset_delta_ms"] == pytest.approx(100.0)
    assert result["end_delta_ms"] == pytest.approx(-100.0)
    assert primary_words == original, "shadow comparison may not mutate official words"


def test_local_backend_returns_primary_words_when_shadow_disagrees(monkeypatch) -> None:
    backend = LocalAsrBackend("cpu")
    primary = [_span(1.0, 2.0)]
    monkeypatch.setattr(
        backend,
        "_align_characters",
        lambda *_args, **_kwargs: (primary, (1.0, 2.0)),
    )
    monkeypatch.setattr(
        backend,
        "_resolve_shadow_alignment_head",
        lambda _log: _FakeShadowHead([_span(1.2, 1.8)], (1.2, 1.8)),
    )
    expected_words, _, _ = build_aligned_word_timestamps(
        "あ", primary, 0.0, 3.0, (1.0, 2.0)
    )
    result, _log = backend._use_boundary_timing_result(
        master_text="あ",
        raw_master_text="あ",
        duration=3.0,
        detected_language="Japanese",
        normalized_path="unused.wav",
        timing_start=0.0,
        timing_end=3.0,
        timing_window_source="chunk",
        log=[],
        cached_features=np.zeros((3, 4), dtype=np.float32),
    )
    assert [
        {key: word[key] for key in ("start", "end", "word", "timestamp_kind")}
        for word in result["words"]
    ] == [
        {key: word[key] for key in ("start", "end", "word", "timestamp_kind")}
        for word in expected_words
    ]
    assert result["timing_meta"]["alignment_shadow"]["onset_delta_ms"] == pytest.approx(200.0)


def test_finalize_log_names_the_checkpoint_file_not_the_algorithm_tag(monkeypatch) -> None:
    """The log used to print ALIGNED_TIMING_SOURCE ("ctc_forced_alignment_v1"),
    an algorithm-version tag easily misread as the checkpoint's own version.
    Show the file that actually ran instead."""
    backend = LocalAsrBackend("cpu")
    primary = [_span(1.0, 2.0)]
    monkeypatch.setattr(
        backend,
        "_align_characters",
        lambda *_args, **_kwargs: (primary, (1.0, 2.0)),
    )
    backend._alignment_head = SimpleNamespace(
        checkpoint_path="D:/models/ctc_aligner_jav_vocalisation_v2.pt"
    )

    _result, log = backend._use_boundary_timing_result(
        master_text="あ",
        raw_master_text="あ",
        duration=3.0,
        detected_language="Japanese",
        normalized_path="unused.wav",
        timing_start=0.0,
        timing_end=3.0,
        timing_window_source="chunk",
        log=[],
        cached_features=np.zeros((3, 4), dtype=np.float32),
    )

    assert any("ctc_aligner_jav_vocalisation_v2.pt" in line for line in log)
    assert not any("ctc_forced_alignment_v1" in line for line in log)


def test_run_details_are_lifted_to_absolute_audio_time(monkeypatch) -> None:
    monkeypatch.setenv(alignment_shadow.SHADOW_HEAD_PATH_ENV, "candidate.pt")
    prepared = [
        (
            {
                "text": "あ",
                "timing_meta": {
                    "alignment_shadow": {
                        "schema": alignment_shadow.SHADOW_COMPARISON_SCHEMA,
                        "status": "ok",
                        "primary_start_s": 1.0,
                        "primary_end_s": 2.0,
                        "shadow_start_s": 1.1,
                        "shadow_end_s": 1.9,
                        "max_abs_delta_ms": 100.0,
                    }
                },
            },
            [],
        )
    ]
    details = alignment_shadow.build_run_details(
        [{"index": 7, "start": 30.0, "end": 35.0}], prepared
    )
    assert details is not None
    row = details["comparisons"][0]
    assert row["primary_start_abs_s"] == pytest.approx(31.0)
    assert row["shadow_end_abs_s"] == pytest.approx(31.9)
    assert details["eligible_disagreement_count"] == 1


def test_shadow_run_persists_outside_disposable_job_dir(monkeypatch, tmp_path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    details = {
        "schema": alignment_shadow.SHADOW_RUN_SCHEMA,
        "comparisons": [{"status": "ok"}],
    }
    path = alignment_shadow.persist_shadow_run(
        run_details=details,
        video_path=video,
        video_duration_s=12.5,
        job_id="job/unsafe",
        audio_cache_key="abcdef1234567890",
        root=tmp_path / "observations",
    )
    assert path is not None and path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source_video_path"] == str(video.resolve())
    assert payload["source_video_duration_s"] == 12.5
    assert "/" not in path.name


def test_shadow_without_primary_does_not_encode_features(monkeypatch, caplog) -> None:
    backend = LocalAsrBackend("cpu")
    backend.model = object()
    backend.processor = object()
    encoded: list[list[str]] = []
    monkeypatch.setattr(backend, "_resolve_alignment_head", lambda _log: None)
    monkeypatch.setattr(alignment_shadow, "shadow_enabled", lambda: True)
    monkeypatch.setattr(
        backend,
        "_encode_chunk_features",
        lambda paths: encoded.append(paths) or {},
    )
    monkeypatch.setattr(
        backend,
        "_finalize_group",
        lambda group, _cache: [({}, []) for _item in group],
    )
    text_results = [
        {
            "text": "あ",
            "normalized_path": "unused.wav",
        }
    ]

    with caplog.at_level(logging.WARNING, logger="asr.local_backend"):
        backend.finalize_text_results(text_results)
        backend.finalize_text_results(text_results)

    assert encoded == []
    messages = [
        record.getMessage()
        for record in caplog.records
        if "alignment shadow disabled" in record.getMessage()
    ]
    assert messages == [
        "alignment shadow disabled: primary alignment head is unavailable"
    ]
