"""Targeted regression tests for ASR & audio performance, VRAM, and correctness fixes."""

from __future__ import annotations

import concurrent.futures
import gc
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr import (  # noqa: E402
    alignment,
    audio_encoder,
    chunking,
    cue_features,
    decode_guard,
    encoder_features,
    local_backend,
    pipeline,
    postgate,
    result_cache,
    subtitle_timing,
    transcribe,
    vram_sizing,
)
from asr.backends import base as backend_base, qwen as qwen_backend  # noqa: E402
from audio import loading as audio_loading  # noqa: E402


def _write_pcm_wav(path: Path, *, seconds: float = 0.25, rate: int = 16000) -> Path:
    nframes = max(1, int(round(seconds * rate)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * nframes, *([0] * nframes)))
    return path


def _test_vocab(chars: list[str]) -> alignment.AlignmentVocab:
    return alignment.AlignmentVocab.from_counts({ch: 1 for ch in chars})


# ---------------------------------------------------------------------------
# P0-7: VRAM budget enforced on allocated memory, reserved probed for diagnostics
# ---------------------------------------------------------------------------


def test_vram_budget_ignores_high_reserved_when_allocated_within_budget(monkeypatch) -> None:
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "4000")
    mb = 1024 * 1024
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _d=0: 3200 * mb)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _d=0: 3100 * mb)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _d=0: 6200 * mb)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _d=0: 6100 * mb)

    stages: list[str] = []
    transcribe._enforce_vram_budget("batch_1", stages.append)
    assert stages == []


def test_vram_budget_raises_when_allocated_exceeds_budget(monkeypatch) -> None:
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "4000")
    mb = 1024 * 1024
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _d=0: 4500 * mb)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _d=0: 4400 * mb)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _d=0: 5000 * mb)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _d=0: 4900 * mb)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d=0: (3000 * mb, 8000 * mb))

    stages: list[str] = []
    with pytest.raises(RuntimeError, match=r"allocated_mb=4500\.0"):
        transcribe._enforce_vram_budget("batch_2", stages.append)
    assert stages and "GPU VRAM budget exceeded" in stages[0]


def test_vram_budget_still_propagates_fatal_cuda_error_from_reserved_probe(monkeypatch) -> None:
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "4000")
    mb = 1024 * 1024
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _d=0: 1000 * mb)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _d=0: 1000 * mb)

    def _boom(_d=0):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(torch.cuda, "max_memory_reserved", _boom)

    with pytest.raises(RuntimeError, match="illegal memory access"):
        transcribe._enforce_vram_budget("batch_3", None)


def test_pipeline_snapshot_vram_budget_shares_allocated_rule(monkeypatch) -> None:
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "4000")
    pipeline._enforce_vram_budget_from_snapshot(
        {
            "stage": "split_done",
            "max_allocated_mb": 3500.0,
            "allocated_mb": 3400.0,
            "max_reserved_mb": 6900.0,
        }
    )
    with pytest.raises(RuntimeError, match=r"allocated_mb=4200\.0"):
        pipeline._enforce_vram_budget_from_snapshot(
            {
                "stage": "split_done",
                "max_allocated_mb": 4200.0,
                "allocated_mb": 4000.0,
                "max_reserved_mb": 6900.0,
            }
        )


# ---------------------------------------------------------------------------
# P0-8: _robust_bounds punctuation filtering, NFD NFKC sync, finite scores
# ---------------------------------------------------------------------------


def _make_span(
    char: str,
    idx: int,
    start_s: float,
    end_s: float,
    *,
    start_frame: int = 1,
    end_frame: int = 2,
    score: float = -0.5,
) -> alignment.CharSpan:
    return alignment.CharSpan(
        char=char,
        index=idx,
        start_frame=start_frame,
        end_frame=end_frame,
        start_s=start_s,
        end_s=end_s,
        score=score,
    )


def test_robust_bounds_filters_zero_width_punctuation_and_trims_dragged_edges() -> None:
    # Leading zero-width punctuation ("。" * 2) sits at dragged start 0.2s,
    # followed by dragged "あ" at 0.2s, while real speech "いうえお" is 4.0..4.7s.
    text = "。。あいうえお"
    spans = [
        _make_span("。", 0, 0.2, 0.2, start_frame=5, end_frame=5, score=float("-inf")),
        _make_span("。", 1, 0.2, 0.2, start_frame=5, end_frame=5, score=float("-inf")),
        _make_span("あ", 2, 0.2, 0.3, start_frame=5, end_frame=8, score=-0.4),
        _make_span("い", 3, 4.0, 4.1, start_frame=104, end_frame=107, score=-0.2),
        _make_span("う", 4, 4.2, 4.3, start_frame=109, end_frame=112, score=-0.2),
        _make_span("え", 5, 4.4, 4.5, start_frame=114, end_frame=117, score=-0.2),
        _make_span("お", 6, 4.6, 4.7, start_frame=119, end_frame=122, score=-0.2),
    ]
    words, method, meta = subtitle_timing.build_aligned_word_timestamps(
        text, spans, 0.0, 6.0
    )
    assert method == "ctc_forced_alignment"
    assert meta["boundary_trimmed"] is True
    # Acoustic gaps are [3.7, 0.1, 0.1, 0.1] -> median 0.1 (clamped to 0.2 window).
    # _robust_bounds pulls dragged leading "あ" (0.2s) in to 4.0 - 0.2 = 3.8s,
    # and clamps the leading zero-width punctuation into the trimmed extent.
    assert words[0]["start"] == pytest.approx(3.8)
    assert words[2]["start"] == pytest.approx(3.8)
    # Segment score ignores -inf punctuation spans: mean(-0.4, -0.2, -0.2, -0.2, -0.2) = -0.24.
    assert meta["alignment_score"] == pytest.approx(-0.24)
    assert words[0]["alignment_score"] is None

    # Trailing dragged character plus zero-width punctuation ("あいうえお。。").
    tail_text = "あいうえお。。"
    tail_spans = [
        _make_span("あ", 0, 1.0, 1.1, start_frame=26, end_frame=29, score=-0.3),
        _make_span("い", 1, 1.2, 1.3, start_frame=31, end_frame=34, score=-0.3),
        _make_span("う", 2, 1.4, 1.5, start_frame=36, end_frame=39, score=-0.3),
        _make_span("え", 3, 1.6, 1.7, start_frame=41, end_frame=44, score=-0.3),
        _make_span("お", 4, 5.5, 5.6, start_frame=143, end_frame=146, score=-0.3),
        _make_span("。", 5, 5.6, 5.6, start_frame=146, end_frame=146, score=float("-inf")),
        _make_span("。", 6, 5.6, 5.6, start_frame=146, end_frame=146, score=float("-inf")),
    ]
    tail_words, _, tail_meta = subtitle_timing.build_aligned_word_timestamps(
        tail_text, tail_spans, 0.0, 6.0
    )
    assert tail_meta["boundary_trimmed"] is True
    assert tail_words[-1]["end"] == pytest.approx(1.9)


def test_alignment_tokens_nfkc_keeps_nfd_combining_chars_aligned() -> None:
    nfd_text = "か\u3099あいう"  # NFD "が" + "あいう" -> 4 normalized chars
    assert alignment.normalize_text(nfd_text) == "があいう"
    spans = [
        _make_span("が", 0, 1.0, 1.2, start_frame=26, end_frame=31, score=-0.1),
        _make_span("あ", 1, 1.2, 1.4, start_frame=31, end_frame=36, score=-0.2),
        _make_span("い", 2, 1.4, 1.6, start_frame=36, end_frame=41, score=-0.3),
        _make_span("う", 3, 1.6, 1.8, start_frame=41, end_frame=46, score=-0.4),
    ]
    words, method, meta = subtitle_timing.build_aligned_word_timestamps(
        nfd_text, spans, 0.0, 3.0
    )
    assert method == "ctc_forced_alignment"
    assert [w["word"] for w in words] == ["が", "あ", "い", "う"]
    assert [w["start"] for w in words] == pytest.approx([1.0, 1.2, 1.4, 1.6])
    assert meta["alignment_score"] == pytest.approx(-0.25)


# ---------------------------------------------------------------------------
# P0-9, P0-13, P1-2, P1-3, P2-1: alignment.py
# ---------------------------------------------------------------------------


def test_forced_align_raises_on_all_neg_inf_log_probs() -> None:
    log_probs = torch.full((8, 6), float("-inf"), dtype=torch.float32)
    with pytest.raises(ValueError, match="no finite alignment path"):
        alignment.forced_align(log_probs, [2, 3], chars="あい")


def test_forced_align_matches_brute_force_with_blank_bias_and_repeats() -> None:
    torch.manual_seed(7)
    raw = torch.randn((9, 6), dtype=torch.float32)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = [2, 2, 3]
    bias = 0.35

    spans = alignment.forced_align(
        log_probs, targets, chars="ああい", upsample=2, blank_bias=bias
    )

    # Brute-force enumerate all valid monotonic CTC paths over T=9, states=7.
    extended = [0, 2, 0, 2, 0, 3, 0]
    searched = log_probs.clone()
    searched[:, 0] -= bias
    best_score = float("-inf")
    best_path: list[int] | None = None

    def _dfs(t: int, s: int, acc: float, path: list[int]) -> None:
        nonlocal best_score, best_path
        acc += float(searched[t, extended[s]])
        if t == 8:
            if s in (5, 6) and acc > best_score:
                best_score = acc
                best_path = list(path)
            return
        candidates = [s]
        if s + 1 < len(extended):
            candidates.append(s + 1)
        if s + 2 < len(extended) and extended[s + 2] != 0 and extended[s + 2] != extended[s]:
            candidates.append(s + 2)
        for nxt in candidates:
            path.append(nxt)
            _dfs(t + 1, nxt, acc, path)
            path.pop()

    for start_s in (0, 1):
        _dfs(0, start_s, 0.0, [start_s])

    assert best_path is not None
    for idx, span in enumerate(spans):
        odd_state = 2 * idx + 1
        frames = [t for t, s in enumerate(best_path) if s == odd_state]
        assert span.start_frame == frames[0]
        assert span.end_frame == frames[-1] + 1
        expected_unbiased = float(log_probs[frames, targets[idx]].mean().item())
        assert span.score == pytest.approx(expected_unbiased, abs=1e-5)


def test_blank_runs_and_non_speech_runs_accept_1d_argmax_and_2d_tensor() -> None:
    argmax_ids = np.array([0, 0, 0, 2, 2, 0, 0, 0, 0, 3, 1, 1, 0, 0], dtype=np.int32)
    log_probs = torch.full((len(argmax_ids), 5), -10.0, dtype=torch.float32)
    for t, idx in enumerate(argmax_ids):
        log_probs[t, int(idx)] = 0.0

    assert alignment.blank_runs(argmax_ids, min_seconds=0.1, upsample=2) == alignment.blank_runs(
        log_probs, min_seconds=0.1, upsample=2
    )
    assert alignment.blank_runs(
        argmax_ids, min_seconds=0.1, upsample=2, silent_classes=frozenset({0, 1})
    ) == alignment.blank_runs(
        log_probs, min_seconds=0.1, upsample=2, silent_classes=frozenset({0, 1})
    )

    posteriors = np.zeros((20, 3), dtype=np.float32)
    posteriors[0:6, 0] = 0.9
    posteriors[6:10, 2] = 0.9
    posteriors[10:20, 1] = 0.8
    runs = alignment.non_speech_runs(
        posteriors, speech_threshold=0.5, min_seconds=0.2, upsample=2
    )
    assert len(runs) == 2
    assert runs[0] == (
        pytest.approx(0.0),
        pytest.approx(alignment.frame_to_seconds(6, upsample=2)),
    )
    assert runs[1] == (
        pytest.approx(alignment.frame_to_seconds(10, upsample=2)),
        pytest.approx(alignment.frame_to_seconds(20, upsample=2)),
    )


def test_span_class_shares_float32_and_speech_max_run_s() -> None:
    posteriors = np.zeros((40, 3), dtype=np.float32)
    posteriors[:, alignment.FRAME_CLASS_SILENCE] = 1.0
    # Frames 10..15 speech (5 frames), 15..17 vocalisation, 17..25 speech (8 frames)
    posteriors[10:15] = [0.0, 0.1, 0.9]
    posteriors[15:17] = [0.0, 0.9, 0.1]
    posteriors[17:25] = [0.1, 0.0, 0.9]

    stats = alignment.span_class_shares(
        posteriors,
        start_s=alignment.frame_to_seconds(10, upsample=2),
        end_s=alignment.frame_to_seconds(25, upsample=2),
        upsample=2,
    )
    assert stats is not None
    assert stats["speech_max_run_s"] == pytest.approx(
        alignment.frame_to_seconds(8, upsample=2), abs=1e-5
    )
    assert stats["speech"] == pytest.approx((13 * 0.9 + 2 * 0.1) / 15, abs=1e-5)
    assert stats["vocalisation"] == pytest.approx((5 * 0.1 + 2 * 0.9) / 15, abs=1e-5)
    assert stats["frames"] == 15


def test_speech_extent_boundary_slice_matches_expected_walk() -> None:
    # 80 frames; blank outside 22..48 except non-blank at 18 and 52 (boundaries).
    # speech_extent walks outward through BLANK frames up to the first non-blank!
    log_probs = torch.full((80, 5), -8.0, dtype=torch.float32)
    log_probs[:, 0] = 0.0  # blank everywhere
    log_probs[18, 0] = -8.0
    log_probs[18, 2] = 0.0  # non-blank stop at 18 -> onset stops at 19
    log_probs[52, 0] = -8.0
    log_probs[52, 2] = 0.0  # non-blank stop at 52 -> coda stops at 52

    spans = [
        _make_span(
            "あ",
            0,
            alignment.frame_to_seconds(22, upsample=2),
            alignment.frame_to_seconds(25, upsample=2),
            start_frame=22,
            end_frame=25,
        ),
        _make_span(
            "い",
            1,
            alignment.frame_to_seconds(45, upsample=2),
            alignment.frame_to_seconds(48, upsample=2),
            start_frame=45,
            end_frame=48,
        ),
    ]
    start_s, end_s = alignment.speech_extent(log_probs, spans, upsample=2)
    assert start_s == pytest.approx(alignment.frame_to_seconds(19, upsample=2))
    # CODA_EXTEND_MAX_S = 0.15s = 3 frames at 26 fps -> capped at 48 + 3 = 51.
    assert end_s == pytest.approx(alignment.frame_to_seconds(51, upsample=2))


def test_alignment_head_gpu_batch_matches_single_clip_and_argmax_and_skip_ctc() -> None:
    torch.manual_seed(11)
    vocab = _test_vocab(["あ", "い", "う", "え"])
    module = alignment.build_head(
        vocab_size=vocab.size,
        input_dim=16,
        hidden_dim=16,
        upsample=2,
        blocks=2,
        dropout=0.0,
        frame_classes=3,
    )
    module.eval()
    head = alignment.AlignmentHead(
        module=module, vocab=vocab, upsample=2, device="cpu", blank_bias=0.0
    )

    f0 = torch.randn((5, 16), dtype=torch.float32)
    f1 = torch.randn((9, 16), dtype=torch.float32)
    concat = torch.cat([f0, f1], dim=0)

    ctc_ref_0, frames_ref_0 = head.log_probs_with_frames(f0)
    ctc_ref_1, frames_ref_1 = head.log_probs_with_frames(f1)

    ctc_batch, frames_batch = head.forward_batch_from_gpu_features(
        concat, [5, 9], need_ctc_log_probs=True, argmax_only=False, need_frames=True
    )
    assert torch.allclose(ctc_batch[0], ctc_ref_0, atol=1e-5)
    assert torch.allclose(ctc_batch[1], ctc_ref_1, atol=1e-5)
    assert np.allclose(frames_batch[0], frames_ref_0.exp().numpy(), atol=1e-5)
    assert np.allclose(frames_batch[1], frames_ref_1.exp().numpy(), atol=1e-5)

    argmax_batch, frames_argmax = head.forward_batch_from_gpu_features(
        concat, [5, 9], need_ctc_log_probs=True, argmax_only=True, need_frames=True
    )
    assert argmax_batch[0].dtype == np.int32
    assert np.array_equal(argmax_batch[0], ctc_ref_0.argmax(dim=-1).numpy())
    assert np.array_equal(argmax_batch[1], ctc_ref_1.argmax(dim=-1).numpy())
    assert frames_argmax[0].shape == (10, 3)

    skipped_ctc, frames_only = head.forward_batch_from_gpu_features(
        concat, [5, 9], need_ctc_log_probs=False, argmax_only=True, need_frames=True
    )
    assert skipped_ctc == [None, None]
    assert np.allclose(frames_only[1], frames_ref_1.exp().numpy(), atol=1e-5)

    # Empty batch edge case (all zero lengths).
    empty_ctc, empty_frames = head.forward_batch_from_gpu_features(
        torch.empty((0, 16), dtype=torch.float32),
        [0, 0],
        need_ctc_log_probs=True,
        argmax_only=True,
        need_frames=True,
    )
    assert all(arr.shape == (0,) and arr.dtype == np.int32 for arr in empty_ctc)
    assert all(arr.shape == (0, 3) for arr in empty_frames)


# ---------------------------------------------------------------------------
# P0-13, P1-1, P1-5, P1-8: backend lifecycle, executor reuse, shared pipeline
# ---------------------------------------------------------------------------


def test_local_backend_unload_and_close_release_alignment_head_when_model_is_none(
    monkeypatch,
) -> None:
    cleared: list[str] = []
    monkeypatch.setattr(
        local_backend, "_clear_cuda_cache", lambda device, **_kw: cleared.append(device)
    )
    backend = local_backend.LocalAsrBackend(device="cuda")
    backend.model = None
    backend._alignment_head = object()

    backend.unload_model()
    assert backend._alignment_head is False
    assert cleared == ["cuda"]

    # close() also releases _alignment_head even if model is already None.
    backend._alignment_head = object()
    backend.close()
    assert backend._alignment_head is False
    assert cleared == ["cuda", "cuda"]


def test_local_backend_reuses_executor_and_skips_normal_cuda_cache_clear(
    monkeypatch, tmp_path: Path
) -> None:
    wav_path = _write_pcm_wav(tmp_path / "c0.wav", seconds=0.2)
    created_pools: list[object] = []
    shutdown_calls: list[tuple[object, bool]] = []
    clear_calls: list[str] = []

    class _FakeFuture:
        def __init__(self, result_val=None, timeout_err: bool = False):
            self._result_val = result_val
            self._timeout_err = timeout_err
            self._done = not timeout_err

        def result(self, timeout=None):
            if self._timeout_err:
                raise concurrent.futures.TimeoutError()
            return self._result_val

        def cancel(self):
            return False

        def done(self):
            return self._done

    class _FakePool:
        def __init__(self, max_workers=1):
            self.max_workers = max_workers
            created_pools.append(self)

        def submit(self, fn, *args, **kwargs):
            if getattr(backend, "_force_timeout", False):
                return _FakeFuture(timeout_err=True)
            return _FakeFuture(result_val=fn(*args, **kwargs))

        def shutdown(self, wait=True, cancel_futures=False):
            shutdown_calls.append((self, bool(wait)))

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _FakePool)
    monkeypatch.setattr(
        concurrent.futures,
        "wait",
        lambda futures, timeout=None: (
            (set(), set(futures))
            if any(not f.done() for f in futures)
            else (set(futures), set())
        ),
    )
    monkeypatch.setattr(
        local_backend, "_clear_cuda_cache", lambda device, **_kw: clear_calls.append(device)
    )
    monkeypatch.setattr(local_backend, "_transcription_timeout_s", lambda: 1.0)
    monkeypatch.setattr(local_backend, "_resolve_asr_batch_size", lambda: 2)

    backend = local_backend.LocalAsrBackend(device="cuda")
    backend.model = object()
    backend.processor = object()
    monkeypatch.setattr(
        backend,
        "_transcribe_native",
        lambda paths, _lang: [
            local_backend.NativeAsrTranscription(language="Japanese", text="ok")
            for _ in paths
        ],
    )

    out1 = backend.transcribe_texts([str(wav_path)])
    out2 = backend.transcribe_texts([str(wav_path)])
    assert out1[0]["text"] == "ok" and out2[0]["text"] == "ok"
    # Single executor reused across normal batches; NO per-batch CUDA cache clear!
    assert len(created_pools) == 1
    assert clear_calls == []

    # Timeout detaches the stuck pool into _zombie_executor, skips _clear_cuda_cache
    # while zombie is active, and survives a missing WAV path via _get_wav_duration_or_zero.
    backend._force_timeout = True
    missing_wav = tmp_path / "missing.wav"
    out_timeout = backend.transcribe_texts([str(missing_wav)])
    assert out_timeout[0]["text"] == ""
    assert out_timeout[0]["duration"] == 0.0
    assert out_timeout[0]["asr_generation"]["error_kind"] == "timeout"
    assert clear_calls == []
    assert backend._executor is None
    assert backend._zombie_executor is created_pools[0]
    assert shutdown_calls == []

    # Clear zombie reference before close() so test doesn't block on fake future.
    backend._zombie_future = None
    backend._zombie_executor = None
    backend.model = None
    backend.close()


def test_local_backend_env_language_and_timeout_refresh_dynamically(monkeypatch) -> None:
    monkeypatch.setenv("ASR_LANGUAGE", "English")
    monkeypatch.setenv("TRANSCRIPTION_TIMEOUT_S", "42.5")
    assert local_backend._asr_language() == "English"
    assert local_backend._transcription_timeout_s() == pytest.approx(42.5)
    monkeypatch.delenv("ASR_LANGUAGE")
    monkeypatch.delenv("TRANSCRIPTION_TIMEOUT_S")
    assert local_backend._asr_language() == "Japanese"
    assert local_backend._transcription_timeout_s() == pytest.approx(180.0)


def test_qwen3_asr_encoder_close_runs_gc_before_empty_cache(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))

    enc = encoder_features.Qwen3AsrEncoder.__new__(encoder_features.Qwen3AsrEncoder)
    enc.device = torch.device("cuda")
    enc.model = object()
    enc.processor = object()
    enc.close()
    assert events == ["gc", "empty_cache"]


def test_pipeline_blank_runs_reuses_backend_and_skips_ctc_in_speech_mode(
    monkeypatch, tmp_path: Path
) -> None:
    wav_path = _write_pcm_wav(tmp_path / "sample.wav", seconds=1.0)
    calls: list[dict] = []

    class _FakeGpuHead:
        upsample = 2
        context_frames = 0
        frame_head_available = True
        silent_classes = frozenset({0})
        vocab = _test_vocab(["あ", "い"])

        def forward_batch_from_gpu_features(
            self,
            features,
            lengths,
            *,
            need_ctc_log_probs: bool = True,
            argmax_only: bool = False,
            need_frames: bool = True,
        ):
            calls.append(
                {
                    "need_ctc_log_probs": need_ctc_log_probs,
                    "argmax_only": argmax_only,
                    "need_frames": need_frames,
                    "lengths": list(lengths),
                }
            )
            ctc = (
                [None for _ in lengths]
                if not need_ctc_log_probs
                else [np.zeros(length * 2, dtype=np.int32) for length in lengths]
            )
            frames = [
                np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (length * 2, 1))
                for length in lengths
            ]
            return ctc, frames

    fake_head = _FakeGpuHead()

    class _FakeBackend:
        device = "cuda"
        dtype = torch.float32
        model = SimpleNamespace(device="cuda", dtype=torch.float32)
        processor = object()
        _alignment_head = fake_head

        def load(self, on_stage=None):
            return None

        def _get_alignment_head(self):
            return self._alignment_head

    monkeypatch.setenv("ASR_ALIGNMENT_HEAD_PATH", str(tmp_path / "head.pt"))
    monkeypatch.setenv("ASR_CHUNK_MIN_PAUSE_S", "0.1")
    monkeypatch.setattr(
        pipeline,
        "_load_asr_model_for_features",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("should reuse backend")),
    )
    monkeypatch.setattr(
        audio_encoder,
        "encode_audio_features_batch",
        lambda _model, _proc, waves, **_kw: (
            torch.zeros((13 * len(waves), 8), dtype=torch.float32),
            [13 for _ in waves],
        ),
    )

    # speech mode with v2 frame head: skips CTC log-probs completely!
    monkeypatch.setenv("ASR_CHUNK_PAUSE_READING", "speech")
    runs, total_s, source, track = pipeline._blank_runs_for_audio(
        str(wav_path),
        backend=_FakeBackend(),
    )
    assert total_s == pytest.approx(1.0)
    assert source == "alignment_head_speech_runs"
    assert track is not None
    assert runs
    assert calls[-1]["need_ctc_log_probs"] is False
    assert calls[-1]["argmax_only"] is True
    assert calls[-1]["need_frames"] is True

    # blank mode: requests argmax_only=True (1D int32 instead of full (T, V) tensor).
    monkeypatch.setenv("ASR_CHUNK_PAUSE_READING", "blank")
    runs_blank, _, source_blank, _ = pipeline._blank_runs_for_audio(
        str(wav_path),
        backend=_FakeBackend(),
    )
    assert runs_blank
    assert source_blank == "alignment_head_blank_runs"
    assert calls[-1]["need_ctc_log_probs"] is True
    assert calls[-1]["argmax_only"] is True


def test_pipeline_build_processing_spans_returns_immutable_chunk_plan(
    monkeypatch, tmp_path: Path
) -> None:
    wav_path = _write_pcm_wav(tmp_path / "sample.wav", seconds=1.0)
    monkeypatch.setattr(
        pipeline,
        "_blank_runs_for_audio",
        lambda *_a, **_kw: ([(20.0, 21.0)], 45.0, "alignment_head_blank", None),
    )

    plan = pipeline._build_processing_spans(str(wav_path))
    assert isinstance(plan, pipeline.ChunkPlan)
    assert len(plan) == 2
    assert list(plan) == list(plan.spans)
    assert pipeline._LAST_BOUNDARY_SIGNATURE == plan.boundary_signature
    with pytest.raises(AttributeError):
        plan.spans = ()  # type: ignore[misc]


def test_pipeline_transcribe_and_align_local_reuses_single_backend(
    monkeypatch, tmp_path: Path
) -> None:
    wav_path = _write_pcm_wav(tmp_path / "sample.wav", seconds=0.5)
    created_backends: list[object] = []
    unloaded: list[object] = []
    closed: list[object] = []

    class _FakeBackend:
        def __init__(self, device="cuda"):
            self.device = device
            self.model = object()
            self._alignment_head = object()
            self.cumulative_load_s = 0.0
            created_backends.append(self)

        def load(self, on_stage=None):
            return None

        def unload_model(self, on_stage=None):
            self.model = None
            self._alignment_head = False
            unloaded.append(self)

        def close(self):
            closed.append(self)

    seen_backend_in_spans: list[object] = []
    seen_backend_in_transcribe: list[object] = []
    seen_backend_in_align: list[object] = []

    monkeypatch.setattr(pipeline, "_resolve_asr_backend", lambda device: _FakeBackend(device))
    monkeypatch.setattr(pipeline, "alignment_head_configured", lambda: False)
    monkeypatch.setattr(
        pipeline,
        "_build_processing_spans",
        lambda _wav, *, on_stage=None, backend=None: (
            seen_backend_in_spans.append(backend)
            or pipeline.ChunkPlan(
                spans=((0.0, 0.5),),
                boundary_signature={"chunking": {"chunk_count": 1}},
                cut_provenance={"chunk_count": 1},
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_wav_chunks",
        lambda _wav, _spans, on_stage=None: (
            tmp_path,
            [
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 0.5,
                    "duration": 0.5,
                    "path": str(wav_path),
                    "audio_sha256": "deadbeef",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_transcribe_asr_chunks_text_only",
        lambda backend, _chunks, _label, on_stage=None: (
            seen_backend_in_transcribe.append(backend)
            or (
                [{"text": "テスト", "duration": 0.5, "normalized_path": str(wav_path)}],
                {"text_transcribe_s": 0.01},
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_align_TRANSCRIPTION_results",
        lambda backend, _texts, on_stage=None: (
            seen_backend_in_align.append(backend)
            or (
                [
                    (
                        {
                            "text": "テスト",
                            "words": [
                                {"start": 0.0, "end": 0.5, "word": "テスト"}
                            ],
                        },
                        [],
                    )
                ],
                {"alignment_s": 0.01},
            )
        ),
    )
    monkeypatch.setattr(pipeline, "_release_stage_gpu_cache", lambda *_a, **_kw: None)
    monkeypatch.setattr(pipeline, "_record_cuda_memory", lambda *_a, **_kw: None)

    segments, _log, _details = pipeline._transcribe_and_align_local(str(wav_path), "cpu")
    assert segments and segments[0]["text"] == "テスト"
    assert len(created_backends) == 1
    assert seen_backend_in_spans == created_backends
    assert seen_backend_in_transcribe == created_backends
    assert seen_backend_in_align == created_backends
    assert closed == created_backends


# ---------------------------------------------------------------------------
# P1-4, P1-6, P1-7, P2-2 ~ P2-4
# ---------------------------------------------------------------------------


def test_result_cache_finalize_signature_tracks_blank_bias_and_normalizes_model_env(
    monkeypatch, tmp_path: Path
) -> None:
    head_path = tmp_path / "head.pt"
    head_path.write_bytes(b"fake-head")
    monkeypatch.setenv("ASR_ALIGNMENT_HEAD_PATH", str(head_path))

    monkeypatch.setenv("ASR_ALIGNMENT_BLANK_BIAS", "0.0")
    sig_zero = result_cache.finalize_signature()
    monkeypatch.setenv("ASR_ALIGNMENT_BLANK_BIAS", "0.35")
    sig_bias = result_cache.finalize_signature()
    assert sig_zero is not None and sig_bias is not None
    assert sig_zero["blank_bias"] == 0.0
    assert sig_bias["blank_bias"] == 0.35
    assert sig_zero != sig_bias

    # Equivalent env spellings produce identical normalized model_signature.
    monkeypatch.setenv("ASR_FORCE_LANGUAGE", "true")
    monkeypatch.setenv("ASR_MAX_NEW_TOKENS", "auto")
    monkeypatch.setenv("ASR_REPETITION_PENALTY", "1.050")
    sig_a = result_cache.model_signature()

    monkeypatch.setenv("ASR_FORCE_LANGUAGE", "1")
    monkeypatch.setenv("ASR_MAX_NEW_TOKENS", "")
    monkeypatch.setenv("ASR_REPETITION_PENALTY", "1.05")
    sig_b = result_cache.model_signature()
    assert sig_a == sig_b


def test_extract_wav_chunks_computes_sha256_throttles_stage_and_handles_short_reads(
    monkeypatch, tmp_path: Path
) -> None:
    wav_path = _write_pcm_wav(tmp_path / "src.wav", seconds=1.0, rate=16000)
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(tmp_path / "chunks"))

    # 60 tiny spans -> stage progress should be throttled well below 60 calls.
    spans = [(i * 0.015, (i + 1) * 0.015) for i in range(60)]
    stages: list[str] = []
    _chunk_dir, infos = chunking._extract_wav_chunks(wav_path, spans, stages.append)
    assert len(infos) == 60
    assert len(stages) < 30
    assert stages[0].startswith("音频切块 1/60")
    assert stages[-1].startswith("音频切块 60/60")
    assert infos[0]["audio_sha256"] == result_cache.chunk_audio_sha256(infos[0]["path"])

    # Short read & zero-frame read handling: simulate truncated reader.
    real_open = wave.open

    class _ShortReader:
        def __init__(self, inner):
            self._inner = inner
            self._call = 0

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def getparams(self):
            return self._inner.getparams()

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def setpos(self, pos):
            self._inner.setpos(pos)

        def readframes(self, n):
            self._call += 1
            if self._call == 1:
                return self._inner.readframes(n // 2)  # short read
            return b""  # zero-frame read -> skipped

    monkeypatch.setattr(
        chunking.wave,
        "open",
        lambda p, mode="rb": _ShortReader(real_open(p, mode)) if mode == "rb" else real_open(p, mode),
    )
    _, short_infos = chunking._extract_wav_chunks(wav_path, [(0.0, 0.2), (0.2, 0.4)], None)
    assert len(short_infos) == 1
    assert short_infos[0]["index"] == 0
    assert short_infos[0]["duration"] == pytest.approx(0.1)
    assert short_infos[0]["end"] == pytest.approx(0.1)


def test_result_cache_skips_wav_read_when_audio_sha256_passed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ASR_RESULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        result_cache,
        "chunk_audio_sha256",
        lambda _p: (_ for _ in ()).throw(AssertionError("should not re-read wav")),
    )
    sig = {"model": "test", "version": 1}
    digest = "0" * 64
    result_cache.store(
        "unused.wav",
        {"text": "こんにちは"},
        audio_sha256=digest,
        signature=sig,
        cache_dir=tmp_path / "cache",
    )
    hit = result_cache.lookup(
        "unused.wav",
        audio_sha256=digest,
        signature=sig,
        cache_dir=tmp_path / "cache",
    )
    assert hit == {"text": "こんにちは"}

    aligned_res = {"alignment_mode": "ctc_forced_alignment", "text": "こんにちは", "words": []}
    result_cache.finalize_store(
        "unused.wav",
        text="こんにちは",
        result=aligned_res,
        log=["aligned"],
        audio_sha256=digest,
        signature=sig,
        cache_dir=tmp_path / "cache",
    )
    fin_hit = result_cache.finalize_lookup(
        "unused.wav",
        text="こんにちは",
        audio_sha256=digest,
        signature=sig,
        cache_dir=tmp_path / "cache",
    )
    assert fin_hit is not None
    assert fin_hit[0] == aligned_res


def test_cue_features_skip_audio_hash_punct_re_and_shifted_unit_repeat(
    monkeypatch, tmp_path: Path
) -> None:
    wav_path = _write_pcm_wav(tmp_path / "a.wav", seconds=0.5)
    monkeypatch.setattr(
        cue_features,
        "_audio_ref",
        lambda _p: (_ for _ in ()).throw(AssertionError("should not hash audio")),
    )
    chunks = [{"index": 0, "start": 0.0, "end": 0.5, "path": str(wav_path)}]
    texts = [{"text": "…（あ）・【い】…", "raw_text": "…（あ）・【い】…"}]
    assert cue_features._strip_punctuation("…（あ）・【い】…") == "あい"
    compact_values = cue_features.precompute_compact_texts(texts)
    cand = cue_features.build_candidate(
        chunk=chunks[0],
        text_result=texts[0],
        position=0,
        chunks=chunks,
        text_results=texts,
        include_audio_ref=False,
        compact_values=compact_values,
    )
    assert cand["audio"]["sha1"] == ""
    assert cand["text_features"]["raw_char_count"] == 2

    # Shifted overlapping repeat: "ab" prefix followed by 4x "bc" -> unit "bc" run=4.
    assert cue_features._max_unit_repeat("ababcbcbcbc")["run"] == 4


def test_postgate_review_all_rejects_explicit_empty_alignment_scores_list(
    tmp_path: Path,
) -> None:
    wav_path = _write_pcm_wav(tmp_path / "p.wav", seconds=0.5)
    chunks = [{"index": 0, "start": 0.0, "end": 0.5, "path": str(wav_path)}]
    texts = [{"text": "テスト", "raw_text": "テスト"}]
    cand = cue_features.build_candidate(
        chunk=chunks[0],
        text_result=texts[0],
        position=0,
        chunks=chunks,
        text_results=texts,
        include_audio_ref=False,
    )
    with pytest.raises(ValueError, match="alignment_scores"):
        postgate.review_all([cand], alignment_scores=[])


def test_transcribe_chunks_dense_batches_descending_and_skips_empty_start_log(
    monkeypatch, tmp_path: Path
) -> None:
    chunks = [
        {"index": 0, "start": 0.0, "end": 5.0, "duration": 5.0, "path": str(tmp_path / "0.wav"), "audio_sha256": "a0"},
        {"index": 1, "start": 5.0, "end": 15.0, "duration": 10.0, "path": str(tmp_path / "1.wav"), "audio_sha256": "a1"},
        {"index": 2, "start": 15.0, "end": 35.0, "duration": 20.0, "path": str(tmp_path / "2.wav"), "audio_sha256": "a2"},
        {"index": 3, "start": 35.0, "end": 43.0, "duration": 8.0, "path": str(tmp_path / "3.wav"), "audio_sha256": "a3"},
    ]
    # Mark chunk 1 cached; pending chunks are 2 (20s), 3 (8s), 0 (5s).
    monkeypatch.setattr(
        result_cache,
        "lookup",
        lambda path, **_kw: {"text": "cached"} if str(path).endswith("1.wav") else None,
    )
    monkeypatch.setattr(result_cache, "store", lambda *_a, **_kw: None)
    monkeypatch.setattr(result_cache, "result_cache_enabled", lambda: True)
    monkeypatch.setattr(result_cache, "_cache_dir", lambda _sig: tmp_path)
    monkeypatch.setattr(result_cache, "model_signature", lambda: {"v": 1})
    monkeypatch.setattr(transcribe, "_enforce_vram_budget", lambda *_a, **_kw: None)

    seen_batches: list[list[str]] = []

    class _FakeBackend:
        request_batch_size = 2
        model = object()

        def transcribe_texts(self, paths, **_kw):
            seen_batches.append([Path(p).name for p in paths])
            return [{"text": f"t:{Path(p).name}", "duration": 1.0} for p in paths]

        def unload_model(self):
            self.model = None

    results, _timings = transcribe._transcribe_asr_chunks_text_only(
        _FakeBackend(), chunks, "ASR 文本转写"
    )
    # Dense batches in descending duration order: [2.wav (20s), 3.wav (8s)] then [0.wav (5s)]
    assert seen_batches == [["2.wav", "3.wav"], ["0.wav"]]
    # Restored in original chronological chunk index order.
    assert [r["text"] for r in results] == ["t:0.wav", "cached", "t:2.wav", "t:3.wav"]

    # All cached -> no "size=0" stage message and backend is not invoked.
    monkeypatch.setattr(
        result_cache,
        "lookup",
        lambda _path, **_kw: {"text": "cached"},
    )
    stages: list[str] = []
    seen_batches.clear()
    transcribe._transcribe_asr_chunks_text_only(
        _FakeBackend(), chunks, "ASR 文本转写", on_stage=stages.append
    )
    assert seen_batches == []
    assert not any("size=0" in entry for entry in stages)


def test_rows_truncated_at_cap_detects_short_row_when_pad_id_in_stop_ids() -> None:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[0, 99], pad_token_id=0),
        config=SimpleNamespace(eos_token_id=[0, 99], pad_token_id=0),
    )
    # Row 0 (cap=2): emitted 2 non-stop tokens then padded with pad_id=0 -> TRUNCATED!
    # Row 1 (cap=4): emitted 4 non-stop tokens -> TRUNCATED!
    # Row 2 (cap=2): stopped at index 1 with eos=0 -> NOT truncated!
    suffix = torch.tensor(
        [
            [11, 12, 0, 0],
            [21, 22, 23, 24],
            [31, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    assert local_backend._rows_truncated_at_cap(suffix, model, [2, 4, 2]) == [
        True,
        True,
        False,
    ]


def _reference_detect_repetition_loop(
    suffix: torch.Tensor, max_ngram: int, min_repeats: int, min_tokens: int
) -> torch.Tensor:
    batch, length = suffix.shape
    hit = torch.zeros(batch, dtype=torch.bool, device=suffix.device)
    if length < min_tokens or max_ngram <= 0:
        return hit
    for ngram, repeats in decode_guard._windows(max_ngram, min_repeats, min_tokens):
        window = ngram * repeats
        if length < window:
            continue
        tail = suffix[:, length - window :].reshape(batch, repeats, ngram)
        hit |= (tail == tail[:, :1, :]).all(dim=(1, 2))
    return hit


def test_detect_repetition_loop_vectorized_matches_reference_and_per_row_min_tokens() -> None:
    torch.manual_seed(123)
    for length in (1, 5, 12, 24, 48):
        # Mix random rows and planted periodic loops of various periods.
        rows = [torch.randint(0, 5, (length,)) for _ in range(4)]
        for period in (1, 2, 3, 5):
            if period * 3 <= length:
                unit = torch.arange(10, 10 + period)
                rep = unit.repeat((length // period) + 1)[:length]
                rows.append(rep)
        batch_t = torch.stack(rows, dim=0)
        for max_ngram, min_repeats, min_tokens in (
            (1, 3, 6),
            (4, 3, 12),
            (6, 3, 13),
            (8, 2, 16),
        ):
            expected = _reference_detect_repetition_loop(
                batch_t, max_ngram, min_repeats, min_tokens
            )
            actual = decode_guard.detect_repetition_loop(
                batch_t,
                max_ngram=max_ngram,
                min_repeats=min_repeats,
                min_tokens=min_tokens,
            )
            assert torch.equal(actual, expected)

    # Per-row min_tokens: row 0 has short threshold (8), row 1 has long threshold (20).
    # Both repeat a 2-gram 5 times (10 tokens): only row 0 should trigger!
    tail_10 = torch.tensor([7, 8] * 5, dtype=torch.long)
    prefix = torch.arange(100, 110, dtype=torch.long)
    seq = torch.cat([prefix, tail_10])
    mixed = torch.stack([seq, seq], dim=0)
    per_row_hit = decode_guard.detect_repetition_loop(
        mixed, max_ngram=[2, 6], min_repeats=3, min_tokens=[8, 20]
    )
    assert per_row_hit.tolist() == [True, False]

    # build_stopping_criteria with per-row token budgets [80, 316].
    # At default loop_budget_fraction (0.50), thresholds are [40, 158]:
    # 45 repeated tokens must stop row 0 without stopping row 1.
    criteria = decode_guard.build_stopping_criteria(
        prompt_length=2, token_budgets=[80, 316]
    )
    assert criteria is not None
    loop_tokens = torch.tensor([42] * 45, dtype=torch.long)
    ids = torch.stack(
        [
            torch.cat([torch.tensor([1, 2]), loop_tokens]),
            torch.cat([torch.tensor([1, 2]), loop_tokens]),
        ],
        dim=0,
    )
    # criteria[0] is TokenBudgetCriteria; criteria[1] is RepetitionLoopCriteria.
    assert len(criteria) == 2
    assert criteria[0](ids, None).tolist() == [False, False]
    stopped = criteria[1](ids, None)
    assert stopped.tolist() == [True, False]


def test_repair_postprocessed_segment_windows_preserves_end_and_prevents_overlap() -> None:
    segments = [
        {
            "start": 1.0,
            "end": 1.0,  # invalid zero-length
            "text": "あ",
            "words": [{"start": 1.0, "end": 1.0, "word": "あ"}],
        },
        {
            "start": 1.02,
            "end": 1.5,
            "text": "い",
            "words": [{"start": 1.02, "end": 1.5, "word": "い"}],
        },
    ]
    repaired = transcribe._repair_postprocessed_segment_windows(segments)
    assert repaired[0]["end"] > repaired[0]["start"]
    assert repaired[0]["end"] <= repaired[1]["start"] + 1e-9

    # Never shrinks end when end is already past start + min_repaired.
    normal = [{"start": 2.0, "end": 2.8, "text": "う", "words": []}]
    out_normal = transcribe._repair_postprocessed_segment_windows(normal)
    assert out_normal[0]["end"] == pytest.approx(2.8)


def test_group_words_to_segments_inserts_space_between_adjacent_ascii_words() -> None:
    words = [
        {"start": 0.0, "end": 0.2, "word": "Hello"},
        {"start": 0.2, "end": 0.4, "word": "world"},
        {"start": 0.4, "end": 0.5, "word": "！"},
        {"start": 0.5, "end": 0.7, "word": "OK"},
        {"start": 0.7, "end": 0.9, "word": "で"},
        {"start": 0.9, "end": 1.1, "word": "す"},
    ]
    segs = transcribe._group_words_to_segments(words)
    assert len(segs) == 1
    assert segs[0]["text"] == "Hello world！OKです"


def test_transcribe_native_rejects_parsed_length_mismatch(monkeypatch) -> None:
    backend = local_backend.LocalAsrBackend(device="cpu")
    backend.processor = SimpleNamespace(
        batch_decode=lambda *_a, **_kw: ["only-one"],
        parse_output=lambda _decoded: [{"language": "ja", "transcription": "only-one"}],
    )
    backend.model = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        generation_config=SimpleNamespace(eos_token_id=[0], pad_token_id=0),
        config=SimpleNamespace(eos_token_id=[0], pad_token_id=0),
        generate=lambda **_kw: torch.tensor([[1, 2, 3], [1, 2, 4]]),
    )
    from asr import qwen_native

    monkeypatch.setattr(
        qwen_native,
        "prepare_transcription_inputs",
        lambda *_a, **_kw: {"input_ids": torch.tensor([[1, 2], [1, 2]])},
    )
    monkeypatch.setattr(
        qwen_native,
        "move_processor_inputs",
        lambda inputs, **_kw: inputs,
    )
    monkeypatch.setattr(local_backend, "_get_wav_duration_or_zero", lambda _p: 1.0)
    monkeypatch.setattr(local_backend, "build_stopping_criteria", lambda *_a, **_kw: None)

    with pytest.raises(RuntimeError, match="mismatch"):
        backend._transcribe_native(["a.wav", "b.wav"], language_hint="Japanese")


def test_qwen_min_physical_vram_env_rejects_non_positive_and_non_integer(
    monkeypatch,
) -> None:
    repo = qwen_backend.active_qwen_asr_model_id()
    for bad_value in (f"{repo}=0", f"{repo}=-512", f"{repo}=abc"):
        monkeypatch.setenv("ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO", bad_value)
        with pytest.raises(ValueError, match="positive"):
            qwen_backend.qwen_asr_min_physical_vram_mb_by_repo()


def test_vram_sizing_closed_form_matches_linear_scan_and_exposes_encoder_batches() -> None:
    for vram_mb in (6144, 8192, 12288, 16384, 24576):
        for dtype in ("bfloat16", "float16", "float32"):
            for attn in ("sdpa", "eager"):
                plan = vram_sizing.select_asr_batch_plan(
                    total_vram_mb=vram_mb,
                    free_vram_mb=vram_mb - 512,
                    dtype=dtype,
                    attention=attn,
                    chunk_max_s=30.0,
                    budget_mb=vram_mb * 0.85,
                )
                # Verify closed-form matches exact linear scan over 1..MAX_AUTO_ASR_BATCH_SIZE.
                expected_batch = 1
                for candidate in range(1, vram_sizing.MAX_AUTO_ASR_BATCH_SIZE + 1):
                    _alloc, est_reserved, _, _ = vram_sizing.estimate_asr_vram_mb(
                        candidate,
                        dtype=dtype,
                        attention=attn,
                        chunk_max_s=30.0,
                    )
                    if est_reserved <= plan.usable_vram_mb:
                        expected_batch = candidate
                assert plan.batch_size == expected_batch
                assert 1 <= plan.feature_batch_size <= vram_sizing.MAX_AUTO_ENCODER_BATCH_SIZE
                assert 1 <= plan.align_batch_size <= vram_sizing.MAX_AUTO_ENCODER_BATCH_SIZE


def test_audio_loading_lazy_scipy_for_16k_and_rejects_invalid_ndim(
    monkeypatch, tmp_path: Path
) -> None:
    wav_16k = _write_pcm_wav(tmp_path / "16k.wav", seconds=0.1, rate=16000)
    # Block scipy import completely: 16 kHz loading must NOT touch scipy.signal.
    saved_scipy = sys.modules.get("scipy")
    saved_signal = sys.modules.get("scipy.signal")
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.signal", None)
    try:
        samples, rate = audio_loading.load_audio_16k_mono(str(wav_16k))
        assert rate == 16000
        assert samples.dtype == np.float32
        assert samples.ndim == 1
        assert len(samples) == 1600
    finally:
        if saved_scipy is not None:
            sys.modules["scipy"] = saved_scipy
        if saved_signal is not None:
            sys.modules["scipy.signal"] = saved_signal

    # Invalid array dimensionality rejected explicitly.
    import soundfile as sf

    monkeypatch.setattr(
        sf, "read", lambda *_a, **_kw: (np.zeros((2, 2, 2), dtype=np.float32), 16000)
    )
    with pytest.raises(ValueError, match="Unsupported audio array shape"):
        audio_loading.load_audio_16k_mono(str(wav_16k))


def test_base_asr_backend_protocol_includes_close_and_cumulative_load_s() -> None:
    backend = local_backend.LocalAsrBackend(device="cpu")
    assert isinstance(backend, backend_base.BaseAsrBackend)
    assert hasattr(backend, "close")
    assert hasattr(backend, "cumulative_load_s")
