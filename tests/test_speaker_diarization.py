import numpy as np
import pytest
import torch
import torchaudio
from pathlib import Path

from audio import speaker_diarization as sd
def _seg(ja: str, zh: str, start: float, end: float) -> dict:
    return {"text": ja, "zh": zh, "start": start, "end": end}


def _make_mock_classifier(embeddings_by_call: list[np.ndarray]):
    call_count = {"n": 0}

    class MockClassifier:
        def eval(self):
            return self

        def encode_batch(self, clip):
            i = call_count["n"]
            call_count["n"] += 1
            emb = embeddings_by_call[i % len(embeddings_by_call)]
            return torch.tensor(emb, dtype=torch.float32).reshape(1, 1, -1)

    return MockClassifier()


def _write_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    samples = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, samples), sr)
    return path


def test_two_speaker_clustering(monkeypatch, tmp_path):
    wav = _write_wav(tmp_path / "test.wav")
    segs = [
        _seg("テスト", "测试", 0.0, 1.0),
        _seg("テスト", "测试", 1.0, 2.0),
        _seg("テスト", "测试", 2.0, 3.0),
        _seg("テスト", "测试", 3.0, 4.0),
    ]
    emb_A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_B = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mock = _make_mock_classifier([emb_A, emb_A, emb_B, emb_B])

    monkeypatch.setenv("EXPERIMENTAL_SPEAKER_DIARIZATION", "1")
    monkeypatch.setattr(sd, "_import_classifier", lambda source, savedir, run_opts: mock)

    result = sd.diarize_segments(segs, wav, cluster_threshold=0.3)
    speakers = [r["speaker"] for r in result]
    assert speakers[0] == speakers[1]
    assert speakers[2] == speakers[3]
    assert speakers[0] != speakers[2]


def test_disabled_returns_original(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENTAL_SPEAKER_DIARIZATION", "0")
    segs = [_seg("テスト", "测试", 0.0, 1.0)]
    result = sd.diarize_segments(segs, tmp_path / "missing.wav")
    assert result is segs


def test_short_segment_backfilled(monkeypatch, tmp_path):
    wav = _write_wav(tmp_path / "test.wav")
    segs = [
        _seg("テスト", "测试", 0.0, 1.0),
        _seg("ア", "啊", 1.0, 1.2),        # 0.2s < min_duration=0.5 -> short
        _seg("テスト", "测试", 1.2, 2.2),
    ]
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mock = _make_mock_classifier([emb, emb])

    monkeypatch.setenv("EXPERIMENTAL_SPEAKER_DIARIZATION", "1")
    monkeypatch.setattr(sd, "_import_classifier", lambda source, savedir, run_opts: mock)

    result = sd.diarize_segments(segs, wav, min_duration_s=0.5)
    assert result[1]["speaker"] == result[0]["speaker"]


def test_kana_only_bgm_stays_none(monkeypatch, tmp_path):
    wav = _write_wav(tmp_path / "test.wav")
    segs = [
        _seg("あああ", "啊", 0.0, 1.0),    # kana-only -> BGM -> None
        _seg("テスト", "测试", 1.0, 2.0),
    ]
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mock = _make_mock_classifier([emb])

    monkeypatch.setenv("EXPERIMENTAL_SPEAKER_DIARIZATION", "1")
    monkeypatch.setattr(sd, "_import_classifier", lambda source, savedir, run_opts: mock)

    result = sd.diarize_segments(segs, wav)
    assert result[0]["speaker"] is None
    assert result[1]["speaker"] == "S0"


def test_model_error_graceful_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENTAL_SPEAKER_DIARIZATION", "1")
    monkeypatch.setattr(sd, "_import_classifier", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("load failed")))

    segs = [_seg("テスト", "测试", 0.0, 1.0)]
    result = sd.diarize_segments(segs, tmp_path / "x.wav")
    assert result == segs


def test_build_speakers_report():
    segs = [
        {"text": "テスト", "zh": "测试", "start": 0.0, "end": 1.0, "speaker": "S0"},
        {"text": "テスト", "zh": "测试", "start": 1.0, "end": 2.0, "speaker": "S1"},
        {"text": "あああ", "zh": "啊",   "start": 2.0, "end": 3.0, "speaker": None},
    ]
    report = sd.build_speakers_report(segs)
    assert report["n_speakers"] == 2
    assert "S0" in report["speakers"]
    assert "S1" in report["speakers"]
    assert report["unassigned"]["bgm"] == 1


def test_max_clusters_capped(monkeypatch, tmp_path):
    wav = _write_wav(tmp_path / "test.wav", duration_s=10.0)
    # 6 segments with 6 distinct embeddings (unit vectors on each axis)
    embs = [np.eye(6, dtype=np.float32)[i] for i in range(6)]
    segs = [_seg("テスト", "测试", float(i), float(i + 1)) for i in range(6)]
    mock = _make_mock_classifier(embs)

    monkeypatch.setenv("EXPERIMENTAL_SPEAKER_DIARIZATION", "1")
    monkeypatch.setattr(sd, "_import_classifier", lambda source, savedir, run_opts: mock)

    result = sd.diarize_segments(segs, wav, cluster_threshold=0.01, max_clusters=3)
    unique_speakers = {r["speaker"] for r in result if r.get("speaker") is not None}
    assert len(unique_speakers) <= 3

