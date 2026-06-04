from dataclasses import dataclass

from asr import local_backend


@dataclass
class _Item:
    text: str
    start_time: float
    end_time: float


class _RecordingAligner:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def align(self, *, audio, text, language):
        assert len(audio) == len(text) == len(language)
        self.batch_sizes.append(len(text))
        return [
            type("Result", (), {"items": [_Item(token, 0.0, 1.0)]})()
            for token in text
        ]


def test_forced_align_words_batch_uses_long_chunk_batch_size(monkeypatch):
    aligner = _RecordingAligner()
    backend = local_backend.LocalAsrBackend("cpu")
    backend.align_batch_size = 2
    monkeypatch.setattr(local_backend, "ALIGN_LONG_CHUNK_BATCH_SIZE", 2)
    monkeypatch.setattr(backend, "_ensure_forced_aligner", lambda on_stage=None: aligner)
    monkeypatch.setattr(local_backend, "_clear_cuda_cache", lambda _device: None)

    results = backend._forced_align_words_batch(
        [
            ("a.wav", "テスト一", "Japanese"),
            ("b.wav", "テスト二", "Japanese"),
            ("c.wav", "テスト三", "Japanese"),
            ("d.wav", "テスト四", "Japanese"),
        ]
    )

    assert aligner.batch_sizes == [2, 2]
    assert len(results) == 4
    assert [item[0]["word"] for item in results] == [
        "テスト一",
        "テスト二",
        "テスト三",
        "テスト四",
    ]


def test_forced_align_words_batch_can_force_single_item_batches(monkeypatch):
    aligner = _RecordingAligner()
    backend = local_backend.LocalAsrBackend("cpu")
    backend.align_batch_size = 4
    monkeypatch.setattr(local_backend, "ALIGN_LONG_CHUNK_BATCH_SIZE", 1)
    monkeypatch.setattr(backend, "_ensure_forced_aligner", lambda on_stage=None: aligner)
    monkeypatch.setattr(local_backend, "_clear_cuda_cache", lambda _device: None)

    results = backend._forced_align_words_batch(
        [
            ("a.wav", "テスト一", "Japanese"),
            ("b.wav", "テスト二", "Japanese"),
            ("c.wav", "テスト三", "Japanese"),
        ]
    )

    assert aligner.batch_sizes == [1, 1, 1]
    assert len(results) == 3


