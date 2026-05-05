from dataclasses import dataclass

from whisper import local_backend
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


def test_forced_align_words_batch_uses_align_batch_size(monkeypatch):
    aligner = _RecordingAligner()
    backend = local_backend.LocalAsrBackend("cpu")
    backend.align_batch_size = 2
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


