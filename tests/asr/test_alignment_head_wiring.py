"""The alignment head must actually run when configured.

The bug these tests pin: `_align_TRANSCRIPTION_results` used to unload the ASR
model unconditionally *before* `finalize_text_results`, while the aligned path
(`_align_characters`) starts with `if self.model is None: return None` — so a
configured head silently degraded every chunk to proportional timing and never
ran in the production batch flow.
"""
from __future__ import annotations

from asr import transcribe


class _LifecycleBackend:
    def __init__(
        self,
        *,
        model: object | None = None,
        load_raises: bool = False,
        finalize_mode: str = "boundary_proportional",
    ) -> None:
        self.model = model
        self.events: list[object] = []
        self._load_raises = load_raises
        self._mode = finalize_mode

    def load(self, on_stage=None) -> None:
        self.events.append("load")
        if self._load_raises:
            raise RuntimeError("no vram for alignment")
        self.model = object()

    def unload_model(self, on_stage=None) -> None:
        self.events.append("unload")
        self.model = None

    def finalize_text_results(self, text_results, on_stage=None):
        self.events.append(("finalize", self.model is not None))
        return [
            (
                {
                    "words": [],
                    "text": str(result.get("text") or ""),
                    "raw_text": str(result.get("raw_text") or ""),
                    "alignment_mode": self._mode,
                    "duration": float(result.get("duration") or 0.0),
                    "language": "Japanese",
                },
                list(result.get("log") or []),
            )
            for result in text_results
        ]


def _text_result(tmp_path) -> dict:
    return {
        "text": "テスト",
        "raw_text": "テスト",
        "duration": 1.0,
        "language": "Japanese",
        "normalized_path": str(tmp_path / "chunk_0000.wav"),
        "log": [],
    }


def _finalize_events(backend: _LifecycleBackend) -> list[object]:
    return [event for event in backend.events if not isinstance(event, str)]


def test_head_configured_keeps_model_loaded_through_finalize(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_ALIGNMENT_HEAD_PATH", str(tmp_path / "head.pt"))
    backend = _LifecycleBackend(model=object())

    results, _ = transcribe._align_TRANSCRIPTION_results(backend, [_text_result(tmp_path)])

    assert len(results) == 1
    assert _finalize_events(backend) == [("finalize", True)]
    finalize_at = backend.events.index(("finalize", True))
    assert "unload" in backend.events[finalize_at + 1 :], "model must be unloaded after finalize"
    assert backend.model is None


def test_no_head_unloads_before_finalize(monkeypatch, tmp_path):
    monkeypatch.delenv("ASR_ALIGNMENT_HEAD_PATH", raising=False)
    backend = _LifecycleBackend(model=object())

    transcribe._align_TRANSCRIPTION_results(backend, [_text_result(tmp_path)])

    # Proportional finalize is pure CPU: the early unload frees VRAM sooner
    # and stays the contract for the head-less path.
    assert backend.events.index("unload") < backend.events.index(("finalize", False))


def test_cached_text_rerun_loads_model_for_alignment(monkeypatch, tmp_path):
    # A fully cache-hit text stage arrives here with the model never loaded.
    monkeypatch.setenv("ASR_ALIGNMENT_HEAD_PATH", str(tmp_path / "head.pt"))
    backend = _LifecycleBackend(model=None)

    transcribe._align_TRANSCRIPTION_results(backend, [_text_result(tmp_path)])

    assert backend.events.index("load") < backend.events.index(("finalize", True))
    assert backend.model is None


def test_model_load_failure_degrades_instead_of_raising(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_ALIGNMENT_HEAD_PATH", str(tmp_path / "head.pt"))
    backend = _LifecycleBackend(model=None, load_raises=True)

    results, _ = transcribe._align_TRANSCRIPTION_results(backend, [_text_result(tmp_path)])

    assert len(results) == 1
    assert ("finalize", False) in backend.events


def test_load_is_deferred_so_a_fully_cached_rerun_never_loads(monkeypatch, tmp_path):
    """A rerun served entirely from cache must not pay for the weights.

    The pipeline used to load eagerly, before the text stage could report that
    every chunk was already cached — about five seconds spent loading a model
    that was then unloaded without transcribing or aligning anything.
    """
    monkeypatch.delenv("ASR_ALIGNMENT_HEAD_PATH", raising=False)
    backend = _LifecycleBackend(model=None)

    # Everything already cached: the text stage returns without transcribing and
    # the head-less alignment path is pure CPU.
    transcribe._align_TRANSCRIPTION_results(backend, [_text_result(tmp_path)])

    assert "load" not in backend.events
