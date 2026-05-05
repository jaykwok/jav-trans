from __future__ import annotations

import json
import os
import uuid
import wave
from pathlib import Path


def _write_wav(path: Path, seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(b"\x00\x00" * int(8000 * seconds))


def main() -> None:
    os.environ["ASR_WORKER_MODE"] = "subprocess"
    os.environ["ALIGNMENT_TIMESTAMP_MODE"] = "forced"
    os.environ.setdefault("ASR_SUBPROCESS_RESPAWN_MAX", "2")
    os.environ.setdefault("ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT", "3")

    from whisper import pipeline as asr
    from whisper.local_backend import SubprocessAsrBackend, WorkerTimeoutError

    label = "ASR 文本转写"
    root = Path("agents") / "temp" / f"step4b_{uuid.uuid4().hex[:8]}"

    def make_chunks(name: str, count: int) -> list[dict]:
        source = root / f"{name}_source.wav"
        _write_wav(source)
        chunks: list[dict] = []
        for index in range(count):
            path = root / f"{name}_{index}.wav"
            _write_wav(path)
            chunks.append(
                {
                    "index": index,
                    "start": index * 0.1,
                    "end": (index + 1) * 0.1,
                    "path": str(path),
                    "source_audio_path": str(source),
                }
            )
        return chunks

    class FakeSubprocessBackend(SubprocessAsrBackend):
        def __init__(self, outcomes: list[str], *, batch_size: int = 1):
            self.outcomes = list(outcomes)
            self.request_batch_size = batch_size
            self.align_batch_size = 1
            self.device = "cpu"

        def transcribe_texts(self, audio_paths, contexts=None, on_stage=None):
            outcome = self.outcomes.pop(0)
            if outcome == "timeout":
                raise WorkerTimeoutError("injected timeout")
            return [
                {
                    "text": f"ok-{Path(path).stem}",
                    "raw_text": f"ok-{Path(path).stem}",
                    "duration": 0.1,
                    "language": "Japanese",
                    "normalized_path": str(Path(path).resolve()),
                    "log": ["fake success"],
                }
                for path in audio_paths
            ]

    class FakeInprocBackend:
        request_batch_size = 1

        def transcribe_texts(self, audio_paths, contexts=None, on_stage=None):
            return [
                {
                    "text": "inproc-ok",
                    "raw_text": "inproc-ok",
                    "duration": 0.1,
                    "language": "Japanese",
                    "normalized_path": str(Path(audio_paths[0]).resolve()),
                    "log": ["inproc fake success"],
                }
            ]

    # subprocess: one timeout, then success; timeout streak resets after success.
    events: list[str] = []
    chunks = make_chunks("single_retry", 1)
    results, timings = asr._transcribe_asr_chunks_text_only(
        FakeSubprocessBackend(["timeout", "success"]),
        chunks,
        label,
        on_stage=events.append,
    )
    assert len(results) == 1 and results[0]["text"].startswith("ok-")
    assert timings["asr_worker_timeout_events"] == 1.0
    assert timings["asr_worker_quarantine_chunks"] == 0.0
    assert timings["asr_worker_fuse_triggered"] == 0.0

    # subprocess: same chunk times out three times, then gets quarantined.
    chunks = make_chunks("same_chunk_timeout", 1)
    results, timings = asr._transcribe_asr_chunks_text_only(
        FakeSubprocessBackend(["timeout", "timeout", "timeout"]),
        chunks,
        label,
        on_stage=events.append,
    )
    assert len(results) == 1
    assert results[0]["segments"] == []
    assert "QUARANTINED: kind=timeout, respawn=3" in results[0]["log"][0]
    run_id = results[0]["log"][1].split("=", 1)[1]
    sidecars = list((Path("agents") / "temp" / "asr_timeouts").glob(f"failures_*_{run_id}.json"))
    assert sidecars, run_id
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["worker_mode"] == "subprocess"
    assert payload["failed_chunks"][0]["failure_kind"] == "timeout"
    assert payload["failed_chunks"][0]["respawn_count"] == 3

    # subprocess: three different chunks time out in one batch, triggering system fuse.
    chunks = make_chunks("fuse", 3)
    checkpoint_source = asr._get_asr_checkpoint_source(chunks, label)
    checkpoint_path = asr._get_asr_checkpoint_path(checkpoint_source)
    try:
        asr._transcribe_asr_chunks_text_only(
            FakeSubprocessBackend(["timeout"], batch_size=3),
            chunks,
            label,
            on_stage=events.append,
        )
        raise AssertionError("ASRWorkerSystemError was not raised")
    except asr.ASRWorkerSystemError:
        assert checkpoint_path.exists(), checkpoint_path

    # inproc path still dispatches to the original sequential implementation.
    chunks = make_chunks("inproc", 1)
    results, _timings = asr._transcribe_asr_chunks_text_only(
        FakeInprocBackend(),
        chunks,
        label,
        on_stage=events.append,
    )
    assert len(results) == 1 and results[0]["text"] == "inproc-ok"

    print(json.dumps({"ok": True, "event_count": len(events)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


