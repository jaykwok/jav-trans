from __future__ import annotations

import json
import os
import sys
import uuid
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
    root = Path("temp") / "tests" / f"step4b_{uuid.uuid4().hex[:8]}"

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
    assert timings["text_transcribe_s"] >= 0.0

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
    assert "QUARANTINED: kind=timeout" in results[0]["log"][0]
    assert "respawn_count=3" in results[0]["log"][0]
    run_id = results[0]["log"][0].split("run_id=", 1)[1].split(",", 1)[0]
    sidecars = list((root.parents[1] / "asr_timeouts").glob(f"timeouts_*_{run_id}.json"))
    assert sidecars, run_id
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["worker_mode"] == "subprocess"
    assert payload["failure_kind"] == "timeout"
    assert payload["respawn_count"] == 3

    # subprocess: three different chunks time out repeatedly, then get quarantined.
    chunks = make_chunks("fuse", 3)
    checkpoint_source = asr._get_asr_checkpoint_source(chunks, label)
    checkpoint_path = asr._get_asr_checkpoint_path(checkpoint_source)
    results, _timings = asr._transcribe_asr_chunks_text_only(
        FakeSubprocessBackend(["timeout", "timeout", "timeout"], batch_size=3),
        chunks,
        label,
        on_stage=events.append,
    )
    assert len(results) == 3
    assert all(result["segments"] == [] for result in results)
    assert all("QUARANTINED: kind=timeout" in result["log"][0] for result in results)
    assert all("respawn_count=3" in result["log"][0] for result in results)
    assert not checkpoint_path.exists(), checkpoint_path

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


