from __future__ import annotations

import json
import os


def main() -> None:
    os.environ["ASR_BACKEND"] = "qwen3-asr-1.7b"
    os.environ["ASR_WORKER_MODE"] = "subprocess"
    os.environ["ASR_WORKER_MOCK"] = "1"
    os.environ.setdefault("ASR_SUBPROCESS_READY_TIMEOUT_S", "10")

    from whisper import pipeline as asr
    from whisper.local_backend import SubprocessAsrBackend

    events: list[str] = []
    backend = asr._create_asr_backend("cpu")
    try:
        backend.load(on_stage=events.append)
        results = backend.transcribe_texts(
            ["temp/mock_factory.wav"],
            contexts=["factory_ctx"],
            on_stage=events.append,
        )
        print(
            json.dumps(
                {
                    "factory_type": type(backend).__name__,
                    "factory_module": type(backend).__module__,
                    "is_subprocess": isinstance(backend, SubprocessAsrBackend),
                    "result_count": len(results),
                    "text": results[0].get("text") if results else "",
                    "events": events,
                },
                ensure_ascii=False,
            )
        )
    finally:
        backend.unload_model(on_stage=events.append)


if __name__ == "__main__":
    main()


