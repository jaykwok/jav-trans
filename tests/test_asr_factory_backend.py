from __future__ import annotations

import json
import os

from helpers import ASR_17B_BACKEND


def main() -> None:
    os.environ["ASR_BACKEND"] = ASR_17B_BACKEND
    os.environ["ASR_WORKER_MODE"] = "subprocess"
    os.environ["ASR_WORKER_MOCK"] = "1"
    os.environ.setdefault("ASR_SUBPROCESS_READY_TIMEOUT_S", "10")

    from asr import pipeline as asr
    from asr.local_backend import SubprocessAsrBackend

    events: list[str] = []
    backend = asr._create_asr_backend("cpu")
    try:
        backend.load(on_stage=events.append)
        results = backend.transcribe_texts(
            ["tmp/tests/mock_factory.wav"],
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


