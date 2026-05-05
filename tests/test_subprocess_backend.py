from __future__ import annotations

import json
import os


def main() -> None:
    os.environ["ASR_WORKER_MOCK"] = "1"
    os.environ.setdefault("ASR_SUBPROCESS_READY_TIMEOUT_S", "10")
    from whisper.local_backend import SubprocessAsrBackend

    events: list[str] = []
    backend = SubprocessAsrBackend("cpu")
    try:
        backend.load(on_stage=events.append)
        results = backend.transcribe_texts(
            ["temp/mock.wav"],
            contexts=["ctx"],
            on_stage=events.append,
        )
        print(
            json.dumps(
                {
                    "events": events,
                    "result_count": len(results),
                    "text": results[0].get("text") if results else "",
                    "request_batch_size": backend.request_batch_size,
                    "align_batch_size": backend.align_batch_size,
                },
                ensure_ascii=False,
            )
        )
    finally:
        backend.unload_model(on_stage=events.append)


if __name__ == "__main__":
    main()


