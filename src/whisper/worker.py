"""Subprocess entry point for local ASR transcription."""

from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any


def _safe_send(conn: Connection, message: dict[str, Any]) -> bool:
    try:
        conn.send(message)
        return True
    except Exception:
        return False


def _is_oom_error(exc: BaseException, torch_module: Any | None) -> bool:
    if torch_module is not None:
        try:
            if isinstance(exc, torch_module.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass

    detail = repr(exc).lower()
    return (
        ("out of memory" in detail and ("cuda" in detail or "gpu" in detail))
        or "cumemalloc" in detail
    )


class _MockAsrBackend:
    def __init__(self, device: str):
        self.device = device

    def load(self) -> None:
        return

    def unload_model(self) -> None:
        return

    def transcribe_texts(self, audio_paths, contexts=None, on_stage=None):
        contexts = contexts or [""] * len(audio_paths)
        return [
            {
                "text": "mock",
                "raw_text": "mock",
                "duration": 0.0,
                "language": "Japanese",
                "normalized_path": path,
                "log": [
                    "ASR worker mock result",
                    f"context_len={len(context or '')}",
                ],
            }
            for path, context in zip(audio_paths, contexts)
        ]


def _extract_request(msg: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    job_id = str(msg.get("job_id") or "")
    chunks = msg.get("chunks")
    if not job_id:
        raise ValueError("missing job_id")
    if not isinstance(chunks, list):
        raise ValueError("chunks must be a list")

    paths: list[str] = []
    contexts: list[str] = []
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunk {position} must be an object")
        path = chunk.get("path")
        if not path:
            raise ValueError(f"chunk {position} missing path")
        paths.append(str(path))
        contexts.append(str(chunk.get("context") or ""))
    return job_id, paths, contexts


def main(parent_conn: Connection, backend_kwargs: dict[str, Any]) -> None:
    import os
    import sys

    sys.stdout = sys.stderr

    backend = None
    torch = None
    clear_cuda_cache = None

    try:
        import torch as torch_module
        from whisper.local_backend import LocalAsrBackend, _clear_cuda_cache

        torch = torch_module
        clear_cuda_cache = _clear_cuda_cache
        kwargs = dict(backend_kwargs or {})
        unexpected_keys = sorted(set(kwargs) - {"device"})
        if unexpected_keys:
            raise TypeError(f"unexpected backend_kwargs: {unexpected_keys}")
        if "device" not in kwargs:
            raise TypeError("missing backend_kwargs['device']")
        mock = os.getenv("ASR_WORKER_MOCK", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if mock:
            backend = _MockAsrBackend(str(kwargs.get("device", "cpu")))
        else:
            backend = LocalAsrBackend(**kwargs)
        backend.load()
        clear_cuda_cache(backend.device)
        if not _safe_send(parent_conn, {"op": "ready", "pid": os.getpid()}):
            return
    except Exception as exc:
        kind = "oom" if _is_oom_error(exc, torch) else "crash"
        _safe_send(
            parent_conn,
            {"op": "error", "job_id": "", "kind": kind, "detail": repr(exc)},
        )
        raise SystemExit(0)

    while True:
        try:
            msg = parent_conn.recv()
        except (EOFError, OSError):
            return

        if not isinstance(msg, dict):
            if not _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": "",
                    "kind": "protocol_error",
                    "detail": "message must be an object",
                },
            ):
                return
            continue

        op = msg.get("op")
        if op == "shutdown":
            try:
                if backend is not None:
                    backend.unload_model()
            except Exception:
                pass
            finally:
                if backend is not None and clear_cuda_cache is not None:
                    clear_cuda_cache(backend.device)
            raise SystemExit(0)

        if op != "transcribe":
            _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": str(msg.get("job_id") or ""),
                    "kind": "protocol_error",
                    "detail": f"unknown op: {op}",
                },
            )
            raise SystemExit(1)

        try:
            job_id, paths, contexts = _extract_request(msg)
        except Exception as exc:
            if not _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": str(msg.get("job_id") or ""),
                    "kind": "protocol_error",
                    "detail": repr(exc),
                },
            ):
                return
            continue

        try:
            results = backend.transcribe_texts(paths, contexts=contexts, on_stage=None)
            if not _safe_send(
                parent_conn,
                {"op": "result", "job_id": job_id, "results": results},
            ):
                return
        except Exception as exc:
            kind = "oom" if _is_oom_error(exc, torch) else "crash"
            _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": job_id,
                    "kind": kind,
                    "detail": repr(exc),
                },
            )
            raise SystemExit(0)


if __name__ == "__main__":
    raise SystemExit("asr_worker.main is intended to run under multiprocessing spawn.")


