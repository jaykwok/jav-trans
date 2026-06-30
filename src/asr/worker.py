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

    def transcribe_texts(
        self,
        audio_paths,
        contexts=None,
        initial_prompts=None,
        on_stage=None,
    ):
        del initial_prompts, on_stage
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
    capturer_cache: list = []  # persists across messages; holds the AsrInternalsCapturer

    try:
        import torch as torch_module
        from asr.local_backend import LocalAsrBackend, _clear_cuda_cache

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

        if op == "capture_internals":
            _handle_capture_internals(parent_conn, msg, backend, capturer_cache)
            continue

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


def _handle_capture_internals(parent_conn, msg, backend, capturer_cache: list):
    """Capture ASR internals (encoder frames + token logits) in the worker, where
    the model is loaded. Returns one internals dict per requested chunk. The
    capturer is cached so the Qwen3-ASR wrapper is reused (no second model load).
    """
    import numpy as np

    job_id = str(msg.get("job_id") or "")
    chunks = msg.get("chunks") or []
    try:
        if not capturer_cache:
            from asr.asr_internals import AsrInternalsCapturer

            model = getattr(backend, "model", None)
            processor = getattr(backend, "processor", None)
            if model is None or processor is None:
                _safe_send(parent_conn, {
                    "op": "result", "job_id": job_id, "internals": [],
                    "error": "backend has no loaded model",
                })
                return
            capturer_cache.append(
                AsrInternalsCapturer(model=model, processor=processor)
            )
        capturer = capturer_cache[0]
        internals_list = []
        for chunk in chunks:
            path = str(chunk.get("path") or "")
            text = str(chunk.get("text") or "")
            start_s = float(chunk.get("start_s") or 0.0)
            end_s = float(chunk.get("end_s") or start_s)
            try:
                internals = capturer.extract(path, text, start_s=start_s, end_s=end_s)
                internals_list.append({
                    "ok": True,
                    "asr_frames": np.asarray(internals["asr_frames"], dtype=np.float32),
                    "token_logprobs": np.asarray(internals["token_logprobs"], dtype=np.float32),
                    "token_entropies": np.asarray(internals["token_entropies"], dtype=np.float32),
                    "token_top1_top2_margins": np.asarray(internals["token_top1_top2_margins"], dtype=np.float32),
                    "token_ids": np.asarray(internals["token_ids"], dtype=np.int64),
                    "decoded_tokens": internals.get("decoded_tokens") or [],
                    "has_timestamps": bool(internals.get("has_timestamps", False)),
                })
            except Exception as exc:  # noqa: BLE001 - per-chunk failure -> mark, keep going
                internals_list.append({"ok": False, "error": repr(exc)})
        _safe_send(parent_conn, {
            "op": "result", "job_id": job_id, "internals": internals_list,
        })
    except Exception as exc:  # noqa: BLE001
        _safe_send(parent_conn, {
            "op": "error", "job_id": job_id, "kind": "crash", "detail": repr(exc),
        })


if __name__ == "__main__":
    raise SystemExit("asr_worker.main is intended to run under multiprocessing spawn.")

