import concurrent.futures
import gc
import inspect
import multiprocessing as mp
import os
import re
import time
import uuid
import wave
from pathlib import Path
from typing import Callable

from utils.model_paths import resolve_model_spec
from asr.backends.qwen import (
    active_qwen_asr_model_id,
    active_qwen_asr_model_path,
    current_qwen_asr_backend,
    qwen_asr_default_batch_size,
)
from asr.subtitle_timing import build_boundary_word_timestamps
from asr.text_normalize import normalize_display_text, strip_text_punctuation

ASR_MODEL_ID = active_qwen_asr_model_id()
ASR_MODEL_PATH = active_qwen_asr_model_path()
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "Japanese").strip() or "Japanese"
ASR_CONTEXT = os.getenv("ASR_CONTEXT", "").strip()


def _resolve_asr_batch_size() -> int:
    raw = os.getenv("ASR_BATCH_SIZE", "auto").strip().lower()
    if raw in {"", "auto"}:
        return max(1, qwen_asr_default_batch_size(current_qwen_asr_backend()))
    return max(1, int(raw))


ASR_BATCH_SIZE = _resolve_asr_batch_size()
ASR_MAX_NEW_TOKENS = max(64, int(os.getenv("ASR_MAX_NEW_TOKENS", "128")))
TRANSCRIPTION_MAX_NEW_TOKENS = max(
    32,
    int(os.getenv("TRANSCRIPTION_MAX_NEW_TOKENS", str(ASR_MAX_NEW_TOKENS))),
)
TRANSCRIPTION_TIMEOUT_S = float(os.getenv("TRANSCRIPTION_TIMEOUT_S", "180"))
ASR_DTYPE = os.getenv("ASR_DTYPE", "auto").strip().lower()
ASR_ATTN = os.getenv("ASR_ATTENTION", "auto").strip().lower()
ASR_REPETITION_PENALTY = float(os.getenv("ASR_REPETITION_PENALTY", "1.05"))
ASR_FORCE_LANGUAGE = os.getenv("ASR_FORCE_LANGUAGE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_ASR_SUBPROCESS_KILL_GRACE_S = float(os.getenv("ASR_SUBPROCESS_KILL_GRACE_S", "5"))
_ASR_SUBPROCESS_READY_TIMEOUT_S = float(
    os.getenv("ASR_SUBPROCESS_READY_TIMEOUT_S", "600")
)

def _get_wav_duration(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / rate if rate else 0.0


def _get_wav_duration_or_zero(audio_path: str) -> float:
    try:
        return _get_wav_duration(audio_path)
    except Exception:
        return 0.0


def _notify(on_stage: Callable[[str], None] | None, message: str) -> None:
    if on_stage:
        on_stage(message)


def _payload_has_timeout_log(payload: dict) -> bool:
    log = payload.get("log", [])
    if isinstance(log, str):
        return "TIMEOUT:" in log
    return any("TIMEOUT:" in str(entry) for entry in log)


def _clear_cuda_cache(device: str) -> None:
    if not device.startswith("cuda"):
        return
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _detect_dtype(device: str):
    import torch

    if ASR_DTYPE == "float32":
        return torch.float32
    if ASR_DTYPE == "float16":
        return torch.float16
    if ASR_DTYPE in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _detect_attention(device: str) -> str:
    if ASR_ATTN != "auto":
        return ASR_ATTN
    if not device.startswith("cuda"):
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _clean_master_text(text: str) -> str:
    return normalize_display_text(text)


def _strip_punctuation(text: str) -> str:
    return strip_text_punctuation(text)


def _first_token_id(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            token_id = _first_token_id(item)
            if token_id is not None:
                return token_id
    return None


def _iter_generation_configs(model) -> list:
    configs = []

    def add_config(config) -> None:
        if config is not None and not any(config is existing for existing in configs):
            configs.append(config)

    for candidate in (
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "thinker", None),
    ):
        add_config(getattr(candidate, "generation_config", None))
        add_config(getattr(candidate, "config", None))
    return configs


def _normalize_deterministic_generation_config(model) -> None:
    generation_configs = _iter_generation_configs(model)
    fallback_eos_token_id = None
    for generation_config in generation_configs:
        fallback_eos_token_id = _first_token_id(
            getattr(generation_config, "eos_token_id", None)
        )
        if fallback_eos_token_id is not None:
            break

    for generation_config in generation_configs:
        if not bool(getattr(generation_config, "do_sample", False)) and getattr(
            generation_config, "temperature", None
        ) is not None:
            generation_config.temperature = None

        if getattr(generation_config, "pad_token_id", None) is None:
            eos_token_id = _first_token_id(
                getattr(generation_config, "eos_token_id", None)
            ) or fallback_eos_token_id
            if eos_token_id is not None:
                generation_config.pad_token_id = eos_token_id


def _apply_generation_safety(model) -> None:
    _normalize_deterministic_generation_config(model)
    if ASR_REPETITION_PENALTY <= 1.0:
        return
    try:
        model.model.thinker.generation_config.repetition_penalty = ASR_REPETITION_PENALTY
    except Exception:
        pass


def _callable_accepts_kwarg(func, name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _asr_max_new_tokens() -> int:
    try:
        return max(64, int(os.getenv("ASR_MAX_NEW_TOKENS", str(ASR_MAX_NEW_TOKENS))))
    except (TypeError, ValueError):
        return ASR_MAX_NEW_TOKENS


def _transcription_max_new_tokens() -> int:
    try:
        fallback = str(_asr_max_new_tokens())
        return max(32, int(os.getenv("TRANSCRIPTION_MAX_NEW_TOKENS", fallback)))
    except (TypeError, ValueError):
        return TRANSCRIPTION_MAX_NEW_TOKENS


def _transcription_timeout_s() -> float:
    try:
        return float(os.getenv("TRANSCRIPTION_TIMEOUT_S", str(TRANSCRIPTION_TIMEOUT_S)))
    except (TypeError, ValueError):
        return TRANSCRIPTION_TIMEOUT_S


def _asr_language() -> str:
    return os.getenv("ASR_LANGUAGE", ASR_LANGUAGE).strip() or "Japanese"


def _asr_force_language() -> bool:
    return os.getenv("ASR_FORCE_LANGUAGE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _asr_context() -> str:
    return os.getenv("ASR_CONTEXT", ASR_CONTEXT).strip()


def _qwen_generation_metadata(
    *,
    error_kind: str | None = None,
    error_detail: str = "",
    worker_mode: str = "inproc",
) -> dict:
    return {
        "backend": current_qwen_asr_backend(),
        "model_id": active_qwen_asr_model_id(),
        "configured_max_new_tokens": _transcription_max_new_tokens(),
        "model_max_target_positions": None,
        "policy": "qwen_transcribe_limit",
        "worker_mode": worker_mode,
        "error_kind": error_kind,
        "error_detail": error_detail,
    }


def normalize_word_dicts(words: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for word in words:
        token = str(word.get("word", "")).strip()
        if not token:
            continue
        start = float(word.get("start", 0.0))
        end = float(word.get("end", 0.0))
        if end < start:
            end = start
        normalized.append({"start": start, "end": end, "word": token})
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized


class WorkerTimeoutError(RuntimeError):
    def __init__(self, detail: str = "ASR worker timed out"):
        super().__init__(detail)
        self.kind = "timeout"
        self.detail = detail


class WorkerError(RuntimeError):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail

class LocalAsrBackend:
    is_subprocess = False
    accepts_contexts = True

    def __init__(self, device: str):
        self.device = device if device.startswith("cuda") else "cpu"
        self.dtype = _detect_dtype(self.device)
        self.attention = _detect_attention(self.device)
        self.model = None
        self.request_batch_size = ASR_BATCH_SIZE

    def load(self, on_stage: Callable[[str], None] | None = None) -> None:
        from qwen_asr import Qwen3ASRModel

        if self.model is not None:
            return

        _notify(on_stage, "加载本地 ASR 模型...")
        model_spec = resolve_model_spec(
            active_qwen_asr_model_path() or None,
            active_qwen_asr_model_id(),
            download=True,
        )
        model_kwargs = {
            "dtype": self.dtype,
            "device_map": self.device,
            "max_inference_batch_size": ASR_BATCH_SIZE,
            "max_new_tokens": _asr_max_new_tokens(),
        }

        if self.attention and self.attention != "sdpa":
            model_kwargs["attn_implementation"] = self.attention

        self.model = Qwen3ASRModel.from_pretrained(model_spec, **model_kwargs)
        _apply_generation_safety(self.model)

    def unload_model(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self.model is None:
            return
        _notify(on_stage, "卸载 ASR 文本模型...")
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        _clear_cuda_cache(self.device)

    def close(self) -> None:
        self.unload_model()

    def _build_text_result(
        self,
        normalized_path: str,
        asr_result,
        language_hint: str | None,
    ) -> tuple[dict, list[str]]:
        duration = _get_wav_duration(normalized_path)
        detected_language = (asr_result.language or language_hint or "Japanese").strip()
        raw_master_text = (asr_result.text or "").strip()
        master_text = _clean_master_text(raw_master_text)

        log = [
            f"ASR 语言: {detected_language}",
            f"ASR 原始文本长度: {len(raw_master_text)}",
        ]
        if master_text != raw_master_text:
            log.append(f"ASR 清洗后文本长度: {len(master_text)}")
        log.append("ASR 输出模式: text_only")

        payload = {
            "text": master_text,
            "raw_text": raw_master_text,
            "duration": duration,
            "language": detected_language,
            "normalized_path": normalized_path,
            "asr_generation": _qwen_generation_metadata(),
        }
        return payload, log

    def transcribe_texts(
        self,
        audio_paths: list[str],
        contexts: list[str] | None = None,
        initial_prompts: list[str | None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict]:
        if self.model is None:
            self.load(on_stage=on_stage)
        if not audio_paths:
            return []
        if initial_prompts is not None and len(initial_prompts) != len(audio_paths):
            raise ValueError(
                f"initial_prompt count mismatch: audio_paths={len(audio_paths)}, initial_prompts={len(initial_prompts)}"
            )

        normalized_paths = [str(Path(audio_path).resolve()) for audio_path in audio_paths]
        language_hint = _asr_language() if _asr_force_language() else None
        request_contexts = contexts if contexts is not None else [_asr_context()] * len(normalized_paths)
        if len(request_contexts) != len(normalized_paths):
            raise ValueError(
                f"context count mismatch: audio_paths={len(normalized_paths)}, contexts={len(request_contexts)}"
            )

        _notify(on_stage, "ASR 文本转录中...")
        transcribe_kwargs = {
            "context": request_contexts,
            "language": language_hint,
            # Runtime timing comes from Boundary/speech-core chunk windows.
            "return_time_stamps": False,
        }
        if _callable_accepts_kwarg(self.model.transcribe, "max_new_tokens"):
            transcribe_kwargs["max_new_tokens"] = _transcription_max_new_tokens()

        asr_results = None
        executor = None
        timed_out = False
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                self.model.transcribe,
                normalized_paths,
                **transcribe_kwargs,
            )
            try:
                timeout_s = _transcription_timeout_s()
                asr_results = future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                timed_out = True
                future.cancel()
                _notify(
                    on_stage,
                    f"[WARN] ASR 超时 ({_transcription_timeout_s()}s)，跳过当前批次",
                )
                return [
                    {
                        "text": "",
                        "raw_text": "",
                        "duration": _get_wav_duration(path),
                        "language": language_hint or "Japanese",
                        "normalized_path": path,
                        "asr_generation": _qwen_generation_metadata(
                            error_kind="timeout",
                            error_detail=f"skipped after {_transcription_timeout_s()}s",
                        ),
                        "log": [
                            f"TIMEOUT: skipped after {_transcription_timeout_s()}s"
                        ],
                    }
                    for path in normalized_paths
                ]

            payloads: list[dict] = []
            for normalized_path, asr_result in zip(normalized_paths, asr_results):
                payload, payload_log = self._build_text_result(
                    normalized_path,
                    asr_result,
                    language_hint,
                )
                payload_log.append(f"ASR 文本生成上限: {_transcription_max_new_tokens()}")
                payload["log"] = payload_log
                payloads.append(payload)
        finally:
            if executor is not None:
                executor.shutdown(wait=not timed_out, cancel_futures=True)
            if asr_results is not None:
                try:
                    del asr_results
                except Exception:
                    pass
            _clear_cuda_cache(self.device)
        return payloads

    def capture_asr_internals(self, chunks: list[dict], **_kwargs) -> list[dict]:
        """Capture ASR internals (encoder frames + token logits) in-process.

        Reuses this backend's loaded Qwen3-ASR (no second model load). Each
        chunk: {path, text, start_s, end_s}. Returns one dict per chunk.
        """
        if not chunks:
            return []
        if self.model is None:
            self.load()
        try:
            from asr.asr_internals import AsrInternalsCapturer

            capturer = AsrInternalsCapturer(wrapper=self.model)
        except Exception:  # noqa: BLE001
            return [{"ok": False, "error": "capturer build failed"}] * len(chunks)
        out: list[dict] = []
        for chunk in chunks:
            path = str(chunk.get("path") or "")
            text = str(chunk.get("text") or "")
            start_s = float(chunk.get("start_s") or 0.0)
            end_s = float(chunk.get("end_s") or start_s)
            try:
                internals = capturer.extract(path, text, start_s=start_s, end_s=end_s)
                out.append({
                    "ok": True,
                    "asr_frames": internals["asr_frames"],
                    "token_logprobs": internals["token_logprobs"],
                    "token_entropies": internals["token_entropies"],
                    "token_top1_top2_margins": internals["token_top1_top2_margins"],
                    "token_ids": internals["token_ids"],
                    "decoded_tokens": internals.get("decoded_tokens") or [],
                    "has_timestamps": bool(internals.get("has_timestamps", False)),
                })
            except Exception as exc:  # noqa: BLE001
                out.append({"ok": False, "error": repr(exc)})
        return out

    def _use_boundary_timing_result(
        self,
        *,
        master_text: str,
        raw_master_text: str,
        duration: float,
        detected_language: str,
        normalized_path: str,
        timing_start: float,
        timing_end: float,
        timing_window_source: str,
        log: list[str],
    ) -> tuple[dict, list[str]]:
        log.append("Subtitle timing: boundary_chunk_timeline")
        word_dicts, alignment_mode, timing_meta = build_boundary_word_timestamps(
            master_text or raw_master_text,
            timing_start,
            timing_end,
            audio_path=normalized_path,
        )
        return self._build_finalize_output(
            word_dicts=normalize_word_dicts(word_dicts),
            master_text=master_text,
            raw_master_text=raw_master_text,
            alignment_mode=alignment_mode,
            duration=duration,
            detected_language=detected_language,
            log=log,
            timing_meta=timing_meta,
            timing_window_source=timing_window_source,
        )

    def _alignment_fallback_window_for_text_result(
        self,
        text_result: dict,
        duration: float,
    ) -> tuple[float, float, str]:
        full_start = 0.0
        full_end = max(0.0, float(duration))
        try:
            start = float(text_result.get("alignment_fallback_start_s"))
            end = float(text_result.get("alignment_fallback_end_s"))
        except (TypeError, ValueError):
            return full_start, full_end, "chunk"

        start = max(full_start, min(full_end, start))
        end = max(start, min(full_end, end))
        if end - start < 0.05:
            return full_start, full_end, "chunk"
        source = str(text_result.get("alignment_fallback_source") or "chunk").strip()
        return start, end, source or "chunk"

    def _build_finalize_output(
        self,
        *,
        word_dicts: list[dict],
        master_text: str,
        raw_master_text: str,
        alignment_mode: str,
        duration: float,
        detected_language: str,
        log: list[str],
        align_error: str = "",
        timing_meta: dict | None = None,
        timing_window_source: str = "",
    ) -> tuple[dict, list[str]]:
        log.append(f"Subtitle timing word count: {len(word_dicts)}")
        if align_error:
            log.append(f"Subtitle timing error: {align_error}")
        if timing_window_source == "speech_core":
            log.append("Subtitle timing window: speech_core")
        if timing_meta is not None and timing_meta.get("timing_source"):
            log.append(f"Subtitle timing source: {timing_meta['timing_source']}")
        log.append(f"Subtitle timing mode: {alignment_mode}")
        return {
            "words": word_dicts,
            "text": master_text,
            "raw_text": raw_master_text,
            "alignment_mode": alignment_mode,
            "duration": duration,
            "language": detected_language,
        }, log

    def finalize_text_results(
        self,
        text_results: list[dict],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[tuple[dict, list[str]]]:
        if not text_results:
            return []

        finalized: list[tuple[dict, list[str]]] = []
        for text_result in text_results:
            log: list[str] = list(text_result.get("log", []))
            normalized_path = str(text_result["normalized_path"])
            duration = float(text_result["duration"])
            detected_language = str(text_result["language"]).strip() or "Japanese"
            raw_master_text = str(text_result.get("raw_text", "")).strip()
            master_text = str(text_result.get("text", "")).strip()
            fallback_start, fallback_end, fallback_window_source = (
                self._alignment_fallback_window_for_text_result(text_result, duration)
            )

            if not master_text:
                finalized.append((
                    {
                        "words": [],
                        "text": "",
                        "raw_text": raw_master_text,
                        "alignment_mode": "empty",
                        "duration": duration,
                        "language": detected_language,
                    },
                    log,
                ))
                continue

            finalized.append(
                self._use_boundary_timing_result(
                    master_text=master_text,
                    raw_master_text=raw_master_text,
                    duration=duration,
                    detected_language=detected_language,
                    normalized_path=normalized_path,
                    timing_start=fallback_start,
                    timing_end=fallback_end,
                    timing_window_source=fallback_window_source,
                    log=log,
                )
            )
        return finalized

    def finalize_text_result(
        self,
        text_result: dict,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[dict, list[str]]:
        return self.finalize_text_results([text_result], on_stage=on_stage)[0]

    def transcribe_to_words(
        self,
        audio_path: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[dict, list[str]]:
        text_result = self.transcribe_texts([audio_path], on_stage=on_stage)[0]
        self.unload_model(on_stage=on_stage)
        return self.finalize_text_result(text_result, on_stage=on_stage)


class SubprocessAsrBackend:
    """Run ASR text inference in a killable child process."""

    is_subprocess = True
    accepts_contexts = True

    def __init__(self, device: str):
        self.device = device if device.startswith("cuda") else "cpu"
        self.request_batch_size = ASR_BATCH_SIZE
        self.kill_grace_s = _ASR_SUBPROCESS_KILL_GRACE_S
        self.ready_timeout_s = _ASR_SUBPROCESS_READY_TIMEOUT_S
        self.model = None
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._conn = None

    def load(self, on_stage: Callable[[str], None] | None = None) -> None:
        self._ensure_worker(on_stage=on_stage)

    def _ensure_worker(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self._process is not None and self._process.is_alive() and self._conn is not None:
            return
        self._start_worker(on_stage=on_stage)

    def _start_worker(self, on_stage: Callable[[str], None] | None = None) -> None:
        from asr.worker import main as worker_main

        self._close_conn()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=worker_main,
            args=(child_conn, {"device": self.device}),
            daemon=False,
        )

        _notify(on_stage, "启动 ASR 子进程...")
        process.start()
        child_conn.close()
        self._process = process
        self._conn = parent_conn

        deadline = time.monotonic() + self.ready_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_child()
                raise WorkerError(
                    "crash",
                    f"worker ready timeout after {self.ready_timeout_s}s",
                )
            if parent_conn.poll(min(0.5, remaining)):
                break
            if process.exitcode is not None:
                exitcode = process.exitcode
                self._kill_child()
                raise WorkerError("crash", f"worker exited before ready: {exitcode}")

        try:
            message = parent_conn.recv()
        except EOFError as exc:
            exitcode = process.exitcode
            self._kill_child()
            raise WorkerError("crash", f"worker exited before ready: {exitcode}") from exc

        if not isinstance(message, dict):
            self._kill_child()
            raise WorkerError("protocol_error", "ready message is not a dict")

        if message.get("op") == "error":
            kind = str(message.get("kind") or "crash")
            detail = str(message.get("detail") or "worker failed during startup")
            self._kill_child()
            raise WorkerError(kind, detail)

        if message.get("op") != "ready":
            self._kill_child()
            raise WorkerError("protocol_error", f"unexpected ready message: {message!r}")

        pid = message.get("pid")
        if not isinstance(pid, int):
            self._kill_child()
            raise WorkerError("protocol_error", f"invalid ready pid: {pid!r}")

        _notify(on_stage, f"ASR 子进程就绪 pid={pid}")

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def _kill_child(self) -> None:
        process = self._process
        self._process = None

        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(self.kill_grace_s)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                if process.is_alive():
                    process.join(1)
            except Exception:
                pass

        self._close_conn()
        _clear_cuda_cache(self.device)

    def _restart_worker(self, on_stage: Callable[[str], None] | None = None) -> None:
        self._kill_child()
        self._start_worker(on_stage=on_stage)

    def _raise_after_worker_restart(
        self,
        exc: WorkerTimeoutError | WorkerError,
        cause: BaseException | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> None:
        try:
            if cause is not None:
                raise exc from cause
            raise exc
        finally:
            try:
                self._restart_worker(on_stage=on_stage)
            except Exception as restart_exc:
                raise WorkerError(
                    "crash",
                    f"worker respawn failed: {restart_exc!r}",
                ) from exc

    def unload_model(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self._process is None:
            return

        _notify(on_stage, "关闭 ASR 子进程...")
        conn = self._conn
        process = self._process
        try:
            if conn is not None and process is not None and process.is_alive():
                conn.send({"op": "shutdown"})
                process.join(5)
        except Exception:
            pass
        finally:
            if process is not None and process.is_alive():
                self._kill_child()
            else:
                self._process = None
                self._close_conn()
                _clear_cuda_cache(self.device)

    def close(self) -> None:
        self.unload_model()

    def transcribe_texts(
        self,
        audio_paths: list[str],
        contexts: list[str] | None = None,
        initial_prompts: list[str | None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict]:
        if not audio_paths:
            return []
        if initial_prompts is not None and len(initial_prompts) != len(audio_paths):
            raise ValueError(
                f"initial_prompt count mismatch: audio_paths={len(audio_paths)}, initial_prompts={len(initial_prompts)}"
            )

        self._ensure_worker(on_stage=on_stage)
        assert self._conn is not None

        if contexts is None:
            request_contexts = [_asr_context()] * len(audio_paths)
        elif len(contexts) != len(audio_paths):
            raise ValueError(
                f"context count mismatch: audio_paths={len(audio_paths)}, contexts={len(contexts)}"
            )
        else:
            request_contexts = contexts

        job_id = uuid.uuid4().hex[:8]
        chunks = [
            {
                "path": str(Path(audio_path).resolve()),
                "context": context,
                "index": idx,
            }
            for idx, (audio_path, context) in enumerate(
                zip(audio_paths, request_contexts)
            )
        ]

        try:
            self._conn.send({"op": "transcribe", "job_id": job_id, "chunks": chunks})
        except (BrokenPipeError, EOFError, OSError) as exc:
            exitcode = self._process.exitcode if self._process is not None else None
            failure = WorkerError(
                "crash",
                f"worker send failed exitcode={exitcode}: {exc!r}",
            )
            self._raise_after_worker_restart(failure, cause=exc, on_stage=on_stage)
        timeout_s = _transcription_timeout_s()
        deadline = time.monotonic() + timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f"worker timeout after {timeout_s}s"
                self._raise_after_worker_restart(
                    WorkerTimeoutError(detail),
                    on_stage=on_stage,
                )
            if not self._conn.poll(min(1.0, remaining)):
                exitcode = self._process.exitcode if self._process is not None else None
                if exitcode is not None:
                    self._raise_after_worker_restart(
                        WorkerError(
                            "crash",
                            f"worker exited before result exitcode={exitcode}",
                        ),
                        on_stage=on_stage,
                    )
                continue

            try:
                message = self._conn.recv()
            except EOFError as exc:
                exitcode = self._process.exitcode if self._process is not None else None
                failure = WorkerError(
                    "crash",
                    f"worker pipe closed exitcode={exitcode}",
                )
                self._raise_after_worker_restart(failure, cause=exc, on_stage=on_stage)

            if not isinstance(message, dict):
                self._raise_after_worker_restart(
                    WorkerError("protocol_error", "worker message is not a dict"),
                    on_stage=on_stage,
                )

            if str(message.get("job_id") or "") != job_id:
                continue

            op = message.get("op")
            if op == "result":
                results = message.get("results")
                if not isinstance(results, list) or len(results) != len(audio_paths):
                    self._raise_after_worker_restart(
                        WorkerError(
                            "protocol_error",
                            "worker result count does not match request",
                        ),
                        on_stage=on_stage,
                    )
                if any(
                    isinstance(result, dict) and _payload_has_timeout_log(result)
                    for result in results
                ):
                    self._raise_after_worker_restart(
                        WorkerTimeoutError("worker returned TIMEOUT payload"),
                        on_stage=on_stage,
                    )
                for result in results:
                    if isinstance(result, dict) and isinstance(
                        result.get("asr_generation"),
                        dict,
                    ):
                        result["asr_generation"]["worker_mode"] = "subprocess"
                return results

            if op == "error":
                kind = str(message.get("kind") or "crash")
                detail = str(message.get("detail") or "worker error")
                self._raise_after_worker_restart(
                    WorkerError(kind, detail),
                    on_stage=on_stage,
                )

            self._raise_after_worker_restart(
                WorkerError("protocol_error", f"unexpected worker op: {op}"),
                on_stage=on_stage,
            )

    def capture_asr_internals(
        self,
        chunks: list[dict],
        *,
        timeout_s: float | None = None,
    ) -> list[dict]:
        """Capture ASR internals (encoder frames + token logits) in the worker.

        Each chunk: {path, text, start_s, end_s}. Returns one dict per chunk
        with asr_frames/token_logprobs/... or {ok: False, error: ...}. The
        capture reuses the worker's loaded Qwen3-ASR (no second model load).
        """
        if not chunks:
            return []
        self._ensure_worker()
        assert self._conn is not None
        job_id = uuid.uuid4().hex[:8]
        payload = [
            {
                "path": str(c.get("path") or ""),
                "text": str(c.get("text") or ""),
                "start_s": float(c.get("start_s") or 0.0),
                "end_s": float(c.get("end_s") or c.get("start_s") or 0.0),
            }
            for c in chunks
        ]
        try:
            self._conn.send({"op": "capture_internals", "job_id": job_id, "chunks": payload})
        except (BrokenPipeError, EOFError, OSError):
            return [{"ok": False, "error": "worker send failed"}] * len(chunks)
        deadline = time.monotonic() + (timeout_s or 600.0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return [{"ok": False, "error": "capture timeout"}] * len(chunks)
            if not self._conn.poll(min(1.0, remaining)):
                continue
            try:
                message = self._conn.recv()
            except EOFError:
                return [{"ok": False, "error": "worker pipe closed"}] * len(chunks)
            if not isinstance(message, dict) or str(message.get("job_id") or "") != job_id:
                continue
            if message.get("op") == "result":
                internals = message.get("internals")
                if isinstance(internals, list) and len(internals) == len(chunks):
                    return internals
                return [{"ok": False, "error": "count mismatch"}] * len(chunks)
            if message.get("op") == "error":
                return [{"ok": False, "error": str(message.get("detail") or "worker error")}] * len(chunks)

    def finalize_text_results(
        self,
        text_results: list[dict],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[tuple[dict, list[str]]]:
        return LocalAsrBackend(self.device).finalize_text_results(text_results, on_stage=on_stage)

    def finalize_text_result(
        self,
        text_result: dict,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[dict, list[str]]:
        return self.finalize_text_results([text_result], on_stage=on_stage)[0]

    def transcribe_to_words(
        self,
        audio_path: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[dict, list[str]]:
        text_result = self.transcribe_texts([audio_path], on_stage=on_stage)[0]
        self.unload_model(on_stage=on_stage)
        return self.finalize_text_result(text_result, on_stage=on_stage)


def transcribe_to_words(
    audio_path: str,
    device: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[dict, list[str]]:
    backend = LocalAsrBackend(device)
    try:
        log = [f"ASR backend: {current_qwen_asr_backend()}"]
        result, extra_log = backend.transcribe_to_words(audio_path, on_stage=on_stage)
        return result, log + extra_log
    finally:
        backend.close()
