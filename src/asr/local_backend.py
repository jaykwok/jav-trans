import concurrent.futures
import gc
import logging
import os
import re
import time
import wave
from dataclasses import dataclass
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

logger = logging.getLogger(__name__)

ASR_MODEL_ID = active_qwen_asr_model_id()
ASR_MODEL_PATH = active_qwen_asr_model_path()
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "Japanese").strip() or "Japanese"


def _resolve_asr_batch_size() -> int:
    raw = os.getenv("ASR_BATCH_SIZE", "auto").strip().lower()
    if raw in {"", "auto"}:
        return max(1, qwen_asr_default_batch_size(current_qwen_asr_backend()))
    return max(1, int(raw))


ASR_MAX_NEW_TOKENS = max(64, int(os.getenv("ASR_MAX_NEW_TOKENS", "128")))
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
# --- Windows Job Object: kill the GPU worker if the parent dies abnormally
# (kill -9 / segfault / OOM-killer / task-manager end). daemon=True only covers
# graceful interpreter exit; a Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
# makes the OS reap the child when the parent process vanishes. Best-effort: on
# failure we fall back to the caller's explicit kill path on close(). ---
if os.name == "nt":
    import ctypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def _create_kill_on_close_job_object():
        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError()
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            raise ctypes.WinError()
        return job

    def _assign_process_to_job_object(job, process) -> None:
        kernel32 = ctypes.windll.kernel32
        # multiprocessing spawn Process keeps the Win32 process handle on its
        # Popen object (process._popen._handle).
        handle = getattr(getattr(process, "_popen", None), "_handle", None)
        if not handle:
            raise RuntimeError("subprocess has no win32 handle to assign")
        if not kernel32.AssignProcessToJobObject(job, handle):
            raise ctypes.WinError()

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

    for candidate in (model,):
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
    model.generation_config.repetition_penalty = ASR_REPETITION_PENALTY


def _asr_max_new_tokens() -> int:
    try:
        return max(64, int(os.getenv("ASR_MAX_NEW_TOKENS", str(ASR_MAX_NEW_TOKENS))))
    except (TypeError, ValueError):
        return ASR_MAX_NEW_TOKENS


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


def _qwen_generation_metadata(
    *,
    error_kind: str | None = None,
    error_detail: str = "",
    worker_mode: str = "gpu_worker",
) -> dict:
    return {
        "backend": current_qwen_asr_backend(),
        "model_id": active_qwen_asr_model_id(),
        "configured_max_new_tokens": _asr_max_new_tokens(),
        "model_max_target_positions": None,
        "policy": "native_transformers_generate",
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
        if end <= start:
            # Drop zero-width / inverted words (floating-point drift in
            # proportional timing can produce them); downstream renderers
            # already ignore words without a positive span.
            continue
        normalized.append({"start": start, "end": end, "word": token})
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized


@dataclass(frozen=True)
class NativeAsrTranscription:
    language: str | None
    text: str


class LocalAsrBackend:
    is_subprocess = False

    def __init__(self, device: str):
        self.device = device if device.startswith("cuda") else "cpu"
        self.dtype = _detect_dtype(self.device)
        self.attention = _detect_attention(self.device)
        self.model = None
        self.processor = None
        # Call-time resolution: reads ASR_BATCH_SIZE env at construction so a
        # persistent worker honors per-job (and per-OOM-retry) batch sizes.
        self.request_batch_size = _resolve_asr_batch_size()
        # References to a previously timed-out worker-local generate that is
        # still running (PyTorch generate cannot be hard-interrupted; see
        # README). Kept so the next call can join the zombie before reusing
        # or replacing self.model, preventing two models from coexisting in
        # VRAM. See _join_zombie_worker.
        self._zombie_future = None
        self._zombie_executor = None

    def load(self, on_stage: Callable[[str], None] | None = None) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

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
        }

        if self.attention and self.attention != "sdpa":
            model_kwargs["attn_implementation"] = self.attention

        self.processor = AutoProcessor.from_pretrained(model_spec)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_spec,
            **model_kwargs,
        )
        self.model.eval()
        _apply_generation_safety(self.model)

    def unload_model(self, on_stage: Callable[[str], None] | None = None) -> None:
        # If a timed-out generate is still running inside the GPU worker, join
        # it first; dropping self.model from under the zombie would let the
        # next load double-allocate.
        if not self._join_zombie_worker(on_stage=on_stage):
            _notify(
                on_stage,
                "[WARN] ASR generate 僵尸线程未结束，跳过卸载以防两模型同驻 OOM",
            )
            return
        if self.model is None:
            return
        _notify(on_stage, "卸载 ASR 文本模型...")
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        self.processor = None
        _clear_cuda_cache(self.device)

    def close(self) -> None:
        self.unload_model()

    def _join_zombie_worker(
        self,
        *,
        on_stage: Callable[[str], None] | None = None,
    ) -> bool:
        """Wait for a previously timed-out generate thread to finish.

        PyTorch's native generate cannot be hard-interrupted mid-flight, so a
        timed-out transcribe leaves a worker thread running with the loaded
        model. If we cleared self.model in that state the next load() would
        allocate a second model while the zombie still holds the old one.

        We instead keep the zombie references on timeout and join here
        (bounded by the transcription timeout) before any reuse/replace.
        Returns True if no zombie is running (or it finished during the
        wait); False if it is still running, in which case the caller must
        NOT clear/replace self.model and must NOT allocate a second model.
        """
        future = self._zombie_future
        if future is None:
            return True
        join_budget = _transcription_timeout_s()
        _notify(
            on_stage,
            f"[WARN] 等待上一轮 ASR generate 超时僵尸线程收尾 (上限 {join_budget}s)",
        )
        try:
            future.result(timeout=join_budget)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "ASR zombie generate still running after %ss join wait; "
                "native generate cannot be hard-interrupted, keeping sole model "
                "reference to avoid double-allocation VRAM OOM",
                join_budget,
            )
            _notify(
                on_stage,
                "[WARN] 僵尸线程仍在运行，跳过本次操作以防两模型同驻 OOM",
            )
            return False
        except Exception:
            # Zombie generate raised; model state is unknown but the worker
            # is done so clearing here has no concurrency risk.
            logger.warning(
                "ASR zombie generate raised; will reload model on next load",
                exc_info=True,
            )
            try:
                del self.model
            except Exception:
                pass
            self.model = None
            self.processor = None

        executor = self._zombie_executor
        if executor is not None:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
        self._zombie_future = None
        self._zombie_executor = None
        _clear_cuda_cache(self.device)
        return True

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
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict]:
        language_hint = _asr_language() if _asr_force_language() else None

        # If a prior generate timed out and is still running, join it
        # (bounded) before touching the model -- never let load() spawn a
        # second model while the zombie holds the old one (VRAM OOM).
        if not self._join_zombie_worker(on_stage=on_stage):
            timeout_s = _transcription_timeout_s()
            _notify(
                on_stage,
                "[WARN] ASR generate 僵尸线程未结束，跳过本批次以防两模型同驻 OOM",
            )
            return [
                {
                    "text": "",
                    "raw_text": "",
                    "duration": _get_wav_duration_or_zero(path),
                    "language": language_hint or "Japanese",
                    "normalized_path": str(Path(path).resolve()),
                    "asr_generation": _qwen_generation_metadata(
                        error_kind="timeout",
                        error_detail=(
                            f"skipped: zombie generate still running after {timeout_s}s"
                        ),
                    ),
                    "log": [
                        (
                            "TIMEOUT: skipped: zombie generate still running "
                            f"after {timeout_s}s"
                        )
                    ],
                }
                for path in audio_paths
            ]

        if self.model is None:
            self.load(on_stage=on_stage)
        if not audio_paths:
            return []

        normalized_paths = [str(Path(audio_path).resolve()) for audio_path in audio_paths]

        _notify(on_stage, "ASR 文本转录中...")
        asr_results = None
        executor = None
        timed_out = False
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                self._transcribe_native,
                normalized_paths,
                language_hint,
            )
            try:
                timeout_s = _transcription_timeout_s()
                asr_results = future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                timed_out = True
                future.cancel()
                # The native generate keeps running in the worker thread and
                # cannot be hard-interrupted. Keep the
                # sole model reference and the running worker so the next call
                # can join the zombie before reusing/reloading -- never drop
                # self.model here, or load() would allocate a second model
                # while the zombie still holds the old one (VRAM OOM).
                self._zombie_future = future
                self._zombie_executor = executor
                logger.warning(
                    "ASR transcribe timed out after %ss; native generate "
                    "cannot be hard-interrupted, deferring model reload until the "
                    "zombie worker finishes (next call joins it)",
                    _transcription_timeout_s(),
                )
                _notify(
                    on_stage,
                    f"[WARN] ASR 超时 ({_transcription_timeout_s()}s)，跳过当前批次"
                    "（native generate 无法硬中断）",
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
                payload_log.append(f"ASR 加载生成上限: {_asr_max_new_tokens()}")
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

    def _transcribe_native(
        self,
        normalized_paths: list[str],
        language_hint: str | None,
    ) -> list[NativeAsrTranscription]:
        from asr.qwen_native import move_processor_inputs, prepare_transcription_inputs

        if self.model is None or self.processor is None:
            raise RuntimeError("ASR model is not loaded")

        results: list[NativeAsrTranscription] = []
        for start in range(0, len(normalized_paths), self.request_batch_size):
            paths = normalized_paths[start : start + self.request_batch_size]
            inputs = prepare_transcription_inputs(
                self.processor,
                audio=paths,
                language=language_hint,
            )
            moved = move_processor_inputs(
                inputs,
                device=self.device,
                dtype=self.dtype,
            )
            generated_ids = self.model.generate(
                **moved,
                max_new_tokens=_asr_max_new_tokens(),
                do_sample=False,
            )
            generated_suffix = generated_ids[:, moved["input_ids"].shape[1] :]
            decoded = self.processor.batch_decode(
                generated_suffix,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            parsed = self.processor.parse_output(decoded)
            if isinstance(parsed, dict):
                parsed = [parsed]
            for item in parsed:
                results.append(
                    NativeAsrTranscription(
                        language=str(item.get("language") or "").strip() or language_hint,
                        text=str(item.get("transcription") or ""),
                    )
                )
        return results

    def capture_asr_internals(self, chunks: list[dict], **_kwargs) -> list[dict]:
        """Capture ASR internals (encoder frames + token logits) in this backend.

        Reuses this backend's loaded Qwen3-ASR (no second model load). Each
        chunk: {path, text, start_s, end_s}. Returns one dict per chunk.
        """
        if not chunks:
            return []
        if not self._join_zombie_worker():
            return [
                {"ok": False, "error": "zombie generate still running"}
            ] * len(chunks)
        if self.model is None:
            self.load()
        try:
            from asr.asr_internals import AsrInternalsCapturer

            capturer = AsrInternalsCapturer(
                model=self.model,
                processor=self.processor,
            )
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

    def _alignment_window_for_text_result(
        self,
        text_result: dict,
        duration: float,
    ) -> tuple[float, float, str]:
        full_start = 0.0
        full_end = max(0.0, float(duration))
        try:
            start = float(text_result.get("alignment_window_start_s"))
            end = float(text_result.get("alignment_window_end_s"))
        except (TypeError, ValueError):
            return full_start, full_end, "chunk"

        start = max(full_start, min(full_end, start))
        end = max(start, min(full_end, end))
        if end - start < 0.05:
            return full_start, full_end, "chunk"
        source = str(text_result.get("alignment_window_source") or "chunk").strip()
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
            window_start, window_end, window_source = (
                self._alignment_window_for_text_result(text_result, duration)
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
                    timing_start=window_start,
                    timing_end=window_end,
                    timing_window_source=window_source,
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
