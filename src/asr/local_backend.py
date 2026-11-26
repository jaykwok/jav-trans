import concurrent.futures
import gc
import logging
import os
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from utils.model_paths import resolve_model_spec
from asr.backends.qwen import (
    DEFAULT_QWEN_ASR_FALLBACK_BATCH_SIZE,
    active_qwen_asr_model_id,
    active_qwen_asr_model_path,
    current_qwen_asr_backend,
)
from asr import alignment
from asr.alignment import AlignmentHead
from asr.decode_guard import (
    DEFAULT_BUDGET_SECONDS,
    build_stopping_criteria,
    plausible_token_budget,
)
from asr.subtitle_timing import (
    ALIGNED_TIMING_SOURCE,
    build_aligned_word_timestamps,
    build_boundary_word_timestamps,
)
from asr.text_normalize import normalize_display_text, strip_text_punctuation
from core.diagnostics import is_fatal_cuda_error
from core.typed_config import env_bool, env_float, env_int, env_text

logger = logging.getLogger(__name__)

ASR_LANGUAGE = env_text("ASR_LANGUAGE", "Japanese")


def _resolve_asr_batch_size() -> int:
    # The GPU worker resolves ASR_BATCH_SIZE=auto via `asr.vram_sizing` before
    # calling here. Direct/standalone callers with "auto" use a safe fallback.
    fallback = DEFAULT_QWEN_ASR_FALLBACK_BATCH_SIZE
    if env_text("ASR_BATCH_SIZE", "auto", lower=True) == "auto":
        return fallback
    return env_int("ASR_BATCH_SIZE", fallback, minimum=1)


TRANSCRIPTION_TIMEOUT_S = env_float("TRANSCRIPTION_TIMEOUT_S", 180.0, minimum=1.0)
# --- Windows Job Object: delegated to `utils.subprocess_tools` so all Windows
# child-management sites share one 64-bit-HANDLE-safe implementation. ---
if os.name == "nt":
    from utils.subprocess_tools import (
        assign_handle_to_job as _assign_handle_to_job,
        create_kill_on_close_job as _create_kill_on_close_job,
    )

    def _create_kill_on_close_job_object():
        return _create_kill_on_close_job()

    def _assign_process_to_job_object(job, process) -> None:
        # multiprocessing spawn Process keeps the Win32 process handle on its
        # Popen object (process._popen._handle).
        handle = getattr(getattr(process, "_popen", None), "_handle", None)
        if not handle:
            raise RuntimeError("subprocess has no win32 handle to assign")
        _assign_handle_to_job(job, int(handle))


def _is_fatal_cuda_oom(exc: BaseException) -> bool:
    return is_fatal_cuda_error(exc)

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


def _clear_cuda_cache(device: str, *, propagate_fatal: bool = True) -> None:
    if not device.startswith("cuda"):
        return
    outer_exc_active = sys.exc_info()[0] is not None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        if propagate_fatal and not outer_exc_active and is_fatal_cuda_error(exc):
            raise
        logger.warning("CUDA cache clear failed on %s: %s", device, exc)


def _detect_dtype(device: str):
    import torch

    dtype = os.getenv("ASR_DTYPE", "auto").strip().lower()
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _detect_attention(device: str) -> str:
    attention = os.getenv("ASR_ATTENTION", "auto").strip().lower()
    if attention != "auto":
        return attention
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
    model.generation_config.repetition_penalty = env_float(
        "ASR_REPETITION_PENALTY", 1.05, minimum=1.0
    )


def _asr_max_new_tokens(duration_s: float = DEFAULT_BUDGET_SECONDS) -> int:
    """The decode budget for a chunk of this length - see asr.decode_guard.

    Kept as a function of duration because a flat budget is the thing that was
    silently truncating dialogue: 128 tokens across a 30s chunk is 4.27 tok/s,
    and this domain reaches 4.45.
    """
    return plausible_token_budget(duration_s)


def _rows_truncated_at_cap(suffix, model, caps: Sequence[int] | int) -> list[bool]:
    """Per row: did it generate more tokens than its audio can hold?

    `caps` is one budget per row (an int is broadcast). Because the budget comes
    from the chunk's duration, reaching it is unambiguous - real speech cannot
    exceed `TOKENS_PER_SECOND_CEILING`, so a row that hit its own budget was
    generating rather than transcribing. That was not true of the old flat cap,
    where hitting 128 could equally mean a chunk with more speech than 128 tokens
    covers, and the tail of it never reached the subtitle.

    A row counts when it contains no stop token and emitted at least its own
    budget. Rows the loop guard cut are padded short of it, so they do not count
    here; they are `runaway_repetition` in the postgate report instead.
    """
    import torch

    stop_ids = getattr(model.generation_config, "eos_token_id", None)
    if stop_ids is None:
        stop_ids = getattr(model.config, "eos_token_id", None)
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    stop_ids = [int(item) for item in (stop_ids or [])]
    pad_id = getattr(model.generation_config, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(model.config, "pad_token_id", None)

    rows, length = suffix.shape
    if not stop_ids:
        # Without a stop token there is nothing to distinguish "finished" from
        # "ran out of budget", and guessing would over-report on every chunk.
        return [False] * rows
    budgets = [int(caps)] * rows if isinstance(caps, int) else [int(c) for c in caps]
    if len(budgets) != rows:
        return [False] * rows
    stopped = torch.isin(
        suffix, torch.tensor(stop_ids, device=suffix.device)
    ).any(dim=1)
    if pad_id is None or int(pad_id) in stop_ids:
        # Padding is indistinguishable from a stop token, so fall back to the
        # full row length. `stopped` already excludes every padded row.
        emitted = torch.full((rows,), length, device=suffix.device)
    else:
        emitted = (suffix != int(pad_id)).sum(dim=1)
    reached = emitted >= torch.tensor(budgets, device=suffix.device)
    return [bool(value) for value in ((~stopped) & reached).tolist()]


def _transcription_timeout_s() -> float:
    return env_float("TRANSCRIPTION_TIMEOUT_S", TRANSCRIPTION_TIMEOUT_S, minimum=1.0)


def _asr_language() -> str:
    return env_text("ASR_LANGUAGE", ASR_LANGUAGE)


def _asr_force_language() -> bool:
    return env_bool("ASR_FORCE_LANGUAGE", True)


def _qwen_generation_metadata(
    *,
    error_kind: str | None = None,
    error_detail: str = "",
    worker_mode: str = "gpu_worker",
    duration_s: float = DEFAULT_BUDGET_SECONDS,
) -> dict:
    return {
        "backend": current_qwen_asr_backend(),
        "model_id": active_qwen_asr_model_id(),
        "configured_max_new_tokens": _asr_max_new_tokens(duration_s),
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
            # Zero width means two different things, and only one of them is a
            # defect. Proportional timing can drift into a degenerate span for a
            # token that was actually spoken - drop those. But an acoustic-only
            # head gives every unpronounced character a zero-width span *on
            # purpose* (`alignment._spans_for_full_text`: a comma occupies no
            # audio, so taking width for it would steal width from a character
            # that was spoken). Dropping those deleted every comma and ellipsis
            # from the subtitle, because `transcribe._group_words_to_segments`
            # rebuilds the segment text by joining these tokens - and with the
            # punctuation went the boundaries that the vocalisation filter and
            # the layout DP both read.
            if any(alignment.is_acoustic_char(char) for char in token):
                continue
            end = start
        item = {"start": start, "end": end, "word": token}
        timestamp_kind = str(word.get("timestamp_kind") or "").strip()
        if timestamp_kind:
            item["timestamp_kind"] = timestamp_kind
        normalized.append(item)
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized


@dataclass(frozen=True)
class NativeAsrTranscription:
    language: str | None
    text: str
    truncated_at_cap: bool = False


def _wait_future_outcome(
    future: Any,
    timeout_s: float,
) -> tuple[str, BaseException | None]:
    """Distinguish a live wait timeout from a completed task raising TimeoutError.

    In Python 3.11+, `concurrent.futures.TimeoutError` is an alias for builtin
    `TimeoutError`, so `future.result(timeout=...)` cannot tell whether the wait
    timed out or the task itself finished by raising `TimeoutError`.
    """
    cancelled_fn = getattr(future, "cancelled", None)
    if callable(cancelled_fn) and cancelled_fn():
        return "done", None
    if isinstance(future, concurrent.futures.Future):
        done, _ = concurrent.futures.wait((future,), timeout=timeout_s)
        if future not in done:
            return "timeout", None
        if future.cancelled():
            return "done", None
        return "done", future.exception()
    try:
        future.result(timeout=timeout_s)
        return "done", None
    except concurrent.futures.TimeoutError as exc:
        if not future.done():
            return "timeout", None
        return "done", exc
    except Exception as exc:
        return "done", exc


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
        self._zombie_timed_out_future = None
        self._last_align_error = ""
        # Loaded lazily on first use and only when ASR_ALIGNMENT_HEAD_PATH is
        # set. `False` means "not looked at yet"; `None` means "looked, absent",
        # which stops a missing checkpoint from being re-probed per chunk.
        self._alignment_head: AlignmentHead | None | bool = False
        # Seconds spent inside load(), summed over every load this backend has
        # done. The pipeline no longer loads eagerly, so this is how the stage
        # timings still report what the weights actually cost - and report zero
        # on a rerun served entirely from the result cache.
        self.cumulative_load_s = 0.0

    def load(self, on_stage: Callable[[str], None] | None = None) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if self.model is not None:
            return
        load_started = time.perf_counter()

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
        self.cumulative_load_s += time.perf_counter() - load_started

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
        if self._zombie_timed_out_future is future and not future.done():
            # Already waited a full budget for this exact stuck generate thread;
            # don't burn another full timeout per chunk while it is still running.
            return False
        outer_exc_active = sys.exc_info()[0] is not None
        join_budget = _transcription_timeout_s()
        _notify(
            on_stage,
            f"[WARN] 等待上一轮 ASR generate 超时僵尸线程收尾 (上限 {join_budget}s)",
        )
        fatal_error: BaseException | None = None
        status, task_exc = _wait_future_outcome(future, join_budget)
        if status == "timeout":
            self._zombie_timed_out_future = future
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
        if task_exc is not None:
            # Zombie generate raised; model state is unknown but the worker
            # is done so clearing here has no concurrency risk. Fatal CUDA
            # faults poison the whole GPU context and must escalate to the
            # worker process after cleanup rather than being treated as recoverable.
            if is_fatal_cuda_error(task_exc):
                fatal_error = task_exc
            logger.warning(
                "ASR zombie generate raised; will reload model on next load",
                exc_info=task_exc,
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
        self._zombie_timed_out_future = None
        self._last_align_error = ""
        if fatal_error is not None:
            _clear_cuda_cache(self.device, propagate_fatal=False)
            if outer_exc_active:
                logger.error(
                    "ASR zombie generate failed with fatal CUDA error during exception teardown; "
                    "preserving active primary exception: %s",
                    fatal_error,
                    exc_info=fatal_error,
                )
                return True
            raise fatal_error
        _clear_cuda_cache(self.device, propagate_fatal=not outer_exc_active)
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

        generation = _qwen_generation_metadata(duration_s=duration)
        if getattr(asr_result, "truncated_at_cap", False):
            generation["truncated_at_cap"] = True
            # Both the flag and the line: the flag is what the pipeline counts,
            # the line is what survives into the run log for one chunk.
            log.append("ASR 解码超出音频可容纳的 token 量（判为失控，尾部已截断）")

        payload = {
            "text": master_text,
            "raw_text": raw_master_text,
            "duration": duration,
            "language": detected_language,
            "normalized_path": normalized_path,
            "asr_generation": generation,
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
            timeout_s = _transcription_timeout_s()
            done, _ = concurrent.futures.wait((future,), timeout=timeout_s)
            if future not in done:
                timed_out = True
                cancelled = future.cancel()
                if cancelled:
                    # The worker never began executing `_transcribe_native`, so
                    # nothing holds `self.model` and there is no zombie to join.
                    logger.warning(
                        "ASR transcribe timed out after %ss before worker started; "
                        "cancelled queued task cleanly",
                        timeout_s,
                    )
                    _notify(
                        on_stage,
                        f"[WARN] ASR 启动等待超时 ({timeout_s}s)，已取消未开始批次",
                    )
                elif future.done():
                    # Finished in the window between wait() and cancel(); if it
                    # ended with a fatal CUDA error, surface it instead of masking.
                    finished_exc = future.exception()
                    if finished_exc is not None and is_fatal_cuda_error(finished_exc):
                        raise finished_exc
                    logger.warning(
                        "ASR transcribe finished right after %ss timeout; skipping batch without zombie",
                        timeout_s,
                    )
                else:
                    # The native generate keeps running in the worker thread and
                    # cannot be hard-interrupted. Keep the sole model reference
                    # and the running worker so the next call can join the zombie
                    # before reusing/reloading -- never drop self.model here, or
                    # load() would allocate a second model while the zombie still
                    # holds the old one (VRAM OOM).
                    self._zombie_future = future
                    self._zombie_executor = executor
                    logger.warning(
                        "ASR transcribe timed out after %ss; native generate "
                        "cannot be hard-interrupted, deferring model reload until the "
                        "zombie worker finishes (next call joins it)",
                        timeout_s,
                    )
                    _notify(
                        on_stage,
                        f"[WARN] ASR 超时 ({timeout_s}s)，跳过当前批次"
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
                            error_detail=f"skipped after {timeout_s}s",
                        ),
                        "log": [
                            f"TIMEOUT: skipped after {timeout_s}s"
                        ],
                    }
                    for path in normalized_paths
                ]
            asr_results = future.result()

            payloads: list[dict] = []
            for normalized_path, asr_result in zip(normalized_paths, asr_results):
                payload, payload_log = self._build_text_result(
                    normalized_path,
                    asr_result,
                    language_hint,
                )
                payload_log.append(
                    "ASR 生成上限: "
                    f"{_asr_max_new_tokens(payload.get('duration') or 0.0)}"
                    f"（{payload.get('duration') or 0.0:.1f}s 音频派生）"
                )
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
            # Each row gets the budget its own duration can fill, and the batch
            # runs to the largest of them - a sequence stuck in a repetition loop
            # never emits EOS, and `generate` returns only when every sequence is
            # done, so without the per-row stop the shortest chunk in the batch
            # would be free to generate the longest chunk's worth of tokens. See
            # asr.decode_guard for why the bound is arithmetic.
            budgets = [
                plausible_token_budget(_get_wav_duration_or_zero(path)) for path in paths
            ]
            cap = max(budgets)
            stopping_criteria = build_stopping_criteria(
                int(moved["input_ids"].shape[1]), token_budgets=budgets
            )
            generated_ids = self.model.generate(
                **moved,
                max_new_tokens=cap,
                do_sample=False,
                **({"stopping_criteria": stopping_criteria} if stopping_criteria else {}),
            )
            generated_suffix = generated_ids[:, moved["input_ids"].shape[1] :]
            try:
                truncated = _rows_truncated_at_cap(generated_suffix, self.model, budgets)
            except Exception as error:  # noqa: BLE001
                if is_fatal_cuda_error(error):
                    raise
                # Instrumentation must never fail a transcription that succeeded.
                logger.warning("decode cap accounting failed: %s", error)
                truncated = [False] * len(paths)
            decoded = self.processor.batch_decode(
                generated_suffix,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            parsed = self.processor.parse_output(decoded)
            if isinstance(parsed, dict):
                parsed = [parsed]
            for offset, item in enumerate(parsed):
                results.append(
                    NativeAsrTranscription(
                        language=str(item.get("language") or "").strip() or language_hint,
                        text=str(item.get("transcription") or ""),
                        truncated_at_cap=bool(
                            truncated[offset] if offset < len(truncated) else False
                        ),
                    )
                )
        return results

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
        cached_features: Any = None,
    ) -> tuple[dict, list[str]]:
        log.append("Subtitle timing: boundary_chunk_timeline")
        text = master_text or raw_master_text
        self._last_align_error = ""
        aligned = self._align_characters(
            normalized_path, text, log, cached_features=cached_features
        )
        align_error = self._last_align_error
        if aligned is not None:
            char_spans, acoustic_extent = aligned
            word_dicts, alignment_mode, timing_meta = build_aligned_word_timestamps(
                text,
                char_spans,
                timing_start,
                timing_end,
                acoustic_extent,
            )
        else:
            word_dicts, alignment_mode, timing_meta = build_boundary_word_timestamps(
                text,
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
            align_error=align_error,
            timing_meta=timing_meta,
            timing_window_source=timing_window_source,
        )

    def _resolve_alignment_head(self, log: list[str]) -> AlignmentHead | None:
        if self._alignment_head is not False:
            return self._alignment_head or None
        try:
            self._alignment_head = AlignmentHead.from_env()
        except Exception as error:  # noqa: BLE001
            if is_fatal_cuda_error(error):
                raise
            # A bad checkpoint must not take the transcription down with it;
            # synthetic timing is a working fallback, and the log says why.
            logger.warning("alignment head unavailable: %s", error)
            log.append(f"Subtitle timing: alignment head unavailable ({error})")
            self._alignment_head = None
        return self._alignment_head or None

    def _align_characters(
        self,
        normalized_path: str,
        text: str,
        log: list[str],
        *,
        cached_features: Any = None,
    ) -> tuple[list, tuple[float, float]] | None:
        """Character times for `text`, measured off this chunk's own audio.

        Returns the spans together with the utterance's acoustic extent, which
        is a different measurement off the same tensor: the spans say where each
        character is, the extent says where the sound starts and stops. Line
        edges need the second one - see `alignment.speech_extent`.

        Runs the encoder a second time on this chunk rather than reusing the
        forward from `generate`. That is affordable precisely because of the
        asymmetry the redesign rests on - encoder RTF is 0.00069 against 0.12273
        for decode, so a whole extra encoder pass costs about half a percent of
        the decode that just happened.
        """
        head = self._resolve_alignment_head(log)
        if head is None or not (text or "").strip():
            return None
        try:
            import numpy as np
            import torch

            from asr.qwen_native import (
                move_processor_inputs,
                prepare_transcription_inputs,
            )
            from audio.loading import load_audio_16k_mono
            from asr.encoder_features import qwen3_asr_audio_output_lengths

            if self.model is None or self.processor is None:
                return None
            if cached_features is not None:
                # Already encoded with this chunk's siblings in one forward pass.
                chunk_features = cached_features
            else:
                if not self._join_zombie_worker():
                    detail = "zombie ASR generate worker still active; skipping encoder pass"
                    self._last_align_error = detail
                    logger.warning("character alignment skipped: %s", detail)
                    log.append(f"Subtitle timing: alignment failed ({detail})")
                    return None
                audio, rate = load_audio_16k_mono(normalized_path)
                if rate != 16000:
                    return None
                inputs = prepare_transcription_inputs(
                    self.processor,
                    audio=[np.asarray(audio, dtype=np.float32)],
                    language=ASR_LANGUAGE,
                )
                moved = move_processor_inputs(
                    inputs, device=self.device, dtype=self.dtype
                )
                with torch.inference_mode():
                    features = self.model.get_audio_features(
                        input_features=moved["input_features"],
                        input_features_mask=moved["input_features_mask"],
                    ).pooler_output
                frames = int(
                    qwen3_asr_audio_output_lengths(
                        moved["input_features_mask"].sum(dim=1)
                    )[0]
                )
                chunk_features = features[:frames].detach().float().cpu().numpy()
            aligned = head.align_extent(chunk_features, text)
            if not aligned:
                log.append("Subtitle timing: alignment declined, using proportional")
                return None
            spans, extent_start, extent_end = aligned
            log.append(f"Subtitle timing: aligned {len(spans)} characters")
            return spans, (extent_start, extent_end)
        except Exception as error:  # noqa: BLE001
            if is_fatal_cuda_error(error):
                raise
            detail = f"{type(error).__name__}: {error}"
            self._last_align_error = detail
            logger.warning("character alignment failed: %s", detail, exc_info=True)
            log.append(f"Subtitle timing: alignment failed ({detail})")
            return None

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
            source = timing_meta["timing_source"]
            head = self._alignment_head if self._alignment_head else None
            if source == ALIGNED_TIMING_SOURCE and head is not None and head.checkpoint_path:
                # The algorithm tag ("ctc_forced_alignment_v1") names the
                # timing method, not the checkpoint -- it looks like a head
                # version and isn't one, so show the actual file in use.
                source = Path(head.checkpoint_path).name
            log.append(f"Subtitle timing source: {source}")
        log.append(f"Subtitle timing mode: {alignment_mode}")
        return {
            "words": word_dicts,
            "text": master_text,
            "raw_text": raw_master_text,
            "alignment_mode": alignment_mode,
            "align_error": align_error,
            "duration": duration,
            "language": detected_language,
            # Carried out rather than only logged: `alignment_score` in here is
            # the post-gate's one signal for text the acoustics do not support,
            # and it is only measurable at this point, where the characters were
            # aligned against this chunk's own audio.
            "timing_meta": dict(timing_meta or {}),
        }, log

    def _encode_chunk_features(self, paths: list[str]) -> dict[str, Any]:
        """Encoder frames for several chunks in one forward pass.

        The alignment pass used to encode one chunk at a time, re-reading the wav
        and recomputing mel on the CPU for each - 37.76s over 384 chunks on a
        2h09m file. The encoder is compute-bound, so the per-call overhead was
        most of that. Returns `{path: frames}` and simply omits anything that
        failed to load, because the caller already degrades to proportional
        timing when a chunk has no features.
        """
        import numpy as np
        import torch

        from asr.qwen_native import move_processor_inputs, prepare_transcription_inputs
        from audio.loading import load_audio_16k_mono
        from asr.encoder_features import qwen3_asr_audio_output_lengths

        loaded: list[tuple[str, Any]] = []
        for path in paths:
            try:
                audio, rate = load_audio_16k_mono(path)
            except Exception as error:  # noqa: BLE001
                logger.warning("alignment audio load failed for %s: %s", path, error)
                continue
            if rate != 16000:
                continue
            loaded.append((path, np.asarray(audio, dtype=np.float32)))
        if not loaded:
            return {}
        if not self._join_zombie_worker():
            logger.warning(
                "Skipping batched alignment encode: zombie ASR generate worker is still active"
            )
            return {}

        inputs = prepare_transcription_inputs(
            self.processor,
            audio=[audio for _, audio in loaded],
            language=ASR_LANGUAGE,
        )
        moved = move_processor_inputs(inputs, device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            features = self.model.get_audio_features(
                input_features=moved["input_features"],
                input_features_mask=moved["input_features_mask"],
            ).pooler_output
        lengths = [
            int(value)
            for value in qwen3_asr_audio_output_lengths(
                moved["input_features_mask"].sum(dim=1)
            )
        ]
        # `pooler_output` concatenates the batch along the frame axis, matching
        # `encoder_features.Qwen3AsrEncoder.encode_batch`.
        hidden = features.detach().float().cpu().numpy()
        expected_total_frames = sum(lengths)
        if expected_total_frames != int(hidden.shape[0]):
            raise RuntimeError(
                "ASR encoder output length mismatch during alignment: "
                f"expected_frames={expected_total_frames} actual_frames={int(hidden.shape[0])}"
            )
        encoded: dict[str, Any] = {}
        offset = 0
        for (path, _), length in zip(loaded, lengths):
            encoded[path] = hidden[offset : offset + length]
            offset += length
        return encoded

    def finalize_text_results(
        self,
        text_results: list[dict],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[tuple[dict, list[str]]]:
        if not text_results:
            return []

        batch_size = env_int("ASR_ALIGN_BATCH_SIZE", 4, minimum=1)
        head_notes: list[str] = []
        primary_head = self._resolve_alignment_head(head_notes)
        head_available = (
            primary_head is not None
            and self.model is not None
            and self.processor is not None
        )

        finalized: list[tuple[dict, list[str]]] = []
        total_results = len(text_results)
        # Check once per finalize pass so a stuck generate thread never burns a
        # full timeout for batched encode and again for each individual chunk.
        zombie_active = head_available and not self._join_zombie_worker(on_stage=on_stage)
        if zombie_active:
            logger.warning(
                "Skipping alignment encoder passes for %d chunks: zombie ASR generate worker is still active",
                total_results,
            )
        for group_start in range(0, total_results, batch_size):
            group = text_results[group_start : group_start + batch_size]
            feature_cache: dict[str, Any] = {}
            if head_available and not zombie_active:
                alignable = [
                    str(item["normalized_path"])
                    for item in group
                    if str(item.get("text", "")).strip()
                ]
                if alignable:
                    try:
                        feature_cache = self._encode_chunk_features(alignable)
                    except Exception as error:  # noqa: BLE001
                        if is_fatal_cuda_error(error):
                            raise
                        # One bad batch must not lose the transcription; every
                        # chunk below falls back to its own encode or to
                        # proportional timing.
                        logger.warning(
                            "batched chunk encode failed: %s", error, exc_info=True
                        )
                        feature_cache = {}
            group_finalized = self._finalize_group(group, feature_cache)
            if head_notes and group_start == 0 and group_finalized:
                group_finalized[0][1].extend(head_notes)
            finalized.extend(group_finalized)
            if on_stage is not None:
                on_stage(f"字幕时间轴 {len(finalized)}/{total_results}...")
        return finalized

    def _finalize_group(
        self,
        text_results: list[dict],
        feature_cache: dict[str, Any],
    ) -> list[tuple[dict, list[str]]]:
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
                    cached_features=feature_cache.get(normalized_path),
                )
            )
        return finalized
