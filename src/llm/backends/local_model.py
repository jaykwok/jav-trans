"""In-process Transformers translation backend.

The backend is intentionally serialized. A Transformers model is already able
to use all configured devices; starting independent Python generations against
the same weights provides little benefit and can multiply KV-cache memory.
For production concurrency, point the OpenAI-compatible backend at vLLM or
another dedicated inference server instead.
"""

from __future__ import annotations

import gc
import os
import threading
from typing import Callable

from llm.backends.base import BaseTranslationBackend
from llm.errors import TranslationContextLengthError


class LocalModelBackend(BaseTranslationBackend):
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_key: tuple[str, str, str, bool] | None = None
        self._load_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._max_length = 0

    def name(self) -> str:
        return "local"

    def cache_identity(self) -> str:
        path = os.getenv("LOCAL_MODEL_PATH", "").strip()
        return f"local:{path}"

    def supports_json_schema(self) -> bool:
        return False

    def supports_reasoning(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return False

    @staticmethod
    def _read_config() -> tuple[str, str, str, bool, int]:
        model_path = os.getenv("LOCAL_MODEL_PATH", "").strip()
        if not model_path:
            raise RuntimeError(
                "使用本地翻译后端前必须设置 LOCAL_MODEL_PATH。"
            )

        device = (os.getenv("LOCAL_MODEL_DEVICE", "cuda") or "cuda").strip().lower()
        if device not in {"cuda", "cpu"}:
            raise RuntimeError("LOCAL_MODEL_DEVICE must be 'cuda' or 'cpu'")

        dtype = (os.getenv("LOCAL_MODEL_DTYPE", "") or "").strip().lower()
        if not dtype:
            dtype = "float32" if device == "cpu" else "bfloat16"
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise RuntimeError(
                "LOCAL_MODEL_DTYPE must be float32, float16, or bfloat16"
            )

        auto_download = (
            os.getenv("LOCAL_MODEL_AUTO_DOWNLOAD", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        try:
            max_length = int(os.getenv("LOCAL_MODEL_MAX_LENGTH", "32768"))
        except ValueError as exc:
            raise RuntimeError("LOCAL_MODEL_MAX_LENGTH must be an integer") from exc
        if max_length < 512:
            raise RuntimeError("LOCAL_MODEL_MAX_LENGTH must be at least 512")
        return model_path, device, dtype, auto_download, max_length

    def _ensure_model(self) -> None:
        model_path, device, dtype_name, auto_download, max_length = self._read_config()
        model_key = (model_path, device, dtype_name, auto_download)

        with self._load_lock:
            if self._model is not None and self._model_key == model_key:
                self._max_length = max_length
                return
            if self._model is not None:
                self._unload_locked()

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "本地翻译后端需要 transformers 和 torch。"
                ) from exc

            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "LOCAL_MODEL_DEVICE=cuda，但当前进程无法使用 CUDA。"
                )

            torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[dtype_name]
            local_only = not auto_download

            print(
                f"[translation/local] loading model={model_path} "
                f"device={device} dtype={dtype_name}",
                flush=True,
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                if device == "cpu":
                    model = model.to("cpu")
                model.eval()
            except Exception as exc:
                if local_only:
                    raise RuntimeError(
                        f"本地没有可用模型 {model_path}；可启用自动下载或填写本地路径。"
                    ) from exc
                raise

            self._tokenizer = tokenizer
            self._model = model
            self._model_key = model_key
            self._max_length = max_length

    def chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 384000,
        response_format: dict | None = None,
        stream: bool = True,
        reasoning_effort: str | None = None,
        api_format: str | None = None,
        expected_count: int = 0,
        cancel_event=None,
        on_progress: Callable[[dict], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
    ) -> str:
        del response_format, stream, reasoning_effort, api_format, expected_count
        self._raise_if_cancelled(cancel_event)

        # A shared model must not receive concurrent generate() calls. Waiting
        # workers poll cancellation rather than blocking indefinitely. Loading
        # is inside the same critical section so a settings reset cannot unload
        # the model between _ensure_model() and generate().
        while not self._inference_lock.acquire(timeout=0.1):
            self._raise_if_cancelled(cancel_event)
        try:
            self._raise_if_cancelled(cancel_event)
            self._ensure_model()
            return self._generate(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                cancel_event=cancel_event,
                on_progress=on_progress,
                on_usage=on_usage,
            )
        finally:
            self._inference_lock.release()

    def _generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        cancel_event,
        on_progress,
        on_usage,
    ) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class CancelStoppingCriteria(StoppingCriteria):
            def __call__(self, *args, **kwargs) -> bool:
                try:
                    return bool(cancel_event is not None and cancel_event.is_set())
                except Exception:
                    return False

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_length = int(inputs["input_ids"].shape[1])

        tokenizer_limit = getattr(self._tokenizer, "model_max_length", None)
        useful_tokenizer_limit = (
            int(tokenizer_limit)
            if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9
            else self._max_length
        )
        context_limit = min(self._max_length, useful_tokenizer_limit)
        available_length = context_limit - input_length
        if available_length <= 0:
            raise TranslationContextLengthError(
                f"本地模型上下文已超限：输入 {input_length} tokens，"
                f"上限 {context_limit} tokens。请减小翻译批次或提高上下文配置。"
            )

        try:
            generation_cap = int(os.getenv("LOCAL_MODEL_MAX_NEW_TOKENS", "8192"))
        except ValueError:
            generation_cap = 8192
        actual_max_tokens = min(
            max(1, int(max_tokens)),
            available_length,
            max(1, generation_cap),
        )

        model_device = getattr(self._model, "device", None)
        if model_device is not None and getattr(model_device, "type", None) != "meta":
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        self._emit_progress(on_progress, {"phase": "translating"})
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=actual_max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=pad_token_id,
                stopping_criteria=StoppingCriteriaList([CancelStoppingCriteria()]),
            )

        self._raise_if_cancelled(cancel_event)
        completion_tokens = int(outputs[0].shape[0]) - input_length
        output_text = self._tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        ).strip()
        self._emit_usage(
            on_usage,
            {
                "prompt_tokens": input_length,
                "completion_tokens": max(0, completion_tokens),
                "total_tokens": input_length + max(0, completion_tokens),
            },
        )
        self._emit_progress(on_progress, {"phase": "done"})
        return output_text

    def _unload_locked(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload_model(self) -> None:
        with self._inference_lock:
            with self._load_lock:
                self._unload_locked()

    def close(self) -> None:
        self.unload_model()
