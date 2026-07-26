# Local model translation backend (transformers + vLLM)

import gc
import json
import os
import threading
import time
from typing import Callable

from llm.backends.base import BaseTranslationBackend


class TranslationCancelledError(RuntimeError):
    pass


class RetryableTranslationFormatError(RuntimeError):
    pass


def _cancel_requested(cancel_event: threading.Event | None) -> bool:
    try:
        return bool(cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if _cancel_requested(cancel_event):
        raise TranslationCancelledError("任务已取消")


class LocalModelBackend(BaseTranslationBackend):
    """本地模型后端（transformers 或 vLLM）"""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._load_lock = threading.Lock()

    def name(self) -> str:
        return "local"

    def supports_json_schema(self) -> bool:
        # 本地模型通过解析支持 JSON 输出
        return False

    def supports_reasoning(self) -> bool:
        # 取决于模型本身是否支持 CoT
        return False

    def supports_streaming(self) -> bool:
        return True

    def _load_model(self):
        """延迟加载模型"""
        model_path = os.getenv("LOCAL_MODEL_PATH", "").strip()
        if not model_path:
            raise RuntimeError(
                "LOCAL_MODEL_PATH must be set for local backend. "
                "Example: Qwen/Qwen2.5-72B-Instruct or Tencent-Hunyuan/Hunyuan-Large"
            )

        with self._load_lock:
            if self._model is not None and self._model_path == model_path:
                return

            device = os.getenv("LOCAL_MODEL_DEVICE", "cuda").strip()
            max_length = int(os.getenv("LOCAL_MODEL_MAX_LENGTH", "32768"))

            print(f"[local-backend] Loading model: {model_path} on {device}")
            print(f"[local-backend] Max context length: {max_length}")

            self._load_transformers_model(model_path, device, max_length)
            self._model_path = model_path

    def _load_transformers_model(self, model_path: str, device: str, max_length: int):
        """使用 transformers 加载模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise RuntimeError(
                "transformers and torch are required for local backend. "
                "Install with: uv pip install transformers torch"
            )

        auto_download = os.getenv("LOCAL_MODEL_AUTO_DOWNLOAD", "1").strip() == "1"

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=not auto_download,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device != "cpu" else None,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=not auto_download,
            )
            self._model.eval()
            self._max_length = max_length
        except Exception as e:
            if not auto_download and "local_files_only" in str(e):
                raise RuntimeError(
                    f"Model not found locally: {model_path}. "
                    "Set LOCAL_MODEL_AUTO_DOWNLOAD=1 to download from HuggingFace."
                ) from e
            raise

    def _load_vllm_model(self, model_path: str, device: str):
        """使用 vLLM 加载模型"""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise RuntimeError(
                "vLLM is required for vllm backend. "
                "Install with: uv pip install vllm"
            )

        from transformers import AutoTokenizer

        tensor_parallel_size = int(os.getenv("LLM_LOCAL_TENSOR_PARALLEL_SIZE", "1"))
        gpu_memory_utilization = float(os.getenv("LLM_LOCAL_GPU_MEMORY_UTILIZATION", "0.85"))
        dtype = os.getenv("LLM_LOCAL_DTYPE", "bfloat16").strip()

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=True,
        )

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
        cancel_event=None,
        on_progress: Callable[[dict], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
    ) -> str:
        """执行本地推理"""
        _raise_if_cancelled(cancel_event)

        self._load_model()

        return self._inference_transformers(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )

    def _inference_transformers(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        cancel_event,
        on_progress,
    ) -> str:
        """使用 transformers 推理"""
        import torch

        _raise_if_cancelled(cancel_event)

        # 应用 chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        self._emit_progress(on_progress, {"phase": "translating"})

        # 动态调整 max_new_tokens
        input_length = inputs["input_ids"].shape[1]
        available_length = self._max_length - input_length
        actual_max_tokens = min(max_tokens, available_length, 8192)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=actual_max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )

        _raise_if_cancelled(cancel_event)

        # 解码输出
        output_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        self._emit_progress(on_progress, {"phase": "done"})
        return output_text.strip()

    def unload_model(self):
        """卸载模型释放显存"""
        with self._load_lock:
            if self._model is not None:
                print("[local-backend] Unloading model to free GPU memory")
                del self._model
                del self._tokenizer
                self._model = None
                self._tokenizer = None
                self._model_path = None

                try:
                    import torch
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
