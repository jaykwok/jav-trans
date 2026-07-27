# 翻译后端架构

## 设计目标

翻译系统只有一个编排核心，负责 Prompt、全片上下文、全局术语、批处理、缓存、缺失项重试、修复、进度和取消。后端只负责执行一次消息请求并返回文本。

这个边界刻意避免把批处理和修复拆成各自维护状态的副本。此前的拆分曾丢失 Responses API、全局术语提取、取消语义和缓存统计，因此 `batching.py`、`repair.py` 与 `translator_legacy.py` 已删除。

```text
src/llm/
├── translator.py             # 唯一编排核心与 OpenAI canonical transport
├── prompt.py                 # Prompt 和字幕 JSON 构造
├── cache.py                  # batch cache 与 translation memory
├── glossary.py               # 用户词表解析
├── errors.py                 # 所有层共享的异常类型
└── backends/
    ├── __init__.py           # 注册表、选择和进程级实例生命周期
    ├── base.py               # 后端抽象基类
    ├── openai_compat.py      # Chat/Responses canonical transport 适配器
    └── local_model.py        # 进程内 Transformers 后端
```

## 后端契约

自定义后端实现 `BaseTranslationBackend.chat_completion()`。输入包括消息、采样参数、结构化输出 schema、任务级 `api_format`、取消事件和进度/用量回调；返回最终文本。

后端必须遵守以下规则：

- 取消统一抛出 `llm.errors.TranslationCancelledError`，不能自行定义同名异常。
- 临时响应格式错误使用共享的 retryable 异常。
- `cache_identity()` 必须能区分会改变译文的模型或服务。
- 一个注册名在进程中只创建一个实例；配置变化通过 `reset_backend()` 释放旧实例。
- 后端不得自行实现字幕分批、translation memory 或 repair pass。

## OpenAI 兼容后端

```env
TRANSLATION_BACKEND=openai
OPENAI_COMPATIBILITY_BASE_URL=https://api.deepseek.com
API_KEY=your-api-key
LLM_MODEL_NAME=your-model
LLM_API_FORMAT=chat
LLM_REASONING_EFFORT=medium
```

`LLM_API_FORMAT` 支持 `chat` 和 `responses`。调用 `translate_segments(..., api_format=...)` 时，任务级参数优先于进程环境变量。

Chat 和 Responses 的流式进度、usage、JSON Schema、DeepSeek `json_object` 兼容以及 Grok provider patch 只有一份 canonical 实现，避免适配器与主流程分叉。

生产环境运行本地大模型时，推荐启动 vLLM/SGLang 等 OpenAI 兼容服务，然后仍选择 `openai` 后端。这样服务负责 continuous batching、KV cache 和多卡调度，本程序不会与 ASR 在同一进程争抢模型生命周期。

## 进程内 Transformers 后端

```env
TRANSLATION_BACKEND=local
LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
LOCAL_MODEL_DEVICE=cuda
LOCAL_MODEL_DTYPE=bfloat16
LOCAL_MODEL_MAX_LENGTH=32768
LOCAL_MODEL_BATCH_SIZE=16
LOCAL_MODEL_MAX_NEW_TOKENS=8192
LOCAL_MODEL_AUTO_DOWNLOAD=1
```

行为约束：

- 注册表复用一个 `LocalModelBackend`，不会为每个 batch worker 加载模型。
- `generate()` 串行执行，避免并发 KV cache 放大显存；等待推理锁时可取消。
- CUDA 不可用、dtype 非法、模型缺失及上下文超限都会给出明确错误。
- CPU 默认使用 `float32`，CUDA 默认使用 `bfloat16`；可用 `LOCAL_MODEL_DTYPE` 覆盖。
- 上下文会使用较小 batch 和有界全片摘要；输入仍超过模型窗口时终止，不把负数传给 `max_new_tokens`。
- 当前实现不是 token streaming，因此 `supports_streaming()` 返回 `False`；仍会发送阶段进度和 token usage。
- 从 Web 切换后端或修改本地模型关键配置时，会先释放旧实例再允许加载新实例。

## 缓存隔离

缓存签名包含 Prompt 版本、目标语言、词表、人物参考和后端 `cache_identity()`。API 模型与本地模型不会复用同一翻译缓存。

全局术语提取使用独立的 `terms` JSON Schema，不再错误套用字幕 `translations` Schema。修复后的译文会写回同一个 batch cache key。

## 扩展后端

```python
from llm.backends import register_backend
from llm.backends.base import BaseTranslationBackend


class MyBackend(BaseTranslationBackend):
    def name(self) -> str:
        return "my-backend"

    def cache_identity(self) -> str:
        return "my-backend:model-v1"

    def chat_completion(self, messages, **kwargs) -> str:
        self._raise_if_cancelled(kwargs.get("cancel_event"))
        return call_my_model(messages)


register_backend("my-backend", MyBackend)
```

注册名重复默认报错；开发期确需替换时显式传 `replace=True`。

## 验证

```powershell
$env:PYTHONIOENCODING = "utf-8"
$translationTests = rg --files tests | Where-Object { $_ -match 'translation|translator|batch_translation|glossary_preextract|compact_prompt|request_backoff' }
uv run --no-sync pytest -q $translationTests
```
