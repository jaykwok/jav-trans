# Translation backend architecture documentation

## 概览

重构后的翻译模块采用可插拔的后端架构，支持：

1. **OpenAI 兼容 API**（DeepSeek、OpenAI、Azure 等）
2. **本地模型**（transformers、vLLM）
3. **未来可扩展**（Claude、通义千问等）

## 架构

```
src/llm/
├── backends/
│   ├── __init__.py           # 后端注册表和工厂
│   ├── base.py              # 抽象基类
│   ├── openai_compat.py     # OpenAI 兼容后端
│   └── local_model.py       # 本地模型后端
├── translator.py            # 统一入口（保持 API 兼容）
├── batching.py              # 批处理和并发
├── repair.py                # 翻译修复
├── cache.py                 # 缓存机制（已有）
├── prompt.py                # Prompt 构建（已有）
└── glossary.py              # 术语表（已有）
```

## 配置

### OpenAI 兼容 API

```env
LLM_BACKEND_TYPE=openai
OPENAI_COMPATIBILITY_BASE_URL=https://api.deepseek.com
API_KEY=your-api-key
LLM_MODEL_NAME=deepseek-v4-flash
LLM_API_FORMAT=chat
LLM_REASONING_EFFORT=medium
```

### 本地模型（transformers）

```env
LLM_BACKEND_TYPE=local
LLM_LOCAL_BACKEND=transformers
LLM_LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
LLM_LOCAL_DEVICE=cuda
LLM_LOCAL_DTYPE=bfloat16
```

### 本地模型（vLLM，推荐）

```env
LLM_BACKEND_TYPE=local
LLM_LOCAL_BACKEND=vllm
LLM_LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
LLM_LOCAL_DEVICE=cuda
LLM_LOCAL_DTYPE=bfloat16
LLM_LOCAL_GPU_MEMORY_UTILIZATION=0.85
LLM_LOCAL_TENSOR_PARALLEL_SIZE=1
```

### vLLM OpenAI 兼容服务（推荐用于生产）

启动 vLLM 服务：
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --api-key dummy \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9
```

配置：
```env
LLM_BACKEND_TYPE=openai
OPENAI_COMPATIBILITY_BASE_URL=http://localhost:8000/v1
API_KEY=dummy
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

## 本地模型推荐

### 小型（7B-14B，适合 8GB+ 显存）
- `Qwen/Qwen2.5-7B-Instruct`
- `internlm/internlm2_5-7b-chat`
- `THUDM/glm-4-9b-chat`

### 中型（20B-32B，适合 24GB+ 显存）
- `Qwen/Qwen2.5-32B-Instruct`
- `internlm/internlm2_5-20b-chat`

### 大型（70B+，需要多卡或量化）
- `Qwen/Qwen2.5-72B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`

## 显存协调

### 方案 1：外部服务（推荐）

翻译服务独立运行：
```bash
# 终端 1：ASR 服务（使用主 GPU）
uv run python launcher.py

# 终端 2：vLLM 翻译服务（使用另一个 GPU 或 CPU）
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

### 方案 2：错峰使用

ASR 和翻译串行执行，自动切换：
```env
LLM_BACKEND_TYPE=local
LLM_LOCAL_BACKEND=transformers
# ASR 完成后自动加载翻译模型
```

**注意**：本地模型与 ASR 共享显存时，需要预留足够空间。

### 方案 3：降低 ASR 显存占用

```env
ASR_STAGE_WORKER_VRAM_RATIO=0.50  # ASR 只用 50% 显存
ASR_BATCH_SIZE=2                   # 降低 ASR batch size
```

## 性能对比

| 后端 | 优点 | 缺点 |
|------|------|------|
| OpenAI API | 质量高、无需显存 | 需要网络、有费用 |
| vLLM 服务 | 高性能、灵活 | 需要额外服务 |
| transformers | 一体化、简单 | 性能较低、显存冲突 |

## 扩展新后端

1. 创建新后端文件 `src/llm/backends/my_backend.py`
2. 继承 `BaseTranslationBackend`
3. 实现 `chat_completion` 方法
4. 在 `backends/__init__.py` 中注册

示例：
```python
from llm.backends.base import BaseTranslationBackend
from llm.backends import register_backend

class MyBackend(BaseTranslationBackend):
    def name(self) -> str:
        return "my_backend"
    
    def chat_completion(self, messages, **kwargs) -> str:
        # 实现翻译逻辑
        ...

# 注册
register_backend("my_backend", MyBackend)
```

配置：
```env
LLM_BACKEND_TYPE=my_backend
```

## 向后兼容

`translator.py` 保持原有 API 接口不变：
```python
from llm.translator import translate_segments

zh_texts, timings, retry_events = translate_segments(
    segments,
    max_workers=4,
    cache_path="./cache",
    target_lang="简体中文",
    glossary="...",
)
```

现有代码无需修改即可使用新架构。

## 测试

```powershell
# 测试 OpenAI 后端
$env:LLM_BACKEND_TYPE="openai"
uv run pytest tests/test_translation*.py

# 测试本地模型后端
$env:LLM_BACKEND_TYPE="local"
$env:LLM_LOCAL_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
uv run pytest tests/test_translation*.py
```

## 迁移指南

旧代码无需修改，但可以利用新功能：

### 使用本地模型

只需修改 `.env`：
```env
LLM_BACKEND_TYPE=local
LLM_LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
```

### 切换后端

运行时动态切换：
```python
import os
os.environ["LLM_BACKEND_TYPE"] = "local"

from llm.translator import translate_segments
# 现在使用本地模型
```

### 卸载模型释放显存

```python
from llm.backends import get_backend

backend = get_backend("local")
if hasattr(backend, "unload_model"):
    backend.unload_model()  # 释放显存
```
