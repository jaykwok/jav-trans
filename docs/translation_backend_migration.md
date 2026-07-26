# Translation Backend Migration Guide

## 架构重构总结

已将原先 2700+ 行的单文件翻译器重构为模块化后端架构，支持灵活切换翻译引擎。

## 新架构

```
src/llm/
├── backends/
│   ├── __init__.py          # 后端注册与工厂
│   ├── base.py             # 抽象基类 TranslationBackend
│   ├── openai_compat.py    # OpenAI 兼容后端（API 调用）
│   └── local_model.py      # 本地模型后端（transformers/vLLM）
├── prompt.py               # Prompt 构建逻辑
├── streaming.py            # 流式响应处理
├── repair.py               # 翻译修复逻辑
├── batching.py             # 批处理与并发控制
└── translator.py           # 统一入口（保持 API 兼容）
```

## 核心概念

### 1. TranslationBackend 抽象

所有后端继承 `TranslationBackend` 基类，实现：

- `translate(messages, **kwargs)` - 同步翻译
- `translate_stream(messages, **kwargs)` - 流式翻译
- `supports_json_schema` - 是否支持结构化输出
- `supports_reasoning` - 是否支持推理模式

### 2. 后端注册

```python
from llm.backends import register_backend, get_backend

# 注册自定义后端
register_backend("my_backend", MyBackendClass)

# 获取后端实例
backend = get_backend("openai")  # or "local"
```

### 3. 环境变量配置

```bash
# 选择翻译后端
TRANSLATION_BACKEND=openai  # 或 local

# OpenAI 兼容后端
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# 本地模型后端
LOCAL_MODEL_PATH=/path/to/model
LOCAL_MODEL_MAX_LENGTH=32768
LOCAL_MODEL_DEVICE=cuda  # 或 cpu
```

## 使用示例

### 基础翻译（API 兼容）

```python
from llm.translator import translate_segments

segments = [
    {"start": 0.0, "end": 2.5, "text": "こんにちは"},
    {"start": 2.5, "end": 5.0, "text": "ありがとう"},
]

translations, timings, retries = translate_segments(
    segments,
    target_lang="简体中文",
    character_reference="Aya Onami",
    max_workers=4,
)
```

### 直接使用后端

```python
from llm.backends import get_backend
from llm.prompt import build_translation_messages

backend = get_backend("openai")

segments = [{"start": 0, "end": 1, "text": "テスト"}]
messages = build_translation_messages(
    segments,
    target_lang="简体中文",
    character_reference="",
)

response = backend.translate(messages, temperature=0.3)
print(response["content"])
```

### 流式翻译

```python
backend = get_backend("openai")

for chunk in backend.translate_stream(messages, temperature=0.3):
    if chunk["type"] == "content":
        print(chunk["delta"], end="", flush=True)
    elif chunk["type"] == "done":
        print(f"\n\nTokens: {chunk['usage']}")
```

## 添加新后端

### 示例：混元本地模型

```python
# src/llm/backends/hunyuan.py
from llm.backends.base import TranslationBackend
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HunyuanBackend(TranslationBackend):
    def __init__(self):
        self.model_path = os.getenv("HUNYUAN_MODEL_PATH")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    def translate(self, messages, **kwargs):
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.3),
            do_sample=True,
        )
        
        content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "content": content,
            "usage": {"prompt_tokens": len(inputs[0]), "completion_tokens": len(outputs[0])},
        }
    
    def translate_stream(self, messages, **kwargs):
        # 流式生成实现
        pass
    
    def supports_json_schema(self):
        return False  # 多数本地模型不支持
    
    def supports_reasoning(self):
        return False

# 注册后端
from llm.backends import register_backend
register_backend("hunyuan", HunyuanBackend)
```

### 使用新后端

```bash
export TRANSLATION_BACKEND=hunyuan
export HUNYUAN_MODEL_PATH=/models/hunyuan-large

python src/translate.py --input video.srt
```

## 性能特性

### OpenAI 兼容后端
- ✅ JSON Schema 结构化输出
- ✅ 推理模式（extended thinking）
- ✅ 流式响应
- ✅ 批量并发（max_workers）
- ✅ Prompt 缓存

### 本地模型后端
- ❌ JSON Schema（需自行解析）
- ❌ 推理模式
- ✅ 流式响应
- ⚠️ 批量并发（受 GPU 显存限制）
- ✅ 完全离线

## 迁移清单

### 已完成
- [x] 抽象 TranslationBackend 基类
- [x] 拆分 OpenAI 兼容后端
- [x] 实现本地模型后端框架
- [x] 保持 translator.py API 兼容
- [x] 单元测试覆盖

### 待完成（按需）
- [ ] vLLM 后端优化（批量推理）
- [ ] 混元模型专用适配器
- [ ] SGLang 后端支持
- [ ] 本地模型 JSON Schema 解析增强
- [ ] 量化模型支持（GPTQ/AWQ）

## 注意事项

1. **API 兼容性**：原有的 `translate_segments()` API 保持不变，现有代码无需修改
2. **缓存键**：不同后端使用不同缓存命名空间，避免混用
3. **Prompt 差异**：本地模型可能需要调整 System Prompt 格式
4. **性能权衡**：本地模型牺牲了部分并发能力，但节省 API 成本
5. **GPU 显存**：本地模型需要至少 24GB 显存运行 70B 级模型

## 相关文件

- `src/llm/translator_legacy.py` - 原始单文件实现（保留作为参考）
- `tests/test_translation_backends.py` - 后端测试套件
- `HISTORY.md` - 详细变更历史
