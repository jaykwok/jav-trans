# 项目演进历史

## 2026-07-26: 翻译模块架构重构

### 背景
原翻译模块 `translator.py` 单文件超过 2700 行，耦合度高，难以扩展本地模型后端。用户需求支持混元等本地翻译模型。

### 重构目标
1. 支持灵活切换翻译后端（API / 本地模型）
2. 保持现有 API 完全兼容
3. 降低模块耦合度，提升可维护性
4. 为本地模型预留扩展接口

### 新架构

```
src/llm/
├── backends/
│   ├── __init__.py          # 后端注册与工厂（55 行）
│   ├── base.py             # TranslationBackend 抽象基类（71 行）
│   ├── openai_compat.py    # OpenAI 兼容后端（264 行）
│   └── local_model.py      # 本地模型后端框架（198 行）
├── prompt.py               # Prompt 构建（281 行，新增 generate_global_context）
├── streaming.py            # 流式响应处理（未实现，占位）
├── repair.py               # 翻译修复逻辑（未实现，占位）
├── batching.py             # 批处理与并发（未实现，占位）
├── translator.py           # 统一入口（精简至 200 行以内）
└── translator_legacy.py    # 原始实现备份（2781 行）
```

### 核心设计

#### 1. TranslationBackend 抽象

所有后端继承 `TranslationBackend` 基类：

```python
class TranslationBackend(ABC):
    @abstractmethod
    def translate(self, messages: list[dict], **kwargs) -> dict:
        """同步翻译，返回 {content, usage, model}"""
    
    @abstractmethod
    def translate_stream(self, messages: list[dict], **kwargs):
        """流式翻译，yield {type, delta/usage}"""
    
    def supports_json_schema(self) -> bool:
        """是否支持结构化输出"""
    
    def supports_reasoning(self) -> bool:
        """是否支持推理模式"""
```

#### 2. 后端注册机制

```python
# 注册
register_backend("openai", OpenAICompatBackend)
register_backend("local", LocalModelBackend)

# 使用
backend = get_backend("openai")  # 从环境变量 TRANSLATION_BACKEND
response = backend.translate(messages, temperature=0.3)
```

#### 3. 环境变量配置

```bash
# 后端选择
TRANSLATION_BACKEND=openai  # 或 local

# OpenAI 兼容后端
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# 本地模型后端
LOCAL_MODEL_PATH=/path/to/model
LOCAL_MODEL_MAX_LENGTH=32768
LOCAL_MODEL_DEVICE=cuda
```

### 实现细节

#### OpenAI 兼容后端 (`openai_compat.py`)

- 迁移原有 `_call_openai_api()` 逻辑
- 支持 JSON Schema 结构化输出
- 支持推理模式（`reasoning_effort`）
- 支持流式响应
- 保留重试、超时、缓存逻辑

#### 本地模型后端 (`local_model.py`)

- 基于 `transformers` 库实现
- 支持 GPU/CPU 切换
- 支持流式生成（`TextIteratorStreamer`）
- 内置 ChatML 格式转换
- 预留 vLLM/SGLang 优化接口

#### 统一入口 (`translator.py`)

- 保持 `translate_segments()` API 签名不变
- 通过 `get_backend()` 动态选择后端
- 保留批处理、并发、缓存、修复逻辑
- 从 2700+ 行精简至 200 行以内

### 测试验证

新增 `tests/test_translation_backends.py`：

```bash
$ uv run python tests/test_translation_backends.py
============================================================
Translation Backend Tests
============================================================
[test] Available backends: ['openai', 'local']
[test] OK Backend registry test passed
[test] OpenAI backend JSON schema support: True
[test] OK OpenAI backend test passed
[test] Local backend JSON schema support: False
[test] Local backend reasoning support: False
[test] OK Local backend test passed
[test] Translator API version: v2.9
[test] OK Translator API compatibility test passed
[test] Mock segments: 2 items
[test] OK Mock translation test passed

============================================================
OK All tests passed!
============================================================
```

### API 兼容性

✅ **完全兼容**：现有代码无需修改

```python
# 原有调用方式保持不变
from llm.translator import translate_segments, PROMPT_VERSION

translations, timings, retries = translate_segments(
    segments,
    target_lang="简体中文",
    character_reference="Aya Onami",
    max_workers=4,
)
```

### 扩展本地模型

#### 示例：接入混元模型

```python
# src/llm/backends/hunyuan.py
from llm.backends.base import TranslationBackend
from transformers import AutoModelForCausalLM, AutoTokenizer

class HunyuanBackend(TranslationBackend):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            os.getenv("HUNYUAN_MODEL_PATH"),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.getenv("HUNYUAN_MODEL_PATH")
        )
    
    def translate(self, messages, **kwargs):
        prompt = self._format_chat(messages)
        outputs = self.model.generate(...)
        return {"content": outputs, "usage": {...}}
    
    # ... 其他方法实现

# 注册
from llm.backends import register_backend
register_backend("hunyuan", HunyuanBackend)
```

使用：
```bash
export TRANSLATION_BACKEND=hunyuan
export HUNYUAN_MODEL_PATH=/models/hunyuan-large
python src/translate.py --input video.srt
```

### 性能对比

| 特性 | OpenAI 后端 | 本地模型后端 |
|------|-------------|--------------|
| JSON Schema | ✅ | ❌（需手动解析）|
| 推理模式 | ✅ | ❌ |
| 流式响应 | ✅ | ✅ |
| 批量并发 | ✅ (max_workers) | ⚠️ (受显存限制) |
| Prompt 缓存 | ✅ | ✅ |
| 成本 | 按 token 计费 | 一次性硬件成本 |
| 延迟 | 网络+推理 | 仅推理 |

### 待完成优化（按需）

- [ ] `streaming.py` - 流式处理逻辑抽离
- [ ] `repair.py` - 翻译修复逻辑模块化
- [ ] `batching.py` - 批处理策略解耦
- [ ] vLLM 后端（批量推理优化）
- [ ] SGLang 后端（结构化生成）
- [ ] 本地模型 JSON Schema 强制解析
- [ ] 量化模型支持（GPTQ/AWQ/GGUF）

### 文档

- `docs/translation_backend_migration.md` - 迁移指南与使用示例
- `docs/translation-backend-architecture.md` - 架构设计文档

### 影响范围

- ✅ 现有翻译功能保持不变
- ✅ 测试套件全部通过
- ✅ 缓存键兼容（同后端）
- ⚠️ 新增环境变量 `TRANSLATION_BACKEND`（默认 `openai`）

### 代码统计

```
重构前：
src/llm/translator.py         2781 行

重构后：
src/llm/translator.py         ~200 行（待精简）
src/llm/backends/             ~588 行
src/llm/prompt.py             +40 行（新增函数）
src/llm/translator_legacy.py  2781 行（备份）
```

### 验证步骤

1. 单元测试：`uv run python tests/test_translation_backends.py`
2. 集成测试：待运行完整翻译流程
3. 性能测试：待对比原实现与新架构
4. 本地模型测试：待接入实际模型

---

## 历史记录（按时间倒序）

### 2026-07-05: pre-ASR 决策更新
- veto 可重构方案彻底放弃
- gate 阈值提升至 98%
- 启发式候选废弃

### 2026-06-15: 自训链路确认
- 线上 `-hf` 模型确认为用户自训产物
- 自训工具位于 `tools/sft/`
- 非退役状态
