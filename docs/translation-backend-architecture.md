# 翻译后端架构

## 设计目标

翻译侧是三层结构：**transport（backends）只执行一次消息请求**；**profile 只拥有一个模型家族的 prompt 合同**（怎么构造消息、怎么把回复解析回逐 id 文本）；**engine 只有一份编排循环**（批规划、缓存/翻译记忆、缺失项重试阶梯、滚动历史调度、进度聚合、计时）。`translator.py` 是门面：装配 seam（chat 调用、退避、崩溃探针）、选择 profile、调用 engine，然后按 profile 声明执行可选阶段（全局术语预抽取、修复批）。

历史教训有两条，边界因此刻意如此划分：其一，2026-07 之前曾把批处理和修复拆成各自维护状态的副本，丢失过 Responses API、术语提取、取消语义和缓存统计——所以现在**编排循环全仓只有一份**（`engine.py`），repair/global_glossary 是无状态的阶段函数，通过参数拿到 chat 调用，不持有第二份调度状态。其二，transport 曾与 translator 循环导入——现在 `backends/`、`profiles/`、`engine.py`、`repair.py`、`global_glossary.py` 都禁止导入 `translator`，`tests/test_no_circular_imports.py` 用子进程锁死这条。

硬不变式：**cue plan 在翻译前冻结，翻译永远 1:1**——任何 profile 都不得合并或拆分行；`parse_response` 只能返回请求过的 id（`tests/test_profiles_contract.py` 钉住）。

```text
src/llm/
├── translator.py             # 门面：seam 装配、profile 选择、阶段编排
├── engine.py                 # 唯一编排循环（批/缓存/重试/历史/进度/计时）
├── settings.py               # 全部 TRANSLATION_* 常量与 env 解析
├── transport_util.py         # 取消/退避/重试事件/usage/流式进度等共享件
├── prompt.py                 # JSON 合同的 prompt 与字幕 JSON 构造（PROMPT_VERSION）
├── repair.py                 # 修复批（profile 经 wants_repair_pass 选入）
├── global_glossary.py        # 全片术语预抽取(profile 经 wants_extra_glossary 选入)
├── cache.py                  # batch cache 与 translation memory
├── glossary.py               # 用户词表解析
├── errors.py                 # 所有层共享的异常类型
├── profiles/
│   ├── base.py               # TranslationProfile ABC + ProfileContext
│   ├── json_v3.py            # 默认 JSON 批合同（解析/规范化/schema 都在这）
│   ├── sakura_galtransl.py   # Sakura/GalTransl 行式合同
│   └── __init__.py           # 注册表、pin 别名、按后端隔离的自动检测
└── backends/
    ├── __init__.py           # 注册表、选择和进程级实例生命周期
    ├── base.py               # 后端抽象基类
    ├── openai_compat.py      # Chat/Responses canonical transport（含重试/流式）
    ├── llamacpp_server.py    # 托管 llama-server 的 GGUF 后端（Sakura 推荐路径）
    └── local_model.py        # 进程内 Transformers 后端
```

## Profile 合同

一个 profile = 一个模型家族的 prompt 合同，实现 `profiles/base.py` 的 `TranslationProfile`：

- `id` / `version`：`cache_signature()` 返回 `id@version`，折进每个缓存/记忆 key。任何会让相同输入产生不同译文的改动都必须升 `version`。
- `build_messages(segments, ids, ctx)` / `parse_response(text, ids)`：消息构造与回复解析。`ids` 是全局字幕 id；解析必须返回 `{id: text}` 且只含请求过的 id，违反合同抛 `RetryableTranslationFormatError`。
- 调度声明：`needs_history`/`history_limit`（连续分片、片内串行，保证滚动历史真实）、`line_capable`（行数不匹配时允许逐行回退）、`supports_partial_reissue`（只补请缺失 id）、`schema`（结构化输出，传给报告 `supports_json_schema()` 的后端）、`sampling(batch_size)`（采样覆盖）。
- 阶段声明：`wants_repair_pass`、`wants_extra_glossary`。两个阶段都用 JSON 合同按全局 id 定位行，所以只有 JSON 系 profile 可以打开。

选择规则（`profiles.select_profile()`）：`TRANSLATION_PROMPT_PROFILE` 显式 pin 优先（别名：`sakura`→`sakura_galtransl`，`off`/`none`→`json`）；`auto` 下按**所选后端自己的模型配置**检测（llamacpp 读 `LLAMACPP_GGUF_PATH`/`MODEL_FILE`/`MODEL_REPO`，openai 只读 `LLM_MODEL_NAME`，其余后端必须显式 pin）——检测按后端隔离，内置的 Sakura 默认文件名不会在 openai 后端下误触发。都不命中回落 `json`。

新增一个微调模型的支持 = 写一个 profile 模块 + `register_profile(profile, match_tokens=...)`，不改 engine。

## Engine

`engine.py` 有两条循环，都由 profile 参数化：

- `run_batched`：JSON 系自由并行批。全片 JSON prefix（provider prompt-cache 友好；超限回落全片概览）、prefix warmup（仅 openai 后端且 >1 个待翻批次）、批级精确缓存 + 行级翻译记忆预填、缺失 id 部分补请（`supports_partial_reissue`）、`TRANSLATION_BATCH_MAX_REQUESTS` 硬顶、逐批诊断事件与计时。单请求小任务与多批大任务走同一条路径（旧的 single-request 专用路径已删除）。
- `run_line_profile`：行式历史合同（Sakura/GalTransl）。连续分片并行、分片内串行滚动 `历史翻译`，行数不匹配整批重试→逐行回退，跳过术语预抽取与修复批（它们说的是 JSON 合同）。

engine 不做 transport 决策：`chat` / `backoff_sleep` / `crash_probe` 由 translator 注入，所以测试在 translator 上的 `_chat`/`_chat_with_reasoning` seam 对 engine 驱动的请求仍然生效。

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

Chat 和 Responses 的流式进度、usage、JSON Schema 与 DeepSeek `json_object` 兼容只有一份 canonical 实现，避免适配器与主流程分叉。协议就是标准 OpenAI Chat/Responses，不带任何 provider 私有 patch；网关不兼容标准协议时应换网关而不是在这里加分支。

生产环境运行本地大模型时，推荐启动 vLLM/SGLang 等 OpenAI 兼容服务，然后仍选择 `openai` 后端。这样服务负责 continuous batching、KV cache 和多卡调度，本程序不会与 ASR 在同一进程争抢模型生命周期。

## 托管 llama-server GGUF 后端（本地推荐）

```env
TRANSLATION_BACKEND=llamacpp
LLAMACPP_SERVER_PATH=            # 留空则取 PATH 中的 llama-server（winget install llama.cpp）
LLAMACPP_MODEL_REPO=SakuraLLM/Sakura-GalTransl-7B-v3.7
LLAMACPP_MODEL_FILE=Sakura-Galtransl-7B-v3.7.gguf
LLAMACPP_GGUF_PATH=              # 本地 GGUF 路径，填了则优先于 repo+file 下载
LLAMACPP_CTX_SIZE=8192           # 每个并发槽的上下文；服务端总上下文 = CTX * PARALLEL
LLAMACPP_N_GPU_LAYERS=999
LLAMACPP_PARALLEL=4
LLAMACPP_STARTUP_TIMEOUT_S=300
```

选它而不是 llama-cpp-python 的原因：项目在 Python 3.14，llama-cpp-python 官方 CUDA
预编译轮只覆盖到 3.12，而 llama.cpp 每个 release 都带官方 Windows CUDA 的
`llama-server`。后端只是一个进程管理器 + 指向 127.0.0.1 的 OpenAI 客户端：
自动选端口、健康轮询、日志落到 `tmp/log/llamacpp_server.log`、Windows Job Object
保证宿主异常退出时服务进程一并被回收。

**显存交接**：服务启动前先关停常驻的 ASR GPU worker（下个 ASR 阶段自动重载），
单个视频的翻译阶段结束后由主流程 `reset_backend("llamacpp")` 关停服务——8G 卡上
6.3GB 的 GGUF 与 ASR 模型无法共存，交接必须是显式的。

**Sakura prompt profile**（`llm/profiles/sakura_galtransl.py`）：`TRANSLATION_PROMPT_PROFILE=auto`
下，当且仅当所选后端自己的模型配置命中 Sakura/GalTransl 时启用（检测按后端隔离，
内置的 Sakura 默认文件名不会在 openai 后端下误触发）。启用后整条 JSON 批合同被
替换为模型卡的行式模板：术语表 `src->dst`、`历史翻译` 滚动上文、N 行进 N 行出，
采样固定 temperature 0.3 / top_p 0.8。engine 的行式循环按连续分片并行
（`SAKURA_WORKERS`），分片内串行以保证历史上文真实；行数不匹配先整批重试一次、
再逐行回退；JSON 修复批与术语预抽取被跳过。缓存签名为 `sakura_galtransl@v3`，
与 JSON 合同互不复用。非 Sakura 的 GGUF 模型（如 Qwen3 量化）不受影响，仍走标准
JSON 批合同。

Sakura/GalTransl 系模型为 CC-BY-NC-SA 4.0 许可，禁止商用。

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

缓存签名包含 profile 签名（`id@version`，JSON 合同当前为 `json@v3.0`）、目标语言、词表、人物参考和后端 `cache_identity()`。API 模型与本地模型不会复用同一翻译缓存；不同 profile 之间也不会。

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
