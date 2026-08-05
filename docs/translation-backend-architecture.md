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
│   ├── json_v3.py            # JSON 批合同（解析/规范化/schema 都在这）
│   ├── hymt2.py              # 本地逐句合同
│   └── __init__.py           # 注册表、pin 别名、按后端隔离的自动检测
└── backends/
    ├── __init__.py           # 注册表、选择和进程级实例生命周期
    ├── base.py               # 后端抽象基类
    ├── openai_compat.py      # Chat/Responses canonical transport（含重试/流式）
    └── llamacpp_server.py    # 托管 llama-server 的 GGUF 后端（本地路径）
```

## Profile 合同

一个 profile = 一个模型家族的 prompt 合同，实现 `profiles/base.py` 的 `TranslationProfile`：

- `id` / `version`：`cache_signature()` 返回 `id@version`，折进每个缓存/记忆 key。任何会让相同输入产生不同译文的改动都必须升 `version`。
- `build_messages(segments, ids, ctx)` / `parse_response(text, ids)`：消息构造与回复解析。`ids` 是全局字幕 id；解析必须返回 `{id: text}` 且只含请求过的 id，违反合同抛 `RetryableTranslationFormatError`。
- 调度声明：`supports_partial_reissue`（只补请缺失 id）、`schema`（结构化输出，传给报告 `supports_json_schema()` 的后端）、`max_batch_size()`（每请求条数硬顶，逐句合同返回 1）、`response_token_budget(segments)`（按源文长度派生的生成上界）、`bounded_schema(segments)`（把 schema 收窄到这批片段能合法产出的形状，例如给 `text` 加 `maxLength`）。
- 阶段声明：`wants_repair_pass`、`wants_extra_glossary`。两个阶段都用 JSON 合同按全局 id 定位行，所以只有 JSON 系 profile 可以打开。

选择规则（`profiles.select_profile()`）：`TRANSLATION_PROMPT_PROFILE` 显式 pin 优先（别名：`off`/`none`→`json`）；`auto` 下按**所选后端自己的模型配置**检测（llamacpp 读 `LLAMACPP_GGUF_PATH`/`MODEL_FILE`/`MODEL_REPO`，openai 只读 `LLM_MODEL_NAME`，其余后端必须显式 pin）——检测按后端隔离，一个后端的模型名不会污染另一个后端的判断。都不命中回落 `json`。

**注册了两个 profile，按部署形态分**（见上表）：`json` 是默认，`hymt2` 由本地 GGUF 的文件名/仓库名命中。行式的 Sakura/GalTransl 合同已于 2026-08-04 删除（`agents/rm/20260804_141838_sakura-branch-removed/`）；进程内 Transformers 后端于 2026-08-05 删除（`agents/rm/20260805_183000_local-transformers-backend-dropped/`），它是第三种跑本地模型的方式，而没有任何随附模型以它为目标。

新增一个微调模型的支持 = 写一个 profile 模块 + `register_profile(profile, match_tokens=...)`，不改 engine。

## Engine

`engine.py` 只有一条循环，由 profile 参数化：

- `run_batched`：JSON 系自由并行批。全片 JSON prefix（provider prompt-cache 友好；超限回落全片概览）、prefix warmup（仅 openai 后端且 >1 个待翻批次）、批级精确缓存 + 行级翻译记忆预填、缺失 id 部分补请（`supports_partial_reissue`）、`TRANSLATION_BATCH_MAX_REQUESTS` 硬顶、逐批诊断事件与计时。单请求小任务与多批大任务走同一条路径（旧的 single-request 专用路径已删除）。

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
LLAMACPP_SERVER_PATH=            # 留空则取 PATH 中的 llama-server（winget install -e --id ggml.llamacpp，装的是 Vulkan 构建）
LLAMACPP_MODEL_REPO=tencent/Hy-MT2-1.8B-GGUF
LLAMACPP_MODEL_FILE=Hy-MT2-1.8B-Q8_0.gguf
LLAMACPP_GGUF_PATH=              # 本地 GGUF 路径，填了则优先于 repo+file 下载
LLAMACPP_CTX_SIZE=8192           # 每个并发槽的上下文；服务端总上下文 = CTX * PARALLEL
LLAMACPP_N_GPU_LAYERS=999
LLAMACPP_PARALLEL=8              # 1.8B 默认模型 ~2GB，8G 卡可开 8 槽；换大模型要调小
LLAMACPP_STARTUP_TIMEOUT_S=300
```

选它而不是 llama-cpp-python 的原因：项目在 Python 3.14，llama-cpp-python 官方 CUDA
预编译轮只覆盖到 3.12，而 llama.cpp 每个 release 都带官方 Windows CUDA 的
`llama-server`。后端只是一个进程管理器 + 指向 127.0.0.1 的 OpenAI 客户端：
自动选端口、健康轮询、日志落到 `tmp/log/llamacpp_server.log`、Windows Job Object
保证宿主异常退出时服务进程一并被回收。

**显存交接**：服务启动前先关停常驻的 ASR GPU worker（下个 ASR 阶段自动重载），
单个视频的翻译阶段结束后由主流程 `reset_backend("llamacpp")` 关停服务——8G 卡上
5GB 级的 GGUF 与 ASR 模型无法共存，交接必须是显式的。

**两套 prompt 合同，按部署形态分而不是按厂商分**（`src/llm/profiles/`）：

| profile | 用途 | 每请求条数 | schema | 上下文 |
|---|---|---|---|---|
| `json` | OpenAI 兼容 API | 至多 64 | JSON | 全片前缀 + 角色表 + 术语表 |
| `hymt2` | 本地 llama.cpp 默认 | 1 | 无 | 无 |

`hymt2` 逐句不是口味问题。同 300 条真实台词实测未翻译率：裸模板 6、加系统提示 26、
加术语表/角色块 30、加邻句背景 60、JSON 批量 152（另有 88 行原样回吐）——**每加一层
上下文退化一档**。而本地那侧本来也买不到上下文：全片 JSON 前缀塞不进 8G 卡上 8192 的
槽，那层是结构性不可用的。所以本地路径上**术语表、角色参考、全片上下文一律不生效**，
运行时会打印一行说明，而不是收下再忽略。

profile 由 `select_profile()` 按配置字符串自动选择（只读配置，绝不加载后端）：
llamacpp 侧看 GGUF 文件名/仓库名，openai 侧看 `LLM_MODEL_NAME`。`TRANSLATION_PROMPT_PROFILE`
可以钉死。自定义 GGUF 若不是 Hunyuan-MT 系，会被当成能按 JSON 批量回答的模型处理；
只认自家行式模板的 Sakura/GalTransl 系仍不支持，`llm/preflight.py` 在建任务时就拦下。

**两套合同共有的硬约束：空译文一律报错，绝不能当成翻译返回。** 这正是 2026-08-04
删掉 Sakura 支路的唯一理由——它把没对上的行写成 `""` 并成功返回，是本流水线唯一
能在不让任务失败的情况下把未翻译字幕送上屏幕的路径。`hymt2` 是行式合同，但这条缺陷
没有跟着回来，contract 测试单独钉住。

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
