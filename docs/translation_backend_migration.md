# 翻译后端迁移说明

本项目尚未发布，本次重构采用断兼容策略：不保证旧的私有函数、内部模块或第三方后端类继续可用。用户侧的 Web 工作流、`translate_segments()` 返回产物、OpenAI Chat/Responses、缓存、术语、进度和取消行为继续保留。

最新架构、配置和扩展方式统一见 [翻译后端架构](translation-backend-architecture.md)。

## 从 2026-07-26 初版后端重构迁移

- 只使用 `TRANSLATION_BACKEND`；删除文档中未实现的 `LLM_BACKEND_TYPE`。
- 本地模型变量统一为 `LOCAL_MODEL_*`；删除未接线的 `LLM_LOCAL_*`。
- 后端实现 `chat_completion()` 和 `cache_identity()`，不再使用文档中未实现的 `translate()` / `translate_stream()` 双接口。
- 后端由注册表进程级复用；不要假设 `get_backend()` 每次返回新实例。
- `batching.py`、`repair.py`、`translator_legacy.py` 已删除。它们曾是功能不完整的重复实现，不是扩展点。
- vLLM 不作为进程内模式。请启动 OpenAI 兼容服务并选择 `TRANSLATION_BACKEND=openai`。
- 本地 Transformers 后端同步生成并串行使用模型；需要并发吞吐时应使用外部推理服务。

## 自定义后端注意事项

自定义后端属于断兼容范围，必须重新基于 `BaseTranslationBackend` 实现，并从 `llm.errors` 导入共享异常。缓存身份改变时更新 `cache_identity()`；否则不同模型可能错误复用译文。
