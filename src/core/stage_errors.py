"""Turn a stage failure into a line a first-time user can act on.

The task bar shows one line per failed job, and that line used to be whatever
``str(exc)`` happened to be: ``The api_key client option must be set...`` for a
forgotten API key, ``[WinError 2] 系统找不到指定的文件。`` for a missing ffmpeg.
Both describe the symptom in the vocabulary of the library that raised, not the
setting the user has to change.

Every rule here names the missing item *and* where to supply it. Recognition is
by duck-typing (``status_code``, exception class name, message substrings) so
this module stays importable from ``core`` without dragging in openai or torch.
Anything unrecognised falls through unchanged - a wrong guess would be worse
than a raw message.
"""

from __future__ import annotations

import re

TRANSLATION_PANEL = "「翻译设置」"

MISSING_API_KEY = (
    "缺少翻译 API Key：请打开「翻译设置」，在「API Key」中填写密钥并保存后重试；"
    "只需要日文字幕可以打开「不翻译（仅日文字幕）」。"
)
MISSING_BASE_URL = (
    "缺少 API Base URL：请在「翻译设置」的「API Base URL」中填写服务地址"
    "（例如 OpenRouter 的 https://openrouter.ai/api/v1，或 DeepSeek 的 "
    "https://api.deepseek.com）并保存后重试。"
)
MISSING_MODEL = (
    "未选择翻译模型：请在「翻译设置」中填好 API Key 与 API Base URL，"
    "点「获取」拉取模型列表后选择一个模型并保存。"
)
INVALID_API_KEY = (
    "翻译 API Key 被服务端拒绝（401/403）：请在「翻译设置」中确认 API Key 没有写错、"
    "并且与「API Base URL」指向的服务商一致。"
)
INSUFFICIENT_BALANCE = (
    "翻译服务返回余额不足（402）：请为该 API Key 充值，或在「翻译设置」中换一个可用的 Key。"
)
MODEL_NOT_FOUND = (
    "翻译服务不认识当前模型（404）：请在「翻译设置」中重新点「获取」，"
    "选择一个该服务商当前可用的模型并保存。"
)
# Also a 404, and the generic one above would send the user hunting for a model
# that is in fact available: OpenRouter returns this when no upstream endpoint
# for the chosen model accepts the strict JSON schema. Two real remedies, so it
# names both rather than picking one.
NO_ROUTE_FOR_STRICT_JSON = (
    "当前模型在 OpenRouter 上没有支持严格 JSON Schema 的供应商（404）：请在「翻译设置」中"
    "换一个支持结构化输出的模型（模型列表里 `supported_parameters` 含 `structured_outputs` 的那些），"
    "或在 .env 中设 LLM_STRUCTURED_OUTPUT=json_object 改用宽松的 JSON 约束后重试。"
)
# Also a 404, and also not a missing/renamed model: OpenRouter's account-level
# privacy settings (data policy / which providers may see the prompt) reject
# the request before it reaches any upstream. Measured 2026-09-01 against a
# real account whose default policy blocked every provider for an NSFW prompt -
# the generic 404 line sent the user re-picking a model that was never the
# problem.
OPENROUTER_GUARDRAIL_BLOCKED = (
    "OpenRouter 因隐私/数据策略拦截了这次请求（404），不是模型选错了：请打开 "
    "https://openrouter.ai/settings/privacy 调整数据策略（例如允许存储训练数据的供应商），"
    "或在「翻译设置」中换一个不受该策略限制的模型/服务商。"
)
RATE_LIMITED = (
    "翻译服务限速（429）且重试已用尽：请把「并行翻译 Worker 数」调低后重试。"
)
# Reached from two very different stages (model download, translation request),
# so it names both remedies instead of guessing which one applies.
CANNOT_REACH_SERVICE = (
    "网络连接失败：请检查网络；下载模型可以在「网络代理」中配置代理，"
    "调用翻译 API 则请再确认「翻译设置」里的「API Base URL」。"
)
VIDEO_FILE_MISSING = (
    "找不到视频文件：请确认文件还在原来的位置"
    "（移动、改名或所在磁盘未挂载时，需要重新添加任务）。"
)
FFMPEG_MISSING = (
    "未找到 ffmpeg：请确认压缩包已完整解压（程序目录下应有 bin\\ffmpeg.exe），"
    "或把 ffmpeg 所在目录加入系统 PATH 后重启应用。"
)
FFMPEG_EXTRACT_FAILED = (
    "ffmpeg 无法从这个视频里提取音频：请确认文件没有损坏、并且带有音轨"
    "（可以先用播放器确认能出声）。"
)
CUDA_UNAVAILABLE = (
    "显卡不可用：本项目需要 NVIDIA 显卡和可用的 CUDA，"
    "请更新显卡驱动后重启应用（也请确认没有在虚拟机/远程桌面里禁用了显卡）。"
)
OUT_OF_MEMORY = (
    "显存不足：请先关掉占用显存的程序（游戏、浏览器硬件加速等），"
    "或在「环境变量覆盖」里填 ASR_BATCH_SIZE=2 降低批大小后重试。"
)

# A message we wrote ourselves already tells the user what to do; re-mapping it
# would only overwrite a more specific instruction with a generic one. Matching
# the instruction verbs rather than a bare "请" keeps a provider's own Chinese
# error ("请求过于频繁") from looking like one of ours.
_ALREADY_ACTIONABLE = re.compile(
    "请(?:在|打开|填|先|为|更新|检查|确认|把|降低|选择|安装|关闭|设置|使用|改)"
)

_STATUS_MESSAGES = {
    401: INVALID_API_KEY,
    402: INSUFFICIENT_BALANCE,
    403: INVALID_API_KEY,
    404: MODEL_NOT_FOUND,
    429: RATE_LIMITED,
}

_CONNECTION_NAME_MARKERS = (
    "apiconnection",
    "connecterror",
    "connectionerror",
    "connecttimeout",
    "sslerror",
    "proxyerror",
)


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _mentions_ffmpeg(exc: BaseException, message: str) -> bool:
    haystack = [message.lower()]
    for attribute in ("filename", "filename2"):
        value = getattr(exc, attribute, None)
        if value:
            haystack.append(str(value).lower())
    command = getattr(exc, "cmd", None)
    if command:
        haystack.append(str(command).lower())
    return any("ffmpeg" in item or "ffprobe" in item for item in haystack)


def describe_stage_failure(exc: BaseException) -> str:
    """Best-effort actionable message for a job that failed in some stage."""
    detail = getattr(exc, "detail", None)
    if isinstance(detail, str) and detail.strip():
        return detail.strip()

    message = str(exc).strip()
    lowered = message.lower()
    name = type(exc).__name__.lower()

    if _ALREADY_ACTIONABLE.search(message):
        return message

    # The openai SDK raises this from the client constructor, before any request.
    if "api_key client option must be set" in lowered:
        return MISSING_API_KEY

    if _mentions_ffmpeg(exc, message):
        if isinstance(exc, FileNotFoundError) or "winerror 2" in lowered:
            return FFMPEG_MISSING
        if name == "calledprocesserror" or "non-zero exit status" in lowered:
            return FFMPEG_EXTRACT_FAILED

    status = _status_code(exc)
    if status == 404 and "no endpoints found that can handle" in lowered:
        return f"{NO_ROUTE_FOR_STRICT_JSON}（服务端原文：{message}）"
    if status == 404 and ("guardrail" in lowered or "data policy" in lowered):
        return f"{OPENROUTER_GUARDRAIL_BLOCKED}（服务端原文：{message}）"
    if status in _STATUS_MESSAGES:
        return f"{_STATUS_MESSAGES[status]}（服务端原文：{message}）" if message else _STATUS_MESSAGES[status]

    if any(marker in name for marker in _CONNECTION_NAME_MARKERS):
        return f"{CANNOT_REACH_SERVICE}（原始错误：{message}）" if message else CANNOT_REACH_SERVICE

    if "cuda is unavailable" in lowered or "requires cuda" in lowered:
        return CUDA_UNAVAILABLE

    if "out of memory" in lowered or "outofmemory" in name:
        return OUT_OF_MEMORY

    return message or type(exc).__name__


__all__ = [
    "CANNOT_REACH_SERVICE",
    "CUDA_UNAVAILABLE",
    "FFMPEG_EXTRACT_FAILED",
    "FFMPEG_MISSING",
    "INSUFFICIENT_BALANCE",
    "INVALID_API_KEY",
    "MISSING_API_KEY",
    "MISSING_BASE_URL",
    "MISSING_MODEL",
    "MODEL_NOT_FOUND",
    "NO_ROUTE_FOR_STRICT_JSON",
    "OPENROUTER_GUARDRAIL_BLOCKED",
    "OUT_OF_MEMORY",
    "RATE_LIMITED",
    "TRANSLATION_PANEL",
    "VIDEO_FILE_MISSING",
    "describe_stage_failure",
]
