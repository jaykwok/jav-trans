"""What a failed job says in the task bar.

Each case here is a failure a first-time user actually hits, and the assertion is
that the message names the missing thing plus where to supply it - not that it
matches some exact wording.
"""

from __future__ import annotations

import subprocess

import pytest

from core import stage_errors
from core.stage_errors import describe_stage_failure


class _HttpError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ResponseHolder(RuntimeError):
    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = self._Response(status_code)


def test_openai_sdk_missing_key_becomes_a_settings_instruction():
    exc = RuntimeError(
        "The api_key client option must be set either by passing api_key to the "
        "client or by setting the OPENAI_API_KEY environment variable"
    )
    message = describe_stage_failure(exc)
    assert message == stage_errors.MISSING_API_KEY
    assert "翻译设置" in message
    assert "API Key" in message


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, stage_errors.INVALID_API_KEY),
        (403, stage_errors.INVALID_API_KEY),
        (402, stage_errors.INSUFFICIENT_BALANCE),
        (404, stage_errors.MODEL_NOT_FOUND),
        (429, stage_errors.RATE_LIMITED),
    ],
)
def test_http_status_maps_to_the_setting_that_fixes_it(status, expected):
    message = describe_stage_failure(_HttpError("upstream said no", status))
    assert message.startswith(expected)
    # The provider's own text is kept as a parenthetical: it is the only clue
    # when a service reuses a status for something else.
    assert "upstream said no" in message


def test_openrouter_has_no_provider_for_the_strict_schema_is_not_a_missing_model():
    """Measured against OpenRouter 2026-08-24: `provider.require_parameters`
    turns a model whose upstreams cannot do strict structured output into a 404.
    The generic 404 line would send the user off to pick another model without
    saying why the current one failed, and hides the one-line escape hatch."""
    exc = _HttpError(
        "No endpoints found that can handle the requested parameters.", 404
    )
    message = describe_stage_failure(exc)
    assert message.startswith(stage_errors.NO_ROUTE_FOR_STRICT_JSON)
    assert "LLM_STRUCTURED_OUTPUT" in message
    assert "json_object" in message
    assert "config/config.json" in message
    assert not message.startswith(stage_errors.MODEL_NOT_FOUND)


def test_openrouter_guardrail_block_is_not_a_missing_model():
    """Measured 2026-09-01 against a real OpenRouter account: the account's
    default data-policy settings rejected every provider for an NSFW prompt,
    returning this 404 before the request ever reached an upstream. The
    generic 404 line sent the user re-picking a model that was never the
    problem; the real fix is the account's own privacy settings page."""
    exc = _HttpError(
        "No endpoints available matching your guardrail restrictions and "
        "data policy. Configure: https://openrouter.ai/settings/privacy",
        404,
    )
    message = describe_stage_failure(exc)
    assert message.startswith(stage_errors.OPENROUTER_GUARDRAIL_BLOCKED)
    assert "openrouter.ai/settings/privacy" in message
    assert not message.startswith(stage_errors.MODEL_NOT_FOUND)


def test_status_code_is_also_read_from_a_wrapped_response():
    message = describe_stage_failure(_ResponseHolder("nope", 401))
    assert message.startswith(stage_errors.INVALID_API_KEY)


def test_connection_failures_cover_both_stages_that_reach_the_network():
    class APIConnectionError(RuntimeError):
        pass

    message = describe_stage_failure(APIConnectionError("getaddrinfo failed"))
    assert message.startswith(stage_errors.CANNOT_REACH_SERVICE)
    assert "getaddrinfo failed" in message
    # A model download fails the same way, so neither remedy may be assumed.
    assert "代理" in message and "API Base URL" in message


def test_missing_ffmpeg_is_distinguished_from_a_failed_extraction():
    missing = FileNotFoundError(2, "系统找不到指定的文件。", "ffmpeg")
    assert describe_stage_failure(missing) == stage_errors.FFMPEG_MISSING

    failed = subprocess.CalledProcessError(
        1,
        ["ffmpeg", "-i", "a.mp4", "out.wav"],
        stderr="a.mp4: Invalid data found when processing input\n",
    )
    message = describe_stage_failure(failed)
    assert message.startswith(stage_errors.FFMPEG_EXTRACT_FAILED)
    assert "退出码 1" in message
    assert "Invalid data found when processing input" in message


def test_a_missing_file_that_is_not_ffmpeg_is_left_alone():
    exc = FileNotFoundError(2, "系统找不到指定的文件。", "D:/videos/gone.mp4")
    assert describe_stage_failure(exc) == str(exc)


def test_cuda_and_oom_get_their_own_advice():
    cuda = RuntimeError(
        "subtitle_timing requires CUDA for runtime inference; CPU fallback is disabled"
    )
    cuda_msg = describe_stage_failure(cuda)
    assert cuda_msg.startswith(stage_errors.CUDA_UNAVAILABLE)
    assert "requires CUDA" in cuda_msg

    oom = RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
    oom_msg = describe_stage_failure(oom)
    assert oom_msg.startswith(stage_errors.OUT_OF_MEMORY)
    assert "512.00 MiB" in oom_msg


def test_connection_error_preserves_underlying_cause_chain():
    class APIConnectionError(RuntimeError):
        pass

    class ReadError(RuntimeError):
        pass

    root = TimeoutError("timed out reading response body")
    wrapped = APIConnectionError("Connection error.")
    wrapped.__cause__ = root
    message = describe_stage_failure(wrapped)
    assert message.startswith(stage_errors.CANNOT_REACH_SERVICE)
    assert "timed out reading response body" in message

    read_msg = describe_stage_failure(ReadError("peer reset connection"))
    assert read_msg.startswith(stage_errors.CANNOT_REACH_SERVICE)
    assert "peer reset connection" in read_msg


def test_local_server_crashed_error_is_not_misreported_as_remote_network():
    from llm.errors import LocalServerCrashedError

    exc = LocalServerCrashedError(
        "本地翻译服务 llama-server 已异常退出（PID=1234，退出码 -1073741819 (0xC0000005 STATUS_ACCESS_VIOLATION)）；"
        "请检查显存占用、显卡驱动或 GGUF 模型文件，完整日志见 logs/llamacpp_server.log。"
    )
    message = describe_stage_failure(exc)
    assert message == str(exc)
    assert not message.startswith(stage_errors.CANNOT_REACH_SERVICE)


def test_a_message_we_already_wrote_is_not_rewritten():
    # llamacpp/local raise their own actionable Chinese; a generic remap would
    # replace a specific instruction with a vaguer one.
    original = "LLAMACPP_GGUF_PATH 指向的 GGUF 文件不存在：D:/a.gguf，请填写正确路径。"
    assert describe_stage_failure(RuntimeError(original)) == original


def test_a_providers_own_chinese_rate_limit_still_maps_by_status():
    # "请求过于频繁" contains 请 but is not an instruction; the 429 rule must win.
    exc = _HttpError("请求过于频繁，请求已被限流", 429)
    assert describe_stage_failure(exc).startswith(stage_errors.RATE_LIMITED)


def test_detail_attribute_wins_and_empty_errors_still_say_something():
    class DetailError(RuntimeError):
        detail = "缺翻译 API Key"

        def __str__(self) -> str:
            return "internal wrapper"

    assert describe_stage_failure(DetailError()) == "缺翻译 API Key"
    assert describe_stage_failure(RuntimeError()) == "RuntimeError"


def test_a_moved_video_is_reported_before_ffmpeg_can_blame_the_file(tmp_path):
    """The stage checks the path itself: ffmpeg on a missing file exits non-zero,
    which would otherwise be described as a damaged or silent video."""
    import main as pipeline_main
    from core.job_context import JobContext

    missing = tmp_path / "gone.mp4"
    ctx = JobContext.from_spec(
        type("Spec", (), {"video_paths": [str(missing)]})(),
        "job",
        str(tmp_path),
        str(tmp_path / "cache.jsonl"),
    )
    with pytest.raises(RuntimeError) as excinfo:
        pipeline_main._run_asr_alignment_impl(str(missing), ctx=ctx)
    assert stage_errors.VIDEO_FILE_MISSING in str(excinfo.value)
    assert str(missing) in str(excinfo.value)


def test_huggingface_http_errors_map_to_model_download_advice_not_translation_api():
    class HfHubHTTPError(RuntimeError):
        def __init__(self, message: str, status_code: int) -> None:
            super().__init__(message)
            self.response = _ResponseHolder._Response(status_code)

    msg_401 = describe_stage_failure(
        HfHubHTTPError("401 Client Error for https://huggingface.co/private/repo", 401)
    )
    assert msg_401.startswith(stage_errors.HF_AUTH_REQUIRED)
    assert not msg_401.startswith(stage_errors.INVALID_API_KEY)

    msg_404 = describe_stage_failure(
        _HttpError("404 Client Error: Not Found for url: https://hf-mirror.com/foo/bar", 404)
    )
    assert msg_404.startswith(stage_errors.HF_REPO_NOT_FOUND)
    assert not msg_404.startswith(stage_errors.MODEL_NOT_FOUND)

    msg_429 = describe_stage_failure(
        HfHubHTTPError("429 Too Many Requests for https://huggingface.co/x", 429)
    )
    assert msg_429.startswith(stage_errors.HF_RATE_LIMITED)
    assert not msg_429.startswith(stage_errors.RATE_LIMITED)


def test_host_ram_oom_is_distinguished_from_gpu_vram_oom():
    ram_exc = RuntimeError("Physical RAM budget exceeded: 31.2 GB / 32.0 GB")
    ram_msg = describe_stage_failure(ram_exc)
    assert ram_msg.startswith(stage_errors.HOST_RAM_OUT_OF_MEMORY)
    assert not ram_msg.startswith(stage_errors.OUT_OF_MEMORY)

    mem_msg = describe_stage_failure(MemoryError("unable to allocate array"))
    assert mem_msg.startswith(stage_errors.HOST_RAM_OUT_OF_MEMORY)


def test_describe_stage_failure_redacts_url_credentials_everywhere():
    exc = RuntimeError("proxy socks5://alice:s3cr@t@127.0.0.1:1080 failed")
    msg = describe_stage_failure(exc)
    assert "alice" not in msg
    assert "s3cr@t" not in msg
    assert "socks5://***@127.0.0.1:1080" in msg


def test_english_detail_continues_rule_matching_while_chinese_detail_short_circuits():
    class _WorkerErr(RuntimeError):
        def __init__(self, detail: str) -> None:
            super().__init__(f"worker: {detail}")
            self.detail = detail

    oom_msg = describe_stage_failure(_WorkerErr("CUDA out of memory. Tried to allocate 256 MiB"))
    assert oom_msg.startswith(stage_errors.OUT_OF_MEMORY)

    ram_msg = describe_stage_failure(_WorkerErr("Physical RAM budget exceeded: 96%"))
    assert ram_msg.startswith(stage_errors.HOST_RAM_OUT_OF_MEMORY)

    # Dead constant removed.
    assert not hasattr(stage_errors, "TRANSLATION_PANEL")
