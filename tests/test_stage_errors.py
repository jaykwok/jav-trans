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
    assert "LLM_STRUCTURED_OUTPUT=json_object" in message
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

    failed = subprocess.CalledProcessError(1, ["ffmpeg", "-i", "a.mp4", "out.wav"])
    assert describe_stage_failure(failed) == stage_errors.FFMPEG_EXTRACT_FAILED


def test_a_missing_file_that_is_not_ffmpeg_is_left_alone():
    exc = FileNotFoundError(2, "系统找不到指定的文件。", "D:/videos/gone.mp4")
    assert describe_stage_failure(exc) == str(exc)


def test_cuda_and_oom_get_their_own_advice():
    cuda = RuntimeError(
        "subtitle_timing requires CUDA for runtime inference; CPU fallback is disabled"
    )
    assert describe_stage_failure(cuda) == stage_errors.CUDA_UNAVAILABLE

    oom = RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
    assert describe_stage_failure(oom) == stage_errors.OUT_OF_MEMORY


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
