"""Shared wire contract for model-generated audio timestamps.

Teacher responses use an unambiguous ``MM:SS.mmm`` string on the wire.  Local
artifacts may keep numeric seconds after this module has validated and parsed
the response.  Numeric teacher timestamps are deliberately unsupported: a
value such as ``105.153`` cannot tell us whether the model meant 105.153
seconds or ``01:05.153``.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_UP
import math
import re
from typing import Any, Mapping


TIMESTAMP_CONTRACT_ID = "omni_audio_timestamp_mmss_mmm_v1"
TIMESTAMP_FORMAT = "MM:SS.mmm"
_TIMESTAMP_RE = re.compile(
    r"^(?P<minutes>[0-9]{2,}):(?P<seconds>[0-5][0-9])\.(?P<milliseconds>[0-9]{3})$"
)
_MILLISECONDS_PER_SECOND = Decimal(1000)
_SECONDS_PER_MINUTE = 60


# Nothing injects this: teacher prompts come from `--prompt/--prompt-file`, so
# this is the canonical text to paste into one. It is kept next to the parser
# that enforces it, because a prompt that states a different format than
# `parse_mmss_timestamp` accepts fails at parse time, after the run is paid for.
TIMESTAMP_PROMPT_CONTRACT_ZH = """时间坐标合同（适用于本请求中的所有区间）：
- 所有区间边界必须使用 JSON 字符串字段 `start_ts` / `end_ts`，格式严格为 `MM:SS.mmm`。
- 单个候选点使用 JSON 字符串字段 `time_ts`，格式同样严格为 `MM:SS.mmm`。
- `MM` 至少两位，`SS` 必须为 00–59，毫秒必须恰好三位；例如 `00:05.153`、`01:05.153`。
- `01:05.153` 表示 65.153 秒。绝对不要删除冒号写成 `105.153`，也不要输出数字秒 `start_s` / `end_s`。
- 坐标使用当前上传音频的 0-based 局部时间轴，不使用原视频时间轴；区间不得超出请求给出的 `duration_ts`。
- 无坐标的状态使用 null；不得用 `00:00.000` 代替 null。
"""


def _finite_nonnegative_decimal(value: float | int | Decimal, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite non-negative number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite non-negative number") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _format_decimal_timestamp(value: Decimal, *, rounding: str) -> str:
    rounding_mode = {
        "nearest": ROUND_HALF_UP,
        "floor": ROUND_FLOOR,
    }.get(rounding)
    if rounding_mode is None:
        raise ValueError("timestamp rounding must be 'nearest' or 'floor'")
    total_milliseconds = int(
        (value * _MILLISECONDS_PER_SECOND).quantize(
            Decimal("1"),
            rounding=rounding_mode,
        )
    )
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    minutes, seconds = divmod(total_seconds, _SECONDS_PER_MINUTE)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_mmss_timestamp(
    seconds: float | int | Decimal,
    *,
    rounding: str = "nearest",
) -> str:
    """Format non-negative seconds as an exact ``MM:SS.mmm`` string."""

    return _format_decimal_timestamp(
        _finite_nonnegative_decimal(seconds, field="timestamp seconds"),
        rounding=rounding,
    )


def format_duration_timestamp(seconds: float | int | Decimal) -> str:
    """Advertise a source duration without rounding beyond available audio."""

    return format_mmss_timestamp(seconds, rounding="floor")


def parse_mmss_timestamp(
    value: Any,
    *,
    field: str = "timestamp",
    duration_s: float | int | Decimal | None = None,
) -> float:
    """Strictly parse a teacher timestamp and optionally enforce source bounds."""

    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a {TIMESTAMP_FORMAT} string; numeric seconds are rejected"
        )
    match = _TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"{field} must exactly match {TIMESTAMP_FORMAT}; received {value!r}"
        )
    total_milliseconds = (
        (
            int(match.group("minutes")) * _SECONDS_PER_MINUTE
            + int(match.group("seconds"))
        )
        * 1000
        + int(match.group("milliseconds"))
    )
    seconds = total_milliseconds / 1000.0
    if not math.isfinite(seconds):
        raise ValueError(f"{field} is not finite")
    if duration_s is not None:
        duration = _finite_nonnegative_decimal(duration_s, field="source duration")
        if Decimal(total_milliseconds) > duration * _MILLISECONDS_PER_SECOND:
            raise ValueError(
                f"{field}={value} ({seconds:.3f}s) exceeds source duration "
                f"{format_duration_timestamp(duration)}"
            )
    return seconds


def parse_mmss_span(
    row: Mapping[str, Any],
    *,
    field: str,
    duration_s: float | int | Decimal,
    allow_null: bool = False,
    start_key: str = "start_ts",
    end_key: str = "end_ts",
) -> tuple[float | None, float | None]:
    """Parse and validate one ordered teacher span from its wire fields."""

    if start_key not in row or end_key not in row:
        raise ValueError(
            f"{field} must contain {start_key}/{end_key} using {TIMESTAMP_FORMAT}"
        )
    start_value = row[start_key]
    end_value = row[end_key]
    if start_value is None or end_value is None:
        if allow_null and start_value is None and end_value is None:
            return None, None
        raise ValueError(
            f"{field} must use either two {TIMESTAMP_FORMAT} strings or two null values"
        )
    start = parse_mmss_timestamp(
        start_value,
        field=f"{field}.{start_key}",
        duration_s=duration_s,
    )
    end = parse_mmss_timestamp(
        end_value,
        field=f"{field}.{end_key}",
        duration_s=duration_s,
    )
    if end <= start:
        raise ValueError(
            f"{field} must satisfy start < end; received {start_value}..{end_value}"
        )
    return start, end


def timestamp_request_contract(duration_s: float | int | Decimal) -> dict[str, str]:
    """Return the common request metadata consumed by interval teachers."""

    return {
        "duration_ts": format_duration_timestamp(duration_s),
        "timestamp_format": TIMESTAMP_FORMAT,
        "timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
        "coordinate_system": "0-based current uploaded-audio timeline",
    }
