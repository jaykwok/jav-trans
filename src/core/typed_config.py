"""Reading configuration values with one set of rules.

Every module used to parse its own settings, and they disagreed. Some clamped a
value into range, some took whatever the string said - `SUBTITLE_LINGER_S=-5`
and `SUBTITLE_READING_CPS=nan` were both accepted, and a NaN propagates through
every duration computed from it without ever raising. Some raised at *import*
time, so a typo in `.env` surfaced as a traceback out of a module three layers
below the thing the user was doing, naming neither the setting nor the file.

So there is one place that turns a string into a number, and it answers three
questions the same way everywhere:

* **Unset or empty is not an error.** It means "not configured", which is how
  most of `.env` is written; the default applies silently.
* **Unparseable is an error, and the default applies.** Refusing to start would
  make one bad character in `.env` fatal to a desktop tool whose user cannot see
  a traceback; running with a value nobody chose would be worse still, so the
  default is used *and the problem is recorded* under the field's own name.
* **Out of range follows the caller's declared policy, and is recorded.** Caps
  clamp to the nearest boundary; rates fall back to a meaningful default.

Recorded problems accumulate per field and are read back by
`configuration_problems()` - startup prints problems from its common-setting
scan and values already read. Lazy stages warn on their first read. A field that
later parses cleanly clears its own problem, because these values are re-read at
call time: fixing the settings panel has to be enough to make the warning go.

What this module deliberately does *not* do is cache. Values are read from the
environment on every call, because the settings page edits `os.environ` while
the process runs and a job started afterwards must see the new value.

Domain-specific contracts remain strict at their own boundaries, including
version stamps in `SubtitleOptions.__post_init__` and ASR pause-reading modes.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)

_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class ConfigProblem:
    field: str
    value: str
    problem: str

    def message(self) -> str:
        return f"{self.field}={self.value!r}：{self.problem}"


_problems: dict[str, ConfigProblem] = {}
_problem_lock = threading.Lock()


def _record(field: str, value: str, problem: str) -> None:
    entry = ConfigProblem(field=field, value=value, problem=problem)
    with _problem_lock:
        previous = _problems.get(field)
        _problems[field] = entry
    if previous != entry:
        # Once per distinct problem: these are read on every job, and a bad
        # value would otherwise fill the log with the same line.
        log.warning("Configuration problem: %s", entry.message())


def _clear(field: str) -> None:
    with _problem_lock:
        _problems.pop(field, None)


def configuration_problems() -> list[str]:
    """Problems recorded by fields read so far, cleared by a later valid read."""
    with _problem_lock:
        entries = list(_problems.values())
    return [entry.message() for entry in sorted(entries, key=lambda item: item.field)]


def clear_configuration_problems() -> None:
    """Forget what has been recorded. For tests and for a settings reload."""
    with _problem_lock:
        _problems.clear()


def _raw(name: str, *, environ: dict[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return str(source.get(name, "") or "")


def env_text(
    name: str,
    default: str = "",
    *,
    lower: bool = False,
    environ: dict[str, str] | None = None,
) -> str:
    """A configured string, or the default when it is unset or blank."""
    value = _raw(name, environ=environ).strip()
    if not value:
        _clear(name)
        return default
    return value.lower() if lower else value


def env_choice(
    name: str,
    default: str,
    *,
    choices: tuple[str, ...],
    environ: dict[str, str] | None = None,
) -> str:
    """One of `choices`, case-insensitively. Anything else is recorded."""
    value = _raw(name, environ=environ).strip().lower()
    if not value:
        _clear(name)
        return default
    if value in choices:
        _clear(name)
        return value
    _record(
        name,
        value,
        f"不是可用取值（{'、'.join(choices)}），已按 {default!r} 处理",
    )
    return default


def env_bool(
    name: str,
    default: bool,
    *,
    environ: dict[str, str] | None = None,
) -> bool:
    value = _raw(name, environ=environ).strip().lower()
    if not value:
        _clear(name)
        return default
    if value in _TRUE_WORDS:
        _clear(name)
        return True
    if value in _FALSE_WORDS:
        _clear(name)
        return False
    _record(name, value, f"不是开关值（1/0、true/false、on/off），已按 {default} 处理")
    return default


# What an out-of-range value means, which is not the same question everywhere.
#
# For a cap - line width, batch size, a retry count - "too big" is a request the
# nearest legal value still honours, so it is clamped. For a rate or a budget,
# the bound is where the quantity stops meaning anything: clamping
# `ASR_DECODE_TOKENS_PER_SECOND=-5` to its floor would hand the decoder a budget
# of nearly zero and truncate every line, which is a far worse answer to a typo
# than the configured default. Callers say which they are.
CLAMP = "clamp"
FALL_BACK = "fall-back"


def _in_range(
    name: str,
    value: float,
    *,
    default: float,
    minimum: float | None,
    maximum: float | None,
    out_of_range: str,
    formatter,
) -> float:
    if minimum is not None and value < minimum:
        bound, limit = minimum, f"低于允许的最小值 {formatter(minimum)}"
    elif maximum is not None and value > maximum:
        bound, limit = maximum, f"高于允许的最大值 {formatter(maximum)}"
    else:
        _clear(name)
        return value
    if out_of_range == CLAMP:
        _record(name, formatter(value), f"{limit}，已按该边界处理")
        return bound
    _record(name, formatter(value), f"{limit}，已按默认值 {formatter(default)} 处理")
    return default


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    out_of_range: str = CLAMP,
    environ: dict[str, str] | None = None,
) -> int:
    raw = _raw(name, environ=environ).strip()
    if not raw:
        _clear(name)
        return default
    try:
        # `int(float(...))` on purpose: `.env` files written by hand and by the
        # settings page both contain values like "4.0", and refusing those would
        # be a rule about formatting rather than about the value.
        parsed = float(raw)
        if not math.isfinite(parsed) or not parsed.is_integer():
            raise ValueError(raw)
        value = int(parsed)
    except (TypeError, ValueError):
        _record(name, raw, f"不是整数，已按默认值 {default} 处理")
        return default
    return int(
        _in_range(
            name,
            value,
            default=default,
            minimum=minimum,
            maximum=maximum,
            out_of_range=out_of_range,
            formatter=lambda item: str(int(item)),
        )
    )


def env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    out_of_range: str = CLAMP,
    environ: dict[str, str] | None = None,
) -> float:
    raw = _raw(name, environ=environ).strip()
    if not raw:
        _clear(name)
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        _record(name, raw, f"不是数字，已按默认值 {default} 处理")
        return default
    if not math.isfinite(value):
        # inf and nan parse as floats and then poison every duration derived
        # from them without raising anywhere.
        _record(name, raw, f"不是有限数字，已按默认值 {default} 处理")
        return default
    return float(
        _in_range(
            name,
            value,
            default=default,
            minimum=minimum,
            maximum=maximum,
            out_of_range=out_of_range,
            formatter=lambda item: f"{float(item):g}",
        )
    )


__all__ = [
    "CLAMP",
    "FALL_BACK",
    "ConfigProblem",
    "clear_configuration_problems",
    "configuration_problems",
    "env_bool",
    "env_choice",
    "env_float",
    "env_int",
    "env_text",
]
