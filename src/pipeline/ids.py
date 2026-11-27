from __future__ import annotations

import hashlib
import re

_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


def sanitize_job_id(value: str) -> str:
    raw = (value or "").strip()
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", raw)
    sanitized = sanitized.strip("._-")
    if not sanitized:
        if raw:
            digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:8]
            return f"job_{digest}"
        return "job"
    stem = sanitized.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_DEVICE_NAMES:
        return f"job_{sanitized}"
    return sanitized
