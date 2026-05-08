from __future__ import annotations

import re


def sanitize_job_id(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", (value or "").strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "job"
