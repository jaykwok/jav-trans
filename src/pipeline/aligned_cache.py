from __future__ import annotations

import json
from pathlib import Path


def try_load_aligned_segments(
    path: str,
    expected_audio_cache_key: str,
    expected_backend: str,
) -> dict | None:
    try:
        cache_path = Path(path)
        if not cache_path.exists():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("audio_cache_key") != expected_audio_cache_key:
            return None
        if payload.get("backend") != expected_backend:
            return None
        segments = payload.get("segments")
        if not isinstance(segments, list):
            return None
        payload["segments"] = [
            dict(segment) for segment in segments if isinstance(segment, dict)
        ]
        payload["asr_details"] = (
            payload.get("asr_details")
            if isinstance(payload.get("asr_details"), dict)
            else {}
        )
        payload["asr_log"] = (
            payload.get("asr_log") if isinstance(payload.get("asr_log"), list) else []
        )
        return payload
    except Exception:
        return None
