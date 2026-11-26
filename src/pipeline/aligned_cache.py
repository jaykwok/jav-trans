from __future__ import annotations

import json
from pathlib import Path


def _stable_signature(value: dict | None) -> str:
    if not isinstance(value, dict):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def try_load_aligned_segments(
    path: str,
    expected_audio_cache_key: str,
    expected_backend: str,
    expected_signature: dict | None = None,
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
        if expected_signature is not None:
            saved_signature = payload.get("cache_signature", payload.get("signature"))
            if not isinstance(saved_signature, dict):
                return None
            if _stable_signature(saved_signature) != _stable_signature(expected_signature):
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
