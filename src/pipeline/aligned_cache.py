from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _stable_signature(value: dict | None) -> str:
    if not isinstance(value, dict):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _diff_signature_keys(saved: dict, expected: dict, *, prefix: str = "") -> list[str]:
    diffs: list[str] = []
    all_keys = sorted(set(saved) | set(expected))
    for key in all_keys:
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if key not in saved or key not in expected:
            diffs.append(full_key)
            continue
        left, right = saved[key], expected[key]
        if isinstance(left, dict) and isinstance(right, dict):
            diffs.extend(_diff_signature_keys(left, right, prefix=full_key))
        elif left != right:
            diffs.append(full_key)
    return diffs


def try_load_aligned_segments(
    path: str,
    expected_audio_cache_key: str,
    expected_backend: str,
    expected_signature: dict | None = None,
) -> dict | None:
    cache_path = Path(path)
    try:
        if not cache_path.exists():
            logger.debug("aligned_cache miss: file not found (%s)", cache_path.name)
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            logger.info("aligned_cache miss: payload is not a JSON object (%s)", cache_path.name)
            return None
        if payload.get("audio_cache_key") != expected_audio_cache_key:
            logger.info(
                "aligned_cache miss: audio_cache_key mismatch (saved=%r expected=%r)",
                payload.get("audio_cache_key"),
                expected_audio_cache_key,
            )
            return None
        if payload.get("backend") != expected_backend:
            logger.info(
                "aligned_cache miss: backend mismatch (saved=%r expected=%r)",
                payload.get("backend"),
                expected_backend,
            )
            return None
        if expected_signature is not None:
            saved_signature = payload.get("cache_signature", payload.get("signature"))
            if not isinstance(saved_signature, dict):
                logger.info(
                    "aligned_cache miss: cache_signature missing or invalid (%s)",
                    cache_path.name,
                )
                return None
            if _stable_signature(saved_signature) != _stable_signature(expected_signature):
                mismatched = _diff_signature_keys(saved_signature, expected_signature)
                logger.info(
                    "aligned_cache miss: signature mismatch keys=%s (%s)",
                    ",".join(mismatched[:12]) or "<unknown>",
                    cache_path.name,
                )
                return None
        segments = payload.get("segments")
        if not isinstance(segments, list):
            logger.info("aligned_cache miss: segments field missing or not a list (%s)", cache_path.name)
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
    except Exception as exc:
        logger.warning(
            "aligned_cache miss: failed to read %s (%s: %s)",
            cache_path.name,
            type(exc).__name__,
            exc,
        )
        return None
