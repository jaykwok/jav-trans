"""Cross-job, content-addressed cache of per-chunk ASR text results.

Replaces the per-job crash-resume checkpoint (2026-08-01). Entries are keyed by
chunk audio content (PCM bytes + wav params) plus the model/decode identity, so
rerunning the same film — or any job that produces byte-identical chunk audio —
skips encoder and decoder entirely. Within one run the per-entry files double as
the crash-resume mechanism the checkpoint used to provide.

Entries are chunk-relative: nothing stored depends on the chunk's absolute
position, index, or path. Timeout/quarantined results never enter the cache.

Layout:
    <ASR_RESULT_CACHE_ROOT>/<model_sig_sha1_12>/<audio_sha256>.json
    <ASR_RESULT_CACHE_ROOT>/<model_sig_sha1_12>/signature.json

The signature intentionally omits the boundary/chunking configuration (the
audio hash already pins the exact samples) and the worker mode (process layout
cannot change what the model transcribes). The reserved "stage" field is fixed
to "text": a finalize-stage cache (word timings) only becomes worthwhile once
the alignment head is enabled, and its key must additionally carry the head
checkpoint digest.
"""

import hashlib
import json
import logging
import os
import uuid
import wave
from pathlib import Path
from typing import Any

from asr.backends.qwen import active_qwen_asr_model_id
from asr.backends.registry import current_asr_backend
from asr.decode_guard import (
    MIN_REPEATS,
    explicit_token_cap,
    loop_budget_fraction,
    loop_guard_enabled,
    tokens_per_second_ceiling,
)
from core.typed_config import env_bool, env_float, env_int, env_text

logger = logging.getLogger(__name__)
CACHE_SCHEMA = "asr_result_cache_v1"


def _env_text(name: str, default: str = "") -> str:
    return env_text(name, default)


def _env_lower(name: str, default: str = "") -> str:
    return env_text(name, default, lower=True)


def result_cache_enabled() -> bool:
    return env_bool("ASR_RESULT_CACHE_ENABLED", True)


def result_cache_root() -> Path:
    return Path(env_text("ASR_RESULT_CACHE_ROOT", str(Path("cache") / "asr"))).resolve()


def _is_timed_out_result(result: dict) -> bool:
    return any("TIMEOUT:" in entry for entry in result.get("log", []))


def _is_quarantined_result(result: dict) -> bool:
    # Circuit-breaker quarantined chunks carry empty text and must be
    # re-transcribed; caching them would freeze the failure. Tagged via
    # asr_generation.policy == "quarantined_result" (and a "QUARANTINED:" log
    # line), distinct from timeout results.
    generation = result.get("asr_generation")
    if isinstance(generation, dict) and generation.get("policy") == "quarantined_result":
        return True
    return any(
        isinstance(entry, str) and entry.startswith("QUARANTINED:")
        for entry in result.get("log", [])
    )


def _is_failed_result(result: dict) -> bool:
    """True for timed-out, quarantined, or error-skipped chunks that must not be cached."""
    if _is_timed_out_result(result) or _is_quarantined_result(result):
        return True
    generation = result.get("asr_generation")
    return isinstance(generation, dict) and bool(generation.get("error_kind"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _jsonable(value.item())
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return _jsonable(value.item())
            array = np.ascontiguousarray(value)
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            return {
                "array_type": "ndarray",
                "dtype": str(array.dtype),
                "shape": [int(item) for item in array.shape],
                "sha256": digest,
            }
    except Exception:
        pass
    return str(value)


def _signature_json(payload: dict) -> str:
    return json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _optional_positive_int_env(name: str) -> int | None:
    if not _env_text(name, ""):
        return None
    parsed = env_int(name, 0, minimum=1)
    return parsed or None


def model_signature() -> dict:
    return {
        "schema": CACHE_SCHEMA,
        "stage": "text",
        "backend": current_asr_backend(),
        "model": {
            "asr_model_id": _env_text("ASR_MODEL_ID", ""),
            "resolved_asr_model_id": active_qwen_asr_model_id(),
            "asr_model_path": _env_text("ASR_MODEL_PATH", ""),
            "asr_dtype": _env_lower("ASR_DTYPE", "auto"),
            "asr_attention": _env_lower("ASR_ATTENTION", "auto"),
        },
        "language": {
            "asr_language": _env_text("ASR_LANGUAGE", "Japanese") or "Japanese",
            "asr_force_language": env_bool("ASR_FORCE_LANGUAGE", True),
        },
        "generation": {
            # None means "derive the budget from each chunk's duration", which
            # is the default; an explicit value is a hard ceiling that can
            # truncate, so it changes the text and belongs here either way.
            "asr_max_new_tokens": explicit_token_cap(),
            # The rate ceiling *is* the budget when no explicit cap is set.
            "asr_decode_tokens_per_second": round(
                float(tokens_per_second_ceiling()), 6
            ),
            "asr_repetition_penalty": round(
                float(env_float("ASR_REPETITION_PENALTY", 1.05, minimum=1.0)), 6
            ),
            # The loop guard ends sequences early, so it is part of what the text
            # *is* - not a speed setting.
            "asr_decode_loop_guard": loop_guard_enabled(),
            "asr_decode_loop_budget_fraction": round(
                float(loop_budget_fraction()), 6
            ),
            "asr_decode_loop_max_ngram": _optional_positive_int_env(
                "ASR_DECODE_LOOP_MAX_NGRAM"
            ),
            "asr_decode_loop_min_repeats": env_int(
                "ASR_DECODE_LOOP_MIN_REPEATS", MIN_REPEATS, minimum=1
            ),
            "asr_decode_loop_min_tokens": _optional_positive_int_env(
                "ASR_DECODE_LOOP_MIN_TOKENS"
            ),
        },
    }


def _signature_hash(signature: dict) -> str:
    return hashlib.sha1(_signature_json(signature).encode()).hexdigest()[:12]


def pcm_audio_sha256(
    pcm_bytes: bytes,
    *,
    nchannels: int,
    sampwidth: int,
    framerate: int,
    nframes: int,
) -> str:
    """Content hash of raw PCM bytes + WAV geometry, matching ``chunk_audio_sha256``."""
    header = (
        f"channels={int(nchannels)};width={int(sampwidth)};"
        f"rate={int(framerate)};frames={int(nframes)}"
    )
    digest = hashlib.sha256(header.encode())
    digest.update(pcm_bytes)
    return digest.hexdigest()


def chunk_audio_sha256(chunk_path: str | Path) -> str:
    with wave.open(str(chunk_path), "rb") as reader:
        params = reader.getparams()
        header = (
            f"channels={params.nchannels};width={params.sampwidth};"
            f"rate={params.framerate};frames={params.nframes}"
        )
        digest = hashlib.sha256(header.encode())
        remaining = params.nframes
        while remaining > 0:
            frames = reader.readframes(min(remaining, 1_048_576))
            if not frames:
                break
            digest.update(frames)
            remaining -= min(remaining, 1_048_576)
    return digest.hexdigest()


def _cache_dir(signature: dict) -> Path:
    return result_cache_root() / _signature_hash(signature)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as writer:
            json.dump(_jsonable(payload), writer, ensure_ascii=False)
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("Could not remove ASR cache temp file %s: %s", tmp_path, exc)


def _read_entry(entry_path: Path) -> dict | None:
    """Read one cache entry, or None for anything that is not a usable one.

    The parse and the shape check belong to the same best-effort boundary. Split
    apart, a file holding `[]` or `null` parses fine and then raises
    AttributeError on the first `.get` - out of the `except`, up through
    finalize, and out of a job that only ever consulted an accelerator. A cache
    is allowed to be a miss; it is not allowed to be a source of failure.
    """
    try:
        if not entry_path.exists():
            return None
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Corrupt or unreadable ASR cache entry %s: %s", entry_path.name, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Ignoring non-object ASR cache entry %s", entry_path.name)
        return None
    return payload


def lookup(
    chunk_path: str | Path,
    *,
    audio_sha256: str | None = None,
    signature: dict | None = None,
    cache_dir: Path | None = None,
) -> dict | None:
    if not result_cache_enabled():
        return None
    try:
        resolved_dir = (
            cache_dir
            if cache_dir is not None
            else _cache_dir(signature if signature is not None else model_signature())
        )
        digest = audio_sha256 or chunk_audio_sha256(chunk_path)
        entry_path = resolved_dir / f"{digest}.json"
    except Exception as exc:
        logger.warning("ASR text cache lookup failed for %s: %s", Path(chunk_path).name, exc)
        return None
    payload = _read_entry(entry_path)
    if payload is None:
        return None
    if payload.get("schema") != CACHE_SCHEMA:
        return None
    text_result = payload.get("text_result")
    if not isinstance(text_result, dict):
        return None
    return dict(text_result)


def store(
    chunk_path: str | Path,
    text_result: dict,
    *,
    audio_sha256: str | None = None,
    signature: dict | None = None,
    cache_dir: Path | None = None,
) -> None:
    if not result_cache_enabled():
        return
    if _is_failed_result(text_result):
        return
    try:
        resolved_signature = signature if signature is not None else model_signature()
        resolved_dir = cache_dir if cache_dir is not None else _cache_dir(resolved_signature)
        digest = audio_sha256 or chunk_audio_sha256(chunk_path)
        sanitized = {
            key: value
            for key, value in text_result.items()
            if key not in {"normalized_path", "audio_sha256"}
        }
        signature_path = resolved_dir / "signature.json"
        if not signature_path.exists():
            _atomic_write_json(signature_path, resolved_signature)
        _atomic_write_json(
            resolved_dir / f"{digest}.json",
            {"schema": CACHE_SCHEMA, "text_result": sanitized},
        )
    except Exception as exc:
        # The cache is an accelerator: a full disk or an unreadable wav must
        # never take down the transcription that just succeeded.
        logger.warning("ASR text cache write failed for %s: %s", Path(chunk_path).name, exc)
        return


def restore_text_result(chunk: dict, cached: dict) -> dict:
    """Rehydrate a cache entry for this job's chunk instance."""
    result = dict(cached)
    current_path = str(Path(chunk["path"]).resolve())
    result["normalized_path"] = current_path
    if chunk.get("audio_sha256"):
        result["audio_sha256"] = str(chunk["audio_sha256"])
    chunk_duration = chunk.get("duration")
    if isinstance(chunk_duration, (int, float)) and float(chunk_duration) > 0.0:
        result["duration"] = float(chunk_duration)
    else:
        try:
            with wave.open(current_path, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
            result["duration"] = frames / rate if rate else 0.0
        except Exception as exc:
            logger.warning(
                "Could not read chunk wav duration for %s (%s); using cached duration",
                Path(current_path).name,
                exc,
            )
            result["duration"] = float(result.get("duration", 0.0))
    result.setdefault("language", "Japanese")
    result.setdefault("text", "")
    result.setdefault("raw_text", result.get("text", ""))
    result_log = list(result.get("log", []))
    result_log.append("ASR result cache hit: restored chunk text")
    result["log"] = result_log
    return result


# --- finalize-stage cache (word timing; active only with an alignment head) ---
#
# The text-stage cache above spares the decoder. Once an alignment head is
# configured, finalize additionally pays one encoder pass + a Viterbi walk per
# chunk, and its output depends on one more artifact: the head checkpoint. So
# the finalize entry lives under its own signature (stage="final" + head
# digest) and stores the source text it aligned, verified on lookup — a text
# mismatch means the entry describes different words, not different audio.
#
# Only real alignments (alignment_mode == "ctc_forced_alignment") are stored.
# Declines and failures re-run on the next pass: an encoder forward on the few
# declined chunks is cheap, and caching a degraded result would freeze what
# might have been a transient error into a permanent proportional timeline.

_ALIGNED_FINALIZE_MODE = "ctc_forced_alignment"
_HEAD_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def _checkpoint_digest(raw_path: str) -> str | None:
    raw_path = str(raw_path or "").strip()
    if not raw_path:
        return None
    try:
        # download=False: this runs before anyone has asked for the head, and a
        # cache-key lookup is no place to start a network fetch. An uncached
        # `hf:` reference yields "" and disables the finalize cache for that one
        # call; the loader downloads it, and the next call keys off the file.
        from asr.alignment import resolve_alignment_head_path

        local_path = resolve_alignment_head_path(raw_path, download=False)
        if not local_path:
            return None
        head_path = Path(local_path).resolve()
        stat = head_path.stat()
        cache_key = (str(head_path), stat.st_size, stat.st_mtime_ns)
        cached = _HEAD_DIGEST_CACHE.get(cache_key)
        if cached is not None:
            return cached
        digest = hashlib.sha256(head_path.read_bytes()).hexdigest()
        _HEAD_DIGEST_CACHE.clear()
        _HEAD_DIGEST_CACHE[cache_key] = digest
        return digest
    except Exception as exc:
        logger.warning("Could not hash alignment checkpoint %r: %s", raw_path, exc)
        return None


def _alignment_head_digest() -> str | None:
    return _checkpoint_digest(os.environ.get("ASR_ALIGNMENT_HEAD_PATH") or "")


def finalize_signature() -> dict | None:
    head_digest = _alignment_head_digest()
    if head_digest is None:
        return None
    signature = model_signature()
    signature["stage"] = "final"
    signature["alignment_head"] = {"sha256": head_digest}
    # The head checkpoint is not the only thing that decides the stored word
    # times: `speech_extent` walks both edges outward afterwards, capped by these
    # two constants, and `blank_bias` shifts character edges during Viterbi.
    from asr.alignment import (
        CODA_EXTEND_MAX_S,
        ONSET_BACKOFF_MAX_S,
        blank_bias_from_env,
    )

    signature["blank_bias"] = round(float(blank_bias_from_env()), 6)
    signature["edge_caps"] = {
        "onset_backoff_max_s": round(float(ONSET_BACKOFF_MAX_S), 6),
        "coda_extend_max_s": round(float(CODA_EXTEND_MAX_S), 6),
    }
    signature["word_build_version"] = 3
    return signature


def finalize_lookup(
    chunk_path: str | Path,
    *,
    text: str,
    audio_sha256: str | None = None,
    signature: dict | None = None,
    cache_dir: Path | None = None,
) -> tuple[dict, list[str]] | None:
    if not result_cache_enabled():
        return None
    resolved_dir = cache_dir
    if resolved_dir is None:
        resolved_signature = signature if signature is not None else finalize_signature()
        if resolved_signature is None:
            return None
        resolved_dir = _cache_dir(resolved_signature)
    try:
        digest = audio_sha256 or chunk_audio_sha256(chunk_path)
        entry_path = resolved_dir / f"{digest}.json"
    except Exception as exc:
        logger.warning("ASR finalize cache lookup failed for %s: %s", Path(chunk_path).name, exc)
        return None
    payload = _read_entry(entry_path)
    if payload is None:
        return None
    if payload.get("schema") != CACHE_SCHEMA or payload.get("stage") != "final":
        return None
    if str(payload.get("source_text") or "") != str(text or ""):
        return None
    result = payload.get("result")
    log = payload.get("log")
    if not isinstance(result, dict) or not isinstance(log, list):
        return None
    restored_log = [str(entry) for entry in log]
    restored_log.append("ASR finalize cache hit: restored word timing")
    return dict(result), restored_log


def finalize_store(
    chunk_path: str | Path,
    *,
    text: str,
    result: dict,
    log: list[str],
    audio_sha256: str | None = None,
    signature: dict | None = None,
    cache_dir: Path | None = None,
) -> None:
    if not result_cache_enabled():
        return
    if str(result.get("alignment_mode") or "") != _ALIGNED_FINALIZE_MODE:
        return
    resolved_signature = signature if signature is not None else finalize_signature()
    if resolved_signature is None:
        return
    try:
        resolved_dir = cache_dir if cache_dir is not None else _cache_dir(resolved_signature)
        digest = audio_sha256 or chunk_audio_sha256(chunk_path)
        signature_path = resolved_dir / "signature.json"
        if not signature_path.exists():
            _atomic_write_json(signature_path, resolved_signature)
        _atomic_write_json(
            resolved_dir / f"{digest}.json",
            {
                "schema": CACHE_SCHEMA,
                "stage": "final",
                "source_text": str(text or ""),
                "result": dict(result),
                "log": [str(entry) for entry in log],
            },
        )
    except Exception as exc:
        logger.warning("ASR finalize cache write failed for %s: %s", Path(chunk_path).name, exc)
        return
