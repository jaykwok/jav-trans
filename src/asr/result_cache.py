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
import os
import uuid
import wave
from pathlib import Path
from typing import Any

from asr.backends.registry import current_asr_backend
from asr.backends.qwen import active_qwen_asr_model_id


CACHE_SCHEMA = "asr_result_cache_v1"


def _env_text(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_lower(name: str, default: str = "") -> str:
    return _env_text(name, default).lower()


def result_cache_enabled() -> bool:
    return _env_lower("ASR_RESULT_CACHE_ENABLED", "1") not in {"0", "false", "no", "off"}


def result_cache_root() -> Path:
    return Path(os.getenv("ASR_RESULT_CACHE_ROOT", Path("tmp") / "asr_cache")).resolve()


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


def _cacheable_text_results(text_results_by_index: dict[int, dict]) -> dict[int, dict]:
    return {
        index: result
        for index, result in text_results_by_index.items()
        if not _is_timed_out_result(result) and not _is_quarantined_result(result)
    }


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
            "asr_force_language": _env_lower("ASR_FORCE_LANGUAGE", "1"),
        },
        "generation": {
            # Empty means "derive the budget from each chunk's duration", which
            # is the default; an explicit value is a hard ceiling that can
            # truncate, so it changes the text and belongs here either way.
            "asr_max_new_tokens": _env_text("ASR_MAX_NEW_TOKENS", ""),
            # The rate ceiling *is* the budget when no explicit cap is set.
            "asr_decode_tokens_per_second": _env_text(
                "ASR_DECODE_TOKENS_PER_SECOND", ""
            ),
            "asr_repetition_penalty": _env_text("ASR_REPETITION_PENALTY", "1.05"),
            # The loop guard ends sequences early, so it is part of what the text
            # *is* - not a speed setting. Left out of the signature it made the
            # cache lie: switching it off replayed guard-on text and made the two
            # settings indistinguishable, which is exactly the comparison anyone
            # touching the guard needs to run.
            "asr_decode_loop_guard": _env_lower("ASR_DECODE_LOOP_GUARD", "1"),
            "asr_decode_loop_budget_fraction": _env_text(
                "ASR_DECODE_LOOP_BUDGET_FRACTION", ""
            ),
            "asr_decode_loop_max_ngram": _env_text("ASR_DECODE_LOOP_MAX_NGRAM", ""),
            "asr_decode_loop_min_repeats": _env_text("ASR_DECODE_LOOP_MIN_REPEATS", ""),
            "asr_decode_loop_min_tokens": _env_text("ASR_DECODE_LOOP_MIN_TOKENS", ""),
        },
    }


def _signature_hash(signature: dict) -> str:
    return hashlib.sha1(_signature_json(signature).encode()).hexdigest()[:12]


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
        except Exception:
            pass


def lookup(chunk_path: str | Path) -> dict | None:
    if not result_cache_enabled():
        return None
    try:
        entry_path = _cache_dir(model_signature()) / f"{chunk_audio_sha256(chunk_path)}.json"
        if not entry_path.exists():
            return None
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("schema") != CACHE_SCHEMA:
        return None
    text_result = payload.get("text_result")
    if not isinstance(text_result, dict):
        return None
    return dict(text_result)


def store(chunk_path: str | Path, text_result: dict) -> None:
    if not result_cache_enabled():
        return
    if _is_timed_out_result(text_result) or _is_quarantined_result(text_result):
        return
    try:
        signature = model_signature()
        cache_dir = _cache_dir(signature)
        sanitized = {
            key: value
            for key, value in text_result.items()
            if key != "normalized_path"
        }
        signature_path = cache_dir / "signature.json"
        if not signature_path.exists():
            _atomic_write_json(signature_path, signature)
        _atomic_write_json(
            cache_dir / f"{chunk_audio_sha256(chunk_path)}.json",
            {"schema": CACHE_SCHEMA, "text_result": sanitized},
        )
    except Exception:
        # The cache is an accelerator: a full disk or an unreadable wav must
        # never take down the transcription that just succeeded.
        return


def restore_text_result(chunk: dict, cached: dict) -> dict:
    """Rehydrate a cache entry for this job's chunk instance."""
    result = dict(cached)
    current_path = str(Path(chunk["path"]).resolve())
    result["normalized_path"] = current_path
    try:
        with wave.open(current_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
        result["duration"] = frames / rate if rate else 0.0
    except Exception:
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


def _alignment_head_digest() -> str | None:
    raw_path = (os.environ.get("ASR_ALIGNMENT_HEAD_PATH") or "").strip()
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
    except Exception:
        return None


def finalize_signature() -> dict | None:
    head_digest = _alignment_head_digest()
    if head_digest is None:
        return None
    signature = model_signature()
    signature["stage"] = "final"
    signature["alignment_head"] = {"sha256": head_digest}
    return signature


def finalize_lookup(chunk_path: str | Path, *, text: str) -> tuple[dict, list[str]] | None:
    if not result_cache_enabled():
        return None
    signature = finalize_signature()
    if signature is None:
        return None
    try:
        entry_path = _cache_dir(signature) / f"{chunk_audio_sha256(chunk_path)}.json"
        if not entry_path.exists():
            return None
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
    except Exception:
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
) -> None:
    if not result_cache_enabled():
        return
    if str(result.get("alignment_mode") or "") != _ALIGNED_FINALIZE_MODE:
        return
    signature = finalize_signature()
    if signature is None:
        return
    try:
        cache_dir = _cache_dir(signature)
        signature_path = cache_dir / "signature.json"
        if not signature_path.exists():
            _atomic_write_json(signature_path, signature)
        _atomic_write_json(
            cache_dir / f"{chunk_audio_sha256(chunk_path)}.json",
            {
                "schema": CACHE_SCHEMA,
                "stage": "final",
                "source_text": str(text or ""),
                "result": dict(result),
                "log": [str(entry) for entry in log],
            },
        )
    except Exception:
        return
