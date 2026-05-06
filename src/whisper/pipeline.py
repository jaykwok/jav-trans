import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import warnings
import wave
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from whisper.backends.base import BaseAsrBackend
from whisper.qc import (
    ASR_QC_ENABLED,
    ASR_RECOVERY_ENABLED,
    evaluate_asr_text_results_qc,
    format_qc_log_items,
)
from whisper.local_backend import (
    LocalAsrBackend,
    ASR_DTYPE,
    ASR_MODEL_ID,
    ALIGNMENT_TIMESTAMP_MODE,
    SubprocessAsrBackend,
    WorkerError,
    WorkerTimeoutError,
    looks_like_word_timing_failure,
)
from whisper.timestamp_fallback import build_word_timestamps_fallback

warnings.filterwarnings("ignore")

ASR_BACKEND = os.getenv("ASR_BACKEND", "anime-whisper").strip().lower()
WHISPER_TIMESTAMP_MODE = os.getenv(
    "WHISPER_TIMESTAMP_MODE",
    "forced",
).strip().lower()

_GRAY_MAX_DURATION_S = float(os.getenv("ASR_GRAY_MAX_DURATION", "2.5"))
# also used by vad/ffmpeg_backend.py
_SEGMENT_MIN_SILENCE_S = float(os.getenv("SEGMENT_MIN_SILENCE", "0.35"))
_SEGMENT_MIN_CHUNK_S = float(os.getenv("SEGMENT_MIN_CHUNK", "1.2"))
_SEGMENT_MAX_CHUNK_S = float(os.getenv("SEGMENT_MAX_CHUNK", "18.0"))
_SEGMENT_TARGET_CHUNK_S = min(
    _SEGMENT_MAX_CHUNK_S,
    float(os.getenv("SEGMENT_TARGET_CHUNK", str(min(60.0, _SEGMENT_MAX_CHUNK_S)))),
)
_SEGMENT_MIN_SPEECH_S = float(os.getenv("SEGMENT_MIN_SPEECH", "0.25"))
_SEGMENT_PAD_S = float(os.getenv("SEGMENT_PAD", "0.15"))
_SEGMENT_CUT_MIN_SILENCE_S = float(os.getenv("SEGMENT_CUT_MIN_SILENCE", "0.5"))
_SEGMENT_SILENCE_DB = os.getenv("SEGMENT_SILENCE_DB", "-32dB").strip()
_KEEP_ASR_CHUNKS = os.getenv("KEEP_ASR_CHUNKS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_ALIGNMENT_STEP_DOWN_CHUNK_S = float(os.getenv("ALIGNMENT_STEP_DOWN_CHUNK", "6.0"))
_ALIGNMENT_COARSE_REFINE_CHUNK_S = float(os.getenv("ALIGNMENT_COARSE_REFINE_CHUNK", "18.0"))
_ALIGNMENT_MAX_REFINE_DEPTH = max(0, int(os.getenv("ALIGNMENT_MAX_REFINE_DEPTH", "2")))
_ASR_CONTEXT = os.getenv("ASR_CONTEXT", "").strip()
_ASR_HEAD_CONTEXT = os.getenv("ASR_HEAD_CONTEXT", "").strip()
_ASR_HEAD_CONTEXT_MAX_START_S = float(os.getenv("ASR_HEAD_CONTEXT_MAX_START_S", "0"))
_ALIGNMENT_MIN_SPAN_MS = float(os.getenv("ALIGNMENT_MIN_SPAN_MS", "120"))
_ALIGNMENT_MAX_ZERO_RATIO = float(
    os.getenv("ALIGNMENT_MAX_ZERO_RATIO", "0.55")
)
_ALIGNMENT_MAX_REPEAT_RATIO = float(
    os.getenv("ALIGNMENT_MAX_REPEAT_RATIO", "0.65")
)
_ALIGNMENT_MAX_COVERAGE_RATIO = float(
    os.getenv("ALIGNMENT_MAX_COVERAGE_RATIO", "0.05")
)
_ALIGNMENT_MAX_CPS = float(os.getenv("ALIGNMENT_MAX_CPS", "50.0"))
_ALIGNMENT_RETRY_SKIP_MAX_TEXT_LEN = max(
    1,
    int(os.getenv("ALIGNMENT_RETRY_SKIP_MAX_TEXT_LEN", "10")),
)


def _is_timed_out_result(result: dict) -> bool:
    return any("TIMEOUT:" in entry for entry in result.get("log", []))


def _checkpointable_text_results(
    text_results_by_index: dict[int, dict],
) -> dict[int, dict]:
    return {
        index: result
        for index, result in text_results_by_index.items()
        if not _is_timed_out_result(result)
    }


_ASR_CONTEXT_LEAK_SIMILARITY = float(os.getenv("ASR_CONTEXT_LEAK_SIMILARITY", "0.88"))
_ASR_FRAGMENT_MERGE_MAX_GAP_S = float(os.getenv("ASR_FRAGMENT_MERGE_MAX_GAP", "1.0"))
_ASR_FRAGMENT_MERGE_MAX_CHARS = max(
    1,
    int(os.getenv("ASR_FRAGMENT_MERGE_MAX_CHARS", "72")),
)
_ASR_FRAGMENT_MERGE_MAX_DURATION_S = float(
    os.getenv("ASR_FRAGMENT_MERGE_MAX_DURATION", "12.5")
)
_ASR_INVALID_SEGMENT_DURATION_S = float(
    os.getenv("ASR_INVALID_SEGMENT_DURATION", "0.1")
)
_ASR_MIN_REPAIRED_SEGMENT_DURATION_S = float(
    os.getenv("ASR_MIN_REPAIRED_SEGMENT_DURATION", "0.6")
)
_ASR_POSTPROCESS_MAX_CHARS = max(
    8,
    int(os.getenv("ASR_POSTPROCESS_MAX_CHARS", "60")),
)
_ASR_POSTPROCESS_MAX_DURATION_S = float(
    os.getenv("ASR_POSTPROCESS_MAX_DURATION", "12.5")
)
_ASR_POSTPROCESS_SPLIT_MIN_CHARS = max(
    4,
    int(os.getenv("ASR_POSTPROCESS_SPLIT_MIN_CHARS", "4")),
)
_ASR_CHUNK_ROOT = Path(
    os.getenv("ASR_CHUNK_ROOT", Path("temp") / "chunks")
).resolve()
_ASR_CHECKPOINT_INTERVAL = max(1, int(os.getenv("ASR_CHECKPOINT_INTERVAL", "50")))
_ASR_WORKER_MODE = os.getenv("ASR_WORKER_MODE", "subprocess").strip().lower()
_ASR_SUBPROCESS_RESPAWN_MAX = max(
    0,
    int(os.getenv("ASR_SUBPROCESS_RESPAWN_MAX", "2")),
)
_ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT = max(
    1,
    int(os.getenv("ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT", "3")),
)
_ASR_SUBPROCESS_KILL_GRACE_S = float(os.getenv("ASR_SUBPROCESS_KILL_GRACE_S", "5"))
_ASR_SUBPROCESS_READY_TIMEOUT_S = float(
    os.getenv("ASR_SUBPROCESS_READY_TIMEOUT_S", "600")
)
_ASR_CHECKPOINT_ENABLED = os.getenv("ASR_CHECKPOINT_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_LAST_VAD_SIGNATURE: dict = {}

from vad.ffmpeg_backend import _SILENCE_END_RE, _SILENCE_START_RE

_TRIVIAL_SEGMENT = re.compile(
    r"^[あっんーっ。！？\s…、ぁぃぅぇぉアイウエオ]{1,5}$"
    r"|^[。！？…、\s,.!?・「」（）【】；：\-—–]+$"
    r"|^[あいうえおぁぃぅぇぉんっーはふへほ]+$"
)
_TOOL_SIGNATURE_RE = re.compile(r"whisperjav\s+\d", re.I)
_STRIP_PUNCT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]\s~〜ー-]+")
_CONTEXT_COMPACT_RE = re.compile(r"[^0-9A-Za-zぁ-ゖァ-ヺ一-龯々〆ヵヶ]+")
_CONTEXT_TOKEN_SPLIT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]\s~〜ー\-；;：:\n\r\t]+")
_SENTENCE_TERMINAL_RE = re.compile(r"[。！？!?…」』）)\]]$")
_SENTENCE_FRAGMENT_RE = re.compile(r"[^。！？!?…]+[。！？!?…]?")
_FRAGMENT_CONTINUATION_START_RE = re.compile(
    r"^(?:ます|ました|ません|です|でした|でしょう|ながら|ので|けど|から|たり|"
    r"程度|ところ|いる|いき|して|され|なり|効果|で[、,]?|に|を|が|は|も)"
)
_MOAN_CHAR_RE = re.compile(
    r"^[ぁ-ゖァ-ヺー〜～っッんンあいうえおアイウエオはひふへほハヒフヘホ]+$"
)
_RHYTHMIC_HALLUCINATION_RE = re.compile(
    r"^.+(?:ささ|まま|だだ|なな|たた|かか|やや|ばば|ぱぱ|ちゃちゃ|じゃじゃ|べべ|ぺぺ)$"
    r"|^(.{2,5})\1+$"
)

_DEFAULT_NOISE_WORDS = (
    "あ,い,う,え,お,ん,"
    "あっ,うん,えっ,おっ,"
    "あー,えー,うー,おー,んー,"
    "ねえ,ねー,はい,もう,ふぁ,はぁ,くっ,あぁ,うぅ,おぉ"
)
_NOISE_WORDS = frozenset(
    w.strip()
    for w in os.getenv("ASR_NOISE_WORDS", _DEFAULT_NOISE_WORDS).split(",")
    if w.strip()
)

_GRAY_WORDS = frozenset(
    {
        "気持ち",
        "好き",
        "すごい",
        "すご",
        "いい",
        "だめ",
        "やばい",
        "ちょっと",
        "あかん",
    }
)
_LOW_VALUE_KEEP_WORDS = frozenset(
    {
        "だめ",
        "ダメ",
        "いや",
        "イヤ",
        "やめ",
        "やめて",
        "もっと",
        "まだ",
        "好き",
        "気持ちいい",
        "すごい",
        "お久しぶり",
        "出てきた",
        "奥",
        "本当に",
    }
)


def get_backend_label() -> str:
    if ASR_BACKEND == "qwen3-asr-1.7b":
        if _ASR_WORKER_MODE == "subprocess":
            return "qwen3-asr-1.7b (subprocess worker)"
        if _ASR_WORKER_MODE == "inproc":
            return "qwen3-asr-1.7b (inproc)"
        return f"qwen3-asr-1.7b ({_ASR_WORKER_MODE})"
    return ASR_BACKEND


_WHISPER_BACKENDS = {
    "anime-whisper",
    "whisper-ja-anime-v0.3",
    "whisper-ja-1.5b",
}
_QWEN_BACKENDS = {"qwen3-asr-1.7b"}
_VALID_ASR_BACKENDS = _WHISPER_BACKENDS | _QWEN_BACKENDS
_VALID_ASR_WORKER_MODES = {"inproc", "subprocess"}


def _create_whisper_backend(device: str) -> BaseAsrBackend:
    from whisper.model_backend import create_whisper_model_backend
    return create_whisper_model_backend(ASR_BACKEND, device)


def _resolve_asr_backend(device: str) -> BaseAsrBackend:
    if ASR_BACKEND not in _VALID_ASR_BACKENDS:
        raise ValueError(
            f"Unsupported ASR_BACKEND={ASR_BACKEND!r}; "
            f"expected one of {sorted(_VALID_ASR_BACKENDS)}"
        )
    if _ASR_WORKER_MODE not in _VALID_ASR_WORKER_MODES:
        raise ValueError(
            f"Unsupported ASR_WORKER_MODE={_ASR_WORKER_MODE!r}; "
            f"expected one of {sorted(_VALID_ASR_WORKER_MODES)}"
        )
    if ASR_BACKEND in _WHISPER_BACKENDS:
        return _create_whisper_backend(device)
    if _ASR_WORKER_MODE == "inproc":
        return LocalAsrBackend(device)
    return SubprocessAsrBackend(device)


def _create_asr_backend(device: str) -> BaseAsrBackend:
    return _resolve_asr_backend(device)


def _is_subprocess_backend(backend: BaseAsrBackend) -> bool:
    return bool(getattr(backend, "is_subprocess", False))


class ASRWorkerSystemError(RuntimeError):
    pass


def _strip_punctuation(text: str) -> str:
    return _STRIP_PUNCT_RE.sub("", text or "")


def _compact_context_text(text: str) -> str:
    return _CONTEXT_COMPACT_RE.sub("", text or "")


def _context_tokens(context: str) -> list[str]:
    tokens = []
    for token in _CONTEXT_TOKEN_SPLIT_RE.split(context or ""):
        compact = _compact_context_text(token)
        if len(compact) >= 2:
            tokens.append(compact)
    return tokens


def _is_context_leak(text: str) -> bool:
    compact = _compact_context_text(text)
    if len(compact) < 3:
        return False

    for context in (_ASR_CONTEXT, _ASR_HEAD_CONTEXT):
        context_compact = _compact_context_text(context)
        if len(context_compact) < 3:
            continue

        if compact == context_compact:
            return True
        if compact in context_compact:
            return True
        if context_compact in compact and len(compact) <= len(context_compact) * 1.25:
            return True
        if SequenceMatcher(None, compact, context_compact).ratio() >= _ASR_CONTEXT_LEAK_SIMILARITY:
            return True

        remainder = compact
        for token in sorted(_context_tokens(context), key=len, reverse=True):
            remainder = remainder.replace(token, "")
        if not remainder:
            return True

    return False


def _collapse_repeated_noise(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", (text or "").strip())
    cleaned = re.sub(r"(.)\1{3,}", r"\1\1", cleaned)

    for _ in range(2):
        updated = re.sub(
            r"([ぁ-ゖァ-ヺ一-龯]{1,8})(?:[、。！？…\s]*\1){2,}",
            r"\1、\1",
            cleaned,
        )
        if updated == cleaned:
            break
        cleaned = updated

    return cleaned.strip()


def _is_noise_token(text: str) -> bool:
    token = _strip_punctuation(text)
    if not token:
        return True
    if token in _LOW_VALUE_KEEP_WORDS:
        return False
    if token in _NOISE_WORDS:
        return True
    return len(token) <= 4 and bool(_MOAN_CHAR_RE.fullmatch(token))


def _is_low_value_text(text: str) -> bool:
    normalized = _collapse_repeated_noise(text)
    compact = _strip_punctuation(normalized)
    if not compact:
        return True
    if compact in _LOW_VALUE_KEEP_WORDS:
        return False

    if _ASR_CONTEXT:
        context_compact = _strip_punctuation(_ASR_CONTEXT)
        if context_compact and context_compact in compact and len(compact) <= len(context_compact) + 3:
            return True

    if _TRIVIAL_SEGMENT.match(normalized):
        return True
        
    if _RHYTHMIC_HALLUCINATION_RE.fullmatch(compact):
        return True

    parts = [part for part in re.split(r"[、。！？…,.!?・\s]+", normalized) if part]
    if parts and all(_is_noise_token(part) for part in parts):
        return True

    return _is_noise_token(normalized)


def _clean_segment_text(text: str) -> str:
    return _collapse_repeated_noise((text or "").replace("\r", " ").replace("\n", " "))


def _remove_context_leak_fragments(text: str) -> str:
    if not (_ASR_CONTEXT or _ASR_HEAD_CONTEXT):
        return text

    context_tokens = sorted(
        {
            token
            for context in (_ASR_CONTEXT, _ASR_HEAD_CONTEXT)
            for token in _context_tokens(context)
            if len(token) >= 2
        },
        key=len,
        reverse=True,
    )
    strong_tokens = [token for token in context_tokens if len(token) >= 3]

    cleaned_text = text or ""
    if len(context_tokens) >= 2 and strong_tokens:
        token_pattern = "|".join(re.escape(token) for token in context_tokens)
        sequence_re = re.compile(
            rf"(?:{token_pattern})(?:[；;、。,.!?！？…・\s]+(?:{token_pattern}))+[。！？!?…]?"
        )

        def _drop_context_sequence(match: re.Match) -> str:
            compact = _compact_context_text(match.group(0))
            if any(token in compact for token in strong_tokens):
                return ""
            return match.group(0)

        cleaned_text = sequence_re.sub(_drop_context_sequence, cleaned_text)

    fragments = _SENTENCE_FRAGMENT_RE.findall(cleaned_text)
    if not fragments:
        return "" if _is_context_leak(cleaned_text) else cleaned_text

    kept_fragments = []
    changed = cleaned_text != text
    for fragment in fragments:
        if _is_context_leak(fragment):
            changed = True
            continue

        cleaned_fragment = fragment
        for token in strong_tokens:
            if token not in cleaned_fragment:
                continue
            without_token = cleaned_fragment.replace(token, "")
            if _is_low_value_text(without_token):
                changed = True
                cleaned_fragment = without_token

        cleaned_fragment = re.sub(r"[、,\s]+([。！？!?…])", r"\1", cleaned_fragment)
        cleaned_fragment = re.sub(r"^[、。！？!?…,.・\s]+|[、,.・\s]+$", "", cleaned_fragment)
        cleaned_fragment = _collapse_repeated_noise(cleaned_fragment)

        if not cleaned_fragment or _is_low_value_text(cleaned_fragment):
            changed = True
            continue
        kept_fragments.append(cleaned_fragment)

    if not changed:
        return text
    return _collapse_repeated_noise("".join(kept_fragments))


def _delete_path_for_cleanup(path: Path) -> None:
    if not path.exists():
        return

    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except Exception:
        pass


def _get_whisper_generation_checkpoint_signature() -> str:
    if ASR_BACKEND not in _WHISPER_BACKENDS:
        return ""
    try:
        from whisper.model_backend import WHISPER_PRESETS

        preset = WHISPER_PRESETS.get(ASR_BACKEND, {})
        beams = os.getenv("WHISPER_BEAMS", "").strip() or str(preset.get("beams", ""))
        no_repeat_ngram = (
            os.getenv("WHISPER_NO_REPEAT_NGRAM", "").strip()
            or str(preset.get("no_repeat_ngram", ""))
        )
        max_new_tokens = (
            os.getenv("WHISPER_MAX_NEW_TOKENS", "").strip()
            or str(preset.get("max_new_tokens", ""))
        )
        forced_fail_ratio = (
            os.getenv("WHISPER_FORCED_FAIL_RATIO", "").strip()
            or str(preset.get("forced_fail_ratio", ""))
        )
        model_path = os.getenv("WHISPER_MODEL_PATH", "").strip()
        return json.dumps(
            {
                "beams": beams,
                "no_repeat_ngram": no_repeat_ngram,
                "max_new_tokens": max_new_tokens,
                "forced_fail_ratio": forced_fail_ratio,
                "model_path": model_path,
            },
            sort_keys=True,
        )
    except Exception:
        return ""


def _get_asr_checkpoint_path(audio_path: str) -> Path:
    vad_signature = json.dumps(_LAST_VAD_SIGNATURE, sort_keys=True)
    whisper_generation_signature = _get_whisper_generation_checkpoint_signature()
    key = hashlib.sha1(
        (
            f"{audio_path}|{ASR_MODEL_ID}|{ASR_DTYPE}|"
            f"{_ASR_WORKER_MODE}|{ALIGNMENT_TIMESTAMP_MODE}|{vad_signature}|"
            f"{ASR_BACKEND}|{WHISPER_TIMESTAMP_MODE}|{whisper_generation_signature}"
        ).encode()
    ).hexdigest()[:10]
    return _ASR_CHUNK_ROOT.parent / f"asr_checkpoint_{key}.json"


def _get_asr_checkpoint_source(chunks: list[dict], text_stage_label: str) -> str:
    source = str(chunks[0].get("source_audio_path") or chunks[0].get("path", ""))
    return f"{source}|{text_stage_label}"


def _chunk_checkpoint_signature(chunks: list[dict]) -> dict[str, dict[str, float | str]]:
    return {
        str(int(chunk["index"])): {
            "start": round(float(chunk.get("start", 0.0)), 3),
            "end": round(float(chunk.get("end", 0.0)), 3),
            "vad_method": _LAST_VAD_SIGNATURE.get("backend", "unknown"),
        }
        for chunk in chunks
    }


def _load_asr_checkpoint(
    checkpoint_path: Path,
    checkpoint_source: str,
    chunks: list[dict],
    run_id: str | None = None,
) -> dict[int, dict]:
    if not _ASR_CHECKPOINT_ENABLED or not checkpoint_path.exists():
        return {}

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as reader:
            payload = json.load(reader)
    except Exception:
        return {}

    if payload.get("audio_path") != checkpoint_source:
        return {}

    raw_results = payload.get("results", {})
    if not isinstance(raw_results, dict):
        return {}

    chunk_by_index = {int(chunk["index"]): chunk for chunk in chunks}
    expected_signature = _chunk_checkpoint_signature(chunks)
    saved_signature = payload.get("chunks", {})
    restored: dict[int, dict] = {}

    for key, value in raw_results.items():
        try:
            chunk_index = int(key)
        except (TypeError, ValueError):
            continue
        if chunk_index not in chunk_by_index or not isinstance(value, dict):
            continue
        if _is_timed_out_result(value):
            continue
        saved_chunk_signature = (
            saved_signature.get(str(chunk_index))
            if isinstance(saved_signature, dict)
            else None
        )
        if isinstance(saved_chunk_signature, dict) and "vad_method" not in saved_chunk_signature:
            print(
                f"[WARN] ASR checkpoint resume: chunk {chunk_index} missing vad_method; skip stale checkpoint entry",
                file=sys.stderr,
            )
            continue
        if (
            isinstance(saved_signature, dict)
            and saved_signature
            and saved_chunk_signature != expected_signature.get(str(chunk_index))
        ):
            continue

        chunk = chunk_by_index[chunk_index]
        result = dict(value)
        current_path = str(Path(chunk["path"]).resolve())
        result["normalized_path"] = current_path
        try:
            result["duration"] = _get_wav_duration(current_path)
        except Exception:
            result["duration"] = float(result.get("duration", 0.0))
        result.setdefault("language", "Japanese")
        result.setdefault("text", "")
        result.setdefault("raw_text", result.get("text", ""))
        result_log = list(result.get("log", []))
        result_log.append("ASR checkpoint resume: restored chunk text")
        saved_run_id = payload.get("run_id")
        if run_id and saved_run_id and saved_run_id != run_id:
            result_log.append(
                f"ASR checkpoint resume: saved_run_id={saved_run_id}, current_run_id={run_id}"
            )
        result["log"] = result_log
        restored[chunk_index] = result

    return restored


def _save_asr_checkpoint(
    checkpoint_path: Path,
    checkpoint_source: str,
    chunks: list[dict],
    text_results_by_index: dict[int, dict],
    run_id: str | None = None,
) -> None:
    if not _ASR_CHECKPOINT_ENABLED:
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(
        f"{checkpoint_path.name}.{uuid.uuid4().hex[:8]}.tmp"
    )
    payload = {
        "audio_path": checkpoint_source,
        "chunks": _chunk_checkpoint_signature(chunks),
        "results": {
            str(index): result
            for index, result in sorted(text_results_by_index.items())
        },
    }
    if run_id:
        payload["run_id"] = run_id
    try:
        with open(tmp_path, "w", encoding="utf-8") as writer:
            json.dump(payload, writer, ensure_ascii=False)
        tmp_path.replace(checkpoint_path)
    except Exception:
        _delete_path_for_cleanup(tmp_path)


def _build_quarantined_text_result(
    chunk: dict,
    *,
    kind: str,
    detail: str,
    respawn_count: int,
    run_id: str | None = None,
) -> dict:
    normalized_path = str(Path(chunk["path"]).resolve())
    try:
        duration = _get_wav_duration(normalized_path)
    except Exception:
        duration = max(
            0.0,
            float(chunk.get("end", 0.0)) - float(chunk.get("start", 0.0)),
        )

    return {
        "text": "",
        "raw_text": "",
        "duration": duration,
        "language": "Japanese",
        "normalized_path": normalized_path,
        "segments": [],
        "log": [
            (
                "QUARANTINED: "
                f"kind={kind}, respawn_count={respawn_count}, "
                f"run_id={run_id or ''}, detail={detail}"
            )
        ],
    }


def _quarantine_failed_chunks(
    checkpoint_source: str,
    chunks: list[dict],
    failure_records: list[dict],
    *,
    run_id: str | None = None,
    worker_mode: str | None = None,
) -> list[Path]:
    if not failure_records:
        return []

    chunk_by_index = {int(chunk["index"]): chunk for chunk in chunks}
    out_dir = _ASR_CHUNK_ROOT.parent / "asr_timeouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for record in failure_records:
        try:
            chunk_index = int(record["index"])
        except (KeyError, TypeError, ValueError):
            continue
        chunk = chunk_by_index.get(chunk_index)
        if chunk is None:
            continue

        record_run_id = str(record.get("run_id") or run_id or "unknown")
        chunk_key = hashlib.sha1(
            f"{checkpoint_source}|{chunk_index}".encode()
        ).hexdigest()[:10]
        target = out_dir / f"timeouts_{chunk_key}_{record_run_id}.json"
        tmp_path = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        payload = {
            "run_id": record_run_id,
            "audio_path": checkpoint_source,
            "chunk_index": chunk_index,
            "start": float(chunk.get("start", 0.0)),
            "end": float(chunk.get("end", 0.0)),
            "model": ASR_MODEL_ID,
            "dtype": ASR_DTYPE,
            "timeout_s": float(os.getenv("TRANSCRIPTION_TIMEOUT_S", "180")),
            "respawn_count": int(record.get("respawn_count", 0)),
            "failure_kind": str(record.get("kind") or "crash"),
            "last_error": str(record.get("detail", "")),
            "worker_mode": str(record.get("worker_mode") or worker_mode or _ASR_WORKER_MODE),
        }
        try:
            with open(tmp_path, "w", encoding="utf-8") as writer:
                json.dump(payload, writer, ensure_ascii=False, indent=2)
            tmp_path.replace(target)
            written.append(target)
        except Exception:
            _delete_path_for_cleanup(tmp_path)

    return written


def aggregate_timeout_fragments(job_id: str) -> Path | None:
    normalized_job_id = re.sub(r"[^0-9A-Za-z._-]+", "_", (job_id or "").strip())
    normalized_job_id = normalized_job_id.strip("._-")
    if not normalized_job_id:
        return None

    out_dir = _ASR_CHUNK_ROOT.parent / "asr_timeouts"
    if not out_dir.exists() or not out_dir.is_dir():
        return None

    fragments: list[Path] = []
    for path in out_dir.glob("timeouts_*.json"):
        if path.name.startswith("timeouts_summary_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates = {
            str(payload.get("job_id") or ""),
            str(payload.get("video_name") or ""),
            str(payload.get("video_stem") or ""),
            str(payload.get("source_job_id") or ""),
        }
        audio_path = str(payload.get("audio_path") or "")
        if normalized_job_id in candidates or f"/{normalized_job_id}/" in audio_path.replace("\\", "/"):
            fragments.append(path)

    if not fragments:
        return None

    records: list[dict] = []
    for path in sorted(fragments, key=lambda item: item.name):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            records.append({"source_file": path.name, "parse_error": repr(exc)})

    summary_path = out_dir / f"timeouts_summary_{normalized_job_id}.json"
    tmp_path = summary_path.with_name(f"{summary_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    payload = {
        "job_id": normalized_job_id,
        "count": len(records),
        "fragments": [path.name for path in sorted(fragments, key=lambda item: item.name)],
        "records": records,
    }
    with open(tmp_path, "w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2)
    tmp_path.replace(summary_path)

    for path in fragments:
        _delete_path_for_cleanup(path)

    return summary_path


def _get_wav_duration(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / rate if rate else 0.0


# Reuse the ffmpeg VAD helpers for fallback segmentation and focused unit tests.
from vad.ffmpeg_backend import (
    _group_speech_spans,
    _merge_spans,
    _run_silence_detect,
    _speech_spans_from_silences,
    _split_long_span,
)


def _build_processing_spans(audio_path: str) -> list[tuple[float, float]]:
    global _LAST_VAD_SIGNATURE
    from vad import get_vad_backend

    vad = get_vad_backend()
    result = vad.segment(audio_path)
    _LAST_VAD_SIGNATURE = result.parameters
    spans = [(g[0].start, g[-1].end) for g in result.groups]
    return spans or [(0.0, result.audio_duration_sec)]


def _record_stage_timing(
    log: list[str],
    timings: dict[str, float],
    key: str,
    label: str,
    elapsed_s: float,
) -> None:
    timings[key] = elapsed_s
    log.append(f"ASR 阶段耗时: {label}={elapsed_s:.2f}s")


def _extract_wav_chunks(
    audio_path: str,
    spans: list[tuple[float, float]],
    on_stage: Callable[[str], None] | None = None,
) -> tuple[Path, list[dict]]:
    root = _ASR_CHUNK_ROOT
    root.mkdir(parents=True, exist_ok=True)
    source_audio_path = str(Path(audio_path).resolve())
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(audio_path).stem)
    chunk_dir = root / f"{safe_prefix}_{uuid.uuid4().hex[:8]}"
    chunk_dir.mkdir(parents=True, exist_ok=False)

    chunk_infos: list[dict] = []
    with wave.open(audio_path, "rb") as reader:
        params = reader.getparams()
        frame_rate = reader.getframerate()
        total_frames = reader.getnframes()
        min_chunk_frames = max(1, int(_SEGMENT_MIN_SPEECH_S * frame_rate))
        valid_spans: list[tuple[int, int, float, float]] = []

        for start, end in spans:
            start_frame = max(0, int(start * frame_rate))
            end_frame = min(total_frames, int(end * frame_rate))
            if end_frame - start_frame < min_chunk_frames:
                continue
            valid_spans.append(
                (
                    start_frame,
                    end_frame,
                    start_frame / frame_rate,
                    end_frame / frame_rate,
                )
            )

        total_chunks = len(valid_spans)

        for idx, (start_frame, end_frame, start_time, end_time) in enumerate(
            valid_spans, 1
        ):
            if on_stage:
                on_stage(f"音频切块 {idx}/{total_chunks}...")

            reader.setpos(start_frame)
            frames = reader.readframes(end_frame - start_frame)

            chunk_path = chunk_dir / f"chunk_{idx - 1:04d}.wav"
            with wave.open(str(chunk_path), "wb") as writer:
                writer.setparams(params)
                writer.writeframes(frames)

            chunk_infos.append(
                {
                    "index": idx - 1,
                    "start": start_time,
                    "end": end_time,
                    "path": str(chunk_path),
                    "source_audio_path": source_audio_path,
                }
            )

    return chunk_dir, chunk_infos


def _build_timestamp_fallback(
    text: str,
    start: float,
    end: float,
    audio_path: str | None = None,
) -> tuple[list[dict], str, dict]:
    return build_word_timestamps_fallback(
        _clean_segment_text(text),
        start,
        end,
        audio_path=audio_path,
    )


def _looks_like_alignment_failure(
    words: list[dict],
    scene_duration_sec: float | None = None,
) -> bool:
    return looks_like_word_timing_failure(
        words,
        min_span_ms=_ALIGNMENT_MIN_SPAN_MS,
        max_zero_ratio=_ALIGNMENT_MAX_ZERO_RATIO,
        max_repeat_ratio=_ALIGNMENT_MAX_REPEAT_RATIO,
        scene_duration_sec=scene_duration_sec,
        max_coverage_ratio=_ALIGNMENT_MAX_COVERAGE_RATIO,
        max_cps=_ALIGNMENT_MAX_CPS,
    )


def _split_span_evenly(
    start: float, end: float, chunk_size: float
) -> list[tuple[float, float]]:
    if end - start <= chunk_size:
        return [(start, end)]

    spans: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + chunk_size)
        spans.append((cursor, chunk_end))
        cursor = chunk_end
    return spans


def _prepare_asr_chunk_results(
    backend: LocalAsrBackend,
    chunks: list[dict],
    text_stage_label: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[tuple[dict, list[str]]], dict[str, float]]:
    stage_started = time.perf_counter()
    text_results, text_timings = _transcribe_asr_chunks_text_only(
        backend,
        chunks,
        text_stage_label,
        on_stage=on_stage,
    )
    unload_started = time.perf_counter()
    backend.unload_model(on_stage=on_stage)
    unload_elapsed = time.perf_counter() - unload_started
    prepared, align_timings = _align_TRANSCRIPTION_results(
        backend,
        text_results,
        on_stage=on_stage,
    )
    total_elapsed = time.perf_counter() - stage_started

    return prepared, {
        "text_transcribe_s": text_timings["text_transcribe_s"],
        "asr_unload_s": unload_elapsed,
        "alignment_s": align_timings["alignment_s"],
        "total_s": total_elapsed,
    }


def _transcribe_asr_chunks_text_only(
    backend: LocalAsrBackend | SubprocessAsrBackend,
    chunks: list[dict],
    text_stage_label: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict[str, float]]:
    if not chunks:
        return [], {"text_transcribe_s": 0.0}

    request_batch_size = max(1, int(getattr(backend, "request_batch_size", 1)))
    checkpoint_source = _get_asr_checkpoint_source(chunks, text_stage_label)
    checkpoint_path = _get_asr_checkpoint_path(checkpoint_source)
    run_id = uuid.uuid4().hex[:8]
    text_results_by_index = _load_asr_checkpoint(
        checkpoint_path,
        checkpoint_source,
        chunks,
        run_id=run_id,
    )
    if text_results_by_index and on_stage:
        on_stage(f"ASR checkpoint 恢复 {len(text_results_by_index)}/{len(chunks)} 个块")

    processed_since_checkpoint = 0
    completed = False
    final_checkpoint_saved = False
    is_subprocess_backend = _is_subprocess_backend(backend)
    respawn_count: defaultdict[int, int] = defaultdict(int)
    consecutive_failures = 0
    failure_records: list[dict] = []
    text_started = time.perf_counter()

    def _save_progress_checkpoint() -> None:
        nonlocal processed_since_checkpoint
        if processed_since_checkpoint < _ASR_CHECKPOINT_INTERVAL:
            return
        _save_asr_checkpoint(
            checkpoint_path,
            checkpoint_source,
            chunks,
            _checkpointable_text_results(text_results_by_index),
            run_id=run_id,
        )
        processed_since_checkpoint = 0
        if on_stage:
            on_stage(
                f"ASR checkpoint 已保存 {len(text_results_by_index)}/{len(chunks)} 个块"
            )

    def _store_text_results(batch_chunks: list[dict], batch_text_results: list[dict]) -> None:
        nonlocal processed_since_checkpoint
        for chunk, text_result in zip(batch_chunks, batch_text_results):
            text_results_by_index[int(chunk["index"])] = text_result
        processed_since_checkpoint += len(batch_text_results)
        _save_progress_checkpoint()

    def _quarantine_chunk(chunk: dict, *, kind: str, detail: str) -> None:
        nonlocal processed_since_checkpoint
        chunk_index = int(chunk["index"])
        if chunk_index in text_results_by_index:
            return
        count = int(respawn_count[chunk_index])
        text_results_by_index[chunk_index] = _build_quarantined_text_result(
            chunk,
            kind=kind,
            detail=detail,
            respawn_count=count,
            run_id=run_id,
        )
        failure_records.append(
            {
                "index": chunk_index,
                "kind": kind,
                "detail": detail,
                "respawn_count": count,
                "run_id": run_id,
                "worker_mode": _ASR_WORKER_MODE,
            }
        )
        processed_since_checkpoint += 1

    def _schedule_retries(
        failed_chunks: list[dict],
        *,
        kind: str,
        detail: str,
    ) -> list[dict]:
        retry_chunks: list[dict] = []
        for chunk in failed_chunks:
            chunk_index = int(chunk["index"])
            if chunk_index in text_results_by_index:
                continue
            respawn_count[chunk_index] += 1
            if respawn_count[chunk_index] > _ASR_SUBPROCESS_RESPAWN_MAX:
                _quarantine_chunk(chunk, kind=kind, detail=detail)
            else:
                retry_chunks.append(chunk)
        _save_progress_checkpoint()
        return retry_chunks

    def _quarantine_many(
        failed_chunks: list[dict],
        *,
        kind: str,
        detail: str,
    ) -> None:
        for chunk in failed_chunks:
            _quarantine_chunk(chunk, kind=kind, detail=detail)
        _save_progress_checkpoint()

    def _transcribe_batch(batch_chunks: list[dict]) -> list[dict]:
        batch_contexts = [_build_ASR_CONTEXT_for_chunk(chunk) for chunk in batch_chunks]
        return backend.transcribe_texts(
            [chunk["path"] for chunk in batch_chunks],
            contexts=batch_contexts,
            on_stage=on_stage,
        )

    try:
        if not is_subprocess_backend:
            for batch_start in range(0, len(chunks), request_batch_size):
                batch_chunks = chunks[batch_start : batch_start + request_batch_size]
                pending_chunks = [
                    chunk
                    for chunk in batch_chunks
                    if int(chunk["index"]) not in text_results_by_index
                ]
                batch_end = batch_start + len(batch_chunks)
                if on_stage:
                    on_stage(f"{text_stage_label} {batch_end}/{len(chunks)}...")
                if not pending_chunks:
                    continue

                batch_text_results = _transcribe_batch(pending_chunks)
                _store_text_results(pending_chunks, batch_text_results)
        else:
            pending_chunks = [
                chunk for chunk in chunks if int(chunk["index"]) not in text_results_by_index
            ]
            while pending_chunks:
                batch_chunks = pending_chunks[:request_batch_size]
                pending_chunks = pending_chunks[request_batch_size:]
                if on_stage:
                    on_stage(
                        f"{text_stage_label} {min(len(text_results_by_index) + len(batch_chunks), len(chunks))}/{len(chunks)}..."
                    )

                try:
                    batch_text_results = _transcribe_batch(batch_chunks)
                except WorkerTimeoutError as exc:
                    consecutive_failures += 1
                    retry_chunks = _schedule_retries(
                        batch_chunks,
                        kind="timeout",
                        detail=str(getattr(exc, "detail", str(exc))),
                    )
                    if (
                        consecutive_failures
                        >= _ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT
                    ):
                        _quarantine_many(
                            retry_chunks + pending_chunks,
                            kind="timeout",
                            detail="circuit breaker",
                        )
                        pending_chunks = []
                        break
                    pending_chunks = retry_chunks + pending_chunks
                    continue
                except WorkerError as exc:
                    failure_kind = str(getattr(exc, "kind", "crash") or "crash")
                    failure_detail = str(getattr(exc, "detail", str(exc)) or str(exc))
                    if failure_kind == "oom":
                        retry_chunks: list[dict] = []
                        for chunk in batch_chunks:
                            chunk_index = int(chunk["index"])
                            if chunk_index in text_results_by_index:
                                continue
                            respawn_count[chunk_index] += 1
                            try:
                                single_result = _transcribe_batch([chunk])
                            except WorkerTimeoutError as single_exc:
                                consecutive_failures += 1
                                retry_chunks.extend(
                                    _schedule_retries(
                                        [chunk],
                                        kind="timeout",
                                        detail=str(
                                            getattr(
                                                single_exc,
                                                "detail",
                                                str(single_exc),
                                            )
                                        ),
                                    )
                                )
                            except WorkerError as single_exc:
                                single_kind = str(
                                    getattr(single_exc, "kind", "crash") or "crash"
                                )
                                single_detail = str(
                                    getattr(single_exc, "detail", str(single_exc))
                                    or str(single_exc)
                                )
                                if single_kind == "oom":
                                    respawn_count[chunk_index] += 1
                                    _quarantine_chunk(
                                        chunk,
                                        kind="oom",
                                        detail=single_detail,
                                    )
                                    _save_progress_checkpoint()
                                elif single_kind in {"crash", "protocol_error"}:
                                    consecutive_failures += 1
                                    retry_chunks.extend(
                                        _schedule_retries(
                                            [chunk],
                                            kind=single_kind,
                                            detail=single_detail,
                                        )
                                    )
                                else:
                                    raise
                            else:
                                _store_text_results([chunk], single_result)
                                consecutive_failures = 0

                        if (
                            consecutive_failures
                            >= _ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT
                        ):
                            _quarantine_many(
                                retry_chunks + pending_chunks,
                                kind="timeout",
                                detail="circuit breaker",
                            )
                            pending_chunks = []
                            break
                        pending_chunks = retry_chunks + pending_chunks
                        continue

                    if failure_kind in {"crash", "protocol_error"}:
                        consecutive_failures += 1
                        retry_chunks = _schedule_retries(
                            batch_chunks,
                            kind=failure_kind,
                            detail=failure_detail,
                        )
                        if (
                            consecutive_failures
                            >= _ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT
                        ):
                            _quarantine_many(
                                retry_chunks + pending_chunks,
                                kind=failure_kind,
                                detail="circuit breaker",
                            )
                            pending_chunks = []
                            break
                        pending_chunks = retry_chunks + pending_chunks
                        continue
                    raise
                else:
                    _store_text_results(batch_chunks, batch_text_results)
                    consecutive_failures = 0

        timeout_count = sum(
            1 for result in text_results_by_index.values() if _is_timed_out_result(result)
        )
        if failure_records:
            quarantine_paths = _quarantine_failed_chunks(
                checkpoint_source,
                chunks,
                failure_records,
                run_id=run_id,
                worker_mode=_ASR_WORKER_MODE,
            )
            if quarantine_paths and on_stage:
                on_stage(f"ASR quarantine 已写入 {len(quarantine_paths)} 个记录")
        checkpointable_results = _checkpointable_text_results(text_results_by_index)
        completed = len(checkpointable_results) >= len(chunks)
        if timeout_count:
            if on_stage:
                on_stage(
                    f"[WARN] 本轮 {timeout_count} 个块超时跳过，已从 checkpoint 中排除，下次续跑将重试"
                )
            _save_asr_checkpoint(
                checkpoint_path,
                checkpoint_source,
                chunks,
                checkpointable_results,
                run_id=run_id,
            )
            final_checkpoint_saved = True
        if completed:
            _delete_path_for_cleanup(checkpoint_path)
    finally:
        if (
            _ASR_CHECKPOINT_ENABLED
            and not completed
            and not final_checkpoint_saved
            and processed_since_checkpoint > 0
        ):
            _save_asr_checkpoint(
                checkpoint_path,
                checkpoint_source,
                chunks,
                _checkpointable_text_results(text_results_by_index),
                run_id=run_id,
            )

    if is_subprocess_backend:
        missing_indices = [
            int(chunk["index"])
            for chunk in chunks
            if int(chunk["index"]) not in text_results_by_index
        ]
        if missing_indices:
            raise ASRWorkerSystemError(
                f"subprocess ASR missing text results for chunks: {missing_indices[:10]}"
            )
        text_results = [text_results_by_index[int(chunk["index"])] for chunk in chunks]
    else:
        text_results = [
            text_results_by_index[int(chunk["index"])]
            for chunk in chunks
            if int(chunk["index"]) in text_results_by_index
        ]

    text_elapsed = time.perf_counter() - text_started
    return text_results, {"text_transcribe_s": text_elapsed}

def _is_empty_segment_text_result(text_result: dict) -> bool:
    if "segments" not in text_result:
        return False
    if text_result.get("segments"):
        return False
    text = str(text_result.get("text") or text_result.get("raw_text") or "").strip()
    return not text


def _empty_alignment_placeholder(
    text_result: dict,
    *,
    reason: str = "Alignment skipped: empty segments placeholder",
) -> tuple[dict, list[str]]:
    log = list(text_result.get("log", []))
    if reason and reason not in log:
        log.append(reason)
    try:
        duration = float(text_result.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return (
        {
            "words": [],
            "text": str(text_result.get("text") or "").strip(),
            "raw_text": str(text_result.get("raw_text") or "").strip(),
            "alignment_mode": "empty",
            "duration": duration,
            "language": str(text_result.get("language") or "Japanese").strip()
            or "Japanese",
        },
        log,
    )


def _empty_segments_quarantine_placeholder() -> tuple[dict, list[str]]:
    return _empty_alignment_placeholder(
        {
            "text": "",
            "raw_text": "",
            "duration": 0.0,
            "language": "Japanese",
            "normalized_path": "",
            "segments": [],
            "log": [
                (
                    "QUARANTINED: "
                    "kind=empty_segment, respawn_count=0, "
                    "run_id=, detail=empty alignment input"
                )
            ],
        }
    )


def _align_TRANSCRIPTION_results(
    backend: BaseAsrBackend,
    text_results: list[dict],
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[tuple[dict, list[str]]], dict[str, float]]:
    if not text_results:
        return [_empty_segments_quarantine_placeholder()], {"alignment_s": 0.0}

    align_started = time.perf_counter()
    prepared: list[tuple[dict, list[str]] | None] = [None] * len(text_results)
    pending_results: list[dict] = []
    pending_indices: list[int] = []
    for idx, text_result in enumerate(text_results):
        if _is_empty_segment_text_result(text_result):
            prepared[idx] = _empty_alignment_placeholder(text_result)
        else:
            pending_results.append(text_result)
            pending_indices.append(idx)

    if not pending_results:
        return [item for item in prepared if item is not None], {
            "alignment_s": time.perf_counter() - align_started
        }

    if _is_subprocess_backend(backend) or getattr(backend, "model", None) is not None:
        backend.unload_model(on_stage=on_stage)

    finalize_batch_size = max(1, int(getattr(backend, "align_batch_size", 1)))

    for batch_start in range(0, len(pending_results), finalize_batch_size):
        batch_results = pending_results[batch_start : batch_start + finalize_batch_size]
        batch_indices = pending_indices[batch_start : batch_start + finalize_batch_size]
        batch_end = batch_start + len(batch_results)
        if on_stage:
            on_stage(f"Alignment 对齐 {batch_end}/{len(pending_results)}...")
        finalized_batch = backend.finalize_text_results(
            batch_results,
            on_stage=on_stage,
        )
        for result_index, finalized in zip(batch_indices, finalized_batch):
            prepared[result_index] = finalized
        if len(finalized_batch) < len(batch_results):
            for missing_index in batch_indices[len(finalized_batch) :]:
                prepared[missing_index] = _empty_alignment_placeholder(
                    text_results[missing_index],
                    reason="Alignment skipped: backend returned no finalize result",
                )

    align_elapsed = time.perf_counter() - align_started
    return [
        item
        if item is not None
        else _empty_alignment_placeholder(
            text_results[idx],
            reason="Alignment skipped: missing finalize result",
        )
        for idx, item in enumerate(prepared)
    ], {"alignment_s": align_elapsed}


def _run_TRANSCRIPTION_qc(
    backend: BaseAsrBackend,
    chunks: list[dict],
    text_results: list[dict],
    log: list[str],
    on_stage: Callable[[str], None] | None = None,
) -> tuple[dict, dict[str, float]]:
    if not ASR_QC_ENABLED:
        return {
            "enabled": False,
            "chunk_count": len(chunks),
            "recoverable_count": 0,
            "warning_count": 0,
            "items": [],
            "recoverable_indices": [],
        }, {"asr_qc_s": 0.0}

    if on_stage:
        on_stage("ASR 质检分析...")

    qc_started = time.perf_counter()
    qc_report = evaluate_asr_text_results_qc(
        chunks,
        text_results,
        is_low_value_text=_is_low_value_text,
        is_context_leak=_is_context_leak,
        backend=backend,
    )
    qc_elapsed = time.perf_counter() - qc_started

    log.append(
        "[qc] context_leak_check={context} repetition_check={repetition} threshold={threshold}".format(
            context=qc_report.get("context_leak_check", "on"),
            repetition=qc_report.get("repetition_check", "on"),
            threshold=qc_report.get("repetition_threshold", 10),
        )
    )
    log.append(
        "ASR QC: recoverable={recoverable}/{total}, warning={warning}".format(
            recoverable=qc_report["recoverable_count"],
            total=qc_report["chunk_count"],
            warning=qc_report["warning_count"],
        )
    )
    log.extend(format_qc_log_items(qc_report))
    if qc_report["recoverable_count"] and not ASR_RECOVERY_ENABLED:
        log.append(
            "ASR Recovery Path: disabled; QC-only prototype keeps original ASR text"
        )

    return qc_report, {"asr_qc_s": qc_elapsed}


def _recover_TRANSCRIPTION_results_if_needed(
    backend: LocalAsrBackend,
    chunks: list[dict],
    text_results: list[dict],
    qc_report: dict,
    log: list[str],
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict[str, float]]:
    recoverable_indices = list(qc_report.get('recoverable_indices', []))
    if not ASR_RECOVERY_ENABLED or not recoverable_indices:
        return text_results, {'asr_recovery_s': 0.0, 'asr_recovered_chunks': 0.0}

    recovery_started = time.perf_counter()
    if on_stage:
        on_stage(f'ASR Recovery Path 调度 {len(recoverable_indices)} 个异常块...')

    timings: dict[str, float] = {
        'asr_recovery_s': 0.0,
        'asr_recovered_chunks': 0.0,
        'asr_recovery_retranscribe_s': 0.0,
    }

    from vad.ffmpeg_backend import FfmpegSilencedetectBackend
    from audio.vad_refine import refine_chunks_via_vad

    recoverable_chunks = [chunks[i] for i in recoverable_indices]
    sub_chunk_paths: list[str] = []
    try:
        refined = refine_chunks_via_vad(
            recoverable_chunks,
            vad_backend=FfmpegSilencedetectBackend(),
            timeout_per_chunk_s=float(os.getenv('ASR_RECOVERY_TIMEOUT_S', '30')),
        )
        sub_chunk_paths = [
            c['path'] for c in refined
            if c['path'] not in {rc['path'] for rc in recoverable_chunks}
        ]
    except Exception as exc:
        log.append(f'ASR Recovery VAD refine failed: {exc}; kept original ASR text')
        timings['asr_recovery_s'] = time.perf_counter() - recovery_started
        return text_results, timings

    log.append(
        f'ASR Recovery VAD: {len(recoverable_chunks)} chunks -> {len(refined)} sub-chunks'
    )

    retranscribe_started = time.perf_counter()
    try:
        flat_results, text_timings = _transcribe_asr_chunks_text_only(
            backend,
            refined,
            'ASR Recovery 重转写',
            on_stage=on_stage,
        )
        timings['asr_recovery_retranscribe_s'] = text_timings.get('text_transcribe_s', 0.0)
    except Exception as exc:
        log.append(f'ASR Recovery retranscribe failed: {exc}; kept original ASR text')
        timings['asr_recovery_s'] = time.perf_counter() - recovery_started
        return text_results, timings

    if not timings['asr_recovery_retranscribe_s']:
        timings['asr_recovery_retranscribe_s'] = time.perf_counter() - retranscribe_started

    groups: dict[int, list[dict]] = {}
    for sub_chunk, sub_result in zip(refined, flat_results):
        parent = int(sub_chunk.get('_vad_parent_index', sub_chunk.get('index', 0)))
        groups.setdefault(parent, []).append(sub_result)

    updated_results = list(text_results)
    recovered_count = 0
    for position in recoverable_indices:
        group = groups.get(position)
        if not group:
            continue
        original_result = text_results[position]
        merged_text = ''.join(
            r.get('text', '').strip() for r in group if r.get('text', '').strip()
        )
        merged_result = dict(group[0])
        merged_result['text'] = merged_text
        result_log = list(merged_result.get('log', []))
        result_log.append('ASR Recovery VAD: text merged from sub-chunk retranscription')
        merged_result['log'] = result_log
        try:
            qc_reasons = qc_report['items'][position].get('reasons', [])
        except (KeyError, IndexError, TypeError):
            qc_reasons = []
        merged_result['recovery'] = {
            'enabled': True,
            'method': 'vad_refine',
            'sub_chunks': len(group),
            'original_text': original_result.get('text', ''),
            'qc_reasons': qc_reasons,
        }
        updated_results[position] = merged_result
        log.append(
            'ASR Recovery VAD replaced chunk {index}: {before} -> {after}'.format(
                index=chunks[position].get('index', position + 1),
                before=(original_result.get('text', '') or '')[:80],
                after=(merged_text or '')[:80],
            )
        )
        recovered_count += 1

    for sub_path in sub_chunk_paths:
        try:
            Path(sub_path).unlink(missing_ok=True)
        except Exception:
            pass

    timings['asr_recovered_chunks'] = float(recovered_count)
    timings['asr_recovery_s'] = time.perf_counter() - recovery_started
    return updated_results, timings


def _build_transcript_chunks(
    chunks: list[dict], text_results: list[dict]
) -> list[dict]:
    transcript_chunks: list[dict] = []
    for chunk, text_result in zip(chunks, text_results):
        transcript_chunks.append(
            {
                "index": chunk["index"],
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "duration": float(text_result.get("duration", 0.0)),
                "language": text_result.get("language", ""),
                "text": text_result.get("text", ""),
                "raw_text": text_result.get("raw_text", ""),
            }
        )
    return transcript_chunks


def _build_ASR_CONTEXT_for_chunk(chunk: dict) -> str:
    parts: list[str] = []
    chunk_start = float(chunk.get("start", 0.0))
    if _ASR_HEAD_CONTEXT and chunk_start <= _ASR_HEAD_CONTEXT_MAX_START_S:
        parts.append(_ASR_HEAD_CONTEXT)
    if _ASR_CONTEXT:
        parts.append(_ASR_CONTEXT)
    return "\n".join(part for part in parts if part)


def _should_skip_alignment_retry(text: str) -> bool:
    cleaned = _clean_segment_text(text)
    compact = _strip_punctuation(cleaned)
    if not compact:
        return True
    if len(compact) <= _ALIGNMENT_RETRY_SKIP_MAX_TEXT_LEN:
        return True
    return _is_low_value_text(cleaned)


def _needs_alignment_fallback(
    words: list[dict],
    text: str,
    scene_duration_sec: float | None = None,
) -> bool:
    compact_len = len(_strip_punctuation(text))
    if compact_len > 0 and not words:
        return True
    if compact_len >= 18 and len(words) <= 1:
        return True
    return _looks_like_alignment_failure(words, scene_duration_sec=scene_duration_sec)


def _finalize_aligned_chunk_without_asr_retry(
    chunk: dict,
    chunk_result: dict,
    chunk_log: list[str],
) -> tuple[list[dict], list[str]]:
    chunk_words = list(chunk_result.get("words", []))
    text = chunk_result.get("text", "")
    start = float(chunk["start"])
    end = float(chunk["end"])
    duration = max(0.0, end - start)

    if not _needs_alignment_fallback(chunk_words, text, scene_duration_sec=duration):
        return chunk_words, chunk_log

    chunk_log.append(
        "Alignment 哨兵触发: 时间轴异常，不重新调用 ASR，改用 VAD/比例回退"
    )
    fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
        text,
        0.0,
        duration,
        audio_path=chunk["path"],
    )
    if fallback_mode == "aligner_vad_fallback":
        chunk_log.append("Alignment 回退: 使用 VAD 约束比例时间戳")
        chunk_log.append(
            f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
        )
    else:
        chunk_log.append("Alignment 回退: 使用等比分配时间戳")
        if fallback_meta.get("vad_error"):
            chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
    return fallback_words, chunk_log


def _refine_chunk_with_subchunks(
    backend: LocalAsrBackend,
    source_audio_path: str,
    chunk: dict,
    subchunk_size: float,
    retry_depth: int,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[str]]:
    start = float(chunk["start"])
    end = float(chunk["end"])
    refine_spans = _split_span_evenly(start, end, subchunk_size)
    if len(refine_spans) <= 1:
        return [], []

    refine_dir, refine_infos = _extract_wav_chunks(source_audio_path, refine_spans)
    refine_words: list[dict] = []
    refine_log: list[str] = []
    try:
        prepared_results, _stage_timings = _prepare_asr_chunk_results(
            backend,
            refine_infos,
            "Alignment 局部细化",
            on_stage=on_stage,
        )
        for refine_idx, refine_chunk, (initial_chunk_result, initial_chunk_log) in zip(
            range(1, len(refine_infos) + 1),
            refine_infos,
            prepared_results,
        ):
            sub_words, sub_log = _transcribe_asr_chunk_with_retry(
                backend,
                source_audio_path,
                refine_chunk,
                on_stage=on_stage,
                retry_depth=retry_depth + 1,
                initial_chunk_result=initial_chunk_result,
                initial_chunk_log=initial_chunk_log,
            )
            refine_log.extend(
                f"refine {refine_idx}: {line}"
                for line in sub_log
                if line.startswith(
                    (
                        "ASR 时间戳策略",
                        "ASR 原生时间戳词数",
                        "ASR 原生时间戳异常",
                        "ASR 原生时间戳需局部细化",
                        "Alignment 模式",
                        "Alignment 异常",
                        "Alignment 哨兵",
                        "Alignment 降级",
                        "Alignment VAD 回退",
                    )
                )
            )
            for word in sub_words:
                refine_words.append(
                    {
                        "start": float(word["start"]) + refine_chunk["start"] - start,
                        "end": float(word["end"]) + refine_chunk["start"] - start,
                        "word": word["word"],
                    }
                )
    finally:
        if refine_dir.exists() and not _KEEP_ASR_CHUNKS:
            _delete_path_for_cleanup(refine_dir)

    refine_words.sort(key=lambda item: (item["start"], item["end"]))
    return refine_words, refine_log


def _transcribe_asr_chunk_with_retry(
    backend: LocalAsrBackend,
    source_audio_path: str,
    chunk: dict,
    on_stage: Callable[[str], None] | None = None,
    retry_depth: int = 0,
    initial_chunk_result: dict | None = None,
    initial_chunk_log: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    if initial_chunk_result is None:
        chunk_result, chunk_log = backend.transcribe_to_words(
            chunk["path"], on_stage=on_stage
        )
    else:
        chunk_result = initial_chunk_result
        chunk_log = list(initial_chunk_log or [])
    chunk_words = list(chunk_result.get("words", []))
    chunk_mode = str(chunk_result.get("alignment_mode", "")).strip()
    start = float(chunk["start"])
    end = float(chunk["end"])
    duration = max(0.0, end - start)

    if (
        chunk_mode == "ASR_NATIVE_refine_needed"
        and retry_depth < _ALIGNMENT_MAX_REFINE_DEPTH
        and duration > _ALIGNMENT_COARSE_REFINE_CHUNK_S
    ):
        chunk_log.append(
            "ASR 原生时间戳需局部细化: 长块不直接整块强制对齐，先拆为中等片段"
        )
        refined_words, refined_log = _refine_chunk_with_subchunks(
            backend,
            source_audio_path,
            chunk,
            _ALIGNMENT_COARSE_REFINE_CHUNK_S,
            retry_depth,
            on_stage=on_stage,
        )
        chunk_log.extend(refined_log)
        if refined_words:
            return refined_words, chunk_log

    if not _looks_like_alignment_failure(chunk_words, scene_duration_sec=duration):
        return chunk_words, chunk_log

    if chunk_mode == "forced_aligner":
        chunk_log.append("Alignment 哨兵触发: forced 模式保留原文，不再对子片段重转写")
        return chunk_words, chunk_log

    if _should_skip_alignment_retry(chunk_result.get("text", "")):
        chunk_log.append("Alignment 哨兵触发: 低价值/短文本直接回退，不再做子片段重试")
        fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
            chunk_result.get("text", ""),
            0.0,
            duration,
            audio_path=chunk["path"],
        )
        if fallback_mode == "aligner_vad_fallback":
            chunk_log.append("Alignment 快速回退: 使用 VAD 约束比例时间戳")
            chunk_log.append(
                f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
            )
        else:
            chunk_log.append("Alignment 快速回退: 使用等比分配时间戳")
            if fallback_meta.get("vad_error"):
                chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
        return fallback_words, chunk_log

    if retry_depth < _ALIGNMENT_MAX_REFINE_DEPTH and duration > _ALIGNMENT_COARSE_REFINE_CHUNK_S:
        chunk_log.append("Alignment 哨兵触发: 长块先细化为中等片段重试")
        refined_words, refined_log = _refine_chunk_with_subchunks(
            backend,
            source_audio_path,
            chunk,
            _ALIGNMENT_COARSE_REFINE_CHUNK_S,
            retry_depth,
            on_stage=on_stage,
        )
        chunk_log.extend(refined_log)
        if refined_words and not _looks_like_alignment_failure(
            refined_words,
            scene_duration_sec=duration,
        ):
            chunk_log.append(f"Alignment 长块细化成功: {len(refined_words)} 词")
            return refined_words, chunk_log

    chunk_log.append(
        "Alignment 哨兵触发: 检测到异常密集/重复时间戳，准备切为更小片段重试"
    )
    retry_spans = _split_span_evenly(start, end, _ALIGNMENT_STEP_DOWN_CHUNK_S)
    if len(retry_spans) <= 1:
        fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
            chunk_result.get("text", ""),
            0.0,
            duration,
            audio_path=chunk["path"],
        )
        if fallback_mode == "aligner_vad_fallback":
            chunk_log.append("Alignment 降级失败: 子片段不足，改用 VAD 约束比例时间戳")
            chunk_log.append(
                f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
            )
        else:
            chunk_log.append("Alignment 降级失败: 子片段不足，改用等比分配时间戳")
            if fallback_meta.get("vad_error"):
                chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
        return fallback_words, chunk_log

    retry_dir, retry_infos = _extract_wav_chunks(source_audio_path, retry_spans)
    retry_words: list[dict] = []
    try:
        prepared_results, _stage_timings = _prepare_asr_chunk_results(
            backend,
            retry_infos,
            "Alignment 降级重试",
            on_stage=on_stage,
        )
        for retry_idx, retry_chunk, (retry_result, retry_log) in zip(
            range(1, len(retry_infos) + 1),
            retry_infos,
            prepared_results,
        ):
            chunk_log.extend(
                f"retry {retry_idx}: {line}"
                for line in retry_log
                if line.startswith(("Alignment 模式", "Alignment 异常"))
            )
            for word in retry_result.get("words", []):
                retry_words.append(
                    {
                        "start": float(word["start"]) + retry_chunk["start"] - start,
                        "end": float(word["end"]) + retry_chunk["start"] - start,
                        "word": word["word"],
                    }
                )
    finally:
        if retry_dir.exists() and not _KEEP_ASR_CHUNKS:
            _delete_path_for_cleanup(retry_dir)

    retry_words.sort(key=lambda item: (item["start"], item["end"]))
    if retry_words and not _looks_like_alignment_failure(
        retry_words,
        scene_duration_sec=duration,
    ):
        chunk_log.append(f"Alignment 降级成功: {len(retry_infos)} 个子片段")
        return retry_words, chunk_log

    fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
        chunk_result.get("text", ""),
        0.0,
        duration,
        audio_path=chunk["path"],
    )
    if fallback_mode == "aligner_vad_fallback":
        chunk_log.append("Alignment 降级后仍异常: 改用 VAD 约束比例时间戳")
        chunk_log.append(
            f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
        )
    else:
        chunk_log.append("Alignment 降级后仍异常: 改用等比分配时间戳")
        if fallback_meta.get("vad_error"):
            chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
    return fallback_words, chunk_log


def _postprocess_segments(segments: list[dict]) -> list[dict]:
    cleaned_segments: list[dict] = []

    for segment in segments:
        text = _clean_segment_text(segment.get("text", ""))
        if not text or _TOOL_SIGNATURE_RE.search(text):
            continue
        text = _remove_context_leak_fragments(text)
        if not text:
            continue
        if _is_context_leak(text):
            continue
        if _is_low_value_text(text):
            continue

        cleaned_segments.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": text,
                "source_chunk_index": segment.get("source_chunk_index"),
                "words": list(segment.get("words") or []),
            }
        )

    cleaned_segments.sort(key=lambda item: (item["start"], item["end"]))
    merged_segments = _merge_fragment_segments(cleaned_segments)
    split_segments = _split_long_postprocessed_segments(merged_segments)
    return _repair_postprocessed_segment_windows(split_segments)


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_TERMINAL_RE.search((text or "").strip()))


def _same_source_chunk(current: dict, following: dict) -> bool:
    current_chunk = current.get("source_chunk_index")
    following_chunk = following.get("source_chunk_index")
    return (
        current_chunk is not None
        and following_chunk is not None
        and current_chunk == following_chunk
    )


def _should_merge_fragment(current: dict, following: dict) -> bool:
    current_text = str(current.get("text", "")).strip()
    following_text = str(following.get("text", "")).strip()
    if not current_text or not following_text:
        return False
    if not _same_source_chunk(current, following):
        return False
    if _ends_sentence(current_text):
        return False

    gap = float(following["start"]) - float(current["end"])
    if gap < -0.05 or gap > _ASR_FRAGMENT_MERGE_MAX_GAP_S:
        return False

    combined_chars = len(_compact_context_text(current_text + following_text))
    if combined_chars > _ASR_FRAGMENT_MERGE_MAX_CHARS:
        return False

    combined_duration = float(following["end"]) - float(current["start"])
    if combined_duration > _ASR_FRAGMENT_MERGE_MAX_DURATION_S:
        return False

    return True


def _join_segment_text(left: str, right: str) -> str:
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    if _FRAGMENT_CONTINUATION_START_RE.match(right):
        return left + right
    return left + right


def _merge_fragment_segments(segments: list[dict]) -> list[dict]:
    if len(segments) < 2:
        return segments

    merged: list[dict] = []
    current = dict(segments[0])

    for following in segments[1:]:
        if _should_merge_fragment(current, following):
            current["end"] = float(following["end"])
            current["text"] = _join_segment_text(current.get("text", ""), following.get("text", ""))
            current["words"] = list(current.get("words") or []) + list(
                following.get("words") or []
            )
            continue

        merged.append(current)
        current = dict(following)

    merged.append(current)
    return merged


def _pick_postprocess_split_index(text: str) -> int:
    text = (text or "").strip()
    if len(text) < _ASR_POSTPROCESS_SPLIT_MIN_CHARS * 2:
        return 0

    target = len(text) // 2
    split_chars = "。！？!?…、,，・ "
    candidates = [
        idx + 1
        for idx, char in enumerate(text[:-1])
        if char in split_chars
        and _ASR_POSTPROCESS_SPLIT_MIN_CHARS <= idx + 1
        and len(text) - (idx + 1) >= _ASR_POSTPROCESS_SPLIT_MIN_CHARS
    ]
    if candidates:
        return min(candidates, key=lambda idx: abs(idx - target))

    return target


def _split_long_postprocessed_segment(segment: dict) -> list[dict]:
    text = str(segment.get("text", "")).strip()
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    duration = max(0.0, end - start)
    compact_len = len(_compact_context_text(text))
    original_words = list(segment.get("words") or [])

    if (
        compact_len <= _ASR_POSTPROCESS_MAX_CHARS
        and duration <= _ASR_POSTPROCESS_MAX_DURATION_S
    ):
        return [segment]

    split_idx = _pick_postprocess_split_index(text)
    if not split_idx:
        return [segment]

    left_text = text[:split_idx].strip()
    right_text = text[split_idx:].strip()
    if not left_text or not right_text:
        return [segment]

    left_weight = max(1, len(_compact_context_text(left_text)))
    right_weight = max(1, len(_compact_context_text(right_text)))
    split_time = start + duration * (left_weight / (left_weight + right_weight))
    if (
        split_time - start <= _ASR_INVALID_SEGMENT_DURATION_S
        or end - split_time <= _ASR_INVALID_SEGMENT_DURATION_S
    ):
        return [segment]

    left_words = [
        word
        for word in original_words
        if float(word.get("start", start)) < split_time
    ]
    right_words = [
        word
        for word in original_words
        if float(word.get("start", start)) >= split_time
    ]
    assert len(left_words) + len(right_words) == len(original_words)

    left_segment = {
        "start": start,
        "end": split_time,
        "text": left_text,
        "source_chunk_index": segment.get("source_chunk_index"),
        "words": left_words,
    }
    right_segment = {
        "start": split_time,
        "end": end,
        "text": right_text,
        "source_chunk_index": segment.get("source_chunk_index"),
        "words": right_words,
    }
    return (
        _split_long_postprocessed_segment(left_segment)
        + _split_long_postprocessed_segment(right_segment)
    )


def _split_long_postprocessed_segments(segments: list[dict]) -> list[dict]:
    split_segments: list[dict] = []
    for segment in segments:
        split_segments.extend(_split_long_postprocessed_segment(dict(segment)))
    return split_segments


def _repair_postprocessed_segment_windows(segments: list[dict]) -> list[dict]:
    if not segments:
        return []

    repaired: list[dict] = []
    epsilon = 0.01
    for idx, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = max(0.0, float(segment.get("start", 0.0)))
        end = max(start, float(segment.get("end", start)))
        if repaired and start < float(repaired[-1]["end"]):
            start = float(repaired[-1]["end"])
            end = max(end, start)

        duration = end - start
        if duration <= _ASR_INVALID_SEGMENT_DURATION_S:
            target_end = start + _ASR_MIN_REPAIRED_SEGMENT_DURATION_S
            if idx + 1 < len(segments):
                next_start = float(segments[idx + 1].get("start", target_end))
                if next_start > start:
                    target_end = min(target_end, max(start, next_start - epsilon))

            if target_end - start <= _ASR_INVALID_SEGMENT_DURATION_S:
                if (
                    repaired
                    and start - float(repaired[-1]["end"])
                    <= _ASR_FRAGMENT_MERGE_MAX_GAP_S
                    and _same_source_chunk(repaired[-1], segment)
                ):
                    repaired[-1]["text"] = _join_segment_text(
                        str(repaired[-1].get("text", "")),
                        text,
                    )
                    repaired[-1]["end"] = max(
                        float(repaired[-1]["end"]),
                        end,
                        start + _ASR_MIN_REPAIRED_SEGMENT_DURATION_S,
                    )
                    repaired[-1]["words"] = list(repaired[-1].get("words") or []) + list(
                        segment.get("words") or []
                    )
                    continue
                target_end = start + _ASR_MIN_REPAIRED_SEGMENT_DURATION_S

            end = max(end, target_end)

        repaired.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "source_chunk_index": segment.get("source_chunk_index"),
                "words": list(segment.get("words") or []),
            }
        )

    repaired.sort(key=lambda item: (item["start"], item["end"]))
    non_overlapping: list[dict] = []
    for segment in repaired:
        start = float(segment["start"])
        end = float(segment["end"])
        if non_overlapping and start < float(non_overlapping[-1]["end"]):
            start = float(non_overlapping[-1]["end"])
        if end - start <= _ASR_INVALID_SEGMENT_DURATION_S:
            end = start + _ASR_MIN_REPAIRED_SEGMENT_DURATION_S
        non_overlapping.append(
            {
                "start": start,
                "end": end,
                "text": str(segment["text"]).strip(),
                "source_chunk_index": segment.get("source_chunk_index"),
                "words": list(segment.get("words") or []),
            }
        )

    return non_overlapping


def _should_split_on_gender(prev_word, next_word, gap, *, min_gap_s=0.15):
    pg, ng = prev_word.get('gender'), next_word.get('gender')
    return pg is not None and ng is not None and pg != ng and gap >= min_gap_s


def _merge_words_to_segments(words: list[dict]) -> list[dict]:
    if not words:
        return []

    japanese_sentence_end = {"。", "！", "？", "…", "!", "?"}
    soft_max_chars = 45
    hard_max_chars = 60
    max_duration = 8.5
    hard_max_duration = float(os.getenv("ASR_MERGE_HARD_MAX_DURATION", "9.0"))
    max_gap = 1.2

    segments: list[dict] = []
    current_words: list[dict] = []

    def _pick_sentence_split(word_list: list[dict]) -> int:
        if len(word_list) < 2:
            return 0

        total_text = "".join(w["word"] for w in word_list).strip()
        total_len = len(total_text)
        segment_start = word_list[0]["start"]
        running_len = 0
        candidates: list[tuple[float, int]] = []

        for idx, item in enumerate(word_list[:-1], 1):
            token = item["word"]
            running_len += len(token)
            if not token or token[-1] not in japanese_sentence_end:
                continue

            left_duration = item["end"] - segment_start
            right_words = word_list[idx:]
            right_text_len = total_len - running_len
            right_duration = (
                word_list[-1]["end"] - right_words[0]["start"] if right_words else 0.0
            )

            if running_len < 8 or left_duration < 1.0:
                continue

            score = abs(running_len - soft_max_chars)
            if right_text_len < 6:
                score += 100
            elif right_text_len < 10:
                score += 12
            if right_duration < 0.5:
                score += 40
            if running_len > hard_max_chars:
                score += 25

            candidates.append((score, idx))

        if not candidates:
            return 0

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def flush(word_list: list[dict]) -> None:
        if not word_list:
            return
        text = "".join(w["word"] for w in word_list).strip()
        if not text:
            return
        segment_duration = word_list[-1]["end"] - word_list[0]["start"]
        if len(text) > soft_max_chars or segment_duration > max_duration:
            split_idx = _pick_sentence_split(word_list)
            if split_idx:
                flush(word_list[:split_idx])
                flush(word_list[split_idx:])
                return
        if text in _NOISE_WORDS:
            return
        if _TRIVIAL_SEGMENT.match(text):
            return
        if text in _GRAY_WORDS and segment_duration > _GRAY_MAX_DURATION_S:
            return
        segments.append(
            {
                "start": word_list[0]["start"],
                "end": word_list[-1]["end"],
                "text": text,
                "source_chunk_index": word_list[0].get("source_chunk_index"),
                "words": list(word_list),
            }
        )

    for word in words:
        if not current_words:
            current_words.append(word)
            continue

        segment_duration = current_words[-1]["end"] - current_words[0]["start"]
        segment_text = "".join(w["word"] for w in current_words)
        gap = word["start"] - current_words[-1]["end"]
        ends_sentence = bool(segment_text) and segment_text[-1] in japanese_sentence_end
        projected_text = segment_text + word["word"]
        projected_duration = word["end"] - current_words[0]["start"]
        current_chunk = current_words[-1].get("source_chunk_index")
        next_chunk = word.get("source_chunk_index")
        crosses_chunk_boundary = (
            current_chunk is not None
            and next_chunk is not None
            and current_chunk != next_chunk
        )
        should_split_turn = (
            (
                ends_sentence
                and segment_duration >= 0.8
                and len(segment_text) >= 4
            )
            or (ends_sentence and gap > 0.05)
            or (
                gap > 0.35
                and segment_duration >= 0.8
                and len(segment_text) >= 4
            )
            or (
                gap > 0.2
                and segment_duration >= 1.2
                and len(segment_text) >= 8
            )
            or (
                segment_duration >= 4.0
                and len(segment_text) >= 20
            )
        )
        should_split_turn = should_split_turn or _should_split_on_gender(current_words[-1], word, gap)
        exceeds_segment_limits = (
            (
                ends_sentence
                and (
                    gap > max_gap
                    or segment_duration >= max_duration
                    or len(segment_text) >= soft_max_chars
                )
            )
            or len(projected_text) > hard_max_chars
            or projected_duration >= hard_max_duration
        )

        if (
            crosses_chunk_boundary
            or should_split_turn
            or exceeds_segment_limits
        ):
            flush(current_words)
            current_words = [word]
        else:
            current_words.append(word)

    flush(current_words)
    return segments


def _transcribe_and_align_local(
    audio_path: str,
    device: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[str], dict]:
    def _notify(message: str) -> None:
        if on_stage:
            on_stage(message)

    log: list[str] = [f"ASR backend: {get_backend_label()}"]
    timings: dict[str, float] = {}
    transcript_chunks: list[dict] = []
    chunk_dir: Path | None = None
    total_started = time.perf_counter()

    try:
        _notify("分析静音并切分音频...")
        split_started = time.perf_counter()
        chunk_spans = _build_processing_spans(audio_path)
        chunk_dir, chunk_infos = _extract_wav_chunks(
            audio_path, chunk_spans, on_stage=on_stage
        )
        split_elapsed = time.perf_counter() - split_started
        log.append(f"切分完成：共 {len(chunk_infos)} 个处理块")
        _record_stage_timing(log, timings, "split_s", "静音分析与切块", split_elapsed)

        backend = _resolve_asr_backend(device)
        word_dicts: list[dict] = []
        try:
            load_started = time.perf_counter()
            backend.load(on_stage=on_stage)
            load_elapsed = time.perf_counter() - load_started
            _record_stage_timing(
                log, timings, "asr_model_load_s", "ASR模型加载", load_elapsed
            )

            text_results, text_timings = _transcribe_asr_chunks_text_only(
                backend,
                chunk_infos,
                "ASR 文本转写",
                on_stage=on_stage,
            )
            _record_stage_timing(
                log,
                timings,
                "asr_text_transcribe_s",
                "ASR文本转写",
                text_timings["text_transcribe_s"],
            )
            for timing_key, timing_value in text_timings.items():
                if timing_key == "text_transcribe_s":
                    continue
                timings[timing_key] = timing_value

            qc_report, qc_timings = _run_TRANSCRIPTION_qc(
                backend,
                chunk_infos,
                text_results,
                log,
                on_stage=on_stage,
            )
            _record_stage_timing(
                log,
                timings,
                "asr_qc_s",
                "ASR质检",
                qc_timings["asr_qc_s"],
            )

            text_results, recovery_timings = _recover_TRANSCRIPTION_results_if_needed(
                backend,
                chunk_infos,
                text_results,
                qc_report,
                log,
                on_stage=on_stage,
            )
            if recovery_timings["asr_recovery_s"] > 0:
                _record_stage_timing(
                    log,
                    timings,
                    "asr_recovery_s",
                    "ASR局部修复",
                    recovery_timings["asr_recovery_s"],
                )
                timings["asr_recovered_chunks"] = recovery_timings[
                    "asr_recovered_chunks"
                ]
                for timing_key, timing_label in (
                    ("asr_recovery_separator_s", "ASR人声分离"),
                    ("asr_recovery_retranscribe_s", "ASR修复重转写"),
                    ("asr_recovery_model_unload_s", "ASR修复模型卸载"),
                ):
                    if recovery_timings.get(timing_key, 0.0) > 0:
                        _record_stage_timing(
                            log,
                            timings,
                            timing_key,
                            timing_label,
                            recovery_timings[timing_key],
                        )

            transcript_chunks = _build_transcript_chunks(chunk_infos, text_results)

            unload_started = time.perf_counter()
            backend.unload_model(on_stage=on_stage)
            unload_elapsed = time.perf_counter() - unload_started
            _record_stage_timing(
                log,
                timings,
                "asr_model_unload_s",
                "ASR模型卸载",
                unload_elapsed,
            )

            prepared_results, align_timings = _align_TRANSCRIPTION_results(
                backend,
                text_results,
                on_stage=on_stage,
            )
            _record_stage_timing(
                log,
                timings,
                "alignment_s",
                "Alignment对齐",
                align_timings["alignment_s"],
            )

            aligner_unload_started = time.perf_counter()
            backend.unload_forced_aligner(on_stage=on_stage)
            aligner_unload_elapsed = time.perf_counter() - aligner_unload_started
            _record_stage_timing(
                log,
                timings,
                "alignment_model_unload_s",
                "Alignment模型卸载",
                aligner_unload_elapsed,
            )

            for idx, chunk, (chunk_result, chunk_log) in zip(
                range(1, len(chunk_infos) + 1),
                chunk_infos,
                prepared_results,
            ):
                chunk_words, chunk_log = _finalize_aligned_chunk_without_asr_retry(
                    chunk,
                    chunk_result,
                    list(chunk_log),
                )
                for entry in chunk_log:
                    if entry.startswith(
                        (
                            "ASR 输出",
                            "Alignment 策略",
                            "Alignment 哨兵",
                            "Alignment 回退",
                            "Alignment 词数",
                            "Alignment 输入文本长度",
                            "ASR 原始文本长度",
                            "Alignment 模式",
                            "Alignment 异常",
                            "Alignment VAD 回退",
                            "AnimeWhisper 对齐模式",
                            "AnimeWhisper VAD 回退",
                        )
                    ):
                        log.append(f"chunk {idx}: {entry}")

                for word in chunk_words:
                    word_dicts.append(
                        {
                            "start": chunk["start"] + float(word["start"]),
                            "end": chunk["start"] + float(word["end"]),
                            "word": word["word"],
                            "source_chunk_index": chunk.get("index", idx - 1),
                        }
                    )
        finally:
            backend.close()

        merge_started = time.perf_counter()
        word_dicts.sort(key=lambda item: (item["start"], item["end"]))
        log.append(f"Alignment 汇总词数: {len(word_dicts)}")
        segments = _merge_words_to_segments(word_dicts)
        segments = _postprocess_segments(segments)
        merge_elapsed = time.perf_counter() - merge_started
        _record_stage_timing(
            log, timings, "subtitle_merge_s", "字幕合并", merge_elapsed
        )

        total_elapsed = time.perf_counter() - total_started
        _record_stage_timing(
            log,
            timings,
            "asr_alignment_total_s",
            "ASR与Alignment总计",
            total_elapsed,
        )
        log.append(f"过滤后保留字幕: {len(segments)}")

        details = {
            "backend": get_backend_label(),
            "audio_path": audio_path,
            "device": device,
            "chunk_count": len(chunk_infos),
            "transcript_chunks": transcript_chunks,
            "asr_qc": qc_report,
            "stage_timings": timings,
            "word_count": len(word_dicts),
            "segment_count": len(segments),
        }
        return segments, log, details
    finally:
        if chunk_dir is not None and chunk_dir.exists() and not _KEEP_ASR_CHUNKS:
            _delete_path_for_cleanup(chunk_dir)


def transcribe_and_align(
    audio_path: str,
    device: str,
    on_stage: Callable[[str], None] | None = None,
    include_details: bool = False,
) -> tuple[list[dict], list[str]] | tuple[list[dict], list[str], dict]:
    segments, log, details = _transcribe_and_align_local(
        audio_path,
        device,
        on_stage=on_stage,
    )
    if include_details:
        return segments, log, details
    return segments, log
