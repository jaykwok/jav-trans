import json
import hashlib
import contextlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from openai import OpenAI

from core.config import load_config


load_config()

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_CLIENT: OpenAI | None = None
_CLIENT_KEY: tuple[str, str] = ("", "")
_CLIENT_LOCK = threading.Lock()
_RETRY_CONTEXT = threading.local()
_cache_lock = threading.Lock()
PROMPT_VERSION = "v2.1"
_LEADING_SPEAKER_RE = re.compile(
    r"^\s*(?:男|女|男性|女性|男优|女优|スタッフ|撮影者|カメラマン|"
    r"[A-Za-z][A-Za-z ._-]{0,20})\s*[：:]\s*"
)


class RetryableTranslationFormatError(RuntimeError):
    pass


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set in config.py or .env")
    return value


OPENAI_COMPATIBILITY_BASE_URL = (
    os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip() or None
)
API_KEY = os.getenv("API_KEY", "").strip() or None
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "").strip()
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "max").strip() or "max"
CHARACTER_FULL_NAME_REFERENCE = (
    os.getenv("ASR_CONTEXT", "").strip()
)

TRANSLATION_MAX_TOKENS = max(0, int(os.getenv("TRANSLATION_MAX_TOKENS", "384000")))
COMPACT_SYSTEM_PROMPT = os.getenv("COMPACT_SYSTEM_PROMPT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRANSLATION_API_RETRIES = max(1, int(os.getenv("TRANSLATION_API_RETRIES", "4")))
TRANSLATION_API_BACKOFF_BASE_S = max(
    0.1,
    float(os.getenv("TRANSLATION_API_BACKOFF_BASE_S", "1.5")),
)
TRANSLATION_API_BACKOFF_MAX_S = max(
    TRANSLATION_API_BACKOFF_BASE_S,
    float(os.getenv("TRANSLATION_API_BACKOFF_MAX_S", "20")),
)


def _get_client() -> OpenAI:
    global _CLIENT, _CLIENT_KEY
    current_key = os.getenv("API_KEY", "").strip() or None
    current_url = os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip() or None
    key_tuple = (current_key or "", current_url or "")
    with _CLIENT_LOCK:
        if _CLIENT is None or key_tuple != _CLIENT_KEY:
            _CLIENT = OpenAI(api_key=current_key, base_url=current_url)
            _CLIENT_KEY = key_tuple
    return _CLIENT


def _current_retry_events() -> list[dict] | None:
    events = getattr(_RETRY_CONTEXT, "events", None)
    return events if isinstance(events, list) else None


def _load_translation_cache(path) -> dict:
    if not path:
        return {}
    try:
        cache_path = _translation_cache_jsonl_path(Path(path))
        legacy_path = _translation_cache_legacy_json_path(Path(path))
        if cache_path.exists():
            return _read_translation_cache_jsonl(cache_path)
        if legacy_path.exists():
            data = _read_translation_cache_json(legacy_path)
            if data:
                _rewrite_translation_cache_jsonl(cache_path, data)
            if legacy_path != cache_path:
                with contextlib.suppress(Exception):
                    legacy_path.unlink()
            return data
        return {}
    except Exception:
        return {}


def _translation_cache_jsonl_path(path: Path) -> Path:
    return path.with_suffix(".jsonl") if path.suffix.lower() == ".json" else path


def _translation_cache_legacy_json_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".json" else path.with_suffix(".json")


def _read_translation_cache_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as reader:
            data = json.load(reader)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_translation_cache_jsonl(path: Path) -> dict:
    cache: dict[str, list] = {}
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
                key = item.get("key")
                value = item.get("value")
                if isinstance(key, str) and isinstance(value, list):
                    cache[key] = value
        return cache
    except Exception:
        return {}


def _rewrite_translation_cache_jsonl(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as writer:
        for key, value in cache.items():
            writer.write(
                json.dumps(
                    {
                        "key": str(key),
                        "value": list(value) if isinstance(value, list) else value,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    tmp_path.replace(path)


def _save_cache_entry(path, batch_key, zh_texts, lock) -> None:
    if not path:
        return
    raw_path = Path(path)
    cache_path = _translation_cache_jsonl_path(raw_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with cache_path.open("a", encoding="utf-8") as writer:
            writer.write(
                json.dumps(
                    {"key": str(batch_key), "value": list(zh_texts)},
                    ensure_ascii=False,
                )
                + "\n"
            )
        legacy_path = _translation_cache_legacy_json_path(raw_path)
        if legacy_path != cache_path and legacy_path.exists():
            with contextlib.suppress(Exception):
                legacy_path.unlink()


def _compute_prompt_signature(
    extra_glossary: str = "",
    *,
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
) -> str:
    model_name = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME).strip()
    compact = "1" if COMPACT_SYSTEM_PROMPT else "0"
    payload = (
        f"{PROMPT_VERSION}\n{target_lang.strip()}\n{glossary.strip()}\n"
        f"{extra_glossary.strip()}\n{(character_reference or '').strip()}\n"
        f"{model_name}\ncompact={compact}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _translation_cache_key(
    batch_index: int,
    batch_segments: list[dict],
    *,
    extra_glossary: str = "",
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
) -> str:
    source_text = "|".join(
        str(seg.get("ja_text") or seg.get("text") or seg.get("ja") or "")
        for seg in batch_segments
    )
    source_sig = hashlib.sha1(source_text.encode("utf-8")).hexdigest()[:8]
    prompt_sig = _compute_prompt_signature(
        extra_glossary,
        glossary=glossary,
        target_lang=target_lang,
        character_reference=character_reference,
    )
    return f"{prompt_sig}::{batch_index}::{source_sig}"


def _filter_global_glossary_terms(raw_terms) -> list[dict]:
    if not isinstance(raw_terms, list):
        return []
    filtered: list[dict] = []
    banned_re = re.compile(r"[,\u3001\u3002\uff0c\uff1f?？\s]")
    for item in raw_terms:
        if not isinstance(item, dict):
            continue
        ja = str(item.get("ja", "")).strip()
        zh = str(item.get("zh", "")).strip()
        if not ja or not zh:
            continue
        if len(ja) > 8 or len(zh) > 8:
            continue
        if banned_re.search(ja) or banned_re.search(zh):
            continue
        filtered.append({"ja": ja, "zh": zh})
        if len(filtered) >= 15:
            break
    return filtered


def _format_global_glossary_terms(
    terms: list[dict],
    *,
    glossary: str = "",
) -> str:
    lines = []
    seen: set[str] = set()
    for item in terms:
        ja = str(item.get("ja", "")).strip()
        zh = str(item.get("zh", "")).strip()
        if not ja or not zh or ja in seen:
            continue
        if glossary and ja in glossary:
            continue
        seen.add(ja)
        lines.append(f"{ja} \u2192 {zh}")
    return "\n".join(lines)


def _global_glossary_cache_path(translation_cache_path: str) -> str:
    cache_path = Path(translation_cache_path)
    return str(cache_path.with_name("translation_global_glossary.json"))


def extract_global_glossary(all_ja_texts: list[str], cache_path: str) -> list[dict]:
    if not cache_path:
        return []
    path = Path(cache_path)
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            terms = payload.get("terms") if isinstance(payload, dict) else payload
            return _filter_global_glossary_terms(terms)
    except Exception as exc:
        print(f"[WARN] failed to load translation global glossary cache: {exc}")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        source_text = "\n".join(str(text or "") for text in all_ja_texts)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是字幕术语提取器。请从全片日文字幕中提取 10-20 个反复出现的核心词，"
                    "范围包括代词、人名、性器官词、高频形容词。给出推荐中文翻译。"
                    '只返回合法 JSON：{"terms":[{"ja":"...","zh":"..."}]}。'
                ),
            },
            {"role": "user", "content": f"【全片日文字幕】\n{source_text}"},
        ]
        raw_output = _chat(messages, expected_count=0, reasoning_effort="low")
        parsed = json.loads(_strip_reasoning_artifacts(raw_output))
        terms = _filter_global_glossary_terms(parsed.get("terms") if isinstance(parsed, dict) else None)
        tmp_path = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
        tmp_path.write_text(
            json.dumps({"terms": terms}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return terms
    except Exception as exc:
        print(f"[WARN] failed to extract translation global glossary: {exc}")
        return []


def _test_crash_translation_batch() -> int:
    raw_value = os.getenv("_TEST_CRASH_TRANSLATION_BATCH", "").strip()
    if not raw_value:
        return 0
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 0


def _normalize_source_text(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", (text or "").strip())
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
    cleaned = re.sub(r"(.)\1{4,}", r"\1\1\1", cleaned)
    return cleaned.strip()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_global_context(
    segments: list[dict],
    current_index: int | None = None,
    batch_size: int = 0,
    max_chars: int | None = None,
) -> str:
    del current_index, batch_size

    lines = []
    for idx, seg in enumerate(segments):
        text = _normalize_source_text(seg.get("text", ""))
        if not text:
            continue
        start = _safe_float(seg.get("start"))
        end = _safe_float(seg.get("end"))
        lines.append(f"{idx:04d} [{start:.2f}->{end:.2f}] {text}")

    context = "\n".join(lines)
    if max_chars is not None and max_chars > 0:
        return context[:max_chars]
    return context


def translate_segments(
    segments: list[dict],
    batch_size: int = 100,
    global_context: str | None = None,
    max_workers: int = 1,
    cache_path: str = "",
    target_lang: str = "简体中文",
    glossary: str = "",
    character_reference: str | None = None,
    reasoning_effort: str | None = None,
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    if not segments:
        return [], [], []

    effective_batch_size = max(0, int(batch_size))
    effective_max_workers = max(1, int(max_workers))
    effective_cache_path = cache_path or ""
    effective_target_lang = (target_lang or "简体中文").strip() or "简体中文"
    effective_glossary = (glossary or "").strip()
    effective_character_reference = (
        CHARACTER_FULL_NAME_REFERENCE
        if character_reference is None
        else (character_reference or "").strip()
    )
    previous_retry_events = getattr(_RETRY_CONTEXT, "events", None)
    retry_events: list[dict] = []
    _RETRY_CONTEXT.events = retry_events
    try:
        if effective_batch_size > 0 and len(segments) > effective_batch_size:
            zh_texts, timings = _translate_segments_batched(
                segments,
                batch_size=effective_batch_size,
                max_workers=effective_max_workers,
                global_context=global_context,
                cache_path=effective_cache_path,
                target_lang=effective_target_lang,
                glossary=effective_glossary,
                character_reference=effective_character_reference,
                reasoning_effort=reasoning_effort,
                on_batch_done=on_batch_done,
                on_progress=on_progress,
            )
        else:
            zh_texts, timings = _translate_segments_oneshot(
                segments,
                global_context=global_context,
                target_lang=effective_target_lang,
                glossary=effective_glossary,
                character_reference=effective_character_reference,
                reasoning_effort=reasoning_effort,
                on_batch_done=on_batch_done,
                on_progress=on_progress,
            )
        return zh_texts, timings, list(retry_events)
    finally:
        if previous_retry_events is None:
            with contextlib.suppress(AttributeError):
                delattr(_RETRY_CONTEXT, "events")
        else:
            _RETRY_CONTEXT.events = previous_retry_events


def _chat_with_reasoning(
    messages: list[dict],
    *,
    expected_count: int,
    reasoning_effort: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> str:
    if reasoning_effort is None:
        return _chat(
            messages,
            expected_count=expected_count,
            on_progress=on_progress,
        )
    return _chat(
        messages,
        expected_count=expected_count,
        reasoning_effort=reasoning_effort,
        on_progress=on_progress,
    )


def _translate_segments_oneshot(
    segments: list[dict],
    *,
    global_context: str | None = None,
    target_lang: str,
    glossary: str,
    character_reference: str,
    reasoning_effort: str | None = None,
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[list[str], list[dict]]:
    started = time.perf_counter()
    full_context = (
        global_context
        if global_context is not None
        else generate_global_context(segments)
    )
    source_payload = _serialize_segments(segments)
    expected_count = len(segments)

    messages = _build_oneshot_messages(
        source_payload=source_payload,
        expected_count=expected_count,
        target_lang=target_lang,
        glossary=glossary,
        character_reference=character_reference,
    )
    missing_indexes: list[int] = []
    zh_texts: list[str | None] = [None] * expected_count
    for attempt in range(TRANSLATION_API_RETRIES):
        _emit_progress(on_progress, {"phase": "reset", "attempt": attempt})
        try:
            raw_output = _chat_with_reasoning(
                messages,
                expected_count=expected_count,
                reasoning_effort=reasoning_effort,
                on_progress=on_progress,
            )
            zh_texts = _parse_translation_output(raw_output, expected_count)
            missing_indexes = _missing_indexes(zh_texts)
            if missing_indexes:
                raise RetryableTranslationFormatError(
                    "DeepSeek JSON mode returned incomplete translations: "
                    f"{len(missing_indexes)} missing of {expected_count}; "
                    f"missing ids={missing_indexes[:50]}"
                )
            break
        except RetryableTranslationFormatError as exc:
            if attempt < TRANSLATION_API_RETRIES - 1:
                _request_backoff_sleep(attempt, exc)
                continue
            raise RuntimeError(
                "One-shot translation returned invalid or incomplete JSON after "
                f"{TRANSLATION_API_RETRIES} attempts: {exc}"
            ) from exc

    timing = {
        "start_index": 0,
        "segment_count": expected_count,
        "elapsed_s": time.perf_counter() - started,
        "mode": "oneshot_full_context",
        "request_count": 1,
        "source_payload_chars": len(source_payload),
        "global_context_chars": len(full_context),
        "missing_count": 0,
        "missing_indexes": [],
    }
    if on_batch_done:
        on_batch_done(timing)
    return [text or "" for text in zh_texts], [timing]


def _translate_segments_batched(
    segments: list[dict],
    *,
    batch_size: int,
    max_workers: int,
    global_context: str | None = None,
    cache_path: str = "",
    target_lang: str,
    glossary: str,
    character_reference: str,
    reasoning_effort: str | None = None,
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[list[str], list[dict]]:
    started = time.perf_counter()
    batches = _split_into_batches(segments, batch_size)
    expected_total = len(segments)
    extra_glossary = ""
    if cache_path:
        glossary_terms = extract_global_glossary(
            [str(seg.get("text", "")) for seg in segments],
            _global_glossary_cache_path(cache_path),
        )
        extra_glossary = _format_global_glossary_terms(
            glossary_terms,
            glossary=glossary,
        )
    full_context = (
        global_context
        if global_context is not None
        else generate_global_context(segments)
    )
    progress_callbacks, _ = _make_aggregated_progress_callback(
        len(batches),
        expected_total,
        on_progress,
    )
    zh_texts: list[str | None] = [None] * expected_total
    timings_by_batch: dict[int, dict] = {}
    translation_cache = (
        _load_translation_cache(cache_path)
        if cache_path
        else {}
    )
    pending_batches: list[tuple[int, list[dict]]] = []

    for batch_index, batch_segments in enumerate(batches):
        batch_key = _translation_cache_key(
            batch_index,
            batch_segments,
            extra_glossary=extra_glossary,
            glossary=glossary,
            target_lang=target_lang,
            character_reference=character_reference,
        )
        cached_texts = translation_cache.get(batch_key)
        if isinstance(cached_texts, list) and len(cached_texts) == len(batch_segments):
            print(f"[translation-cache] restored batch {batch_index} cache_key={batch_key}")
            start_index = batch_index * batch_size
            for offset, text in enumerate(cached_texts):
                zh_texts[start_index + offset] = str(text or "")
            timing = {
                "batch_index": batch_index,
                "start_index": start_index,
                "segment_count": len(batch_segments),
                "elapsed_s": 0.0,
                "mode": "translation_cache_hit",
                "request_count": 0,
                "source_payload_chars": 0,
                "global_context_chars": len(full_context),
                "missing_count": 0,
                "missing_indexes": [],
                "cache_hit": True,
            }
            timings_by_batch[batch_index] = timing
            _emit_progress(
                progress_callbacks[batch_index],
                {
                    "phase": "done",
                    "translated": len(batch_segments),
                    "expected": len(batch_segments),
                },
            )
            if on_batch_done:
                on_batch_done(timing)
        else:
            pending_batches.append((batch_index, batch_segments))

    def run_batch(batch_index: int, batch_segments: list[dict]) -> tuple[int, list[str | None], dict]:
        batch_started = time.perf_counter()
        start_index = batch_index * batch_size
        expected_count = len(batch_segments)
        source_payload = _serialize_segments(batch_segments, start_index=start_index)
        messages = _build_batch_messages(
            batch_segments,
            full_context,
            start_index,
            character_reference,
            expected_count,
            batch_index=batch_index,
            extra_glossary=extra_glossary,
            target_lang=target_lang,
            glossary=glossary,
        )
        batch_results: list[str | None] = [None] * expected_total
        missing_indexes: list[int] = []
        progress_callback = progress_callbacks[batch_index]

        for attempt in range(TRANSLATION_API_RETRIES):
            _emit_progress(progress_callback, {"phase": "reset", "attempt": attempt})
            try:
                raw_output = _chat_with_reasoning(
                    messages,
                    expected_count=expected_count,
                    reasoning_effort=reasoning_effort,
                    on_progress=progress_callback,
                )
                parsed = _parse_translation_output_by_global_id(
                    raw_output,
                    expected_ids=list(range(start_index, start_index + expected_count)),
                    total_count=expected_total,
                )
                batch_results = parsed
                missing_indexes = [
                    index
                    for index in range(start_index, start_index + expected_count)
                    if batch_results[index] is None
                ]
                if missing_indexes:
                    raise RetryableTranslationFormatError(
                        "DeepSeek JSON mode returned incomplete batch translations: "
                        f"{len(missing_indexes)} missing of {expected_count}; "
                        f"missing ids={missing_indexes[:50]}"
                    )
                break
            except RetryableTranslationFormatError as exc:
                if attempt < TRANSLATION_API_RETRIES - 1:
                    _request_backoff_sleep(attempt, exc)
                    continue
                raise RuntimeError(
                    "Batch translation returned invalid or incomplete JSON after "
                    f"{TRANSLATION_API_RETRIES} attempts: batch={batch_index}, "
                    f"start_index={start_index}, size={expected_count}, error={exc}"
                ) from exc

        timing = {
            "batch_index": batch_index,
            "start_index": start_index,
            "segment_count": expected_count,
            "elapsed_s": time.perf_counter() - batch_started,
            "mode": "batched_full_context",
            "request_count": 1,
            "source_payload_chars": len(source_payload),
            "global_context_chars": len(full_context),
            "missing_count": len(missing_indexes),
            "missing_indexes": missing_indexes,
        }
        return batch_index, batch_results, timing

    if pending_batches:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pending_batches))) as executor:
            pending_by_index = {
                batch_index: batch for batch_index, batch in pending_batches
            }
            futures = {
                executor.submit(run_batch, batch_index, batch): batch_index
                for batch_index, batch in pending_batches
            }
            try:
                for future in as_completed(futures):
                    batch_index, batch_results, timing = future.result()
                    timings_by_batch[batch_index] = timing
                    start_index = int(timing["start_index"])
                    segment_count = int(timing["segment_count"])
                    local_texts: list[str] = []
                    for offset in range(segment_count):
                        global_index = start_index + offset
                        text = batch_results[global_index] or ""
                        zh_texts[global_index] = text
                        local_texts.append(text)
                    if cache_path:
                        batch_key = _translation_cache_key(
                            batch_index,
                            pending_by_index[batch_index],
                            extra_glossary=extra_glossary,
                            glossary=glossary,
                            target_lang=target_lang,
                            character_reference=character_reference,
                        )
                        _save_cache_entry(
                            cache_path,
                            batch_key,
                            local_texts,
                            _cache_lock,
                        )
                        print(f"[translation-cache] saved batch {batch_index} cache_key={batch_key}")
                        if _test_crash_translation_batch() == batch_index + 1:
                            for pending_future in futures:
                                if pending_future is not future:
                                    pending_future.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise SystemExit(1)
                    if on_batch_done:
                        on_batch_done(timing)
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    missing = _missing_indexes(zh_texts)
    if missing:
        raise RuntimeError(
            "Batched translation finished with missing translations: "
            f"{len(missing)} missing; ids={missing[:50]}"
        )

    timings = [timings_by_batch[index] for index in sorted(timings_by_batch)]
    timings.append(
        {
            "start_index": 0,
            "segment_count": expected_total,
            "elapsed_s": time.perf_counter() - started,
            "mode": "batched_full_context_total",
            "request_count": len(pending_batches),
            "batch_size": batch_size,
            "max_workers": max_workers,
            "cache_hit_count": len(batches) - len(pending_batches),
            "missing_count": 0,
            "missing_indexes": [],
        }
    )
    return [text or "" for text in zh_texts], timings


def _split_into_batches(segments: list[dict], batch_size: int) -> list[list[dict]]:
    if not segments:
        return []
    if batch_size <= 0:
        return [segments]
    return [segments[index : index + batch_size] for index in range(0, len(segments), batch_size)]


def _serialize_segments(segments: list[dict], *, start_index: int = 0) -> str:
    payload = []
    for idx, seg in enumerate(segments):
        start = _safe_float(seg.get("start"))
        end = _safe_float(seg.get("end"))
        ja_text = _normalize_source_text(seg.get("text", ""))
        gender = seg.get("gender")
        if gender == "M":
            ja_text = f"[M]{ja_text}"
        elif gender == "F":
            ja_text = f"[F]{ja_text}"
        payload.append(
            {
                "id": start_index + idx,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_sec": round(max(0.0, end - start), 3),
                "ja": ja_text,
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_full_segments_summary(segments: list[dict], *, limit_chars: int = 1800) -> str:
    lines = []
    for idx, seg in enumerate(segments):
        start = _safe_float(seg.get("start"))
        text = _normalize_source_text(seg.get("text", ""))
        if not text:
            continue
        lines.append(f"{idx}: {start:.2f}s {text}")
    summary = "\n".join(lines)
    if len(summary) > limit_chars:
        return summary[:limit_chars].rstrip() + "\n..."
    return summary


_SYSTEM_PROMPT_FULL = (
    "你是专业的日语成人视频字幕翻译，目标语言是{target_lang}。\n"
    "本片译文要体现主动撩拨的语气，露骨直接；遇到性器官性行为时优先用粗口口语（肉棒、操进来、淫穴、骚逼等），不用阴茎性交插入等书面语。\n"
    "你会收到全片字幕的完整 JSON 数组，必须基于全局上下文一次性完成翻译。\n"
    "本视频出场人物全名参考：【{character_reference}】。\n"
    "要求：\n"
    "1. 翻译要自然、口语化、适合字幕阅读，避免书面腔。\n"
    "2. 成人、暧昧、调情、呻吟、下流语气要保留原本强度，不要净化、弱化或说教。\n"
    "3. 人名不要翻译成中文；如果原文出现人物姓名，直接输出罗马音，格式用 Title Case，并用空格分隔名和姓，例如 Aya Onami。\n"
    "4. {name_boundary}\n"
    "5. {name_homophone}\n"
    "6. 允许根据前后文修正明显 ASR 误听，但不要编造未出现的信息。\n"
    "7. 输入中部分日文前可能带 [M]（男声）或 [F]（女声）声学标签，请利用此信息理解对话切换、调整语气和人称（例如男声用更直白的命令式，女声用更贴合的女性口吻）。**译文中绝对不要保留或输出任何说话人前缀**——[M]/[F] 仅作为输入提示，输出时只给纯净的中文翻译。\n"
    "8. 每条输入必须单独翻译，不能合并、拆分、漏译、调换顺序。\n"
    "9. 输出尽量短，贴近屏幕阅读节奏；短促呻吟和语气词也要简短自然。\n"
    "10. 如果一行里大部分是呻吟、喘息、重复语气词，只保留清晰语义核心，重复部分可以压缩；映射参考：あんっ/はあん 译 啊嗯啊，気持ちいい 译 好舒服要爽死了，イッちゃう/イク 译 要去了要射了，避免感觉很舒服即将达到高潮等翻译腔。\n"
    "11. DeepSeek JSON Mode 要求 prompt 明确包含 json 字样；最终只输出合法 JSON 对象。\n"
    '12. 你必须只输出 JSON：{{"translations":[{{"id":0,"text":"..."}}]}}，并且恰好返回 {expected_count} 条。\n'
    "13. 最终 content 不能为空；即使开启思考模式，也必须把完整 JSON 对象写进最终 content。\n"
    "14. 不要输出 Markdown，不要解释，不要额外字段；思考过程不要写进最终 content。\n\n"
    "EXAMPLE JSON OUTPUT:\n"
    '{{"translations":[{{"id":0,"text":"第一句中文翻译"}},{{"id":1,"text":"第二句中文翻译"}}]}}'
)

_SYSTEM_PROMPT_COMPACT = (
    "你是日语成人视频字幕译者，目标语言是{target_lang}。保持口语、露骨语气和人名罗马音；"
    "可修正明显ASR误听；每条独立翻译，不合并、不漏译、不调序。"
    '只输出合法 JSON：{{"translations":[{{"id":0,"text":"..."}}]}}，恰好 {expected_count} 条。'
)


def _build_system_prompt(
    expected_count: int,
    character_reference: str,
    *,
    target_lang: str,
    glossary: str,
    compact: bool = False,
    extra_glossary: str = "",
) -> str:
    name_guidance = _build_character_name_guidance(character_reference)
    template = _SYSTEM_PROMPT_COMPACT if compact else _SYSTEM_PROMPT_FULL
    effective_target_lang = (target_lang or "简体中文").strip() or "简体中文"
    prompt = template.format(
        target_lang=effective_target_lang,
        character_reference=character_reference,
        name_boundary=name_guidance["boundary"],
        name_homophone=name_guidance["homophone"],
        expected_count=expected_count,
    )
    effective_glossary = (glossary or "").strip()
    if effective_glossary:
        prompt += f"\n\n以下词汇表必须严格遵守，不得自行创造译名：\n{effective_glossary}"
    extra_glossary = (extra_glossary or "").strip()
    if extra_glossary:
        prompt += (
            "\n\n<glossary>\n"
            "本片已确定译法（必须沿用）：\n"
            f"{extra_glossary}\n"
            "</glossary>"
        )
    return prompt


def _build_oneshot_messages(
    source_payload: str,
    expected_count: int,
    compact_system_prompt: bool = False,
    extra_glossary: str = "",
    target_lang: str = "简体中文",
    glossary: str = "",
    character_reference: str | None = None,
) -> list[dict]:
    effective_character_reference = (
        CHARACTER_FULL_NAME_REFERENCE
        if character_reference is None
        else (character_reference or "").strip()
    )
    system_prompt = _build_system_prompt(
        expected_count,
        effective_character_reference,
        target_lang=target_lang,
        glossary=glossary,
        compact=compact_system_prompt,
        extra_glossary=extra_glossary,
    )

    user_parts = [
        "【任务】把下面完整 JSON 数组里的日文字幕逐条翻译成中文字幕。",
        "每个元素的 `id` 必须原样保留，`text` 只能是翻译结果本身。",
        f"【全片字幕 JSON】\n{source_payload}",
    ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _build_batch_messages(
    batch_segments: list[dict],
    full_segments_summary: str | list[dict],
    batch_offset: int,
    character_reference: str,
    expected_count: int,
    batch_index: int = 0,
    extra_glossary: str = "",
    target_lang: str = "简体中文",
    glossary: str = "",
) -> list[dict]:
    if isinstance(full_segments_summary, str):
        summary = full_segments_summary
    else:
        summary = _build_full_segments_summary(full_segments_summary)

    source_payload = _serialize_segments(batch_segments, start_index=batch_offset)
    messages = _build_oneshot_messages(
        source_payload=source_payload,
        expected_count=expected_count,
        compact_system_prompt=COMPACT_SYSTEM_PROMPT and batch_index > 0,
        extra_glossary=extra_glossary,
        target_lang=target_lang,
        glossary=glossary,
        character_reference=character_reference,
    )
    messages[0]["content"] = (
        messages[0]["content"]
        + "\n\n全片字幕概览（仅作上下文连贯参考，不要翻译，原 id 不在本批的不要返回）：\n"
        + summary
    )
    user_content = (
        "【任务】把下面当前批次 JSON 数组里的日文字幕逐条翻译成中文字幕。\n"
        "每个元素的 `id` 是全片全局 id，必须原样保留；只返回本批 id，不要返回概览里的其他 id。\n"
        "每个 `text` 只能是翻译结果本身。\n"
        f"【当前批次字幕 JSON】\n{source_payload}"
    )
    if extra_glossary.strip():
        user_content += "\n\n注意：必须严格使用 System Prompt 中 <glossary> 标签内的术语表翻译。"
    messages[1]["content"] = user_content
    return messages


def _build_character_name_guidance(character_reference: str) -> dict[str, str]:
    normalized = (character_reference or "").strip()
    return {
        "boundary": (
            f"智能拆解边界：请自动根据日本姓名习惯识别人物全名的姓氏和名字边界。"
            f"本片参考名“{normalized}”只作为人名参考；剧中如果只称呼姓氏或名字，只翻译实际出现的部分，"
            "不要强行补全全名。人名用罗马音输出。"
        ),
        "homophone": (
            "ASR 同音纠错：源日文文本由语音识别生成，可能把人物姓名识别成同音字、谐音字或近音词。"
            f"请结合全片上下文，将明显的人名称呼纠正为参考名“{normalized}”对应的罗马音；"
            "纠错只限明显人名称呼，不要把普通名词误改成人名。"
        ),
    }


def _is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    name = type(exc).__name__.lower()
    return any(
        marker in name
        for marker in (
            "ratelimit",
            "timeout",
            "connection",
            "serviceunavailable",
            "internalserver",
        )
    )


def _request_backoff_delay(attempt: int) -> float:
    return min(
        TRANSLATION_API_BACKOFF_MAX_S,
        TRANSLATION_API_BACKOFF_BASE_S * (2**attempt),
    )


def _record_api_retry_event(exc: Exception, attempt: int, delay_s: float) -> None:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    event = {
        "attempt": attempt + 1,
        "delay_s": delay_s,
        "status_code": status_code,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
    }
    local_events = _current_retry_events()
    if local_events is not None:
        local_events.append(event)


def _request_backoff_sleep(attempt: int, exc: Exception) -> None:
    delay = _request_backoff_delay(attempt)
    _record_api_retry_event(exc, attempt, delay)
    time.sleep(delay)


def _create_chat_completion(request: dict):
    last_error: Exception | None = None

    for attempt in range(TRANSLATION_API_RETRIES):
        try:
            return _get_client().chat.completions.create(**request)
        except Exception as exc:
            last_error = exc
            if not _is_retryable_api_error(exc):
                raise

            if attempt < TRANSLATION_API_RETRIES - 1:
                _request_backoff_sleep(attempt, exc)

    if last_error is not None:
        raise last_error
    raise RuntimeError("chat completion failed without an exception")


def _emit_progress(
    on_progress: Callable[[dict], None] | None,
    payload: dict,
) -> None:
    if on_progress is None:
        return
    try:
        on_progress(payload)
    except Exception:
        pass


def _make_aggregated_progress_callback(
    num_batches: int,
    expected_total: int,
    on_progress: Callable[[dict], None] | None,
) -> tuple[list[Callable[[dict], None]], None]:
    lock = threading.Lock()
    batch_states: dict[int, dict] = {}
    last_emit_ts = 0.0
    done_emitted = False

    def emit(payload: dict, *, force: bool = False) -> None:
        nonlocal last_emit_ts
        now = time.monotonic()
        if not force and now - last_emit_ts < 0.25:
            return
        last_emit_ts = now
        _emit_progress(on_progress, payload)

    def build_payload() -> tuple[dict, bool] | None:
        nonlocal done_emitted
        if not batch_states:
            return None

        translated = sum(
            int(state.get("translated", 0))
            for state in batch_states.values()
            if state.get("phase") in {"translating", "done"}
        )
        if (
            num_batches > 0
            and len(batch_states) >= num_batches
            and all(state.get("phase") == "done" for state in batch_states.values())
        ):
            if done_emitted:
                return None
            done_emitted = True
            return (
                {"phase": "done", "translated": expected_total, "expected": expected_total},
                True,
            )

        if any(state.get("phase") == "translating" for state in batch_states.values()):
            return (
                {
                    "phase": "translating",
                    "translated": translated,
                    "expected": expected_total,
                    "content_chars": sum(
                        int(state.get("content_chars", 0))
                        for state in batch_states.values()
                    ),
                },
                False,
            )

        if any(state.get("phase") == "thinking" for state in batch_states.values()):
            return (
                {
                    "phase": "thinking",
                    "reasoning_chars": max(
                        int(state.get("reasoning_chars", 0))
                        for state in batch_states.values()
                    ),
                },
                False,
            )

        if any(state.get("phase") == "reset" for state in batch_states.values()):
            return (
                {
                    "phase": "reset",
                    "attempt": max(
                        int(state.get("attempt", 0))
                        for state in batch_states.values()
                    ),
                },
                True,
            )
        return None

    def make_wrapper(batch_id: int) -> Callable[[dict], None]:
        def wrapper(evt: dict) -> None:
            try:
                payload: tuple[dict, bool] | None
                with lock:
                    batch_states[batch_id] = dict(evt)
                    payload = build_payload()
                if payload is None:
                    return
                emit(payload[0], force=payload[1])
            except Exception:
                pass

        return wrapper

    return [make_wrapper(batch_id) for batch_id in range(num_batches)], None


def _chat(
    messages: list[dict],
    expected_count: int = 0,
    on_progress: Callable[[dict], None] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    model_name = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME).strip()
    if not model_name:
        raise RuntimeError("请先在「翻译设置」中获取并选择翻译模型，再提交任务")
    request = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "response_format": {"type": "json_object"},
        "reasoning_effort": reasoning_effort
        or os.getenv("LLM_REASONING_EFFORT", "max").strip()
        or "max",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    if TRANSLATION_MAX_TOKENS > 0:
        request["max_tokens"] = TRANSLATION_MAX_TOKENS

    response_stream = _create_chat_completion(request)

    final_content = []
    finish_reason = None
    reasoning_chars = 0
    content_chars = 0
    translated_count = 0
    id_scan_tail = ""
    id_marker = '"id":'
    last_emit = 0.0
    debounce_s = 0.25

    def maybe_emit(payload: dict, *, force: bool = False) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if not force and now - last_emit < debounce_s:
            return
        last_emit = now
        _emit_progress(on_progress, payload)

    for chunk in response_stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            reasoning_chars += len(reasoning_content)
            maybe_emit(
                {"phase": "thinking", "reasoning_chars": reasoning_chars}
            )
        if hasattr(delta, "content") and delta.content:
            content_piece = delta.content
            final_content.append(content_piece)
            content_chars += len(content_piece)
            scan_text = id_scan_tail + content_piece
            tail_len = len(id_scan_tail)
            translated_count += sum(
                1 for match in re.finditer(re.escape(id_marker), scan_text) if match.end() > tail_len
            )
            id_scan_tail = scan_text[-(len(id_marker) - 1) :]
            maybe_emit(
                {
                    "phase": "translating",
                    "translated": translated_count,
                    "expected": expected_count,
                    "content_chars": content_chars,
                }
            )
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    if finish_reason == "length":
        raise RetryableTranslationFormatError(
            "DeepSeek JSON mode response was cut off by max_tokens; "
            "increase TRANSLATION_MAX_TOKENS."
        )

    final_content_str = "".join(final_content)
    if not final_content_str.strip():
        raise RetryableTranslationFormatError("DeepSeek JSON mode returned empty content.")
    _emit_progress(
        on_progress,
        {
            "phase": "done",
            "translated": translated_count,
            "expected": expected_count,
        },
    )
    return final_content_str.strip()


def _strip_reasoning_artifacts(raw_output: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", raw_output or "")
    close_tag = "</think>"
    close_idx = cleaned.lower().rfind(close_tag)
    if close_idx != -1:
        cleaned = cleaned[close_idx + len(close_tag) :]
    return cleaned.strip()


def _parse_translation_output(
    raw_output: str,
    expected_count: int,
) -> list[str | None]:
    raw_output = _strip_reasoning_artifacts(raw_output)
    if not raw_output.strip():
        raise RetryableTranslationFormatError(
            "DeepSeek JSON mode returned empty content."
        )

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise RetryableTranslationFormatError(
            "DeepSeek JSON mode response was not valid JSON."
        ) from exc
    return _extract_translations_from_json(parsed, expected_count)


def _parse_translation_output_by_global_id(
    raw_output: str,
    *,
    expected_ids: list[int],
    total_count: int,
) -> list[str | None]:
    raw_output = _strip_reasoning_artifacts(raw_output)
    if not raw_output.strip():
        raise RetryableTranslationFormatError(
            "DeepSeek JSON mode returned empty content."
        )

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise RetryableTranslationFormatError(
            "DeepSeek JSON mode response was not valid JSON."
        ) from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get("translations"), list):
        raise RetryableTranslationFormatError(
            'DeepSeek JSON mode response must be {"translations":[...]} .'
        )

    expected_id_set = set(expected_ids)
    translations = parsed["translations"]
    if len(translations) != len(expected_ids):
        raise RetryableTranslationFormatError(
            "DeepSeek JSON mode returned wrong batch translation count: "
            f"{len(translations)} of {len(expected_ids)}."
        )

    results: list[str | None] = [None] * total_count
    seen_ids: set[int] = set()
    for item in translations:
        if not isinstance(item, dict):
            raise RetryableTranslationFormatError(
                "DeepSeek JSON mode translations must contain objects."
            )
        idx = _coerce_int(item.get("id"))
        if idx is None or idx not in expected_id_set or idx >= total_count:
            raise RetryableTranslationFormatError(
                f"DeepSeek JSON mode returned invalid batch translation id: {item.get('id')!r}."
            )
        if idx in seen_ids:
            raise RetryableTranslationFormatError(
                f"DeepSeek JSON mode returned duplicate translation id: {idx}."
            )
        seen_ids.add(idx)
        results[idx] = _normalize_translation_text(item.get("text"))

    return results


def _extract_translations_from_json(data, expected_count: int) -> list[str | None]:
    if not isinstance(data, dict) or not isinstance(data.get("translations"), list):
        raise RetryableTranslationFormatError(
            'DeepSeek JSON mode response must be {"translations":[...]} .'
        )

    translations = data["translations"]
    if len(translations) != expected_count:
        raise RetryableTranslationFormatError(
            "DeepSeek JSON mode returned wrong translation count: "
            f"{len(translations)} of {expected_count}."
        )

    results: list[str | None] = [None] * expected_count
    seen_ids: set[int] = set()
    for item in translations:
        if not isinstance(item, dict):
            raise RetryableTranslationFormatError(
                "DeepSeek JSON mode translations must contain objects."
            )

        idx = _coerce_int(item.get("id"))
        if idx is None or idx < 0 or idx >= expected_count:
            raise RetryableTranslationFormatError(
                f"DeepSeek JSON mode returned invalid translation id: {item.get('id')!r}."
            )
        if idx in seen_ids:
            raise RetryableTranslationFormatError(
                f"DeepSeek JSON mode returned duplicate translation id: {idx}."
            )

        seen_ids.add(idx)
        results[idx] = _normalize_translation_text(item.get("text"))

    return results


def _coerce_int(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _normalize_translation_text(text) -> str | None:
    if text is None:
        return None

    cleaned = str(text).strip()
    if not cleaned:
        return None

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"^Translation>\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^Original>\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^['\"“”‘’]+|['\"“”‘’]+$", "", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    cleaned = _LEADING_SPEAKER_RE.sub("", cleaned, count=1)
    if "\n" in cleaned:
        cleaned = cleaned.replace("\n", "\\n")
    return cleaned or None


def _missing_indexes(values: list[str | None]) -> list[int]:
    return [idx for idx, value in enumerate(values) if value is None or value == ""]


