import hashlib
import json
from pathlib import Path

from llm.glossary import normalize_glossary_text


def _warn_translation_cache(message: str) -> None:
    print(f"[WARN] translation cache {message}", flush=True)


def _load_translation_cache(path) -> dict:
    if not path:
        return {}
    try:
        cache_path = _translation_cache_jsonl_path(Path(path))
        if cache_path.exists():
            return _read_translation_cache_jsonl(cache_path)
        return {}
    except Exception as exc:
        _warn_translation_cache(f"load failed for {path}: {exc}")
        return {}


def _translation_cache_jsonl_path(path: Path) -> Path:
    return path.with_suffix(".jsonl") if path.suffix.lower() == ".json" else path


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
    except Exception as exc:
        _warn_translation_cache(f"JSONL load failed for {path}: {exc}")
        return {}


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


def _compute_prompt_signature(
    extra_glossary: str = "",
    *,
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
    prompt_version: str,
    model_name: str,
    compact_system_prompt: bool,
) -> str:
    compact = "1" if compact_system_prompt else "0"
    normalized_glossary = normalize_glossary_text(glossary)
    normalized_extra_glossary = normalize_glossary_text(extra_glossary)
    payload = (
        f"{prompt_version}\n{target_lang.strip()}\n{normalized_glossary}\n"
        f"{normalized_extra_glossary}\n{(character_reference or '').strip()}\n"
        f"{model_name.strip()}\ncompact={compact}"
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
    prompt_version: str,
    model_name: str,
    compact_system_prompt: bool,
) -> str:
    source_payload = []
    for seg in batch_segments:
        try:
            start = float(seg.get("start", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            end = start
        source_payload.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_sec": round(max(0.0, end - start), 3),
                "ja": str(seg.get("ja_text") or seg.get("text") or seg.get("ja") or ""),
            }
        )
    source_sig = hashlib.sha1(
        json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    prompt_sig = _compute_prompt_signature(
        extra_glossary,
        glossary=glossary,
        target_lang=target_lang,
        character_reference=character_reference,
        prompt_version=prompt_version,
        model_name=model_name,
        compact_system_prompt=compact_system_prompt,
    )
    return f"{prompt_sig}::{batch_index}::{source_sig}"
