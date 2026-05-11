import contextlib
import hashlib
import json
import threading
from pathlib import Path


def _warn_translation_cache(message: str) -> None:
    print(f"[WARN] translation cache {message}", flush=True)


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
    except Exception as exc:
        _warn_translation_cache(f"load failed for {path}: {exc}")
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
    except Exception as exc:
        _warn_translation_cache(f"JSON load failed for {path}: {exc}")
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
    except Exception as exc:
        _warn_translation_cache(f"JSONL load failed for {path}: {exc}")
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
    prompt_version: str,
    model_name: str,
    compact_system_prompt: bool,
) -> str:
    compact = "1" if compact_system_prompt else "0"
    payload = (
        f"{prompt_version}\n{target_lang.strip()}\n{glossary.strip()}\n"
        f"{extra_glossary.strip()}\n{(character_reference or '').strip()}\n"
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
        prompt_version=prompt_version,
        model_name=model_name,
        compact_system_prompt=compact_system_prompt,
    )
    return f"{prompt_sig}::{batch_index}::{source_sig}"
