"""Shared project path bootstrap and UTF-8 JSON/JSONL helpers for tools/."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def ensure_src_on_path() -> tuple[Path, Path]:
    """Ensure both repository root and ``src/`` are importable via ``sys.path``."""
    for root in (PROJECT_ROOT, SRC_ROOT):
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)
    return PROJECT_ROOT, SRC_ROOT


def read_json(path: Path | str, *, encoding: str = "utf-8-sig") -> Any:
    """Read and parse a JSON file as UTF-8 (tolerating an optional BOM)."""
    return json.loads(Path(path).read_text(encoding=encoding))


def write_json(
    path: Path | str,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    trailing_newline: bool = True,
) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating parent directories if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    if trailing_newline:
        text += "\n"
    target.write_text(text, encoding="utf-8")
    return target


def read_jsonl(
    path: Path | str,
    *,
    encoding: str = "utf-8-sig",
    require_mapping: bool = False,
) -> list[Any]:
    """Read non-empty lines from a UTF-8 JSONL file."""
    target = Path(path)
    rows: list[Any] = []
    with target.open("r", encoding=encoding) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if require_mapping:
                if not isinstance(payload, Mapping):
                    raise ValueError(f"JSONL row must be an object: {target}:{line_number}")
                payload = dict(payload)
            rows.append(payload)
    return rows


def write_jsonl(
    path: Path | str,
    rows: Iterable[Any],
    *,
    sort_keys: bool = False,
) -> Path:
    """Write an iterable of JSON-serializable rows as UTF-8 JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n"
        for row in rows
    ]
    target.write_text("".join(lines), encoding="utf-8")
    return target


__all__ = [
    "PROJECT_ROOT",
    "SRC_ROOT",
    "ensure_src_on_path",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]
