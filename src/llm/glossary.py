from __future__ import annotations

import re


_GLOSSARY_COMMA_SPLIT_RE = re.compile(r"[,，]+")


def _iter_raw_glossary_items(text: str | None) -> list[str]:
    """Split glossary input into pair strings.

    Supports both newline-separated entries (preserving commas inside English
    targets such as ``えっちな-naughty, lewd`` when subsequent comma segments
    are not new ``source-target`` pairs) and legacy comma-separated lists
    (such as ``ちんぽ-肉棒, チンポ-肉棒``).
    """
    items: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        comma_parts = [part.strip() for part in _GLOSSARY_COMMA_SPLIT_RE.split(line) if part.strip()]
        if len(comma_parts) > 1 and all("-" in part for part in comma_parts):
            items.extend(comma_parts)
        else:
            items.append(line)
    return items


def parse_glossary_pairs(text: str | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in _iter_raw_glossary_items(text):
        if "→" in item or "->" in item or "-" not in item:
            continue
        source, target = item.split("-", 1)
        source = source.strip()
        target = target.strip()
        if source and target:
            pairs.append((source, target))
    return pairs


def normalize_glossary_text(text: str | None, *, separator: str = "\n") -> str:
    return separator.join(f"{source}-{target}" for source, target in parse_glossary_pairs(text))
