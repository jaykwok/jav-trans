from __future__ import annotations

import re


_GLOSSARY_ITEM_SPLIT_RE = re.compile(r"[,，\n]+")


def parse_glossary_pairs(text: str | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_item in _GLOSSARY_ITEM_SPLIT_RE.split(str(text or "")):
        item = raw_item.strip()
        if not item:
            continue
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
