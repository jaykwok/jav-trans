"""Resolve reusable human-audit instructions from CLI text or a UTF-8 file."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class ResolvedAuditPrompt:
    text: str
    source: str
    sha256: str


def resolve_audit_prompt(
    *,
    prompt: str = "",
    prompt_file: str = "",
    default_prompt: str,
) -> ResolvedAuditPrompt:
    """Resolve one audit prompt without allowing ambiguous override precedence."""

    if prompt and prompt_file:
        raise ValueError("use only one of --prompt or --prompt-file")
    if prompt_file:
        path = Path(prompt_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8-sig")
        source = str(path)
    elif prompt:
        text = prompt
        source = "cli:--prompt"
    else:
        text = default_prompt
        source = "builtin-default"
    text = text.strip()
    if not text:
        raise ValueError("audit prompt must not be empty")
    return ResolvedAuditPrompt(
        text=text,
        source=source,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
