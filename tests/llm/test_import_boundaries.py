"""The preflight check must not depend on the stage it precedes.

`llm.backends → openai_compat → preflight → translator → backends` was the only
multi-module cycle in `src/`, and it existed for one reason: "how much room does
this endpoint give" had no answer outside the engine that spends it. Lazy
imports kept it from failing at startup, which is not the same as it being
right - the question is about configuration and learned limits, so it now lives
in `llm.token_budget`, which imports neither the registry nor the engine.

Read statically, including lazy imports, because a runtime check would only
prove that today's call order happens to work.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def _imports_of(module: str) -> set[str]:
    """Every `llm.*`/`core.*` module named by an import anywhere in the file."""
    path = SRC / (module.replace(".", "/") + ".py")
    if not path.exists():
        path = SRC / module.replace(".", "/") / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_preflight_does_not_import_the_translation_engine() -> None:
    imported = _imports_of("llm.preflight")

    assert not [name for name in imported if name.startswith("llm.translator")]
    assert not [
        name
        for name in imported
        if name == "llm.backends" or name.startswith("llm.backends.")
    ]


def test_the_budget_module_stays_pure() -> None:
    # If this ever imports the registry or the engine, the cycle is back and the
    # preflight check goes with it.
    imported = _imports_of("llm.token_budget")

    assert not [
        name
        for name in imported
        if name.startswith("llm.translator") or name.startswith("llm.backends")
    ]


def test_no_import_cycle_reaches_back_into_the_engine() -> None:
    """Walk the four modules that used to form the cycle."""
    seen: set[str] = set()
    frontier = ["llm.preflight"]
    while frontier:
        module = frontier.pop()
        if module in seen:
            continue
        seen.add(module)
        for name in _imports_of(module):
            if not name.startswith("llm."):
                continue
            candidate = name if (SRC / (name.replace(".", "/") + ".py")).exists() else None
            if candidate is None and (SRC / name.replace(".", "/") / "__init__.py").exists():
                candidate = name
            if candidate and candidate not in seen:
                frontier.append(candidate)

    assert "llm.translator" not in seen
