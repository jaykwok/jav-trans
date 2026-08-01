"""Layering guard: transport/profiles/engine must import without the facade.

The 2026-08 refactor killed the translator<->backend circular import; this
locks it. Runs each import in a subprocess so this test cannot be poisoned by
modules other tests already imported.
"""

import subprocess
import sys

_CHECK = """
import sys
sys.path.insert(0, {src!r})
import llm.{module}
banned = [name for name in sys.modules if name == "llm.translator"]
assert not banned, f"importing llm.{module} pulled in llm.translator"
print("ok")
"""


def _assert_imports_clean(module: str):
    import pathlib

    src = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", _CHECK.format(src=src, module=module)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_backends_do_not_import_translator():
    _assert_imports_clean("backends.openai_compat")


def test_engine_does_not_import_translator():
    _assert_imports_clean("engine")


def test_profiles_do_not_import_translator():
    _assert_imports_clean("profiles")


def test_repair_does_not_import_translator():
    _assert_imports_clean("repair")


def test_global_glossary_does_not_import_translator():
    _assert_imports_clean("global_glossary")
