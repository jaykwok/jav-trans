"""Shared pytest configuration.

pytest's default ``tmp_path`` root lives under the OS temp directory
(``<tempdir>/pytest-of-<user>``). On some machines that directory has corrupted
permissions (e.g. Windows WinError 5), which makes every test using ``tmp_path``
fail before it even starts. Rather than pin a fixed ``--basetemp`` in ``addopts``
(which would clutter the working tree in every environment), probe the default
root once and fall back to a project-local directory only when the OS temp is
not writable.
"""

from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path

import pytest


# Tests must never inherit a developer machine's saved local backend.  The
# translation tests mock the OpenAI transport; if `.env` selects llama.cpp,
# those mocks are bypassed and pytest launches a real multi-GB llama-server in
# a second process alongside the Web app's server. Individual llama.cpp tests
# still opt in explicitly with monkeypatch.setenv after collection.
os.environ["TRANSLATION_BACKEND"] = "openai"


@pytest.fixture(autouse=True)
def _close_test_local_translation_backend():
    yield
    # Defence in depth for a test that explicitly opted into llama.cpp and then
    # failed before closing it. This only sees instances owned by the pytest
    # process; it cannot touch a server owned by the running Web application.
    try:
        from llm.backends import reset_backend

        reset_backend("llamacpp")
    except Exception:
        pass


def _default_tmp_root() -> Path:
    try:
        user = getpass.getuser() or "user"
    except Exception:
        user = "user"
    return Path(tempfile.gettempdir()) / f"pytest-of-{user}"


def pytest_configure(config):
    if config.option.basetemp:
        return
    root = _default_tmp_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".pytest-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        fallback = Path(__file__).resolve().parent.parent / "tmp" / "pytest"
        fallback.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(fallback)
