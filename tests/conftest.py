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
import tempfile
from pathlib import Path


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
