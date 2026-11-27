import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_proxy_settings_interactions():
    node = shutil.which("node")
    if node is None:
        msg = "node is required for the frontend interaction harness"
        if os.getenv("JAV_SUB_CI_STRICT") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, str(Path(__file__).parent / "js" / "proxy_settings.cjs"), str(root / "src/web/static/js/settings.js")],
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
