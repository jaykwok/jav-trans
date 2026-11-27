"""Cross-module contract tests pinning bootstrap.py against src/core/*, launcher.py, and packaging/*.

`bootstrap.py` MUST stay strictly Python standard-library-only so it can run
before `.venv` exists. Because it cannot import `src/core/*`, shared constants
and protocols are mirrored between `bootstrap.py` and `src/core/*`; this module
ensures they can never silently drift apart.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import bootstrap
import launcher
from core import atomic_io, config, diagnostics, updater

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_imports_only_python_stdlib_via_ast() -> None:
    """Hard invariant: every import anywhere in bootstrap.py must belong to stdlib."""
    source = (ROOT / "bootstrap.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="bootstrap.py")
    allowed = set(sys.stdlib_module_names) | set(sys.builtin_module_names) | {"__future__"}

    non_stdlib: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top not in allowed:
                    non_stdlib.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                non_stdlib.append(f".{node.module or ''}")
                continue
            if node.module:
                top = node.module.split(".", 1)[0]
                if top not in allowed:
                    non_stdlib.append(node.module)

    assert non_stdlib == [], f"bootstrap.py must only import stdlib, found: {non_stdlib}"


def test_updater_constants_and_security_rules_match_bootstrap() -> None:
    assert bootstrap.UPDATE_EXIT_CODE == updater.UPDATE_EXIT_CODE
    assert bootstrap.DEFAULT_UPDATE_REPO == updater.DEFAULT_UPDATE_REPO
    assert bootstrap.UPDATE_ARCHIVE_NAME == updater.UPDATE_ARCHIVE_NAME
    assert bootstrap.UPDATE_SHA256_NAME == updater.UPDATE_SHA256_NAME
    assert bootstrap.UPDATE_MANIFEST_NAME == updater.UPDATE_MANIFEST_NAME
    assert bootstrap.RELEASE_INDEX_NAME == updater.RELEASE_INDEX_NAME
    assert bootstrap.RELEASE_INDEX_SCHEMA == updater.RELEASE_INDEX_SCHEMA
    assert bootstrap.RELEASE_MANIFEST_SCHEMA == updater.RELEASE_MANIFEST_SCHEMA
    assert bootstrap.PENDING_UPDATE_SCHEMA == updater.PENDING_UPDATE_SCHEMA
    assert bootstrap.PENDING_UPDATE_FILE_NAME == updater.PENDING_UPDATE_FILE_NAME
    assert bootstrap._PROTECTED_UPDATE_TOP_DIRS == updater._PROTECTED_TOP_DIRS
    assert bootstrap._REQUIRED_PAYLOAD_ENTRIES == updater._REQUIRED_PAYLOAD_ENTRIES
    assert bootstrap._WINDOWS_RESERVED_DEVICE_NAMES == updater._WINDOWS_RESERVED_DEVICE_NAMES
    assert bootstrap._GITHUB_ALLOWED_REDIRECT_SUFFIXES == updater._GITHUB_ALLOWED_REDIRECT_SUFFIXES
    assert bootstrap._MAX_ARCHIVE_MEMBERS == updater._MAX_ARCHIVE_MEMBERS
    assert bootstrap._MAX_ARCHIVE_UNCOMPRESSED_BYTES == updater._MAX_ARCHIVE_UNCOMPRESSED_BYTES
    assert bootstrap._LOOPBACK_HOSTS == updater._LOOPBACK_HOSTS

    # Verify packaging manifest schema agrees with both updater and bootstrap.
    if str(ROOT / "packaging") not in sys.path:
        sys.path.insert(0, str(ROOT / "packaging"))
    import record_release_manifest

    assert record_release_manifest.MANIFEST_SCHEMA == updater.RELEASE_MANIFEST_SCHEMA
    assert record_release_manifest.MANIFEST_SCHEMA == bootstrap.RELEASE_MANIFEST_SCHEMA


def test_config_proxy_and_active_default_models_match_bootstrap(tmp_path: Path) -> None:
    assert set(bootstrap.PROXY_ENV_KEYS) == set(config._PROXY_ENV_KEYS)
    assert set(bootstrap.NO_PROXY_ENV_KEYS) == set(config._NO_PROXY_ENV_KEYS)
    assert set(bootstrap.SUPPORTED_PROXY_PROTOCOLS) == set(config._SUPPORTED_PROXY_PROTOCOLS)
    assert bootstrap._LOOPBACK_NO_PROXY == config._LOOPBACK_NO_PROXY

    # Core default models in DEFAULT_SETTINGS must ALWAYS be protected from retired-model pruning.
    protected_names, _protected_paths = bootstrap._active_model_keys(tmp_path)
    for key in ("LLAMACPP_MODEL_REPO", "ASR_BACKEND"):
        repo = config.DEFAULT_SETTINGS[key].strip("/")
        assert repo, f"DEFAULT_SETTINGS[{key!r}] must be non-empty"
        hub_name = f"models--{repo.replace('/', '--')}".casefold()
        flat_name = repo.replace("/", "-").casefold()
        assert hub_name in protected_names
        assert flat_name in protected_names


def test_diagnostics_ntstatus_and_exit_code_formatting_match_bootstrap() -> None:
    assert bootstrap._NTSTATUS_NAMES == diagnostics._NTSTATUS_NAMES
    for code in (None, 0, 1, 42, 0xC0000005, 0xC0000135, -1073741819):
        assert bootstrap._format_exit_code(code) == diagnostics.format_exit_code(code)


def test_workspace_gate_protocol_matches_across_bootstrap_launcher_and_atomic_io(
    tmp_path: Path,
) -> None:
    assert bootstrap._LAUNCHER_INSTANCE_DIR_NAME == launcher._INSTANCE_DIR_NAME
    assert bootstrap._LAUNCHER_INSTANCE_DIR_NAME == atomic_io._INSTANCE_DIR_NAME
    assert bootstrap._WORKSPACE_GATE_NAME == launcher._GATE_NAME
    assert bootstrap._WORKSPACE_GATE_NAME == atomic_io._GATE_NAME
    assert bootstrap._WORKSPACE_GATE_NAME == "workspace.gate"

    # Live OS mutex interop across all three modules.
    with atomic_io.workspace_gate(tmp_path, wait_s=1.0) as held:
        assert held is True
        assert bootstrap._other_launcher_instances_active(tmp_path, wait_s=0.0) is True
        assert launcher._acquire_gate(tmp_path, 0.05) is None

    assert bootstrap._other_launcher_instances_active(tmp_path, wait_s=0.0) is False
    gate = launcher._acquire_gate(tmp_path, 0.1)
    assert gate is not None
    try:
        assert atomic_io.other_launcher_instances_running(tmp_path, wait_s=0.0) is True
        assert bootstrap._other_launcher_instances_active(tmp_path, wait_s=0.0) is True
    finally:
        launcher._release_gate(gate)


def test_build_setup_ps1_is_strictly_ascii() -> None:
    """PowerShell 5.1 reads BOM-less .ps1 files using the system code page;
    non-ASCII bytes break packaging on non-UTF-8 Windows locales."""
    raw = (ROOT / "packaging" / "build_setup.ps1").read_bytes()
    non_ascii = [(idx, byte) for idx, byte in enumerate(raw) if byte > 127]
    assert non_ascii == [], f"packaging/build_setup.ps1 contains non-ASCII bytes: {non_ascii[:10]}"


@pytest.mark.parametrize(
    "origin,target,expected",
    [
        (
            "https://github.com/jaykwok/jav-sub/releases/download/v1.0.1/a.zip",
            "https://release-assets.githubusercontent.com/a.zip",
            True,
        ),
        (
            "https://github.com/jaykwok/jav-sub/releases/download/v1.0.1/a.zip",
            "http://release-assets.githubusercontent.com/a.zip",
            False,
        ),
        (
            "https://github.com/jaykwok/jav-sub/releases/download/v1.0.1/a.zip",
            "https://untrusted.example.com/a.zip",
            False,
        ),
        (
            "http://127.0.0.1:8080/latest",
            "http://127.0.0.1:8080/archive.zip",
            True,
        ),
    ],
)
def test_redirect_policy_matches_updater_exactly(
    origin: str, target: str, expected: bool
) -> None:
    assert bootstrap._is_allowed_update_redirect(origin, target) is expected
    assert updater._is_allowed_redirect_target(origin, target) is expected
