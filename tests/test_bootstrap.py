"""The installer runs before anything else exists, so its failures are silent.

There is no app to report an error through and no log to inspect: whatever
`bootstrap.py` gets wrong, the user sees only a console that closed. So the
properties pinned here are the ones with no second chance - that it imports
without any dependency installed, that it never destroys settings it did not
write, and that it cannot be tricked into reporting a broken environment as
ready.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import bootstrap

ROOT = Path(__file__).resolve().parents[1]


class TestStdlibOnly:
    def test_it_imports_with_no_third_party_packages(self) -> None:
        """The whole premise. If this ever needs a dependency, it cannot run on
        the machine it exists to set up."""
        completed = subprocess.run(
            [sys.executable, "-S", "-c", "import bootstrap"],
            cwd=str(ROOT),
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")


class TestConfigFile:
    def test_values_are_read(self, tmp_path: Path) -> None:
        path = tmp_path / "config" / "config.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            '{"PROXY_HOST": "127.0.0.1", "PROXY_PORT": 7890, "AUDIO_DYNAUDNORM": true}\n',
            encoding="utf-8",
        )
        assert bootstrap.read_config_file(path) == {
            "PROXY_HOST": "127.0.0.1",
            "PROXY_PORT": "7890",
            "AUDIO_DYNAUDNORM": "1",
        }

    def test_malformed_json_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        path = tmp_path / "config" / "config.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not valid json", encoding="utf-8")
        assert bootstrap.read_config_file(path) == {}

    def test_a_missing_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert bootstrap.read_config_file(tmp_path / "absent.json") == {}

    def test_a_proxy_switched_off_in_the_ui_yields_no_proxy(self, tmp_path: Path) -> None:
        path = tmp_path / "config" / "config.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            '{"PROXY_PROTOCOL": "http", "PROXY_HOST": "", "PROXY_PORT": ""}\n',
            encoding="utf-8",
        )
        assert bootstrap.proxy_url_from(bootstrap.read_config_file(path)) == ""

    def test_writing_preserves_unrelated_keys(self, tmp_path: Path) -> None:
        """The settings page owns this file too. Rewriting it wholesale would
        drop the user's API key on the next launch."""
        path = tmp_path / "config" / "config.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"API_KEY": "secret"}\n', encoding="utf-8")
        bootstrap.update_config_file({"PROXY_HOST": "127.0.0.1"}, path)
        assert bootstrap.read_config_file(path) == {
            "API_KEY": "secret",
            "PROXY_HOST": "127.0.0.1",
        }

    def test_an_existing_key_is_replaced_in_place(self, tmp_path: Path) -> None:
        path = tmp_path / "config" / "config.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"PROXY_HOST": "old", "API_KEY": "secret"}\n', encoding="utf-8")
        bootstrap.update_config_file({"PROXY_HOST": "new"}, path)
        assert bootstrap.read_config_file(path) == {"PROXY_HOST": "new", "API_KEY": "secret"}

    def test_writing_to_a_missing_file_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "config" / "config.json"
        bootstrap.update_config_file({"PROXY_HOST": "127.0.0.1"}, path)
        assert path.is_file()
        assert bootstrap.read_config_file(path)["PROXY_HOST"] == "127.0.0.1"

    def test_writing_refuses_to_overwrite_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "config" / "config.json"
        path.parent.mkdir(parents=True)
        broken = '{\n  "API_KEY": "keep-me",\n}\n'
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValueError, match="拒绝覆盖"):
            bootstrap.update_config_file({"PROXY_HOST": "127.0.0.1"}, path)
        assert path.read_text(encoding="utf-8") == broken


class TestProxyUrl:
    def test_host_and_port_compose_a_url(self) -> None:
        values = {"PROXY_PROTOCOL": "socks5", "PROXY_HOST": "127.0.0.1", "PROXY_PORT": "1080"}
        assert bootstrap.proxy_url_from(values) == "socks5://127.0.0.1:1080"

    def test_the_protocol_defaults_to_http(self) -> None:
        assert bootstrap.proxy_url_from({"PROXY_HOST": "h", "PROXY_PORT": "1"}) == "http://h:1"

    def test_an_unknown_protocol_falls_back_rather_than_failing(self) -> None:
        values = {"PROXY_PROTOCOL": "gopher", "PROXY_HOST": "h", "PROXY_PORT": "1"}
        assert bootstrap.proxy_url_from(values) == "http://h:1"

    def test_a_half_configured_proxy_is_no_proxy(self) -> None:
        """Better to attempt a direct download than to point uv at a URL with a
        missing port and have it fail with a connection error."""
        assert bootstrap.proxy_url_from({"PROXY_HOST": "127.0.0.1"}) == ""
        assert bootstrap.proxy_url_from({"PROXY_PORT": "7890"}) == ""
        assert bootstrap.proxy_url_from({}) == ""

    @pytest.mark.parametrize(
        "values",
        [
            {"PROXY_PROTOCOL": "socks5", "PROXY_HOST": "10.0.0.2", "PROXY_PORT": "1080"},
            {"PROXY_HOST": "10.0.0.2", "PROXY_PORT": "1080"},
            {"PROXY_PROTOCOL": "gopher", "PROXY_HOST": "h", "PROXY_PORT": "1"},
            {"PROXY_HOST": "10.0.0.2"},
        ],
    )
    def test_it_matches_the_rule_the_app_uses(
        self, values: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """bootstrap restates this rule because it cannot import src/. If the
        two ever disagree, the installer downloads through a different proxy
        than the app then uses for the ASR weights."""
        from core import config

        for key in ("PROXY_PROTOCOL", "PROXY_HOST", "PROXY_PORT"):
            monkeypatch.delenv(key, raising=False)
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        assert bootstrap.proxy_url_from(values) == config.network_proxy_url_from_env()

    def test_the_exported_keys_are_the_ones_the_app_reads(self) -> None:
        from core import config

        assert set(bootstrap.PROXY_ENV_KEYS) == set(config._PROXY_ENV_KEYS)


class TestApplyProxy:
    def test_every_variant_is_exported_and_loopback_no_proxy_merged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap.os, "environ", dict(bootstrap.os.environ))
        for key in bootstrap.PROXY_ENV_KEYS + bootstrap.NO_PROXY_ENV_KEYS:
            bootstrap.os.environ.pop(key, None)
        bootstrap.os.environ["NO_PROXY"] = "corp.internal"
        bootstrap.apply_proxy("http://127.0.0.1:7890")
        for key in bootstrap.PROXY_ENV_KEYS:
            assert bootstrap.os.environ[key] == "http://127.0.0.1:7890"
        for key in bootstrap.NO_PROXY_ENV_KEYS:
            entries = {item.strip() for item in bootstrap.os.environ[key].split(",")}
            assert {"corp.internal", "127.0.0.1", "localhost", "::1"} <= entries

    def test_an_empty_url_preserves_shell_proxy_by_default_and_clears_when_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default clear_existing=False keeps an external shell proxy intact;
        explicit clear_existing=True removes stale proxy variables."""
        monkeypatch.setattr(bootstrap.os, "environ", dict(bootstrap.os.environ))
        bootstrap.apply_proxy("http://127.0.0.1:7890")
        bootstrap.apply_proxy("")
        for key in bootstrap.PROXY_ENV_KEYS:
            assert bootstrap.os.environ[key] == "http://127.0.0.1:7890"

        bootstrap.apply_proxy("", clear_existing=True)
        for key in bootstrap.PROXY_ENV_KEYS:
            assert key not in bootstrap.os.environ
        assert bootstrap.os.environ["PROXY_HOST"] == ""
        assert bootstrap.os.environ["PROXY_PORT"] == ""

    def test_opener_falls_back_to_all_proxy_and_preserves_system_proxy_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap.os, "environ", dict(bootstrap.os.environ))
        for key in bootstrap.PROXY_ENV_KEYS:
            bootstrap.os.environ.pop(key, None)
        bootstrap.os.environ["ALL_PROXY"] = "socks5://127.0.0.1:1080"
        handler = bootstrap._build_proxy_handler()
        assert handler.proxies == {
            "http": "socks5://127.0.0.1:1080",
            "https": "socks5://127.0.0.1:1080",
        }

        bootstrap.os.environ.pop("ALL_PROXY", None)
        sentinel = {"http": "http://system-proxy.local:8080"}
        monkeypatch.setattr(bootstrap.urllib.request, "getproxies", lambda: dict(sentinel))
        fallback_handler = bootstrap._build_proxy_handler()
        assert fallback_handler.proxies == sentinel


class TestProxyPromptAndConfigSafety:
    def test_prompt_proxy_strips_pasted_url_and_host_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answers = iter(["socks5://127.0.0.1:1080"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
        assert bootstrap.prompt_proxy() == {
            "PROXY_PROTOCOL": "socks5",
            "PROXY_HOST": "127.0.0.1",
            "PROXY_PORT": "1080",
        }

        answers2 = iter(["127.0.0.1:7890", "https"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(answers2))
        assert bootstrap.prompt_proxy() == {
            "PROXY_PROTOCOL": "https",
            "PROXY_HOST": "127.0.0.1",
            "PROXY_PORT": "7890",
        }

    @pytest.mark.parametrize(
        "inputs",
        [
            ["127.0.0.1", "0", "http"],
            ["127.0.0.1", "99999", "http"],
            ["127.0.0.1", "abc", "http"],
            ["127.0.0.1/bad"],
            ["127.0.0.1 7890"],
            ["http://127.0.0.1:7890/extra"],
            ["http://127.0.0.1:99999"],
            ["socks4://127.0.0.1:1080"],
        ],
    )
    def test_prompt_proxy_rejects_invalid_inputs(
        self, inputs: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answers = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
        assert bootstrap.prompt_proxy() is None

    def test_ensure_proxy_configured_validates_before_persisting_and_survives_bad_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "config" / "config.json"
        cfg.parent.mkdir(parents=True)
        # Pre-existing corrupt proxy in config.json must NOT lock out launcher / --doctor.
        cfg.write_text('{"PROXY_HOST": "127.0.0.1", "PROXY_PORT": "99999"}\n', encoding="utf-8")
        monkeypatch.setattr(bootstrap, "CONFIG_PATH", cfg)
        monkeypatch.setattr(bootstrap.os, "environ", dict(bootstrap.os.environ))
        monkeypatch.setattr(bootstrap, "interactive", lambda: False)
        bootstrap.ensure_proxy_configured(assume_yes=False)

        # Interactive retry: if prompt_proxy yields invalid settings, apply_proxy rejects them
        # BEFORE update_config_file is ever called; valid retry then persists cleanly.
        monkeypatch.setattr(bootstrap, "interactive", lambda: True)
        monkeypatch.setattr(bootstrap, "measure_download", lambda: (1024 * 1024.0, 1024 * 1024))
        yes_answers = iter([True, True])
        monkeypatch.setattr(bootstrap, "ask_yes", lambda _q, **_k: next(yes_answers))
        prompts = iter(
            [
                {"PROXY_PROTOCOL": "http", "PROXY_HOST": "bad host", "PROXY_PORT": "99999"},
                {"PROXY_PROTOCOL": "http", "PROXY_HOST": "127.0.0.1", "PROXY_PORT": "7890"},
            ]
        )
        monkeypatch.setattr(bootstrap, "prompt_proxy", lambda: next(prompts))
        writes: list[dict[str, str]] = []
        real_update = bootstrap.update_config_file

        def spy_update(settings: dict[str, str], path: Path | None = None) -> None:
            writes.append(dict(settings))
            real_update(settings, path)

        monkeypatch.setattr(bootstrap, "update_config_file", spy_update)
        bootstrap.ensure_proxy_configured(assume_yes=False)
        assert writes == [{"PROXY_PROTOCOL": "http", "PROXY_HOST": "127.0.0.1", "PROXY_PORT": "7890"}]
        assert bootstrap.read_config_file(cfg)["PROXY_PORT"] == "7890"

    def test_update_config_file_cleans_temp_file_on_replace_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "config" / "config.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text('{"API_KEY": "keep"}\n', encoding="utf-8")
        monkeypatch.setattr(bootstrap, "_replace_atomically", lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("locked")))
        with pytest.raises(PermissionError):
            bootstrap.update_config_file({"PROXY_HOST": "127.0.0.1"}, cfg)
        assert list(cfg.parent.glob("*.tmp")) == []
        assert bootstrap.read_config_file(cfg) == {"API_KEY": "keep"}


class TestProbeTarget:
    def test_the_lockfile_is_present_in_a_checkout(self) -> None:
        """It is tracked on purpose. The release ships it and installs with
        `uv sync --frozen`, so without it a clone cannot build the artifact
        users are running."""
        assert bootstrap.LOCK_PATH.is_file()

    def test_the_real_torch_wheel_is_read_from_the_lockfile(self) -> None:
        """Measuring against the file the user actually downloads, from the host
        it actually comes from - PyPI being quick says nothing about the CUDA
        index."""
        url = bootstrap.torch_wheel_url()
        assert "torch-" in url and url.endswith("win_amd64.whl")

    def test_the_free_threaded_build_is_not_chosen(self) -> None:
        assert "cp314t" not in bootstrap.torch_wheel_url()

    def test_a_missing_lockfile_degrades_to_the_fallback(self, tmp_path: Path) -> None:
        assert bootstrap.torch_wheel_url(tmp_path / "absent") == ""

    def test_a_lockfile_without_torch_degrades_to_the_fallback(self, tmp_path: Path) -> None:
        path = tmp_path / "uv.lock"
        path.write_text('[[package]]\nname = "numpy"\n', encoding="utf-8")
        assert bootstrap.torch_wheel_url(path) == ""

    def test_another_package_named_like_torch_is_not_matched(self, tmp_path: Path) -> None:
        """`torchcodec` and `torchvision` sort next to torch and are three
        orders of magnitude smaller."""
        path = tmp_path / "uv.lock"
        path.write_text(
            '[[package]]\nname = "torchcodec"\n'
            'wheels = [{ url = "https://example/torchcodec-1-win_amd64.whl" }]\n',
            encoding="utf-8",
        )
        assert bootstrap.torch_wheel_url(path) == ""


class TestSpeedReport:
    def test_the_estimate_is_the_measured_rate_applied_to_the_real_size(self) -> None:
        text = bootstrap.describe_speed((1024 ** 2, 1024 ** 3 * 2))
        assert "1.00 MB/s" in text
        assert "2.0GB" in text
        assert "约 34 分钟" in text

    def test_a_failed_probe_says_so_instead_of_reporting_zero(self) -> None:
        """A silent 0 MB/s reads as "fine, continue" and produces an install
        that hangs for an hour."""
        assert "连接失败" in bootstrap.describe_speed(None)

    def test_long_downloads_are_reported_in_hours(self) -> None:
        assert "小时" in bootstrap.describe_speed((20 * 1024, 1024 ** 3 * 2))

    def test_a_short_download_is_not_rounded_to_zero_minutes(self) -> None:
        assert bootstrap.format_duration(20.0) == "不到 1 分钟"


class TestReadiness:
    def test_a_venv_without_a_stamp_is_not_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An interrupted first run leaves .venv behind with a partial install.
        Treating the directory as proof would launch straight into ImportError."""
        python = tmp_path / ".venv" / ("Scripts/python.exe" if bootstrap.os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "VENV_PATH", tmp_path / ".venv")
        monkeypatch.setattr(bootstrap, "STAMP_PATH", tmp_path / ".venv" / "stamp")
        assert bootstrap.environment_is_current() is False

    def test_a_stamp_without_a_venv_is_not_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp = tmp_path / "stamp"
        stamp.write_text(bootstrap.lock_digest(), encoding="utf-8")
        monkeypatch.setattr(bootstrap, "VENV_PATH", tmp_path / ".venv")
        monkeypatch.setattr(bootstrap, "STAMP_PATH", stamp)
        assert bootstrap.environment_is_current() is False

    def test_a_matching_stamp_skips_the_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        python = tmp_path / ".venv" / ("Scripts/python.exe" if bootstrap.os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        stamp = tmp_path / ".venv" / "stamp"
        stamp.write_text(bootstrap.lock_digest(), encoding="utf-8")
        monkeypatch.setattr(bootstrap, "VENV_PATH", tmp_path / ".venv")
        monkeypatch.setattr(bootstrap, "STAMP_PATH", stamp)
        assert bootstrap.environment_is_current() is True

    def test_a_changed_lockfile_forces_a_resync(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This is how a patch release that bumps a dependency reaches users who
        already installed the previous one."""
        python = tmp_path / ".venv" / ("Scripts/python.exe" if bootstrap.os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        stamp = tmp_path / ".venv" / "stamp"
        stamp.write_text("digest of the previous release", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "VENV_PATH", tmp_path / ".venv")
        monkeypatch.setattr(bootstrap, "STAMP_PATH", stamp)
        assert bootstrap.environment_is_current() is False

    def test_the_digest_covers_both_dependency_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock = tmp_path / "uv.lock"
        lock.write_text("a", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "LOCK_PATH", lock)
        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        before = bootstrap.lock_digest()
        lock.write_text("b", encoding="utf-8")
        assert bootstrap.lock_digest() != before
        (tmp_path / "pyproject.toml").write_text("c", encoding="utf-8")
        assert bootstrap.lock_digest() != before


class TestVersionOnlyRelease:
    """Every release bumps the project's own version in pyproject.toml and in
    the lock. The project is never installed into .venv, so a release that
    changes nothing else must not look like a dependency change: that costs a
    pointless sync and, before it, the speed test and the proxy question."""

    @staticmethod
    def _dependency_files(root: Path, *, version: str = "1.8.2", numpy: str = "2.3.0",
                          source: str = 'virtual = "."', requirement: str = "numpy>=2") -> None:
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "jav-sub"\nversion = "{version}"\ndependencies = ["{requirement}"]\n',
            encoding="utf-8",
        )
        (root / "uv.lock").write_text(
            'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n\n'
            f'[[package]]\nname = "jav-sub"\nversion = "{version}"\nsource = {{ {source} }}\n'
            'dependencies = [{ name = "numpy" }]\n\n'
            f'[[package]]\nname = "numpy"\nversion = "{numpy}"\n'
            'source = { registry = "https://pypi.org/simple" }\n',
            encoding="utf-8",
        )

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        monkeypatch.setattr(bootstrap, "LOCK_PATH", tmp_path / "uv.lock")
        return tmp_path

    def test_a_version_bump_keeps_an_installed_environment_current(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        python = root / ".venv" / ("Scripts/python.exe" if bootstrap.os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        stamp = root / ".venv" / "stamp"
        monkeypatch.setattr(bootstrap, "VENV_PATH", root / ".venv")
        monkeypatch.setattr(bootstrap, "STAMP_PATH", stamp)
        self._dependency_files(root, version="1.8.2")
        stamp.write_text(bootstrap.lock_digest(), encoding="utf-8")
        self._dependency_files(root, version="1.8.3")
        assert bootstrap.environment_is_current() is True

    @pytest.mark.parametrize(
        "change",
        [{"numpy": "2.3.1"}, {"requirement": "numpy>=2.3"}],
        ids=["locked-version", "declared-requirement"],
    )
    def test_a_dependency_change_still_forces_a_resync(self, root: Path, change: dict) -> None:
        self._dependency_files(root)
        before = bootstrap.lock_digest()
        self._dependency_files(root, version="1.8.3", **change)
        assert bootstrap.lock_digest() != before

    def test_the_version_counts_once_the_project_is_installed(self, root: Path) -> None:
        """An editable install writes the version into .venv; only a virtual
        project may leave it out."""
        self._dependency_files(root, source='editable = "."')
        before = bootstrap.lock_digest()
        self._dependency_files(root, version="1.8.3", source='editable = "."')
        assert bootstrap.lock_digest() != before

    def test_the_shipped_dependency_files_ignore_a_version_bump(self, root: Path) -> None:
        """Against the real files, so a change in uv's lock format that hides
        the project entry fails here instead of silently re-prompting users."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
        version = bootstrap.tomllib.loads(pyproject)["project"]["version"]
        (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        (root / "uv.lock").write_text(lock, encoding="utf-8")
        before = bootstrap.lock_digest()

        project_line = f'version = "{version}"'
        lock_entry = f'name = "jav-sub"\nversion = "{version}"'
        assert pyproject.count(project_line) == 1 and lock.count(lock_entry) == 1
        (root / "pyproject.toml").write_text(pyproject.replace(project_line, 'version = "99.0.0"'), encoding="utf-8")
        (root / "uv.lock").write_text(
            lock.replace(lock_entry, 'name = "jav-sub"\nversion = "99.0.0"'), encoding="utf-8"
        )
        assert bootstrap.lock_digest() == before


class TestUvDiscovery:
    def test_the_bundled_copy_wins_over_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A release must install with the uv it shipped, not with whatever
        version the user happens to have."""
        name = "uv.exe" if bootstrap.os.name == "nt" else "uv"
        bundled = tmp_path / name
        bundled.write_text("", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        monkeypatch.setattr(bootstrap.shutil, "which", lambda _: "C:/elsewhere/uv.exe")
        assert bootstrap.find_uv() == bundled

    def test_path_is_the_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        monkeypatch.setattr(bootstrap.shutil, "which", lambda _: str(tmp_path / "found"))
        assert bootstrap.find_uv() == tmp_path / "found"

    def test_an_earlier_download_is_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user who lost their network mid-install must not pay for the uv
        download a second time."""
        name = "uv.exe" if bootstrap.os.name == "nt" else "uv"
        fetched = tmp_path / "bin" / name
        fetched.parent.mkdir()
        fetched.write_text("", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        monkeypatch.setattr(bootstrap.shutil, "which", lambda _: None)
        assert bootstrap.find_uv() == fetched

    def test_no_uv_anywhere_is_reported_rather_than_guessed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        monkeypatch.setattr(bootstrap.shutil, "which", lambda _: None)
        assert bootstrap.find_uv() is None


class TestUvDownload:
    INDEX = """<!DOCTYPE html><html><body>
    <a href="https://files.pythonhosted.org/x/uv-0.9.0-py3-none-win_amd64.whl#sha256=aa">uv-0.9.0</a>
    <a href="https://files.pythonhosted.org/x/uv-0.12.1-py3-none-win_amd64.whl#sha256=bb">uv-0.12.1</a>
    <a href="https://files.pythonhosted.org/x/uv-0.11.29-py3-none-win_amd64.whl#sha256=cc">uv-0.11.29</a>
    <a href="https://files.pythonhosted.org/x/uv-0.12.1-py3-none-manylinux_2_17_x86_64.whl">linux</a>
    <a href="https://files.pythonhosted.org/x/uv-0.12.1.tar.gz#sha256=dd">sdist</a>
    </body></html>"""

    def test_the_newest_windows_wheel_is_chosen(self) -> None:
        url = bootstrap.latest_uv_wheel_url(self.INDEX)
        assert url.endswith("uv-0.12.1-py3-none-win_amd64.whl")

    def test_versions_are_compared_numerically_not_as_text(self) -> None:
        """0.9.0 sorts after 0.12.1 as a string, and the index is ordered by
        upload time rather than by version."""
        assert "0.9.0" not in bootstrap.latest_uv_wheel_url(self.INDEX)

    def test_the_hash_fragment_is_stripped(self) -> None:
        assert "#" not in bootstrap.latest_uv_wheel_url(self.INDEX)

    def test_other_platforms_and_sdists_are_ignored(self) -> None:
        index = """<a href="https://x/uv-0.12.1-py3-none-manylinux_2_17_x86_64.whl">l</a>
                   <a href="https://x/uv-0.12.1.tar.gz">s</a>"""
        assert bootstrap.latest_uv_wheel_url(index) == ""

    def test_an_empty_index_yields_nothing_rather_than_raising(self) -> None:
        assert bootstrap.latest_uv_wheel_url("") == ""

    def test_skips_yanked_and_prerelease_and_resolves_relative_url_with_sha256(self) -> None:
        digest = "a" * 64
        index = f"""<!DOCTYPE html><html><body>
        <a href="../../packages/uv-0.14.0-py3-none-win_amd64.whl#sha256={"b" * 64}" data-yanked="broken">yanked</a>
        <a href="../../packages/uv-0.15.0rc1-py3-none-win_amd64.whl#sha256={"c" * 64}">prerelease</a>
        <a href="../../packages/uv-0.13.2-py3-none-win_amd64.whl?foo=1&amp;bar=2#sha256={digest}">stable</a>
        </body></html>"""
        url, sha = bootstrap._latest_uv_wheel(index, base_url="https://mirror.example/simple/uv/")
        assert url == "https://mirror.example/packages/uv-0.13.2-py3-none-win_amd64.whl?foo=1&bar=2"
        assert sha == digest

    def test_download_uv_verifies_sha256_and_replaces_atomically(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io
        import zipfile

        wheel_buf = io.BytesIO()
        with zipfile.ZipFile(wheel_buf, "w") as zf:
            zf.writestr("uv-0.13.2.data/scripts/uv.exe", b"real-uv-exe")
        wheel_bytes = wheel_buf.getvalue()
        good_sha = bootstrap.hashlib.sha256(wheel_bytes).hexdigest()
        bad_sha = "0" * 64

        class _Resp(io.BytesIO):
            def __init__(self, data: bytes, url: str) -> None:
                super().__init__(data)
                self._url = url
                self.headers = {"Content-Length": str(len(data))}

            def geturl(self) -> str:
                return self._url

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        monkeypatch.setattr(bootstrap, "UV_PYPI_INDEX", "https://mirror.example/simple/uv/")

        bad_index = f'<a href="../../packages/uv-0.13.2-py3-none-win_amd64.whl#sha256={bad_sha}">uv</a>'.encode()
        good_index = f'<a href="../../packages/uv-0.13.2-py3-none-win_amd64.whl#sha256={good_sha}">uv</a>'.encode()
        wheel_url = "https://mirror.example/packages/uv-0.13.2-py3-none-win_amd64.whl"

        current_index = [bad_index]

        class _FakeOpener:
            def open(self, req, timeout=20.0):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if url == bootstrap.UV_PYPI_INDEX:
                    return _Resp(current_index[0], url)
                assert url == wheel_url
                return _Resp(wheel_bytes, url)

        monkeypatch.setattr(bootstrap, "_opener", lambda: _FakeOpener())
        assert bootstrap.download_uv() is None
        assert not (tmp_path / "bin" / ("uv.exe" if bootstrap.os.name == "nt" else "uv")).exists()

        current_index[0] = good_index
        installed = bootstrap.download_uv()
        assert installed is not None and installed.read_bytes() == b"real-uv-exe"
        assert list((tmp_path / "bin").glob("*.tmp")) == []

    @pytest.mark.skipif(
        bootstrap.os.getenv("JAV_SUB_NETWORK_TESTS") != "1",
        reason="Set JAV_SUB_NETWORK_TESTS=1 to check the live PyPI index",
    )
    def test_the_real_index_still_parses(self) -> None:
        """Guards against PyPI changing its simple-index markup: the installer
        would otherwise report "没有 Windows 版 uv" on a machine that has none."""
        try:
            with bootstrap._opener().open(
                bootstrap.urllib.request.Request(
                    bootstrap.UV_PYPI_INDEX,
                    headers={"Accept": "application/vnd.pypi.simple.v1+html"},
                ),
                timeout=15,
            ) as response:
                index = response.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - offline is not a failure of this code
            pytest.skip("PyPI unreachable")
        assert bootstrap.latest_uv_wheel_url(index).endswith("win_amd64.whl")


class TestOverwriteUpgrade:
    """Extracting a new release over the old folder replaces files, never removes
    one. What the new release dropped stays behind - and a leftover package
    directory shadows a new module of the same name - unless the launch moves it
    out of the way using the index the build ships."""

    SHIPPED = (
        "launcher.py",
        "src/main.py",
        "src/llm/__init__.py",
        "src/llm/backend.py",
        "bin/ffmpeg.exe",
        "bin/avcodec-62.dll",
    )

    @staticmethod
    def _touch(root: Path, *paths: str) -> None:
        for relative in paths:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def _upgraded(self, root: Path, *leftovers: str) -> Path:
        """An install that had an older release and then got this one on top."""
        self._touch(root, *self.SHIPPED)
        bootstrap.write_release_index(root)
        self._touch(root, *leftovers)
        return root

    def _run_build_script(self, payload: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "packaging" / "write_release_index.py"), "--payload", str(payload)],
            capture_output=True,
        )

    def test_the_index_lists_every_shipped_file_but_itself(self, tmp_path: Path) -> None:
        self._touch(tmp_path, *self.SHIPPED)
        index = bootstrap.write_release_index(tmp_path)
        assert index.name == bootstrap.RELEASE_INDEX_NAME
        assert bootstrap.read_release_index(tmp_path) == frozenset(self.SHIPPED)

    def test_a_package_directory_left_by_an_older_release_is_moved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case that breaks a launch: src/llm/backend/ wins over the
        src/llm/backend.py the new release ships."""
        monkeypatch.setattr(bootstrap.time, "strftime", lambda _: "20260929_120000")
        root = self._upgraded(
            tmp_path,
            "src/llm/backend/__init__.py",
            "src/llm/backend/__pycache__/__init__.cpython-314.pyc",
        )
        moved, leftovers = bootstrap.prune_stale_release_files(root)
        assert moved == ["src/llm/backend"]
        assert leftovers == root / bootstrap.LEFTOVERS_DIR / "20260929_120000"
        assert (root / "src/llm/backend.py").is_file()
        assert not (root / "src/llm/backend").exists()
        # Moved, not deleted: someone may have edited it.
        assert (leftovers / "src/llm/backend/__init__.py").is_file()

    def test_dropped_modules_and_ffmpeg_dlls_go_and_the_rest_stays(self, tmp_path: Path) -> None:
        root = self._upgraded(
            tmp_path,
            "src/llm/old_client.py",
            "src/retired/__init__.py",
            "bin/avcodec-61.dll",
            # Not the release's: uv is downloaded on the first run, and a cache
            # is harmless once its source is gone.
            "bin/uv.exe",
            "src/llm/__pycache__/backend.cpython-314.pyc",
        )
        moved, _ = bootstrap.prune_stale_release_files(root)
        assert sorted(moved) == ["bin/avcodec-61.dll", "src/llm/old_client.py", "src/retired"]
        for relative in (*self.SHIPPED, "bin/uv.exe", "src/llm/__pycache__/backend.cpython-314.pyc"):
            assert (root / relative).is_file(), relative

    def test_other_tools_in_bin_are_left_whole(self, tmp_path: Path) -> None:
        """A llama.cpp build unpacked into bin/ carries its own DLLs. Moving
        those but not llama-server.exe would break local translation."""
        tools = (
            "bin/llama-server.exe",
            "bin/llama.dll",
            "bin/ggml-cuda.dll",
            "bin/ggml-base.dll",
            "bin/cudart64_12.dll",
        )
        root = self._upgraded(tmp_path, *tools, "bin/postproc-58.dll", "bin/SWSCALE-9.DLL")
        moved, _ = bootstrap.prune_stale_release_files(root)
        assert sorted(moved) == ["bin/SWSCALE-9.DLL", "bin/postproc-58.dll"]
        for relative in tools:
            assert (root / relative).is_file(), relative

    def test_an_old_spelling_of_a_shipped_path_is_the_same_path(self, tmp_path: Path) -> None:
        """Copying src/llm/ over an old src/LLM/ writes into the existing folder
        and keeps its spelling, as Windows does. The new release's files in it
        are not leftovers."""
        payload = tmp_path / "payload"
        self._touch(payload, *self.SHIPPED)
        bootstrap.write_release_index(payload)
        root = tmp_path / "install"
        self._touch(
            root,
            "launcher.py",
            "src/Main.py",
            "src/LLM/__init__.py",
            "src/LLM/Backend.py",
            "bin/FFmpeg.exe",
            "bin/AVCODEC-62.DLL",
        )
        shutil.copy2(payload / bootstrap.RELEASE_INDEX_NAME, root / bootstrap.RELEASE_INDEX_NAME)
        assert bootstrap.prune_stale_release_files(root) == ([], None)
        assert (root / "src/LLM/Backend.py").is_file()

    def test_a_second_upgrade_in_the_same_second_keeps_the_first_leftovers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap.time, "strftime", lambda _: "20260929_120000")
        root = self._upgraded(tmp_path, "src/old.py")
        earlier = root / bootstrap.LEFTOVERS_DIR / "20260929_120000"
        self._touch(earlier, "src/old.py")
        _, leftovers = bootstrap.prune_stale_release_files(root)
        assert leftovers == earlier.with_name("20260929_120000-1")
        assert (earlier / "src/old.py").is_file()

    def test_a_file_that_will_not_move_is_skipped_not_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A leftover costs less than a launcher that refuses to start."""
        root = self._upgraded(tmp_path, "src/locked.py", "src/old.py")
        real_move = bootstrap.shutil.move

        def move(source: str, destination: str) -> str:
            if source.endswith("locked.py"):
                raise PermissionError("in use")
            return real_move(source, destination)

        monkeypatch.setattr(bootstrap.shutil, "move", move)
        moved, _ = bootstrap.prune_stale_release_files(root)
        assert moved == ["src/old.py"]
        assert (root / "src/locked.py").is_file()

    def test_a_git_checkout_is_never_touched(self, tmp_path: Path) -> None:
        """src/ there is the developer's work in progress."""
        root = self._upgraded(tmp_path, "src/new_module.py")
        (root / ".git").mkdir()
        assert bootstrap.prune_stale_release_files(root) == ([], None)
        assert (root / "src/new_module.py").is_file()

    def test_without_an_index_nothing_is_touched(self, tmp_path: Path) -> None:
        """Archives from before the index existed ship none."""
        self._touch(tmp_path, *self.SHIPPED, "src/old.py")
        assert bootstrap.prune_stale_release_files(tmp_path) == ([], None)

    @pytest.mark.parametrize(
        "content",
        [
            '{"schema": "jav_sub_release_files_v1", "files": ["README.txt"]}',
            '{"schema": "jav_sub_release_files_v1", "files": ["launcher.py"]}',
            '{"schema": "some_other_schema", "files": ["launcher.py", "src/main.py"]}',
            '{"schema": "jav_sub_release_files_v1", "files": [',
        ],
    )
    def test_an_index_that_would_empty_src_is_ignored(self, tmp_path: Path, content: str) -> None:
        """Everything an index leaves out counts as stale, so a truncated one
        must read as no index at all."""
        self._touch(tmp_path, *self.SHIPPED)
        (tmp_path / bootstrap.RELEASE_INDEX_NAME).write_text(content, encoding="utf-8")
        assert bootstrap.read_release_index(tmp_path) is None
        assert bootstrap.prune_stale_release_files(tmp_path) == ([], None)
        assert (tmp_path / "src/llm/backend.py").is_file()

    def test_the_launch_prunes_before_anything_reads_src(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._upgraded(tmp_path, "src/llm/backend/__init__.py")
        monkeypatch.setattr(bootstrap, "ROOT", root)
        seen = []

        class Stop(Exception):
            pass

        def diagnose(*args, **kwargs):
            seen.append((root / "src/llm/backend").exists())
            raise Stop

        monkeypatch.setattr(bootstrap, "diagnose", diagnose)
        with pytest.raises(Stop):
            bootstrap.main([])
        assert seen == [False]

    def test_the_build_script_writes_an_index_the_launcher_accepts(self, tmp_path: Path) -> None:
        self._touch(tmp_path, *self.SHIPPED)
        completed = self._run_build_script(tmp_path)
        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
        assert bootstrap.read_release_index(tmp_path) == frozenset(self.SHIPPED)

    def test_the_build_script_refuses_a_payload_the_launcher_would_ignore(self, tmp_path: Path) -> None:
        """Shipping an unusable index would silently turn the cleanup off."""
        self._touch(tmp_path, "launcher.py")
        assert self._run_build_script(tmp_path).returncode == 1


class TestNestedExtraction:
    """Extracting the new release *into* the old folder, rather than over it,
    starts a second install that downloads every model again."""

    def test_an_installed_parent_is_recognised(self, tmp_path: Path) -> None:
        (tmp_path / "launcher.py").write_text("", encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / "models").mkdir()
        assert bootstrap.enclosing_install(tmp_path / "jav-sub") == tmp_path

    def test_a_parent_that_never_installed_is_not(self, tmp_path: Path) -> None:
        """An extracted-but-never-run copy one level up has nothing to lose."""
        (tmp_path / "launcher.py").write_text("", encoding="utf-8")
        (tmp_path / "src").mkdir()
        assert bootstrap.enclosing_install(tmp_path / "jav-sub") is None

    def test_an_ordinary_parent_is_not(self, tmp_path: Path) -> None:
        (tmp_path / "models").mkdir()
        assert bootstrap.enclosing_install(tmp_path / "jav-sub") is None

    def test_the_first_run_asks_before_installing_a_second_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "launcher.py").write_text("", encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / ".venv").mkdir()
        nested = tmp_path / "jav-sub"
        (nested / "src").mkdir(parents=True)
        (nested / "launcher.py").write_text("", encoding="utf-8")
        (nested / "pyproject.toml").write_text("", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "ROOT", nested)
        monkeypatch.setattr(bootstrap, "VENV_PATH", nested / ".venv")
        monkeypatch.setattr(bootstrap, "LOCK_PATH", nested / "uv.lock")
        monkeypatch.setattr(bootstrap, "STAMP_PATH", nested / ".venv" / "stamp")
        monkeypatch.setattr(bootstrap, "interactive", lambda: True)
        questions = []
        monkeypatch.setattr(bootstrap, "ask_yes", lambda question, **_: questions.append(question) or False)
        monkeypatch.setattr(bootstrap, "ensure_uv", lambda: pytest.fail("declined, nothing to install"))
        assert bootstrap.main([]) == 1
        assert len(questions) == 1


class TestRetiredModelCleanup:
    """In-place upgrades reclaim multi-GB caches of retired local models while
    sparing Hy-MT3, active Qwen3-ASR 1.7B, the managed llama runtime, custom
    folders, and anything explicitly referenced in `.env`."""

    SHIPPED = ("launcher.py", "src/main.py")

    def _upgraded(self, tmp_path: Path, *extra_files: str) -> Path:
        root = tmp_path / "install"
        for relative in (*self.SHIPPED, *extra_files):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 16)
        payload = tmp_path / "payload"
        for relative in self.SHIPPED:
            path = payload / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        bootstrap.write_release_index(payload)
        shutil.copy2(payload / bootstrap.RELEASE_INDEX_NAME, root / bootstrap.RELEASE_INDEX_NAME)
        return root

    def test_removes_retired_models_and_spares_active_ones(self, tmp_path: Path) -> None:
        root = self._upgraded(
            tmp_path,
            # Retired local translation & 0.6B ASR caches (should be removed)
            "models/hub/models--tencent--Hy-MT2-7B-GGUF/blobs/old.gguf",
            "models/hub/.locks/models--tencent--Hy-MT2-7B-GGUF/old.lock",
            "models/hub/models--tencent--Hy-MT2-1.8B-GGUF/blobs/small.gguf",
            "models/hub/models--SakuraLLM--Sakura-7B-Qwen2.5-v1.0-GGUF/blobs/s.gguf",
            "models/hub/models--Qwen--Qwen3-8B/blobs/q.bin",
            "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf/model.safetensors",
            # Active models, runtime, and unknown user folders (must stay)
            "models/hub/models--tencent--Hy-MT3-7B-GGUF/blobs/mt3.gguf",
            "models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/model.safetensors",
            "models/hub/models--jaykwok--Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/blobs/ctc.pt",
            "models/runtime/llama-cpp/llama-server.exe",
            "models/custom-user-model/weights.gguf",
        )

        removed, reclaimed = bootstrap.prune_retired_model_caches(root)
        assert sorted(removed) == [
            "models/hub/models--Qwen--Qwen3-8B",
            "models/hub/models--SakuraLLM--Sakura-7B-Qwen2.5-v1.0-GGUF",
            "models/hub/models--tencent--Hy-MT2-1.8B-GGUF",
            "models/hub/models--tencent--Hy-MT2-7B-GGUF",
            "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf",
        ]
        assert reclaimed == 5 * 16
        assert not (root / "models/hub/models--tencent--Hy-MT2-7B-GGUF").exists()
        assert not (root / "models/hub/.locks/models--tencent--Hy-MT2-7B-GGUF").exists()
        assert (root / "models/hub/models--tencent--Hy-MT3-7B-GGUF/blobs/mt3.gguf").is_file()
        assert (root / "models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/model.safetensors").is_file()
        assert (root / "models/hub/models--jaykwok--Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/blobs/ctc.pt").is_file()
        assert (root / "models/runtime/llama-cpp/llama-server.exe").is_file()
        assert (root / "models/custom-user-model/weights.gguf").is_file()

    def test_spares_retired_model_still_configured_in_config(self, tmp_path: Path) -> None:
        root = self._upgraded(
            tmp_path,
            "models/hub/models--tencent--Hy-MT2-7B-GGUF/blobs/old.gguf",
            "models/hub/models--tencent--Hy-MT2-1.8B-GGUF/blobs/other.gguf",
        )
        cfg = root / "config" / "config.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(
            '{"LLAMACPP_MODEL_REPO": "tencent/Hy-MT2-7B-GGUF"}\n',
            encoding="utf-8",
        )
        removed, _ = bootstrap.prune_retired_model_caches(root)
        assert removed == ["models/hub/models--tencent--Hy-MT2-1.8B-GGUF"]
        assert (root / "models/hub/models--tencent--Hy-MT2-7B-GGUF/blobs/old.gguf").is_file()

    def test_prune_legacy_env_file_removes_env_only_for_packaged_upgrade(self, tmp_path: Path) -> None:
        root = self._upgraded(tmp_path)
        legacy = root / ".env"
        legacy.write_text("API_KEY=old\n", encoding="utf-8")

        # Git checkout is never touched.
        (root / ".git").mkdir()
        assert bootstrap.prune_legacy_env_file(root) is False
        assert legacy.is_file()
        (root / ".git").rmdir()

        # Packaged install removes legacy .env cleanly without migrating it.
        assert bootstrap.prune_legacy_env_file(root) is True
        assert not legacy.exists()
        assert not (root / "config" / "config.json").exists()

    def test_spares_retired_model_referenced_by_process_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._upgraded(
            tmp_path,
            "models/hub/models--Qwen--Qwen3-8B/blobs/custom.gguf",
            "models/hub/models--tencent--Hy-MT2-1.8B-GGUF/blobs/other.gguf",
        )
        gguf = root / "models/hub/models--Qwen--Qwen3-8B/blobs/custom.gguf"
        monkeypatch.setenv("LLAMACPP_GGUF_PATH", str(gguf))
        removed, _ = bootstrap.prune_retired_model_caches(root)
        assert removed == ["models/hub/models--tencent--Hy-MT2-1.8B-GGUF"]
        assert gguf.is_file()

    def test_skips_cleanup_when_config_json_is_malformed(self, tmp_path: Path) -> None:
        root = self._upgraded(
            tmp_path,
            "models/hub/models--Qwen--Qwen3-8B/blobs/custom.gguf",
            "models/hub/models--tencent--Hy-MT2-1.8B-GGUF/blobs/other.gguf",
        )
        gguf = root / "models/hub/models--Qwen--Qwen3-8B/blobs/custom.gguf"
        cfg = root / "config" / "config.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        # Trailing comma makes config.json invalid JSON; cleanup must skip deletion.
        cfg.write_text(
            f'{{"LLAMACPP_GGUF_PATH": "{gguf.as_posix()}",}}\n',
            encoding="utf-8",
        )
        assert bootstrap.retired_model_cache_paths(root) == []
        assert bootstrap.prune_retired_model_caches(root) == ([], 0)
        assert gguf.is_file()

    def test_skips_cleanup_when_config_json_is_truncated_empty(self, tmp_path: Path) -> None:
        root = self._upgraded(
            tmp_path,
            "models/hub/models--Qwen--Qwen3-8B/blobs/custom.gguf",
            "models/hub/models--tencent--Hy-MT2-1.8B-GGUF/blobs/other.gguf",
        )
        gguf = root / "models/hub/models--Qwen--Qwen3-8B/blobs/custom.gguf"
        cfg = root / "config" / "config.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        # 0-byte truncated config.json must NOT be treated as empty {} config.
        cfg.write_text("", encoding="utf-8")
        assert bootstrap.retired_model_cache_paths(root) == []
        assert bootstrap.prune_retired_model_caches(root) == ([], 0)
        with pytest.raises(ValueError, match="拒绝覆盖"):
            bootstrap.update_config_file({"PROXY_HOST": "127.0.0.1"}, cfg)
        assert gguf.is_file()

        # Whitespace-only truncated config.json is likewise protected.
        cfg.write_text("   \n", encoding="utf-8")
        assert bootstrap.retired_model_cache_paths(root) == []

        # Explicit empty JSON object {} IS valid and permits cleaning unreferenced retired caches.
        cfg.write_text("{}\n", encoding="utf-8")
        assert set(bootstrap.retired_model_cache_paths(root)) == {
            "models/hub/models--Qwen--Qwen3-8B",
            "models/hub/models--tencent--Hy-MT2-1.8B-GGUF",
        }

    def test_does_nothing_in_git_checkout_or_without_release_index(self, tmp_path: Path) -> None:
        root = self._upgraded(
            tmp_path,
            "models/hub/models--tencent--Hy-MT2-7B-GGUF/blobs/old.gguf",
        )
        (root / ".git").mkdir()
        assert bootstrap.prune_retired_model_caches(root) == ([], 0)
        assert (root / "models/hub/models--tencent--Hy-MT2-7B-GGUF/blobs/old.gguf").is_file()

        (root / ".git").rmdir()
        (root / bootstrap.RELEASE_INDEX_NAME).unlink()
        assert bootstrap.prune_retired_model_caches(root) == ([], 0)
        assert (root / "models/hub/models--tencent--Hy-MT2-7B-GGUF/blobs/old.gguf").is_file()


class TestLegacyPythonRuntimeMigration:
    def test_migrates_tmp_python_to_dot_python_and_rewrites_pyvenv_cfg(self, tmp_path: Path) -> None:
        root = tmp_path / "install"
        legacy_home = root / "tmp" / "python" / "cpython-3.14.6-windows-x86_64-none"
        legacy_home.mkdir(parents=True)
        (legacy_home / "python.exe").write_bytes(b"py")
        venv = root / ".venv"
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(
            f"home = {legacy_home}\nimplementation = CPython\ninclude-system-site-packages = false\n",
            encoding="utf-8",
        )

        assert bootstrap.venv_uses_legacy_tmp_python(root) is True
        result = bootstrap.migrate_legacy_python_runtime(root, verify_python=lambda _p: True)
        assert result == "migrated"
        assert bootstrap.venv_uses_legacy_tmp_python(root) is False
        new_home = (root / ".python" / "cpython-3.14.6-windows-x86_64-none").resolve()
        assert (new_home / "python.exe").is_file()
        assert f"home = {new_home}" in (venv / "pyvenv.cfg").read_text(encoding="utf-8")
        assert not (root / "tmp" / "python").exists()

    def test_missing_tmp_python_marks_venv_for_rebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "install"
        venv = root / ".venv"
        py_rel = "Scripts/python.exe" if bootstrap.os.name == "nt" else "bin/python"
        py_exe = venv / py_rel
        py_exe.parent.mkdir(parents=True)
        py_exe.write_bytes(b"")
        (venv / "pyvenv.cfg").write_text(
            f"home = {root / 'tmp' / 'python' / 'cpython-3.14.6'}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(bootstrap, "ROOT", root)
        monkeypatch.setattr(bootstrap, "VENV_PATH", venv)
        monkeypatch.setattr(bootstrap, "STAMP_PATH", venv / "stamp")
        (venv / "stamp").write_text(bootstrap.lock_digest(), encoding="utf-8")

        assert bootstrap.migrate_legacy_python_runtime(root) == "needs_rebuild"
        assert bootstrap.environment_is_current() is False
        report = bootstrap.diagnose()
        assert "tmp\\python" in report["venv"]

    def test_failed_verification_after_cfg_rewrite_forces_rebuild_across_restarts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "install"
        legacy_home = root / "tmp" / "python" / "cpython-3.14.6-windows-x86_64-none"
        legacy_home.mkdir(parents=True)
        (legacy_home / "python.exe").write_bytes(b"py")

        venv = root / ".venv"
        py_rel = "Scripts/python.exe" if bootstrap.os.name == "nt" else "bin/python"
        py_exe = venv / py_rel
        py_exe.parent.mkdir(parents=True)
        py_exe.write_bytes(b"broken-launcher")
        sentinel = venv / "old-broken-sentinel.txt"
        sentinel.write_text("stale", encoding="utf-8")
        (venv / "pyvenv.cfg").write_text(
            f"home = {legacy_home}\ninclude-system-site-packages = false\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(bootstrap, "ROOT", root)
        monkeypatch.setattr(bootstrap, "VENV_PATH", venv)
        monkeypatch.setattr(bootstrap, "STAMP_PATH", venv / "stamp")
        (venv / "stamp").write_text(bootstrap.lock_digest(), encoding="utf-8")

        # Rewrite succeeds, but interpreter verification fails -> must persist rebuild need.
        assert bootstrap.migrate_legacy_python_runtime(root, verify_python=lambda _p: False) == "needs_rebuild"
        assert bootstrap.venv_uses_legacy_tmp_python(root) is False
        assert bootstrap.venv_needs_rebuild(root) is True
        assert bootstrap.environment_is_current() is False
        assert bootstrap.diagnose()["venv"]

        # Restart still reports needs_rebuild and refuses to treat .venv as current.
        assert bootstrap.migrate_legacy_python_runtime(root, verify_python=lambda _p: True) == "needs_rebuild"
        assert bootstrap.environment_is_current() is False

        # install() removes the broken .venv before running uv sync and restores readiness.
        def fake_sync(_cmd):
            assert not sentinel.exists(), "install() must remove broken .venv before uv sync"
            py_exe.parent.mkdir(parents=True, exist_ok=True)
            py_exe.write_bytes(b"rebuilt-python")
            return 0

        monkeypatch.setattr(bootstrap, "run_streaming", fake_sync)
        monkeypatch.setattr(bootstrap, "imports_work", lambda: True)
        assert bootstrap.install(Path("uv")) == 0
        assert bootstrap.venv_needs_rebuild(root) is False
        assert bootstrap.environment_is_current() is True

    def test_interruption_before_verification_returns_still_enforces_verification_and_rebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "install"
        legacy_home = root / "tmp" / "python" / "cpython-3.14.6-windows-x86_64-none"
        legacy_home.mkdir(parents=True)
        (legacy_home / "python.exe").write_bytes(b"py")

        venv = root / ".venv"
        py_rel = "Scripts/python.exe" if bootstrap.os.name == "nt" else "bin/python"
        py_exe = venv / py_rel
        py_exe.parent.mkdir(parents=True)
        py_exe.write_bytes(b"broken-launcher")
        (venv / "pyvenv.cfg").write_text(
            f"home = {legacy_home}\ninclude-system-site-packages = false\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(bootstrap, "ROOT", root)
        monkeypatch.setattr(bootstrap, "VENV_PATH", venv)
        monkeypatch.setattr(bootstrap, "STAMP_PATH", venv / "stamp")
        (venv / "stamp").write_text(bootstrap.lock_digest(), encoding="utf-8")

        def crash_during_verify(_exe: Path) -> bool:
            raise SystemExit("simulated crash before verification returns")

        with pytest.raises(SystemExit):
            bootstrap.migrate_legacy_python_runtime(root, verify_python=crash_during_verify)

        # Even though pyvenv.cfg now points to .python and the stamp matches,
        # the pending-verification marker prevents treating .venv as current.
        assert bootstrap.venv_uses_legacy_tmp_python(root) is False
        assert bootstrap.venv_needs_rebuild(root) is True
        assert bootstrap.environment_is_current() is False
        assert bootstrap.diagnose()["venv"]

        # Restart re-runs verification: failure transitions permanently to needs_rebuild.
        assert bootstrap.migrate_legacy_python_runtime(root, verify_python=lambda _p: False) == "needs_rebuild"
        assert bootstrap.environment_is_current() is False

        # Verify happy-path recovery too: a crash before verification on a healthy
        # interpreter clears the pending marker when verification succeeds on restart.
        root_ok = tmp_path / "install-ok"
        legacy_home_ok = root_ok / "tmp" / "python" / "cpython-3.14.6-windows-x86_64-none"
        legacy_home_ok.mkdir(parents=True)
        (legacy_home_ok / "python.exe").write_bytes(b"py")
        venv_ok = root_ok / ".venv"
        py_ok = venv_ok / py_rel
        py_ok.parent.mkdir(parents=True)
        py_ok.write_bytes(b"healthy-launcher")
        (venv_ok / "pyvenv.cfg").write_text(
            f"home = {legacy_home_ok}\ninclude-system-site-packages = false\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(bootstrap, "ROOT", root_ok)
        monkeypatch.setattr(bootstrap, "VENV_PATH", venv_ok)
        monkeypatch.setattr(bootstrap, "STAMP_PATH", venv_ok / "stamp")
        (venv_ok / "stamp").write_text(bootstrap.lock_digest(), encoding="utf-8")

        with pytest.raises(SystemExit):
            bootstrap.migrate_legacy_python_runtime(root_ok, verify_python=crash_during_verify)
        assert bootstrap.environment_is_current() is False
        assert bootstrap.migrate_legacy_python_runtime(root_ok, verify_python=lambda _p: True) == "migrated"
        assert bootstrap.venv_needs_rebuild(root_ok) is False
        assert bootstrap.environment_is_current() is True


class TestOnlineSelfUpdate:
    """In-place online update apply, rollback, cleanup, and CLI fallback."""

    @staticmethod
    def _make_install(root: Path, version: str = "1.9.0") -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "jav-sub.exe").write_bytes(f"exe-{version}".encode("utf-8"))
        (root / "launcher.py").write_text(f"# launcher {version}\n", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "jav-sub"\nversion = "{version}"\n',
            encoding="utf-8",
        )
        (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "src" / "main.py").write_text(f"VERSION = '{version}'\n", encoding="utf-8")
        bootstrap.write_release_index(root)
        # User/runtime folders that must never be touched by self-update
        for rel, text in (
            (".venv/sentinel.txt", "keep-venv"),
            (".python/sentinel.txt", "keep-python"),
            ("models/hub/sentinel.bin", "keep-model"),
            ("config/config.json", '{"API_KEY": "secret-placeholder"}\n'),
            ("outputs/sample-a.srt", "1\n"),
            ("cache/web/jobs.json", "[]\n"),
        ):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return root

    @staticmethod
    def _stage_pending(root: Path, target_version: str = "1.9.1") -> Path:
        staged = root / bootstrap.UPDATES_DIR / f"v{target_version}-stage" / "staged"
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "jav-sub.exe").write_bytes(f"exe-{target_version}".encode("utf-8"))
        (staged / "launcher.py").write_text(f"# launcher {target_version}\n", encoding="utf-8")
        (staged / "pyproject.toml").write_text(
            f'[project]\nname = "jav-sub"\nversion = "{target_version}"\n',
            encoding="utf-8",
        )
        (staged / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (staged / "src").mkdir(parents=True, exist_ok=True)
        (staged / "src" / "main.py").write_text(f"VERSION = '{target_version}'\n", encoding="utf-8")
        (staged / "src" / "new_feature.py").write_text("OK = True\n", encoding="utf-8")
        bootstrap.write_release_index(staged)
        descriptor = {
            "schema": bootstrap.PENDING_UPDATE_SCHEMA,
            "from_version": "1.9.0",
            "target_version": target_version,
            "tag": f"v{target_version}",
            "staged_dir": staged.relative_to(root).as_posix(),
            "zip_sha256": "a" * 64,
            "manifest_verified": True,
        }
        pending = root / bootstrap.UPDATES_DIR / bootstrap.PENDING_UPDATE_FILE_NAME
        pending.write_text(json.dumps(descriptor), encoding="utf-8")
        return staged

    def test_apply_pending_update_overlays_payload_replaces_exe_and_preserves_user_data(
        self, tmp_path: Path
    ) -> None:
        root = self._make_install(tmp_path / "install", "1.9.0")
        self._stage_pending(root, "1.9.1")

        apply_status, version = bootstrap.apply_pending_update(root, wait_instances_s=0.0)
        assert (apply_status, version) == ("applied", "1.9.1")
        assert (root / "jav-sub.exe").read_bytes() == b"exe-1.9.1"
        assert (root / "jav-sub.exe.old").read_bytes() == b"exe-1.9.0"
        assert (root / "src" / "new_feature.py").is_file()
        assert bootstrap.current_project_version(root) == "1.9.1"
        assert not (root / bootstrap.UPDATES_DIR).exists()

        # User/runtime data untouched
        assert (root / ".venv/sentinel.txt").read_text(encoding="utf-8") == "keep-venv"
        assert (root / ".python/sentinel.txt").read_text(encoding="utf-8") == "keep-python"
        assert (root / "models/hub/sentinel.bin").read_text(encoding="utf-8") == "keep-model"
        assert "secret-placeholder" in (root / "config/config.json").read_text(encoding="utf-8")
        assert (root / "outputs/sample-a.srt").is_file()
        assert (root / "cache/web/jobs.json").is_file()

        # Next launch cleans jav-sub.exe.old
        removed = bootstrap.prune_old_launcher_backups(root)
        assert "jav-sub.exe.old" in removed
        assert not (root / "jav-sub.exe.old").exists()

    def test_replace_running_exe_rolls_back_on_copy_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "jav-sub.exe"
        staged = tmp_path / "staged.exe"
        target.write_bytes(b"original-exe")
        staged.write_bytes(b"new-exe")

        def fail_copy(_src: Path, _dst: Path, **_kw) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(bootstrap, "_copy_file_atomic", fail_copy)
        with pytest.raises(OSError, match="disk full"):
            bootstrap._replace_running_exe(staged, target)
        assert target.read_bytes() == b"original-exe"
        assert not (tmp_path / "jav-sub.exe.old").exists()

    def test_apply_pending_update_defers_when_other_instance_active_and_skips_git(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._make_install(tmp_path / "install", "1.9.0")
        self._stage_pending(root, "1.9.1")

        # Another launcher window active -> defer without deleting pending-update.json
        monkeypatch.setattr(bootstrap, "_other_launcher_instances_active", lambda _r, **_k: True)
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.0) == ("deferred", "1.9.1")
        assert (root / bootstrap.UPDATES_DIR / bootstrap.PENDING_UPDATE_FILE_NAME).is_file()
        assert bootstrap.current_project_version(root) == "1.9.0"

        # Startup with deferred update must also refuse to launch a second window.
        monkeypatch.setattr(bootstrap, "ROOT", root)
        monkeypatch.setattr(
            bootstrap,
            "prune_stale_release_files",
            lambda *_a, **_k: pytest.fail("must not continue startup on deferred update"),
        )
        assert bootstrap.main([]) == 1

        # Git checkout -> never touches files
        monkeypatch.setattr(bootstrap, "_other_launcher_instances_active", lambda _r, **_k: False)
        (root / ".git").mkdir()
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.0) == ("none", "")
        assert bootstrap.current_project_version(root) == "1.9.0"

    def test_startup_aborts_on_partial_overlay_failure_and_resumes_after_unlock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._make_install(tmp_path / "install", "1.9.0")
        self._stage_pending(root, "1.9.1")
        monkeypatch.setattr(bootstrap, "ROOT", root)
        monkeypatch.setattr(bootstrap, "CONFIG_PATH", root / "config" / "config.json")
        monkeypatch.setattr(bootstrap, "_other_launcher_instances_active", lambda _r, **_k: False)

        real_copy = bootstrap._copy_file_atomic

        def locked_main_copy(src: Path, dst: Path, **kwargs) -> None:
            if dst.name == "main.py" and dst.parent.name == "src":
                raise PermissionError("src/main.py locked by another process")
            real_copy(src, dst, **kwargs)

        monkeypatch.setattr(bootstrap, "_copy_file_atomic", locked_main_copy)
        monkeypatch.setattr(
            bootstrap,
            "prune_stale_release_files",
            lambda *_a, **_k: pytest.fail("must not prune or launch a half-updated directory"),
        )
        monkeypatch.setattr(
            bootstrap,
            "launch_with_recovery",
            lambda *_a, **_k: pytest.fail("must not launch a half-updated directory"),
        )

        # Attempt 1: launcher.py / pyproject.toml updated, src/main.py locked -> must abort (exit 1)
        # and keep tmp/updates/pending-update.json intact for retry.
        assert bootstrap.main([]) == 1
        assert (root / "src" / "main.py").read_text(encoding="utf-8") == "VERSION = '1.9.0'\n"
        assert bootstrap.current_project_version(root) == "1.9.1"
        assert (root / bootstrap.UPDATES_DIR / bootstrap.PENDING_UPDATE_FILE_NAME).is_file()

        # Attempt 2: target unlocked, but staged release-files.json temporarily unreadable ->
        # must return failed, preserve pending-update.json & staging, and still abort startup!
        monkeypatch.setattr(bootstrap, "_copy_file_atomic", real_copy)
        staged_dir = root / bootstrap.UPDATES_DIR / "v1.9.1-stage" / "staged"
        staged_index = (staged_dir / bootstrap.RELEASE_INDEX_NAME).resolve()
        real_read_text = Path.read_text

        def locked_staged_index_read(self: Path, *args, **kwargs) -> str:
            if self.resolve() == staged_index:
                raise PermissionError("staged release-files.json locked")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", locked_staged_index_read)
        assert bootstrap.main([]) == 1
        assert (root / bootstrap.UPDATES_DIR / bootstrap.PENDING_UPDATE_FILE_NAME).is_file()
        assert staged_index.is_file()
        assert (root / "src" / "main.py").read_text(encoding="utf-8") == "VERSION = '1.9.0'\n"

        # Transient I/O lock on staged pyproject.toml or pending-update.json must likewise
        # return failed and keep staging intact.
        monkeypatch.setattr(Path, "read_text", real_read_text)
        staged_pyproject = (staged_dir / "pyproject.toml").resolve()
        real_read_bytes = Path.read_bytes

        def locked_staged_pyproject(self: Path, *args, **kwargs) -> bytes:
            if self.resolve() == staged_pyproject:
                raise PermissionError("staged pyproject.toml locked")
            return real_read_bytes(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_bytes", locked_staged_pyproject)
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.0) == ("failed", "1.9.1")
        assert (root / bootstrap.UPDATES_DIR / bootstrap.PENDING_UPDATE_FILE_NAME).is_file()
        monkeypatch.setattr(Path, "read_bytes", real_read_bytes)

        pending_resolved = (root / bootstrap.UPDATES_DIR / bootstrap.PENDING_UPDATE_FILE_NAME).resolve()

        def locked_pending_read(self: Path, *args, **kwargs) -> str:
            if self.resolve() == pending_resolved:
                raise PermissionError("pending-update.json locked")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", locked_pending_read)
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.0) == ("failed", "")
        assert pending_resolved.is_file()
        monkeypatch.setattr(Path, "read_text", real_read_text)

        # Attempt 3: all locks cleared -> resume completes cleanly and cleans tmp/updates/.
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.0) == ("applied", "1.9.1")
        assert (root / "src" / "main.py").read_text(encoding="utf-8") == "VERSION = '1.9.1'\n"
        assert (root / "jav-sub.exe").read_bytes() == b"exe-1.9.1"
        assert not (root / bootstrap.UPDATES_DIR).exists()

    def test_apply_pending_update_rejects_traversal_unverified_manifest_and_protected_dirs(
        self, tmp_path: Path
    ) -> None:
        root = self._make_install(tmp_path / "install", "1.9.0")
        updates = root / bootstrap.UPDATES_DIR
        updates.mkdir(parents=True, exist_ok=True)
        # Escape staged_dir outside tmp/updates
        (updates / bootstrap.PENDING_UPDATE_FILE_NAME).write_text(
            json.dumps(
                {
                    "schema": bootstrap.PENDING_UPDATE_SCHEMA,
                    "target_version": "1.9.1",
                    "staged_dir": "../escape",
                    "manifest_verified": True,
                }
            ),
            encoding="utf-8",
        )
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.0) == ("invalid", "")
        assert not updates.exists()

        # Descriptor with manifest_verified=False must be rejected and cleaned.
        self._stage_pending(root, "1.9.1")
        pending = root / bootstrap.UPDATES_DIR / bootstrap.PENDING_UPDATE_FILE_NAME
        desc = json.loads(pending.read_text(encoding="utf-8"))
        desc["manifest_verified"] = False
        pending.write_text(json.dumps(desc), encoding="utf-8")
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.0) == ("invalid", "")
        assert not (root / bootstrap.UPDATES_DIR).exists()

        # Protected directory inside staged release-files.json
        staged = self._stage_pending(root, "1.9.1")
        (staged / "config").mkdir()
        (staged / "config" / "config.json").write_text("{}", encoding="utf-8")
        bootstrap.write_release_index(staged)
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.0) == ("invalid", "")
        assert not (root / bootstrap.UPDATES_DIR).exists()

    def test_prune_old_launcher_backups_spares_pending_or_active_downloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._make_install(tmp_path / "install", "1.9.0")
        (root / "jav-sub.exe.old").write_bytes(b"old")
        updates = root / bootstrap.UPDATES_DIR
        (updates / "v1.9.1-part").mkdir(parents=True)

        # Active launcher window -> must NOT wipe tmp/updates (could be downloading)
        monkeypatch.setattr(bootstrap, "_other_launcher_instances_active", lambda _r, **_k: True)
        removed = bootstrap.prune_old_launcher_backups(root)
        assert "jav-sub.exe.old" in removed
        assert updates.is_dir()

        # Pending update descriptor present -> must preserve tmp/updates
        monkeypatch.setattr(bootstrap, "_other_launcher_instances_active", lambda _r, **_k: False)
        pending = updates / bootstrap.PENDING_UPDATE_FILE_NAME
        pending.write_text("{}", encoding="utf-8")
        bootstrap.prune_old_launcher_backups(root)
        assert updates.is_dir()

        # No pending descriptor and no active window -> cleans stale tmp/updates
        pending.unlink()
        removed2 = bootstrap.prune_old_launcher_backups(root)
        assert bootstrap.UPDATES_DIR.as_posix() in removed2
        assert not updates.exists()

    def test_launch_with_recovery_passes_through_update_exit_code_and_relaunches_clean_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "launch", lambda *a, **k: bootstrap.UPDATE_EXIT_CODE)
        monkeypatch.setattr(bootstrap, "doctor", lambda **k: pytest.fail("doctor must not run on exit 42"))
        code = bootstrap.launch_with_recovery([], assume_yes=True, repaired_already=False)
        assert code == bootstrap.UPDATE_EXIT_CODE

        # Verify _relaunch_updated_exe strips PyInstaller bootloader env vars
        root = self._make_install(tmp_path / "install", "1.9.1")
        monkeypatch.setattr(bootstrap, "ROOT", root)
        monkeypatch.setenv("_MEIPASS2", "C:\\old-mei")
        monkeypatch.setenv("_PYI_ARCHIVE_FILE", "C:\\old\\jav-sub.exe.old")
        captured: list[tuple[list[str], dict[str, str]]] = []

        def fake_popen(cmd, *, cwd, env):
            captured.append((cmd, env))
            return None

        monkeypatch.setattr(bootstrap.subprocess, "Popen", fake_popen)
        assert bootstrap._relaunch_updated_exe(["--port", "2233"], keep_console=True) is True
        assert len(captured) == 1
        cmd, env = captured[0]
        assert cmd[0] == str(root / "jav-sub.exe")
        assert "--keep-console" in cmd and "--port" in cmd
        assert "_MEIPASS2" not in env
        assert "_PYI_ARCHIVE_FILE" not in env

    def test_stage_online_update_cli_stages_and_rejects_bad_hash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io

        root = self._make_install(tmp_path / "install", "1.9.0")
        staged_src = tmp_path / "src-release" / "jav-sub"
        staged_src.mkdir(parents=True)
        (staged_src / "jav-sub.exe").write_bytes(b"exe-1.9.2")
        (staged_src / "launcher.py").write_text("# launcher 1.9.2\n", encoding="utf-8")
        (staged_src / "pyproject.toml").write_text(
            '[project]\nname = "jav-sub"\nversion = "1.9.2"\n',
            encoding="utf-8",
        )
        (staged_src / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (staged_src / "src").mkdir()
        (staged_src / "src" / "main.py").write_text("VERSION = '1.9.2'\n", encoding="utf-8")
        bootstrap.write_release_index(staged_src)

        manifest_files = []
        zip_buf = io.BytesIO()
        with bootstrap.zipfile.ZipFile(zip_buf, "w") as zf:
            for p in sorted(staged_src.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(staged_src).as_posix()
                raw = p.read_bytes()
                zf.writestr(f"jav-sub/{rel}", raw)
                manifest_files.append(
                    {
                        "path": rel,
                        "root": "dist",
                        "bytes": len(raw),
                        "sha256": bootstrap.hashlib.sha256(raw).hexdigest(),
                    }
                )
        zip_bytes = zip_buf.getvalue()
        zip_sha = bootstrap.hashlib.sha256(zip_bytes).hexdigest()
        manifest_bytes = json.dumps(
            {"schema": bootstrap.RELEASE_MANIFEST_SCHEMA, "files": manifest_files}
        ).encode("utf-8")

        api_url = "https://example.invalid/latest"
        dl_url = "https://example.invalid/jav-sub-windows-x64.zip"
        sha_url = "https://example.invalid/jav-sub-windows-x64.zip.sha256"
        man_url = "https://example.invalid/release-manifest.json"
        release_json = json.dumps(
            {
                "tag_name": "v1.9.2",
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": bootstrap.UPDATE_ARCHIVE_NAME,
                        "browser_download_url": dl_url,
                        "digest": f"sha256:{zip_sha}",
                    },
                    {"name": bootstrap.UPDATE_SHA256_NAME, "browser_download_url": sha_url},
                    {"name": bootstrap.UPDATE_MANIFEST_NAME, "browser_download_url": man_url},
                ],
            }
        ).encode("utf-8")

        class _FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        responses = {
            api_url: release_json,
            sha_url: f"{zip_sha}  {bootstrap.UPDATE_ARCHIVE_NAME}\n".encode("utf-8"),
            man_url: manifest_bytes,
            dl_url: zip_bytes,
        }

        class _FakeOpener:
            def open(self, req, timeout=20.0):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                return _FakeResp(responses[url])

        monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", api_url)
        monkeypatch.setattr(bootstrap, "_opener", lambda: _FakeOpener())

        # Happy path: stages and applies cleanly
        status, ver = bootstrap.stage_online_update_cli(root, assume_yes=True)
        assert (status, ver) == ("staged", "1.9.2")
        apply_status, applied_ver = bootstrap.apply_pending_update(root, wait_instances_s=0.0)
        assert (apply_status, applied_ver) == ("applied", "1.9.2")
        assert bootstrap.current_project_version(root) == "1.9.2"

        # Already up-to-date now
        status2, ver2 = bootstrap.stage_online_update_cli(root, assume_yes=True)
        assert (status2, ver2) == ("up_to_date", "1.9.2")

        # Release missing release-manifest.json must be rejected
        root_no_man = self._make_install(tmp_path / "install-no-man", "1.9.0")
        responses[api_url] = json.dumps(
            {
                "tag_name": "v1.9.2",
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": bootstrap.UPDATE_ARCHIVE_NAME,
                        "browser_download_url": dl_url,
                        "digest": f"sha256:{zip_sha}",
                    },
                    {"name": bootstrap.UPDATE_SHA256_NAME, "browser_download_url": sha_url},
                ],
            }
        ).encode("utf-8")
        assert bootstrap.stage_online_update_cli(root_no_man, assume_yes=True) == ("error", "")
        assert not (root_no_man / bootstrap.UPDATES_DIR / bootstrap.PENDING_UPDATE_FILE_NAME).exists()

        # Restore release JSON with manifest, corrupt zip bytes -> error status and clean temp dir
        responses[api_url] = release_json
        root_bad = self._make_install(tmp_path / "install-bad", "1.9.0")
        responses[dl_url] = b"corrupted-zip"
        status_bad, _ = bootstrap.stage_online_update_cli(root_bad, assume_yes=True)
        assert status_bad == "error"
        assert not (root_bad / bootstrap.UPDATES_DIR / bootstrap.PENDING_UPDATE_FILE_NAME).exists()

        # CLI main --update exits non-zero on error
        monkeypatch.setattr(bootstrap, "ROOT", root_bad)
        monkeypatch.setattr(bootstrap, "CONFIG_PATH", root_bad / "config" / "config.json")
        assert bootstrap.main(["--update", "--yes", "--install-only"]) == 1

    def test_finish_after_launch_and_main_update_return_nonzero_when_relaunch_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "set_console_visible", lambda _v: None)
        monkeypatch.setattr(bootstrap, "apply_pending_update", lambda **_k: ("applied", "1.9.2"))
        monkeypatch.setattr(bootstrap, "_relaunch_updated_exe", lambda *_a, **_k: False)
        assert bootstrap._finish_after_launch(bootstrap.UPDATE_EXIT_CODE, []) == 1

        root = self._make_install(tmp_path / "install", "1.9.0")
        monkeypatch.setattr(bootstrap, "ROOT", root)
        monkeypatch.setattr(bootstrap, "CONFIG_PATH", root / "config" / "config.json")
        monkeypatch.setattr(bootstrap, "stage_online_update_cli", lambda **_k: ("staged", "1.9.2"))
        assert bootstrap.main(["--update", "--yes"]) == 1

    def test_launch_with_recovery_retry_accepts_update_exit_code_quietly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        codes = iter([1, bootstrap.UPDATE_EXIT_CODE])
        printed: list[str] = []
        monkeypatch.setattr(bootstrap, "launch", lambda *_a, **_k: next(codes))
        monkeypatch.setattr(bootstrap, "doctor", lambda **_k: (True, True))
        monkeypatch.setattr(bootstrap, "log", lambda msg="": printed.append(str(msg)))
        assert bootstrap.launch_with_recovery([], assume_yes=True, repaired_already=False) == bootstrap.UPDATE_EXIT_CODE
        assert not any("重试后仍退出" in line for line in printed)

    def test_apply_pending_update_defers_when_workspace_gate_is_held(
        self, tmp_path: Path
    ) -> None:
        root = self._make_install(tmp_path / "install", "1.9.0")
        self._stage_pending(root, "1.9.1")
        with bootstrap._workspace_gate(root, wait_s=1.0) as acquired:
            assert acquired is True
            assert bootstrap.apply_pending_update(root, wait_instances_s=0.05) == ("deferred", "1.9.1")
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.05) == ("applied", "1.9.1")

    def test_stage_online_update_cli_rejects_active_instance_and_unsafe_member_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._make_install(tmp_path / "install", "1.9.0")
        monkeypatch.setattr(bootstrap, "_other_launcher_instances_active", lambda _r, **_k: True)
        assert bootstrap.stage_online_update_cli(root, assume_yes=True) == ("error", "")

        for unsafe in ("", ".", "src/./a.py", "src/a.txt:ads", "src/CON.py", "src/NUL", "src/foo "):
            with pytest.raises(ValueError):
                bootstrap._validate_update_relpath(unsafe)

        assert bootstrap._is_allowed_update_redirect(
            "https://github.com/jaykwok/jav-sub/releases/download/v1.9.2/a.zip",
            "https://objects.githubusercontent.com/bucket/a.zip",
        ) is True
        assert bootstrap._is_allowed_update_redirect(
            "https://github.com/jaykwok/jav-sub/releases/download/v1.9.2/a.zip",
            "http://objects.githubusercontent.com/bucket/a.zip",
        ) is False
        assert bootstrap._is_allowed_update_redirect(
            "https://github.com/jaykwok/jav-sub/releases/download/v1.9.2/a.zip",
            "https://evil.example.net/a.zip",
        ) is False

    def test_apply_pending_update_rejects_missing_required_payload_entry(
        self, tmp_path: Path
    ) -> None:
        root = self._make_install(tmp_path / "install", "1.9.0")
        staged = self._stage_pending(root, "1.9.1")
        (staged / "uv.lock").unlink()
        bootstrap.write_release_index(staged)
        assert bootstrap.apply_pending_update(root, wait_instances_s=0.0) == ("invalid", "")
        assert not (root / bootstrap.UPDATES_DIR).exists()
