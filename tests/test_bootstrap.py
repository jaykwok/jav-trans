"""The installer runs before anything else exists, so its failures are silent.

There is no app to report an error through and no log to inspect: whatever
`bootstrap.py` gets wrong, the user sees only a console that closed. So the
properties pinned here are the ones with no second chance - that it imports
without any dependency installed, that it never destroys settings it did not
write, and that it cannot be tricked into reporting a broken environment as
ready.
"""

from __future__ import annotations

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


class TestEnvFile:
    def test_values_are_read(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("PROXY_HOST=127.0.0.1\nPROXY_PORT=7890\n", encoding="utf-8")
        assert bootstrap.read_env_file(path) == {
            "PROXY_HOST": "127.0.0.1",
            "PROXY_PORT": "7890",
        }

    def test_comments_and_junk_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        """The shipped `.env` is comment-led, and users hand-edit it."""
        path = tmp_path / ".env"
        path.write_text("# comment\n\nnot-an-assignment\nA=1\n", encoding="utf-8")
        assert bootstrap.read_env_file(path) == {"A": "1"}

    def test_a_missing_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert bootstrap.read_env_file(tmp_path / "absent") == {}

    def test_quotes_are_stripped_like_the_app_strips_them(self, tmp_path: Path) -> None:
        """The settings page writes every value quoted, and the app reads the
        same file through python-dotenv. Reading it literally here would make
        the two halves of the program disagree about the value."""
        path = tmp_path / ".env"
        path.write_text(
            'PROXY_PROTOCOL="socks5"\nPROXY_HOST="127.0.0.1"\nPROXY_PORT=\'1080\'\n',
            encoding="utf-8",
        )
        assert bootstrap.read_env_file(path) == {
            "PROXY_PROTOCOL": "socks5",
            "PROXY_HOST": "127.0.0.1",
            "PROXY_PORT": "1080",
        }

    def test_a_proxy_switched_off_in_the_ui_yields_no_proxy(self, tmp_path: Path) -> None:
        """`PROXY_HOST=""` is how the settings page records "no proxy". Kept as
        the literal two-character string it is truthy, and every launch exported
        `HTTP_PROXY=http://"":""` to uv and to the model downloader."""
        path = tmp_path / ".env"
        path.write_text(
            'PROXY_PROTOCOL="http"\nPROXY_HOST=""\nPROXY_PORT=""\n', encoding="utf-8"
        )
        assert bootstrap.proxy_url_from(bootstrap.read_env_file(path)) == ""

    def test_writing_preserves_unrelated_lines(self, tmp_path: Path) -> None:
        """The settings page owns this file too. Rewriting it wholesale would
        drop the user's API key on the next launch."""
        path = tmp_path / ".env"
        path.write_text("# header\nAPI_KEY=secret\n", encoding="utf-8")
        bootstrap.update_env_file({"PROXY_HOST": "127.0.0.1"}, path)
        text = path.read_text(encoding="utf-8")
        assert "# header\n" in text
        assert "API_KEY=secret\n" in text
        assert "PROXY_HOST=127.0.0.1\n" in text

    def test_an_existing_key_is_replaced_in_place(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("PROXY_HOST=old\nAPI_KEY=secret\n", encoding="utf-8")
        bootstrap.update_env_file({"PROXY_HOST": "new"}, path)
        assert bootstrap.read_env_file(path) == {"PROXY_HOST": "new", "API_KEY": "secret"}
        assert path.read_text(encoding="utf-8").count("PROXY_HOST") == 1

    def test_a_commented_example_is_not_treated_as_the_key(self, tmp_path: Path) -> None:
        """The shipped file documents keys as commented examples; matching them
        would edit the documentation and leave the real setting unwritten."""
        path = tmp_path / ".env"
        path.write_text("# PROXY_HOST=example\n", encoding="utf-8")
        bootstrap.update_env_file({"PROXY_HOST": "127.0.0.1"}, path)
        text = path.read_text(encoding="utf-8")
        assert "# PROXY_HOST=example\n" in text
        assert "PROXY_HOST=127.0.0.1\n" in text

    def test_writing_to_a_missing_file_creates_it(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        bootstrap.update_env_file({"PROXY_HOST": "127.0.0.1"}, path)
        assert bootstrap.read_env_file(path)["PROXY_HOST"] == "127.0.0.1"

    def test_new_file_does_not_start_with_a_blank_line(self, tmp_path):
        path = tmp_path / ".env"
        bootstrap.update_env_file({"PROXY_HOST": "127.0.0.1"}, path)
        assert path.read_text(encoding="utf-8").startswith("# ---")


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
    def test_every_variant_is_exported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in bootstrap.PROXY_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(bootstrap.os, "environ", dict(bootstrap.os.environ))
        bootstrap.apply_proxy("http://127.0.0.1:7890")
        for key in bootstrap.PROXY_ENV_KEYS:
            assert bootstrap.os.environ[key] == "http://127.0.0.1:7890"

    def test_an_empty_url_clears_stale_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A proxy the user has removed must not survive in the environment of
        the process that downloads 1.8 GB."""
        monkeypatch.setattr(bootstrap.os, "environ", dict(bootstrap.os.environ))
        bootstrap.apply_proxy("http://127.0.0.1:7890")
        bootstrap.apply_proxy("")
        for key in bootstrap.PROXY_ENV_KEYS:
            assert key not in bootstrap.os.environ


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
