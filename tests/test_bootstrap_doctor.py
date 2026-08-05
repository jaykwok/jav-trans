"""Self-repair and the console window.

Two failures with no second chance. The first: an install that once finished and
then lost files - a folder deleted to reclaim disk space - which the stamp still
reports as ready, so the app starts into ImportError inside a daemon thread and
the window opens onto nothing. The second: hiding the console. It is the only
place a startup failure is visible, so hiding it a moment too early turns a
diagnosable error into "双击了没反应".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import bootstrap


def _fake_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, roots=(), stamped=True) -> Path:
    """A program directory that looks installed, with only `roots` in site-packages."""
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "LOCK_PATH", tmp_path / "uv.lock")
    venv = tmp_path / ".venv"
    monkeypatch.setattr(bootstrap, "VENV_PATH", venv)
    stamp = venv / "stamp"
    monkeypatch.setattr(bootstrap, "STAMP_PATH", stamp)

    monkeypatch.setattr(bootstrap, "ENV_PATH", tmp_path / ".env")

    python = venv / ("Scripts/python.exe" if bootstrap.os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    site = bootstrap.site_packages_dir()
    site.mkdir(parents=True, exist_ok=True)
    for name in roots:
        (site / name).mkdir()

    # Before the stamp: it is a digest of the lockfile and pyproject.toml, so
    # writing it first would leave it stale the moment the payload appears.
    for name in bootstrap.PAYLOAD_PATHS:
        target = tmp_path / name
        if target.suffix:
            target.write_text("", encoding="utf-8")
        else:
            target.mkdir(exist_ok=True)
    if stamped:
        stamp.write_text(bootstrap.lock_digest(), encoding="utf-8")

    suffix = ".exe" if bootstrap.os.name == "nt" else ""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("ffmpeg", "ffprobe"):
        (bin_dir / f"{name}{suffix}").write_text("", encoding="utf-8")
    (bin_dir / "avcodec-62.dll").write_text("", encoding="utf-8")
    return venv


ALL_ROOTS = tuple(bootstrap.CRITICAL_IMPORT_ROOTS)


class TestEnvironmentIntegrity:
    def test_a_deleted_package_is_caught_even_with_a_valid_stamp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reported failure mode: a finished install, then a folder deleted.
        The stamp still says "installed", so readiness cannot be the only
        question asked before launching."""
        kept = [name for name in ALL_ROOTS if name != "torch"]
        _fake_install(tmp_path, monkeypatch, roots=kept)
        assert bootstrap.environment_is_current() is True
        assert bootstrap.missing_import_roots() == ["torch"]
        report = bootstrap.diagnose()
        assert "torch" in report["venv"]
        assert report["broken"] == ["torch"]
        assert bootstrap.report_is_healthy(report) is False

    def test_a_complete_install_is_reported_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_install(tmp_path, monkeypatch, roots=ALL_ROOTS)
        assert bootstrap.missing_import_roots() == []
        assert bootstrap.report_is_healthy(bootstrap.diagnose()) is True

    def test_the_structural_check_costs_no_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It runs on every launch, which is only affordable while it stays a
        handful of stat() calls - the import probe is the --doctor path."""
        _fake_install(tmp_path, monkeypatch, roots=ALL_ROOTS)
        monkeypatch.setattr(
            bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("no subprocess here")
        )
        bootstrap.diagnose()

    def test_missing_roots_name_the_distribution_that_provides_them(self) -> None:
        """The repair reinstalls distributions, not import names, and `webview`
        comes from `pywebview` - getting that wrong makes the repair a no-op."""
        assert bootstrap.distributions_for(["webview"]) == ["pywebview"]
        assert bootstrap.distributions_for(["torch", "webview"]) == ["pywebview", "torch"]
        assert bootstrap.distributions_for(["not-a-dependency"]) == []

    def test_every_critical_root_maps_to_a_real_dependency(self) -> None:
        declared = (bootstrap.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        for distribution in bootstrap.CRITICAL_IMPORT_ROOTS.values():
            assert f'"{distribution}' in declared, distribution

    def test_a_missing_venv_marks_everything_broken(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        monkeypatch.setattr(bootstrap, "VENV_PATH", tmp_path / ".venv")
        report = bootstrap.diagnose()
        assert ".venv" in report["venv"]
        assert "torch" in report["broken"]

    def test_deleted_program_files_are_reported_as_unrepairable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """uv can rebuild .venv from these files; it cannot rebuild these files."""
        _fake_install(tmp_path, monkeypatch, roots=ALL_ROOTS)
        (tmp_path / "launcher.py").unlink()
        assert bootstrap.diagnose()["payload"] == ["launcher.py"]

    def test_ffmpeg_exes_without_their_dlls_are_still_broken(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """torchcodec loads the shared libraries, not the executables."""
        if bootstrap.os.name != "nt":
            pytest.skip("Windows-only packaging detail")
        _fake_install(tmp_path, monkeypatch, roots=ALL_ROOTS)
        (tmp_path / "bin" / "avcodec-62.dll").unlink()
        assert "av*.dll" in bootstrap.ffmpeg_problem()

    def test_an_ffmpeg_on_path_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        monkeypatch.setattr(bootstrap.shutil, "which", lambda _: "C:/tools/ffmpeg.exe")
        assert bootstrap.ffmpeg_problem() == ""

    def test_the_import_probe_reports_which_package_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parsed per package, because that is what a targeted reinstall needs."""
        monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
        monkeypatch.setattr(bootstrap, "venv_python", lambda: Path(sys.executable))
        failures = bootstrap.probe_imports(("json", "a_package_that_does_not_exist"))
        assert [name for name, _ in failures] == ["a_package_that_does_not_exist"]
        assert "ModuleNotFoundError" in failures[0][1]

    def test_a_dead_interpreter_is_reported_rather_than_read_as_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "venv_python", lambda: tmp_path / "not-python.exe")
        failures = bootstrap.probe_imports(("torch",))
        assert [name for name, _ in failures] == ["torch"]


class TestRepair:
    def test_the_broken_packages_are_named_before_the_big_hammer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plain `uv sync` audits metadata: the dist-info of a package whose
        files were deleted satisfies it and nothing is reinstalled. So the repair
        names the packages, and only escalates when imports still fail."""
        commands: list[list[str]] = []
        monkeypatch.setattr(bootstrap, "LOCK_PATH", tmp_path / "uv.lock")
        monkeypatch.setattr(bootstrap, "STAMP_PATH", tmp_path / "stamp")
        monkeypatch.setattr(
            bootstrap,
            "run_streaming",
            lambda command: commands.append([str(part) for part in command]) or 0,
        )
        monkeypatch.setattr(bootstrap, "probe_imports", lambda names=(): [])

        assert bootstrap.repair(Path("uv"), ["torch", "pywebview"]) is True
        assert len(commands) == 1
        assert commands[0][-4:] == [
            "--reinstall-package", "torch", "--reinstall-package", "pywebview",
        ]
        assert (tmp_path / "stamp").read_text(encoding="utf-8") == bootstrap.lock_digest()

    def test_a_targeted_reinstall_that_does_not_help_escalates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commands: list[list[str]] = []
        monkeypatch.setattr(bootstrap, "LOCK_PATH", tmp_path / "uv.lock")
        monkeypatch.setattr(bootstrap, "STAMP_PATH", tmp_path / "stamp")
        monkeypatch.setattr(
            bootstrap,
            "run_streaming",
            lambda command: commands.append([str(part) for part in command]) or 0,
        )
        probes = iter([[("torch", "ImportError: DLL load failed")], []])
        monkeypatch.setattr(bootstrap, "probe_imports", lambda names=(): next(probes))

        assert bootstrap.repair(Path("uv"), ["torch"]) is True
        assert "--reinstall" in commands[1]
        assert "--reinstall-package" not in commands[1]

    def test_a_repair_that_never_works_reports_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "LOCK_PATH", tmp_path / "uv.lock")
        monkeypatch.setattr(bootstrap, "STAMP_PATH", tmp_path / "stamp")
        monkeypatch.setattr(bootstrap, "run_streaming", lambda command: 0)
        monkeypatch.setattr(bootstrap, "probe_imports", lambda names=(): [("torch", "boom")])

        assert bootstrap.repair(Path("uv"), ["torch"]) is False
        # No stamp written: claiming success here would make the next launch skip
        # the repair and fail the same way.
        assert not (tmp_path / "stamp").exists()


class TestDoctor:
    def test_a_healthy_install_is_reported_without_touching_uv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_install(tmp_path, monkeypatch, roots=ALL_ROOTS)
        monkeypatch.setattr(bootstrap, "probe_imports", lambda names=(): [])
        monkeypatch.setattr(bootstrap, "ensure_uv", lambda: pytest.fail("nothing to repair"))
        assert bootstrap.doctor(assume_yes=True, deep=True) == (True, False)

    def test_missing_program_files_are_not_papered_over_with_a_sync(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_install(tmp_path, monkeypatch, roots=ALL_ROOTS)
        (tmp_path / "launcher.py").unlink()
        monkeypatch.setattr(bootstrap, "ensure_uv", lambda: pytest.fail("uv cannot restore these"))
        assert bootstrap.doctor(assume_yes=True) == (False, False)

    def test_a_broken_dependency_is_repaired_after_the_proxy_question(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Order matters: the proxy has to be settled before uv starts
        downloading, and one answer then covers the model downloads too."""
        order: list[str] = []
        _fake_install(tmp_path, monkeypatch, roots=[n for n in ALL_ROOTS if n != "torch"])
        monkeypatch.setattr(
            bootstrap, "ensure_proxy_configured", lambda **kwargs: order.append("proxy")
        )
        monkeypatch.setattr(bootstrap, "ensure_uv", lambda: order.append("uv") or Path("uv"))
        monkeypatch.setattr(
            bootstrap, "repair", lambda uv, broken: order.append(f"repair:{broken}") or True
        )
        assert bootstrap.doctor(assume_yes=True) == (True, True)
        assert order == ["proxy", "uv", "repair:['torch']"]

    def test_without_uv_the_repair_is_declined_rather_than_faked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_install(tmp_path, monkeypatch, roots=[n for n in ALL_ROOTS if n != "torch"])
        monkeypatch.setattr(bootstrap, "ensure_proxy_configured", lambda **kwargs: None)
        monkeypatch.setattr(bootstrap, "ensure_uv", lambda: None)
        assert bootstrap.doctor(assume_yes=True) == (False, False)

    def test_a_report_only_run_reports_and_stops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--doctor` in a git checkout: repairing means `uv sync --no-dev`,
        which would strip pytest out of the developer's own .venv."""
        _fake_install(tmp_path, monkeypatch, roots=[n for n in ALL_ROOTS if n != "torch"])
        monkeypatch.setattr(
            bootstrap, "ensure_proxy_configured", lambda **kwargs: pytest.fail("not ours to fix")
        )
        assert bootstrap.doctor(assume_yes=True, allow_repair=False) == (False, False)
        # The checklist is still the point of the run, so it has to be printed.
        assert "torch" in capsys.readouterr().out

    def test_the_doctor_flag_declines_to_repair_a_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard that protects a checkout sits after the --doctor branch, so
        --doctor has to carry its own."""
        _fake_install(tmp_path, monkeypatch, roots=[n for n in ALL_ROOTS if n != "torch"])
        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(bootstrap, "ensure_uv", lambda: pytest.fail("would strip dev deps"))
        monkeypatch.setattr(bootstrap, "launch", lambda *a, **k: pytest.fail("not launched"))
        assert bootstrap.main(["--doctor"]) == 1


class TestConsoleWindow:
    def test_a_source_checkout_never_hides_the_developers_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unfrozen, GetConsoleWindow returns the terminal the developer is
        working in - not a window this program is entitled to close."""
        monkeypatch.setattr(bootstrap, "_FROZEN", False)
        monkeypatch.setattr(
            bootstrap, "_apply_console_visibility", lambda visible: pytest.fail("not ours")
        )
        bootstrap.set_console_visible(False)

    def test_hiding_is_idempotent_and_reversible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shown: list[bool] = []
        monkeypatch.setattr(bootstrap, "_FROZEN", True)
        monkeypatch.setattr(bootstrap, "_console_hidden", False)
        monkeypatch.setattr(
            bootstrap, "_apply_console_visibility", lambda visible: shown.append(visible) or True
        )
        bootstrap.set_console_visible(False)
        bootstrap.set_console_visible(False)
        bootstrap.set_console_visible(True)
        assert shown == [False, True]

    def test_a_failed_hide_does_not_pretend_it_worked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bootstrap, "_FROZEN", True)
        monkeypatch.setattr(bootstrap, "_console_hidden", False)
        monkeypatch.setattr(bootstrap, "_apply_console_visibility", lambda visible: False)
        bootstrap.set_console_visible(False)
        assert bootstrap._console_hidden is False


class TestReadyHandshake:
    """The two halves of the handshake live in different files and must agree.

    A rename on either side is silent: the console would simply never hide, or
    would hide before the window existed.
    """

    def _launcher_source(self) -> str:
        return (Path(bootstrap.__file__).parent / "launcher.py").read_text(encoding="utf-8")

    def test_the_app_signals_the_variable_the_launcher_sets(self) -> None:
        assert 'os.environ["JAV_TRANS_READY_FILE"] = str(READY_PATH)' in Path(
            bootstrap.__file__
        ).read_text(encoding="utf-8")
        assert 'os.getenv("JAV_TRANS_READY_FILE"' in self._launcher_source()

    def test_the_signal_waits_for_the_server_instead_of_sleeping(self) -> None:
        source = self._launcher_source()
        # Signalled from _bind, i.e. once the window object exists, and only
        # after the socket answers - not after a fixed sleep.
        assert "if not _wait_for_server(PORT):" in source
        assert "time.sleep(1.5)" not in source

    def test_a_dead_server_thread_is_reported_not_hidden(self) -> None:
        source = self._launcher_source()
        assert "_server_error.append(exc)" in source
        assert "--doctor" in source


class TestLaunchRecovery:
    def test_the_ready_file_hides_the_console(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ready = tmp_path / "tmp" / ".app-ready"
        monkeypatch.setattr(bootstrap, "READY_PATH", ready)
        visible: list[bool] = []
        monkeypatch.setattr(bootstrap, "set_console_visible", lambda value: visible.append(value))
        monkeypatch.setattr(bootstrap, "venv_python", lambda: Path("python"))

        def fake_run(command):
            assert bootstrap.os.environ["JAV_TRANS_READY_FILE"] == str(ready)
            ready.parent.mkdir(parents=True, exist_ok=True)
            ready.write_text("123", encoding="utf-8")
            for _ in range(200):
                if visible:
                    return 0
                bootstrap.time.sleep(0.02)
            return 0

        monkeypatch.setattr(bootstrap, "run_streaming", fake_run)
        assert bootstrap.launch([]) == 0
        assert visible == [False]
        assert not ready.exists()

    def test_a_failed_launch_brings_the_console_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "READY_PATH", tmp_path / "tmp" / ".app-ready")
        visible: list[bool] = []
        monkeypatch.setattr(bootstrap, "set_console_visible", lambda value: visible.append(value))
        monkeypatch.setattr(bootstrap, "run_streaming", lambda command: 1)
        monkeypatch.setattr(bootstrap, "venv_python", lambda: Path("python"))
        assert bootstrap.launch([]) == 1
        assert visible == [True]

    def test_keep_console_leaves_the_signal_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "READY_PATH", tmp_path / "tmp" / ".app-ready")
        monkeypatch.setenv("JAV_TRANS_READY_FILE", "a stale value")
        monkeypatch.setattr(bootstrap, "set_console_visible", lambda value: None)
        monkeypatch.setattr(bootstrap, "venv_python", lambda: Path("python"))
        seen: list[str | None] = []
        monkeypatch.setattr(
            bootstrap,
            "run_streaming",
            lambda command: seen.append(bootstrap.os.environ.get("JAV_TRANS_READY_FILE")) or 0,
        )
        bootstrap.launch([], hide_when_ready=False)
        assert seen == [None]

    def test_a_failed_launch_is_diagnosed_and_retried_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launches: list[int] = []
        monkeypatch.setattr(bootstrap, "launch", lambda extra, **kwargs: launches.append(1) or 1)
        monkeypatch.setattr(bootstrap, "doctor", lambda **kwargs: (True, True))
        assert bootstrap.launch_with_recovery([], assume_yes=True, repaired_already=False) == 1
        # Two launches, not a loop: the second failure is reported, not re-doctored.
        assert len(launches) == 2

    def test_a_launch_that_fails_right_after_a_repair_is_not_re_repaired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The environment was rebuilt seconds ago; re-diagnosing it would only
        repeat the verification the repair already did."""
        monkeypatch.setattr(bootstrap, "launch", lambda extra, **kwargs: 1)
        monkeypatch.setattr(
            bootstrap, "doctor", lambda **kwargs: pytest.fail("must not re-run the doctor")
        )
        assert bootstrap.launch_with_recovery([], assume_yes=True, repaired_already=True) == 1

    def test_a_healthy_launch_never_calls_the_doctor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bootstrap, "launch", lambda extra, **kwargs: 0)
        monkeypatch.setattr(
            bootstrap, "doctor", lambda **kwargs: pytest.fail("no diagnosis without a failure")
        )
        assert bootstrap.launch_with_recovery([], assume_yes=True, repaired_already=False) == 0

    def test_a_failure_with_a_healthy_environment_points_elsewhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WebView2 missing is not a dependency problem, and saying "已修复" would
        send the user round the same loop."""
        printed: list[str] = []
        monkeypatch.setattr(bootstrap, "launch", lambda extra, **kwargs: 1)
        monkeypatch.setattr(bootstrap, "doctor", lambda **kwargs: (True, False))
        monkeypatch.setattr(bootstrap, "log", lambda message="": printed.append(message))
        assert bootstrap.launch_with_recovery([], assume_yes=True, repaired_already=False) == 1
        assert any("WebView2" in line for line in printed)
        assert any("--doctor" in line for line in printed)
