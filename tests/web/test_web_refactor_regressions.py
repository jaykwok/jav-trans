"""Targeted regression tests for Web backend/frontend security, reliability, and UX fixes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from pipeline import artifact_store
from web import broadcaster, output_leases, pipeline_manager as pm, security, workers
from web.app import create_app
from web.job_store import JobPersistenceError, JobStoreWriter, _CommitGate
from web.models import JobSpec, JobState, SettingsUpdate
from web.routes import config as config_routes, events as events_routes, files as files_routes
from web.state_clock import StateClock


# ---------------------------------------------------------------------------
# P0-1: Security helpers (pure sync unit tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1:2233", "127.0.0.1"),
        ("LOCALHOST:2233", "localhost"),
        ("[::1]", "::1"),
        ("[::1]:2233", "::1"),
        ("::1", "::1"),
        ("test", "test"),
        ("testserver", "testserver"),
        ("evil.example", "evil.example"),
        ("", None),
        ("localhost:99999", None),
        ("localhost:abc", None),
        ("localhost/path", None),
        ("user@localhost", None),
        ("[::1", None),
    ],
)
def test_security_host_header_parsing(header: str, expected: str | None) -> None:
    assert security.extract_host_hostname(header) == expected


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("http://127.0.0.1:2233", "127.0.0.1"),
        ("https://localhost:8443", "localhost"),
        ("http://[::1]:2233", "::1"),
        ("http://test", "test"),
        ("null", None),
        ("NULL", None),
        ("file:///C:/evil.html", None),
        ("javascript:alert(1)", None),
        ("http://user:pass@127.0.0.1:2233", None),
        ("http://127.0.0.1:2233/path", None),
        ("https://attacker.example", "attacker.example"),
    ],
)
def test_security_origin_parsing(origin: str, expected: str | None) -> None:
    assert security.extract_origin_hostname(origin) == expected


def test_security_validate_local_request_rules() -> None:
    def _headers(**pairs: str) -> list[tuple[bytes, bytes]]:
        return [(k.encode("latin-1"), v.encode("latin-1")) for k, v in pairs.items()]

    # Allowed hosts & same-origin / direct calls.
    assert security.validate_local_request("GET", "/", _headers(host="127.0.0.1:2233"))[0]
    assert security.validate_local_request("POST", "/api/jobs", _headers(host="test"))[0]
    assert security.validate_local_request(
        "POST",
        "/api/jobs",
        _headers(host="[::1]:2233", origin="http://[::1]:2233", **{"sec-fetch-site": "same-origin"}),
    )[0]
    assert security.validate_local_request(
        "GET",
        "/api/jobs",
        _headers(host="localhost:2233", **{"sec-fetch-site": "none"}),
    )[0]

    # Blocked DNS rebinding / missing / duplicate conflicting Host.
    assert not security.validate_local_request("GET", "/", [])[0]
    assert not security.validate_local_request("GET", "/", _headers(host="rebind.attacker.example:2233"))[0]
    assert not security.validate_local_request(
        "GET",
        "/",
        [(b"host", b"127.0.0.1:2233"), (b"host", b"localhost:2233")],
    )[0]

    # Blocked cross-origin & Sec-Fetch-Site on unsafe methods and /api/*.
    assert not security.validate_local_request(
        "POST", "/api/jobs", _headers(host="127.0.0.1:2233", origin="https://evil.example")
    )[0]
    assert not security.validate_local_request(
        "POST", "/api/jobs", _headers(host="127.0.0.1:2233", origin="null")
    )[0]
    assert not security.validate_local_request(
        "GET",
        "/api/settings",
        _headers(host="127.0.0.1:2233", **{"sec-fetch-site": "cross-site"}),
    )[0]
    assert not security.validate_local_request(
        "DELETE",
        "/api/jobs/abc",
        _headers(host="127.0.0.1:2233", **{"sec-fetch-site": "same-site"}),
    )[0]


# ---------------------------------------------------------------------------
# P1-5: Pydantic input validation
# ---------------------------------------------------------------------------


def test_settings_update_and_job_spec_validation() -> None:
    valid = SettingsUpdate(
        api_key="sk-valid-123",
        base_url="https://api.example.test/v1",
        proxy_host="alice:pass@127.0.0.1",
        proxy_port=7890,
    )
    assert valid.base_url == "https://api.example.test/v1"
    assert SettingsUpdate(base_url="").base_url == ""
    assert SettingsUpdate(proxy_host="").proxy_host == ""

    for bad_url in ("ftp://example.com", "javascript:alert(1)", "not-a-url", "https://bad host/v1"):
        with pytest.raises(ValidationError):
            SettingsUpdate(base_url=bad_url)

    for bad_host in ("http://127.0.0.1", "127.0.0.1/path", "127.0.0.1?q=1", "127.0.0.1#frag", "bad host"):
        with pytest.raises(ValidationError):
            SettingsUpdate(proxy_host=bad_host)

    for bad_key in ("line1\nline2", "line1\rline2", "null\x00byte", "x" * 5000):
        with pytest.raises(ValidationError):
            SettingsUpdate(api_key=bad_key)

    with pytest.raises(ValidationError):
        JobSpec(video_paths=["   "])
    with pytest.raises(ValidationError):
        JobSpec(video_paths=["bad\x00path.mp4"])
    with pytest.raises(ValidationError):
        JobSpec(video_paths=["ok.mp4"], output_dir="bad\x00dir")


# ---------------------------------------------------------------------------
# P1-1 / P1-2: Broadcaster backpressure, newline stripping, sync progress update
# ---------------------------------------------------------------------------


def test_events_matches_job_delivers_control_events() -> None:
    assert events_routes._matches_job('{"type": "connected"}', "job-1") is True
    assert events_routes._matches_job('{"type": "resync"}', "job-1") is True
    assert events_routes._matches_job('{"job_id": "job-1", "stage": "asr"}', "job-1") is True
    assert events_routes._matches_job('{"job_id": "job-2", "stage": "asr"}', "job-1") is False


def test_broadcaster_backpressure_strips_newlines_and_updates_memory_before_push(monkeypatch) -> None:
    assert not hasattr(broadcaster, "tcp_listener")

    # Backpressure drops oldest progress and injects resync without losing terminal `done`.
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
    q.put_nowait('{"type": "connected"}')
    q.put_nowait('{"job_id": "j", "stage": "translation", "phase": "progress", "extra": {"translated": 1}}')
    q.put_nowait('{"job_id": "j", "stage": "translation", "phase": "progress", "extra": {"translated": 2}}')
    q.put_nowait('{"job_id": "j", "stage": "translation", "phase": "progress", "extra": {"translated": 3}}')

    done_line = '{"job_id": "j", "stage": "write_output", "phase": "done"}'
    broadcaster._enqueue_with_backpressure(q, done_line, is_critical=True)
    drained = []
    while not q.empty():
        drained.append(q.get_nowait())
    assert done_line in drained
    assert broadcaster.RESYNC_EVENT_LINE in drained
    assert len(drained) == 4

    # Sync memory update happens BEFORE subscriber queue receives the SSE frame,
    # and embedded newlines are stripped.
    job = JobState(
        id="sync-order-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-11-27T00:00:00.000+00:00",
        run_id="run-sync",
        status="asr",
        revision=1,
    )
    monkeypatch.setattr(pm, "_jobs", {job.id: job})
    observed_revisions: list[int] = []

    class ObservingQueue(asyncio.Queue):
        def put_nowait(self, item: str) -> None:
            observed_revisions.append(pm._jobs[job.id].revision)
            assert "\n" not in item and "\r" not in item
            super().put_nowait(item)

    sub = ObservingQueue(maxsize=10)
    monkeypatch.setattr(broadcaster, "_subscribers", [sub])
    raw_line = (
        '{"job_id": "sync-order-job", "run_id": "run-sync", '
        '"stage": "translation", "phase": "progress", "extra": {"translated": 5, "expected": 20}}\r\n'
    )
    broadcaster.publish(raw_line)
    assert observed_revisions == [pm._jobs[job.id].revision]
    assert pm._jobs[job.id].revision > 1
    assert pm._jobs[job.id].progress["translated"] == 5
    assert not sub.empty()


# ---------------------------------------------------------------------------
# StateClock, OutputLeases (JOB_TEMP_DIR), JobStoreWriter, Logger cleanup
# ---------------------------------------------------------------------------


def test_state_clock_atomic_epoch_and_tombstones(tmp_path: Path) -> None:
    clock = StateClock(tombstone_limit=3)
    epoch_file = tmp_path / "state-epoch.json"
    first = clock.resolve_epoch(epoch_file, time_fn=lambda: 1_700_000_000.0)
    assert json.loads(epoch_file.read_text(encoding="utf-8")) == {"epoch": first}

    clock2 = StateClock(tombstone_limit=3)
    second = clock2.resolve_epoch(epoch_file, time_fn=lambda: 1_700_000_000.0)
    assert second == first + 1

    for idx in range(5):
        clock2.entomb(f"job-{idx}", idx + 1)
    assert len(clock2.tombstones) == 3
    assert clock2.deleted_revision("job-0") is None
    assert clock2.deleted_revision("job-4") == 5


def test_output_leases_respects_job_temp_dir_env(tmp_path: Path, monkeypatch) -> None:
    custom_rel = "custom_scratch/jobs"
    monkeypatch.setenv("JOB_TEMP_DIR", custom_rel)
    assert output_leases.jobs_temp_root(tmp_path) == tmp_path / custom_rel

    custom_abs = tmp_path / "abs_jobs"
    monkeypatch.setenv("JOB_TEMP_DIR", str(custom_abs))
    assert output_leases.jobs_temp_root(tmp_path) == custom_abs

    monkeypatch.delenv("JOB_TEMP_DIR", raising=False)
    assert output_leases.jobs_temp_root(tmp_path) == tmp_path / "tmp" / "jobs"


def test_commit_gate_skips_fsync_when_cancelled_and_writer_caches_terminal_jobs(
    tmp_path: Path, monkeypatch
) -> None:
    gate = _CommitGate()
    assert gate.cancel() is True
    assert gate.is_cancelled() is True

    fsync_called = False

    def fail_if_fsync(_fd):
        nonlocal fsync_called
        fsync_called = True
        raise AssertionError("fsync must not be called when commit gate is already cancelled")

    monkeypatch.setattr(os, "fsync", fail_if_fsync)
    target = tmp_path / "jobs.json"
    sample = JobState(
        id="job-1",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-11-27T00:00:00.000+00:00",
        status="done",
        revision=5,
    )
    with pytest.raises(JobPersistenceError):
        pm._write_jobs_snapshot([sample], path=target, commit=gate.commit)
    assert fsync_called is False
    assert not target.exists()

    # Terminal jobs with revision > 0 are cached across snapshots instead of deep-copied each time.
    writer = JobStoreWriter(lambda _snap: None)
    snap1 = writer._freeze_snapshot([sample])
    snap2 = writer._freeze_snapshot([sample])
    assert snap1[0] is snap2[0]


def test_close_asr_artifacts_logger_releases_file_handle_and_ignores_mocks(tmp_path: Path) -> None:
    log_file = tmp_path / "sample.run.log"
    logger = logging.getLogger("test.web.refactor.runlog")
    handler = logging.FileHandler(log_file, encoding="utf-8")
    logger.addHandler(handler)
    artifacts = SimpleNamespace(logger=logger)

    workers.close_asr_artifacts_logger(artifacts, pm.pipeline_main)
    assert artifacts.logger is None
    assert logger.handlers == []
    # File must be unlocked and removable on Windows.
    log_file.unlink()
    assert not log_file.exists()

    # Mock objects without a real logging.Logger must not error.
    workers.close_asr_artifacts_logger(object(), pm.pipeline_main)
    workers.close_asr_artifacts_logger({"job_id": "x"}, pm.pipeline_main)
    workers.close_asr_artifacts_logger(None, pm.pipeline_main)


# ---------------------------------------------------------------------------
# Smart player launch, Explorer/Finder reveal, artifact roots, folder scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exe", "expected_tail"),
    [
        (r"C:\Program Files\DAUM\PotPlayer\PotPlayerMini64.exe", ["V", "/sub=S"]),
        (r"C:\PotPlayer\PotPlayerMini.exe", ["V", "/sub=S"]),
        (r"C:\PotPlayer\PotPlayer64.exe", ["V", "/sub=S"]),
        (r"C:\PotPlayer\PotPlayer.exe", ["V", "/sub=S"]),
        (r"C:\Tools\mpv\mpv.exe", ["V", "--sub-file=S"]),
        ("/usr/bin/mpv", ["V", "--sub-file=S"]),
        (r"C:\Program Files\VideoLAN\VLC\vlc.exe", ["V", "--sub-file=S"]),
        ("/Applications/VLC.app/Contents/MacOS/vlc", ["V", "--sub-file=S"]),
        (r"C:\Program Files\MPC-HC\mpc-hc64.exe", ["V", "/sub", "S"]),
        (r"C:\Program Files\MPC-HC\mpc-hc.exe", ["V", "/sub", "S"]),
        (r"C:\Program Files\MPC-BE\mpc-be64.exe", ["V", "/sub", "S"]),
        (r"C:\Program Files\MPC-BE\mpc-be.exe", ["V", "/sub", "S"]),
        (r"C:\Program Files\SMPlayer\smplayer.exe", ["-sub", "S", "V"]),
        ("/usr/bin/smplayer", ["-sub", "S", "V"]),
    ],
)
def test_build_player_launch_command_supported_players(
    exe: str, expected_tail: list[str]
) -> None:
    video = r"D:\My Videos\sample a.mp4"
    srt = r"D:\My Videos\sample a.srt"
    expected = [exe] + [
        video if tok == "V" else srt if tok == "S" else tok.replace("=S", f"={srt}")
        for tok in expected_tail
    ]
    assert files_routes._build_player_launch_command(exe, video, srt) == expected


@pytest.mark.parametrize(
    ("exe", "srt"),
    [
        (r"C:\Windows\System32\wmplayer.exe", r"D:\Videos\a.srt"),
        (r"C:\Program Files\WindowsApps\Microsoft.Media.Player.exe", r"D:\Videos\a.srt"),
        (None, r"D:\Videos\a.srt"),
        (r"C:\Program Files\DAUM\PotPlayer\PotPlayerMini64.exe", None),
        ("", r"D:\Videos\a.srt"),
    ],
)
def test_build_player_launch_command_returns_none_for_unknown_or_missing_subtitle(
    exe: str | None, srt: str | None
) -> None:
    assert files_routes._build_player_launch_command(exe, r"D:\Videos\a.mp4", srt) is None


def test_open_video_with_subtitle_launches_player_and_falls_back(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "sample a.mp4"
    video.write_bytes(b"video")
    srt = tmp_path / "sample a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")

    launched: list[tuple[list[str], dict]] = []
    fallback: list[Path] = []

    monkeypatch.setattr(files_routes.os, "name", "nt")
    monkeypatch.setattr(
        files_routes,
        "_query_windows_associated_executable",
        lambda _ext: r"C:\Tools\PotPlayerMini64.exe",
    )
    monkeypatch.setattr(
        files_routes.subprocess,
        "Popen",
        lambda cmd, **kwargs: launched.append((list(cmd), kwargs)),
    )
    monkeypatch.setattr(
        files_routes,
        "_open_path_with_os",
        lambda target, *, label: fallback.append(target),
    )

    files_routes._open_video_with_subtitle(video, srt)
    assert len(launched) == 1
    assert launched[0][0] == [r"C:\Tools\PotPlayerMini64.exe", str(video), f"/sub={srt}"]
    assert fallback == []

    # When Popen fails, fall back cleanly to OS default launch.
    def failing_popen(_cmd, **_kwargs):
        raise OSError("player missing")

    monkeypatch.setattr(files_routes.subprocess, "Popen", failing_popen)
    files_routes._open_video_with_subtitle(video, srt)
    assert fallback == [video]

    # When player is unknown, skip Popen and go straight to OS default launch.
    fallback.clear()
    monkeypatch.setattr(
        files_routes,
        "_query_windows_associated_executable",
        lambda _ext: r"C:\Windows\wmplayer.exe",
    )
    files_routes._open_video_with_subtitle(video, srt)
    assert fallback == [video]


def test_reveal_path_in_folder_selects_file_and_opens_dir(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "My Videos"
    folder.mkdir()
    srt = folder / "sample a.srt"
    srt.write_text("1\n", encoding="utf-8")

    popen_calls: list[object] = []
    os_open_calls: list[Path] = []
    monkeypatch.setattr(
        files_routes.subprocess,
        "Popen",
        lambda cmd, **_kwargs: popen_calls.append(cmd),
    )
    monkeypatch.setattr(
        files_routes,
        "_open_path_with_os",
        lambda target, *, label: os_open_calls.append(target),
    )

    # Windows file -> explorer.exe /select,"<path>"
    monkeypatch.setattr(files_routes.os, "name", "nt")
    files_routes._reveal_path_in_folder_with_os(srt)
    assert popen_calls == [f'explorer.exe /select,"{srt}"']
    assert os_open_calls == []

    # Windows directory -> direct _open_path_with_os
    popen_calls.clear()
    files_routes._reveal_path_in_folder_with_os(folder)
    assert popen_calls == []
    assert os_open_calls == [folder]

    # Windows Explorer failure -> graceful fallback to parent directory
    os_open_calls.clear()
    monkeypatch.setattr(
        files_routes.subprocess,
        "Popen",
        lambda _cmd, **_kwargs: (_ for _ in ()).throw(OSError("explorer failed")),
    )
    files_routes._reveal_path_in_folder_with_os(srt)
    assert os_open_calls == [folder]

    # macOS file -> open -R <path>
    popen_calls.clear()
    os_open_calls.clear()
    monkeypatch.setattr(files_routes.os, "name", "posix")
    monkeypatch.setattr(files_routes.sys, "platform", "darwin")
    monkeypatch.setattr(
        files_routes.subprocess,
        "Popen",
        lambda cmd, **_kwargs: popen_calls.append(cmd),
    )
    files_routes._reveal_path_in_folder_with_os(srt)
    assert popen_calls == [["open", "-R", str(srt)]]


def test_authorized_artifact_roots_handles_moved_source_video(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    video_dir = tmp_path / "external_videos"
    video_dir.mkdir()
    # Source video was moved/deleted after subtitle generation, but its folder still exists!
    missing_video = video_dir / "sample-a.mp4"
    exported_srt = video_dir / "sample-a.srt"
    exported_srt.write_text("1\n", encoding="utf-8")

    monkeypatch.setattr(files_routes, "PROJECT_ROOT", project_root)
    job = JobState(
        id="moved-video-job",
        spec=JobSpec(video_paths=[str(missing_video)], output_dir=None),
        created_at="2026-11-27T00:00:00.000+00:00",
        status="done",
        output_target=str(exported_srt),
        output_generation=1,
        run_id="run-1",
    )

    roots = files_routes._authorized_artifact_roots(job)
    assert roots[0] == video_dir.resolve()
    assert project_root.resolve() in roots
    assert roots.index(video_dir.resolve()) < roots.index(project_root.resolve())
    resolved = files_routes._resolve_open_folder_target_blocking(job, str(exported_srt))
    assert resolved == exported_srt.resolve()


def test_scan_video_folder_limits_depth_m2ts_and_picker_escaping(
    tmp_path: Path, monkeypatch
) -> None:
    # Directory named foo.mp4 and NUL paths must be rejected.
    fake_dir_video = tmp_path / "dir_named.mp4"
    fake_dir_video.mkdir()
    assert files_routes._resolve_existing_video_path(str(fake_dir_video)) is None
    assert files_routes._resolve_existing_video_path("bad\x00.mp4") is None

    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    (scan_root / "clip1.m2ts").write_bytes(b"m2ts")
    (scan_root / "clip2.mp4").write_bytes(b"mp4")
    (scan_root / "clip3.mkv").write_bytes(b"mkv")
    deep = scan_root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "too_deep.mp4").write_bytes(b"deep")

    found, truncated = files_routes._scan_video_folder_with_limit(
        str(scan_root), max_files=2, max_depth=1
    )
    assert truncated is True
    assert len(found) == 2

    found_all, truncated_all = files_routes._scan_video_folder_with_limit(
        str(scan_root), max_files=10, max_depth=1
    )
    assert truncated_all is False
    names = {Path(p).name for p in found_all}
    assert names == {"clip1.m2ts", "clip2.mp4", "clip3.mkv"}
    assert "too_deep.mp4" not in names

    # Picker parameter escaping & .m2ts filters.
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        captured_cmds.append(list(cmd))
        return files_routes.subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(files_routes.subprocess, "run", fake_run)
    monkeypatch.setattr(files_routes.shutil, "which", lambda _name: "/usr/bin/zenity")

    files_routes._pick_files_windows()
    assert "*.m2ts" in captured_cmds[-1][-1]

    files_routes._pick_folder_windows("Select 'quoted'\nfolder")
    assert "Select ''quoted'' folder" in captured_cmds[-1][-1]

    files_routes._pick_files_macos()
    assert '"m2ts"' in captured_cmds[-1][-1]

    files_routes._pick_folder_macos('Choose "quoted" folder')
    assert captured_cmds[-1][-2:] == ["--", 'Choose "quoted" folder']

    files_routes._pick_files_linux()
    assert any("*.m2ts" in arg for arg in captured_cmds[-1])


# ---------------------------------------------------------------------------
# Async integration tests (full ASGI / worker / cancel / shutdown paths)
# ---------------------------------------------------------------------------


def test_security_middleware_blocks_bad_host_and_origin_over_asgi() -> None:
    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            ok_resp = await client.get("/api/config")
            assert ok_resp.status_code == 200

            bad_host = await client.get("/api/config", headers={"Host": "attacker.example:2233"})
            assert bad_host.status_code == 403

            bad_origin = await client.post(
                "/api/settings",
                json={"target_lang": "简体中文"},
                headers={"Origin": "https://attacker.example"},
            )
            assert bad_origin.status_code == 403

            null_origin = await client.post(
                "/api/settings",
                json={"target_lang": "简体中文"},
                headers={"Origin": "null"},
            )
            assert null_origin.status_code == 403

            cross_site = await client.get(
                "/api/settings",
                headers={"Sec-Fetch-Site": "cross-site"},
            )
            assert cross_site.status_code == 403

            ipv6_ok = await client.get(
                "/api/config",
                headers={
                    "Host": "[::1]:2233",
                    "Origin": "http://[::1]:2233",
                    "Sec-Fetch-Site": "same-origin",
                },
            )
            assert ipv6_ok.status_code == 200

    asyncio.run(scenario())


def test_cancel_and_remove_protect_uncommitted_and_saving_outputs(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
        monkeypatch.setattr(pm, "_job_store_writer", pm._JobStoreWriter())
        monkeypatch.setattr(pm, "_jobs", {})
        monkeypatch.setattr(pm, "_cancel_events", {})
        monkeypatch.setattr(pm, "_tombstones", {})
        monkeypatch.setattr(pm, "gpu_queue", asyncio.Queue())
        monkeypatch.setattr(pm, "trans_queue", asyncio.Queue())

        job = (
            await pm.create_job(
                JobSpec(video_paths=[str(tmp_path / "sample-a.mp4")], skip_translation=True)
            )
        )[0]
        lease = pm.output_lease_for_job(job)
        assert lease is not None
        revoke_calls: list[artifact_store.OutputLease] = []
        orig_revoke = artifact_store.revoke_output
        monkeypatch.setattr(
            artifact_store,
            "revoke_output",
            lambda item: revoke_calls.append(item) or orig_revoke(item),
        )

        # Cancelling a publish_failed or save_failed job must NEVER revoke its lease.
        async with pm._state_lock:
            pm._jobs[job.id].status = "publish_failed"
        assert await pm.cancel_job(job.id) is True
        assert revoke_calls == []

        async with pm._state_lock:
            pm._jobs[job.id].status = "save_failed"
        assert await pm.cancel_job(job.id) is True
        assert revoke_calls == []

        # DELETE /api/jobs/{id} on save_failed returns 409 without revoking output lease.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.delete(f"/api/jobs/{job.id}")
            assert resp.status_code == 409
        assert revoke_calls == []

        # Cancelling an active job with corrupt output identity still succeeds cleanly.
        async with pm._state_lock:
            pm._jobs[job.id].status = "asr"
            pm._jobs[job.id].output_target = str(tmp_path / "wrong.srt")
        assert await pm.cancel_job(job.id) is True
        assert (await pm.get_job(job.id)).status == "cancelling"

    asyncio.run(scenario())


def test_workers_and_shutdown_close_run_loggers_and_skip_translation_stage(
    tmp_path: Path, monkeypatch
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
        monkeypatch.setattr(pm, "_job_store_writer", pm._JobStoreWriter())
        monkeypatch.setattr(pm, "_jobs", {})
        monkeypatch.setattr(pm, "_cancel_events", {})
        monkeypatch.setattr(pm, "_tombstones", {})
        monkeypatch.setattr(pm, "gpu_queue", asyncio.Queue())
        monkeypatch.setattr(pm, "trans_queue", asyncio.Queue())

        # Verify shutdown_executor closes queued trans_queue loggers and stays lazy (_executor=None).
        log_path = tmp_path / "queued.run.log"
        logger = logging.getLogger("test.web.refactor.shutdown_drain")
        logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
        queued_artifacts = SimpleNamespace(logger=logger)
        dummy_job = JobState(
            id="queued-drain",
            spec=JobSpec(video_paths=["sample.mp4"], skip_translation=True),
            created_at="2026-11-27T00:00:00.000+00:00",
            status="writing",
        )
        await pm.trans_queue.put((dummy_job, queued_artifacts))
        await pm.shutdown_executor()
        assert queued_artifacts.logger is None
        assert pm._executor is None
        log_path.unlink()

        # Verify skip_translation=True transitions through `writing`/`write_output`, not `translating`.
        observed_statuses: list[str] = []
        job = (
            await pm.create_job(
                JobSpec(video_paths=[str(tmp_path / "sample-ja.mp4")], skip_translation=True)
            )
        )[0]
        monkeypatch.setattr(pm, "_run_asr_alignment", lambda *_args: SimpleNamespace(logger=None))

        def fake_trans(j, _arts, _cancel, _run_id):
            observed_statuses.append(j.status)
            return []

        monkeypatch.setattr(pm, "_run_translation_and_write", fake_trans)
        w1 = asyncio.create_task(pm.gpu_worker())
        w2 = asyncio.create_task(pm.translation_worker())
        try:
            await asyncio.wait_for(pm.gpu_queue.join(), 3.0)
            await asyncio.wait_for(pm.trans_queue.join(), 3.0)
        finally:
            w1.cancel()
            w2.cancel()
            await asyncio.gather(w1, w2, return_exceptions=True)
            await pm.shutdown_executor()

        assert observed_statuses == ["writing"]
        assert (await pm.get_job(job.id)).status == "done"

    asyncio.run(scenario())


def test_models_endpoint_redacts_credentials_and_handles_invalid_url(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setenv("API_KEY", "sk-secret-key")
        monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://alice:topsecret@api.example.test/v1")

        class BrokenClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, **_kwargs):
                raise httpx.InvalidURL(f"bad url {url}")

        original_async_client = httpx.AsyncClient
        monkeypatch.setattr(config_routes.httpx, "AsyncClient", BrokenClient)
        transport = httpx.ASGITransport(app=create_app())
        async with original_async_client(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/models")
        assert resp.status_code == 502
        detail = resp.json()["detail"]
        assert "topsecret" not in detail
        assert "***@" in detail

    asyncio.run(scenario())


def test_icon_asset_missing_returns_404_and_present_sets_cache_control(
    tmp_path: Path, monkeypatch
) -> None:
    async def scenario() -> None:
        import web.app as web_app

        monkeypatch.setattr(web_app, "resource_root", lambda: tmp_path)
        (tmp_path / "src" / "assets" / "images").mkdir(parents=True)
        (tmp_path / "src" / "assets" / "images" / "icon.png").write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIEND\xaeB`\x82"
        )
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            ok = await client.get("/assets/images/icon.png")
            missing = await client.get("/assets/images/icon.ico")
        assert ok.status_code == 200
        assert "max-age=" in ok.headers.get("cache-control", "")
        assert missing.status_code == 404

    asyncio.run(scenario())
