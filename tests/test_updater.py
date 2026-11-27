from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

import bootstrap
from core import updater
from web import pipeline_manager
from web.app import create_app
from web.models import JobSpec, JobState
from web.routes import config as config_routes


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_release_zip(
    *,
    version: str,
    extra_files: dict[str, bytes] | None = None,
    wrap_folder: str | None = "jav-sub",
    include_index: bool = True,
    corrupt_index: dict[str, Any] | None = None,
    raw_members: list[tuple[str | zipfile.ZipInfo, bytes]] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Build a realistic packaged release zip and matching release_manifest_v2."""
    files: dict[str, bytes] = {
        "jav-sub.exe": f"fake-exe-{version}".encode("utf-8"),
        "launcher.py": f"# launcher {version}\n".encode("utf-8"),
        "pyproject.toml": f'[project]\nname = "jav-sub"\nversion = "{version}"\n'.encode("utf-8"),
        "uv.lock": b"version = 1\n",
        "README.txt": b"readme\n",
        "src/main.py": f"VERSION = '{version}'\n".encode("utf-8"),
        "bin/ffmpeg.exe": b"ffmpeg-bytes",
    }
    if extra_files:
        files.update(extra_files)

    if include_index:
        index_obj = (
            corrupt_index
            if corrupt_index is not None
            else {
                "schema": updater.RELEASE_INDEX_SCHEMA,
                "files": sorted(files.keys()),
            }
        )
        files[updater.RELEASE_INDEX_NAME] = (
            json.dumps(index_obj, ensure_ascii=False, indent=1) + "\n"
        ).encode("utf-8")

    manifest_entries = [
        {
            "path": rel,
            "root": "dist",
            "category": "payload",
            "bytes": len(content),
            "sha256": _sha256_bytes(content),
        }
        for rel, content in sorted(files.items())
    ]
    manifest = {
        "schema": updater.RELEASE_MANIFEST_SCHEMA,
        "dist_root": wrap_folder or "jav-sub",
        "files": manifest_entries,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, content in sorted(files.items()):
            arcname = f"{wrap_folder}/{rel}" if wrap_folder else rel
            zf.writestr(arcname, content)
        for member, content in raw_members or []:
            zf.writestr(member, content)
    return buf.getvalue(), manifest


def _make_installed_dir(root: Path, version: str = "1.9.0") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "jav-sub.exe").write_bytes(f"old-exe-{version}".encode("utf-8"))
    (root / "launcher.py").write_text(f"# launcher {version}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "jav-sub"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "main.py").write_text(f"VERSION = '{version}'\n", encoding="utf-8")
    bootstrap.write_release_index(root)
    return root


def _release_payload(
    *,
    tag: str,
    zip_bytes: bytes,
    zip_sha: str | None = None,
    include_digest: bool = True,
    include_sha_asset: bool = True,
    include_manifest: bool = True,
    draft: bool = False,
    prerelease: bool = False,
    archive_url: str = "https://example.invalid/releases/jav-sub-windows-x64.zip",
) -> dict[str, Any]:
    sha = zip_sha if zip_sha is not None else _sha256_bytes(zip_bytes)
    assets: list[dict[str, Any]] = []
    archive_item: dict[str, Any] = {
        "name": updater.UPDATE_ARCHIVE_NAME,
        "browser_download_url": archive_url,
        "size": len(zip_bytes),
    }
    if include_digest and sha:
        archive_item["digest"] = f"sha256:{sha}"
    assets.append(archive_item)
    if include_sha_asset:
        assets.append(
            {
                "name": updater.UPDATE_SHA256_NAME,
                "browser_download_url": "https://example.invalid/releases/jav-sub-windows-x64.zip.sha256",
                "size": 100,
            }
        )
    if include_manifest:
        assets.append(
            {
                "name": updater.UPDATE_MANIFEST_NAME,
                "browser_download_url": "https://example.invalid/releases/release-manifest.json",
                "size": 500,
            }
        )
    return {
        "tag_name": tag,
        "draft": draft,
        "prerelease": prerelease,
        "html_url": f"https://github.com/jaykwok/jav-sub/releases/tag/{tag}",
        "body": f"Release notes for {tag}",
        "published_at": "2026-11-26T12:00:00Z",
        "assets": assets,
    }


def _mock_http_client(routes: dict[str, tuple[int, bytes | str | dict[str, Any], dict[str, str] | None]]):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url not in routes:
            return httpx.Response(404, text=f"not found: {url}")
        status, body, headers = routes[url]
        hdrs = dict(headers or {})
        if isinstance(body, dict):
            return httpx.Response(status, json=body, headers=hdrs)
        if isinstance(body, str):
            return httpx.Response(status, text=body, headers=hdrs)
        hdrs.setdefault("Content-Length", str(len(body)))
        return httpx.Response(status, content=body, headers=hdrs)

    transport = httpx.MockTransport(handler)
    return lambda _timeout_s: httpx.Client(transport=transport, follow_redirects=True)


class TestSemverAndSelection:
    def test_parse_semver_accepts_strict_xyz_and_rejects_prereleases(self) -> None:
        assert updater.parse_semver("1.9.0") == (1, 9, 0)
        assert updater.parse_semver("v1.9.10") == (1, 9, 10)
        assert updater.parse_semver("V2.0.1") == (2, 0, 1)
        assert updater.parse_semver("v1.9.10") > updater.parse_semver("v1.9.2")
        assert updater.parse_semver("v2.0.0") > updater.parse_semver("v1.9.9")

        for bad in (
            None,
            "",
            "1.9",
            "1.9.0.1",
            "v1.9.1-rc1",
            "v1.9.1-beta.1",
            "v1.9.1+build1",
            "latest",
        ):
            assert updater.parse_semver(bad) is None

    def test_select_release_ignores_draft_prerelease_and_non_xyz_tags(self) -> None:
        zip_bytes, _ = _build_release_zip(version="1.9.1")
        for kwargs in (
            {"tag": "v1.9.1", "draft": True},
            {"tag": "v1.9.1", "prerelease": True},
            {"tag": "v1.9.1-rc1"},
            {"tag": "nightly"},
        ):
            payload = _release_payload(zip_bytes=zip_bytes, **kwargs)
            is_newer, info = updater.select_release_update(payload, current_version="1.9.0")
            assert is_newer is False
            assert info is None

    def test_select_release_numeric_comparison(self) -> None:
        zip_bytes, _ = _build_release_zip(version="1.9.10")
        payload = _release_payload(tag="v1.9.10", zip_bytes=zip_bytes)

        is_newer, info = updater.select_release_update(payload, current_version="1.9.2")
        assert is_newer is True
        assert info is not None
        assert info.version == "1.9.10"

        is_newer_same, info_same = updater.select_release_update(payload, current_version="1.9.10")
        assert is_newer_same is False
        assert info_same is not None

        is_newer_older, _ = updater.select_release_update(payload, current_version="1.10.0")
        assert is_newer_older is False

    def test_select_release_rejects_missing_sha256_or_insecure_url(self) -> None:
        zip_bytes, _ = _build_release_zip(version="1.9.1")
        no_sha = _release_payload(
            tag="v1.9.1",
            zip_bytes=zip_bytes,
            include_digest=False,
            include_sha_asset=False,
        )
        with pytest.raises(updater.UpdateError, match="SHA-256"):
            updater.select_release_update(no_sha, current_version="1.9.0")

        bad_url = _release_payload(
            tag="v1.9.1",
            zip_bytes=zip_bytes,
            archive_url="http://evil.example.com/jav-sub-windows-x64.zip",
        )
        with pytest.raises(updater.UpdateError, match="安装包资产"):
            updater.select_release_update(bad_url, current_version="1.9.0")

        no_manifest = _release_payload(
            tag="v1.9.1",
            zip_bytes=zip_bytes,
            include_manifest=False,
        )
        with pytest.raises(updater.UpdateError, match="发行清单"):
            updater.select_release_update(no_manifest, current_version="1.9.0")

    def test_parse_sha256_sidecar_variants(self) -> None:
        h1 = "a" * 64
        h2 = "b" * 64
        assert updater.parse_sha256_sidecar(f"{h1}  jav-sub-windows-x64.zip\n") == h1
        assert updater.parse_sha256_sidecar(f"{h1.upper()} *dist/JAV-SUB-WINDOWS-X64.ZIP\n") == h1
        assert updater.parse_sha256_sidecar(f"{h2}  other.zip\n{h1}  jav-sub-windows-x64.zip\n") == h1
        assert updater.parse_sha256_sidecar(f"{h1}\n") == h1
        assert updater.parse_sha256_sidecar(f"{h2}  unrelated.zip\n") is None
        assert updater.parse_sha256_sidecar("not a hash") is None


class TestExtractionAndPayloadVerification:
    def test_safe_extract_unwraps_top_folder_and_verifies_manifest(self, tmp_path: Path) -> None:
        zip_bytes, manifest = _build_release_zip(version="1.9.1")
        archive = tmp_path / "update.zip"
        archive.write_bytes(zip_bytes)

        staged = updater.safe_extract_zip(archive, tmp_path / "extract")
        assert (staged / "release-files.json").is_file()
        assert (staged / "jav-sub.exe").is_file()
        assert updater.verify_staged_payload(staged, expected_version="1.9.1", manifest_payload=manifest) is True

    def test_safe_extract_rejects_zip_slip_absolute_drive_and_symlinks(self, tmp_path: Path) -> None:
        bad_cases = [
            ("../escape.txt", b"boom"),
            ("jav-sub/../../escape.txt", b"boom"),
            ("/etc/passwd", b"boom"),
            ("C:\\Windows\\evil.dll", b"boom"),
        ]
        for idx, (member_name, data) in enumerate(bad_cases):
            archive = tmp_path / f"bad-{idx}.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(member_name, data)
            with pytest.raises(updater.UpdateError, match=r"越界路径|绝对路径"):
                updater.safe_extract_zip(archive, tmp_path / f"out-{idx}")

        # Symlink inside ZIP must be refused.
        sym_archive = tmp_path / "sym.zip"
        with zipfile.ZipFile(sym_archive, "w") as zf:
            info = zipfile.ZipInfo("jav-sub/link")
            info.external_attr = (0o120777 << 16)
            zf.writestr(info, "target")
        with pytest.raises(updater.UpdateError, match="符号链接"):
            updater.safe_extract_zip(sym_archive, tmp_path / "out-sym")

    def test_verify_staged_payload_rejects_protected_dirs_version_mismatch_and_tampered_manifest(
        self, tmp_path: Path
    ) -> None:
        # Protected directory entry in release-files.json
        zip_prot, _ = _build_release_zip(
            version="1.9.1",
            extra_files={"config/config.json": b"{}"},
        )
        arch_prot = tmp_path / "prot.zip"
        arch_prot.write_bytes(zip_prot)
        staged_prot = updater.safe_extract_zip(arch_prot, tmp_path / "ex-prot")
        with pytest.raises(updater.UpdateError, match="受保护目录"):
            updater.verify_staged_payload(staged_prot, expected_version="1.9.1")

        # Valid zip checked against wrong expected version
        zip_ok, manifest = _build_release_zip(version="1.9.1")
        arch_ok = tmp_path / "ok.zip"
        arch_ok.write_bytes(zip_ok)
        staged_ok = updater.safe_extract_zip(arch_ok, tmp_path / "ex-ok")
        with pytest.raises(updater.UpdateError, match="版本号"):
            updater.verify_staged_payload(staged_ok, expected_version="1.9.2")

        # Missing release-manifest.json must be refused
        with pytest.raises(updater.UpdateError, match="发行清单"):
            updater.verify_staged_payload(staged_ok, expected_version="1.9.1", manifest_payload=None)

        # Manifest hash mismatch
        bad_manifest = json.loads(json.dumps(manifest))
        bad_manifest["files"][0]["sha256"] = "0" * 64
        with pytest.raises(updater.UpdateError, match="SHA-256"):
            updater.verify_staged_payload(staged_ok, expected_version="1.9.1", manifest_payload=bad_manifest)

        # Extra unrecorded file inside staged_dir when manifest is provided
        (staged_ok / "sneaky.dll").write_bytes(b"sneaky")
        with pytest.raises(updater.UpdateError, match="未登记的额外文件"):
            updater.verify_staged_payload(staged_ok, expected_version="1.9.1", manifest_payload=manifest)

    def test_validate_archive_member_path_rejects_dot_ads_reserved_and_trailing_chars(self) -> None:
        bad_names = [
            "",
            ".",
            "./",
            "./jav-sub/file.txt",
            "jav-sub/./file.txt",
            "jav-sub//file.txt",
            "jav-sub/file.txt:evil",
            "jav-sub/CON",
            "jav-sub/con.txt",
            "jav-sub/src/NUL.py",
            "jav-sub/COM1.dll",
            "jav-sub/lpt9.a.b",
            "jav-sub/file.txt ",
            "jav-sub/file.txt.",
        ]
        for bad in bad_names:
            with pytest.raises(updater.UpdateError):
                updater._validate_archive_member_path(bad)

    def test_safe_extract_rejects_zip_bomb_and_windows_case_collisions(self, tmp_path: Path) -> None:
        # Member count cap
        many_zip = tmp_path / "many.zip"
        with zipfile.ZipFile(many_zip, "w") as zf:
            for idx in range(5):
                zf.writestr(f"jav-sub/f{idx}.txt", b"x")
        with pytest.raises(updater.UpdateError, match="成员数量"):
            updater.safe_extract_zip(many_zip, tmp_path / "out-many", max_members=3)

        # Uncompressed byte cap
        big_zip = tmp_path / "big.zip"
        with zipfile.ZipFile(big_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("jav-sub/big.bin", b"A" * 4096)
        with pytest.raises(updater.UpdateError, match="体积超过安全上限"):
            updater.safe_extract_zip(big_zip, tmp_path / "out-big", max_total_bytes=1024)

        # Windows case-folding file collision
        case_zip = tmp_path / "case.zip"
        with zipfile.ZipFile(case_zip, "w") as zf:
            zf.writestr("jav-sub/src/Main.py", b"a")
            zf.writestr("jav-sub/src/main.py", b"b")
        with pytest.raises(updater.UpdateError, match="大小写冲突"):
            updater.safe_extract_zip(case_zip, tmp_path / "out-case")

        # Windows case-folding directory collision
        dir_case_zip = tmp_path / "dir-case.zip"
        with zipfile.ZipFile(dir_case_zip, "w") as zf:
            zf.writestr("jav-sub/SRC/a.py", b"a")
            zf.writestr("jav-sub/src/b.py", b"b")
        with pytest.raises(updater.UpdateError, match="大小写冲突"):
            updater.safe_extract_zip(dir_case_zip, tmp_path / "out-dir-case")

    def test_verify_staged_payload_rejects_dirty_git_protected_manifest_and_duplicate_or_case_paths(
        self, tmp_path: Path
    ) -> None:
        zip_ok, manifest = _build_release_zip(version="1.9.1")
        arch_ok = tmp_path / "ok.zip"
        arch_ok.write_bytes(zip_ok)
        staged = updater.safe_extract_zip(arch_ok, tmp_path / "ex-ok")

        # git.dirty is True must be refused; git.dirty is False passes.
        dirty_manifest = json.loads(json.dumps(manifest))
        dirty_manifest["git"] = {"commit": "a" * 40, "dirty": True}
        with pytest.raises(updater.UpdateError, match="脏构建"):
            updater.verify_staged_payload(staged, expected_version="1.9.1", manifest_payload=dirty_manifest)

        clean_manifest = json.loads(json.dumps(manifest))
        clean_manifest["git"] = {"commit": "a" * 40, "dirty": False}
        assert updater.verify_staged_payload(staged, expected_version="1.9.1", manifest_payload=clean_manifest) is True

        # Protected directory in manifest["files"] must be refused.
        prot_manifest = json.loads(json.dumps(manifest))
        prot_manifest["files"].append(
            {
                "path": "config/config.json",
                "root": "dist",
                "bytes": 2,
                "sha256": _sha256_bytes(b"{}"),
            }
        )
        with pytest.raises(updater.UpdateError, match="受保护目录"):
            updater.verify_staged_payload(staged, expected_version="1.9.1", manifest_payload=prot_manifest)

        # Duplicate path in manifest["files"] must be refused.
        dup_manifest = json.loads(json.dumps(manifest))
        dup_manifest["files"].append(dict(dup_manifest["files"][0]))
        with pytest.raises(updater.UpdateError, match="重复路径"):
            updater.verify_staged_payload(staged, expected_version="1.9.1", manifest_payload=dup_manifest)

        # Windows case-folding collision in manifest["files"] must be refused.
        cf_manifest = json.loads(json.dumps(manifest))
        readme_entry = next(e for e in cf_manifest["files"] if e["path"] == "README.txt")
        colliding = dict(readme_entry)
        colliding["path"] = "readme.txt"
        cf_manifest["files"].append(colliding)
        with pytest.raises(updater.UpdateError, match="大小写冲突"):
            updater.verify_staged_payload(staged, expected_version="1.9.1", manifest_payload=cf_manifest)

    def test_http_client_blocks_unsafe_redirects_and_exempts_loopback_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 1. Redirect hook checks HTTPS->HTTP downgrade & untrusted domain redirects.
        def _run_redirect(origin: str, location: str) -> httpx.Response:
            def handler(request: httpx.Request) -> httpx.Response:
                if str(request.url) == origin:
                    return httpx.Response(302, headers={"Location": location})
                return httpx.Response(200, text="ok")

            with httpx.Client(
                transport=httpx.MockTransport(handler),
                follow_redirects=True,
                event_hooks={"response": [updater._validate_redirect_response]},
            ) as client:
                return client.get(origin)

        # HTTPS -> HTTP downgrade (even to loopback) must be refused.
        with pytest.raises(updater.UpdateError, match=r"协议降级|不受信任"):
            _run_redirect("https://github.com/a", "http://github.com/b")
        with pytest.raises(updater.UpdateError, match=r"协议降级|不受信任"):
            _run_redirect("https://github.com/a", "http://127.0.0.1/b")

        # Cross-domain redirect to untrusted domain must be refused.
        with pytest.raises(updater.UpdateError, match="不受信任"):
            _run_redirect("https://github.com/a", "https://evil.attacker.invalid/b")

        # GitHub release asset CDN redirect and same-host redirect are allowed.
        ok_gh = _run_redirect(
            "https://github.com/jaykwok/jav-sub/releases/download/v1.0.0/a.zip",
            "https://release-assets.githubusercontent.com/blob/a.zip",
        )
        assert ok_gh.status_code == 200
        ok_same = _run_redirect("https://example.invalid/a", "https://example.invalid/b")
        assert ok_same.status_code == 200

        # 2. _http_client exempts loopback from configured proxy.
        monkeypatch.setenv("PROXY_PROTOCOL", "http")
        monkeypatch.setenv("PROXY_HOST", "127.0.0.1")
        monkeypatch.setenv("PROXY_PORT", "7890")
        with updater._http_client(5.0) as client:
            for url in ("http://127.0.0.1:8080/a", "http://localhost:8080/a", "http://[::1]:8080/a"):
                t = client._transport_for_url(httpx.URL(url))
                assert getattr(t._pool, "_proxy_url", None) is None
            remote = client._transport_for_url(httpx.URL("https://api.github.com/repos"))
            assert getattr(remote._pool, "_proxy_url", None) is not None


class TestUpdateManagerAndPolicy:
    def test_auto_check_toggle_and_startup_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = _make_installed_dir(tmp_path / "pkg")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("JAV_SUB_UPDATE_API_URL", raising=False)

        monkeypatch.delenv("JAV_SUB_DISABLE_UPDATE_CHECK", raising=False)
        assert updater.auto_update_check_enabled() is True
        assert updater.should_auto_check_on_startup(install) is True

        monkeypatch.setenv("JAV_SUB_DISABLE_UPDATE_CHECK", "1")
        assert updater.auto_update_check_enabled() is False
        assert updater.should_auto_check_on_startup(install) is False

        # Git checkout skips automatic startup check unless explicit API URL is set.
        monkeypatch.setenv("JAV_SUB_DISABLE_UPDATE_CHECK", "0")
        (install / ".git").mkdir()
        assert updater.should_auto_check_on_startup(install) is False
        allowed, reason = updater.can_self_update(install)
        assert allowed is False
        assert "git" in reason

    def test_manager_end_to_end_stage_and_trigger_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = _make_installed_dir(tmp_path / "pkg", version="1.9.0")
        zip_bytes, manifest = _build_release_zip(version="1.9.1")
        zip_sha = _sha256_bytes(zip_bytes)
        api_url = "https://example.invalid/repos/jaykwok/jav-sub/releases/latest"
        payload = _release_payload(tag="v1.9.1", zip_bytes=zip_bytes, zip_sha=zip_sha)

        monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", api_url)
        monkeypatch.setattr(updater.os, "name", "nt")
        monkeypatch.setattr(
            updater,
            "_http_client",
            _mock_http_client(
                {
                    api_url: (200, payload, None),
                    "https://example.invalid/releases/jav-sub-windows-x64.zip.sha256": (
                        200,
                        f"{zip_sha}  jav-sub-windows-x64.zip\n",
                        None,
                    ),
                    "https://example.invalid/releases/release-manifest.json": (200, manifest, None),
                    "https://example.invalid/releases/jav-sub-windows-x64.zip": (200, zip_bytes, None),
                }
            ),
        )

        restart_calls: list[str] = []
        updater.register_update_exit_callback(lambda: restart_calls.append("exit"))
        try:
            mgr = updater.UpdateManager(root=install)
            snap = mgr.check_now()
            assert snap["state"] == "available"
            assert snap["update_available"] is True
            assert snap["latest_version"] == "1.9.1"

            mgr.start_update(active_jobs_count=0)
            assert mgr._worker is not None
            mgr._worker.join(timeout=10.0)
            final = mgr.snapshot()
            assert final["state"] == "restarting"
            assert restart_calls == ["exit"]

            pending_path = install / "tmp" / "updates" / updater.PENDING_UPDATE_FILE_NAME
            assert pending_path.is_file()
            descriptor = json.loads(pending_path.read_text(encoding="utf-8"))
            assert descriptor["schema"] == updater.PENDING_UPDATE_SCHEMA
            assert descriptor["target_version"] == "1.9.1"
            assert descriptor["zip_sha256"] == zip_sha
            assert descriptor["manifest_verified"] is True
            staged_dir = install / descriptor["staged_dir"]
            assert (staged_dir / "jav-sub.exe").read_bytes() == b"fake-exe-1.9.1"
        finally:
            updater.register_update_exit_callback(None)

    def test_manager_sha256_mismatch_rejects_and_cleans_temp_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = _make_installed_dir(tmp_path / "pkg", version="1.9.0")
        zip_bytes, manifest = _build_release_zip(version="1.9.1")
        wrong_sha = "f" * 64
        api_url = "https://example.invalid/repos/jaykwok/jav-sub/releases/latest"
        payload = _release_payload(tag="v1.9.1", zip_bytes=zip_bytes, zip_sha=wrong_sha)

        monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", api_url)
        monkeypatch.setattr(updater.os, "name", "nt")
        monkeypatch.setattr(
            updater,
            "_http_client",
            _mock_http_client(
                {
                    api_url: (200, payload, None),
                    "https://example.invalid/releases/jav-sub-windows-x64.zip.sha256": (
                        200,
                        f"{wrong_sha}  jav-sub-windows-x64.zip\n",
                        None,
                    ),
                    "https://example.invalid/releases/release-manifest.json": (200, manifest, None),
                    "https://example.invalid/releases/jav-sub-windows-x64.zip": (200, zip_bytes, None),
                }
            ),
        )

        mgr = updater.UpdateManager(root=install)
        mgr.check_now()
        mgr.start_update(active_jobs_count=0)
        assert mgr._worker is not None
        mgr._worker.join(timeout=10.0)
        snap = mgr.snapshot()
        assert snap["state"] == "error"
        assert "SHA-256" in snap["error"]
        assert not (install / "tmp" / "updates" / updater.PENDING_UPDATE_FILE_NAME).exists()
        updates_dir = install / "tmp" / "updates"
        assert not updates_dir.exists() or list(updates_dir.iterdir()) == []

    def test_manager_cancel_mid_download_cleans_temp_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = _make_installed_dir(tmp_path / "pkg", version="1.9.0")
        zip_bytes, manifest = _build_release_zip(version="1.9.1")
        zip_sha = _sha256_bytes(zip_bytes)
        api_url = "https://example.invalid/repos/jaykwok/jav-sub/releases/latest"
        payload = _release_payload(tag="v1.9.1", zip_bytes=zip_bytes, zip_sha=zip_sha)

        started_stream = threading.Event()
        release_stream = threading.Event()

        class SlowStream(httpx.SyncByteStream):
            def __iter__(self):
                yield zip_bytes[:16]
                started_stream.set()
                release_stream.wait(timeout=5.0)
                yield zip_bytes[16:]

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url == api_url:
                return httpx.Response(200, json=payload)
            if url.endswith(".sha256"):
                return httpx.Response(200, text=f"{zip_sha}  jav-sub-windows-x64.zip\n")
            if url.endswith("release-manifest.json"):
                return httpx.Response(200, json=manifest)
            if url.endswith(".zip"):
                return httpx.Response(
                    200,
                    headers={"Content-Length": str(len(zip_bytes))},
                    stream=SlowStream(),
                )
            return httpx.Response(404)

        monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", api_url)
        monkeypatch.setattr(updater.os, "name", "nt")
        monkeypatch.setattr(
            updater,
            "_http_client",
            lambda _t: httpx.Client(transport=httpx.MockTransport(handler)),
        )

        mgr = updater.UpdateManager(root=install)
        mgr.check_now()
        mgr.start_update(active_jobs_count=0)
        assert started_stream.wait(timeout=5.0)
        # Duplicate check_now() / start_update() while downloading must NOT deadlock.
        dup_check = mgr.check_now()
        assert dup_check["state"] == "downloading"
        dup_start = mgr.start_update(active_jobs_count=0)
        assert dup_start["state"] == "downloading"

        mgr.cancel_update()
        release_stream.set()
        assert mgr._worker is not None
        mgr._worker.join(timeout=10.0)
        snap = mgr.snapshot()
        assert snap["state"] == "available"
        assert snap["downloaded_bytes"] == 0
        updates_dir = install / "tmp" / "updates"
        assert not updates_dir.exists() or list(updates_dir.iterdir()) == []

    def test_manager_cancel_during_verifying_aborts_before_descriptor_or_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = _make_installed_dir(tmp_path / "pkg", version="1.9.0")
        zip_bytes, manifest = _build_release_zip(version="1.9.1")
        zip_sha = _sha256_bytes(zip_bytes)
        api_url = "https://example.invalid/repos/jaykwok/jav-sub/releases/latest"
        payload = _release_payload(tag="v1.9.1", zip_bytes=zip_bytes, zip_sha=zip_sha)

        monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", api_url)
        monkeypatch.setattr(updater.os, "name", "nt")
        monkeypatch.setattr(
            updater,
            "_http_client",
            _mock_http_client(
                {
                    api_url: (200, payload, None),
                    "https://example.invalid/releases/jav-sub-windows-x64.zip.sha256": (
                        200,
                        f"{zip_sha}  jav-sub-windows-x64.zip\n",
                        None,
                    ),
                    "https://example.invalid/releases/release-manifest.json": (200, manifest, None),
                    "https://example.invalid/releases/jav-sub-windows-x64.zip": (200, zip_bytes, None),
                }
            ),
        )

        mgr = updater.UpdateManager(root=install)
        real_sha256_file = updater._sha256_file
        cancel_triggered = False

        def cancelling_sha256_file(path: Path) -> str:
            nonlocal cancel_triggered
            if not cancel_triggered:
                cancel_triggered = True
                # Duplicate check/start during verifying must not deadlock either.
                assert mgr.check_now()["state"] == "verifying"
                assert mgr.start_update(active_jobs_count=0)["state"] == "verifying"
                mgr.cancel_update()
            return real_sha256_file(path)

        monkeypatch.setattr(updater, "_sha256_file", cancelling_sha256_file)

        restart_calls: list[str] = []
        updater.register_update_exit_callback(lambda: restart_calls.append("exit"))
        try:
            mgr.check_now()
            mgr.start_update(active_jobs_count=0)
            assert mgr._worker is not None
            mgr._worker.join(timeout=10.0)

            assert cancel_triggered is True
            assert restart_calls == []
            assert not (install / "tmp" / "updates" / updater.PENDING_UPDATE_FILE_NAME).exists()
            updates_dir = install / "tmp" / "updates"
            assert not updates_dir.exists() or list(updates_dir.iterdir()) == []
            assert mgr.snapshot()["state"] == "available"
            assert mgr.is_update_in_progress() is False
        finally:
            updater.register_update_exit_callback(None)

    def test_manager_cancel_closes_active_http_client_and_restart_failure_cleans_pending_descriptor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = _make_installed_dir(tmp_path / "pkg", version="1.9.0")
        zip_bytes, manifest = _build_release_zip(version="1.9.1")
        zip_sha = _sha256_bytes(zip_bytes)
        api_url = "https://example.invalid/repos/jaykwok/jav-sub/releases/latest"
        payload = _release_payload(tag="v1.9.1", zip_bytes=zip_bytes, zip_sha=zip_sha)

        # 1. Active HTTP client is closed on cancel_update(), unblocking a stalled read.
        in_read = threading.Event()
        closed_evt = threading.Event()

        class _TrackedClient(httpx.Client):
            def close(self) -> None:
                closed_evt.set()
                super().close()

        class _BlockingStream(httpx.SyncByteStream):
            def __iter__(self):
                in_read.set()
                closed_evt.wait(timeout=5.0)
                raise httpx.ReadError("client closed on cancel")

        def stall_handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url == api_url:
                return httpx.Response(200, json=payload)
            if url.endswith(".sha256"):
                return httpx.Response(200, text=f"{zip_sha}  jav-sub-windows-x64.zip\n")
            if url.endswith("release-manifest.json"):
                return httpx.Response(200, json=manifest)
            if url.endswith(".zip"):
                return httpx.Response(200, stream=_BlockingStream())
            return httpx.Response(404)

        monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", api_url)
        monkeypatch.setattr(updater.os, "name", "nt")
        monkeypatch.setattr(
            updater,
            "_http_client",
            lambda _t: _TrackedClient(transport=httpx.MockTransport(stall_handler)),
        )

        mgr = updater.UpdateManager(root=install)
        mgr.check_now()
        mgr.start_update(active_jobs_count=0)
        assert in_read.wait(timeout=5.0)
        mgr.cancel_update()
        assert closed_evt.is_set()
        assert mgr._worker is not None
        mgr._worker.join(timeout=5.0)
        assert mgr.snapshot()["state"] == "available"

        # 2. If request_update_restart() raises after staging, pending-update.json
        #    MUST be cleaned up alongside staged_dir (no dangling descriptor).
        monkeypatch.setattr(
            updater,
            "_http_client",
            _mock_http_client(
                {
                    api_url: (200, payload, None),
                    "https://example.invalid/releases/jav-sub-windows-x64.zip.sha256": (
                        200,
                        f"{zip_sha}  jav-sub-windows-x64.zip\n",
                        None,
                    ),
                    "https://example.invalid/releases/release-manifest.json": (200, manifest, None),
                    "https://example.invalid/releases/jav-sub-windows-x64.zip": (200, zip_bytes, None),
                }
            ),
        )

        def _failing_exit() -> None:
            raise RuntimeError("exit hook boom")

        updater.register_update_exit_callback(_failing_exit)
        try:
            mgr.start_update(active_jobs_count=0)
            assert mgr._worker is not None
            mgr._worker.join(timeout=10.0)
            snap = mgr.snapshot()
            assert snap["state"] == "error"
            assert "exit hook boom" in snap["error"]
            pending_path = install / "tmp" / "updates" / updater.PENDING_UPDATE_FILE_NAME
            assert not pending_path.exists()
            updates_dir = install / "tmp" / "updates"
            assert not updates_dir.exists() or list(updates_dir.iterdir()) == []
        finally:
            updater.register_update_exit_callback(None)


class TestUpdateWebRoutes:
    def test_routes_check_guard_conflicts_and_auto_check_persistence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = _make_installed_dir(tmp_path / "pkg", version="1.9.0")
        zip_bytes, _ = _build_release_zip(version="1.9.1")
        api_url = "https://example.invalid/repos/jaykwok/jav-sub/releases/latest"
        payload = _release_payload(tag="v1.9.1", zip_bytes=zip_bytes)

        monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", api_url)
        monkeypatch.delenv("JAV_SUB_DISABLE_UPDATE_CHECK", raising=False)
        monkeypatch.setattr(updater.os, "name", "nt")
        monkeypatch.setattr(config_routes, "PROJECT_ROOT", install)
        monkeypatch.setattr(pipeline_manager, "PROJECT_ROOT", install)
        monkeypatch.setattr(pipeline_manager, "_jobs_path", install / "cache" / "web" / "jobs.json")
        monkeypatch.setattr(pipeline_manager, "_jobs", {})
        monkeypatch.setattr(pipeline_manager, "_cancel_events", {})
        mgr = updater.UpdateManager(root=install)
        monkeypatch.setattr(updater, "get_update_manager", lambda: mgr)
        monkeypatch.setattr(
            updater,
            "_http_client",
            _mock_http_client({api_url: (200, payload, None)}),
        )

        fake_running_job = JobState(
            id="job-1",
            status="asr",
            spec=JobSpec(video_paths=["sample-a.mp4"]),
            created_at="2026-11-26T12:00:00Z",
        )

        async def _run() -> None:
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r0 = await client.get("/api/update/status")
                assert r0.status_code == 200
                assert r0.json()["current_version"] == "1.9.0"
                assert r0.json()["auto_check_enabled"] is True

                r_chk = await client.post("/api/update/check")
                assert r_chk.status_code == 200
                assert r_chk.json()["state"] == "available"
                assert r_chk.json()["latest_version"] == "1.9.1"

                # Active job must block self-update with 409.
                pipeline_manager._jobs[fake_running_job.id] = fake_running_job
                r_busy = await client.post("/api/update/start")
                assert r_busy.status_code == 409
                assert "任务" in r_busy.json()["detail"]

                # Another open launcher window must block self-update with 409.
                pipeline_manager._jobs.clear()
                monkeypatch.setattr(updater, "other_launcher_instances_running", lambda _r: True)
                r_win = await client.post("/api/update/start")
                assert r_win.status_code == 409
                assert "窗口" in r_win.json()["detail"]
                monkeypatch.setattr(updater, "other_launcher_instances_running", lambda _r: False)

                # Toggle auto-check off -> persisted in config/config.json and reflected in /api/settings.
                r_off = await client.post("/api/update/auto-check", json={"enabled": False})
                assert r_off.status_code == 200
                assert r_off.json()["auto_check_enabled"] is False
                cfg = json.loads((install / "config" / "config.json").read_text(encoding="utf-8"))
                assert cfg["JAV_SUB_DISABLE_UPDATE_CHECK"] == "1"

                r_settings = await client.get("/api/settings")
                assert r_settings.status_code == 200
                assert r_settings.json()["disable_update_check"] is True

                # Re-enable via /api/settings.
                r_set_on = await client.post("/api/settings", json={"disable_update_check": False})
                assert r_set_on.status_code == 200
                r_status_on = await client.get("/api/update/status")
                assert r_status_on.json()["auto_check_enabled"] is True

                # Corrupt/truncated config/config.json must refuse overwrite (400).
                (install / "config" / "config.json").write_text("", encoding="utf-8")
                r_bad_cfg = await client.post("/api/update/auto-check", json={"enabled": False})
                assert r_bad_cfg.status_code == 400
                assert (install / "config" / "config.json").read_text(encoding="utf-8") == ""

        asyncio.run(_run())

    def test_active_update_blocks_job_creation_and_retry_until_cancelled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = _make_installed_dir(tmp_path / "pkg", version="1.9.0")
        video_a = tmp_path / "sample-a.mp4"
        video_b = tmp_path / "sample-b.mp4"
        video_a.write_bytes(b"video-a")
        video_b.write_bytes(b"video-b")

        zip_bytes, manifest = _build_release_zip(version="1.9.1")
        zip_sha = _sha256_bytes(zip_bytes)
        api_url = "https://example.invalid/repos/jaykwok/jav-sub/releases/latest"
        payload = _release_payload(tag="v1.9.1", zip_bytes=zip_bytes, zip_sha=zip_sha)

        started_stream = threading.Event()
        release_stream = threading.Event()

        class SlowStream(httpx.SyncByteStream):
            def __iter__(self):
                yield zip_bytes[:16]
                started_stream.set()
                release_stream.wait(timeout=5.0)
                yield zip_bytes[16:]

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url == api_url:
                return httpx.Response(200, json=payload)
            if url.endswith(".sha256"):
                return httpx.Response(200, text=f"{zip_sha}  jav-sub-windows-x64.zip\n")
            if url.endswith("release-manifest.json"):
                return httpx.Response(200, json=manifest)
            if url.endswith(".zip"):
                return httpx.Response(
                    200,
                    headers={"Content-Length": str(len(zip_bytes))},
                    stream=SlowStream(),
                )
            return httpx.Response(404)

        monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", api_url)
        monkeypatch.setattr(updater.os, "name", "nt")
        monkeypatch.setattr(config_routes, "PROJECT_ROOT", install)
        monkeypatch.setattr(pipeline_manager, "PROJECT_ROOT", install)
        monkeypatch.setattr(pipeline_manager, "_jobs_path", install / "cache" / "web" / "jobs.json")
        monkeypatch.setattr(pipeline_manager, "_jobs", {})
        monkeypatch.setattr(pipeline_manager, "_cancel_events", {})
        monkeypatch.setattr(pipeline_manager, "gpu_queue", asyncio.Queue())
        monkeypatch.setattr(pipeline_manager, "trans_queue", asyncio.Queue())

        failed_job = JobState(
            id="failed-job-1",
            status="failed",
            spec=JobSpec(video_paths=[str(video_b)], skip_translation=True),
            created_at="2026-11-26T12:00:00Z",
            run_id="run-old",
        )
        pipeline_manager._jobs[failed_job.id] = failed_job

        mgr = updater.UpdateManager(root=install)
        monkeypatch.setattr(updater, "get_update_manager", lambda: mgr)
        monkeypatch.setattr(
            updater,
            "_http_client",
            lambda _t: httpx.Client(transport=httpx.MockTransport(handler)),
        )

        async def _run() -> None:
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r_chk = await client.post("/api/update/check")
                assert r_chk.status_code == 200
                r_start = await client.post("/api/update/start")
                assert r_start.status_code == 200
                assert r_start.json()["state"] == "downloading"
                assert started_stream.wait(timeout=5.0)

                # While updating, creating or retrying jobs must be rejected with 409.
                r_job = await client.post(
                    "/api/jobs",
                    json={"video_paths": [str(video_a)], "skip_translation": True},
                )
                assert r_job.status_code == 409
                assert "在线升级" in r_job.json()["detail"]

                r_retry = await client.post(f"/api/jobs/{failed_job.id}/retry")
                assert r_retry.status_code == 409
                assert "在线升级" in r_retry.json()["detail"]
                assert pipeline_manager._jobs[failed_job.id].status == "failed"

                # Duplicate check/start over HTTP must not deadlock.
                r_dup_chk = await client.post("/api/update/check")
                assert r_dup_chk.status_code == 200
                assert r_dup_chk.json()["state"] == "downloading"
                r_dup_start = await client.post("/api/update/start")
                assert r_dup_start.status_code == 200
                assert r_dup_start.json()["state"] == "downloading"

                # Cancel update and wait for worker exit -> job creation & retry resume.
                r_cancel = await client.post("/api/update/cancel")
                assert r_cancel.status_code == 200
                release_stream.set()
                assert mgr._worker is not None
                await asyncio.to_thread(mgr._worker.join, 10.0)
                assert mgr.snapshot()["state"] == "available"

                r_job_ok = await client.post(
                    "/api/jobs",
                    json={"video_paths": [str(video_a)], "skip_translation": True},
                )
                assert r_job_ok.status_code == 201
                assert len(r_job_ok.json()["ids"]) == 1

                r_retry_ok = await client.post(f"/api/jobs/{failed_job.id}/retry")
                assert r_retry_ok.status_code == 200
                assert r_retry_ok.json()["status"] == "queued"

        asyncio.run(_run())

    def test_in_flight_export_retry_blocks_update_start_until_recovery_finishes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = _make_installed_dir(tmp_path / "pkg", version="1.9.0")
        video_a = tmp_path / "sample-a.mp4"
        video_a.write_bytes(b"video-a")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        target_srt = (out_dir / "sample-a.ja.srt").resolve()

        zip_bytes, manifest = _build_release_zip(version="1.9.1")
        zip_sha = _sha256_bytes(zip_bytes)
        api_url = "https://example.invalid/repos/jaykwok/jav-sub/releases/latest"
        payload = _release_payload(tag="v1.9.1", zip_bytes=zip_bytes, zip_sha=zip_sha)

        # Prepare a real committed generation whose SRT export is paused during retry.
        lease = pipeline_manager.artifact_store.claim_output(
            target_srt,
            project_root=install,
            job_id="export-job-1",
            run_id="run-export-1",
        )
        srt_payload = "1\n00:00:00,000 --> 00:00:01,000\nテスト\n"
        run_outputs = pipeline_manager.artifact_store.RunOutputs(lease=lease)
        run_outputs.stage(
            target_srt,
            name="srt",
            write=lambda path: path.write_text(srt_payload, encoding="utf-8"),
        )
        run_outputs.publish(subtitle_cue_count=1)

        export_job = JobState(
            id="export-job-1",
            status="export_failed",
            spec=JobSpec(
                video_paths=[str(video_a)],
                output_dir=str(out_dir),
                skip_translation=True,
            ),
            created_at="2026-11-26T12:00:00Z",
            run_id="run-export-1",
            output_target=str(target_srt),
            output_generation=lease.sequence,
        )

        entered_resume = threading.Event()
        release_resume = threading.Event()
        real_resume_output = pipeline_manager.artifact_store.resume_output

        def paused_resume_output(job_lease):
            entered_resume.set()
            assert release_resume.wait(timeout=5.0)
            return real_resume_output(job_lease)

        monkeypatch.setattr(pipeline_manager.artifact_store, "resume_output", paused_resume_output)
        monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", api_url)
        monkeypatch.setattr(updater.os, "name", "nt")
        monkeypatch.setattr(config_routes, "PROJECT_ROOT", install)
        monkeypatch.setattr(pipeline_manager, "PROJECT_ROOT", install)
        monkeypatch.setattr(pipeline_manager, "_jobs_path", install / "cache" / "web" / "jobs.json")
        monkeypatch.setattr(pipeline_manager, "_jobs", {export_job.id: export_job})
        monkeypatch.setattr(pipeline_manager, "_cancel_events", {})
        monkeypatch.setattr(pipeline_manager, "_active_retry_recoveries", 0)

        download_started = threading.Event()
        release_download = threading.Event()

        class SlowStream(httpx.SyncByteStream):
            def __iter__(self):
                yield zip_bytes[:16]
                download_started.set()
                release_download.wait(timeout=5.0)
                yield zip_bytes[16:]

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url == api_url:
                return httpx.Response(200, json=payload)
            if url.endswith(".sha256"):
                return httpx.Response(200, text=f"{zip_sha}  jav-sub-windows-x64.zip\n")
            if url.endswith("release-manifest.json"):
                return httpx.Response(200, json=manifest)
            if url.endswith(".zip"):
                return httpx.Response(
                    200,
                    headers={"Content-Length": str(len(zip_bytes))},
                    stream=SlowStream(),
                )
            return httpx.Response(404)

        mgr = updater.UpdateManager(root=install)
        monkeypatch.setattr(updater, "get_update_manager", lambda: mgr)
        monkeypatch.setattr(
            updater,
            "_http_client",
            lambda _t: httpx.Client(transport=httpx.MockTransport(handler)),
        )

        async def _run() -> None:
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r_chk = await client.post("/api/update/check")
                assert r_chk.status_code == 200
                assert r_chk.json()["state"] == "available"

                # Start export retry FIRST and pause inside real resume_output().
                retry_task = asyncio.create_task(client.post(f"/api/jobs/{export_job.id}/retry"))
                assert await asyncio.to_thread(entered_resume.wait, 5.0)

                # While resume_output() is in-flight, starting an update MUST be rejected (409).
                r_blocked = await client.post("/api/update/start")
                assert r_blocked.status_code == 409
                assert "任务" in r_blocked.json()["detail"]
                assert mgr.snapshot()["state"] == "available"

                # Release export retry -> completes as done, counter returns to 0.
                release_resume.set()
                r_retry = await retry_task
                assert r_retry.status_code == 200
                assert r_retry.json()["status"] == "done"
                assert target_srt.is_file()
                assert pipeline_manager._active_retry_recoveries == 0

                # Now update can start normally.
                r_start = await client.post("/api/update/start")
                assert r_start.status_code == 200
                assert r_start.json()["state"] == "downloading"
                assert await asyncio.to_thread(download_started.wait, 5.0)
                await client.post("/api/update/cancel")
                release_download.set()
                assert mgr._worker is not None
                await asyncio.to_thread(mgr._worker.join, 10.0)

        asyncio.run(_run())
