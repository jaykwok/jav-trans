"""Unit tests for managed CUDA llama-server discovery, download, and verification."""

from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from pathlib import Path

import pytest

from llm.backends import llamacpp_runtime, llamacpp_server
from llm.backends.llamacpp_runtime import (
    FALLBACK_CUDA13_RELEASE,
    _safe_extract_zip,
    detect_cuda_major,
    ensure_managed_llama_server,
    resolve_latest_release_asset,
    select_release_asset,
)
from llm.backends.llamacpp_server import LlamaCppServerBackend, resolve_server_executable
from llm.errors import TranslationCancelledError


def _make_zip(path: Path, files: dict[str, bytes]) -> tuple[Path, str]:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_detect_cuda_major_prefers_torch_lib_dlls(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "cublas64_13.dll").write_bytes(b"")
    (lib_dir / "cudart64_13.dll").write_bytes(b"")
    assert detect_cuda_major(torch_lib_dir=lib_dir) == 13

    lib12 = tmp_path / "lib12"
    lib12.mkdir()
    (lib12 / "cublas64_12.dll").write_bytes(b"")
    assert detect_cuda_major(torch_lib_dir=lib12) == 12


def test_detect_cuda_major_falls_back_to_torch_metadata(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(
        llamacpp_runtime.importlib.metadata,
        "version",
        lambda name: "2.14.1+cu130" if name == "torch" else "0.0",
    )
    assert detect_cuda_major(torch_lib_dir=empty_dir) == 13


def test_select_release_asset_matches_cuda_major_and_keeps_prereleases():
    releases = [
        {
            "tag_name": "b99999",
            "draft": True,
            "prerelease": True,
            "assets": [
                {
                    "name": "llama-b99999-bin-win-cuda-13.9-x64.zip",
                    "browser_download_url": "https://example.test/draft.zip",
                    "size": 100,
                }
            ],
        },
        {
            "tag_name": "v1.0.0",
            "draft": False,
            "prerelease": False,
            "assets": [
                {
                    "name": "source-only.tar.gz",
                    "browser_download_url": "https://example.test/src.tar.gz",
                    "size": 100,
                }
            ],
        },
        {
            "tag_name": "b12738",
            "draft": False,
            "prerelease": True,
            "assets": [
                {
                    "name": "cudart-llama-bin-win-cuda-13.5-x64.zip",
                    "browser_download_url": "https://example.test/cudart-13.5.zip",
                    "digest": "sha256:" + "a" * 64,
                    "size": 400,
                },
                {
                    "name": "llama-b12738-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example.test/cuda-12.4.zip",
                    "size": 300,
                },
                {
                    "name": "llama-b12738-bin-win-cuda-13.5-arm64.zip",
                    "browser_download_url": "https://example.test/cuda-13.5-arm64.zip",
                    "size": 200,
                },
                {
                    "name": "llama-b12738-bin-win-cuda-13.3-x64.zip",
                    "browser_download_url": "https://example.test/cuda-13.3.zip",
                    "size": 210,
                },
                {
                    "name": "llama-b12738-bin-win-cuda-13.5-x64.zip",
                    "browser_download_url": "https://example.test/cuda-13.5.zip",
                    "digest": "sha256:" + "b" * 64,
                    "size": 220,
                },
            ],
        },
    ]

    picked = select_release_asset(releases, cuda_major=13)
    assert picked is not None
    assert picked.tag == "b12738"
    assert picked.asset_name == "llama-b12738-bin-win-cuda-13.5-x64.zip"
    assert picked.cuda_major == 13
    assert picked.cuda_minor == 5
    assert picked.sha256 == "b" * 64
    assert picked.cudart_url == "https://example.test/cudart-13.5.zip"
    assert picked.cudart_sha256 == "a" * 64


def test_resolve_latest_release_asset_falls_back_on_github_rate_limit():
    def fail_rate_limit():
        raise RuntimeError("403 rate limit exceeded")

    asset = resolve_latest_release_asset(cuda_major=13, fetch_releases=fail_rate_limit)
    assert asset == FALLBACK_CUDA13_RELEASE
    assert asset.tag == "b12738"


def test_safe_extract_zip_blocks_zip_slip(tmp_path):
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../escape.dll", b"bad")
    with pytest.raises(RuntimeError, match="非法路径"):
        _safe_extract_zip(evil_zip, tmp_path / "out")


def test_ensure_managed_llama_server_installs_verifies_and_writes_meta(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    monkeypatch.setattr(llamacpp_runtime, "MODELS_ROOT", models_root)

    src_zip, digest = _make_zip(
        tmp_path / "llama-b12800.zip",
        {
            "nested/llama-server.exe": b"fake-exe",
            "nested/ggml-cuda.dll": b"fake-cuda",
        },
    )

    def fake_releases():
        return [
            {
                "tag_name": "b12800",
                "draft": False,
                "prerelease": True,
                "assets": [
                    {
                        "name": "llama-b12800-bin-win-cuda-13.5-x64.zip",
                        "browser_download_url": "https://example.test/b12800.zip",
                        "digest": f"sha256:{digest}",
                        "size": src_zip.stat().st_size,
                    }
                ],
            }
        ]

    def fake_download(url, dest, *, expected_sha256="", **_kwargs):
        assert url == "https://example.test/b12800.zip"
        assert expected_sha256 == digest
        dest.write_bytes(src_zip.read_bytes())

    exe_path = ensure_managed_llama_server(
        cuda_major=13,
        fetch_releases=fake_releases,
        download_fn=fake_download,
        probe_devices=lambda exe, env: ["CUDA0: RTX 4060 Ti (8187 MiB)"],
        env_factory=lambda: {},
    )
    installed = Path(exe_path)
    assert installed == models_root / "runtime" / "llama-cpp" / "llama-server.exe"
    assert installed.is_file()
    assert (installed.parent / "ggml-cuda.dll").is_file()

    meta = json.loads((installed.parent / "install-meta.json").read_text(encoding="utf-8"))
    assert meta["tag"] == "b12800"
    assert meta["cudart_installed"] is False
    assert meta["devices"] == ["CUDA0: RTX 4060 Ti (8187 MiB)"]


def test_ensure_managed_llama_server_downloads_cudart_if_first_probe_empty(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    monkeypatch.setattr(llamacpp_runtime, "MODELS_ROOT", models_root)

    bin_zip, bin_sha = _make_zip(tmp_path / "bin.zip", {"llama-server.exe": b"exe"})
    cudart_zip, cudart_sha = _make_zip(tmp_path / "cudart.zip", {"cublas64_13.dll": b"dll"})
    urls_seen: list[str] = []

    def fake_releases():
        return [
            {
                "tag_name": "b12801",
                "draft": False,
                "prerelease": True,
                "assets": [
                    {
                        "name": "llama-b12801-bin-win-cuda-13.6-x64.zip",
                        "browser_download_url": "https://example.test/bin.zip",
                        "digest": f"sha256:{bin_sha}",
                        "size": bin_zip.stat().st_size,
                    },
                    {
                        "name": "cudart-llama-bin-win-cuda-13.6-x64.zip",
                        "browser_download_url": "https://example.test/cudart.zip",
                        "digest": f"sha256:{cudart_sha}",
                        "size": cudart_zip.stat().st_size,
                    },
                ],
            }
        ]

    def fake_download(url, dest, **_kwargs):
        urls_seen.append(url)
        src = bin_zip if url.endswith("bin.zip") else cudart_zip
        dest.write_bytes(src.read_bytes())

    def fake_probe(exe, _env):
        # Sees CUDA only once companion cublas64_13.dll is extracted beside exe.
        if (Path(exe).parent / "cublas64_13.dll").is_file():
            return ["CUDA0: RTX 4060 Ti"]
        return []

    exe_path = ensure_managed_llama_server(
        cuda_major=13,
        fetch_releases=fake_releases,
        download_fn=fake_download,
        probe_devices=fake_probe,
        env_factory=lambda: {},
    )
    assert urls_seen == ["https://example.test/bin.zip", "https://example.test/cudart.zip"]
    meta = json.loads((Path(exe_path).parent / "install-meta.json").read_text(encoding="utf-8"))
    assert meta["cudart_installed"] is True


def test_ensure_managed_llama_server_falls_back_to_b12738_when_latest_fails_cuda(
    tmp_path, monkeypatch
):
    models_root = tmp_path / "models"
    monkeypatch.setattr(llamacpp_runtime, "MODELS_ROOT", models_root)

    latest_zip, latest_sha = _make_zip(tmp_path / "latest.zip", {"llama-server.exe": b"latest"})
    fallback_zip, _ = _make_zip(tmp_path / "fallback.zip", {"llama-server.exe": b"fallback"})
    urls_seen: list[str] = []

    def fake_releases():
        return [
            {
                "tag_name": "b19999",
                "draft": False,
                "prerelease": True,
                "assets": [
                    {
                        "name": "llama-b19999-bin-win-cuda-13.9-x64.zip",
                        "browser_download_url": "https://example.test/b19999.zip",
                        "digest": f"sha256:{latest_sha}",
                        "size": latest_zip.stat().st_size,
                    }
                ],
            }
        ]

    def fake_download(url, dest, **_kwargs):
        urls_seen.append(url)
        if "b19999" in url:
            dest.write_bytes(latest_zip.read_bytes())
        else:
            dest.write_bytes(fallback_zip.read_bytes())

    def fake_probe(exe, _env):
        # Latest fails CUDA detection; verified fallback succeeds.
        return ["CUDA0: RTX 4060 Ti"] if Path(exe).read_bytes() == b"fallback" else []

    exe_path = ensure_managed_llama_server(
        cuda_major=13,
        fetch_releases=fake_releases,
        download_fn=fake_download,
        probe_devices=fake_probe,
        env_factory=lambda: {},
    )
    assert any("b19999" in u for u in urls_seen)
    assert any(FALLBACK_CUDA13_RELEASE.tag in u for u in urls_seen)
    meta = json.loads((Path(exe_path).parent / "install-meta.json").read_text(encoding="utf-8"))
    assert meta["tag"] == FALLBACK_CUDA13_RELEASE.tag


def test_ensure_managed_llama_server_refuses_cpu_only_installation(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    monkeypatch.setattr(llamacpp_runtime, "MODELS_ROOT", models_root)

    fake_zip, _ = _make_zip(tmp_path / "only.zip", {"llama-server.exe": b"cpu"})

    with pytest.raises(RuntimeError, match="CUDA"):
        ensure_managed_llama_server(
            cuda_major=13,
            fetch_releases=lambda: [],
            download_fn=lambda url, dest, **_kw: dest.write_bytes(fake_zip.read_bytes()),
            probe_devices=lambda exe, env: [],
            env_factory=lambda: {},
        )
    assert not (models_root / "runtime" / "llama-cpp").exists()


def test_resolve_server_executable_priority_and_auto_download(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    monkeypatch.setattr(llamacpp_runtime, "MODELS_ROOT", models_root)
    monkeypatch.delenv("LLAMACPP_SERVER_PATH", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("LLAMACPP_AUTO_DOWNLOAD", raising=False)

    # 1. Auto-download runs only when download_enabled=True and PATH misses.
    monkeypatch.setattr(llamacpp_server.shutil, "which", lambda name: None)
    monkeypatch.setattr(llamacpp_runtime, "auto_download_allowed", lambda: True)
    downloaded = models_root / "runtime" / "llama-cpp" / "llama-server.exe"

    def fake_install(*, cancel_event=None):
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(b"managed")
        return str(downloaded)

    monkeypatch.setattr(llamacpp_runtime, "ensure_managed_llama_server", fake_install)
    assert resolve_server_executable(download_enabled=True) == str(downloaded)

    # 2. Managed directory beats PATH once installed.
    path_exe = tmp_path / "path" / "llama-server.exe"
    path_exe.parent.mkdir()
    path_exe.write_bytes(b"path")
    monkeypatch.setattr(llamacpp_server.shutil, "which", lambda name: str(path_exe))
    assert resolve_server_executable() == str(downloaded)

    # 3. Explicit LLAMACPP_SERVER_PATH still beats managed directory.
    explicit_exe = tmp_path / "explicit" / "llama-server.exe"
    explicit_exe.parent.mkdir()
    explicit_exe.write_bytes(b"explicit")
    monkeypatch.setenv("LLAMACPP_SERVER_PATH", str(explicit_exe))
    assert resolve_server_executable() == str(explicit_exe)


def test_backend_resolve_server_executable_is_cancellable(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    monkeypatch.setattr(llamacpp_runtime, "MODELS_ROOT", models_root)
    monkeypatch.delenv("LLAMACPP_SERVER_PATH", raising=False)
    monkeypatch.setattr(llamacpp_server.shutil, "which", lambda name: None)
    monkeypatch.setattr(llamacpp_runtime, "auto_download_allowed", lambda: True)

    started = threading.Event()
    release = threading.Event()

    def slow_install(*, cancel_event=None):
        started.set()
        release.wait(timeout=5.0)
        return "never-used"

    monkeypatch.setattr(llamacpp_runtime, "ensure_managed_llama_server", slow_install)
    cancel = threading.Event()

    def trigger_cancel():
        started.wait(timeout=2.0)
        cancel.set()

    threading.Thread(target=trigger_cancel, daemon=True).start()
    try:
        with pytest.raises(TranslationCancelledError):
            LlamaCppServerBackend()._resolve_server_executable(cancel)
    finally:
        release.set()


def test_start_managed_llama_server_prefetch_states_and_single_flight(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    monkeypatch.setattr(llamacpp_runtime, "MODELS_ROOT", models_root)
    monkeypatch.setattr(llamacpp_runtime, "_prefetch_thread", None)
    monkeypatch.delenv("LLAMACPP_SERVER_PATH", raising=False)

    # Disabled when auto-download is off (e.g. default test env).
    monkeypatch.setenv("LLAMACPP_AUTO_DOWNLOAD", "0")
    assert llamacpp_runtime.start_managed_llama_server_prefetch() == "disabled"

    # Explicit override skips prefetch.
    monkeypatch.setenv("LLAMACPP_AUTO_DOWNLOAD", "1")
    monkeypatch.setenv("LLAMACPP_SERVER_PATH", str(tmp_path / "custom.exe"))
    assert llamacpp_runtime.start_managed_llama_server_prefetch() == "explicit_override"
    monkeypatch.delenv("LLAMACPP_SERVER_PATH", raising=False)

    # Single-flight background prefetch starts once and reports in_progress while running.
    monkeypatch.setattr(llamacpp_runtime, "auto_download_allowed", lambda: True)
    started = threading.Event()
    finish = threading.Event()
    calls: list[int] = []

    def fake_install():
        calls.append(1)
        started.set()
        finish.wait(timeout=5.0)
        exe = models_root / "runtime" / "llama-cpp" / "llama-server.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_bytes(b"ok")
        return str(exe)

    monkeypatch.setattr(llamacpp_runtime, "ensure_managed_llama_server", fake_install)
    assert llamacpp_runtime.start_managed_llama_server_prefetch() == "started"
    assert started.wait(timeout=2.0)
    assert llamacpp_runtime.start_managed_llama_server_prefetch() == "in_progress"
    finish.set()
    if llamacpp_runtime._prefetch_thread is not None:
        llamacpp_runtime._prefetch_thread.join(timeout=2.0)
    assert calls == [1]
    assert llamacpp_runtime.start_managed_llama_server_prefetch() == "ready"


def test_download_archive_retries_transient_transport_error(tmp_path, monkeypatch):
    import httpx

    payload = b"managed-llama-zip-bytes"
    expected_sha = hashlib.sha256(payload).hexdigest()
    attempts = {"count": 0}
    sleeps: list[float] = []

    class _FakeResp:
        headers = {"content-length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size=0):
            del chunk_size
            yield payload

    class _FakeClient:
        def __init__(self, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def stream(self, method, url):
            del method, url
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ConnectError("[Errno 11001] getaddrinfo failed")
            return _FakeResp()

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    monkeypatch.setattr(
        llamacpp_runtime, "_sleep_interruptible", lambda delay, _ev=None: sleeps.append(delay)
    )

    dest = tmp_path / "llama.zip"
    llamacpp_runtime._download_archive(
        "https://example.test/llama.zip",
        dest,
        display_name="llama.zip",
        expected_sha256=expected_sha,
        expected_size=len(payload),
    )

    assert attempts["count"] == 2
    assert sleeps == [llamacpp_runtime._DOWNLOAD_RETRY_DELAYS_S[0]]
    assert dest.read_bytes() == payload


def test_download_archive_does_not_retry_http_404(tmp_path, monkeypatch):
    import httpx

    attempts = {"count": 0}

    class _Fake404Resp:
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def raise_for_status(self):
            req = httpx.Request("GET", "https://example.test/missing.zip")
            resp = httpx.Response(404, request=req)
            raise httpx.HTTPStatusError("404 Not Found", request=req, response=resp)

    class _FakeClient:
        def __init__(self, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def stream(self, method, url):
            del method, url
            attempts["count"] += 1
            return _Fake404Resp()

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    with pytest.raises(httpx.HTTPStatusError):
        llamacpp_runtime._download_archive(
            "https://example.test/missing.zip",
            tmp_path / "missing.zip",
            display_name="missing.zip",
            expected_sha256="a" * 64,
        )
    assert attempts["count"] == 1


def test_select_release_asset_skips_assets_without_valid_sha256():
    unsigned = [
        {
            "tag_name": "b19999",
            "draft": False,
            "prerelease": True,
            "assets": [
                {
                    "name": "llama-b19999-bin-win-cuda-13.5-x64.zip",
                    "browser_download_url": "https://example.test/unsigned.zip",
                    "size": 100,
                }
            ],
        }
    ]
    assert select_release_asset(unsigned, cuda_major=13) is None
    resolved = resolve_latest_release_asset(cuda_major=13, fetch_releases=lambda: unsigned)
    assert resolved == FALLBACK_CUDA13_RELEASE


def test_download_archive_refuses_missing_sha256_before_network(tmp_path, monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **_kw: pytest.fail("must refuse unsigned download before opening HTTP client"),
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        llamacpp_runtime._download_archive(
            "https://example.test/unsigned.zip",
            tmp_path / "unsigned.zip",
            display_name="unsigned.zip",
            expected_sha256="",
        )
