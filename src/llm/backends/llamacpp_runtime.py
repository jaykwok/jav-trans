"""Managed CUDA ``llama-server`` runtime installer.

Downloads and verifies an official Windows x64 CUDA ``llama.cpp`` release into
``./models/runtime/llama-cpp/`` when no explicit ``LLAMACPP_SERVER_PATH`` or
system ``llama-server`` is configured.

Design constraints:
1. Match by CUDA major version detected from ``.venv``'s PyTorch runtime DLLs
   (e.g. ``cublas64_13.dll`` -> ``cuda-13.*``) rather than exact minor version,
   because upstream publishes rolling ``cuda-13.x`` assets (currently 13.5) that
   reuse PyTorch's ``cu130`` shared libraries without needing the ~426 MB
   ``cudart-*`` archive.
2. Query ``/releases`` (filtering out drafts only, NOT prereleases — upstream
   marks rolling ``bXXXXX`` builds as prereleases while ``/releases/latest``
   points to a source-only tag) and keep a real-GPU-verified fallback build
   (``b12738``) for GitHub API rate limits or transient outages.
3. Verify SHA-256 when provided, guard against Zip Slip, and run
   ``probe_compute_devices`` on the staged binary before publishing into the
   managed directory — auto-fetching the sibling ``cudart-*`` archive or falling
   back to the verified build if needed, and failing loudly instead of silently
   installing a CPU-only runtime.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from llm.async_transport import run_cancellable_request
from llm.backends import llamacpp_env
from llm.errors import TranslationCancelledError
from utils import hf_progress
from utils.model_paths import MODELS_ROOT

logger = logging.getLogger(__name__)

_SERVER_EXE_NAMES = ("llama-server.exe", "llama-server")
_INSTALL_COND = threading.Condition(threading.Lock())
_INSTALL_IN_PROGRESS = False
_INSTALL_WAIT_POLL_S = 0.2
_PREFETCH_LOCK = threading.Lock()
_prefetch_thread: threading.Thread | None = None
_DOWNLOAD_CHUNK_BYTES = 512 * 1024
_DOWNLOAD_MAX_ATTEMPTS = 3
_DOWNLOAD_RETRY_DELAYS_S = (1.5, 3.0)
_RETRYABLE_HTTP_STATUS = frozenset({502, 503, 504})
_PROGRESS_INTERVAL_S = 0.5
_DEFAULT_CUDA_MAJOR = 13

LLAMACPP_RELEASES_API = (
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
)
_USER_AGENT = "jav-sub-llamacpp-runtime"


@dataclass(frozen=True)
class LlamaReleaseAsset:
    tag: str
    asset_name: str
    url: str
    sha256: str = ""
    size: int = 0
    cuda_major: int = _DEFAULT_CUDA_MAJOR
    cuda_minor: int = 0
    cudart_name: str = ""
    cudart_url: str = ""
    cudart_sha256: str = ""
    cudart_size: int = 0


# Verified 2026-11-26 on RTX 4060 Ti (8 GB) + torch 2.14.1+cu130 + Hy-MT3-7B-Q4_K_M.
FALLBACK_CUDA13_RELEASE = LlamaReleaseAsset(
    tag="b12738",
    asset_name="llama-b12738-bin-win-cuda-13.5-x64.zip",
    url=(
        "https://github.com/ggml-org/llama.cpp/releases/download/b12738/"
        "llama-b12738-bin-win-cuda-13.5-x64.zip"
    ),
    sha256="cf667446db7eca5801fafe817e8b8aa12dbd6dbb41c1a74bbc6514a50e1b67c4",
    size=226_093_012,
    cuda_major=13,
    cuda_minor=5,
    cudart_name="cudart-llama-bin-win-cuda-13.5-x64.zip",
    cudart_url=(
        "https://github.com/ggml-org/llama.cpp/releases/download/b12738/"
        "cudart-llama-bin-win-cuda-13.5-x64.zip"
    ),
    cudart_sha256="c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba",
    cudart_size=426_300_000,
)


def managed_runtime_dir() -> Path:
    """Managed directory under ``./models/runtime/llama-cpp``."""
    return MODELS_ROOT / "runtime" / "llama-cpp"


def managed_server_executable() -> Path | None:
    """Return the managed ``llama-server`` binary if already installed."""
    runtime_dir = managed_runtime_dir()
    if not runtime_dir.is_dir():
        return None
    for name in _SERVER_EXE_NAMES:
        candidate = runtime_dir / name
        if candidate.is_file():
            return candidate
    return None


def auto_download_allowed() -> bool:
    """Whether automatic runtime download is permitted in the current env."""
    if sys.platform != "win32":
        return False
    offline = str(os.getenv("HF_HUB_OFFLINE", "") or "").strip().lower()
    if offline in {"1", "true", "yes"}:
        return False
    enabled = str(os.getenv("LLAMACPP_AUTO_DOWNLOAD", "1") or "1").strip().lower()
    return enabled not in {"0", "false", "no"}


def detect_cuda_major(*, torch_lib_dir: Path | None = None) -> int:
    """Detect the installed PyTorch CUDA major version without importing torch."""
    lib_dir = torch_lib_dir if torch_lib_dir is not None else llamacpp_env._torch_library_dir()
    if lib_dir is not None and lib_dir.is_dir():
        majors: list[int] = []
        for pattern in ("cublas64_*.dll", "cudart64_*.dll"):
            for dll in lib_dir.glob(pattern):
                match = re.search(r"64_(\d+)\.dll$", dll.name, re.I)
                if match:
                    majors.append(int(match.group(1)))
        if majors:
            return max(majors)
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = ""
    match = re.search(r"\+cu(\d+)", str(torch_version or ""), re.I)
    if match:
        raw = int(match.group(1))
        return raw // 10 if raw >= 100 else raw
    return _DEFAULT_CUDA_MAJOR


def _clean_sha256(digest: str | None) -> str:
    text = str(digest or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :].strip()
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else ""


def select_release_asset(
    releases: list[dict], *, cuda_major: int
) -> LlamaReleaseAsset | None:
    """Select the newest Windows x64 CUDA-``<cuda_major>.*`` asset from GitHub releases.

    Rolling ``bXXXXX`` builds are marked ``prerelease=True`` on
    ``ggml-org/llama.cpp``, so drafts are skipped while prereleases are kept.
    """
    bin_re = re.compile(
        rf"^llama-(b\d+)-bin-win-cuda-{int(cuda_major)}\.(\d+)-x64\.zip$",
        re.I,
    )
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft"):
            continue
        assets = rel.get("assets") or []
        if not isinstance(assets, list):
            continue
        candidates: list[tuple[int, str, dict, str]] = []
        by_name: dict[str, dict] = {}
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name") or "")
            if name:
                by_name[name.lower()] = asset
            match = bin_re.match(name)
            if match:
                sha256 = _clean_sha256(asset.get("digest"))
                if not sha256:
                    logger.warning(
                        "Skipping llama.cpp release asset %s without a valid SHA-256 digest",
                        name,
                    )
                    continue
                candidates.append((int(match.group(2)), match.group(1), asset, sha256))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        minor, build_tag, chosen, chosen_sha256 = candidates[0]
        url = str(chosen.get("browser_download_url") or "").strip()
        if not url:
            continue
        tag = str(rel.get("tag_name") or build_tag).strip() or build_tag
        cudart_key = f"cudart-llama-bin-win-cuda-{int(cuda_major)}.{minor}-x64.zip"
        cudart = by_name.get(cudart_key.lower()) or {}
        cudart_sha256 = _clean_sha256(cudart.get("digest"))
        cudart_url = str(cudart.get("browser_download_url") or "").strip() if cudart_sha256 else ""
        cudart_name = str(cudart.get("name") or "") if cudart_url else ""
        cudart_size = int(cudart.get("size") or 0) if cudart_url else 0
        return LlamaReleaseAsset(
            tag=tag,
            asset_name=str(chosen.get("name") or ""),
            url=url,
            sha256=chosen_sha256,
            size=int(chosen.get("size") or 0),
            cuda_major=int(cuda_major),
            cuda_minor=minor,
            cudart_name=cudart_name,
            cudart_url=cudart_url,
            cudart_sha256=cudart_sha256,
            cudart_size=cudart_size,
        )
    return None


def _raise_if_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TranslationCancelledError("用户取消了本地翻译引擎下载。")


def _fallback_for_cuda(cuda_major: int) -> LlamaReleaseAsset | None:
    if int(cuda_major) == FALLBACK_CUDA13_RELEASE.cuda_major:
        return FALLBACK_CUDA13_RELEASE
    return None


def resolve_latest_release_asset(
    *,
    cuda_major: int,
    cancel_event=None,
    fetch_releases: Callable[[], list[dict]] | None = None,
) -> LlamaReleaseAsset:
    """Resolve the newest matching CUDA asset, falling back on API errors/rate limits."""
    _raise_if_cancelled(cancel_event)
    fallback = _fallback_for_cuda(cuda_major)
    try:
        if fetch_releases is not None:
            releases = fetch_releases()
        else:
            import httpx

            async def _fetch_once() -> list[dict]:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    trust_env=True,
                    timeout=httpx.Timeout(20.0, connect=15.0),
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Accept": "application/vnd.github+json",
                    },
                ) as client:
                    response = await client.get(LLAMACPP_RELEASES_API)
                    response.raise_for_status()
                    payload = response.json()
                    return payload if isinstance(payload, list) else []

            releases = run_cancellable_request(_fetch_once, cancel_event=cancel_event)
        asset = select_release_asset(releases, cuda_major=cuda_major)
        if asset is not None:
            return asset
        logger.warning(
            "No llama.cpp release asset matched Windows x64 CUDA %s.x; checking fallback",
            cuda_major,
        )
    except TranslationCancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Failed to query llama.cpp GitHub releases (CUDA %s.x): %s; checking fallback",
            cuda_major,
            exc,
        )

    if fallback is not None:
        logger.info(
            "Using verified fallback llama.cpp release %s (%s)",
            fallback.tag,
            fallback.asset_name,
        )
        return fallback
    raise RuntimeError(
        f"未能自动匹配适用于 CUDA {cuda_major}.x 的 Windows x64 llama.cpp 版本，"
        "且当前主版本无内置兜底包。请检查网络/代理配置，"
        "或在 config/config.json 中设置 LLAMACPP_SERVER_PATH 手动指定路径。"
    )


def _sleep_interruptible(seconds: float, cancel_event=None) -> None:
    if seconds <= 0:
        return
    if cancel_event is not None:
        if cancel_event.wait(timeout=seconds):
            _raise_if_cancelled(cancel_event)
        return
    time.sleep(seconds)


def _download_archive(
    url: str,
    dest: Path,
    *,
    display_name: str,
    expected_sha256: str = "",
    expected_size: int = 0,
    require_sha256: bool = True,
    cancel_event=None,
) -> None:
    """Stream-download a release archive with retries, proxy support, progress, and SHA-256."""
    import httpx

    clean_expected = _clean_sha256(expected_sha256)
    if require_sha256 and not clean_expected:
        raise RuntimeError(
            f"{display_name} 缺少有效的 SHA-256 校验摘要，拒绝下载未经核验的可执行包。"
        )

    identity = hf_progress.current_identity()
    total_started = time.perf_counter()
    start_extra: dict[str, object] = {"file": display_name}
    if expected_size > 0:
        start_extra["size_mb"] = round(expected_size / (1024.0 * 1024.0), 2)
    hf_progress._emit("start", start_extra, identity)

    part = dest.with_suffix(dest.suffix + ".part")
    max_attempts = max(1, int(_DOWNLOAD_MAX_ATTEMPTS))

    try:
        for attempt in range(1, max_attempts + 1):
            _raise_if_cancelled(cancel_event)
            attempt_started = time.perf_counter()

            async def _stream_once(started_at: float = attempt_started) -> str:
                hasher = hashlib.sha256()
                downloaded = 0
                total_bytes = expected_size
                last_emit = 0.0
                last_pct = -1
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    trust_env=True,
                    timeout=httpx.Timeout(60.0, connect=20.0),
                    headers={"User-Agent": _USER_AGENT},
                ) as client:
                    async with client.stream("GET", url) as resp:
                        resp.raise_for_status()
                        cl = resp.headers.get("content-length")
                        if cl and cl.isdigit():
                            total_bytes = int(cl)
                        with open(part, "wb") as handle:
                            async for chunk in resp.aiter_bytes(
                                chunk_size=_DOWNLOAD_CHUNK_BYTES
                            ):
                                _raise_if_cancelled(cancel_event)
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                hasher.update(chunk)
                                downloaded += len(chunk)
                                now = time.perf_counter()
                                pct = (
                                    min(100, int(downloaded * 100 / total_bytes))
                                    if total_bytes > 0
                                    else 0
                                )
                                if (
                                    now - last_emit >= _PROGRESS_INTERVAL_S
                                    or pct >= last_pct + 5
                                ):
                                    elapsed = max(now - started_at, 1e-3)
                                    extra: dict[str, object] = {
                                        "file": display_name,
                                        "downloaded_mb": round(
                                            downloaded / (1024.0 * 1024.0), 2
                                        ),
                                        "speed_mb": round(
                                            downloaded / (1024.0 * 1024.0) / elapsed, 2
                                        ),
                                        "elapsed_s": round(elapsed, 2),
                                    }
                                    if total_bytes > 0:
                                        extra["size_mb"] = round(
                                            total_bytes / (1024.0 * 1024.0), 2
                                        )
                                        extra["pct"] = pct
                                    hf_progress._emit("progress", extra, identity)
                                    last_emit = now
                                    last_pct = pct
                return hasher.hexdigest().lower()

            try:
                digest = run_cancellable_request(
                    _stream_once, cancel_event=cancel_event
                )
                if clean_expected and digest != clean_expected:
                    raise RuntimeError(
                        f"{display_name} SHA-256 校验失败（期望 {clean_expected[:12]}…，"
                        f"实际 {digest[:12]}…），已中止安装。"
                    )
                os.replace(part, dest)
                hf_progress._emit(
                    "done",
                    {
                        "file": display_name,
                        "elapsed_s": round(time.perf_counter() - total_started, 2),
                    },
                    identity,
                )
                return
            except (httpx.TransportError, httpx.HTTPStatusError, OSError) as exc:
                if part.exists():
                    try:
                        part.unlink()
                    except OSError:
                        pass
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response is not None
                    and exc.response.status_code in _RETRYABLE_HTTP_STATUS
                )
                if not retryable or attempt >= max_attempts:
                    raise
                delay = _DOWNLOAD_RETRY_DELAYS_S[
                    min(attempt - 1, len(_DOWNLOAD_RETRY_DELAYS_S) - 1)
                ]
                logger.warning(
                    "Download attempt %d/%d for %s failed (%s); retrying in %.1fs",
                    attempt,
                    max_attempts,
                    display_name,
                    exc,
                    delay,
                )
                _sleep_interruptible(delay, cancel_event)
    except BaseException as exc:
        if part.exists():
            try:
                part.unlink()
            except OSError:
                pass
        hf_progress._emit(
            "error",
            {
                "file": display_name,
                "elapsed_s": round(time.perf_counter() - total_started, 2),
                "error": str(exc)[:300],
            },
            identity,
        )
        raise


def _safe_extract_zip(archive: Path, target_dir: Path) -> None:
    """Extract a zip safely into ``target_dir``, rejecting Zip Slip paths."""
    resolved_root = target_dir.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            member_path = (resolved_root / member.filename).resolve()
            if not member_path.is_relative_to(resolved_root):
                raise RuntimeError(f"压缩包包含非法路径：{member.filename}")
        zf.extractall(resolved_root)


def _locate_and_flatten_server_dir(extract_dir: Path) -> Path:
    """Find ``llama-server.exe`` inside ``extract_dir`` and return its containing dir."""
    for name in _SERVER_EXE_NAMES:
        matches = list(extract_dir.rglob(name))
        if matches:
            return matches[0].parent
    raise RuntimeError("解压后的 llama.cpp 压缩包中未找到 llama-server 可执行文件。")


def _has_cuda_device(devices: list[str]) -> bool:
    return any(str(entry).strip().lower().startswith("cuda") for entry in devices)


def _stage_and_verify_asset(
    asset: LlamaReleaseAsset,
    *,
    staging_root: Path,
    env: dict[str, str],
    probe_devices: Callable[[str, dict[str, str] | None], list[str]],
    download_fn: Callable[..., None],
    cancel_event=None,
) -> tuple[Path, list[str], bool]:
    """Download, extract, and CUDA-probe one asset. Returns (payload_dir, devices, cudart_used)."""
    work_dir = staging_root / f"stage-{asset.tag}-{uuid.uuid4().hex[:8]}"
    extract_dir = work_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    archive = work_dir / asset.asset_name
    try:
        download_fn(
            asset.url,
            archive,
            display_name=asset.asset_name,
            expected_sha256=asset.sha256,
            expected_size=asset.size,
            cancel_event=cancel_event,
        )
        _raise_if_cancelled(cancel_event)
        _safe_extract_zip(archive, extract_dir)
        payload_dir = _locate_and_flatten_server_dir(extract_dir)
        exe = next(
            (payload_dir / name for name in _SERVER_EXE_NAMES if (payload_dir / name).is_file()),
            None,
        )
        assert exe is not None

        devices = probe_devices(str(exe), env)
        if _has_cuda_device(devices):
            return payload_dir, devices, False

        # Second chance inside the same release: fetch companion cudart DLLs in case
        # the newer CUDA 13.x binary requires symbols absent from torch's cu130 DLLs.
        if asset.cudart_url:
            logger.warning(
                "llama-server %s detected no CUDA devices with torch DLLs (%s); "
                "downloading companion %s",
                asset.tag,
                devices,
                asset.cudart_name or "cudart",
            )
            cudart_archive = work_dir / (asset.cudart_name or f"cudart-{asset.tag}.zip")
            download_fn(
                asset.cudart_url,
                cudart_archive,
                display_name=asset.cudart_name or cudart_archive.name,
                expected_sha256=asset.cudart_sha256,
                expected_size=asset.cudart_size,
                cancel_event=cancel_event,
            )
            _raise_if_cancelled(cancel_event)
            _safe_extract_zip(cudart_archive, payload_dir)
            devices = probe_devices(str(exe), env)
            if _has_cuda_device(devices):
                return payload_dir, devices, True

        shutil.rmtree(work_dir, ignore_errors=True)
        return Path(), devices, False
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def _publish_staged_runtime(
    payload_dir: Path,
    target_dir: Path,
    *,
    meta: dict[str, object],
) -> Path:
    """Atomically publish ``payload_dir`` to ``target_dir``."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    (payload_dir / "install-meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    old_dir: Path | None = None
    if target_dir.exists():
        old_dir = target_dir.parent / f".old-{target_dir.name}-{uuid.uuid4().hex[:8]}"
        os.replace(target_dir, old_dir)
    try:
        os.replace(payload_dir, target_dir)
    except BaseException:
        if old_dir is not None and old_dir.exists() and not target_dir.exists():
            os.replace(old_dir, target_dir)
        raise
    finally:
        if old_dir is not None and old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)

    for name in _SERVER_EXE_NAMES:
        candidate = target_dir / name
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"受管目录 {target_dir} 中缺少 llama-server 可执行文件。")


def ensure_managed_llama_server(
    *,
    cancel_event=None,
    cuda_major: int | None = None,
    fetch_releases: Callable[[], list[dict]] | None = None,
    download_fn: Callable[..., None] | None = None,
    probe_devices: Callable[[str, dict[str, str] | None], list[str]] | None = None,
    env_factory: Callable[[], dict[str, str]] | None = None,
) -> str:
    """Ensure a CUDA-verified ``llama-server`` exists under ``./models/runtime/llama-cpp/``."""
    global _INSTALL_IN_PROGRESS

    while True:
        _raise_if_cancelled(cancel_event)
        existing = managed_server_executable()
        if existing is not None:
            return str(existing)
        with _INSTALL_COND:
            existing = managed_server_executable()
            if existing is not None:
                return str(existing)
            if not _INSTALL_IN_PROGRESS:
                _INSTALL_IN_PROGRESS = True
                break
            _INSTALL_COND.wait(timeout=_INSTALL_WAIT_POLL_S)

    try:
        resolved_cuda_major = (
            int(cuda_major) if cuda_major is not None else detect_cuda_major()
        )
        env = env_factory() if env_factory is not None else llamacpp_env.server_environment()
        probe = probe_devices or llamacpp_env.probe_compute_devices
        downloader = download_fn or _download_archive

        target_dir = managed_runtime_dir()
        staging_root = target_dir.parent / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)

        primary = resolve_latest_release_asset(
            cuda_major=resolved_cuda_major,
            cancel_event=cancel_event,
            fetch_releases=fetch_releases,
        )
        fallback = _fallback_for_cuda(resolved_cuda_major)
        plan = [primary]
        if fallback is not None and fallback.tag != primary.tag:
            plan.append(fallback)

        last_devices: list[str] = []
        last_error: Exception | None = None
        for idx, asset in enumerate(plan):
            _raise_if_cancelled(cancel_event)
            try:
                payload_dir, devices, cudart_used = _stage_and_verify_asset(
                    asset,
                    staging_root=staging_root,
                    env=env,
                    probe_devices=probe,
                    download_fn=downloader,
                    cancel_event=cancel_event,
                )
            except TranslationCancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if idx + 1 < len(plan):
                    logger.warning(
                        "Managed llama.cpp install failed for %s (%s); retrying fallback %s",
                        asset.tag,
                        exc,
                        plan[idx + 1].tag,
                    )
                    continue
                raise RuntimeError(
                    f"自动下载本地翻译引擎 llama.cpp（{asset.tag}）失败：{exc}。"
                    "请检查网络或代理设置，或在 config/config.json 中设置 LLAMACPP_SERVER_PATH 手动指定路径。"
                ) from exc

            if payload_dir != Path():
                work_root = payload_dir
                while work_root.parent != staging_root and work_root.parent != work_root:
                    work_root = work_root.parent
                try:
                    exe = _publish_staged_runtime(
                        payload_dir,
                        target_dir,
                        meta={
                            "tag": asset.tag,
                            "asset_name": asset.asset_name,
                            "sha256": asset.sha256,
                            "cuda_major": asset.cuda_major,
                            "cuda_minor": asset.cuda_minor,
                            "cudart_installed": cudart_used,
                            "devices": devices,
                            "installed_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                finally:
                    if work_root.parent == staging_root:
                        shutil.rmtree(work_root, ignore_errors=True)
                logger.info(
                    "Installed managed llama-server %s into %s (devices=%s, cudart=%s)",
                    asset.tag,
                    target_dir,
                    devices,
                    cudart_used,
                )
                return str(exe)

            last_devices = devices
            if idx + 1 < len(plan):
                logger.warning(
                    "llama-server %s detected no CUDA devices (%s); retrying verified fallback %s",
                    asset.tag,
                    devices,
                    plan[idx + 1].tag,
                )

        raise RuntimeError(
            "自动下载的 CUDA 版 llama-server 未检测到任何 NVIDIA CUDA 设备"
            f"（探测结果：{last_devices or '无'}），已拒绝安装以避免静默回退到 CPU。"
            f"请确认 NVIDIA 显卡驱动满足 CUDA {resolved_cuda_major} 要求，"
            "或在 config/config.json 中设置 LLAMACPP_SERVER_PATH 手动指定路径。"
        ) from last_error
    finally:
        with _INSTALL_COND:
            _INSTALL_IN_PROGRESS = False
            _INSTALL_COND.notify_all()


def start_managed_llama_server_prefetch() -> str:
    """Kick off a single-flight background install of ``./models/runtime/llama-cpp/``.

    Called when the user enables local translation (saving ``TRANSLATION_BACKEND=llamacpp``
    or launching with it active) so the ~226 MB CUDA runtime is already ready by the
    time the first video runs. Never raises to callers; failures are logged and also
    surfaced via ``model_download`` error events, and the first translation task will
    retry synchronously if the directory is still missing.
    """
    global _prefetch_thread

    if str(os.getenv("LLAMACPP_SERVER_PATH", "") or "").strip():
        return "explicit_override"
    if managed_server_executable() is not None:
        return "ready"
    if not auto_download_allowed():
        return "disabled"

    with _PREFETCH_LOCK:
        if _prefetch_thread is not None and _prefetch_thread.is_alive():
            return "in_progress"

        def _worker() -> None:
            hf_progress.adopt_identity(("", "", ""))
            try:
                ensure_managed_llama_server()
            except Exception as exc:  # noqa: BLE001 - background prefetch never crashes app
                logger.warning("Managed llama-server background prefetch failed: %s", exc)

        thread = threading.Thread(
            target=_worker,
            name="llamacpp-runtime-prefetch",
            daemon=True,
        )
        _prefetch_thread = thread
        thread.start()
        return "started"
