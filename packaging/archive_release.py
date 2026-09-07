"""Build, unpack and verify an archive before exposing one complete release directory."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from record_release_manifest import ROOT, sha256_of, verify
from utils.subprocess_tools import no_window_subprocess_kwargs


def archive_release(
    source: Path, output_dir: Path, manifest_path: Path, *,
    seven_zip: Path | None = None, archive_name: str = "jav-trans-windows-x64.7z", threads: int = 1,
) -> Path:
    source, output_dir = source.resolve(), output_dir.resolve()
    if not source.is_dir():
        raise ValueError("发行载荷目录不存在")
    if source == output_dir or source in output_dir.parents:
        raise ValueError("归档目录不能放在载荷目录内")
    suffix = Path(archive_name).suffix.lower()
    if Path(archive_name).name != archive_name or suffix not in {".7z", ".zip"}:
        raise ValueError("归档名称必须是不含路径的 .7z 或 .zip 文件名")
    if suffix == ".7z" and seven_zip is None:
        raise ValueError("7-Zip 归档需要指定 --seven-zip")
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = output_dir / f"release-{datetime.now():%Y%m%d_%H%M%S}-{uuid.uuid4().hex[:8]}"
    # Only this invocation's generated scratch files are cleaned. Existing
    # release directories and archives are never overwritten or removed.
    with tempfile.TemporaryDirectory(prefix=".archive-", dir=output_dir) as scratch:
        scratch = Path(scratch)
        bundle = scratch / "bundle"
        bundle.mkdir()
        recorded = bundle / "release-manifest.json"
        shutil.copyfile(manifest_path, recorded)
        if verify(recorded, source):
            raise ValueError("载荷与构建清单不符；未创建发行代次")
        archive = bundle / archive_name
        unpacked = scratch / "unpacked"
        if suffix == ".zip":
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as writer:
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        writer.write(path, (Path(source.name) / path.relative_to(source)).as_posix())
            with zipfile.ZipFile(archive) as reader:
                reader.extractall(unpacked)
        else:
            subprocess.run([
                str(seven_zip), "a", "-t7z", "-m0=LZMA2", "-mx=5", f"-mmt={max(1, threads)}",
                str(archive), str(source),
            ], check=True, capture_output=True, **no_window_subprocess_kwargs())
            subprocess.run([
                str(seven_zip), "x", "-y", f"-o{unpacked}", str(archive),
            ], check=True, capture_output=True, **no_window_subprocess_kwargs())
        # Checking only the source before compression leaves a race. Verify the
        # actual archive bytes after extraction, before publishing the bundle.
        children = list(unpacked.iterdir())
        if len(children) != 1 or children[0].name != source.name or not children[0].is_dir():
            raise ValueError("压缩包目录结构与载荷不符")
        if verify(recorded, children[0]):
            raise ValueError("压缩包实际内容与构建清单不符；未发布发行代次")
        checksum = bundle / f"{archive_name}.sha256"
        checksum.write_text(f"{sha256_of(archive)}  {archive_name}\n", encoding="ascii")
        for path in bundle.iterdir():
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        # Archive, checksum and manifest become visible together. A fresh name
        # also prevents one release command from replacing another's result.
        bundle.rename(generation)
    return generation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dist/jav-trans")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/release-assets")
    parser.add_argument("--manifest", type=Path, default=ROOT / "dist/release-assets/release-manifest.json")
    parser.add_argument("--seven-zip", type=Path)
    parser.add_argument("--name", default="jav-trans-windows-x64.7z")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    try:
        result = archive_release(
            args.source, args.output, args.manifest, seven_zip=args.seven_zip,
            archive_name=args.name, threads=args.threads,
        )
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        print(f"[packaging] 归档失败：{exc}")
        return 1
    print(f"[packaging] 已验证并保存完整发行代次：{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
