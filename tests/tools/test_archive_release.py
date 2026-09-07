"""Small real ZIP and 7-Zip payloads, including mutation during compression."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import archive_release as archive_tool
import record_release_manifest as manifest_tool


@pytest.fixture
def seven_zip():
    choices = [
        shutil.which("7z"), shutil.which("7zz"),
        "C:/Program Files/7-Zip/7z.exe", "C:/Program Files/7-Zip-Zstandard/7z.exe",
    ]
    for choice in choices:
        if choice and Path(choice).is_file():
            return Path(choice)
    pytest.skip("7-Zip CLI is required for the real archive smoke")


@pytest.fixture
def payload(tmp_path):
    source = tmp_path / "sample-app"
    source.mkdir()
    (source / "settings.json").write_text('{"version":1}', "utf-8")
    (source / "app.pyc").write_bytes(b"sample executable payload")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_tool.build_manifest(source)), "utf-8")
    return source, manifest


def test_archive_and_manifest_are_published_as_one_new_generation(tmp_path, payload, seven_zip):
    source, manifest = payload
    releases = tmp_path / "releases"
    old = releases / "previous"
    old.mkdir(parents=True)
    (old / "sample.7z").write_bytes(b"previous release")
    generation = archive_tool.archive_release(source, releases, manifest, seven_zip=seven_zip, threads=1)
    assert generation != old
    assert {p.name for p in generation.iterdir()} == {
        "jav-trans-windows-x64.7z", "jav-trans-windows-x64.7z.sha256", "release-manifest.json",
    }
    archive = generation / "jav-trans-windows-x64.7z"
    assert archive.with_suffix(".7z.sha256").read_text("ascii").startswith(manifest_tool.sha256_of(archive))
    assert (old / "sample.7z").read_bytes() == b"previous release"


def test_archive_verifies_actual_compressed_bytes_before_publication(tmp_path, payload, seven_zip, monkeypatch):
    source, manifest = payload
    releases = tmp_path / "releases"
    run = archive_tool.subprocess.run

    def mutate_before_compression(command, **kwargs):
        if len(command) > 1 and command[1] == "a":
            (source / "app.pyc").write_bytes(b"changed after preflight")
        return run(command, **kwargs)

    monkeypatch.setattr(archive_tool.subprocess, "run", mutate_before_compression)
    with pytest.raises(ValueError, match="压缩包实际内容"):
        archive_tool.archive_release(source, releases, manifest, seven_zip=seven_zip, threads=1)
    assert not list(releases.glob("release-*"))


def test_setup_zip_uses_the_same_verified_generation_contract(tmp_path, payload):
    source, manifest = payload
    releases = tmp_path / "releases"
    generation = archive_tool.archive_release(source, releases, manifest, archive_name="sample-setup.zip")
    assert {path.name for path in generation.iterdir()} == {
        "sample-setup.zip", "sample-setup.zip.sha256", "release-manifest.json",
    }
    archive = generation / "sample-setup.zip"
    assert archive.with_suffix(".zip.sha256").read_text("ascii").startswith(manifest_tool.sha256_of(archive))
    with archive_tool.zipfile.ZipFile(archive) as reader:
        assert reader.read("sample-app/app.pyc") == b"sample executable payload"
    second = archive_tool.archive_release(source, releases, manifest, archive_name="sample-setup.zip")
    assert second != generation and archive.is_file()


def test_setup_zip_checks_extracted_bytes_after_source_changes(tmp_path, payload, monkeypatch):
    source, manifest = payload
    write = archive_tool.zipfile.ZipFile.write

    def mutate_before_write(writer, filename, *args, **kwargs):
        if Path(filename).name == "app.pyc":
            Path(filename).write_bytes(b"changed after preflight")
        return write(writer, filename, *args, **kwargs)

    monkeypatch.setattr(archive_tool.zipfile.ZipFile, "write", mutate_before_write)
    releases = tmp_path / "releases"
    with pytest.raises(ValueError, match="压缩包实际内容"):
        archive_tool.archive_release(source, releases, manifest, archive_name="sample-setup.zip")
    assert not list(releases.glob("release-*"))
