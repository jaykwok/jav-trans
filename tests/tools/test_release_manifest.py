"""A release records what it contains, and can be checked against it later.

Two builds of one commit can carry different model weights, a different
llama-server binary, or a lock file edited between them, and the version number
says nothing about any of it. The manifest is what makes "the release" a
specific set of bytes rather than a commit plus whatever was on the machine.
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "packaging") not in sys.path:
    sys.path.insert(0, str(ROOT / "packaging"))

import record_release_manifest as manifest_tool  # noqa: E402
import prepare_default_model as prepare_tool  # noqa: E402
from utils import model_paths  # noqa: E402

SHA = "1" * 40


def _model(root: Path, content: bytes = b"pinned weights") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}", "utf-8")
    (root / "model.safetensors").write_bytes(content)
    (root / "tokenizer.json").write_text("{}", "utf-8")
    return root


def _dist(tmp_path: Path) -> Path:
    dist = tmp_path / "jav-trans"
    (dist / "models" / "asr").mkdir(parents=True)
    (dist / "_internal").mkdir(parents=True)
    (dist / "models" / "asr" / "model.safetensors").write_bytes(b"weights")
    (dist / "models" / "asr" / "chat_template.jinja").write_text("{{ x }}", "utf-8")
    (dist / "llama-server.exe").write_bytes(b"binary")
    # Bytecode is executable behaviour and must be covered as well.
    (dist / "_internal" / "app.pyc").write_bytes(b"\x00" * 1024)
    return dist


def test_the_things_that_decide_behaviour_are_hashed(tmp_path):
    dist = _dist(tmp_path)

    manifest = manifest_tool.build_manifest(dist)

    categories = {item["category"] for item in manifest["files"]}
    assert {"weights", "template", "binary"} <= categories
    # The lock is hashed from the repo, where it lives.
    assert any(item["path"] == "uv.lock" for item in manifest["files"])
    assert manifest["unhashed"] == {"count": 0, "bytes": 0}
    assert any(item["path"] == "_internal/app.pyc" for item in manifest["files"])


def test_paths_are_relative_to_the_roots_not_to_the_build_machine(tmp_path):
    dist = _dist(tmp_path)

    manifest = manifest_tool.build_manifest(dist)

    for item in manifest["files"]:
        assert not Path(item["path"]).is_absolute()
        assert str(tmp_path) not in item["path"]


def test_verification_passes_on_an_untouched_tree(tmp_path, capsys):
    dist = _dist(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_tool.build_manifest(dist)), encoding="utf-8"
    )

    assert manifest_tool.verify(manifest_path, dist) == 0


def test_a_swapped_model_file_is_caught(tmp_path, capsys):
    dist = _dist(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_tool.build_manifest(dist)), encoding="utf-8"
    )

    (dist / "models" / "asr" / "model.safetensors").write_bytes(b"other weights")

    assert manifest_tool.verify(manifest_path, dist) == 1
    assert "model.safetensors" in capsys.readouterr().out


def test_a_file_added_after_the_fact_is_caught(tmp_path, capsys):
    dist = _dist(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_tool.build_manifest(dist)), encoding="utf-8"
    )

    (dist / "extra.dll").write_bytes(b"smuggled")

    assert manifest_tool.verify(manifest_path, dist) == 1
    assert "extra.dll" in capsys.readouterr().out


class TestPinningTheModel:
    def test_a_build_without_a_pin_refuses_to_run(self, monkeypatch, capsys):
        """The alignment head has been pinned by sha since it moved into the
        defaults; the ASR model was not, so two builds of one commit could ship
        different weights with nothing recording which."""
        monkeypatch.setattr(prepare_tool, "read_pins", dict)
        monkeypatch.setattr(
            prepare_tool,
            "resolve_model_spec",
            lambda *a, **k: pytest_fail("a build with no pin must not download"),
        )

        assert prepare_tool.prepare() == 1
        assert "--pin" in capsys.readouterr().err

    def test_the_pin_is_passed_to_the_download(self, monkeypatch, tmp_path):
        repo = prepare_tool.REQUIRED_MODEL_SPECS[0]["repo_id"]
        monkeypatch.setattr(prepare_tool, "read_pins", lambda: {repo: SHA})
        seen: dict = {}

        def fake_resolve(_path, repo_id, *, download, revision, allow_patterns):
            seen.update(repo_id=repo_id, revision=revision, download=download)
            target = _model(tmp_path / "model")
            model_paths.record_model_revision(target, repo_id, revision)
            return str(target)

        monkeypatch.setattr(prepare_tool, "resolve_model_spec", fake_resolve)

        assert prepare_tool.prepare() == 0
        assert seen == {"repo_id": repo, "revision": SHA, "download": True}

    def test_unpinned_escape_hatch_is_not_accepted(self):
        with pytest.raises(SystemExit) as error:
            prepare_tool.main(["--allow-unpinned"])
        assert error.value.code == 2


def test_old_canonical_cache_cannot_satisfy_a_new_pin(tmp_path, monkeypatch):
    repo = prepare_tool.REQUIRED_MODEL_SPECS[0]["repo_id"]
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")
    canonical = _model(model_paths.canonical_model_dir(repo), b"old weights")
    calls = []

    def download(repo_id, target, **kwargs):
        calls.append((repo_id, kwargs["revision"]))
        _model(target)
        return str(target)

    monkeypatch.setattr(model_paths, "_download_snapshot", download)
    resolved = Path(model_paths.resolve_model_spec(None, repo, download=True, revision=SHA))
    assert resolved == model_paths.revision_model_dir(repo, SHA).resolve()
    assert resolved != canonical
    assert calls == [(repo, SHA)]
    assert (canonical / "model.safetensors").read_bytes() == b"old weights"
    model_paths.verify_model_revision(resolved, repo, SHA)
    assert Path(model_paths.resolve_model_spec(None, repo, download=False, revision=SHA)) == resolved
    assert calls == [(repo, SHA)]


def test_modified_pinned_cache_is_refused_before_download(tmp_path, monkeypatch):
    repo = prepare_tool.REQUIRED_MODEL_SPECS[0]["repo_id"]
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")
    target = _model(model_paths.revision_model_dir(repo, SHA))
    model_paths.record_model_revision(target, repo, SHA)
    (target / "config.json").write_text('{"changed":true}', "utf-8")
    monkeypatch.setattr(model_paths, "_download_snapshot", lambda *a, **k: pytest.fail("do not replace a tampered cache silently"))
    with pytest.raises(ValueError, match="哈希"):
        model_paths.resolve_model_spec(None, repo, download=True, revision=SHA)


def test_spec_input_and_actual_dist_verify_the_same_model_revision(tmp_path, monkeypatch):
    repo = prepare_tool.REQUIRED_MODEL_SPECS[0]["repo_id"]
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(prepare_tool, "read_pins", lambda: {repo: SHA})
    monkeypatch.setattr(manifest_tool, "model_pins", lambda: {repo: SHA})
    target = _model(model_paths.revision_model_dir(repo, SHA))
    model_paths.record_model_revision(target, repo, SHA)
    sources = prepare_tool.verified_release_models()
    assert sources[0][0] == target.resolve()
    dist = tmp_path / "dist"
    bundled = dist / "_internal" / "models" / model_paths.model_dir_name(repo)
    for path in model_paths.inference_model_files(target, include_receipt=True):
        destination = bundled / path.relative_to(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    payload = manifest_tool.build_manifest(dist, require_models=True)
    assert payload["model_payloads"][0]["revision"] == SHA
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), "utf-8")
    assert manifest_tool.verify(manifest_path, dist) == 0
    (bundled / "model.safetensors").write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="哈希"):
        manifest_tool.build_manifest(dist, require_models=True)
    assert manifest_tool.verify(manifest_path, dist) == 1


def test_real_pyinstaller_spec_collects_only_verified_model_files(tmp_path, monkeypatch):
    repo = prepare_tool.REQUIRED_MODEL_SPECS[0]["repo_id"]
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(prepare_tool, "read_pins", lambda: {repo: SHA})
    monkeypatch.setattr(manifest_tool, "model_pins", lambda: {repo: SHA})
    monkeypatch.setattr(sys, "path", list(sys.path))
    pinned = _model(model_paths.revision_model_dir(repo, SHA))
    model_paths.record_model_revision(pinned, repo, SHA)
    _model(model_paths.canonical_model_dir(repo), b"unrelated canonical cache")
    (pinned / "optimizer.pt").write_bytes(b"training only")
    head = tmp_path / "ctc_aligner.pt"
    head.write_bytes(b"synthetic head for collection only")
    import huggingface_hub

    revisions = []

    def alignment_download(*, repo_id, filename, revision):
        revisions.append(model_paths.require_commit_sha(revision))
        return str(head)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", alignment_download)
    hooks = ModuleType("PyInstaller.utils.hooks")
    hooks.collect_data_files = lambda *_args, **_kwargs: []
    hooks.collect_submodules = lambda *_args, **_kwargs: []
    hooks.copy_metadata = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    spec_path = ROOT / "packaging/jav-trans-web.spec"
    tree = ast.parse(spec_path.read_text("utf-8"), str(spec_path))
    # Execute the real spec through data collection. Binary analysis and the
    # build itself need installed tools and are outside this small offline smoke.
    boundary = next(index for index, node in enumerate(tree.body)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "binaries" for target in node.targets))
    namespace = {"SPECPATH": str(spec_path.parent)}
    monkeypatch.delenv("JAV_TRANS_SKIP_MODELS", raising=False)
    exec(compile(ast.Module(body=tree.body[:boundary], type_ignores=[]), str(spec_path), "exec"), namespace)
    dist = tmp_path / "dist"
    model_data = [(Path(source), Path(destination)) for source, destination in namespace["datas"]
                  if Path(destination).parts[0] == "models"]
    assert revisions and all(pinned in source.parents or source == head for source, _ in model_data)
    assert not any(source.name == "optimizer.pt" for source, _ in model_data)
    for source, destination in model_data:
        target = dist / "_internal" / destination / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    payload = manifest_tool.build_manifest(dist, require_models=True)
    assert payload["model_payloads"][0]["revision"] == SHA
    assert any(item["path"].endswith("model-revision.json") for item in payload["files"])


@pytest.mark.parametrize("filename", ["_internal/app.pyc", "settings.json", "tokenizer.json"])
def test_non_weight_payload_changes_are_detected(tmp_path, filename):
    dist = _dist(tmp_path)
    path = dist / filename
    path.write_bytes(b"before")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_tool.build_manifest(dist)), "utf-8")
    path.write_bytes(b"after!")
    assert manifest_tool.verify(manifest_path, dist) == 1


@pytest.mark.parametrize("revision", ["main", "abc123", "z" * 40, ""])
def test_pins_require_full_commit_sha(revision):
    with pytest.raises(ValueError):
        model_paths.require_commit_sha(revision)


def pytest_fail(message: str):
    raise AssertionError(message)
