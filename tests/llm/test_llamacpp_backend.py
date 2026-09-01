"""Managed llama.cpp server backend.

Split out of the retired Sakura test file, which is where the llamacpp coverage
happened to live. The schema tests are new and come from a measurement: on
Hy-MT2-1.8B-Q8_0 (2026-08-04) this backend's free-form output was not parseable
JSON in 3/3 attempts - the model echoed the prompt back - while the same server
with the schema attached answered every id in 3/3 and did it 2.5-4x faster.
llama-server compiles the schema into a GBNF grammar, so this is the difference
between a local model being usable and not.
"""

from __future__ import annotations

import os

import pytest

from llm.backends import list_backends
from llm.backends import llamacpp_server
from llm.backends.llamacpp_server import (
    LlamaCppServerBackend,
    _wrap_response_format,
    cuda_library_dirs,
    probe_compute_devices,
    resolve_gguf_model_path,
    resolve_server_executable,
    server_environment,
)

_ENV_KEYS = (
    "LLAMACPP_MODEL_FILE",
    "LLAMACPP_MODEL_REPO",
    "LLAMACPP_GGUF_PATH",
    "LLAMACPP_SERVER_PATH",
    "LLAMACPP_CTX_SIZE",
    "LLAMACPP_PARALLEL",
    "TRANSLATION_BACKEND",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


def test_llamacpp_backend_registered():
    assert "llamacpp" in list_backends()


# --- locating the server and the model ---------------------------------------


def test_resolve_server_executable_missing_is_actionable(monkeypatch):
    monkeypatch.setattr("llm.backends.llamacpp_server.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        resolve_server_executable()
    message = str(excinfo.value)
    # The exact package id, not a name to guess at.
    assert "winget install -e --id ggml.llamacpp" in message
    assert "releases" in message


def test_resolve_server_executable_accepts_directory(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("LLAMACPP_SERVER_PATH", str(tmp_path))
    assert resolve_server_executable() == str(exe)


def test_resolve_server_executable_bad_explicit_path(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMACPP_SERVER_PATH", str(tmp_path / "nope.exe"))
    with pytest.raises(RuntimeError):
        resolve_server_executable()


def test_resolve_gguf_explicit_path(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    monkeypatch.setenv("LLAMACPP_GGUF_PATH", str(model))
    assert resolve_gguf_model_path() == str(model)


def test_resolve_gguf_missing_config_is_actionable(monkeypatch):
    with pytest.raises(RuntimeError) as excinfo:
        resolve_gguf_model_path()
    assert "GGUF" in str(excinfo.value)


def test_build_command_shape(monkeypatch):
    monkeypatch.setenv("LLAMACPP_CTX_SIZE", "4096")
    monkeypatch.setenv("LLAMACPP_PARALLEL", "2")
    backend = LlamaCppServerBackend()
    command = backend._build_command("llama-server.exe", "D:\\m.gguf", 12345)
    assert command[0] == "llama-server.exe"
    assert command[command.index("-m") + 1] == "D:\\m.gguf"
    assert command[command.index("--port") + 1] == "12345"
    assert command[command.index("-c") + 1] == "8192"  # ctx * parallel
    assert command[command.index("-np") + 1] == "2"
    assert "--no-webui" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_build_command_defaults_to_eight_slots_for_the_7b_q4_model():
    command = LlamaCppServerBackend()._build_command(
        "llama-server.exe", "D:\\m.gguf", 12345
    )
    assert command[command.index("-np") + 1] == "8"
    assert command[command.index("-c") + 1] == "8192"


def test_cache_identity_reflects_model(monkeypatch):
    backend = LlamaCppServerBackend()
    monkeypatch.setenv("LLAMACPP_MODEL_REPO", "tencent/Hy-MT2-1.8B-GGUF")
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "Hy-MT2-1.8B-Q8_0.gguf")
    assert (
        backend.cache_identity()
        == "llamacpp:tencent/Hy-MT2-1.8B-GGUF/Hy-MT2-1.8B-Q8_0.gguf"
    )
    monkeypatch.setenv("LLAMACPP_GGUF_PATH", "D:\\models\\custom-q4.gguf")
    assert backend.cache_identity() == "llamacpp:custom-q4.gguf"


# --- structured output --------------------------------------------------------


def test_the_backend_advertises_schema_support():
    """The engine only sends a schema to a backend that claims it, so this flag
    is what decides whether local decoding is grammar-constrained at all."""
    assert LlamaCppServerBackend().supports_json_schema() is True


def test_a_bare_schema_is_wrapped_for_the_server():
    schema = {"type": "object", "properties": {"translations": {"type": "array"}}}
    wrapped = _wrap_response_format(schema)
    assert wrapped["type"] == "json_schema"
    assert wrapped["json_schema"]["schema"] is schema
    assert wrapped["json_schema"]["strict"] is True


def test_an_already_wrapped_format_is_left_alone():
    value = {"type": "json_object"}
    assert _wrap_response_format(value) is value


def test_no_schema_means_no_response_format_key():
    assert _wrap_response_format(None) is None
    assert _wrap_response_format({}) is None


def test_the_schema_reaches_the_request(monkeypatch):
    """The regression this file exists for: the old code did
    `del response_format` and silently dropped it."""
    sent: dict = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            sent.update(kwargs)

            class _Message:
                content = '{"translations": []}'

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]
                usage = None

            return _Response()

    class _FakeClient:
        class chat:  # noqa: N801 - mirrors the openai client shape
            completions = _FakeCompletions()

    backend = LlamaCppServerBackend()
    monkeypatch.setattr(backend, "_ensure_server", lambda cancel_event=None: None)
    backend._client = _FakeClient()

    schema = {"type": "object", "properties": {"translations": {"type": "array"}}}
    backend.chat_completion([{"role": "user", "content": "x"}], response_format=schema)

    assert sent["response_format"]["json_schema"]["schema"] is schema


# --- getting the CUDA backend to load at all -----------------------------------


def test_the_torch_cuda_libraries_are_offered_to_the_server(monkeypatch, tmp_path):
    """Regression for a silent 40x slowdown: the CUDA llama.cpp zip has no CUDA
    runtime in it, so without cuBLAS on PATH ggml-cuda.dll never registers and
    the server runs on the CPU without saying so."""
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents=True)
    (lib / "cublas64_13.dll").write_bytes(b"")
    monkeypatch.setattr(llamacpp_server, "_torch_library_dir", lambda: lib)

    assert cuda_library_dirs() == (str(lib),)
    monkeypatch.setenv("PATH", "C:\\existing")
    path = server_environment()["PATH"]
    assert path.split(os.pathsep)[0] == str(lib)
    assert "C:\\existing" in path


def test_a_torch_without_cublas_adds_nothing(monkeypatch, tmp_path):
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setattr(llamacpp_server, "_torch_library_dir", lambda: lib)
    assert cuda_library_dirs() == ()
    monkeypatch.setenv("PATH", "C:\\existing")
    assert server_environment()["PATH"] == "C:\\existing"


def test_a_missing_torch_adds_nothing(monkeypatch):
    monkeypatch.setattr(llamacpp_server, "_torch_library_dir", lambda: None)
    assert cuda_library_dirs() == ()


def test_only_accelerators_count_as_devices(monkeypatch):
    """`--list-devices` prints a header and, with no GPU backend loaded, the
    literal `(none)`; neither is a device."""

    class _Result:
        stdout = (
            "Available devices:\n"
            "  CUDA0: NVIDIA GeForce RTX 4060 Ti (8187 MiB, 7075 MiB free)\n"
        )

    monkeypatch.setattr(llamacpp_server.subprocess, "run", lambda *a, **k: _Result())
    assert probe_compute_devices("llama-server.exe") == [
        "CUDA0: NVIDIA GeForce RTX 4060 Ti (8187 MiB, 7075 MiB free)"
    ]


def test_no_devices_when_the_cuda_backend_failed_to_load(monkeypatch):
    class _Result:
        stdout = "Available devices:\n  (none)\n"

    monkeypatch.setattr(llamacpp_server.subprocess, "run", lambda *a, **k: _Result())
    assert probe_compute_devices("llama-server.exe") == []


def test_an_unlaunchable_binary_reports_no_devices(monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr(llamacpp_server.subprocess, "run", _boom)
    assert probe_compute_devices("llama-server.exe") == []


def test_without_a_schema_the_request_stays_plain(monkeypatch):
    sent: dict = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            sent.update(kwargs)

            class _Message:
                content = "hi"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]
                usage = None

            return _Response()

    class _FakeClient:
        class chat:  # noqa: N801
            completions = _FakeCompletions()

    backend = LlamaCppServerBackend()
    monkeypatch.setattr(backend, "_ensure_server", lambda cancel_event=None: None)
    backend._client = _FakeClient()
    backend.chat_completion([{"role": "user", "content": "x"}])
    assert "response_format" not in sent
