from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

from asr.backends import qwen
from core.job_context import JobContext
from core import config


ROOT = Path(__file__).resolve().parents[1]


def test_config_defaults_loaded_before_private_config(monkeypatch, tmp_path):
    private_path = tmp_path / "config" / "config.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps(
            {
                "OPENAI_COMPATIBILITY_BASE_URL": "https://private.example",
                "API_KEY": "secret",
                "LLAMACPP_CTX_SIZE": 6144,
                "AUDIO_DYNAUDNORM": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        config,
        "DEFAULT_SETTINGS",
        {
            "LLM_MODEL_NAME": "from_config",
            "OPENAI_COMPATIBILITY_BASE_URL": "https://config.example",
        },
    )
    monkeypatch.setattr(config, "CONFIG_PATH", private_path)
    for key in (
        "LLM_MODEL_NAME",
        "OPENAI_COMPATIBILITY_BASE_URL",
        "API_KEY",
        "LLAMACPP_CTX_SIZE",
        "AUDIO_DYNAUDNORM",
    ):
        monkeypatch.delenv(key, raising=False)

    config.load_config()

    assert os.environ["LLM_MODEL_NAME"] == "from_config"
    assert os.environ["OPENAI_COMPATIBILITY_BASE_URL"] == "https://private.example"
    assert os.environ["API_KEY"] == "secret"
    assert os.environ["LLAMACPP_CTX_SIZE"] == "6144"
    assert os.environ["AUDIO_DYNAUDNORM"] == "0"


def test_existing_process_env_wins(monkeypatch, tmp_path):
    private_path = tmp_path / "config" / "config.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps({"LLM_MODEL_NAME": "from_private"}), encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_SETTINGS", {"LLM_MODEL_NAME": "from_config"})
    monkeypatch.setattr(config, "CONFIG_PATH", private_path)
    monkeypatch.setenv("LLM_MODEL_NAME", "from_process")

    config.load_config()

    assert os.environ["LLM_MODEL_NAME"] == "from_process"


def test_empty_hf_endpoint_is_not_loaded_into_environment(monkeypatch, tmp_path):
    private_path = tmp_path / "config" / "config.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps({"HF_ENDPOINT": ""}), encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_SETTINGS", {"HF_ENDPOINT": ""})
    monkeypatch.setattr(config, "CONFIG_PATH", private_path)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    config.load_config()

    assert "HF_ENDPOINT" not in os.environ


def test_public_asr_config_loads_generic_names(monkeypatch, tmp_path):
    private_path = tmp_path / "config" / "config.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        config,
        "DEFAULT_SETTINGS",
        {
            "ASR_MODEL_PATH": "./models/asr",
            "ASR_BATCH_SIZE": "3",
        },
    )
    monkeypatch.setattr(config, "CONFIG_PATH", private_path)
    for key in (
        "ASR_MODEL_PATH",
        "ASR_BATCH_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

    config.load_config()

    assert os.environ["ASR_MODEL_PATH"] == "./models/asr"
    assert os.environ["ASR_BATCH_SIZE"] == "3"


def test_legacy_env_and_unrelated_files_are_not_read(monkeypatch, tmp_path):
    private_path = tmp_path / "config" / "config.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text("{}", encoding="utf-8")
    for path in (tmp_path / ".env", tmp_path / "ignored_a.env", tmp_path / "ignored_b.env"):
        path.write_text("LLM_MODEL_NAME=ignored\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "DEFAULT_SETTINGS", {"LLM_MODEL_NAME": "from_config_py"})
    monkeypatch.setattr(config, "CONFIG_PATH", private_path)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    config.load_config()

    assert os.environ["LLM_MODEL_NAME"] == "from_config_py"


def test_default_model_download_root_is_project_models():
    assert config.DEFAULT_SETTINGS["HF_HOME"] == "./models"
    assert config.DEFAULT_SETTINGS["TORCH_HOME"] == "./cache/torch"
    assert config.DEFAULT_SETTINGS["ASR_RESULT_CACHE_ROOT"] == "./cache/asr"
    assert config.DEFAULT_SETTINGS["JOB_TEMP_DIR"] == "./tmp/jobs"
    assert config.DEFAULT_SETTINGS["ASR_CHUNK_ROOT"] == "./tmp/chunks"
    assert config.DEFAULT_SETTINGS["RUN_LOG_ENABLED"] == "1"
    assert config.DEFAULT_SETTINGS["RUN_LOG_DIR"] == "./logs/jobs"
    assert (
        config.DEFAULT_SETTINGS["OPENAI_COMPATIBILITY_BASE_URL"]
        == "https://openrouter.ai/api/v1"
    )
    assert config.DEFAULT_SETTINGS["LLM_REASONING_EFFORT"] == "low"
    assert "KEEP_TEMP_FILES" not in config.DEFAULT_SETTINGS
    assert "ASR_CONTEXT" not in config.DEFAULT_SETTINGS
    assert config.DEFAULT_SETTINGS["ASR_BACKEND"] == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    assert qwen.DEFAULT_QWEN_ASR_BACKEND == qwen.QWEN_ASR_17B_REPO_ID
    assert qwen.QWEN_ASR_REPO_ID == qwen.QWEN_ASR_17B_REPO_ID
    assert config.DEFAULT_SETTINGS["ASR_MODEL_ID"] == ""
    assert "ASR_STAGE_WORKER_MODE" not in config.DEFAULT_SETTINGS
    assert config.DEFAULT_SETTINGS["ASR_STAGE_WORKER_TIMEOUT_S"] == "0"
    assert config.DEFAULT_SETTINGS["ASR_STAGE_WORKER_OOM_RETRY_LIMIT"] == "6"
    assert config.DEFAULT_SETTINGS["ASR_STAGE_WORKER_VRAM_BUDGET_MB"] == "auto"
    assert config.DEFAULT_SETTINGS["ASR_STAGE_WORKER_VRAM_RATIO"] == "0.95"
    minimum_vram_table = config.DEFAULT_SETTINGS["ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO"]
    assert "Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=7168" in minimum_vram_table
    # The 0.6B tier was dropped on 2026-07-31; a leftover row would
    # advertise a model that no longer ships.
    assert "0.6B" not in minimum_vram_table
    assert config.DEFAULT_SETTINGS["ASR_STAGE_WORKER_HEARTBEAT_S"] == "10"
    assert config.DEFAULT_SETTINGS["ASR_BATCH_SIZE"] == "auto"
    assert "ASR_BATCH_SIZE_BY_REPO" not in config.DEFAULT_SETTINGS
    assert "GPU_BATCH_PROFILE_ENABLED" not in config.DEFAULT_SETTINGS
    assert "GPU_BATCH_PROFILE_GROWTH_THRESHOLD" not in config.DEFAULT_SETTINGS
    assert "GPU_BATCH_PROFILE_MAX_ENTRIES" not in config.DEFAULT_SETTINGS
    assert "GPU_BATCH_PROFILE_PATH" not in config.DEFAULT_SETTINGS
    assert "ASR_WORKER_MODE_BY_REPO" not in config.DEFAULT_SETTINGS
    assert config.DEFAULT_SETTINGS["ASR_CHUNK_MAX_S"] == "30.0"
    assert config.DEFAULT_SETTINGS["ASR_CHUNK_MIN_PAUSE_S"] == "0.6"
    # The retired chain must not leave settings behind: an advertised knob that
    # nothing reads is worse than a missing one, because it looks adjustable.
    for retired in (
        "ASR_BOUNDARY_BACKEND",
        "BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        "CUEQC_MODEL_PATH_BY_REPO",
        "SPEECH_BOUNDARY_JA_WINDOW_S",
        "SPEECH_BOUNDARY_JA_OVERLAP_S",
        "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
        "OUTER_EDGE_REFINER_DEVICE",
        "SEMANTIC_SPLIT_DEVICE",
        "INNER_EDGE_REFINER_DEVICE",
        "ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES",
        "BOUNDARY_CACHE_ENABLED",
        "BOUNDARY_CACHE_DIR",
        "BOUNDARY_FEATURE_FRAME_HOP_S",
        "PRE_ASR_CUEQC_ENABLED",
        # Chunks run to ASR_CHUNK_MAX_S now. A separate target length only ever
        # took effect once the alignment head was configured, and then it cut the
        # decode window by a third without saying so.
        "ASR_CHUNK_TARGET_S",
    ):
        assert retired not in config.DEFAULT_SETTINGS
    assert not hasattr(qwen, "QWEN_ASR_06B_REPO_ID")
    assert "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT" not in config.DEFAULT_SETTINGS
    assert not any(key.startswith("ALIGN") for key in config.DEFAULT_SETTINGS)
    assert "BOUNDARY_REFINER_ENABLED" not in config.DEFAULT_SETTINGS
    assert "BOUNDARY_PLANNER_TARGET_CHUNK_S" not in config.DEFAULT_SETTINGS
    assert "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S" not in config.DEFAULT_SETTINGS
    assert "BOUNDARY_PLANNER_MIN_CHUNK_S" not in config.DEFAULT_SETTINGS
    assert "BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT" not in config.DEFAULT_SETTINGS
    assert "CUEQC_MODEL_VERSION" not in config.DEFAULT_SETTINGS
    assert "CUEQC_DECISION_VERSION" not in config.DEFAULT_SETTINGS
    assert "BOUNDARY_CONTEXT_MAX_PADDING_S" not in config.DEFAULT_SETTINGS
    assert "BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S" not in config.DEFAULT_SETTINGS
    assert "BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S" not in config.DEFAULT_SETTINGS
    assert "MAX_SUBTITLE_DURATION" not in config.DEFAULT_SETTINGS
    assert "ASR_SUBPROCESS_READY_TIMEOUT_S" not in config.DEFAULT_SETTINGS
    assert "CUEQC_SHADOW_ENABLED" not in config.DEFAULT_SETTINGS
    assert "CUEQC_DROP_THRESHOLD" not in config.DEFAULT_SETTINGS
    assert "CUEQC_DEVICE" not in config.DEFAULT_SETTINGS
    assert "CUEQC_EXPORT_CANDIDATES_PATH" not in config.DEFAULT_SETTINGS
    assert "CUEQC_EXPORT_CANDIDATES_APPEND" not in config.DEFAULT_SETTINGS
    assert "CUEQC_SHADOW_EMBED_CANDIDATES" not in config.DEFAULT_SETTINGS


def test_boundary_registries_are_gone_after_retirement() -> None:
    """The chain was retired on 2026-07-31 and its tools left with it.

    The empty per-repo registries were kept briefly for the offline tools, but
    those tools moved to dev/rm/ too - so the registries and the machinery
    that resolved through them must not exist at all.
    """
    for name in (
        "DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO",
        "DEFAULT_SPEECH_BOUNDARY_PROPOSAL_CHECKPOINT_BY_REPO",
        "DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO",
        "DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO",
        "DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO",
        "DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO",
        "BOUNDARY_PIPELINE_STATUS_BY_REPO",
    ):
        assert not hasattr(qwen, name), name


def test_job_context_defaults_runtime_logs_from_config(monkeypatch, tmp_path):
    monkeypatch.delenv("RUN_LOG_ENABLED", raising=False)
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)

    ctx = JobContext.from_spec(
        SimpleNamespace(),
        job_id="job-1",
        job_temp_dir=str(tmp_path / "jobs" / "job-1"),
        cache_path=str(tmp_path / "jobs" / "job-1" / "translation_cache.jsonl"),
    )

    assert ctx.run_log_enabled is True
    assert ctx.run_log_dir == "./logs/jobs"

    monkeypatch.setenv("RUN_LOG_ENABLED", "0")
    monkeypatch.setenv("RUN_LOG_DIR", "./custom-log")
    disabled = JobContext.from_spec(
        SimpleNamespace(),
        job_id="job-2",
        job_temp_dir=str(tmp_path / "jobs" / "job-2"),
        cache_path=str(tmp_path / "jobs" / "job-2" / "translation_cache.jsonl"),
    )

    assert disabled.run_log_enabled is False
    assert disabled.run_log_dir == "./custom-log"


def test_asr_chunk_min_duration_removed_from_active_config_surface():
    active_files = (
        "src/asr/chunking.py",
        "src/asr/chunking.py",
        "src/core/config.py",
        "src/main.py",
        "README.md",
    )
    for relative_path in active_files:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "ASR_CHUNK_MIN_DURATION_S" not in text
    assert "ASR_CHUNK_MIN_DURATION_S" not in config.DEFAULT_SETTINGS


def test_asr_sliding_text_context_removed_from_active_config_surface():
    active_files = (
        "src/asr/transcribe.py",
        "src/asr/local_backend.py",
        "src/asr/backends/base.py",
        "src/asr/result_cache.py",
        "src/asr/pipeline.py",
        "src/core/config.py",
        "src/main.py",
        "README.md",
    )
    removed_terms = (
        "ASR_SLIDING_CONTEXT_SEGS",
        "ASR_CONTEXT_RESET_GAP_S",
        "ASR_INITIAL_PROMPT_MAX_CHARS",
        "initial_prompts",
        "_build_initial_prompt_for_chunk",
    )
    for relative_path in active_files:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for term in removed_terms:
            assert term not in text
    for term in removed_terms[:3]:
        assert term not in config.DEFAULT_SETTINGS


def test_subtitle_max_duration_clamp_removed_from_active_config_surface():
    active_files = (
        "src/core/config.py",
        "src/main.py",
        "src/subtitles/options.py",
        "src/subtitles/writer.py",
        "README.md",
    )
    for relative_path in active_files:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "MAX_SUBTITLE_DURATION" not in text
        assert "max_duration" not in text


def test_pre_asr_cueqc_modules_are_retired():
    # The pre-ASR CueQC gate and its training tools moved to dev/rm/ on
    # 2026-07-31; a resurrected module path would mean an accidental restore.
    for relative_path in (
        "src/asr/pre_asr_cueqc.py",
        "tools/asr/cueqc",
        "src/boundary",
        "src/audio/chunk_packer.py",
        "tools/boundary",
    ):
        assert not (ROOT / relative_path).exists(), relative_path


def test_asr_after_cueqc_removed_from_active_runtime_surface():
    active_files = (
        "src/asr/pipeline.py",
        "src/asr/result_cache.py",
        "src/core/config.py",
        "tools/workflows/run_full_workflow.py",
        "tools/web/smoke/submit_job.py",
        "README.md",
    )
    retired_patterns = (
        r"(?<!PRE_ASR_)CUEQC_SHADOW_ENABLED",
        r"(?<!PRE_ASR_)CUEQC_MODEL_PATH_BY_REPO",
        r"(?<!PRE_ASR_)CUEQC_INFERENCE_BATCH_SIZE",
        r"(?<!PRE_ASR_)CUEQC_EXPORT_CANDIDATES_PATH",
        r"(?<!PRE_ASR_)CUEQC_SHADOW_EMBED_CANDIDATES",
        r"--cueqc-shadow-enabled",
        r"--cueqc-model-path-by-repo",
        r"--cueqc-inference-batch-size",
        r"cueqc_shadow",
    )
    for relative_path in active_files:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for pattern in retired_patterns:
            assert not re.search(pattern, text), f"{pattern} remains in {relative_path}"


def test_frozen_path_defaults_resolve_to_runtime_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config" / "config.json")
    monkeypatch.setattr(config, "is_frozen", lambda: True)
    monkeypatch.setattr(
        config,
        "DEFAULT_SETTINGS",
        {
            "HF_HOME": "./models",
            "TORCH_HOME": "./cache/torch",
            "ASR_RESULT_CACHE_ROOT": "./cache/asr",
            "JOB_TEMP_DIR": "./tmp/jobs",
            "RUN_LOG_DIR": "./logs/jobs",
            "LLM_MODEL_NAME": "from_config",
        },
    )
    for key in (
        "HF_HOME",
        "TORCH_HOME",
        "ASR_RESULT_CACHE_ROOT",
        "JOB_TEMP_DIR",
        "RUN_LOG_DIR",
        "LLM_MODEL_NAME",
    ):
        monkeypatch.delenv(key, raising=False)

    config.load_config()

    assert os.environ["HF_HOME"] == str((tmp_path / "models").resolve())
    assert os.environ["TORCH_HOME"] == str((tmp_path / "cache" / "torch").resolve())
    assert os.environ["ASR_RESULT_CACHE_ROOT"] == str((tmp_path / "cache" / "asr").resolve())
    assert os.environ["JOB_TEMP_DIR"] == str((tmp_path / "tmp" / "jobs").resolve())
    assert os.environ["RUN_LOG_DIR"] == str((tmp_path / "logs" / "jobs").resolve())
    assert os.environ["LLM_MODEL_NAME"] == "from_config"


class TestProxyExemptsLoopback:
    """A proxy is configured so model downloads can leave the machine. The
    llamacpp backend, meanwhile, talks to a llama-server it started itself on
    127.0.0.1 -- and neither urllib nor httpx exempts loopback on its own, so
    without NO_PROXY the same setting that fixes downloads breaks local
    translation.
    """

    @staticmethod
    def _clear(monkeypatch):
        for key in config._PROXY_ENV_KEYS + config._NO_PROXY_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

    def test_setting_a_proxy_exempts_loopback(self, monkeypatch):
        self._clear(monkeypatch)
        config.apply_network_proxy_environment("http://127.0.0.1:7890")
        for key in config._NO_PROXY_ENV_KEYS:
            entries = os.environ[key].split(",")
            assert "127.0.0.1" in entries
            assert "localhost" in entries
            assert "::1" in entries

    def test_urllib_and_httpx_agree_that_loopback_is_direct(self, monkeypatch):
        """The two clients that actually reach the local server: `_health_ok`
        uses urllib, the OpenAI SDK uses httpx."""
        import urllib.request

        import httpx

        self._clear(monkeypatch)
        config.apply_network_proxy_environment("http://127.0.0.1:7890")

        assert urllib.request.proxy_bypass("127.0.0.1") is True
        assert not urllib.request.proxy_bypass("huggingface.co")

        client = httpx.Client(trust_env=True)
        try:
            local = client._transport_for_url(httpx.URL("http://127.0.0.1:8080/v1"))
            remote = client._transport_for_url(httpx.URL("https://huggingface.co/x"))
            assert getattr(local._pool, "_proxy_url", None) is None
            assert getattr(remote._pool, "_proxy_url", None) is not None
        finally:
            client.close()

    def test_a_user_supplied_no_proxy_is_kept(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("NO_PROXY", "internal.example,10.0.0.1")
        config.apply_network_proxy_environment("http://127.0.0.1:7890")
        entries = os.environ["NO_PROXY"].split(",")
        assert entries[:2] == ["internal.example", "10.0.0.1"]
        assert "127.0.0.1" in entries

    def test_loopback_is_not_added_twice(self, monkeypatch):
        self._clear(monkeypatch)
        config.apply_network_proxy_environment("http://127.0.0.1:7890")
        config.apply_network_proxy_environment("http://127.0.0.1:7890")
        assert os.environ["NO_PROXY"].split(",").count("127.0.0.1") == 1

    def test_a_wildcard_bypass_is_left_alone(self, monkeypatch):
        """`*` already means "never proxy"; appending hosts to it would only
        make the value harder to read."""
        self._clear(monkeypatch)
        monkeypatch.setenv("NO_PROXY", "*")
        config.apply_network_proxy_environment("http://127.0.0.1:7890")
        assert os.environ["NO_PROXY"] == "*"

    def test_no_proxy_configured_touches_nothing(self, monkeypatch):
        self._clear(monkeypatch)
        config.apply_network_proxy_environment("")
        assert "NO_PROXY" not in os.environ
        assert "HTTP_PROXY" not in os.environ


def test_update_config_file_uses_stringify_and_rejects_non_scalars(tmp_path):
    import pytest

    cfg_path = tmp_path / "config" / "config.json"
    config.update_config_file(
        {"AUDIO_DYNAUDNORM": False, "RUN_LOG_ENABLED": True, "PROXY_PORT": 7890, "EMPTY": None},
        path=cfg_path,
    )
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved == {
        "AUDIO_DYNAUDNORM": "0",
        "RUN_LOG_ENABLED": "1",
        "PROXY_PORT": "7890",
        "EMPTY": "",
    }

    with pytest.raises(ValueError, match="non-scalar"):
        config.update_config_file({"BAD_LIST": [1, 2, 3]}, path=cfg_path)
    with pytest.raises(ValueError, match="non-scalar"):
        config.update_config_file({"BAD_DICT": {"a": 1}}, path=cfg_path)
    # Existing file stays untouched after rejected non-scalar writes.
    assert json.loads(cfg_path.read_text(encoding="utf-8")) == saved


def test_network_proxy_url_from_env_validates_host_port_and_ipv6(monkeypatch):
    from core import typed_config

    typed_config.clear_configuration_problems()
    try:
        monkeypatch.setenv("PROXY_PROTOCOL", "http")
        # Bare IPv6 gets bracketed automatically.
        monkeypatch.setenv("PROXY_HOST", "::1")
        monkeypatch.setenv("PROXY_PORT", "7890")
        assert config.network_proxy_url_from_env() == "http://[::1]:7890"
        assert typed_config.configuration_problems() == []

        # Bracketed IPv6 and userinfo@IPv6 also work cleanly.
        monkeypatch.setenv("PROXY_HOST", "[2001:db8::1]")
        assert config.network_proxy_url_from_env() == "http://[2001:db8::1]:7890"
        monkeypatch.setenv("PROXY_HOST", "user:pass@::1")
        assert config.network_proxy_url_from_env() == "http://user:pass@[::1]:7890"

        # Invalid ports are rejected and recorded.
        monkeypatch.setenv("PROXY_HOST", "127.0.0.1")
        for bad_port in ("0", "65536", "-1", "abc", "7890.5"):
            monkeypatch.setenv("PROXY_PORT", bad_port)
            assert config.network_proxy_url_from_env() == ""
            assert any(p.startswith("PROXY_PORT=") for p in typed_config.configuration_problems())

        # Invalid hosts are rejected and recorded (with credentials redacted).
        monkeypatch.setenv("PROXY_PORT", "7890")
        for bad_host in (
            "http://127.0.0.1",
            "127.0.0.1/path",
            "127.0.0.1?a=1",
            "127.0.0.1#frag",
            "127.0.0.1 8080",
            "127.0.0.1:7890",
            "secret:pass@bad/host",
        ):
            monkeypatch.setenv("PROXY_HOST", bad_host)
            assert config.network_proxy_url_from_env() == ""
            problems = typed_config.configuration_problems()
            assert any(p.startswith("PROXY_HOST=") for p in problems)
            assert "secret" not in " ".join(problems)

        # Fixing both clears all proxy problems.
        monkeypatch.setenv("PROXY_HOST", "127.0.0.1")
        monkeypatch.setenv("PROXY_PORT", "7890")
        assert config.network_proxy_url_from_env() == "http://127.0.0.1:7890"
        assert typed_config.configuration_problems() == []
    finally:
        typed_config.clear_configuration_problems()


def test_frozen_private_config_normalizes_frozen_path_keys(monkeypatch, tmp_path):
    private_path = tmp_path / "config" / "config.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps({"HF_HOME": "./custom_models", "RUN_LOG_DIR": "./custom_logs"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", private_path)
    monkeypatch.setattr(config, "is_frozen", lambda: True)
    monkeypatch.setattr(config, "DEFAULT_SETTINGS", {"HF_HOME": "./models", "RUN_LOG_DIR": "./logs/jobs"})
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)

    config.load_config()

    assert os.environ["HF_HOME"] == str((tmp_path / "custom_models").resolve())
    assert os.environ["RUN_LOG_DIR"] == str((tmp_path / "custom_logs").resolve())


def test_config_load_error_warning_is_deduplicated_across_repeated_loads(monkeypatch, tmp_path, capsys):
    bad_path = tmp_path / "config" / "config.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", bad_path)
    monkeypatch.setattr(config, "DEFAULT_SETTINGS", {})
    monkeypatch.setattr(config, "_CONFIG_LOAD_ERROR", "")
    monkeypatch.setattr(config, "_LAST_REPORTED_CONFIG_LOAD_ERROR", "")

    config.load_config()
    first_out = capsys.readouterr().out
    assert "[WARN]" in first_out
    assert config.config_load_error()

    # Second load with same error must not spam warnings again.
    config.load_config()
    second_out = capsys.readouterr().out
    assert second_out == ""
    assert config.config_load_error()


def test_job_context_flag_and_none_workers(monkeypatch, tmp_path):
    from core import job_context, typed_config

    typed_config.clear_configuration_problems()
    assert job_context._flag("", True) is True
    assert job_context._flag("   ", True) is True
    assert job_context._flag("0", True) is False
    assert job_context._flag("invalid", True) is True
    # _flag uses explicit environ dict, so global typed_config problems stay untouched.
    assert typed_config.configuration_problems() == []

    monkeypatch.delenv("RUN_LOG_ENABLED", raising=False)
    ctx = JobContext.from_spec(
        SimpleNamespace(translation_max_workers=None, run_log_enabled=""),
        job_id="job-none",
        job_temp_dir=str(tmp_path),
        cache_path=str(tmp_path / "cache.jsonl"),
    )
    assert ctx.translation_max_workers == 4
    assert ctx.run_log_enabled is True
