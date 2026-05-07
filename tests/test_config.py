from __future__ import annotations

import os

from core import config


def test_config_defaults_loaded_before_private_env(monkeypatch, tmp_path):
    private_path = tmp_path / ".env"
    private_path.write_text(
        "OPENAI_COMPATIBILITY_BASE_URL=https://private.example\nAPI_KEY=secret\n",
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
    monkeypatch.setattr(config, "PRIVATE_ENV_PATH", private_path)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBILITY_BASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    config.load_config()

    assert os.environ["LLM_MODEL_NAME"] == "from_config"
    assert os.environ["OPENAI_COMPATIBILITY_BASE_URL"] == "https://private.example"
    assert os.environ["API_KEY"] == "secret"


def test_existing_process_env_wins(monkeypatch, tmp_path):
    private_path = tmp_path / ".env"
    private_path.write_text("LLM_MODEL_NAME=from_private\n", encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_SETTINGS", {"LLM_MODEL_NAME": "from_config"})
    monkeypatch.setattr(config, "PRIVATE_ENV_PATH", private_path)
    monkeypatch.setenv("LLM_MODEL_NAME", "from_process")

    config.load_config()

    assert os.environ["LLM_MODEL_NAME"] == "from_process"


def test_empty_hf_endpoint_is_not_loaded_into_environment(monkeypatch, tmp_path):
    private_path = tmp_path / ".env"
    private_path.write_text("HF_ENDPOINT=\n", encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_SETTINGS", {"HF_ENDPOINT": ""})
    monkeypatch.setattr(config, "PRIVATE_ENV_PATH", private_path)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    config.load_config()

    assert "HF_ENDPOINT" not in os.environ


def test_public_asr_config_loads_generic_names(monkeypatch, tmp_path):
    private_path = tmp_path / ".env"
    private_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        config,
        "DEFAULT_SETTINGS",
        {
            "ASR_MODEL_PATH": "./models/asr",
            "ALIGNER_MODEL_PATH": "./models/aligner",
            "ASR_CONTEXT": "sample-name",
            "ALIGNER_BATCH_SIZE": "3",
        },
    )
    monkeypatch.setattr(config, "PRIVATE_ENV_PATH", private_path)
    for key in (
        "ASR_MODEL_PATH",
        "ALIGNER_MODEL_PATH",
        "ASR_CONTEXT",
        "ALIGNER_BATCH_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

    config.load_config()

    assert os.environ["ASR_MODEL_PATH"] == "./models/asr"
    assert os.environ["ALIGNER_MODEL_PATH"] == "./models/aligner"
    assert os.environ["ASR_CONTEXT"] == "sample-name"
    assert os.environ["ALIGNER_BATCH_SIZE"] == "3"


def test_non_python_shared_files_are_not_read(monkeypatch, tmp_path):
    private_path = tmp_path / ".env"
    private_path.write_text("", encoding="utf-8")
    for path in (tmp_path / "ignored_a.env", tmp_path / "ignored_b.env"):
        path.write_text("LLM_MODEL_NAME=ignored\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "DEFAULT_SETTINGS", {"LLM_MODEL_NAME": "from_config_py"})
    monkeypatch.setattr(config, "PRIVATE_ENV_PATH", private_path)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    config.load_config()

    assert os.environ["LLM_MODEL_NAME"] == "from_config_py"


def test_default_model_download_root_is_project_models():
    assert config.DEFAULT_SETTINGS["HF_HOME"] == "./models"
    assert config.DEFAULT_SETTINGS["TORCH_HOME"] == "./temp/torch"
    assert config.DEFAULT_SETTINGS["JOB_TEMP_DIR"] == "./temp/jobs"
    assert "KEEP_TEMP_FILES" not in config.DEFAULT_SETTINGS
    assert config.DEFAULT_SETTINGS["ASR_CONTEXT"] == ""
    assert config.DEFAULT_SETTINGS["ASR_BACKEND"] == "anime-whisper"
    assert config.DEFAULT_SETTINGS["ASR_SUBPROCESS_READY_TIMEOUT_S"] == "600"
    assert config.DEFAULT_SETTINGS["LLM_API_FORMAT"] == "chat"


def test_frozen_path_defaults_resolve_to_runtime_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "PRIVATE_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(config, "is_frozen", lambda: True)
    monkeypatch.setattr(
        config,
        "DEFAULT_SETTINGS",
        {
            "HF_HOME": "./models",
            "TORCH_HOME": "./temp/torch",
            "JOB_TEMP_DIR": "./temp/jobs",
            "LLM_MODEL_NAME": "from_config",
        },
    )
    for key in ("HF_HOME", "TORCH_HOME", "JOB_TEMP_DIR", "LLM_MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)

    config.load_config()

    assert os.environ["HF_HOME"] == str((tmp_path / "models").resolve())
    assert os.environ["TORCH_HOME"] == str((tmp_path / "temp" / "torch").resolve())
    assert os.environ["JOB_TEMP_DIR"] == str((tmp_path / "temp" / "jobs").resolve())
    assert os.environ["LLM_MODEL_NAME"] == "from_config"
