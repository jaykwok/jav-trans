from __future__ import annotations

import os
import re
from pathlib import Path

from asr.backends import qwen
from core import config


ROOT = Path(__file__).resolve().parents[1]


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
            "ASR_BATCH_SIZE": "3",
        },
    )
    monkeypatch.setattr(config, "PRIVATE_ENV_PATH", private_path)
    for key in (
        "ASR_MODEL_PATH",
        "ASR_BATCH_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

    config.load_config()

    assert os.environ["ASR_MODEL_PATH"] == "./models/asr"
    assert os.environ["ASR_BATCH_SIZE"] == "3"


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
    assert config.DEFAULT_SETTINGS["TORCH_HOME"] == "./tmp/cache/torch"
    assert config.DEFAULT_SETTINGS["JOB_TEMP_DIR"] == "./tmp/jobs"
    assert config.DEFAULT_SETTINGS["ASR_CHUNK_ROOT"] == "./tmp/chunks"
    assert "KEEP_TEMP_FILES" not in config.DEFAULT_SETTINGS
    assert "ASR_CONTEXT" not in config.DEFAULT_SETTINGS
    assert config.DEFAULT_SETTINGS["ASR_BACKEND"] == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    assert qwen.DEFAULT_QWEN_ASR_BACKEND == qwen.QWEN_ASR_17B_REPO_ID
    assert qwen.QWEN_ASR_REPO_ID == qwen.QWEN_ASR_17B_REPO_ID
    assert config.DEFAULT_SETTINGS["ASR_MODEL_ID"] == ""
    assert "ASR_STAGE_WORKER_MODE" not in config.DEFAULT_SETTINGS
    assert config.DEFAULT_SETTINGS["ASR_STAGE_WORKER_TIMEOUT_S"] == "0"
    assert config.DEFAULT_SETTINGS["ASR_STAGE_WORKER_OOM_RETRY_LIMIT"] == "3"
    assert config.DEFAULT_SETTINGS["ASR_STAGE_WORKER_VRAM_BUDGET_MB"] == "5600"
    assert config.DEFAULT_SETTINGS["ASR_BATCH_SIZE"] == "auto"
    assert "Qwen3-ASR-0.6B-JA-Anime-Galgame-hf=12" in config.DEFAULT_SETTINGS["ASR_BATCH_SIZE_BY_REPO"]
    assert "Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4" in config.DEFAULT_SETTINGS["ASR_BATCH_SIZE_BY_REPO"]
    assert "ASR_WORKER_MODE_BY_REPO" not in config.DEFAULT_SETTINGS
    assert config.DEFAULT_SETTINGS["SPEECH_BOUNDARY_JA_WINDOW_S"] == "20.0"
    assert config.DEFAULT_SETTINGS["SPEECH_BOUNDARY_JA_OVERLAP_S"] == "4.0"
    assert "BOUNDARY_REFINER_MODEL_PATH_BY_REPO" not in config.DEFAULT_SETTINGS
    assert "CUEQC_MODEL_PATH_BY_REPO" not in config.DEFAULT_SETTINGS
    assert "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO" in config.DEFAULT_SETTINGS
    for mapping in (
        qwen.DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO,
        qwen.DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
        qwen.DEFAULT_CUT_EDGE_REFINER_CHECKPOINT_BY_REPO,
        qwen.DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO,
        qwen.DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO,
    ):
        assert set(mapping) == {
            qwen.QWEN_ASR_06B_REPO_ID,
            qwen.QWEN_ASR_17B_REPO_ID,
        }
    assert qwen.QWEN_ASR_17B_REPO_ID in qwen.DEFAULT_CUEQC_CHECKPOINT_BY_REPO
    assert qwen.QWEN_ASR_06B_REPO_ID in qwen.DEFAULT_CUEQC_CHECKPOINT_BY_REPO
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
    assert config.DEFAULT_SETTINGS["PRE_ASR_CUEQC_ENABLED"] == "1"
    assert config.DEFAULT_SETTINGS["PRE_ASR_CUEQC_DROP_THRESHOLD"] == "0.95"
    assert config.DEFAULT_SETTINGS["LLM_API_FORMAT"] == "chat"


def test_asr_chunk_min_duration_removed_from_active_config_surface():
    active_files = (
        "src/asr/chunking.py",
        "src/boundary/cache.py",
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
        "src/asr/checkpoint.py",
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


def test_pre_asr_cueqc_old_versions_removed_from_active_runtime_surface():
    active_files = (
        "src/asr/pre_asr_cueqc.py",
        "src/asr/backends/qwen.py",
        "src/core/config.py",
        "README.md",
        "tools/workflows/run_full_workflow.py",
        "tools/asr/cueqc/compile_pre_asr_v11_features.py",
        "tools/asr/cueqc/train_pre_asr_v11_binary.py",
        "tools/asr/cueqc/export_pre_asr_v10_audit_candidates.py",
    )
    retired_tokens = (
        "cueqc_pre_asr_mamba_v6",
        "cueqc_pre_asr_mamba_v8",
        "cueqc_pre_asr_mamba_v9",
        "pre_asr_cueqc_features_v2",
        "pre_asr_cueqc_features_v4",
        "pre_asr_cueqc_features_v5",
        "compile_pre_asr_v6",
        "compile_pre_asr_v8",
        "compile_pre_asr_v9",
        "train_pre_asr_v6",
        "train_pre_asr_v8",
        "train_pre_asr_v9",
        "export_pre_asr_v6",
        "export_pre_asr_v8",
        "export_pre_asr_v9",
        "Pre-ASR CueQC v6",
        "Pre-ASR CueQC v8",
        "Pre-ASR CueQC v9",
    )
    for relative_path in active_files:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for token in retired_tokens:
            assert token not in text
    assert qwen.QWEN_ASR_17B_REPO_ID in qwen.DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO


def test_asr_after_cueqc_removed_from_active_runtime_surface():
    active_files = (
        "src/asr/pipeline.py",
        "src/asr/checkpoint.py",
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
    monkeypatch.setattr(config, "PRIVATE_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(config, "is_frozen", lambda: True)
    monkeypatch.setattr(
        config,
        "DEFAULT_SETTINGS",
        {
            "HF_HOME": "./models",
            "TORCH_HOME": "./tmp/cache/torch",
            "JOB_TEMP_DIR": "./tmp/jobs",
            "LLM_MODEL_NAME": "from_config",
        },
    )
    for key in ("HF_HOME", "TORCH_HOME", "JOB_TEMP_DIR", "LLM_MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)

    config.load_config()

    assert os.environ["HF_HOME"] == str((tmp_path / "models").resolve())
    assert os.environ["TORCH_HOME"] == str((tmp_path / "tmp" / "cache" / "torch").resolve())
    assert os.environ["JOB_TEMP_DIR"] == str((tmp_path / "tmp" / "jobs").resolve())
    assert os.environ["LLM_MODEL_NAME"] == "from_config"
