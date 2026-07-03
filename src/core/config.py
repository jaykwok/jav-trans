import os
from pathlib import Path

from dotenv import dotenv_values

from utils.runtime_paths import is_frozen, runtime_root

PROJECT_ROOT = runtime_root()
PRIVATE_ENV_PATH = PROJECT_ROOT / ".env"
_FROZEN_PATH_KEYS = {
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "TORCH_HOME",
    "JOB_TEMP_DIR",
    "ASR_CHUNK_ROOT",
    "RUN_LOG_DIR",
    "QUALITY_REPORT_DIR",
}


# Runtime configuration source of truth.
#
# Values are strings because they are copied into os.environ for modules that
# read configuration at import time. Keep shared defaults here and put local
# machine/API overrides in .env. Web task options are carried by JobContext,
# not process-wide environment values.
DEFAULT_SETTINGS: dict[str, str] = {
    # --- HuggingFace Path ---
    # Local HuggingFace/model cache path. Relative paths are resolved from project root after loading.
    "HF_HOME": "./models",
    # torch.hub runtime cache; not a model directory and not part of models/.
    "TORCH_HOME": "./tmp/cache/torch",
    # HuggingFace Hub endpoint. Set to https://hf-mirror.com for mainland China acceleration.
    # Empty string means use the default huggingface.co. Takes effect on next app start.
    "HF_ENDPOINT": "",

    # --- ASR Model Settings ---
    # Transcription backend. Use the Hugging Face repo id as the stable key.
    "ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    # Voice activity detection backend used before ASR chunking.
    "ASR_BOUNDARY_BACKEND": "speech_boundary_ja",
    # Optional explicit HuggingFace model id override. Empty auto-selects by ASR_BACKEND.
    "ASR_MODEL_ID": "",
    # Optional local ASR model directory override. Empty uses models/<namespace>-<repo> for the selected backend.
    "ASR_MODEL_PATH": "",
    # Model precision; bfloat16 is the current CUDA-friendly default.
    "ASR_DTYPE": "bfloat16",
    # Attention implementation. sdpa uses PyTorch scaled-dot-product attention.
    "ASR_ATTENTION": "sdpa",

    # --- ASR Language & Generation ---
    # Source audio language hint passed to ASR.
    "ASR_LANGUAGE": "Japanese",
    # 1 forces the ASR language prompt instead of letting the model infer language.
    "ASR_FORCE_LANGUAGE": "1",

    # --- Batch Size & Limits ---
    # The ASR stage always runs in the unified GPU worker process; the Web/main
    # process only orchestrates and must not own CUDA.
    # 0 disables the coarse whole-stage timeout; per-batch ASR timeouts still apply.
    "ASR_STAGE_WORKER_TIMEOUT_S": "0",
    "ASR_STAGE_WORKER_READY_TIMEOUT_S": "60",
    # On worker-level CUDA OOM, restart the GPU worker and retry with half batch size.
    # Default 3 lets the built-in batch table fall to 1 before giving up.
    "ASR_STAGE_WORKER_OOM_RETRY_LIMIT": "3",
    # Soft OOM guard for 6GB cards: if worker-side peak reserved VRAM exceeds this
    # budget, treat it as OOM before Windows falls back to shared GPU memory.
    "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "5600",
    # ASR inference batch size. auto resolves by ASR_BACKEND repo id.
    # Defaults target 6GB-class cards.
    "ASR_BATCH_SIZE": "auto",
    "ASR_BATCH_SIZE_BY_REPO": (
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4,"
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf=12"
    ),
    # Max generated tokens configured when loading the Qwen ASR wrapper.
    "ASR_MAX_NEW_TOKENS": "128",
    # Generation penalty to reduce repeated ASR text.
    "ASR_REPETITION_PENALTY": "1.05",

    # --- Semantic boundary pipeline / ASR Chunking ---
    # Speech islands are refined first, then split decisions and shared cut timestamps.
    # This is a feature-score grid fallback, not the source video frame rate.
    "BOUNDARY_FEATURE_FRAME_HOP_S": "0.02",
    "OUTER_EDGE_REFINER_DEVICE": "auto",
    "SEMANTIC_SPLIT_DEVICE": "auto",
    "CUT_EDGE_REFINER_DEVICE": "auto",
    "BOUNDARY_FRAME_SEQUENCE_LEFT_CONTEXT_S": "0.60",
    "BOUNDARY_FRAME_SEQUENCE_RIGHT_CONTEXT_S": "0.60",
    "BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS": "128",
    "BOUNDARY_FRAME_SEQUENCE_INCLUDE_MFCC": "1",
    "SPEECH_BOUNDARY_JA_WINDOW_S": "20.0",
    "SPEECH_BOUNDARY_JA_OVERLAP_S": "4.0",
    # Optional learned SpeechBoundary-JA Mamba2 scorer override. Empty uses the registered repo-id scorer when available; auto resolves the same registry explicitly.
    "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO": "",
    "SPEECH_BOUNDARY_JA_SCORER_DEVICE": "auto",
    # Frame-score mask dilation before raw SpeechBoundary segment extraction. This is not ASR padding.
    "SPEECH_BOUNDARY_JA_FRAME_DILATION_S": "0.2",
    # 1 caches SpeechBoundary frame score -> Boundary Planner outputs separately from ASR generation settings.
    "BOUNDARY_CACHE_ENABLED": "1",
    # Persistent boundary cache directory. Versioned by src/boundary/cache.py.
    "BOUNDARY_CACHE_DIR": "./tmp/cache/boundary",

    # --- Pre-ASR CueQC v11 semantic chunk keep/drop router ---
    # Low-VRAM default: drop obvious non-speech chunks before ASR.
    "PRE_ASR_CUEQC_ENABLED": "1",
    "PRE_ASR_CUEQC_MODEL_PATH_BY_REPO": "",
    "PRE_ASR_CUEQC_DEVICE": "auto",
    "PRE_ASR_CUEQC_DROP_THRESHOLD": "0.95",
    # Optional JSONL export for cold-start clustering/training candidates.
    "PRE_ASR_CUEQC_EXPORT_CANDIDATES_PATH": "",
    "PRE_ASR_CUEQC_EXPORT_CANDIDATES_APPEND": "1",

    # --- Subtitle Timings ---
    # Minimum displayed subtitle duration in seconds.
    "MIN_SUBTITLE_DURATION": "0.6",
    # Estimated Chinese reading speed in characters per second.
    "SUBTITLE_READING_CPS": "7.0",
    # Fixed reading-time buffer added to each subtitle.
    "SUBTITLE_READING_BASE": "0.35",
    # Max stretch ratio compared with the original segment duration in reading mode.
    "SUBTITLE_DURATION_RATIO_CAP": "1.65",
    # Extra seconds allowed when extending short subtitles.
    "SUBTITLE_DURATION_GRACE": "0.9",
    # Extra reading-time weight for the Japanese line in bilingual mode.
    "SUBTITLE_BILINGUAL_SECONDARY_WEIGHT": "0.4",

    # --- LLM Translation Settings ---
    # Base URL for providers that expose an OpenAI-compatible API; DeepSeek by default.
    "OPENAI_COMPATIBILITY_BASE_URL": "https://api.deepseek.com",
    # Translation model name sent to the SDK client.
    "LLM_MODEL_NAME": "deepseek-v4-flash",
    # OpenAI-compatible API surface for translation requests. Valid values: chat, responses.
    "LLM_API_FORMAT": "chat",
    # Reasoning effort parameter for models that support it. Valid values: medium, xhigh.
    "LLM_REASONING_EFFORT": "xhigh",
    # Final subtitle language.
    "TARGET_LANG": "简体中文",
    # Comma-separated Japanese-to-Chinese term mapping injected into translation prompts.
    "TRANSLATION_GLOSSARY": "ちんぽ-肉棒, チンポ-肉棒, おちんちん-肉棒, チンポコ-肉棒",

    # --- Output & Cache ---
    # Root directory for per-video temporary files.
    "JOB_TEMP_DIR": "./tmp/jobs",
    # Root directory for transient ASR wav chunks and crash-resume checkpoints.
    "ASR_CHUNK_ROOT": "./tmp/chunks",
    # 1 writes per-job run logs and persistent timing sidecars under RUN_LOG_DIR.
    "RUN_LOG_ENABLED": "1",
    # Persistent diagnostics root. Runtime creates one subdirectory per job id.
    "RUN_LOG_DIR": "./tmp/log",
    # Internal TCP port used by the web console to receive StageEvent lines.
    "JAVTRANS_EVENTS_PORT": "17322",
    # Web console HTTP port used by launcher.py.
    "JAVTRANS_PORT": "17321",

    # --- Audio ---
    # 1 applies dynamic normalization before ASR.
    "AUDIO_DYNAUDNORM": "1",
    # 1 applies ffmpeg loudnorm when dynaudnorm is disabled.
    "AUDIO_USE_LOUDNORM": "0",

    # --- Quality Report ---
    # 1 stops the pipeline when quality_report warnings are present.
    "QC_HARD_FAIL": "0",
    # 1 writes {video}.quality_report.md plus a machine-readable JSON sidecar.
    "QUALITY_REPORT_ENABLED": "0",
    # Optional override for quality reports. Empty means video/<video-stem>/.
    "QUALITY_REPORT_DIR": "",
    # Maximum allowed empty Chinese translation ratio.
    "QC_MAX_EMPTY_ZH": "0.02",
    # Maximum allowed repeated-translation ratio.
    "QC_MAX_REPETITION": "0.05",
    # Maximum allowed kana-only source ratio; JAV content usually needs a loose threshold.
    "QC_MAX_KANA_ONLY": "0.30",
    # Maximum allowed short-subtitle ratio.
    "QC_MAX_SHORT_SEG": "0.15",
    # Maximum subtitle count per minute before warning.
    "QC_MAX_PER_MIN": "8",
    # Minimum required glossary hit rate when glossary terms are present.
    "QC_MIN_GLOSSARY_HIT": "0.80",
    # Maximum ASR generation failures before quality report warning.
    "QC_MAX_ASR_GENERATION_ERRORS": "0",
    # Maximum ASR generation overflow failures before quality report warning.
    "QC_MAX_ASR_GENERATION_OVERFLOWS": "0",

    # --- Debug / Advanced ---
    # Test-only crash injection for translation resume tests.
    "_TEST_CRASH_TRANSLATION_BATCH": "",
}


def _coerce_default_value(key: str, value: str) -> str:
    if not is_frozen() or key not in _FROZEN_PATH_KEYS or not str(value or "").strip():
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((PROJECT_ROOT / path).resolve())


def _apply_values(values: dict[str, str], protected_keys: set[str]) -> None:
    """Copy string settings into os.environ unless a higher-priority source owns them."""

    for key, value in values.items():
        if key in protected_keys:
            continue
        value = _coerce_default_value(key, value)
        if key == "HF_ENDPOINT" and not str(value or "").strip():
            os.environ.pop(key, None)
            continue
        os.environ[key] = value


def _load_private_env(path: Path, protected_keys: set[str]) -> None:
    """Load .env without clobbering protected process environment keys."""

    if not path.exists():
        return

    for key, value in dotenv_values(path).items():
        if key in protected_keys:
            continue
        if key == "HF_ENDPOINT" and not str(value or "").strip():
            os.environ.pop(key, None)
            continue
        os.environ[key] = "" if value is None else value


def load_config(*, override_existing_env: bool = False) -> None:
    """Load shared defaults, then private local overrides.

    Precedence by default is: existing process env > .env > DEFAULT_SETTINGS.
    That keeps test/process overrides intact while still letting .env override
    shared defaults from this file.
    """

    protected_keys = set() if override_existing_env else set(os.environ)
    _apply_values(DEFAULT_SETTINGS, protected_keys)
    _load_private_env(PRIVATE_ENV_PATH, protected_keys)
