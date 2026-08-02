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
    # HuggingFace Hub endpoint override. Empty string means use the default huggingface.co.
    "HF_ENDPOINT": "",
    # Optional network proxy. When host+port are set, load_config exports HTTP_PROXY/
    # HTTPS_PROXY/ALL_PROXY at config.py:317.
    "PROXY_PROTOCOL": "http",
    "PROXY_HOST": "",
    "PROXY_PORT": "",

    # --- ASR Model Settings ---
    # Transcription backend. Use the Hugging Face repo id as the stable key.
    "ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    # Optional explicit HuggingFace model id override. Empty auto-selects by ASR_BACKEND.
    "ASR_MODEL_ID": "",
    # Optional local ASR model directory override. Empty uses models/<namespace>-<repo> for the selected backend.
    "ASR_MODEL_PATH": "",
    # Model precision; bfloat16 is the current CUDA-friendly default.
    "ASR_DTYPE": "bfloat16",
    # Attention implementation. sdpa uses PyTorch scaled-dot-product attention.
    "ASR_ATTENTION": "sdpa",
    # CTC alignment head over the ASR encoder: word-level subtitle timing and
    # pause-aware chunk cuts. The head is encoder-specific (trained on the
    # 1.7B SFT encoder); clear this to fall back to proportional timing and
    # fixed-length chunks. Accuracy validated on clean speech; real-domain
    # onset accuracy still under audit.
    "ASR_ALIGNMENT_HEAD_PATH": "./src/checkpoints/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/ctc_aligner.pt",

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
    # Default 6 also lets VRAM-scaled auto batches fall to 1 before giving up.
    "ASR_STAGE_WORKER_OOM_RETRY_LIMIT": "6",
    # "auto" resolves inside the CUDA-owner worker to physical VRAM * ratio.
    # A numeric MB value remains available as an exact expert override.
    "ASR_STAGE_WORKER_VRAM_BUDGET_MB": "auto",
    "ASR_STAGE_WORKER_VRAM_RATIO": "0.95",
    "ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=6144",
    "ASR_STAGE_WORKER_RAM_RATIO": "0.95",
    # The Windows PDH shared-VRAM counter jitters by a few MB even when the
    # allocator is hard-capped and cannot spill; only growth beyond this
    # tolerance counts as a real WDDM spill.
    "ASR_SHARED_VRAM_SPILL_TOLERANCE_MB": "64",
    "ASR_STAGE_WORKER_HEARTBEAT_S": "10",
    # Cross-job auto-batch learning. Successful jobs below the utilization
    # threshold probe between the safe batch and current upper bound; OOM
    # records the unsafe bound.
    "GPU_BATCH_PROFILE_ENABLED": "1",
    "GPU_BATCH_PROFILE_GROWTH_THRESHOLD": "0.80",
    "GPU_BATCH_PROFILE_PATH": "./tmp/cache/gpu_batch_profiles.json",
    # Persistent-worker idle self-exit to shed CUDA state on long Web sessions:
    # the worker self-exits after this many seconds with no inbound request (0 =
    # never; default 300s). A per-job restart cadence is intentionally not
    # offered -- every job already gc+empty_cache's on completion, so VRAM does
    # not accumulate across jobs.
    "ASR_STAGE_WORKER_MAX_IDLE_S": "300",
    # ASR inference batch size. auto resolves by ASR_BACKEND repo id.
    # The repo table is the 5600MB baseline. In auto mode the worker scales it
    # to the resolved VRAM budget, while an explicit ASR_BATCH_SIZE stays exact.
    "ASR_BATCH_SIZE": "auto",
    "ASR_BATCH_SIZE_BY_REPO": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4",
    # Max generated tokens configured when loading the Qwen ASR wrapper.
    "ASR_MAX_NEW_TOKENS": "128",
    # Generation penalty to reduce repeated ASR text.
    "ASR_REPETITION_PENALTY": "1.05",

    # --- ASR Chunking ---
    # Cuts are chosen at blank runs read from the CTC alignment head, and the
    # chunks tile the file exactly - nothing here can drop audio. The five
    # acoustic models that used to live in front of the decoder (Scorer, Outer,
    # Split v4, CueQC v13, Inner v2) were retired on 2026-07-31; their settings
    # are gone from here because leaving them advertised knobs that do nothing.
    # With no alignment head configured this degrades to fixed-length chunks.
    "ASR_CHUNK_TARGET_S": "20.0",
    "ASR_CHUNK_MAX_S": "30.0",
    "ASR_CHUNK_MIN_S": "2.0",
    # Below ~0.5s a pause is between words, not between sentences: an earlier
    # 0.35s cut produced ~1s fragments that the decoder answered with whole lines.
    "ASR_CHUNK_MIN_PAUSE_S": "0.6",

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
    # Translation backend type: openai (OpenAI-compatible API) | local (本地模型)
    "TRANSLATION_BACKEND": "openai",
    # Base URL for providers that expose an OpenAI-compatible API; DeepSeek by default.
    "OPENAI_COMPATIBILITY_BASE_URL": "https://api.deepseek.com",
    # Translation model name sent to the SDK client.
    "LLM_MODEL_NAME": "deepseek-v4-flash",
    # OpenAI-compatible API surface for translation requests. Valid values: chat, responses.
    "LLM_API_FORMAT": "chat",
    # Reasoning effort parameter for models that support it. Valid values: medium, xhigh.
    "LLM_REASONING_EFFORT": "medium",
    # Sampling temperature for translation. Higher = more colloquial/varied; the
    # JSON-format retry loop tolerates the extra variance. Read at import time; a
    # change requires a restart (not hot-reloaded by the web settings page).
    "LLM_TEMPERATURE": "0.6",
    # Subtitles per translation request, independent of worker count. Smaller
    # batches trade throughput for higher per-line quality (less long-output
    # decay). Clamped to [8, 400]. Read at import time; restart to apply.
    "TRANSLATION_BATCH_SIZE": "64",
    # Final subtitle language.
    "TARGET_LANG": "简体中文",
    # Comma-separated Japanese-to-Chinese term mapping injected into translation prompts.
    "TRANSLATION_GLOSSARY": "ちんぽ-肉棒, チンポ-肉棒, おちんちん-肉棒, チンポコ-肉棒",

    # --- Local Model Settings (when TRANSLATION_BACKEND=local) ---
    # Local model path (HuggingFace model ID or local path)
    # Examples: Qwen/Qwen2.5-72B-Instruct, Tencent-Hunyuan/Hunyuan-Large
    "LOCAL_MODEL_PATH": "",
    # Device for local model inference: cuda | cpu
    "LOCAL_MODEL_DEVICE": "cuda",
    # Empty selects float32 on CPU and bfloat16 on CUDA.
    "LOCAL_MODEL_DTYPE": "",
    # Maximum context length for local model
    "LOCAL_MODEL_MAX_LENGTH": "32768",
    # Conservative batch and generation caps for in-process inference.
    "LOCAL_MODEL_BATCH_SIZE": "16",
    "LOCAL_MODEL_MAX_NEW_TOKENS": "8192",
    # Auto-download model from HuggingFace if not present locally
    "LOCAL_MODEL_AUTO_DOWNLOAD": "1",

    # --- llama.cpp GGUF Settings (when TRANSLATION_BACKEND=llamacpp) ---
    # Runs Sakura/GalTransl and other quantized GGUF models through a managed
    # local llama-server process (official prebuilt CUDA binaries; OpenAI
    # protocol on 127.0.0.1). Empty server path means "find llama-server on
    # PATH" (winget install llama.cpp).
    "LLAMACPP_SERVER_PATH": "",
    # Default model: GalTransl 7B v3.7 Q6_K -- the galgame-specialized JA->ZH
    # tier the model card recommends for 8GB GPUs (~6.3GB file, CC-BY-NC-SA).
    "LLAMACPP_MODEL_REPO": "SakuraLLM/Sakura-GalTransl-7B-v3.7",
    "LLAMACPP_MODEL_FILE": "Sakura-Galtransl-7B-v3.7.gguf",
    # Explicit local GGUF path wins over repo+file download.
    "LLAMACPP_GGUF_PATH": "",
    # Context per server slot; total server context is CTX_SIZE * PARALLEL.
    "LLAMACPP_CTX_SIZE": "8192",
    "LLAMACPP_N_GPU_LAYERS": "999",
    "LLAMACPP_PARALLEL": "4",
    "LLAMACPP_STARTUP_TIMEOUT_S": "300",
    # Sakura line-oriented prompt profile: auto | sakura | json. auto switches
    # on exactly when the configured model is a Sakura/GalTransl variant.
    "TRANSLATION_PROMPT_PROFILE": "auto",
    "SAKURA_BATCH_SIZE": "8",
    "SAKURA_WORKERS": "4",
    "SAKURA_HISTORY_LINES": "8",

    # --- Output & Cache ---
    # Root directory for per-video temporary files.
    "JOB_TEMP_DIR": "./tmp/jobs",
    # Root directory for transient ASR wav chunks.
    "ASR_CHUNK_ROOT": "./tmp/chunks",
    # Cross-job content-addressed ASR result cache (survives job cleanup);
    # set ASR_RESULT_CACHE_ENABLED=0 to disable reads and writes.
    "ASR_RESULT_CACHE_ROOT": "./tmp/asr_cache",
    "ASR_RESULT_CACHE_ENABLED": "1",
    # 1 writes per-job run logs and persistent timing sidecars under RUN_LOG_DIR.
    "RUN_LOG_ENABLED": "1",
    # Persistent diagnostics root. Runtime creates one subdirectory per job id.
    "RUN_LOG_DIR": "./tmp/log",
    # Internal TCP port used by the web console to receive StageEvent lines.
    "JAV_TRANS_EVENTS_PORT": "17322",
    # Web console HTTP port used by launcher.py.
    "JAV_TRANS_PORT": "17321",

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


_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_SUPPORTED_PROXY_PROTOCOLS = {"http", "https", "socks5"}


def network_proxy_url_from_env() -> str:
    protocol = os.getenv("PROXY_PROTOCOL", "http").strip().lower() or "http"
    host = os.getenv("PROXY_HOST", "").strip()
    port = os.getenv("PROXY_PORT", "").strip()
    if not host or not port:
        return ""
    if protocol not in _SUPPORTED_PROXY_PROTOCOLS:
        protocol = "http"
    return f"{protocol}://{host}:{port}"


def apply_network_proxy_environment(
    proxy_url: str,
    *,
    clear_existing: bool = False,
) -> None:
    proxy_url = str(proxy_url or "").strip()
    if proxy_url:
        for key in _PROXY_ENV_KEYS:
            os.environ[key] = proxy_url
        return
    if clear_existing:
        for key in _PROXY_ENV_KEYS:
            os.environ.pop(key, None)


def sync_network_proxy_environment(*, clear_existing: bool = False) -> str:
    proxy_url = network_proxy_url_from_env()
    apply_network_proxy_environment(proxy_url, clear_existing=clear_existing)
    return proxy_url


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
    sync_network_proxy_environment(clear_existing=False)
