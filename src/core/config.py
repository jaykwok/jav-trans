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
    # `hf:<repo>@<sha>#<file>` downloads once into the HF cache and is offline
    # afterwards; a plain path still works as a local override. The sha is
    # pinned deliberately - under a moving branch a retrained head would change
    # every subtitle's timing with nothing in the run saying so.
    "ASR_ALIGNMENT_HEAD_PATH": "hf:jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf@2d46a169b71232ff08800472c457fdc092084bdf#ctc_aligner.pt",

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
    # Decode budget. Empty derives it from each chunk's duration: audio cannot
    # contain more than ASR_DECODE_TOKENS_PER_SECOND tokens of speech per second,
    # so the bound cannot cut a transcription short. Setting a number makes it a
    # hard ceiling instead, which bounds decode cost but can truncate dialogue -
    # a flat 128 was doing exactly that at 30s chunks (4.27 tok/s against a
    # measured 4.45). See src/asr/decode_guard.py.
    "ASR_MAX_NEW_TOKENS": "",
    # Tokens per second of audio that no real speech can exceed. ~10 mora/s for
    # fast Japanese, ~1 token per mora on this checkpoint, plus punctuation.
    "ASR_DECODE_TOKENS_PER_SECOND": "10.0",
    # Generation penalty to reduce repeated ASR text.
    "ASR_REPETITION_PENALTY": "1.05",

    # --- ASR Chunking ---
    # Cuts are chosen at blank runs read from the CTC alignment head, and the
    # chunks tile the file exactly - nothing here can drop audio. The five
    # acoustic models that used to live in front of the decoder (Scorer, Outer,
    # Split v4, CueQC v13, Inner v2) were retired on 2026-07-31; their settings
    # are gone from here because leaving them advertised knobs that do nothing.
    # With no alignment head configured this degrades to fixed-length chunks.
    #
    # There is no separate target length: cuts take the *latest* pause that fits
    # under ASR_CHUNK_MAX_S, so chunks run to the ceiling. The ceiling is the
    # encoder's own audio window and the processor pads shorter chunks up to it,
    # so a shorter target gives away context for free - see
    # `asr.chunking.cut_at_pauses` for what the 20s target measured on 2026-08-02.
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
    # Thinking budget for models that support it. Valid values: none, medium, max.
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

    # --- llama.cpp GGUF Settings (when TRANSLATION_BACKEND=llamacpp) ---
    # Runs quantized GGUF models through a managed local llama-server process
    # (official prebuilt CUDA binaries; OpenAI protocol on 127.0.0.1). Empty
    # server path means "find llama-server on PATH" (winget install -e --id
    # ggml.llamacpp, which is the Vulkan build; the CUDA zip is faster on
    # NVIDIA and has to be pointed at explicitly).
    "LLAMACPP_SERVER_PATH": "",
    # Default model: Hy-MT2-1.8B Q8_0 (~2GB), driven by the `hymt2` per-line
    # profile rather than the JSON batch contract. Chosen for local hardware on
    # 2026-08-05: a 9B at Q4 fits an 8GB card only at two server slots, where a
    # 1.8B at Q8 runs eight, and the measured end-to-end difference on 300 real
    # cues was ~11x (0.88 vs 9.97 lines/s). The context layers the JSON contract
    # buys are largely unavailable here anyway - a whole-transcript prefix does
    # not fit the local context budget.
    "LLAMACPP_MODEL_REPO": "tencent/Hy-MT2-1.8B-GGUF",
    "LLAMACPP_MODEL_FILE": "Hy-MT2-1.8B-Q8_0.gguf",
    # Explicit local GGUF path wins over repo+file download.
    "LLAMACPP_GGUF_PATH": "",
    # Context per server slot; total server context is CTX_SIZE * PARALLEL.
    "LLAMACPP_CTX_SIZE": "8192",
    "LLAMACPP_N_GPU_LAYERS": "999",
    # Eight slots, because the default model is now ~2GB rather than ~5.5GB and
    # the per-line contract makes requests small. Raise CTX_SIZE, not this, if a
    # custom GGUF needs the JSON contract's longer prompts.
    "LLAMACPP_PARALLEL": "8",
    "LLAMACPP_STARTUP_TIMEOUT_S": "300",
    # Prompt contract: auto | json (off/none are accepted aliases for json).
    # Only the JSON contract ships, so auto resolves to json everywhere; the
    # switch stays because adding a model family means registering a profile.
    "TRANSLATION_PROMPT_PROFILE": "auto",

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
    "JAV_TRANS_EVENTS_PORT": "2234",
    # Web console HTTP port used by launcher.py.
    "JAV_TRANS_PORT": "2233",

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


# The three thinking tiers, in one place because they used to live in three:
# `llm.settings`, `web.models` and `core.job_context` each kept their own copy
# and they drifted -- job_context's list was missing the no-thinking tier
# entirely, so a job submitted with it silently ran at "medium" no matter what
# the UI said.
#
# "none" is the off switch (not "minimal", which is the smallest *nonzero*
# budget on OpenAI, Gemini and DeepSeek alike) and "max" is the top tier.
REASONING_EFFORTS = ("none", "medium", "max")


def normalize_reasoning_effort(value: str | None, fallback: str = "medium") -> str:
    """Clamp a thinking tier to the supported set."""
    normalized = (value or fallback or "medium").strip().lower()
    if normalized in REASONING_EFFORTS:
        return normalized
    return fallback if fallback in REASONING_EFFORTS else "medium"


_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_NO_PROXY_ENV_KEYS = ("NO_PROXY", "no_proxy")
# The llamacpp backend talks to a llama-server it started itself on 127.0.0.1,
# and neither urllib nor httpx exempts loopback on its own (verified: with
# HTTP_PROXY set, `proxy_bypass("127.0.0.1")` is False and httpx picks the proxy
# transport for a 127.0.0.1 URL). Without this, configuring a proxy for model
# downloads would send local translation requests to the proxy instead.
_LOOPBACK_NO_PROXY = ("127.0.0.1", "localhost", "::1")
_SUPPORTED_PROXY_PROTOCOLS = {"http", "https", "socks5"}


def _no_proxy_value_with_loopback(existing: str) -> str:
    """Add the loopback hosts to a NO_PROXY list, keeping what the user set."""

    entries = [item.strip() for item in str(existing or "").split(",")]
    entries = [item for item in entries if item]
    if "*" in entries:
        return ",".join(entries)
    lowered = {item.lower() for item in entries}
    for host in _LOOPBACK_NO_PROXY:
        if host not in lowered:
            entries.append(host)
            lowered.add(host)
    return ",".join(entries)


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
        existing_no_proxy = ""
        for key in _NO_PROXY_ENV_KEYS:
            existing_no_proxy = os.environ.get(key) or existing_no_proxy
        merged = _no_proxy_value_with_loopback(existing_no_proxy)
        for key in _NO_PROXY_ENV_KEYS:
            os.environ[key] = merged
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
