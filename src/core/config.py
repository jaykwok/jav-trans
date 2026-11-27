import ipaddress
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from core.atomic_io import write_json_atomic
from core.typed_config import (
    clear_configuration_problem,
    record_configuration_problem,
)
from utils.runtime_paths import is_frozen, runtime_root

logger = logging.getLogger(__name__)
_CONFIG_LOAD_ERROR = ""
_LAST_REPORTED_CONFIG_LOAD_ERROR = ""
_CONFIG_FILE_LOCK = threading.RLock()
_CONFIG_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

PROJECT_ROOT = runtime_root()
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.json"
_FROZEN_PATH_KEYS = {
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "TORCH_HOME",
    "ASR_RESULT_CACHE_ROOT",
    "JOB_TEMP_DIR",
    "ASR_CHUNK_ROOT",
    "RUN_LOG_DIR",
}


# Runtime configuration source of truth.
#
# Values are strings because they are copied into os.environ for modules that
# read configuration at import time. Keep shared defaults here and put local
# machine/API overrides in config/config.json. Web task options are carried by
# JobContext, not process-wide environment values.
DEFAULT_SETTINGS: dict[str, str] = {
    # --- HuggingFace Path ---
    # Local HuggingFace/model cache path. Relative paths are resolved from project root after loading.
    "HF_HOME": "./models",
    # torch.hub runtime cache; not a model directory and not part of models/.
    "TORCH_HOME": "./cache/torch",
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
    # 1.7B SFT encoder); clear this to fall back to coarse timing and fixed-
    # length chunks. This project selects the separate JAV/non-semantic-
    # vocalisation variant. The original general `ctc_aligner.pt` remains in
    # the same HF repo and can be selected explicitly; it was not overwritten.
    # `hf:<repo>@<sha>#<file>` downloads once into the HF cache and is offline
    # afterwards; a plain path still works as a local override. The sha is
    # pinned deliberately - under a moving branch a retrained head would change
    # every subtitle's timing with nothing in the run saying so.
    "ASR_ALIGNMENT_HEAD_PATH": "hf:jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf@368788ce95dbd10fc47717442cd5bb63a0e00769#ctc_aligner_jav_vocalisation_v3.pt",

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
    "ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=7168",
    "ASR_STAGE_WORKER_RAM_RATIO": "0.95",
    # The Windows PDH shared-VRAM counter jitters by a few MB even when the
    # allocator is hard-capped and cannot spill; only growth beyond this
    # tolerance counts as a real WDDM spill.
    "ASR_SHARED_VRAM_SPILL_TOLERANCE_MB": "64",
    "ASR_STAGE_WORKER_HEARTBEAT_S": "10",
    # Persistent-worker idle self-exit to shed CUDA state on long Web sessions:
    # the worker self-exits after this many seconds with no inbound request (0 =
    # never; default 300s). A per-job restart cadence is intentionally not
    # offered -- every job already gc+empty_cache's on completion, so VRAM does
    # not accumulate across jobs.
    "ASR_STAGE_WORKER_MAX_IDLE_S": "300",
    # ASR inference batch size. "auto" derives the batch directly from Qwen3-ASR
    # 1.7B VRAM physics (total/free VRAM, dtype, chunk_max_s, attention) capped
    # at 24, with runtime OOM half-step retry as safety backstop. Set an integer
    # for an exact override.
    "ASR_BATCH_SIZE": "auto",
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
    # Plan complete source sentences before translation. Overlong sentences
    # may split only at a completed clause with >=600ms measured silence.
    # A word gap or a translated-text ratio cannot create a subtitle boundary.
    "SUBTITLE_MAX_SOURCE_CHARS": "32",
    # Best-effort lexical-duration target paired with the source-character
    # target above. A measured safe boundary always outranks this value.
    "SUBTITLE_MAX_DISPLAY_DURATION_S": "7.0",
    # Minimum displayed subtitle duration in seconds, for cues that have no
    # measured timeline. Layout v3 never stretches a measured cue to reach it:
    # the cue ends at the last spoken character, so 7% of real cues land under
    # this value and QC reports them as a scale, not as a defect (README,
    # "字幕与文本策略").
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
    # Translation backend type: llamacpp (default, fixed Hy-MT3-7B Q4_K_M GGUF
    # served by llama.cpp) | openai (OpenAI-compatible API).
    "TRANSLATION_BACKEND": "llamacpp",
    # Base URL for providers that expose an OpenAI-compatible API; OpenRouter by
    # default. DeepSeek's own endpoint is `https://api.deepseek.com` (no version
    # path), and it is the one endpoint with a documented divergence: no strict
    # `json_schema`, so the transport sends `json_object` there.
    "OPENAI_COMPATIBILITY_BASE_URL": "https://openrouter.ai/api/v1",
    # Translation model name sent to the SDK client, in the id space of the base
    # URL above. OpenRouter's canonical form is `author/slug`; a bare slug is
    # resolved when it matches exactly one model and 400s as ambiguous when it
    # matches several (measured 2026-08-24: `deepseek-v4-flash` resolves,
    # `deepseek-chat` does not). The default is spelled the canonical way so it
    # cannot start failing when a second model matches the short name.
    "LLM_MODEL_NAME": "deepseek/deepseek-v4-flash",
    # Thinking budget for models that support it. Valid values: none, low, high.
    "LLM_REASONING_EFFORT": "low",
    # Sampling temperature for translation. Higher = more colloquial/varied; the
    # JSON-format retry loop tolerates the extra variance. Read at import time; a
    # change requires a restart (not hot-reloaded by the web settings page).
    #
    # Note it may do nothing while thinking is on: DeepSeek documents temperature
    # and top_p as accepted-but-inert in thinking mode, on both the Chat and
    # Responses surfaces, and other providers make their own call. It always
    # matters again at LLM_REASONING_EFFORT=none.
    "LLM_TEMPERATURE": "0.6",
    # Cues per translation request, and since the worker coupling was removed
    # this is the operating point rather than a ceiling - the only number
    # deciding how many requests a film costs. Clamped to [8, 400], read at
    # import time. Lower only for a provider that repeatedly fails large replies:
    # it costs money, because reasoning is charged per request and smaller
    # batches mean more of them.
    "TRANSLATION_BATCH_SIZE": "200",
    # Final subtitle language.
    "TARGET_LANG": "简体中文",
    # Comma-separated Japanese-to-target-language term mapping injected into API prompts.
    "TRANSLATION_GLOSSARY": "ちんぽ-肉棒, チンポ-肉棒, おちんちん-肉棒, チンポコ-肉棒",

    # --- llama.cpp GGUF Settings (when TRANSLATION_BACKEND=llamacpp) ---
    # Runs quantized GGUF models through a managed local llama-server process
    # (official prebuilt CUDA binaries; OpenAI protocol on 127.0.0.1). Empty
    # server path checks ./models/runtime/llama-cpp/, then PATH, and on Windows
    # auto-downloads the latest CUDA-major-matched llama.cpp release (with a
    # verified fallback build) on first local translation run.
    "LLAMACPP_SERVER_PATH": "",
    # Default model: official Hy-MT3-7B Q4_K_M (~4.79 GiB / 5.15 GB GGUF),
    # sharing the unified JSON batch contract (`json`) with API models, sized
    # to small local batches to stay comfortably inside the 8GB VRAM floor.
    "LLAMACPP_MODEL_REPO": "tencent/Hy-MT3-7B-GGUF",
    "LLAMACPP_MODEL_FILE": "Hy-MT3-7B-Q4_K_M.gguf",
    # Explicit local GGUF path wins over repo+file download.
    "LLAMACPP_GGUF_PATH": "",
    # Local VRAM-bounded batch sizing for the unified JSON contract (API uses
    # TRANSLATION_BATCH_SIZE=200 with full-film prefix caching; local uses
    # 32-cue windowed batches — the measured 7B Q4 quality sweet spot).
    "LLAMACPP_TRANSLATION_BATCH_SIZE": "32",
    "LLAMACPP_BATCH_MAX_SOURCE_CHARS": "2400",
    # Context per server slot and parallel slot count. "auto" derives safe
    # values from the actual GGUF KV geometry and available GPU VRAM (e.g.
    # 4096 x 2 on 8 GB cards, scaling up to 6144/8192 x 3-4 on 10-16+ GB cards);
    # set explicit integers to override.
    "LLAMACPP_CTX_SIZE": "auto",
    "LLAMACPP_N_GPU_LAYERS": "999",
    "LLAMACPP_PARALLEL": "auto",
    "LLAMACPP_STARTUP_TIMEOUT_S": "300",
    # Admission may fail without abandoning the retiring resource owner.
    "LOCAL_BACKEND_WAIT_TIMEOUT_S": "600",
    # Bound only the caller's wait; cleanup continues on one thread per backend.
    "LOCAL_BACKEND_CLOSE_WAIT_S": "15",
    # Prompt contract: auto | json (off/none alias json). Local Hy-MT3 and API
    # models both speak the unified JSON batch contract.
    "TRANSLATION_PROMPT_PROFILE": "auto",

    # --- Output & Cache ---
    # Root directory for per-video temporary files.
    "JOB_TEMP_DIR": "./tmp/jobs",
    # Root directory for transient ASR wav chunks.
    "ASR_CHUNK_ROOT": "./tmp/chunks",
    # Cross-job content-addressed ASR result cache (survives job cleanup);
    # set ASR_RESULT_CACHE_ENABLED=0 to disable reads and writes.
    "ASR_RESULT_CACHE_ROOT": "./cache/asr",
    "ASR_RESULT_CACHE_ENABLED": "1",
    # 1 writes per-job run logs and persistent timing sidecars under RUN_LOG_DIR.
    "RUN_LOG_ENABLED": "1",
    # Persistent per-job diagnostics root. Runtime creates one subdirectory per job id.
    "RUN_LOG_DIR": "./logs/jobs",
    # Internal TCP port used by the web console to receive StageEvent lines.
    "JAV_SUB_EVENTS_PORT": "2234",
    # Web console HTTP port used by launcher.py.
    "JAV_SUB_PORT": "2233",
    # Set to "1" to disable automatic background check for GitHub Releases on startup.
    # Default "0" enables automatic checks (manual checks and --update always work).
    "JAV_SUB_DISABLE_UPDATE_CHECK": "0",

    # --- Audio ---
    # 1 applies dynamic normalization before ASR.
    "AUDIO_DYNAUDNORM": "1",
    # 1 applies ffmpeg loudnorm when dynaudnorm is disabled.
    "AUDIO_USE_LOUDNORM": "0",

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
# These are wire values, not labels. DeepSeek's Chat surface accepts exactly
# `low` / `high` / `max` for `reasoning_effort` and switches thinking off
# through `extra_body={"thinking": {"type": "disabled"}}`; its Responses
# surface spells the same axis as `reasoning.effort` in `none|low|high|max`.
# Both readings agree that off-vs-on is one more point on the effort axis, so
# it is modelled here as the "none" tier rather than a second parameter.
#
# "medium" was never one of them. It shipped as this project's default from
# 2026-08-14 to 2026-08-24 and DeepSeek silently ignored it, falling back to
# its documented default of `high` - so every job billed at the second-most
# expensive tier while the UI read "medium（默认）". That is the whole of the
# cost anomaly: on sample-x (1,396 cues) one run spent 570,031 output tokens,
# ~93% of them reasoning, and output was ~91% of the bill at either price
# tier.
#
# "max" is dropped rather than kept: on deepseek-v4-flash it maps to `max`,
# the single most expensive setting, and nothing in this pipeline was ever
# shown to need it.
REASONING_EFFORTS = ("none", "low", "high")
# Saved settings, job records and .env files carry values these tiers replaced.
# `medium` resolves to what it actually ran as, so a re-run reproduces the old
# result rather than quietly changing tier along with the rename. `max`/`xhigh`
# fold down to `high`, which is both the nearest surviving tier and the cheaper
# direction - the point of the change.
_RETIRED_REASONING_EFFORTS = {"medium": "high", "xhigh": "high", "max": "high"}
# Keep the default aligned with DEFAULT_SETTINGS and all job/settings readers.
# `none` remains an explicit choice; the default enables a low thinking budget.
DEFAULT_REASONING_EFFORT = "low"


def normalize_reasoning_effort(
    value: str | None, fallback: str = DEFAULT_REASONING_EFFORT
) -> str:
    """Clamp a thinking tier to the supported set."""
    normalized = (value or fallback or DEFAULT_REASONING_EFFORT).strip().lower()
    normalized = _RETIRED_REASONING_EFFORTS.get(normalized, normalized)
    if normalized in REASONING_EFFORTS:
        return normalized
    return fallback if fallback in REASONING_EFFORTS else DEFAULT_REASONING_EFFORT


def recognized_reasoning_effort(value: str | None) -> str | None:
    """The tier this names, or None if it names nothing.

    `normalize_reasoning_effort` cannot answer this: it clamps an unknown value
    to the default, so a typo comes back as `low` and is indistinguishable from
    someone asking for `low`. Callers that must reject garbage rather than
    silently accept a tier need this instead.
    """
    normalized = (value or "").strip().lower()
    normalized = _RETIRED_REASONING_EFFORTS.get(normalized, normalized)
    return normalized if normalized in REASONING_EFFORTS else None


def escalated_reasoning_effort(effort: str | None) -> str:
    """The tier the repair pass reissues a flagged cue at: one step up, clamped.

    One step rather than "always the top" is what makes the cascade cheap at
    both ends. The base pass sets the price of the whole film; the repair pass
    only pays for the cues a detector actually flagged, so it can afford to
    think - but jumping straight to `high` from `none` would also make the
    repair the expensive half on a film where the base pass flagged a lot.

    `high` escalates to itself. There is nowhere left to go, and a repair that
    reissues at the same tier is still worth making: it carries the neighbouring
    lines as context and an explicit statement of what looked wrong, neither of
    which the base request had.
    """
    normalized = normalize_reasoning_effort(effort)
    index = REASONING_EFFORTS.index(normalized)
    return REASONING_EFFORTS[min(index + 1, len(REASONING_EFFORTS) - 1)]


_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_NO_PROXY_ENV_KEYS = ("NO_PROXY", "no_proxy")
NETWORK_PROXY_ENV_KEYS = ("PROXY_PROTOCOL", "PROXY_HOST", "PROXY_PORT") + _PROXY_ENV_KEYS + _NO_PROXY_ENV_KEYS
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


def _redact_proxy_host_for_log(raw_host: str) -> str:
    if "@" not in raw_host:
        return raw_host
    _, _, host_part = raw_host.rpartition("@")
    return f"***@{host_part}"


def _normalize_proxy_host(raw_host: str) -> str | None:
    if (
        "://" in raw_host
        or any(ch.isspace() for ch in raw_host)
        or any(ch in raw_host for ch in ("/", "?", "#", "\\", "\x00"))
    ):
        return None
    userinfo, sep, host_part = raw_host.rpartition("@")
    if not host_part:
        return None
    if host_part.startswith("[") or host_part.endswith("]"):
        if not (host_part.startswith("[") and host_part.endswith("]")):
            return None
        try:
            ipaddress.IPv6Address(host_part[1:-1])
        except ValueError:
            return None
        formatted_host = host_part
    elif ":" in host_part:
        try:
            ipaddress.IPv6Address(host_part)
        except ValueError:
            return None
        formatted_host = f"[{host_part}]"
    else:
        formatted_host = host_part
    return f"{userinfo}@{formatted_host}" if sep else formatted_host


def network_proxy_url_from_env() -> str:
    raw_protocol = os.getenv("PROXY_PROTOCOL", "").strip().lower()
    protocol = raw_protocol or "http"
    if protocol not in _SUPPORTED_PROXY_PROTOCOLS:
        record_configuration_problem(
            "PROXY_PROTOCOL",
            raw_protocol,
            "不是支持的代理协议（http、https、socks5），已按 'http' 处理",
        )
        protocol = "http"
    else:
        clear_configuration_problem("PROXY_PROTOCOL")

    raw_host = os.getenv("PROXY_HOST", "").strip()
    raw_port = os.getenv("PROXY_PORT", "").strip()

    normalized_host: str | None = None
    if raw_host:
        normalized_host = _normalize_proxy_host(raw_host)
        if normalized_host is None:
            record_configuration_problem(
                "PROXY_HOST",
                _redact_proxy_host_for_log(raw_host),
                "代理主机格式无效（不能包含协议头、空白、/、?、# 或无效端口），已忽略代理配置",
            )
        else:
            clear_configuration_problem("PROXY_HOST")
    else:
        clear_configuration_problem("PROXY_HOST")

    port_num: int | None = None
    if raw_port:
        if raw_port.isascii() and raw_port.isdigit():
            candidate_port = int(raw_port)
            if 1 <= candidate_port <= 65535:
                port_num = candidate_port
        if port_num is None:
            record_configuration_problem(
                "PROXY_PORT",
                raw_port,
                "代理端口必须是 1~65535 之间的整数，已忽略代理配置",
            )
        else:
            clear_configuration_problem("PROXY_PORT")
    else:
        clear_configuration_problem("PROXY_PORT")

    if normalized_host is None or port_num is None:
        return ""

    proxy_url = f"{protocol}://{normalized_host}:{port_num}"
    try:
        parsed = urlsplit(proxy_url)
        if not parsed.hostname or parsed.port != port_num:
            raise ValueError("invalid proxy authority")
    except ValueError:
        record_configuration_problem(
            "PROXY_HOST",
            _redact_proxy_host_for_log(raw_host),
            "代理主机无法构成合法 URL，已忽略代理配置",
        )
        return ""
    return proxy_url


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


def config_load_error() -> str:
    """Return the last config/config.json decode/read problem encountered during load_config, if any."""
    with _CONFIG_FILE_LOCK:
        return _CONFIG_LOAD_ERROR


def _set_config_load_error(message: str) -> None:
    """Record a config load error under lock and deduplicate repeated warnings."""
    global _CONFIG_LOAD_ERROR, _LAST_REPORTED_CONFIG_LOAD_ERROR
    with _CONFIG_FILE_LOCK:
        _CONFIG_LOAD_ERROR = message
        if not message:
            _LAST_REPORTED_CONFIG_LOAD_ERROR = ""
            return
        if message == _LAST_REPORTED_CONFIG_LOAD_ERROR:
            return
        _LAST_REPORTED_CONFIG_LOAD_ERROR = message
    logger.warning("%s", message)
    print(f"[WARN] {message}", flush=True)


def _stringify_config_value(key: str, value: Any) -> str | None:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    logger.warning(
        "Ignoring non-scalar config value for %s in config/config.json (%s)",
        key,
        type(value).__name__,
    )
    return None


def _parse_config_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("配置文件内容为空（可能已被截断）")
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象（{...}）")
    return loaded


def read_config_file(path: Path | None = None) -> dict[str, str]:
    """Read key-value overrides from config/config.json leniently."""
    target = Path(path) if path is not None else CONFIG_PATH
    with _CONFIG_FILE_LOCK:
        if not target.exists():
            return {}
        try:
            raw = _parse_config_mapping(target)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Skipping unreadable config at %s: %s", target, exc)
            return {}

    values: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not _CONFIG_KEY_RE.fullmatch(key):
            continue
        coerced = _stringify_config_value(key, value)
        if coerced is not None:
            values[key] = coerced
    return values


def update_config_file(
    updates: dict[str, Any],
    path: Path | None = None,
) -> None:
    """Atomically merge updates into config/config.json while preserving existing keys."""
    coerced_updates: dict[str, str] = {}
    for key, value in updates.items():
        if not isinstance(key, str) or not _CONFIG_KEY_RE.fullmatch(key):
            raise ValueError(f"Invalid config key: {key!r}")
        coerced = _stringify_config_value(key, value)
        if coerced is None:
            raise ValueError(
                f"Invalid non-scalar config value for {key!r}: {type(value).__name__}"
            )
        coerced_updates[key] = coerced

    target = Path(path) if path is not None else CONFIG_PATH
    with _CONFIG_FILE_LOCK:
        existing: dict[str, Any] = {}
        if target.exists():
            try:
                existing = _parse_config_mapping(target)
            except UnicodeDecodeError as exc:
                raise ValueError(
                    "config/config.json 不是有效的 UTF-8 编码，请先转换为 UTF-8 后再保存设置。"
                ) from exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"config/config.json 不是有效的 JSON 对象（{exc}），请修正或删除后再保存设置。"
                ) from exc
        merged = dict(existing)
        merged.update(coerced_updates)
        write_json_atomic(target, merged, indent=2, ensure_ascii=False)


def _load_private_config(path: Path, protected_keys: set[str]) -> None:
    """Load config/config.json without clobbering protected process environment keys."""
    with _CONFIG_FILE_LOCK:
        if not path.exists():
            _set_config_load_error("")
            return

        try:
            raw = _parse_config_mapping(path)
        except UnicodeDecodeError as exc:
            _set_config_load_error(
                f"配置文件（{path}）不是有效的 UTF-8 编码（{exc}），已跳过加载并使用默认配置；"
                "请先将其转换为 UTF-8 编码后重启，或在设置页重新保存。"
            )
            return
        except (json.JSONDecodeError, ValueError) as exc:
            _set_config_load_error(
                f"配置文件（{path}）不是有效的 JSON 对象（{exc}），已跳过加载并使用默认配置；"
                "请修正格式后重启，或删除后在设置页重新保存。"
            )
            return
        except OSError as exc:
            _set_config_load_error(
                f"读取配置文件（{path}）失败（{exc}），已使用默认配置。"
            )
            return

        _set_config_load_error("")

        for key, value in raw.items():
            if not isinstance(key, str) or not _CONFIG_KEY_RE.fullmatch(key):
                continue
            if key in protected_keys:
                continue
            coerced = _stringify_config_value(key, value)
            if coerced is None:
                continue
            coerced = _coerce_default_value(key, coerced)
            if key == "HF_ENDPOINT" and not coerced.strip():
                os.environ.pop(key, None)
                continue
            os.environ[key] = coerced


def load_config(*, override_existing_env: bool = False) -> None:
    """Load shared defaults, then private local overrides.

    Precedence by default is: existing process env > config/config.json > DEFAULT_SETTINGS.
    That keeps test/process overrides intact while still letting config/config.json
    override shared defaults from this file.
    """
    with _CONFIG_FILE_LOCK:
        protected_keys = set() if override_existing_env else set(list(os.environ))
        _apply_values(DEFAULT_SETTINGS, protected_keys)
        _load_private_config(CONFIG_PATH, protected_keys)
        sync_network_proxy_environment(clear_existing=False)
