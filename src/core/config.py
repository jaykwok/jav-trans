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
    "TORCH_HOME": "./temp/torch",
    # HuggingFace Hub endpoint. Set to https://hf-mirror.com for mainland China acceleration.
    # Empty string means use the default huggingface.co. Takes effect on next app start.
    "HF_ENDPOINT": "",

    # --- ASR & Alignment Model Settings ---
    # Transcription backend. Use the Hugging Face repo id as the stable key.
    "ASR_BACKEND": "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame",
    # Local/Qwen ASR timestamp mode. Valid values: forced, native, hybrid.
    "ALIGNMENT_TIMESTAMP_MODE": "forced",
    # Character cap for sliding ASR context prompt before tokenizer-level budgeting.
    "ASR_INITIAL_PROMPT_MAX_CHARS": "240",
    # Max prompt tokens kept before generation budgeting.
    "ASR_INITIAL_PROMPT_MAX_TOKENS": "180",
    # Minimum output budget preserved when prompt tokens compete with decoder context.
    "ASR_MIN_EFFECTIVE_NEW_TOKENS": "64",
    # Voice activity detection backend used before ASR chunking.
    "ASR_BOUNDARY_BACKEND": "speech_boundary_ja",
    # Optional explicit HuggingFace model id override. Empty auto-selects by ASR_BACKEND.
    "ASR_MODEL_ID": "",
    # Optional local ASR model directory override. Empty uses models/<namespace>-<repo> for the selected backend.
    "ASR_MODEL_PATH": "",
    # Remote HuggingFace forced-aligner model id used as fallback.
    "ALIGNER_MODEL_ID": "Qwen/Qwen3-ForcedAligner-0.6B",
    # Optional local forced-aligner directory override. Empty uses models/<namespace>-<repo>.
    "ALIGNER_MODEL_PATH": "",
    # Model precision; bfloat16 is the current 8GB VRAM-friendly default.
    "ASR_DTYPE": "bfloat16",
    # Attention implementation. sdpa uses PyTorch scaled-dot-product attention.
    "ASR_ATTENTION": "sdpa",

    # --- ASR Recognition Context & Generation ---
    # Source audio language hint passed to ASR.
    "ASR_LANGUAGE": "Japanese",
    # 1 forces the ASR language prompt instead of letting the model infer language.
    "ASR_FORCE_LANGUAGE": "1",
    # Short recognition hint such as performer names or domain terms.
    "ASR_CONTEXT": "",
    # Optional extra prompt applied only near the start of the video.
    "ASR_HEAD_CONTEXT": "",
    # Latest segment start time, in seconds, that can receive ASR_HEAD_CONTEXT.
    "ASR_HEAD_CONTEXT_MAX_START_S": "16",

    # --- Batch Size & Limits ---
    # ASR inference batch size; keep low on 8GB VRAM.
    "ASR_BATCH_SIZE": "1",
    # Forced-alignment batch size.
    "ALIGNER_BATCH_SIZE": "4",
    # Forced-alignment batch size used when ASR chunk packing is enabled.
    "ALIGN_LONG_CHUNK_BATCH_SIZE": "1",
    # Max generated tokens per ASR chunk.
    "ASR_MAX_NEW_TOKENS": "128",
    # Subprocess transcription token cap; usually matches ASR_MAX_NEW_TOKENS.
    "TRANSCRIPTION_MAX_NEW_TOKENS": "128",
    # Max seconds to wait for a subprocess ASR worker to load and report ready.
    "ASR_SUBPROCESS_READY_TIMEOUT_S": "600",
    # Generation penalty to reduce repeated ASR text.
    "ASR_REPETITION_PENALTY": "1.05",

    # --- ASR Native Timing Guards ---
    # Minimum native word/span duration accepted from ASR timestamps.
    "ASR_NATIVE_MIN_SPAN_MS": "80",
    # Reject native timing when too many spans have near-zero duration.
    "ASR_NATIVE_MAX_ZERO_RATIO": "0.55",
    # Reject native timing when repeated token spans dominate.
    "ASR_NATIVE_MAX_REPEAT_RATIO": "0.65",
    # Reject native timing when each timing item carries too much text.
    "ASR_NATIVE_MAX_CHARS_PER_ITEM": "12.0",

    # --- Boundary Refiner / ASR Chunking ---
    # Discard exported ASR wav chunks shorter than this many seconds.
    "ASR_CHUNK_MIN_DURATION_S": "0.25",
    # Reset sliding ASR text context when adjacent chunks are separated by this gap.
    "ASR_CONTEXT_RESET_GAP_S": "0.5",
    # Boundary Refiner is the current pre-ASR chunk planning path.
    # This is a feature-score grid fallback, not the source video frame rate.
    "BOUNDARY_FEATURE_FRAME_HOP_S": "0.02",
    "BOUNDARY_REFINER_ENABLED": "1",
    "BOUNDARY_REFINER_MODEL_PATH": "",
    "BOUNDARY_REFINER_BACKBONE": "transformers.Mamba2Model",
    "BOUNDARY_REFINER_THRESHOLD": "0.5",
    "BOUNDARY_PLANNER_MAX_CHUNK_S": "30.0",
    "BOUNDARY_PLANNER_TARGET_CHUNK_S": "9.0",
    "BOUNDARY_PLANNER_MIN_CHUNK_S": "0.4",
    "BOUNDARY_PLANNER_START_WEIGHT": "1.5",
    "BOUNDARY_PLANNER_TARGET_PADDING_S": "2.0",
    "BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT": "16",
    # 1 stores SpeechBoundary frame scores in the SpeechBoundary result. Boundary Refiner enables
    # this at runtime even when this explicit diagnostics flag stays off.
    "SPEECH_BOUNDARY_JA_EXPORT_FRAME_SCORES": "0",
    # 1 enables dropping very short low-energy spans before ASR (opt-in).
    "BOUNDARY_DROP_LOW_ENERGY_ENABLED": "0",
    # Spans shorter than this value (seconds) are candidates for dropping.
    "BOUNDARY_DROP_LOW_ENERGY_MIN_DURATION_S": "0.20",
    # Spans with RMS energy below this dBFS level are candidates for dropping.
    # Both duration and energy thresholds must be met (AND logic).
    "BOUNDARY_DROP_LOW_ENERGY_RMS_DBFS": "-40.0",
    # 1 caches SpeechBoundary frame score -> Boundary Planner outputs separately from ASR generation settings.
    "BOUNDARY_CACHE_ENABLED": "1",
    # Persistent boundary cache directory. Versioned as boundary-cache v1.
    "BOUNDARY_CACHE_DIR": "./temp/boundary-cache",

    # --- Alignment Retry & Refine ---
    # Max chunk length that hybrid alignment can force-align directly.
    "ALIGNMENT_HYBRID_FORCE_MAX_CHUNK": "24.0",
    # Chunk size used by coarse-to-fine alignment refinement.
    "ALIGNMENT_COARSE_REFINE_CHUNK": "18.0",
    # Maximum recursive refinement depth for difficult alignment chunks.
    "ALIGNMENT_MAX_REFINE_DEPTH": "2",
    # Fallback chunk size when alignment retries step down.
    "ALIGNMENT_STEP_DOWN_CHUNK": "6.0",

    # --- ASR Post-Processing ---
    # Similarity threshold for removing prompt/context leakage from ASR output.
    "ASR_CONTEXT_LEAK_SIMILARITY": "0.88",
    # Max gap, in seconds, for merging adjacent ASR fragments.
    "ASR_FRAGMENT_MERGE_MAX_GAP": "1.0",
    # Max combined text length after fragment merging.
    "ASR_FRAGMENT_MERGE_MAX_CHARS": "72",
    # Max combined duration after fragment merging.
    "ASR_FRAGMENT_MERGE_MAX_DURATION": "12.5",
    # Hard cap for merging ASR fragments into one subtitle candidate.
    "ASR_MERGE_HARD_MAX_DURATION": "9.0",

    # --- ASR QC / Conservative Filtering ---
    # 1 enables ASR text quality checks before translation.
    "ASR_QC_ENABLED": "1",
    # 1 allows QC to clear high-risk ASR text. Default 0 keeps QC diagnostic-only.
    "ASR_QC_DROP_UNCERTAIN": "0",
    # Adaptive precision drops high-risk ASR chunks while relaxing low-logprob true dialogue.
    "ASR_QC_ADAPTIVE_BASE_LOGPROB": "-0.7",
    "ASR_QC_ADAPTIVE_MIN_LOGPROB": "-0.95",
    "ASR_QC_ADAPTIVE_MAX_LOGPROB": "-0.55",
    "ASR_QC_ADAPTIVE_VIDEO_MAD_MULTIPLIER": "1.8",
    "ASR_QC_ADAPTIVE_VIDEO_MAX_LOGPROB": "-0.70",
    "ASR_QC_ADAPTIVE_HARD_NOSPEECH_THRESHOLD": "0.5",
    "ASR_QC_ADAPTIVE_HARD_COMPRESSION_THRESHOLD": "2.0",
    "ASR_QC_ADAPTIVE_HARD_MAX_CHARS_PER_SEC": "14.0",
    "ASR_QC_ADAPTIVE_HARD_REPEAT_RATIO": "0.45",
    # --- Subtitle Timings ---
    # Conservative hard maximum for one subtitle cue; 7s is the industry ceiling, 6.5s avoids hanging text.
    "MAX_SUBTITLE_DURATION": "6.5",
    # Soft split target before the hard cap.
    "SUBTITLE_SOFT_MAX_S": "5.5",
    # 1 enables word/punctuation based soft splitting before SRT output.
    "SUBTITLE_SOFT_SPLIT_ENABLED": "1",
    # Minimum displayed subtitle duration in seconds.
    "MIN_SUBTITLE_DURATION": "0.6",
    # Estimated Chinese reading speed in characters per second.
    "SUBTITLE_READING_CPS": "7.0",
    # Fixed reading-time buffer added to each subtitle.
    "SUBTITLE_READING_BASE": "0.35",
    # Max stretch ratio compared with the original segment duration.
    "SUBTITLE_DURATION_RATIO_CAP": "1.65",
    # Extra seconds allowed when extending short subtitles.
    "SUBTITLE_DURATION_GRACE": "0.9",
    # Extra reading-time weight for the Japanese line in bilingual mode.
    "SUBTITLE_BILINGUAL_SECONDARY_WEIGHT": "0.4",

    # --- LLM Translation Settings ---
    # Base URL for providers that expose an OpenAI-compatible API; DeepSeek by default.
    "OPENAI_COMPATIBILITY_BASE_URL": "https://api.deepseek.com",
    # Translation model name sent to the SDK client.
    "LLM_MODEL_NAME": "deepseek-v4-pro",
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
    "JOB_TEMP_DIR": "./temp/jobs",
    # 1 keeps optional quality reports requested by the web console.
    "KEEP_QUALITY_REPORT": "0",
    # Optional StageEvent sink: empty disables events; file:, tcp:, and memory are supported.
    "STAGE_EVENT_SINK": "",
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
    # Maximum allowed forced-alignment fallback ratio.
    "QC_MAX_ALIGN_FALLBACK": "0.20",
    # Maximum ASR generation failures before quality report warning.
    "QC_MAX_ASR_GENERATION_ERRORS": "0",
    # Maximum ASR generation overflow failures before quality report warning.
    "QC_MAX_ASR_GENERATION_OVERFLOWS": "0",

    # --- Speaker Diarization ---
    # 1 enables experimental speaker clustering.
    "EXPERIMENTAL_SPEAKER_DIARIZATION": "0",
    # SpeechBrain speaker embedding model.
    "SPEAKER_MODEL": "speechbrain/spkrec-ecapa-voxceleb",
    # Minimum segment duration for speaker embedding.
    "SPEAKER_MIN_DURATION": "0.5",
    # Agglomerative clustering distance threshold.
    "SPEAKER_CLUSTER_THRESHOLD": "0.5",
    # Maximum detected speaker clusters.
    "SPEAKER_MAX_CLUSTERS": "5",
    # 1 prefixes subtitles with speaker labels such as [S0].
    "SUBTITLE_SHOW_SPEAKER": "0",

    # --- F0 Gender Detection ---
    # 1 enables gender-aware word-level cue splitting; 0 disables gender splitting.
    "MULTI_CUE_SPLIT_ENABLED": "1",
    # 1 documents/enables F0 processing after forced alignment word timestamps.
    "F0_GENDER_POST_ALIGNMENT": "0",
    # Consecutive unknown-gender words needed to form a separate None group.
    "F0_GENDER_NONE_TOLERANCE": "3",
    # 1 lets short unknown-gender segments inherit gender from matching nearby anchors.
    "F0_GENDER_CARRYOVER_ENABLED": "1",
    # Max left-anchor-end to right-anchor-start gap for None segment carry-over.
    "F0_GENDER_CARRYOVER_MAX_GAP_S": "15.0",
    # Max duration of a None segment eligible for gender carry-over.
    "F0_GENDER_CARRYOVER_MAX_SEGMENT_S": "12.0",
    # Minimum duration for gender-turn split pieces.
    "SUBTITLE_MIN_DURATION_GENDER_TURN": "0.4",
    # 1 preserves word timestamps in final output artifacts.
    "KEEP_WORD_TIMESTAMPS": "0",
    # Median F0 threshold; below is treated as male, above/equal as female.
    "F0_THRESHOLD_HZ": "160",
    # High unvoiced/invalid F0 ratio marks gender as unknown.
    "F0_NAN_RATIO_THRESHOLD": "0.6",
    # pYIN word-level analysis window size in milliseconds.
    "F0_WORD_WINDOW_MS": "300",
    # RMS gate threshold in dB; lower-energy frames are excluded from gender.
    "F0_RMS_GATE_DB": "-45.0",
    # Minimum word-level gender span duration in milliseconds.
    "F0_WORD_MIN_SPAN_MS": "500",
    # 1 removes segments whose F0 gender is unknown before translation/SRT output.
    "F0_FILTER_NONE_SEGMENTS": "0",

    # --- Debug / Advanced ---
    # Test-only crash injection for translation resume tests.
    "_TEST_CRASH_TRANSLATION_BATCH": "",
}


STAGE_EVENT_SINK: str = os.getenv("STAGE_EVENT_SINK", "")


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
