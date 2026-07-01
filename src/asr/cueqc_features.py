"""CueQC Mamba v4 binary feature construction from ASR internals.

Three builders that turn an ASR candidate into the model feature arms:

* ``build_token_trace``  → [L, K_tok]   per-token features from teacher-forced logits
* ``build_decoder_stats`` → [K_dec]     chunk-level aggregate statistics
* ``build_structured_features`` → [S]   metadata (timing / text stats / boundary)

All outputs are ``float32`` and free of NaN/inf. An empty token sequence yields a
single zero row so the downstream Mamba always has ``L >= 1``. Feature names are frozen here and echoed into the checkpoint so the
runtime refiner can rebuild the exact same layout.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

# Per-token features. ``has_token_timestamp`` flags whether ``token_duration`` /
# ``token_gap_to_prev`` are real timestamps (1) or step-level proxies (0).
TOKEN_FEATURE_NAMES: list[str] = [
    "token_logprob",
    "token_entropy",
    "top1_top2_margin",
    "token_duration",
    "token_gap_to_prev",
    "relative_token_pos",
    "is_repeat_token",
    "is_punctuation",
    "is_special_token",
    "is_language_token",
    "is_numeric_token",
    "token_char_len",
    "has_token_timestamp",
]

# Chunk-level aggregate statistics describing overall ASR reliability.
DECODER_FEATURE_NAMES: list[str] = [
    "avg_logprob",
    "min_logprob",
    "std_logprob",
    "low_conf_token_ratio",
    "entropy_mean",
    "entropy_max",
    "margin_mean",
    "margin_min",
    "token_len",
    "text_char_len",
    "chars_per_second",
    "tokens_per_second",
    "repeat_token_ratio",
    "repeat_char_ratio",
    "repeat_bigram_ratio",
    "unique_char_ratio",
    "punct_ratio",
    "special_token_ratio",
    "language_token_ratio",
    "timestamp_coverage",
    "avg_token_duration",
    "max_token_gap",
]

# Structured metadata: timing, text statistics, optional boundary signal. Missing
# optional fields are filled with 0.0 so the vector layout is fixed.
STRUCTURED_FEATURE_NAMES: list[str] = [
    "start_s",
    "end_s",
    "duration_s",
    "num_chars",
    "num_tokens",
    "chars_per_second",
    "tokens_per_second",
    "asr_confidence",
    "chunk_index_norm",
    "prev_gap_s",
    "next_gap_s",
    "boundary_score_optional",
    "boundary_left_margin_optional",
    "boundary_right_margin_optional",
]

K_TOK = len(TOKEN_FEATURE_NAMES)
K_DEC = len(DECODER_FEATURE_NAMES)
S_DIM = len(STRUCTURED_FEATURE_NAMES)

# Punctuation / special-token character classes for the token trace flags.
_PUNCT_CHARS = set("、。！？，．.!?~「」『』（）()[][]【】〈〉《》…‥・ーー−~～")
# Japanese kana + common CJK + latin/digit ranges for language-token detection.
_KANA_RE = __import__("re").compile(r"[\u3040-\u30ff]")
_KANJI_RE = __import__("re").compile(r"[\u3400-\u9fff]")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clean_finite(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0 and ensure float32 contiguity."""
    out = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def _is_repeat(decoded_tokens: Sequence[str]) -> list[int]:
    """1 where a token equals the previous decoded token, else 0."""
    flags = [0] * len(decoded_tokens)
    for i in range(1, len(decoded_tokens)):
        if decoded_tokens[i] and decoded_tokens[i] == decoded_tokens[i - 1]:
            flags[i] = 1
    return flags


def _token_char_flags(decoded_token: str) -> tuple[int, int, int, int, int]:
    """Return (is_punct, is_special, is_language, is_numeric, char_len)."""
    if not decoded_token:
        return 0, 0, 0, 0, 0
    char_len = len(decoded_token)
    # Special tokens are HF-style: <...> or the ASR control tags.
    is_special = 1 if decoded_token.startswith("<") and decoded_token.endswith(">") else 0
    is_punct = 1 if decoded_token.strip() and all(ch in _PUNCT_CHARS for ch in decoded_token.strip()) else 0
    is_language = 1 if (_KANA_RE.search(decoded_token) or _KANJI_RE.search(decoded_token)) else 0
    is_numeric = 1 if any(ch.isdigit() for ch in decoded_token) else 0
    return is_punct, is_special, is_language, is_numeric, char_len


def build_token_trace(
    *,
    token_logprobs: Sequence[float],
    token_entropies: Sequence[float],
    token_margins: Sequence[float],
    decoded_tokens: Sequence[str],
    has_timestamps: bool = False,
    token_durations: Sequence[float] | None = None,
    token_gaps_to_prev: Sequence[float] | None = None,
) -> np.ndarray:
    """Construct the [L, K_tok] token-trace feature matrix.

    ``token_logprobs`` / ``token_entropies`` / ``token_margins`` come from the
    teacher-forced forward (one scalar per generated token). ``decoded_tokens``
    drives the repeat/punct/special/language/numeric flags and char length.

    If ``has_timestamps`` is False, ``token_duration`` uses a step-level proxy
    (1.0 per token) and ``token_gap_to_prev`` is 0, with ``has_token_timestamp``
    flagged 0. When true, real per-token durations/gaps are used.
    """
    n = len(token_logprobs)
    if n == 0:
        return np.zeros((1, K_TOK), dtype=np.float32)

    decoded_tokens = list(decoded_tokens)
    if len(decoded_tokens) != n:
        # Defensive: align lengths; pad/truncate decoded_tokens.
        decoded_tokens = (decoded_tokens + [""] * n)[:n]

    repeats = _is_repeat(decoded_tokens)
    flag_lists = [_token_char_flags(tok) for tok in decoded_tokens]
    durations = list(token_durations) if (token_durations and has_timestamps) else [1.0] * n
    gaps = list(token_gaps_to_prev) if (token_gaps_to_prev and has_timestamps) else [0.0] * n
    durations = (durations + [0.0] * n)[:n]
    gaps = (gaps + [0.0] * n)[:n]

    rows: list[list[float]] = []
    for i in range(n):
        is_punct, is_special, is_language, is_numeric, char_len = flag_lists[i]
        rows.append([
            _safe_float(token_logprobs[i]),
            _safe_float(token_entropies[i]),
            _safe_float(token_margins[i]),
            _safe_float(durations[i]),
            _safe_float(gaps[i]),
            (i / max(1, n - 1)) if n > 1 else 0.0,
            float(repeats[i]),
            float(is_punct),
            float(is_special),
            float(is_language),
            float(is_numeric),
            float(char_len),
            1.0 if has_timestamps else 0.0,
        ])
    return _clean_finite(np.array(rows, dtype=np.float32))


def build_decoder_stats(
    *,
    token_trace: np.ndarray,
    text: str,
    duration_s: float,
    has_timestamps: bool = False,
) -> np.ndarray:
    """Aggregate a sample's token-trace into the [K_dec] decoder-statistics vector."""
    token_trace = np.asarray(token_trace, dtype=np.float32)
    # Map feature names to columns for readability.
    idx = {name: col for col, name in enumerate(TOKEN_FEATURE_NAMES)}
    logprobs = token_trace[:, idx["token_logprob"]]
    entropies = token_trace[:, idx["token_entropy"]]
    margins = token_trace[:, idx["top1_top2_margin"]]
    durations = token_trace[:, idx["token_duration"]]
    gaps = token_trace[:, idx["token_gap_to_prev"]]
    is_repeat = token_trace[:, idx["is_repeat_token"]]
    is_punct = token_trace[:, idx["is_punctuation"]]
    is_special = token_trace[:, idx["is_special_token"]]
    is_language = token_trace[:, idx["is_language_token"]]
    char_lens = token_trace[:, idx["token_char_len"]]

    n_tok = token_trace.shape[0]
    text = str(text or "")
    text_chars = len(text)
    unique_chars = len(set(text))
    duration = max(duration_s, 1e-6)

    avg_logprob = float(np.mean(logprobs)) if n_tok else 0.0
    std_logprob = float(np.std(logprobs)) if n_tok else 0.0
    low_conf_ratio = float(np.mean(logprobs < -1.0)) if n_tok else 0.0
    entropy_mean = float(np.mean(entropies)) if n_tok else 0.0
    entropy_max = float(np.max(entropies)) if n_tok else 0.0
    margin_mean = float(np.mean(margins)) if n_tok else 0.0
    margin_min = float(np.min(margins)) if n_tok else 0.0

    repeat_token_ratio = float(np.mean(is_repeat)) if n_tok else 0.0
    punct_ratio = float(np.mean(is_punct)) if n_tok else 0.0
    special_ratio = float(np.mean(is_special)) if n_tok else 0.0
    language_ratio = float(np.mean(is_language)) if n_tok else 0.0

    # Character-level repetition over the cleaned text.
    compact = "".join(ch for ch in text if ch.isalnum())
    repeat_char_ratio = 0.0
    if compact:
        repeat_char_ratio = 1.0 - (len(set(compact)) / len(compact))
    bigrams = [compact[i : i + 2] for i in range(len(compact) - 1)]
    repeat_bigram_ratio = (1.0 - (len(set(bigrams)) / max(1, len(bigrams)))) if bigrams else 0.0
    unique_char_ratio = (unique_chars / text_chars) if text_chars else 0.0

    timestamp_coverage = 1.0 if has_timestamps else 0.0
    avg_token_duration = float(np.mean(durations)) if n_tok else 0.0
    # gaps[0] is always 0 by construction; consider only interior gaps.
    interior_gaps = gaps[1:] if n_tok > 1 else gaps
    max_token_gap = float(np.max(interior_gaps)) if interior_gaps.size else 0.0

    stats = [
        avg_logprob,
        float(np.min(logprobs)) if n_tok else 0.0,
        std_logprob,
        low_conf_ratio,
        entropy_mean,
        entropy_max,
        margin_mean,
        margin_min,
        float(n_tok),
        float(text_chars),
        text_chars / duration,
        n_tok / duration,
        repeat_token_ratio,
        repeat_char_ratio,
        repeat_bigram_ratio,
        unique_char_ratio,
        punct_ratio,
        special_ratio,
        language_ratio,
        timestamp_coverage,
        avg_token_duration,
        max_token_gap,
    ]
    # language_token_ratio captures the language-script token share (kana/kanji
    # etc.). Numeric tokens stay available via the token trace's is_numeric_token
    # column but are not duplicated as a decoder stat; vector length is fixed at K_DEC.
    assert len(stats) == K_DEC, f"decoder stats length {len(stats)} != {K_DEC}"
    return _clean_finite(np.array(stats, dtype=np.float32))


def build_structured_features(
    candidate: Mapping[str, Any],
    *,
    n_tokens: int | None = None,
    asr_confidence: float | None = None,
) -> np.ndarray:
    """Build the [S] structured-metadata vector from a CueQC candidate dict.

    Optional fields (boundary_score_*, asr_confidence) default to 0.0 when
    absent, keeping the layout fixed regardless of upstream availability.
    """
    text_features = candidate.get("text_features") if isinstance(candidate.get("text_features"), Mapping) else {}
    adjacency = candidate.get("adjacency") if isinstance(candidate.get("adjacency"), Mapping) else {}
    boundary = candidate.get("boundary") if isinstance(candidate.get("boundary"), Mapping) else {}
    asr_signals = candidate.get("asr_signals") if isinstance(candidate.get("asr_signals"), Mapping) else {}

    start_s = _safe_float(candidate.get("start"))
    end_s = _safe_float(candidate.get("end"), start_s)
    duration_s = max(_safe_float(candidate.get("duration_s"), end_s - start_s), 1e-6)
    num_chars = _safe_float(text_features.get("char_count"))
    num_tokens = float(n_tokens) if n_tokens is not None else 0.0
    chars_per_sec = num_chars / duration_s
    tokens_per_sec = num_tokens / duration_s
    confidence = asr_confidence
    if confidence is None:
        confidence = _optional_float(asr_signals.get("avg_logprob"))
    confidence_val = confidence if confidence is not None else 0.0
    chunk_index_norm = 0.0  # set by caller when batch context is available
    prev_gap_s = _safe_float(adjacency.get("prev_gap_s"))
    next_gap_s = _safe_float(adjacency.get("next_gap_s"))
    boundary_score = _safe_float(boundary.get("boundary_score"))
    boundary_left_margin = _safe_float(boundary.get("boundary_left_margin_optional"))
    boundary_right_margin = _safe_float(boundary.get("boundary_right_margin_optional"))

    vec = [
        start_s,
        end_s,
        duration_s,
        num_chars,
        num_tokens,
        chars_per_sec,
        tokens_per_sec,
        confidence_val,
        chunk_index_norm,
        prev_gap_s,
        next_gap_s,
        boundary_score,
        boundary_left_margin,
        boundary_right_margin,
    ]
    assert len(vec) == S_DIM, f"structured length {len(vec)} != {S_DIM}"
    return _clean_finite(np.array(vec, dtype=np.float32))
