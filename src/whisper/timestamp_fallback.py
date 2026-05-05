import contextlib
import io
import os
import re
from pathlib import Path
from threading import Lock

_VAD_MIN_OFF_S = float(os.getenv("VAD_MIN_OFF", "0.1"))
_VAD_PAD_S = float(os.getenv("VAD_PAD", "0.15"))
_TIMESTAMP_VAD_ONSET = float(os.getenv("TIMESTAMP_VAD_ONSET", "0.5"))
_TIMESTAMP_VAD_MIN_SPEECH_S = float(os.getenv("TIMESTAMP_VAD_MIN_SPEECH", "0.25"))
_SILERO_MODEL_CACHE = None
_SILERO_MODEL_LOCK = Lock()
_TEN_VAD_ENABLED = os.getenv("TEN_VAD_BACKEND", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _clean_text(text: str) -> str:
    cleaned = (text or "").replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"[ \t]+", " ", cleaned)


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\S+|.", text) if token.strip()]


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []

    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _merge_close_spans(
    spans: list[tuple[float, float]], max_gap_s: float
) -> list[tuple[float, float]]:
    if not spans:
        return []

    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not merged or start - merged[-1][1] > max_gap_s:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _pad_spans(
    spans: list[tuple[float, float]], pad_s: float, duration_s: float
) -> list[tuple[float, float]]:
    if not spans:
        return []
    return [
        (max(0.0, start - pad_s), min(duration_s, end + pad_s))
        for start, end in spans
        if end > start
    ]


def _load_silero_vad():
    global _SILERO_MODEL_CACHE
    if _SILERO_MODEL_CACHE is not None:
        return _SILERO_MODEL_CACHE

    with _SILERO_MODEL_LOCK:
        if _SILERO_MODEL_CACHE is not None:
            return _SILERO_MODEL_CACHE

    import torch
    from utils.model_paths import PROJECT_ROOT

    torch_home = Path(os.getenv("TORCH_HOME", "./temp/torch")).expanduser()
    if not torch_home.is_absolute():
        torch_home = PROJECT_ROOT / torch_home
    torch_home = torch_home.resolve()
    os.environ["TORCH_HOME"] = str(torch_home)
    hub_dir = torch_home / "hub"
    cached_repo = hub_dir / "snakers4_silero-vad_master"
    load_kwargs = {
        "repo_or_dir": str(cached_repo) if cached_repo.exists() else "snakers4/silero-vad",
        "model": "silero_vad",
        "force_reload": False,
        "onnx": False,
        "trust_repo": True,
    }
    if cached_repo.exists():
        load_kwargs["source"] = "local"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        vad_model, vad_utils = torch.hub.load(**load_kwargs)
    _SILERO_MODEL_CACHE = (vad_model, vad_utils[0])
    return _SILERO_MODEL_CACHE


def _load_audio_for_vad(audio_path: str):
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.contiguous()
    if sample_rate not in {8000, 16000}:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    return waveform, sample_rate


def _detect_speech_spans_ten_vad(audio_path: str) -> tuple[list[tuple[float, float]], str]:
    try:
        import numpy as np
        import torchaudio
        from ten_vad import TenVad

        waveform, sample_rate = _load_audio_for_vad(audio_path)
        if waveform.numel() == 0:
            return [], "empty audio"
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        samples = waveform.detach().cpu().numpy()
        samples = np.squeeze(samples)
        if samples.size == 0:
            return [], "empty audio"
        if samples.dtype != np.int16:
            if np.issubdtype(samples.dtype, np.floating):
                samples = np.clip(samples, -1.0, 1.0)
                samples = (samples * 32767.0).astype(np.int16)
            else:
                samples = np.clip(samples, -32768, 32767).astype(np.int16)
        samples = np.ascontiguousarray(samples)

        hop_size = 256
        vad = TenVad(hop_size, _TIMESTAMP_VAD_ONSET)
        spans: list[tuple[float, float]] = []
        active_start: int | None = None
        frame_count = int(len(samples) // hop_size)

        for frame_index in range(frame_count):
            frame_start = frame_index * hop_size
            frame = samples[frame_start : frame_start + hop_size]
            _prob, flag = vad.process(frame)
            if int(flag) == 1:
                if active_start is None:
                    active_start = frame_index
            elif active_start is not None:
                spans.append(
                    (
                        active_start * hop_size / sample_rate,
                        frame_index * hop_size / sample_rate,
                    )
                )
                active_start = None

        if active_start is not None:
            spans.append(
                (
                    active_start * hop_size / sample_rate,
                    frame_count * hop_size / sample_rate,
                )
            )

        duration_s = len(samples) / sample_rate
        spans = _merge_close_spans(spans, _VAD_MIN_OFF_S)
        spans = _pad_spans(spans, _VAD_PAD_S, duration_s)
        spans = _merge_close_spans(spans, _VAD_MIN_OFF_S)
        return spans, ""
    except Exception as exc:
        return [], str(exc)


def detect_speech_spans(audio_path: str) -> tuple[list[tuple[float, float]], str]:
    if _TEN_VAD_ENABLED:
        return _detect_speech_spans_ten_vad(audio_path)

    try:
        waveform, sample_rate = _load_audio_for_vad(audio_path)
        if waveform.numel() == 0:
            return [], "empty audio"

        vad_model, get_speech_timestamps = _load_silero_vad()
        timestamps = get_speech_timestamps(
            waveform,
            model=vad_model,
            sampling_rate=sample_rate,
            threshold=_TIMESTAMP_VAD_ONSET,
            min_speech_duration_ms=max(1, int(_TIMESTAMP_VAD_MIN_SPEECH_S * 1000)),
            min_silence_duration_ms=max(0, int(_VAD_MIN_OFF_S * 1000)),
            speech_pad_ms=max(0, int(_VAD_PAD_S * 1000)),
            return_seconds=True,
        )

        spans = []
        for item in timestamps:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
            if end > start:
                spans.append((start, end))
        return _merge_spans(spans), ""
    except Exception as exc:
        return [], str(exc)


def _project_offset_to_time(offset: float, spans: list[tuple[float, float]]) -> float:
    if not spans:
        return 0.0
    if offset <= 0:
        return spans[0][0]

    remaining = offset
    for start, end in spans:
        span_duration = end - start
        if remaining <= span_duration:
            return start + remaining
        remaining -= span_duration
    return spans[-1][1]


def _build_tokens_over_spans(tokens: list[str], spans: list[tuple[float, float]]) -> list[dict]:
    if not tokens or not spans:
        return []

    total_chars = sum(max(1, len(token.strip())) for token in tokens)
    total_duration = sum(max(0.0, end - start) for start, end in spans)
    if total_chars <= 0 or total_duration <= 0:
        return []

    cursor = 0.0
    words: list[dict] = []
    for idx, token in enumerate(tokens):
        weight = max(1, len(token.strip()))
        start_offset = cursor
        end_offset = (
            total_duration
            if idx == len(tokens) - 1
            else min(total_duration, cursor + total_duration * (weight / total_chars))
        )
        word_start = _project_offset_to_time(start_offset, spans)
        word_end = _project_offset_to_time(end_offset, spans)
        if word_end < word_start:
            word_end = word_start
        words.append({"start": word_start, "end": word_end, "word": token})
        cursor = end_offset
    return words


def build_word_timestamps_fallback(
    text: str,
    start: float,
    end: float,
    audio_path: str | None = None,
) -> tuple[list[dict], str, dict]:
    cleaned = _clean_text(text)
    tokens = _tokenize(cleaned)
    if not tokens:
        return [], "empty", {"speech_span_count": 0, "vad_error": ""}

    clipped_end = max(start, end)
    spans: list[tuple[float, float]] = []
    meta = {"speech_span_count": 0, "vad_error": ""}

    if audio_path:
        speech_spans, vad_error = detect_speech_spans(audio_path)
        meta["speech_span_count"] = len(speech_spans)
        meta["vad_error"] = vad_error
        spans = [
            (max(start, start + span_start), min(clipped_end, start + span_end))
            for span_start, span_end in speech_spans
            if start + span_end > start + span_start
        ]
        spans = _merge_spans(spans)

    if spans:
        return _build_tokens_over_spans(tokens, spans), "aligner_vad_fallback", meta

    return _build_tokens_over_spans(tokens, [(start, clipped_end)]), "even_fallback", meta

