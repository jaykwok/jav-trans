import re


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


def detect_speech_spans(audio_path: str) -> tuple[list[tuple[float, float]], str]:
    del audio_path
    return [], ""


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
            (max(start, span_start), min(clipped_end, span_end))
            for span_start, span_end in speech_spans
            if span_end > span_start and span_end > start and span_start < clipped_end
        ]
        spans = _merge_spans(spans)

    if spans:
        return _build_tokens_over_spans(tokens, spans), "aligner_vad_fallback", meta

    return _build_tokens_over_spans(tokens, [(start, clipped_end)]), "even_fallback", meta

