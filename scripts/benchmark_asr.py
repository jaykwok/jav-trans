import argparse
import json
import os
import re
import statistics
import sys
import time
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.config import load_config


_ALIGN_MODE_RE = re.compile(r"Alignment 模式: (?P<mode>[\w_]+)")
_CHUNK_COUNT_RE = re.compile(r"切分完成：共 (?P<count>\d+) 个处理块")
_RAW_SEGMENTS_RE = re.compile(r"识别完成：共 (?P<count>\d+) 个粗粒度片段")
_VAD_SPANS_RE = re.compile(r"Alignment VAD 回退语音区间: (?P<count>\d+)")
_STAGE_TIMING_RE = re.compile(r"ASR 阶段耗时: (?P<label>[^=]+)=(?P<seconds>\d+(?:\.\d+)?)s")


def _project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _project_relative(path: str | Path | None) -> str | None:
    if path is None:
        return None
    raw = str(path)
    if not raw:
        return raw
    normalized = raw.replace("\\", "/")
    root_pattern = re.compile(
        re.escape(PROJECT_ROOT.resolve().as_posix()) + r"/?",
        re.IGNORECASE,
    )
    normalized = root_pattern.sub("", normalized)
    if normalized != raw.replace("\\", "/"):
        return normalized or "."
    candidate = Path(raw)
    try:
        if candidate.is_absolute():
            return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return raw.replace("\\", "/")
    return raw.replace("\\", "/")


def _relativize_payload_paths(value):
    if isinstance(value, dict):
        return {key: _relativize_payload_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_payload_paths(item) for item in value]
    if isinstance(value, str):
        return _project_relative(value)
    return value


def _get_wav_duration(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / rate if rate else 0.0


def _build_log_stats(log_lines: list[str]) -> dict:
    align_modes: dict[str, int] = {}
    total_vad_spans = 0
    chunk_count = None
    raw_segments = None
    sentinel_hits = 0
    step_down_success = 0
    step_down_fallback = 0
    stage_timings: dict[str, float] = {}

    for line in log_lines:
        if "Alignment 哨兵触发" in line:
            sentinel_hits += 1
        if "Alignment 降级成功" in line:
            step_down_success += 1
        if "Alignment 降级失败" in line or "Alignment 降级后仍异常" in line:
            step_down_fallback += 1

        align_match = _ALIGN_MODE_RE.search(line)
        if align_match:
            mode = align_match.group("mode")
            align_modes[mode] = align_modes.get(mode, 0) + 1

        chunk_match = _CHUNK_COUNT_RE.search(line)
        if chunk_match:
            chunk_count = int(chunk_match.group("count"))

        raw_match = _RAW_SEGMENTS_RE.search(line)
        if raw_match:
            raw_segments = int(raw_match.group("count"))

        vad_match = _VAD_SPANS_RE.search(line)
        if vad_match:
            total_vad_spans += int(vad_match.group("count"))

        timing_match = _STAGE_TIMING_RE.search(line)
        if timing_match:
            stage_timings[timing_match.group("label")] = float(timing_match.group("seconds"))

    return {
        "chunk_count": chunk_count,
        "raw_segments": raw_segments,
        "alignment_modes": align_modes,
        "sentinel_hits": sentinel_hits,
        "step_down_success": step_down_success,
        "step_down_fallback": step_down_fallback,
        "total_vad_spans": total_vad_spans,
        "stage_timings": stage_timings,
    }


def _build_segment_stats(segments: list[dict]) -> dict:
    durations = [max(0.0, float(seg["end"]) - float(seg["start"])) for seg in segments]
    chars = [len((seg.get("text") or "").strip()) for seg in segments]
    non_empty_chars = [count for count in chars if count > 0]

    return {
        "segment_count": len(segments),
        "total_chars": sum(chars),
        "avg_segment_duration_s": statistics.mean(durations) if durations else 0.0,
        "median_segment_duration_s": statistics.median(durations) if durations else 0.0,
        "avg_chars_per_segment": statistics.mean(non_empty_chars) if non_empty_chars else 0.0,
    }


def main() -> None:
    load_config()
    if os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = str(Path(os.getenv("HF_HOME")).resolve())
    if os.getenv("TORCH_HOME"):
        os.environ["TORCH_HOME"] = str(Path(os.getenv("TORCH_HOME")).resolve())
    if os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--asr-model-id", dest="asr_model_id")
    parser.add_argument("--asr-model-path", dest="asr_model_path")
    parser.add_argument("--aligner-model-id", dest="aligner_model_id")
    parser.add_argument("--aligner-model-path", dest="aligner_model_path")
    parser.add_argument("--asr-qc-enabled", choices=["0", "1"])
    parser.add_argument("--asr-recovery-enabled", choices=["0", "1"])
    args = parser.parse_args()

    audio_path = str(_project_path(args.audio))
    json_out = _project_path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["ASR_BACKEND"] = args.backend
    if args.asr_model_id:
        os.environ["ASR_MODEL_ID"] = args.asr_model_id
    if args.asr_model_path is not None:
        os.environ["ASR_MODEL_PATH"] = args.asr_model_path
    if args.aligner_model_id:
        os.environ["ALIGNER_MODEL_ID"] = args.aligner_model_id
    if args.aligner_model_path is not None:
        os.environ["ALIGNER_MODEL_PATH"] = args.aligner_model_path
    if args.asr_qc_enabled is not None:
        os.environ["ASR_QC_ENABLED"] = args.asr_qc_enabled
    if args.asr_recovery_enabled is not None:
        os.environ["ASR_RECOVERY_ENABLED"] = args.asr_recovery_enabled

    import torch
    from whisper import pipeline as asr

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    progress_lines: list[str] = []

    def _on_stage(message: str) -> None:
        progress_lines.append(message)
        print(message, flush=True)

    audio_duration_s = _get_wav_duration(audio_path)
    started = time.perf_counter()
    segments, log_lines, asr_details = asr.transcribe_and_align(
        audio_path,
        device,
        on_stage=_on_stage,
        include_details=True,
    )
    elapsed_s = time.perf_counter() - started

    result = {
        "backend": args.backend,
        "backend_label": asr.get_backend_label(),
        "asr_model_id": os.getenv("ASR_MODEL_ID"),
        "asr_model_path": os.getenv("ASR_MODEL_PATH", ""),
        "aligner_model_id": os.getenv("ALIGNER_MODEL_ID"),
        "aligner_model_path": os.getenv("ALIGNER_MODEL_PATH", ""),
        "ASR_CONTEXT": os.getenv("ASR_CONTEXT", ""),
        "asr_qc_enabled": os.getenv("ASR_QC_ENABLED", ""),
        "asr_recovery_enabled": os.getenv("ASR_RECOVERY_ENABLED", ""),
        "audio_path": audio_path,
        "audio_duration_s": audio_duration_s,
        "elapsed_s": elapsed_s,
        "seconds_per_audio_minute": (elapsed_s / (audio_duration_s / 60.0)) if audio_duration_s > 0 else None,
        "realtime_factor": (elapsed_s / audio_duration_s) if audio_duration_s > 0 else None,
        "device": device,
        "segment_stats": _build_segment_stats(segments),
        "segments": segments,
        "log_stats": _build_log_stats(log_lines),
        "asr_details": asr_details,
        "log_lines": log_lines,
        "stage_messages": progress_lines,
    }

    json_out.write_text(
        json.dumps(_relativize_payload_paths(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "backend": result["backend"],
        "elapsed_s": round(result["elapsed_s"], 2),
        "segment_count": result["segment_stats"]["segment_count"],
        "alignment_modes": result["log_stats"]["alignment_modes"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


