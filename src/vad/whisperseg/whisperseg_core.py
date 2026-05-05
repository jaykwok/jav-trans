# Vendored from WhisperJAV (https://github.com/meizhong986/WhisperJAV) — MIT License
# Original: whisperjav/modules/speech_segmentation/backends/whisperseg.py @ v1.8.12
# Local modifications: import paths, env-driven config, project logger

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from vad.base import SegmentationResult, SpeechSegment
from vad.whisperseg.postprocess import group_segments
from utils.model_paths import MODELS_ROOT, resolve_model_spec

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False

_HF_REPO_ID = "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx"
_HF_REVISION = "6ac29e2cbf2f4f8e9b639861766a8639dd666e9c"
_MODEL_FILENAME = "model.onnx"
_METADATA_FILENAME = "model_metadata.json"

_WHISPER_BASE_MODEL_ID = "openai/whisper-base"
_SAMPLE_RATE = 16000
_DEFAULT_METADATA: dict[str, Any] = {
    "whisper_model_name": _WHISPER_BASE_MODEL_ID,
    "frame_duration_ms": 20,
    "total_duration_ms": 30000,
}


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_bool(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _hf_cache_dir() -> str:
    return str(MODELS_ROOT)


class WhisperSegSpeechSegmenter:
    """WhisperSeg ONNX speech segmentation backend vendored from WhisperJAV."""

    def __init__(
        self,
        threshold: float | None = None,
        min_speech_duration_ms: int | None = None,
        min_silence_duration_ms: int | None = None,
        speech_pad_ms: int | None = None,
        max_speech_duration_s: float | None = None,
        chunk_threshold_s: float | None = None,
        max_group_duration_s: float | None = None,
        force_cpu: bool | None = None,
        num_threads: int = 1,
        model_path: str | None = None,
    ) -> None:
        self.threshold = float(
            threshold if threshold is not None else _env_float("WHISPERSEG_THRESHOLD", "0.25")
        )
        self.min_speech_duration_ms = int(
            min_speech_duration_ms
            if min_speech_duration_ms is not None
            else _env_int("WHISPERSEG_MIN_SPEECH_MS", "80")
        )
        self.min_silence_duration_ms = int(
            min_silence_duration_ms
            if min_silence_duration_ms is not None
            else _env_int("WHISPERSEG_MIN_SILENCE_MS", "80")
        )
        self.speech_pad_ms = int(
            speech_pad_ms if speech_pad_ms is not None else _env_int("WHISPERSEG_PAD_MS", "400")
        )
        self.force_cpu = bool(
            force_cpu if force_cpu is not None else _env_bool("WHISPERSEG_FORCE_CPU", "0")
        )
        self.num_threads = int(num_threads)
        self.model_path = model_path

        self.chunk_threshold_s = float(
            chunk_threshold_s
            if chunk_threshold_s is not None
            else _env_float("WHISPERSEG_CHUNK_THRESHOLD_S", "1.0")
        )

        self.max_group_duration_s = float(
            max_group_duration_s
            if max_group_duration_s is not None
            else _env_float("WHISPERSEG_MAX_GROUP_S", "5.0")
        )

        self.max_speech_duration_s = float(
            max_speech_duration_s
            if max_speech_duration_s is not None
            else _env_float("WHISPERSEG_MAX_SPEECH_S", "4.0")
        )

        self._session = None
        self._feature_extractor = None
        self._input_name: str | None = None
        self._output_names: list[str] | None = None
        self._metadata: dict[str, Any] | None = None
        self._frame_duration_ms: int = _DEFAULT_METADATA["frame_duration_ms"]
        self._chunk_duration_ms: int = _DEFAULT_METADATA["total_duration_ms"]
        self._chunk_samples: int = int(self._chunk_duration_ms * _SAMPLE_RATE / 1000)
        self._actual_device: str = "CPU"
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "whisperseg"

    @property
    def display_name(self) -> str:
        return "WhisperSeg (JA-ASMR)"

    def get_supported_sample_rates(self) -> list[int]:
        return [_SAMPLE_RATE]

    def _download_model(self) -> tuple[str, str | None]:
        if self.model_path and os.path.exists(self.model_path):
            sidecar = self.model_path.replace(".onnx", "_metadata.json")
            sidecar_path = sidecar if os.path.exists(sidecar) else None
            log.info(f"WhisperSeg using local model: {self.model_path}")
            return self.model_path, sidecar_path

        try:
            model_root = Path(
                resolve_model_spec(
                    None,
                    _HF_REPO_ID,
                    download=True,
                    revision=_HF_REVISION,
                    allow_patterns=[_MODEL_FILENAME, _METADATA_FILENAME],
                )
            )
            model_path = str(model_root / _MODEL_FILENAME)
            metadata_path = str(model_root / _METADATA_FILENAME)
            if not Path(metadata_path).exists():
                metadata_path = None
        except Exception as exc:
            raise ImportError(
                f"WhisperSeg failed to download model from HuggingFace Hub "
                f"({_HF_REPO_ID}@{_HF_REVISION[:8]}). Original error: {exc}"
            ) from exc

        log.info(f"WhisperSeg model resolved: {model_path}")
        return model_path, metadata_path

    def _load_feature_extractor(self):
        from transformers import WhisperFeatureExtractor

        whisper_model_name = self._metadata.get("whisper_model_name", _WHISPER_BASE_MODEL_ID)
        try:
            feature_root = resolve_model_spec(
                None,
                whisper_model_name,
                download=True,
                allow_patterns=[
                    "preprocessor_config.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "normalizer.json",
                ],
            )
        except Exception as exc:
            raise ImportError(
                f"WhisperSeg failed to download WhisperFeatureExtractor for "
                f"{whisper_model_name!r}. Original error: {exc}"
            ) from exc
        return WhisperFeatureExtractor.from_pretrained(feature_root)

    def _load_metadata(self, metadata_path: str | None) -> dict[str, Any]:
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    data = json.load(f)
                merged = dict(_DEFAULT_METADATA)
                merged.update(data)
                return merged
            except Exception as exc:
                log.warning(f"WhisperSeg metadata unreadable, using defaults: {exc}")
        return dict(_DEFAULT_METADATA)

    def _ensure_model(self) -> None:
        if self._session is not None:
            return

        with self._lock:
            if self._session is not None:
                return

            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError("WhisperSeg requires onnxruntime or onnxruntime-gpu.") from exc

            model_path, metadata_path = self._download_model()
            self._metadata = self._load_metadata(metadata_path)

            self._frame_duration_ms = int(self._metadata.get("frame_duration_ms", 20))
            self._chunk_duration_ms = int(self._metadata.get("total_duration_ms", 30000))
            self._chunk_samples = int(self._chunk_duration_ms * _SAMPLE_RATE / 1000)

            opts = ort.SessionOptions()
            available_providers = ort.get_available_providers()
            use_gpu = not self.force_cpu and "CUDAExecutionProvider" in available_providers
            if use_gpu:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._actual_device = "GPU (CUDA)"
                opts.inter_op_num_threads = self.num_threads
                opts.intra_op_num_threads = self.num_threads
            else:
                providers = ["CPUExecutionProvider"]
                self._actual_device = "CPU"
                if self.num_threads == 1:
                    optimal = max(1, multiprocessing.cpu_count() // 2)
                    opts.inter_op_num_threads = optimal
                    opts.intra_op_num_threads = optimal
                else:
                    opts.inter_op_num_threads = self.num_threads
                    opts.intra_op_num_threads = self.num_threads

            self._session = ort.InferenceSession(model_path, providers=providers, sess_options=opts)
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [output.name for output in self._session.get_outputs()]

            self._feature_extractor = self._load_feature_extractor()

            log.info(
                f"WhisperSeg ready: device={self._actual_device}, "
                f"chunk={self._chunk_duration_ms}ms, frame={self._frame_duration_ms}ms"
            )

    def _process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        if len(audio_chunk) < self._chunk_samples:
            audio_chunk = np.pad(
                audio_chunk,
                (0, self._chunk_samples - len(audio_chunk)),
                mode="constant",
            )
        elif len(audio_chunk) > self._chunk_samples:
            audio_chunk = audio_chunk[: self._chunk_samples]

        inputs = self._feature_extractor(
            audio_chunk,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
        )
        outputs = self._session.run(
            self._output_names,
            {self._input_name: inputs.input_features},
        )
        frame_logits = outputs[0][0]
        frame_probs = 1.0 / (1.0 + np.exp(-frame_logits))
        return frame_probs.astype(np.float32)

    def _audio_forward(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.mean(axis=0 if audio.shape[0] > audio.shape[1] else 1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) == 0:
            return np.zeros(0, dtype=np.float32)

        all_probs: list[np.ndarray] = []
        for i in range(0, len(audio), self._chunk_samples):
            all_probs.append(self._process_chunk(audio[i : i + self._chunk_samples]))

        if not all_probs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(all_probs)

    def _probs_to_segments(
        self,
        speech_probs: np.ndarray,
        audio_duration_sec: float,
    ) -> list[SpeechSegment]:
        if len(speech_probs) == 0:
            return []

        frame_ms = float(self._frame_duration_ms)
        threshold = float(self.threshold)
        neg_threshold = max(threshold - 0.15, 0.01)

        min_speech_frames = max(1, int(self.min_speech_duration_ms / frame_ms))
        min_silence_frames = max(1, int(self.min_silence_duration_ms / frame_ms))
        speech_pad_frames = max(0, int(self.speech_pad_ms / frame_ms))
        if self.max_speech_duration_s and self.max_speech_duration_s > 0:
            max_speech_frames = int(self.max_speech_duration_s * 1000.0 / frame_ms)
        else:
            max_speech_frames = len(speech_probs)

        triggered = False
        speeches: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        current_probs: list[float] = []
        temp_end = 0

        for i, prob_raw in enumerate(speech_probs):
            prob = float(prob_raw)

            if triggered:
                current_probs.append(prob)

            if prob >= threshold and not triggered:
                triggered = True
                current["start"] = i
                current_probs = [prob]
                continue

            if triggered and "start" in current:
                duration = i - current["start"]
                if duration > max_speech_frames:
                    current["end"] = current["start"] + max_speech_frames
                    if current_probs:
                        valid = current_probs[: current["end"] - current["start"]]
                        if valid:
                            current["avg_prob"] = float(np.mean(valid))
                    speeches.append(current)
                    current = {}
                    current_probs = []
                    triggered = False
                    temp_end = 0
                    continue

            if prob < neg_threshold and triggered:
                if not temp_end:
                    temp_end = i

                if i - temp_end >= min_silence_frames:
                    current["end"] = temp_end
                    if current["end"] - current["start"] >= min_speech_frames:
                        if current_probs:
                            valid = current_probs[: temp_end - current["start"]]
                            if valid:
                                current["avg_prob"] = float(np.mean(valid))
                        speeches.append(current)
                    current = {}
                    current_probs = []
                    triggered = False
                    temp_end = 0
            elif prob >= threshold and temp_end:
                temp_end = 0

        if triggered and "start" in current:
            current["end"] = len(speech_probs)
            if current["end"] - current["start"] >= min_speech_frames:
                if current_probs:
                    current["avg_prob"] = float(np.mean(current_probs))
                speeches.append(current)

        for idx, seg in enumerate(speeches):
            if idx == 0:
                seg["start"] = max(0, seg["start"] - speech_pad_frames)
            else:
                seg["start"] = max(speeches[idx - 1]["end"], seg["start"] - speech_pad_frames)
            if idx < len(speeches) - 1:
                seg["end"] = min(speeches[idx + 1]["start"], seg["end"] + speech_pad_frames)
            else:
                seg["end"] = min(len(speech_probs), seg["end"] + speech_pad_frames)

        results: list[SpeechSegment] = []
        for seg in speeches:
            start_sec = seg["start"] * frame_ms / 1000.0
            end_sec = min(seg["end"] * frame_ms / 1000.0, audio_duration_sec)
            if end_sec > start_sec:
                results.append(SpeechSegment(start=start_sec, end=end_sec))
        return results

    def segment(
        self,
        audio: np.ndarray | Path | str,
        sample_rate: int = _SAMPLE_RATE,
        **kwargs: Any,
    ) -> SegmentationResult:
        start_time = time.time()
        self._ensure_model()

        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        duration_sec = len(audio_data) / actual_sr if actual_sr > 0 else 0.0

        if actual_sr != _SAMPLE_RATE:
            audio_data = self._resample_audio(audio_data, actual_sr, _SAMPLE_RATE)

        try:
            probs = self._audio_forward(audio_data)
        except Exception as exc:
            log.error(f"WhisperSeg inference failed: {exc}", exc_info=True)
            return SegmentationResult(
                segments=[],
                groups=[],
                method=self.name,
                audio_duration_sec=duration_sec,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
            )

        segments = self._probs_to_segments(probs, duration_sec)
        groups = group_segments(
            segments,
            max_group_duration_s=self.max_group_duration_s,
            chunk_threshold_s=self.chunk_threshold_s,
        )

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration_sec,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    def _load_audio(self, audio: np.ndarray | Path | str, sample_rate: int) -> tuple[np.ndarray, int]:
        if isinstance(audio, np.ndarray):
            return audio, sample_rate

        audio_path = Path(audio) if isinstance(audio, str) else audio
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError("soundfile is required to load audio files.") from exc

        audio_data, actual_sr = sf.read(str(audio_path), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        return audio_data, int(actual_sr)

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        try:
            from scipy import signal

            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples).astype(audio.dtype)
        except ImportError:
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(audio), 1 / ratio)
            indices = np.clip(indices, 0, len(audio) - 1).astype(int)
            return audio[indices]

    def _get_parameters(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "max_speech_duration_s": self.max_speech_duration_s,
            "chunk_threshold_s": self.chunk_threshold_s,
            "max_group_duration_s": self.max_group_duration_s,
            "force_cpu": self.force_cpu,
            "num_threads": self.num_threads,
            "device": self._actual_device,
            "frame_duration_ms": self._frame_duration_ms,
            "chunk_duration_ms": self._chunk_duration_ms,
        }

    def cleanup(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session = None
                self._feature_extractor = None
                self._input_name = None
                self._output_names = None
                log.debug("WhisperSeg resources released")

    def __repr__(self) -> str:
        return f"WhisperSegSpeechSegmenter(threshold={self.threshold}, device={self._actual_device})"

