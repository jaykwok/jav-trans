from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio.loading import load_audio_16k_mono
from utils.model_paths import WHISPER_MODEL_PATH, resolve_model_spec
from whisper.model_backend import WHISPER_PRESETS

QWEN3_ASR_PRESETS = {
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
}


@dataclass(frozen=True)
class FeatureConfig:
    ptm: str = "whisper-ja-1.5b"
    frame_hop_s: float = 0.02
    n_mfcc: int = 40
    n_fft: int = 400
    feature_dim: int | None = None
    device: str = "cuda"
    dtype: str = "float16"
    revision: str | None = None
    model_path: str = ""
    download: bool = True
    attention: str = "sdpa"
    language: str = "Japanese"


@dataclass(frozen=True)
class CachedFeature:
    audio_id: str
    source: str
    audio_path: str
    feature_path: str
    duration_s: float
    frame_hop_s: float
    frame_count: int
    whisper_dim: int
    mfcc_dim: int
    ptm: str
    cache_key: str


def cache_key_for_audio(*, audio_path: Path, config: FeatureConfig) -> str:
    stat = audio_path.stat()
    payload = {
        "path": str(audio_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "config": asdict(config),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def extract_feature_bundle(
    *,
    audio_path: Path,
    config: FeatureConfig,
    whisper_extractor: Any | None = None,
) -> dict[str, Any]:
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    duration_s = len(audio) / sample_rate if sample_rate else 0.0
    mfcc = extract_mfcc(audio, sample_rate=sample_rate, config=config)
    if whisper_extractor is None:
        whisper = extract_ptm_features(audio, sample_rate=sample_rate, config=config)
    else:
        whisper = whisper_extractor.extract(audio, sample_rate=sample_rate)
    whisper, mfcc = align_feature_frames(
        whisper,
        mfcc,
        resize_ptm=is_low_frame_rate_ptm(config.ptm),
    )
    return {
        "whisper": whisper.astype("float32", copy=False),
        "mfcc": mfcc.astype("float32", copy=False),
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
    }


def torch_dtype_from_config(dtype: str):
    import torch

    normalized = dtype.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


class WhisperEncoderFeatureExtractor:
    def __init__(self, config: FeatureConfig) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if config.ptm not in WHISPER_PRESETS:
            raise ValueError(f"unsupported whisper PTM preset: {config.ptm}")
        self.config = config
        preset = WHISPER_PRESETS[config.ptm]
        self.model_path = resolve_model_spec(
            config.model_path or WHISPER_MODEL_PATH or None,
            str(preset["repo_id"]),
            download=config.download,
            revision=config.revision,
        )
        if not config.download and self.model_path == str(preset["repo_id"]):
            raise FileNotFoundError(
                f"{config.ptm} is not available locally. Put the model under "
                f"models/{str(preset['repo_id']).replace('/', '-')} or pass --model-path, "
                "or rerun without --no-download."
            )
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        self.torch_dtype = torch_dtype_from_config(config.dtype)
        self.device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()

    def extract(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        return self.extract_batch([audio], sample_rate=sample_rate)[0]

    def extract_batch(self, audios: list[np.ndarray], *, sample_rate: int) -> list[np.ndarray]:
        import torch

        if not audios:
            return []
        inputs = self.processor(
            [np.asarray(audio, dtype=np.float32) for audio in audios],
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=self.device, dtype=self.torch_dtype)
        with torch.inference_mode():
            encoder_output = self.model.model.encoder(input_features=input_features)
            hidden = encoder_output.last_hidden_state.detach().float().cpu().numpy()
        return [np.asarray(item, dtype=np.float32) for item in hidden]

    def close(self) -> None:
        import torch

        self.model = None
        self.processor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def normalize_ptm_name(ptm: str) -> str:
    return (ptm or "").strip()


def is_qwen3_asr_ptm(ptm: str) -> bool:
    normalized = normalize_ptm_name(ptm)
    lowered = normalized.lower()
    return lowered in QWEN3_ASR_PRESETS or lowered.startswith("qwen/qwen3-asr-")


def is_low_frame_rate_ptm(ptm: str) -> bool:
    return is_qwen3_asr_ptm(ptm)


def qwen3_asr_repo_id(ptm: str) -> str:
    normalized = normalize_ptm_name(ptm)
    return QWEN3_ASR_PRESETS.get(normalized.lower(), normalized)


def qwen3_asr_audio_output_lengths(input_lengths: Any) -> Any:
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class Qwen3AsrFeatureExtractor:
    def __init__(self, config: FeatureConfig) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self.config = config
        self.repo_id = qwen3_asr_repo_id(config.ptm)
        self.model_path = resolve_model_spec(
            config.model_path or None,
            self.repo_id,
            download=config.download,
            revision=config.revision,
        )
        if not config.download and self.model_path == self.repo_id:
            raise FileNotFoundError(
                f"{config.ptm} is not available locally. Put the model under "
                f"models/{self.repo_id.replace('/', '-')} or pass --model-path, "
                "or rerun without --no-download."
            )
        self.torch_dtype = torch_dtype_from_config(config.dtype)
        self.device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
        model_kwargs: dict[str, Any] = {
            "dtype": self.torch_dtype,
            "device_map": str(self.device),
            "max_inference_batch_size": 1,
            "max_new_tokens": 1,
        }
        if config.attention and config.attention != "sdpa":
            model_kwargs["attn_implementation"] = config.attention
        self.handle = Qwen3ASRModel.from_pretrained(self.model_path, **model_kwargs)
        self.model = self.handle.model
        self.processor = self.handle.processor
        self.model.eval()

    def extract(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        return self.extract_batch([audio], sample_rate=sample_rate)[0]

    def extract_batch(self, audios: list[np.ndarray], *, sample_rate: int) -> list[np.ndarray]:
        import torch

        if not audios:
            return []
        if sample_rate != 16000:
            raise ValueError(f"Qwen3-ASR feature extraction expects 16kHz audio, got {sample_rate}")
        prompt = self.handle._build_text_prompt(context="", force_language=self.config.language or None)
        inputs = self.processor(
            text=[prompt] * len(audios),
            audio=[np.asarray(audio, dtype=np.float32) for audio in audios],
            return_tensors="pt",
            padding=True,
        )
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if key == "input_features":
                    moved[key] = value.to(device=self.device, dtype=self.torch_dtype)
                else:
                    moved[key] = value.to(device=self.device)
            else:
                moved[key] = value
        input_lengths = moved["feature_attention_mask"].sum(dim=1)
        output_lengths = qwen3_asr_audio_output_lengths(input_lengths).detach().cpu().tolist()
        with torch.inference_mode():
            audio_features = self.model.thinker.get_audio_features(
                input_features=moved["input_features"],
                feature_attention_mask=moved["feature_attention_mask"],
            )
            hidden = audio_features.detach().float().cpu().numpy()
        result: list[np.ndarray] = []
        offset = 0
        for length in output_lengths:
            length_int = int(length)
            result.append(np.asarray(hidden[offset : offset + length_int], dtype=np.float32))
            offset += length_int
        return result

    def close(self) -> None:
        import torch

        self.model = None
        self.processor = None
        self.handle = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def build_ptm_feature_extractor(config: FeatureConfig) -> Any:
    if config.ptm in WHISPER_PRESETS:
        return WhisperEncoderFeatureExtractor(config)
    if is_qwen3_asr_ptm(config.ptm):
        return Qwen3AsrFeatureExtractor(config)
    raise ValueError(
        f"unsupported PTM feature extractor: {config.ptm}. "
        f"Whisper presets={sorted(WHISPER_PRESETS)} qwen presets={sorted(QWEN3_ASR_PRESETS)}"
    )


def extract_mfcc(
    audio: np.ndarray,
    *,
    sample_rate: int,
    config: FeatureConfig,
) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for FusionVAD-JA MFCC extraction") from exc
    hop_length = max(1, int(round(config.frame_hop_s * sample_rate)))
    mfcc = librosa.feature.mfcc(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=hop_length,
    )
    return np.asarray(mfcc.T, dtype=np.float32)


def extract_whisper_encoder_features(
    audio: np.ndarray,
    *,
    sample_rate: int,
    config: FeatureConfig,
) -> np.ndarray:
    extractor = WhisperEncoderFeatureExtractor(config)
    try:
        return extractor.extract(audio, sample_rate=sample_rate)
    finally:
        extractor.close()


def extract_ptm_features(
    audio: np.ndarray,
    *,
    sample_rate: int,
    config: FeatureConfig,
) -> np.ndarray:
    extractor = build_ptm_feature_extractor(config)
    try:
        return extractor.extract(audio, sample_rate=sample_rate)
    finally:
        extractor.close()


def resize_feature_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    source = np.asarray(features, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={source.shape}")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    source_frames = int(source.shape[0])
    if source_frames <= 0:
        raise ValueError("cannot resize empty feature array")
    if source_frames == target_frames:
        return np.ascontiguousarray(source, dtype=np.float32)
    if source_frames == 1:
        return np.ascontiguousarray(np.repeat(source, target_frames, axis=0), dtype=np.float32)
    positions = np.linspace(0, source_frames - 1, target_frames, dtype=np.float32)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, source_frames - 1)
    weight = (positions - left).reshape(-1, 1).astype(np.float32)
    resized = source[left] * (1.0 - weight) + source[right] * weight
    return np.ascontiguousarray(resized, dtype=np.float32)


def align_feature_frames(
    whisper: np.ndarray,
    mfcc: np.ndarray,
    *,
    resize_ptm: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if resize_ptm and int(whisper.shape[0]) != int(mfcc.shape[0]):
        whisper = resize_feature_frames(whisper, int(mfcc.shape[0]))
    frame_count = min(int(whisper.shape[0]), int(mfcc.shape[0]))
    if frame_count <= 0:
        raise ValueError("feature extraction produced no frames")
    return (
        np.ascontiguousarray(whisper[:frame_count], dtype=np.float32),
        np.ascontiguousarray(mfcc[:frame_count], dtype=np.float32),
    )


def write_feature_cache(
    *,
    output_dir: Path,
    audio_id: str,
    source: str,
    audio_path: Path,
    config: FeatureConfig,
    bundle: dict[str, Any],
) -> CachedFeature:
    output_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key_for_audio(audio_path=audio_path, config=config)
    feature_path = output_dir / f"{audio_id}-{key}.npz"
    np.savez_compressed(
        feature_path,
        whisper=bundle["whisper"],
        mfcc=bundle["mfcc"],
        duration_s=np.asarray([bundle["duration_s"]], dtype=np.float32),
        sample_rate=np.asarray([bundle["sample_rate"]], dtype=np.int32),
    )
    whisper = bundle["whisper"]
    mfcc = bundle["mfcc"]
    return CachedFeature(
        audio_id=audio_id,
        source=source,
        audio_path=str(audio_path),
        feature_path=str(feature_path),
        duration_s=float(bundle["duration_s"]),
        frame_hop_s=config.frame_hop_s,
        frame_count=int(whisper.shape[0]),
        whisper_dim=int(whisper.shape[1]),
        mfcc_dim=int(mfcc.shape[1]),
        ptm=config.ptm,
        cache_key=key,
    )


def load_cached_feature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        whisper = np.asarray(data["whisper"], dtype=np.float32)
        mfcc = np.asarray(data["mfcc"], dtype=np.float32)
    return whisper, mfcc
