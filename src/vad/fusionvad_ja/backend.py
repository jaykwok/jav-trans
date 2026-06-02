from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from audio.loading import load_audio_16k_mono
from vad.base import SegmentationResult, SpeechSegment
from vad.fusionvad_ja.dataset import frame_count
from vad.fusionvad_ja.features import (
    FeatureConfig,
    align_feature_frames,
    build_ptm_feature_extractor,
    extract_mfcc,
    is_low_frame_rate_ptm,
)
from vad.fusionvad_ja.model import AdditionFusionBiLSTM, AdditionFusionEndpointBiLSTM
from vad.fusionvad_ja.model import AdditionFusionImitationBiLSTM
from vad.whisperseg.postprocess import group_segments


DEFAULT_CHECKPOINT = str(
    Path(__file__).resolve().parent
    / "checkpoints"
    / "fusionvad_ja_v1_19b_splitcut_touch4096_endpoint_refiner.pt"
)
DEFAULT_IMITATION_CHECKPOINT = (
    "datasets/train/fusionvad-ja/v1-21/qwen3-asr-0.6b-full29239/"
    "imitation-head-dropgaponly-poswin128-balanced-resizedtarget-batch8-lr2e-4-steps512/"
    "fusionvad_ja_imitation_head.pt"
)
DEFAULT_MODEL_PATH = "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame"
DEFAULT_OPERATING_POINT = "v1.19b-splitcut-touch4096-speech0.200-cut0.500-pad0.2"


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: str) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return int(float(default))


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _model_device(requested: str):
    import torch

    value = requested.strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        value = "cpu"
    return torch.device(value)


def _load_addition_model(checkpoint_path: Path, *, device):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config") or {})
    model_type = str(checkpoint.get("model_type") or "addition_bilstm")
    if model_type == "addition_endpoint_bilstm":
        model = AdditionFusionEndpointBiLSTM(
            whisper_dim=int(checkpoint["whisper_dim"]),
            mfcc_dim=int(checkpoint["mfcc_dim"]),
            fusion_dim=int(config.get("fusion_dim", 256)),
            hidden_dim=int(config.get("hidden_dim", 192)),
            layers=int(config.get("layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
    elif model_type in {"addition_bilstm", ""}:
        model = AdditionFusionBiLSTM(
            whisper_dim=int(checkpoint["whisper_dim"]),
            mfcc_dim=int(checkpoint["mfcc_dim"]),
            fusion_dim=int(config.get("fusion_dim", 256)),
            hidden_dim=int(config.get("hidden_dim", 192)),
            layers=int(config.get("layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
    else:
        raise ValueError(f"unsupported FusionVAD-JA checkpoint model_type={model_type!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, config, model_type


def _load_imitation_model(checkpoint_path: Path, *, device):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config") or {})
    model_type = str(checkpoint.get("model_type") or "")
    if model_type and model_type != "addition_imitation_bilstm":
        raise ValueError(f"unsupported FusionVAD-JA imitation checkpoint model_type={model_type!r}")
    model = AdditionFusionImitationBiLSTM(
        whisper_dim=int(checkpoint["whisper_dim"]),
        mfcc_dim=int(checkpoint["mfcc_dim"]),
        fusion_dim=int(config.get("fusion_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 192)),
        layers=int(config.get("layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, config, model_type or "addition_imitation_bilstm"


def _sigmoid_outputs(logits):
    import torch

    if isinstance(logits, dict):
        return {
            name: torch.sigmoid(value).detach().float().cpu().numpy().reshape(-1)
            for name, value in logits.items()
        }
    return {"speech": torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1)}


def _apply_cut_gate(
    speech_probs: np.ndarray,
    cut_probs: np.ndarray | None,
    *,
    cut_threshold: float,
    apply_cut: bool,
) -> np.ndarray:
    if not apply_cut or cut_probs is None:
        return speech_probs
    frame_total = min(int(speech_probs.size), int(cut_probs.size))
    if frame_total <= 0:
        return speech_probs
    gated = speech_probs.copy()
    active = gated[:frame_total]
    active[cut_probs[:frame_total] >= cut_threshold] = 0.0
    return gated


def _first_parameter_device_dtype(model) -> tuple[str, str]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "unknown", "unknown"
    try:
        parameter = next(parameters())
    except StopIteration:
        return "none", "none"
    except Exception as exc:  # pragma: no cover - defensive logging only
        return "error", type(exc).__name__
    return str(parameter.device), str(parameter.dtype)


def _padded_frames(values: np.ndarray, *, pad_frames: int) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if pad_frames <= 0 or mask.size == 0:
        return mask.astype(np.int8, copy=False)
    out = mask.copy()
    active = np.flatnonzero(mask)
    for index in active:
        start = max(0, int(index) - pad_frames)
        end = min(out.size, int(index) + pad_frames + 1)
        out[start:end] = True
    return out.astype(np.int8, copy=False)


def frames_to_segments(
    frames: Iterable[int],
    *,
    frame_hop_s: float,
    duration_s: float,
    scores: np.ndarray | None = None,
) -> list[SpeechSegment]:
    values = [1 if int(value) else 0 for value in frames]
    segments: list[SpeechSegment] = []
    start_index: int | None = None
    for index, value in enumerate(values + [0]):
        if value and start_index is None:
            start_index = index
        if not value and start_index is not None:
            start = max(0.0, min(float(start_index) * frame_hop_s, duration_s))
            end = max(0.0, min(float(index) * frame_hop_s, duration_s))
            score = None
            if scores is not None and index > start_index:
                score = float(np.max(scores[start_index:index]))
            if end > start:
                segments.append(SpeechSegment(start=start, end=end, score=score))
            start_index = None
    return segments


def merge_segments(
    segments: Iterable[SpeechSegment],
    *,
    duration_s: float,
    merge_gap_s: float,
    min_segment_s: float,
) -> list[SpeechSegment]:
    ordered = sorted(
        (
            SpeechSegment(
                start=max(0.0, min(float(segment.start), duration_s)),
                end=max(0.0, min(float(segment.end), duration_s)),
                score=segment.score,
            )
            for segment in segments
        ),
        key=lambda item: (item.start, item.end),
    )
    merged: list[SpeechSegment] = []
    for segment in ordered:
        if segment.end - segment.start < min_segment_s:
            continue
        if not merged or segment.start - merged[-1].end > merge_gap_s:
            merged.append(segment)
            continue
        merged[-1].end = max(merged[-1].end, segment.end)
        if merged[-1].score is None:
            merged[-1].score = segment.score
        elif segment.score is not None:
            merged[-1].score = max(float(merged[-1].score), float(segment.score))
    return [segment for segment in merged if segment.end - segment.start >= min_segment_s]


@dataclass(frozen=True)
class FusionVadJaConfig:
    checkpoint: Path
    threshold: float = 0.200
    pad_s: float = 0.2
    frame_hop_s: float = 0.02
    ptm: str = "qwen3-asr-0.6b"
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    window_s: float = 30.0
    overlap_s: float = 1.0
    min_segment_s: float = 0.05
    merge_gap_s: float = 0.0
    max_group_s: float = 6.0
    chunk_threshold_s: float = 1.0
    cut_threshold: float = 0.500
    apply_cut_to_speech: bool = True
    imitation_checkpoint: Path = Path(DEFAULT_IMITATION_CHECKPOINT)
    no_download: bool = False

    @classmethod
    def from_env(cls) -> "FusionVadJaConfig":
        return cls(
            checkpoint=_resolve_project_path(
                os.getenv("FUSIONVAD_JA_CHECKPOINT", DEFAULT_CHECKPOINT).strip()
                or DEFAULT_CHECKPOINT
            ),
            threshold=_env_float("FUSIONVAD_JA_THRESHOLD", "0.200"),
            pad_s=_env_float("FUSIONVAD_JA_PAD_S", "0.2"),
            frame_hop_s=_env_float("FUSIONVAD_JA_FRAME_HOP_S", "0.02"),
            ptm=os.getenv("FUSIONVAD_JA_PTM", "qwen3-asr-0.6b").strip()
            or "qwen3-asr-0.6b",
            model_path=os.getenv("FUSIONVAD_JA_MODEL_PATH", DEFAULT_MODEL_PATH).strip(),
            device=os.getenv("FUSIONVAD_JA_DEVICE", "auto").strip() or "auto",
            dtype=os.getenv("FUSIONVAD_JA_DTYPE", "bfloat16").strip() or "bfloat16",
            attention=os.getenv("FUSIONVAD_JA_ATTENTION", "sdpa").strip() or "sdpa",
            window_s=_env_float("FUSIONVAD_JA_WINDOW_S", "30.0"),
            overlap_s=_env_float("FUSIONVAD_JA_OVERLAP_S", "1.0"),
            min_segment_s=_env_float("FUSIONVAD_JA_MIN_SEGMENT_S", "0.05"),
            merge_gap_s=_env_float("FUSIONVAD_JA_MERGE_GAP_S", "0.0"),
            max_group_s=_env_float("FUSIONVAD_JA_MAX_GROUP_S", "6.0"),
            chunk_threshold_s=_env_float("FUSIONVAD_JA_CHUNK_THRESHOLD_S", "1.0"),
            cut_threshold=_env_float("FUSIONVAD_JA_CUT_THRESHOLD", "0.500"),
            apply_cut_to_speech=_env_bool("FUSIONVAD_JA_APPLY_CUT_TO_SPEECH", "1"),
            imitation_checkpoint=_resolve_project_path(
                os.getenv("FUSIONVAD_JA_IMITATION_CHECKPOINT", DEFAULT_IMITATION_CHECKPOINT).strip()
                or DEFAULT_IMITATION_CHECKPOINT
            ),
            no_download=_env_bool("FUSIONVAD_JA_NO_DOWNLOAD", "0"),
        )


class FusionVadJaBackend:
    name = "fusionvad_ja_v1_19b_splitcut_endpoint_refiner"

    def __init__(self, config: FusionVadJaConfig | None = None) -> None:
        self.config = config or FusionVadJaConfig.from_env()

    def signature(self) -> dict:
        cfg = self.config
        return {
            "backend": self.name,
            "checkpoint": str(cfg.checkpoint),
            "threshold": float(cfg.threshold),
            "pad_s": float(cfg.pad_s),
            "frame_hop_s": float(cfg.frame_hop_s),
            "ptm": cfg.ptm,
            "model_path": cfg.model_path,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "attention": cfg.attention,
            "window_s": float(cfg.window_s),
            "overlap_s": float(cfg.overlap_s),
            "min_segment_s": float(cfg.min_segment_s),
            "merge_gap_s": float(cfg.merge_gap_s),
            "max_group_s": float(cfg.max_group_s),
            "chunk_threshold_s": float(cfg.chunk_threshold_s),
            "cut_threshold": float(cfg.cut_threshold),
            "apply_cut_to_speech": bool(cfg.apply_cut_to_speech),
            "imitation_checkpoint": str(cfg.imitation_checkpoint),
            "operating_point": DEFAULT_OPERATING_POINT,
            "allow_empty": True,
        }

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del target_sr
        started = time.perf_counter()
        cfg = self.config
        if not cfg.checkpoint.exists():
            raise FileNotFoundError(f"FusionVAD-JA checkpoint not found: {cfg.checkpoint}")
        need_drop_gap_scores = _env_bool("FUSIONVAD_JA_EXPORT_DROP_GAP_SCORES", "0") or _env_bool(
            "ASR_PRE_ASR_DROP_GAP_SPLIT_ENABLED", "0"
        )
        if need_drop_gap_scores and not cfg.imitation_checkpoint.exists():
            raise FileNotFoundError(
                f"FusionVAD-JA imitation checkpoint not found: {cfg.imitation_checkpoint}"
            )
        if cfg.window_s <= 0.0:
            raise ValueError("FUSIONVAD_JA_WINDOW_S must be positive")
        if cfg.overlap_s < 0.0:
            raise ValueError("FUSIONVAD_JA_OVERLAP_S must be non-negative")
        if cfg.overlap_s >= cfg.window_s:
            raise ValueError("FUSIONVAD_JA_OVERLAP_S must be smaller than FUSIONVAD_JA_WINDOW_S")

        import torch

        device = _model_device(cfg.device)
        threshold = float(cfg.threshold if threshold_override is None else threshold_override)
        feature_config = FeatureConfig(
            ptm=cfg.ptm,
            frame_hop_s=cfg.frame_hop_s,
            n_mfcc=40,
            n_fft=400,
            device=str(device),
            dtype=cfg.dtype,
            model_path=cfg.model_path,
            download=not cfg.no_download,
            attention=cfg.attention,
            language="Japanese",
        )
        imitation_model = None
        imitation_checkpoint = None
        imitation_model_config: dict | None = None
        imitation_model_type: str | None = None
        addition_model, checkpoint, model_config, model_type = _load_addition_model(
            cfg.checkpoint,
            device=device,
        )
        if need_drop_gap_scores:
            (
                imitation_model,
                imitation_checkpoint,
                imitation_model_config,
                imitation_model_type,
            ) = _load_imitation_model(
                cfg.imitation_checkpoint,
                device=device,
            )
        ptm_extractor = build_ptm_feature_extractor(feature_config)
        ptm_param_device, ptm_param_dtype = _first_parameter_device_dtype(
            getattr(ptm_extractor, "model", None)
        )
        model_param_device, model_param_dtype = _first_parameter_device_dtype(addition_model)
        runtime_device = {
            "requested_device": cfg.device,
            "actual_device": str(device),
            "dtype": cfg.dtype,
            "ptm_param_device": ptm_param_device,
            "ptm_param_dtype": ptm_param_dtype,
            "model_param_device": model_param_device,
            "model_param_dtype": model_param_dtype,
        }
        print(
            "[vad] fusionvad_ja device "
            f"requested_device={runtime_device['requested_device']} "
            f"actual_device={runtime_device['actual_device']} "
            f"dtype={runtime_device['dtype']} "
            f"ptm_param_device={runtime_device['ptm_param_device']} "
            f"ptm_param_dtype={runtime_device['ptm_param_dtype']} "
            f"model_param_device={runtime_device['model_param_device']} "
            f"model_param_dtype={runtime_device['model_param_dtype']}",
            flush=True,
        )
        try:
            audio, sample_rate = load_audio_16k_mono(audio_path)
            duration_s = float(len(audio) / sample_rate) if sample_rate else 0.0
            total_frames = frame_count(duration_s, cfg.frame_hop_s)
            probability_sum = np.zeros(total_frames, dtype=np.float64)
            probability_count = np.zeros(total_frames, dtype=np.float32)
            cut_probability_sum = np.zeros(total_frames, dtype=np.float64)
            cut_probability_count = np.zeros(total_frames, dtype=np.float32)
            drop_gap_probability_sum = np.zeros(total_frames, dtype=np.float64)
            drop_gap_probability_count = np.zeros(total_frames, dtype=np.float32)
            window_samples = max(1, int(round(cfg.window_s * sample_rate)))
            stride_samples = max(1, int(round((cfg.window_s - cfg.overlap_s) * sample_rate)))
            starts = list(range(0, max(1, len(audio)), stride_samples))

            with torch.inference_mode():
                for window_index, start_sample in enumerate(starts):
                    end_sample = min(len(audio), start_sample + window_samples)
                    if start_sample >= end_sample:
                        continue
                    chunk = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
                    mfcc = extract_mfcc(chunk, sample_rate=sample_rate, config=feature_config)
                    ptm = ptm_extractor.extract(chunk, sample_rate=sample_rate)
                    ptm, mfcc = align_feature_frames(
                        ptm,
                        mfcc,
                        resize_ptm=is_low_frame_rate_ptm(cfg.ptm),
                    )
                    frame_total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
                    if frame_total <= 0:
                        continue
                    whisper_tensor = torch.from_numpy(
                        np.ascontiguousarray(ptm[:frame_total], dtype=np.float32)
                    ).to(device).unsqueeze(0)
                    mfcc_tensor = torch.from_numpy(
                        np.ascontiguousarray(mfcc[:frame_total], dtype=np.float32)
                    ).to(device).unsqueeze(0)
                    logits = addition_model(whisper_tensor, mfcc_tensor)
                    output_probs = _sigmoid_outputs(logits)
                    probs = output_probs["speech"]
                    cut_probs = output_probs.get("cut")
                    if cut_probs is None:
                        cut_drop_probs = output_probs.get("cut_drop")
                        cut_point_probs = output_probs.get("cut_point")
                        if cut_drop_probs is not None and cut_point_probs is not None:
                            cut_probs = np.maximum(cut_drop_probs, cut_point_probs)
                        elif cut_drop_probs is not None:
                            cut_probs = cut_drop_probs
                        elif cut_point_probs is not None:
                            cut_probs = cut_point_probs
                    drop_gap_probs = None
                    if imitation_model is not None:
                        imitation_logits = imitation_model(whisper_tensor, mfcc_tensor)
                        imitation_probs = _sigmoid_outputs(imitation_logits)
                        drop_gap_probs = imitation_probs.get("drop_gap")
                    window_start_s = start_sample / sample_rate
                    global_start = max(0, int(round(window_start_s / cfg.frame_hop_s)))
                    global_end = min(total_frames, global_start + probs.size)
                    local_end = max(0, global_end - global_start)
                    if local_end <= 0:
                        continue
                    probability_sum[global_start:global_end] += probs[:local_end]
                    probability_count[global_start:global_end] += 1.0
                    if cut_probs is not None:
                        cut_local_end = min(local_end, cut_probs.size)
                        cut_probability_sum[global_start : global_start + cut_local_end] += cut_probs[
                            :cut_local_end
                        ]
                        cut_probability_count[global_start : global_start + cut_local_end] += 1.0
                    if drop_gap_probs is not None:
                        drop_gap_local_end = min(local_end, drop_gap_probs.size)
                        drop_gap_probability_sum[
                            global_start : global_start + drop_gap_local_end
                        ] += drop_gap_probs[:drop_gap_local_end]
                        drop_gap_probability_count[
                            global_start : global_start + drop_gap_local_end
                        ] += 1.0
                    print(
                        "[vad] fusionvad_ja window "
                        f"{window_index + 1}/{len(starts)} start={window_start_s:.1f}s "
                        f"frames={local_end}",
                        flush=True,
                    )

            probabilities = np.divide(
                probability_sum,
                np.maximum(probability_count, 1.0),
                out=np.zeros_like(probability_sum, dtype=np.float64),
                where=probability_count > 0,
            ).astype(np.float32)
            cut_probabilities: np.ndarray | None = None
            if np.any(cut_probability_count > 0):
                cut_probabilities = np.divide(
                    cut_probability_sum,
                    np.maximum(cut_probability_count, 1.0),
                    out=np.zeros_like(cut_probability_sum, dtype=np.float64),
                    where=cut_probability_count > 0,
                ).astype(np.float32)
            drop_gap_probabilities: np.ndarray | None = None
            if np.any(drop_gap_probability_count > 0):
                drop_gap_probabilities = np.divide(
                    drop_gap_probability_sum,
                    np.maximum(drop_gap_probability_count, 1.0),
                    out=np.zeros_like(drop_gap_probability_sum, dtype=np.float64),
                    where=drop_gap_probability_count > 0,
                ).astype(np.float32)
            effective_probabilities = _apply_cut_gate(
                probabilities,
                cut_probabilities,
                cut_threshold=cfg.cut_threshold,
                apply_cut=cfg.apply_cut_to_speech,
            )
            raw_frames = effective_probabilities >= threshold
            padded = _padded_frames(
                raw_frames,
                pad_frames=max(0, int(round(cfg.pad_s / cfg.frame_hop_s))),
            )
            segments = frames_to_segments(
                padded,
                frame_hop_s=cfg.frame_hop_s,
                duration_s=duration_s,
                scores=probabilities,
            )
            segments = merge_segments(
                segments,
                duration_s=duration_s,
                merge_gap_s=cfg.merge_gap_s,
                min_segment_s=cfg.min_segment_s,
            )
            groups = group_segments(
                segments,
                max_group_duration_s=cfg.max_group_s,
                chunk_threshold_s=cfg.chunk_threshold_s,
            )
            params = self.signature()
            params.update(
                {
                    "audio_stats": {
                        "duration_s": duration_s,
                        "frames": int(total_frames),
                        "windows": len(starts),
                        "probability_mean": float(probabilities.mean()) if probabilities.size else 0.0,
                        "probability_max": float(probabilities.max()) if probabilities.size else 0.0,
                        "effective_probability_mean": (
                            float(effective_probabilities.mean()) if effective_probabilities.size else 0.0
                        ),
                        "effective_probability_max": (
                            float(effective_probabilities.max()) if effective_probabilities.size else 0.0
                        ),
                        "cut_probability_mean": (
                            float(cut_probabilities.mean()) if cut_probabilities is not None and cut_probabilities.size else 0.0
                        ),
                        "cut_probability_max": (
                            float(cut_probabilities.max()) if cut_probabilities is not None and cut_probabilities.size else 0.0
                        ),
                        "drop_gap_probability_mean": (
                            float(drop_gap_probabilities.mean())
                            if drop_gap_probabilities is not None and drop_gap_probabilities.size
                            else 0.0
                        ),
                        "drop_gap_probability_max": (
                            float(drop_gap_probabilities.max())
                            if drop_gap_probabilities is not None and drop_gap_probabilities.size
                            else 0.0
                        ),
                        "raw_speech_ratio": float(raw_frames.mean()) if raw_frames.size else 0.0,
                        "padded_speech_ratio": float(padded.mean()) if padded.size else 0.0,
                        "uncovered_frame_ratio": float((probability_count <= 0).mean())
                        if probability_count.size
                        else 0.0,
                    },
                    "checkpoint_config": model_config,
                    "checkpoint_model_type": model_type,
                    "checkpoint_ptm": checkpoint.get("ptm"),
                    "imitation_checkpoint_config": imitation_model_config,
                    "imitation_checkpoint_model_type": imitation_model_type,
                    "imitation_checkpoint_ptm": (
                        imitation_checkpoint.get("ptm") if imitation_checkpoint is not None else None
                    ),
                    "runtime_device": runtime_device,
                }
            )
            if _env_bool("FUSIONVAD_JA_EXPORT_FRAME_SCORES", "0") or _env_bool(
                "ASR_PRE_ASR_VALLEY_SPLIT_ENABLED", "0"
            ) or _env_bool(
                "ASR_PRE_ASR_CUT_SPLIT_ENABLED", "0"
            ) or _env_bool(
                "ASR_PRE_ASR_DROP_GAP_SPLIT_ENABLED", "0"
            ):
                params["frame_scores"] = [float(value) for value in probabilities]
                if cut_probabilities is not None:
                    params["cut_frame_scores"] = [float(value) for value in cut_probabilities]
                if drop_gap_probabilities is not None:
                    params["drop_gap_frame_scores"] = [
                        float(value) for value in drop_gap_probabilities
                    ]
            return SegmentationResult(
                segments=segments,
                groups=groups,
                method=self.name,
                audio_duration_sec=duration_s,
                parameters=params,
                processing_time_sec=time.perf_counter() - started,
            )
        finally:
            close = getattr(ptm_extractor, "close", None)
            if callable(close):
                close()
            del addition_model
            if imitation_model is not None:
                del imitation_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
