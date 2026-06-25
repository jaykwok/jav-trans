from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


FRAME_SEQUENCE_FEATURE_SCHEMA = "edge_sequence_features_v1"
FRAME_SEQUENCE_FRAMES_SCHEMA = "speech_boundary_ja_sequence_feature_frames_v1"
CHUNK_POOLED_PTM_SCHEMA = "pre_asr_chunk_pooled_ptm_v1"
DEFAULT_CHUNK_POOLED_PTM_BINS = 4


@dataclass(frozen=True)
class FrameSequenceFeatureConfig:
    left_context_s: float = 0.60
    right_context_s: float = 0.60
    max_ptm_dims: int = 64
    include_mfcc: bool = True

    def signature(self) -> dict:
        return {
            "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
            "left_context_s": self.left_context_s,
            "right_context_s": self.right_context_s,
            "max_ptm_dims": self.max_ptm_dims,
            "include_mfcc": self.include_mfcc,
        }


@dataclass(frozen=True)
class FrameSequenceFeatureProvider:
    duration_s: float
    frame_hop_s: float
    ptm: Sequence[Sequence[float]]
    mfcc: Sequence[Sequence[float]]
    config: FrameSequenceFeatureConfig = FrameSequenceFeatureConfig()
    _ptm_array: np.ndarray = field(init=False, repr=False)
    _mfcc_array: np.ndarray = field(init=False, repr=False)
    _frame_count: int = field(init=False, repr=False)
    _ptm_used_dim: int = field(init=False, repr=False)
    _mfcc_dim: int = field(init=False, repr=False)
    _ptm_used: np.ndarray = field(init=False, repr=False)
    _mfcc_used: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ptm_array = _frame_array(self.ptm, name="ptm")
        mfcc_array = _frame_array(self.mfcc, name="mfcc")
        frame_count = min(int(ptm_array.shape[0]), int(mfcc_array.shape[0]))
        if frame_count <= 0:
            raise ValueError("edge sequence features require at least one frame")
        ptm_used_dim = min(int(ptm_array.shape[1]), int(self.config.max_ptm_dims))
        object.__setattr__(self, "_ptm_array", ptm_array)
        object.__setattr__(self, "_mfcc_array", mfcc_array)
        object.__setattr__(self, "_frame_count", frame_count)
        object.__setattr__(self, "_ptm_used_dim", ptm_used_dim)
        object.__setattr__(self, "_mfcc_dim", int(mfcc_array.shape[1]))
        object.__setattr__(self, "_ptm_used", ptm_array[:frame_count, :ptm_used_dim])
        object.__setattr__(self, "_mfcc_used", mfcc_array[:frame_count])

    def frame_dims(self) -> tuple[int, int, int]:
        return self._frame_count, self._ptm_used_dim, self._mfcc_dim

    def feature_names(self) -> list[str]:
        _, ptm_dim, mfcc_dim = self.frame_dims()
        return frame_sequence_feature_names(
            config=self.config,
            ptm_dim=ptm_dim,
            mfcc_dim=mfcc_dim,
        )

    def feature_dim(self) -> int:
        _, ptm_dim, mfcc_dim = self.frame_dims()
        return get_feature_dim(
            config=self.config,
            ptm_dim=ptm_dim,
            mfcc_dim=mfcc_dim,
        )

    def feature_schema_hash(self) -> str:
        return feature_extraction_hash(
            config=self.config,
            feature_names=self.feature_names(),
        )

    def signature(self) -> dict:
        names = self.feature_names()
        frame_count, ptm_dim, mfcc_dim = self.frame_dims()
        return {
            "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
            "feature_schema_hash": feature_extraction_hash(
                config=self.config,
                feature_names=names,
            ),
            "feature_config": self.config.signature(),
            "feature_dim": len(names),
            "ptm_used_dim": ptm_dim,
            "mfcc_dim": mfcc_dim,
            "frame_count": frame_count,
            "frame_hop_s": self.frame_hop_s,
        }

    def validate_for_checkpoint(self, feature_names: Sequence[str], feature_schema_hash: str) -> None:
        runtime_names = tuple(self.feature_names())
        expected_names = tuple(str(name) for name in feature_names)
        if runtime_names != expected_names:
            raise ValueError("runtime edge sequence feature_names do not match checkpoint")
        runtime_hash = self.feature_schema_hash()
        if runtime_hash != str(feature_schema_hash):
            raise ValueError(
                "runtime edge sequence feature schema hash does not match checkpoint: "
                f"{runtime_hash} != {feature_schema_hash}"
            )

    def features_for_boundary(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]:
        return _boundary_window_sequence_features_from_arrays(
            left_start_s=left_start_s,
            left_end_s=left_end_s,
            right_start_s=right_start_s,
            right_end_s=right_end_s,
            duration_s=self.duration_s,
            frame_hop_s=self.frame_hop_s,
            ptm_used=self._ptm_used,
            mfcc_used=self._mfcc_used,
            config=self.config,
        )

    def chunk_pooled_ptm_feature_names(
        self,
        *,
        bins: int = DEFAULT_CHUNK_POOLED_PTM_BINS,
    ) -> list[str]:
        return chunk_pooled_ptm_feature_names(ptm_dim=self._ptm_used_dim, bins=bins)

    def chunk_pooled_ptm_signature(
        self,
        *,
        bins: int = DEFAULT_CHUNK_POOLED_PTM_BINS,
    ) -> dict:
        names = self.chunk_pooled_ptm_feature_names(bins=bins)
        return {
            "schema": CHUNK_POOLED_PTM_SCHEMA,
            "bins": int(bins),
            "ptm_used_dim": int(self._ptm_used_dim),
            "feature_dim": len(names),
            "feature_names_hash": hashlib.sha1(
                json.dumps(names, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "frame_hop_s": float(self.frame_hop_s),
        }

    def chunk_pooled_ptm_features(
        self,
        *,
        start_s: float,
        end_s: float,
        bins: int = DEFAULT_CHUNK_POOLED_PTM_BINS,
    ) -> list[float]:
        return _chunk_pooled_ptm_features_from_array(
            self._ptm_used,
            frame_hop_s=self.frame_hop_s,
            start_s=start_s,
            end_s=end_s,
            bins=bins,
        )


def get_default_config() -> FrameSequenceFeatureConfig:
    return FrameSequenceFeatureConfig()


def get_feature_dim(
    *,
    config: FrameSequenceFeatureConfig,
    ptm_dim: int,
    mfcc_dim: int,
) -> int:
    return len(frame_sequence_feature_names(config=config, ptm_dim=ptm_dim, mfcc_dim=mfcc_dim))


def frame_sequence_feature_names(
    *,
    config: FrameSequenceFeatureConfig,
    ptm_dim: int,
    mfcc_dim: int,
) -> list[str]:
    if ptm_dim <= 0:
        raise ValueError("ptm_dim must be positive")
    if mfcc_dim < 0:
        raise ValueError("mfcc_dim must be non-negative")
    names = [
        "gap_s",
        "left_duration_s",
        "right_duration_s",
    ]
    for region in ("left", "gap", "right"):
        names.extend(f"{region}_ptm_mean_{index:03d}" for index in range(ptm_dim))
        names.extend(f"{region}_ptm_std_{index:03d}" for index in range(ptm_dim))
    if config.include_mfcc:
        for region in ("left", "gap", "right"):
            names.extend(f"{region}_mfcc_mean_{index:03d}" for index in range(mfcc_dim))
            names.extend(f"{region}_mfcc_std_{index:03d}" for index in range(mfcc_dim))
    return names


def chunk_pooled_ptm_feature_names(
    *,
    ptm_dim: int,
    bins: int = DEFAULT_CHUNK_POOLED_PTM_BINS,
) -> list[str]:
    if ptm_dim <= 0:
        raise ValueError("ptm_dim must be positive")
    if bins <= 0:
        raise ValueError("bins must be positive")
    names: list[str] = []
    names.extend(f"chunk_ptm_mean_{index:03d}" for index in range(ptm_dim))
    names.extend(f"chunk_ptm_std_{index:03d}" for index in range(ptm_dim))
    for bin_index in range(int(bins)):
        names.extend(
            f"chunk_ptm_bin{bin_index:02d}_mean_{index:03d}"
            for index in range(ptm_dim)
        )
    return names


def feature_extraction_signature(
    *,
    config: FrameSequenceFeatureConfig,
    feature_names: Sequence[str],
) -> dict:
    names = [str(name) for name in feature_names]
    if not names:
        raise ValueError("feature_names must not be empty")
    return {
        "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
        "feature_config": config.signature(),
        "feature_names": names,
        "feature_dim": len(names),
    }


def feature_extraction_hash(
    *,
    config: FrameSequenceFeatureConfig,
    feature_names: Sequence[str],
) -> str:
    payload = json.dumps(
        feature_extraction_signature(config=config, feature_names=feature_names),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def validate_sequence_features(
    features: Sequence[Sequence[float]],
    *,
    feature_names: Sequence[str],
    expected_feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("feature_names must not be empty")
    if expected_feature_names is not None:
        expected = tuple(str(name) for name in expected_feature_names)
        if names != expected:
            raise ValueError("feature_names do not match expected_feature_names")
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("sequence_features must have shape [time, dim]")
    if array.shape[0] <= 0:
        raise ValueError("sequence_features must contain at least one item")
    if array.shape[1] != len(names):
        raise ValueError("sequence_features dim does not match feature_names")
    if not np.isfinite(array).all():
        raise ValueError("sequence_features must not contain NaN or inf")
    return array


def boundary_window_sequence_features(
    *,
    left_start_s: float,
    left_end_s: float,
    right_start_s: float,
    right_end_s: float,
    duration_s: float,
    frame_hop_s: float,
    ptm: Sequence[Sequence[float]],
    mfcc: Sequence[Sequence[float]],
    config: FrameSequenceFeatureConfig,
) -> list[float]:
    if frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    ptm_array = _frame_array(ptm, name="ptm")
    mfcc_array = _frame_array(mfcc, name="mfcc")
    frame_count = min(int(ptm_array.shape[0]), int(mfcc_array.shape[0]))
    if frame_count <= 0:
        raise ValueError("edge sequence features require at least one frame")
    ptm_used_dim = min(int(ptm_array.shape[1]), int(config.max_ptm_dims))
    ptm_used = ptm_array[:frame_count, :ptm_used_dim]
    mfcc_used = mfcc_array[:frame_count]
    return _boundary_window_sequence_features_from_arrays(
        left_start_s=left_start_s,
        left_end_s=left_end_s,
        right_start_s=right_start_s,
        right_end_s=right_end_s,
        duration_s=duration_s,
        frame_hop_s=frame_hop_s,
        ptm_used=ptm_used,
        mfcc_used=mfcc_used,
        config=config,
    )


def _boundary_window_sequence_features_from_arrays(
    *,
    left_start_s: float,
    left_end_s: float,
    right_start_s: float,
    right_end_s: float,
    duration_s: float,
    frame_hop_s: float,
    ptm_used: np.ndarray,
    mfcc_used: np.ndarray,
    config: FrameSequenceFeatureConfig,
) -> list[float]:
    if frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    gap_s = right_start_s - left_end_s
    ranges = {
        "left": (max(0.0, left_end_s - config.left_context_s), left_end_s),
        "gap": (left_end_s, right_start_s),
        "right": (right_start_s, min(duration_s, right_start_s + config.right_context_s)),
    }
    values = [
        float(gap_s),
        float(max(0.0, left_end_s - left_start_s)),
        float(max(0.0, right_end_s - right_start_s)),
    ]
    for name in ("left", "gap", "right"):
        start_s, end_s = ranges[name]
        values.extend(_stats_for_range(ptm_used, frame_hop_s=frame_hop_s, start_s=start_s, end_s=end_s))
    if config.include_mfcc:
        for name in ("left", "gap", "right"):
            start_s, end_s = ranges[name]
            values.extend(_stats_for_range(mfcc_used, frame_hop_s=frame_hop_s, start_s=start_s, end_s=end_s))
    return values


def _frame_array(values: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [frames, dim]")
    if array.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one frame")
    if array.shape[1] <= 0:
        raise ValueError(f"{name} feature dim must be positive")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or inf")
    return array


def _frame_bounds_for_range(
    frame_count: int,
    *,
    frame_hop_s: float,
    start_s: float,
    end_s: float,
) -> tuple[int, int]:
    lower = max(0, int(round(max(0.0, start_s) / frame_hop_s)))
    upper = min(int(frame_count), int(round(max(start_s, end_s) / frame_hop_s)))
    return lower, max(lower, upper)


def _stats_for_range(
    array: np.ndarray,
    *,
    frame_hop_s: float,
    start_s: float,
    end_s: float,
) -> list[float]:
    lower, upper = _frame_bounds_for_range(
        int(array.shape[0]),
        frame_hop_s=frame_hop_s,
        start_s=start_s,
        end_s=end_s,
    )
    if upper <= lower:
        return [0.0] * (int(array.shape[1]) * 2)
    window = np.asarray(array[lower:upper], dtype=np.float32)
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    return [float(value) for value in np.concatenate([mean, std], axis=0)]


def _chunk_pooled_ptm_features_from_array(
    array: np.ndarray,
    *,
    frame_hop_s: float,
    start_s: float,
    end_s: float,
    bins: int,
) -> list[float]:
    if frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    if bins <= 0:
        raise ValueError("bins must be positive")
    dim = int(array.shape[1])
    lower, upper = _frame_bounds_for_range(
        int(array.shape[0]),
        frame_hop_s=frame_hop_s,
        start_s=start_s,
        end_s=end_s,
    )
    if upper <= lower:
        return [0.0] * (dim * (2 + int(bins)))
    window = np.asarray(array[lower:upper], dtype=np.float32)
    values = [float(value) for value in window.mean(axis=0)]
    values.extend(float(value) for value in window.std(axis=0))
    for part in np.array_split(window, int(bins), axis=0):
        if part.shape[0] <= 0:
            values.extend([0.0] * dim)
        else:
            values.extend(float(value) for value in part.mean(axis=0))
    return values
