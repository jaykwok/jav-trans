from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


FRAME_SEQUENCE_FEATURE_SCHEMA = "edge_sequence_features_v2"
FRAME_SEQUENCE_FRAMES_SCHEMA = "speech_boundary_ja_sequence_feature_frames_v1"
CHUNK_POOLED_PTM_SCHEMA = "pre_asr_chunk_pooled_ptm_v1"
CHUNK_PROJECTED_PTM_SCHEMA = "pre_asr_chunk_projected_ptm_v2"
DEFAULT_CHUNK_POOLED_PTM_BINS = 4
SPLIT_CANDIDATE_SCALAR_NAMES = (
    "candidate_score",
    "candidate_prominence",
    "candidate_speech_valley",
    "candidate_strength",
    "core_duration_s",
    "left_duration_s",
    "right_duration_s",
    "candidate_position_ratio",
    "left_speech_mean",
    "right_speech_mean",
    "gap_speech_mean",
    "left_speech_active_ratio",
    "right_speech_active_ratio",
)


PTM_PROJECTION_SCHEMA = "speech_boundary_ja_ptm_projection_v1"


def parse_extra_context_scales(raw: str) -> list[dict]:
    """Parse ``"3.2:4,6.4:4"`` into split-candidate extra-scale dicts; "" disables."""

    scales: list[dict] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        seconds_text, _, bins_text = part.partition(":")
        seconds = float(seconds_text)
        bins = int(bins_text or 4)
        if seconds <= 0 or bins <= 0:
            raise ValueError(f"invalid context scale: {part!r}")
        scales.append(
            {
                "left_context_s": seconds,
                "right_context_s": seconds,
                "left_bins": bins,
                "right_bins": bins,
            }
        )
    return scales


def ptm_projection_digest(mean: np.ndarray, components: np.ndarray) -> str:
    """Stable identity for one projection basis, comparable across processes."""

    import hashlib

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mean, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(components, dtype=np.float32).tobytes())
    return digest.hexdigest()


def load_ptm_projection(path: str) -> dict | None:
    """Load a variance-preserving PTM projection npz (compute_ptm_projection.py)."""

    if not path:
        return None
    bundle = np.load(path)
    if str(bundle["schema"]) != PTM_PROJECTION_SCHEMA:
        raise ValueError(f"unknown ptm projection schema: {path}")
    mean = np.asarray(bundle["mean"], dtype=np.float32)
    components = np.asarray(bundle["components"], dtype=np.float32)
    return {
        "schema": str(bundle["schema"]),
        "source": str(path),
        "mean": mean,
        "components": components,
        "digest": ptm_projection_digest(mean, components),
    }


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
    ptm_projected: Sequence[Sequence[float]] | None = None
    ptm_projected_digest: str = ""
    semantic_ptm_projected: Sequence[Sequence[float]] | None = None
    semantic_scorer_sha256: str = ""
    _ptm_array: np.ndarray = field(init=False, repr=False)
    _mfcc_array: np.ndarray = field(init=False, repr=False)
    _ptm_projected_array: np.ndarray | None = field(init=False, repr=False)
    _semantic_ptm_projected_array: np.ndarray | None = field(init=False, repr=False)
    _frame_count: int = field(init=False, repr=False)
    _ptm_used_dim: int = field(init=False, repr=False)
    _mfcc_dim: int = field(init=False, repr=False)
    _ptm_used: np.ndarray = field(init=False, repr=False)
    _mfcc_used: np.ndarray = field(init=False, repr=False)
    _combined_cache: dict[int, np.ndarray] = field(init=False, repr=False)
    _projected_ptm_cache: dict[int, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ptm_array = _frame_array(self.ptm, name="ptm")
        mfcc_array = _frame_array(self.mfcc, name="mfcc")
        frame_count = min(int(ptm_array.shape[0]), int(mfcc_array.shape[0]))
        if frame_count <= 0:
            raise ValueError("edge sequence features require at least one frame")
        ptm_used_dim = min(int(ptm_array.shape[1]), int(self.config.max_ptm_dims))
        projected_array: np.ndarray | None = None
        if self.ptm_projected is not None:
            projected_array = _frame_array(self.ptm_projected, name="ptm_projected")
            if int(projected_array.shape[0]) < frame_count:
                raise ValueError(
                    "ptm_projected has fewer frames than ptm/mfcc: "
                    f"{projected_array.shape[0]} < {frame_count}"
                )
        semantic_projected_array: np.ndarray | None = None
        if self.semantic_ptm_projected is not None:
            semantic_projected_array = _frame_array(
                self.semantic_ptm_projected,
                name="semantic_ptm_projected",
            )
            if int(semantic_projected_array.shape[0]) < frame_count:
                raise ValueError(
                    "semantic_ptm_projected has fewer frames than ptm/mfcc: "
                    f"{semantic_projected_array.shape[0]} < {frame_count}"
                )
        object.__setattr__(self, "_ptm_array", ptm_array)
        object.__setattr__(self, "_mfcc_array", mfcc_array)
        object.__setattr__(self, "_ptm_projected_array", projected_array)
        object.__setattr__(
            self,
            "_semantic_ptm_projected_array",
            semantic_projected_array,
        )
        object.__setattr__(self, "_frame_count", frame_count)
        object.__setattr__(self, "_ptm_used_dim", ptm_used_dim)
        object.__setattr__(self, "_mfcc_dim", int(mfcc_array.shape[1]))
        object.__setattr__(self, "_ptm_used", ptm_array[:frame_count, :ptm_used_dim])
        object.__setattr__(self, "_mfcc_used", mfcc_array[:frame_count])
        object.__setattr__(self, "_combined_cache", {})
        object.__setattr__(self, "_projected_ptm_cache", {})

    @property
    def has_pre_projected_ptm(self) -> bool:
        return self._ptm_projected_array is not None

    def _combined_features(self, ptm_dim: int) -> np.ndarray:
        frame_dim = min(int(ptm_dim), self._ptm_used_dim)
        cached = self._combined_cache.get(frame_dim)
        if cached is None:
            cached = np.concatenate(
                (
                    self._ptm_used[:, :frame_dim],
                    self._mfcc_used,
                ),
                axis=1,
            )
            self._combined_cache[frame_dim] = cached
        return cached

    def _combined_projected_features(
        self,
        *,
        projection_mean: np.ndarray,
        projection_components: np.ndarray,
    ) -> np.ndarray:
        """PTM projected through a variance-preserving basis instead of a
        leading-dim slice. Uses the FULL cached PTM dim, so max_ptm_dims does
        not truncate the projection input. Cached per projection identity."""

        key = -1 - id(projection_components)
        cached = self._combined_cache.get(key)
        if cached is None:
            projected = self._projected_ptm_features(
                projection_mean=projection_mean,
                projection_components=projection_components,
            )
            cached = np.concatenate((projected, self._mfcc_used), axis=1)
            self._combined_cache[key] = cached
        return cached

    def _projected_ptm_features(
        self,
        *,
        projection_mean: np.ndarray,
        projection_components: np.ndarray,
    ) -> np.ndarray:
        components = np.asarray(projection_components, dtype=np.float32)
        if self._ptm_projected_array is not None:
            projected = self._ptm_projected_array[: self._frame_count]
            if int(projected.shape[1]) != int(components.shape[0]):
                raise ValueError(
                    "pre-projected ptm dim mismatch: payload has "
                    f"{projected.shape[1]}, projection outputs "
                    f"{components.shape[0]}"
                )
            return projected
        key = id(projection_components)
        cached = self._projected_ptm_cache.get(key)
        if cached is None:
            full = self._ptm_array[: self._frame_count]
            mean = np.asarray(projection_mean, dtype=np.float32).reshape(1, -1)
            if mean.shape[1] != full.shape[1] or components.shape[1] != full.shape[1]:
                raise ValueError(
                    "ptm projection dim mismatch: projection expects "
                    f"{components.shape[1]}, provider has {full.shape[1]}. "
                    "At runtime the backend must pre-project sequence "
                    "features (set SPEECH_BOUNDARY_JA_SEQUENCE_PTM_PROJECTION "
                    "to the training projection npz)."
                )
            cached = (full - mean) @ components.T
            self._projected_ptm_cache[key] = cached
        return cached

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

    def features_for_split_candidate(
        self,
        *,
        core_start_s: float,
        core_end_s: float,
        candidate: dict,
        speech_probabilities: Sequence[float],
        left_context_s: float,
        right_context_s: float,
        gap_context_s: float,
        left_bins: int,
        gap_bins: int,
        right_bins: int,
        ptm_dim: int,
        extra_context_scales: Sequence[dict] = (),
        ptm_projection_mean: np.ndarray | None = None,
        ptm_projection_components: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_s = float(candidate["time_s"])
        if ptm_projection_components is not None:
            if ptm_projection_mean is None:
                raise ValueError("ptm projection requires both mean and components")
            ptm_values = self._projected_ptm_features(
                projection_mean=ptm_projection_mean,
                projection_components=ptm_projection_components,
            )
        else:
            frame_dim = min(int(ptm_dim), self._ptm_used_dim)
            ptm_values = self._ptm_used[:, :frame_dim]
        ranges = [
            (
                max(core_start_s, candidate_s - left_context_s),
                max(core_start_s, candidate_s - gap_context_s),
                left_bins,
            ),
            (
                max(core_start_s, candidate_s - gap_context_s),
                min(core_end_s, candidate_s + gap_context_s),
                gap_bins,
            ),
            (
                min(core_end_s, candidate_s + gap_context_s),
                min(core_end_s, candidate_s + right_context_s),
                right_bins,
            ),
        ]
        # Coarser context scales append after the base bins so the base layout
        # (and v1-warm-started encoders) keep their positions.
        for scale in extra_context_scales:
            scale_left_s = float(scale["left_context_s"])
            scale_right_s = float(scale["right_context_s"])
            scale_left_bins = int(scale["left_bins"])
            scale_right_bins = int(scale["right_bins"])
            ranges.append(
                (
                    max(core_start_s, candidate_s - scale_left_s),
                    max(core_start_s, candidate_s - gap_context_s),
                    scale_left_bins,
                )
            )
            ranges.append(
                (
                    min(core_end_s, candidate_s + gap_context_s),
                    min(core_end_s, candidate_s + scale_right_s),
                    scale_right_bins,
                )
            )
        pooled = np.concatenate(
            [
                _pool_feature_bins(
                    ptm_values,
                    self._mfcc_used,
                    frame_hop_s=self.frame_hop_s,
                    start_s=start_s,
                    end_s=end_s,
                    bins=bins,
                )
                for start_s, end_s, bins in ranges
            ],
            axis=0,
        )
        speech = np.asarray(speech_probabilities, dtype=np.float32).reshape(-1)
        left_speech = _frame_window(
            speech,
            frame_hop_s=self.frame_hop_s,
            start_s=max(core_start_s, candidate_s - left_context_s),
            end_s=candidate_s,
        )
        right_speech = _frame_window(
            speech,
            frame_hop_s=self.frame_hop_s,
            start_s=candidate_s,
            end_s=min(core_end_s, candidate_s + right_context_s),
        )
        gap_speech = _frame_window(
            speech,
            frame_hop_s=self.frame_hop_s,
            start_s=max(core_start_s, candidate_s - gap_context_s),
            end_s=min(core_end_s, candidate_s + gap_context_s),
        )
        core_duration = max(0.0, core_end_s - core_start_s)
        scalar = np.asarray(
            [
                float(candidate.get("score") or 0.0),
                float(candidate.get("prominence") or 0.0),
                float(candidate.get("speech_valley") or 0.0),
                float(candidate.get("strength") or 0.0),
                core_duration,
                max(0.0, candidate_s - core_start_s),
                max(0.0, core_end_s - candidate_s),
                (candidate_s - core_start_s) / core_duration if core_duration > 0.0 else 0.0,
                _array_mean(left_speech),
                _array_mean(right_speech),
                _array_mean(gap_speech),
                _active_ratio(left_speech),
                _active_ratio(right_speech),
            ],
            dtype=np.float32,
        )
        return pooled, scalar

    def features_for_outer_island(
        self,
        *,
        start_s: float,
        end_s: float,
        speech_probabilities: Sequence[float],
        context_s: float,
        ptm_dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        start_frame = int(round(start_s / self.frame_hop_s))
        end_frame = int(round(end_s / self.frame_hop_s))
        context_frames = int(round(context_s / self.frame_hop_s))
        used_ptm_dim = min(int(ptm_dim), self._ptm_used_dim)
        ptm_values = self._ptm_used[:, :used_ptm_dim]
        pooled = np.concatenate(
            (
                _pool_feature_bins_by_index(
                    ptm_values, self._mfcc_used, start_frame - context_frames, start_frame, 4
                ),
                _pool_feature_bins_by_index(
                    ptm_values, self._mfcc_used, start_frame, start_frame + context_frames, 4
                ),
                _pool_feature_bins_by_index(
                    ptm_values, self._mfcc_used, end_frame - context_frames, end_frame, 4
                ),
                _pool_feature_bins_by_index(
                    ptm_values, self._mfcc_used, end_frame, end_frame + context_frames, 4
                ),
            ),
            axis=0,
        )
        speech = np.asarray(speech_probabilities, dtype=np.float32).reshape(-1)
        window = speech[max(0, start_frame) : min(speech.size, end_frame)]
        scalar = np.asarray(
            (
                max(0.0, end_s - start_s),
                (start_frame + end_frame) / max(1, 2 * speech.size),
                _array_mean(window),
                float(window.min()) if window.size else 0.0,
                float(window.max()) if window.size else 0.0,
                _active_ratio(window),
            ),
            dtype=np.float32,
        )
        return pooled, scalar

    def features_for_outer_island_v2(
        self,
        *,
        start_s: float,
        end_s: float,
        raw_ptm_dim: int,
    ) -> np.ndarray:
        """Return raw PTM2048 + MFCC frames for learned full-island edges."""

        start_frame = max(0, int(round(start_s / self.frame_hop_s)))
        end_frame = min(
            int(self._ptm_used.shape[0]),
            int(self._mfcc_used.shape[0]),
            int(round(end_s / self.frame_hop_s)),
        )
        if end_frame <= start_frame:
            return np.zeros(
                (
                    0,
                    int(raw_ptm_dim) + int(self._mfcc_used.shape[1]) + 1,
                ),
                dtype=np.float32,
            )
        if int(self._ptm_used.shape[1]) < int(raw_ptm_dim):
            raise ValueError("Outer Edge Refiner v2 requires full raw PTM frames")
        frame_total = end_frame - start_frame
        position = (
            np.arange(frame_total, dtype=np.float32) / max(1, frame_total - 1)
        ).reshape(-1, 1)
        return np.ascontiguousarray(
            np.concatenate(
                (
                    self._ptm_used[start_frame:end_frame, : int(raw_ptm_dim)],
                    self._mfcc_used[start_frame:end_frame],
                    position,
                ),
                axis=1,
            ),
            dtype=np.float32,
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

    def chunk_pooled_projected_ptm_signature(
        self,
        *,
        bins: int = DEFAULT_CHUNK_POOLED_PTM_BINS,
    ) -> dict:
        if self._ptm_projected_array is None or not self.ptm_projected_digest:
            raise ValueError("projected PTM pooling requires projected frames and digest")
        dim = int(self._ptm_projected_array.shape[1])
        names = chunk_pooled_ptm_feature_names(ptm_dim=dim, bins=bins)
        return {
            "schema": CHUNK_PROJECTED_PTM_SCHEMA,
            "bins": int(bins),
            "ptm_used_dim": dim,
            "feature_dim": len(names),
            "feature_names_hash": hashlib.sha1(
                json.dumps(names, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "frame_hop_s": float(self.frame_hop_s),
            "ptm_projection_digest": str(self.ptm_projected_digest),
        }

    def chunk_pooled_projected_ptm_features(
        self,
        *,
        start_s: float,
        end_s: float,
        bins: int = DEFAULT_CHUNK_POOLED_PTM_BINS,
    ) -> list[float]:
        if self._ptm_projected_array is None or not self.ptm_projected_digest:
            raise ValueError("projected PTM pooling requires projected frames and digest")
        return _chunk_pooled_ptm_features_from_array(
            self._ptm_projected_array[: self._frame_count],
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
        "total_duration_s",
        "left_center_s",
        "right_center_s",
        "gap_center_s",
        "left_duration_ratio",
        "right_duration_ratio",
        "gap_duration_ratio",
        "relative_gap_center",
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
    left_duration_s = max(0.0, left_end_s - left_start_s)
    right_duration_s = max(0.0, right_end_s - right_start_s)
    total_duration_s = max(0.0, float(duration_s))
    gap_duration_s = max(0.0, gap_s)
    left_center_s = (left_start_s + left_end_s) / 2.0
    right_center_s = (right_start_s + right_end_s) / 2.0
    gap_center_s = (left_end_s + right_start_s) / 2.0
    ranges = {
        "left": (max(0.0, left_end_s - config.left_context_s), left_end_s),
        "gap": (left_end_s, right_start_s),
        "right": (right_start_s, min(duration_s, right_start_s + config.right_context_s)),
    }
    values = [
        float(gap_s),
        float(left_duration_s),
        float(right_duration_s),
        float(total_duration_s),
        float(left_center_s),
        float(right_center_s),
        float(gap_center_s),
        _safe_ratio(left_duration_s, total_duration_s),
        _safe_ratio(right_duration_s, total_duration_s),
        _safe_ratio(gap_duration_s, total_duration_s),
        _safe_ratio(gap_center_s, total_duration_s),
    ]
    for name in ("left", "gap", "right"):
        start_s, end_s = ranges[name]
        values.extend(_stats_for_range(ptm_used, frame_hop_s=frame_hop_s, start_s=start_s, end_s=end_s))
    if config.include_mfcc:
        for name in ("left", "gap", "right"):
            start_s, end_s = ranges[name]
            values.extend(_stats_for_range(mfcc_used, frame_hop_s=frame_hop_s, start_s=start_s, end_s=end_s))
    return values


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


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


def _frame_window(
    values: np.ndarray,
    *,
    frame_hop_s: float,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    lower, upper = _frame_bounds_for_range(
        int(values.shape[0]),
        frame_hop_s=frame_hop_s,
        start_s=start_s,
        end_s=end_s,
    )
    return values[lower:upper]


def _pool_feature_bins(
    ptm_values: np.ndarray,
    mfcc_values: np.ndarray,
    *,
    frame_hop_s: float,
    start_s: float,
    end_s: float,
    bins: int,
) -> np.ndarray:
    return np.concatenate(
        (
            _pool_frame_bins(
                ptm_values,
                frame_hop_s=frame_hop_s,
                start_s=start_s,
                end_s=end_s,
                bins=bins,
            ),
            _pool_frame_bins(
                mfcc_values,
                frame_hop_s=frame_hop_s,
                start_s=start_s,
                end_s=end_s,
                bins=bins,
            ),
        ),
        axis=1,
    )


def _pool_feature_bins_by_index(
    ptm_values: np.ndarray,
    mfcc_values: np.ndarray,
    start: int,
    end: int,
    bins: int,
) -> np.ndarray:
    return np.concatenate(
        (
            _pool_frame_bins_by_index(ptm_values, start, end, bins),
            _pool_frame_bins_by_index(mfcc_values, start, end, bins),
        ),
        axis=1,
    )


def _pool_frame_bins(
    values: np.ndarray,
    *,
    frame_hop_s: float,
    start_s: float,
    end_s: float,
    bins: int,
) -> np.ndarray:
    window = _frame_window(
        values,
        frame_hop_s=frame_hop_s,
        start_s=start_s,
        end_s=end_s,
    )
    return _pool_window_bins(window, dim=int(values.shape[1]), bins=bins)


def _pool_frame_bins_by_index(
    values: np.ndarray,
    start: int,
    end: int,
    bins: int,
) -> np.ndarray:
    window = values[max(0, start) : min(values.shape[0], end)]
    return _pool_window_bins(window, dim=int(values.shape[1]), bins=bins)


def _pool_window_bins(window: np.ndarray, *, dim: int, bins: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for part in np.array_split(window, int(bins), axis=0):
        rows.append(
            part.mean(axis=0).astype(np.float32)
            if part.shape[0]
            else np.zeros(dim, dtype=np.float32)
        )
    return np.stack(rows)


def _array_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _active_ratio(values: np.ndarray) -> float:
    return float((values >= 0.5).mean()) if values.size else 0.0


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
