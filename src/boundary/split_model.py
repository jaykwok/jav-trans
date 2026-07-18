from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from asr.backends.qwen import validate_checkpoint_repo_id
from boundary.backbones import Mamba2TemporalEncoder
from boundary.contracts import (
    ACOUSTIC_BINARY_V12_CONTRACT,
    require_boundary_contract_id,
)


SEMANTIC_SPLIT_LABELS = ("cut", "continue", "unsure")
SEMANTIC_SPLIT_FEATURE_SCHEMA = "semantic_split_candidate_features_v1"

SEMANTIC_SPLIT_V4_SCHEMA = "semantic_split_model_v4"
SEMANTIC_SPLIT_V4_MODEL_ARCH = "acoustic_candidate_sequence_mamba_binary_v2"
SEMANTIC_SPLIT_V4_RUNTIME_ADAPTER = "acoustic_candidate_binary_event_runs_v2"
SEMANTIC_SPLIT_V4_ARTIFACT = {
    "name": "semantic_split_model",
    "display_name": "Acoustic Split Model",
    "version": "v4",
    "pipeline_stage": 3,
    "pipeline_role": "acoustic_binary_boundary_event_planner",
}
SEMANTIC_SPLIT_V4_DECISION = {"decision_mode": "binary_argmax_cut"}
SEMANTIC_SPLIT_TRAINING_LABELS = ("cut", "continue")
SEMANTIC_SPLIT_STRUCTURAL_ROLES = (
    "none",
    "speech_to_speech",
    "speech_to_noise",
    "noise_to_speech",
)


class IslandCandidateSequenceNetwork:
    """Binary acoustic cut/continue candidate-sequence network factory.

    Encodes every candidate with a per-candidate frame stack
    (``frame_proj`` -> bidirectional Mamba2 over bins -> mean pool + scalar arm)
    and then runs a second
    bidirectional Mamba2 over the ordered candidates of one speech island and
    emits per-candidate heads:

    - ``label``: cut / continue logits used directly by runtime argmax
    - ``role``: structural-role auxiliary logits (none / speech_to_speech /
      speech_to_noise / noise_to_speech)
    """

    def __new__(
        cls,
        *,
        frame_dim: int,
        scalar_dim: int,
        hidden_size: int = 128,
        candidate_layers: int = 2,
        island_layers: int = 2,
        state_size: int = 32,
        num_heads: int = 4,
        head_dim: int = 64,
        n_groups: int = 2,
        conv_kernel: int = 4,
        chunk_size: int = 8,
        bidirectional: bool = True,
        dropout: float = 0.1,
        num_classes: int = 2,
        structural_role_dim: int = 4,
        left_bins: int = 8,
        gap_bins: int = 4,
        right_bins: int = 8,
        extra_scale_bins: Sequence[Sequence[int]] = (),
        ptm_input_dim: int = 0,
        ptm_projected_dim: int = 0,
        ptm_projector_residual: bool = False,
    ):
        import torch
        from torch import nn

        if frame_dim <= 0 or scalar_dim <= 0:
            raise ValueError("frame_dim and scalar_dim must be positive")
        if bool(ptm_input_dim) != bool(ptm_projected_dim):
            raise ValueError(
                "ptm_input_dim and ptm_projected_dim must be set together"
            )
        if ptm_input_dim and ptm_projected_dim >= frame_dim:
            raise ValueError(
                "ptm_projected_dim must leave room for non-PTM frame features"
            )
        if ptm_input_dim and ptm_input_dim < ptm_projected_dim:
            raise ValueError("ptm_input_dim must be >= ptm_projected_dim")
        if num_classes != len(SEMANTIC_SPLIT_TRAINING_LABELS):
            raise ValueError("Acoustic Split network requires num_classes=2")
        if structural_role_dim != len(SEMANTIC_SPLIT_STRUCTURAL_ROLES):
            raise ValueError("island split network requires structural_role_dim=4")
        if left_bins <= 0 or gap_bins <= 0 or right_bins <= 0:
            raise ValueError("left/gap/right bins must be positive")
        scale_bins = [(int(pair[0]), int(pair[1])) for pair in extra_scale_bins]
        if any(left <= 0 or right <= 0 for left, right in scale_bins):
            raise ValueError("extra_scale_bins entries must be positive")

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.left_bins = int(left_bins)
                self.gap_bins = int(gap_bins)
                self.right_bins = int(right_bins)
                self.scale_bins = list(scale_bins)
                self.ptm_input_dim = int(ptm_input_dim)
                self.ptm_projected_dim = int(ptm_projected_dim)
                if self.ptm_input_dim:
                    self.ptm_projector = nn.Linear(
                        self.ptm_input_dim, self.ptm_projected_dim, bias=False
                    )
                    # Identity-slice init keeps step-0 projected inputs stable
                    # before training moves the projection weights.
                    with torch.no_grad():
                        self.ptm_projector.weight.copy_(
                            torch.eye(self.ptm_projected_dim, self.ptm_input_dim)
                        )
                self.ptm_projector_residual = bool(ptm_projector_residual)
                if self.ptm_projector_residual:
                    # Non-foldable residual on top of the linear projector. The
                    # final layer is zero-init, so at step 0 the residual
                    # contributes nothing and z_A == z_B exactly — a clean
                    # B-continue vs A-residual control isolates whether the
                    # non-linear path adds information beyond the linear
                    # subspace B already learned.
                    residual_hidden = 2 * self.ptm_projected_dim
                    self.ptm_residual = nn.Sequential(
                        nn.LayerNorm(self.ptm_input_dim),
                        nn.Linear(self.ptm_input_dim, residual_hidden),
                        nn.SiLU(),
                        nn.Linear(residual_hidden, self.ptm_projected_dim),
                    )
                    nn.init.zeros_(self.ptm_residual[-1].weight)
                    nn.init.zeros_(self.ptm_residual[-1].bias)
                self.frame_proj = nn.Linear(frame_dim, hidden_size)
                # Zero-initialized so checkpoint continuation starts unchanged;
                # learned later to mark left/gap/right bins
                # (ids 3+ mark the coarser multi-scale context regions).
                self.segment_embedding = nn.Parameter(
                    torch.zeros(3 + 2 * len(self.scale_bins), hidden_size)
                )
                self.encoder = Mamba2TemporalEncoder(
                    hidden_size=hidden_size,
                    num_layers=candidate_layers,
                    state_size=state_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    conv_kernel=conv_kernel,
                    chunk_size=chunk_size,
                    bidirectional=bidirectional,
                )
                self.scalar_arm = nn.Sequential(
                    nn.LayerNorm(scalar_dim),
                    nn.Linear(scalar_dim, hidden_size),
                    nn.GELU(),
                )
                # Structured readout keeps the left/gap/right contrast that a
                # global mean pool erases: [global mean, gap mean, left-right].
                readout_dim = self.encoder.output_dim * 3
                self.candidate_fuse = nn.Sequential(
                    nn.LayerNorm(readout_dim + hidden_size),
                    nn.Linear(readout_dim + hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.island_encoder = Mamba2TemporalEncoder(
                    hidden_size=hidden_size,
                    num_layers=island_layers,
                    state_size=state_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    conv_kernel=conv_kernel,
                    chunk_size=chunk_size,
                    bidirectional=bidirectional,
                )
                joint_dim = hidden_size + self.island_encoder.output_dim
                self.trunk = nn.Sequential(
                    nn.LayerNorm(joint_dim),
                    nn.Linear(joint_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.label_head = nn.Linear(hidden_size, num_classes)
                self.role_head = nn.Linear(hidden_size, structural_role_dim)

            def forward(self, frame_features, scalar_features, candidate_mask):
                if frame_features.ndim != 4:
                    raise ValueError(
                        "frame_features must have shape [islands,candidates,bins,dim]"
                    )
                if scalar_features.ndim != 3:
                    raise ValueError(
                        "scalar_features must have shape [islands,candidates,dim]"
                    )
                if candidate_mask.ndim != 2:
                    raise ValueError(
                        "candidate_mask must have shape [islands,candidates]"
                    )
                islands, candidates, bins, dim = frame_features.shape
                base_bins = self.left_bins + self.gap_bins + self.right_bins
                extra_bins = sum(left + right for left, right in self.scale_bins)
                expected_bins = base_bins + extra_bins
                if bins != expected_bins:
                    raise ValueError(
                        f"frame_features bins must be {expected_bins}, got {bins}"
                    )
                flat_frames = frame_features.reshape(islands * candidates, bins, dim)
                if self.ptm_input_dim:
                    non_ptm_dim = frame_dim - self.ptm_projected_dim
                    if dim != self.ptm_input_dim + non_ptm_dim:
                        raise ValueError(
                            f"frame_features dim must be "
                            f"{self.ptm_input_dim + non_ptm_dim} with the PTM "
                            f"projector enabled, got {dim}"
                        )
                    ptm_block = flat_frames[..., : self.ptm_input_dim]
                    ptm_proj = self.ptm_projector(ptm_block)
                    if self.ptm_projector_residual:
                        ptm_proj = ptm_proj + self.ptm_residual(ptm_block)
                    flat_frames = torch.cat(
                        (
                            ptm_proj,
                            flat_frames[..., self.ptm_input_dim :],
                        ),
                        dim=-1,
                    )
                projected = self.frame_proj(flat_frames)
                segment_parts = [
                    torch.zeros(self.left_bins, dtype=torch.long),
                    torch.ones(self.gap_bins, dtype=torch.long),
                    torch.full((self.right_bins,), 2, dtype=torch.long),
                ]
                for scale_index, (scale_left, scale_right) in enumerate(self.scale_bins):
                    segment_parts.append(
                        torch.full((scale_left,), 3 + 2 * scale_index, dtype=torch.long)
                    )
                    segment_parts.append(
                        torch.full((scale_right,), 4 + 2 * scale_index, dtype=torch.long)
                    )
                segment_ids = torch.cat(segment_parts).to(projected.device)
                projected = projected + self.segment_embedding[segment_ids]
                encoded = self.encoder(projected)
                left = encoded[:, : self.left_bins].mean(dim=1)
                gap = encoded[
                    :, self.left_bins : self.left_bins + self.gap_bins
                ].mean(dim=1)
                right = encoded[
                    :, self.left_bins + self.gap_bins : base_bins
                ].mean(dim=1)
                readout = torch.cat(
                    (encoded.mean(dim=1), gap, left - right), dim=-1
                )
                scalars = self.scalar_arm(
                    scalar_features.reshape(islands * candidates, -1)
                )
                fused = self.candidate_fuse(torch.cat((readout, scalars), dim=-1))
                fused = fused.reshape(islands, candidates, -1)
                mask = candidate_mask.to(dtype=fused.dtype).unsqueeze(-1)
                fused = fused * mask
                island_encoded = self.island_encoder(
                    fused,
                    attention_mask=candidate_mask.long(),
                )
                joint = self.trunk(torch.cat((fused, island_encoded), dim=-1))
                return {
                    "label": self.label_head(joint),
                    "role": self.role_head(joint),
                }

        return _Network()


@dataclass(frozen=True)
class SplitDecision:
    label: str
    p_cut: float
    p_continue: float
    p_unsure: float
    role: str = ""
    p_role: float = 0.0
    offset_s: float = 0.0


@dataclass(frozen=True)
class SplitEvent:
    event_id: str
    candidate_start_index: int
    candidate_end_index: int
    representative_index: int
    representative_time_s: float
    p_cut: float


@dataclass(frozen=True)
class AcousticSplitV4Planner:
    """v4 acoustic planner using binary cut/continue argmax only."""

    path: str
    sha256: str
    model: Any
    model_config: dict[str, Any]
    feature_config: dict[str, Any]
    normalization: dict[str, Any]
    metadata: dict[str, Any]
    device: str

    @property
    def decision_config(self) -> dict[str, str]:
        return dict(SEMANTIC_SPLIT_V4_DECISION)

    def signature(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_SPLIT_V4_SCHEMA,
            "model_arch": SEMANTIC_SPLIT_V4_MODEL_ARCH,
            "runtime_adapter": SEMANTIC_SPLIT_V4_RUNTIME_ADAPTER,
            "boundary_serialization_contract_id": require_boundary_contract_id(
                self.metadata.get("boundary_serialization_contract_id")
            ),
            "path": self.path,
            "sha256": self.sha256,
            "model_config": self.model_config,
            "feature_config": self.feature_config,
            "decision_config": self.decision_config,
            "metadata": self.metadata,
        }

    def decide_islands(
        self,
        *,
        island_frame_features: Sequence[np.ndarray],
        island_scalar_features: Sequence[np.ndarray],
        max_padded_candidates: int | None = None,
    ) -> list[list[SplitDecision]]:
        import torch

        if len(island_frame_features) != len(island_scalar_features):
            raise ValueError("island frame/scalar feature counts must match")
        counts = [int(np.asarray(frames).shape[0]) for frames in island_frame_features]
        if max_padded_candidates is None:
            raw_limit = os.getenv("ACOUSTIC_SPLIT_MAX_BATCH_CANDIDATES", "auto").strip()
            max_padded_candidates = 128 if raw_limit in {"", "auto"} else int(raw_limit)
        if max_padded_candidates <= 0:
            raise ValueError("max_padded_candidates must be positive")

        decisions: list[list[SplitDecision]] = [[] for _count in counts]
        start = 0
        current_max = 0
        for index, count in enumerate(counts):
            proposed_max = max(current_max, count)
            proposed_width = index - start + 1
            if start < index and proposed_max * proposed_width > max_padded_candidates:
                _fill_split_batch(
                    self,
                    start,
                    index,
                    island_frame_features,
                    island_scalar_features,
                    decisions,
                )
                start = index
                current_max = 0
            current_max = max(current_max, count)
        if start < len(counts):
            _fill_split_batch(
                self,
                start,
                len(counts),
                island_frame_features,
                island_scalar_features,
                decisions,
            )
        return decisions

    def plan_events(
        self,
        *,
        candidate_times_by_island: Sequence[Sequence[float]],
        island_frame_features: Sequence[np.ndarray],
        island_scalar_features: Sequence[np.ndarray],
        event_id_prefix: str = "split-v4",
    ) -> list[list[SplitEvent]]:
        decisions = self.decide_islands(
            island_frame_features=island_frame_features,
            island_scalar_features=island_scalar_features,
        )
        if len(candidate_times_by_island) != len(decisions):
            raise ValueError("candidate time/island counts must match")
        return [
            aggregate_cut_event_runs(
                candidate_times_s=times,
                decisions=rows,
                event_id_prefix=f"{event_id_prefix}-{island_index}",
            )
            for island_index, (times, rows) in enumerate(
                zip(candidate_times_by_island, decisions)
            )
        ]


def _fill_split_batch(
    planner: AcousticSplitV4Planner,
    start: int,
    end: int,
    island_frame_features: Sequence[np.ndarray],
    island_scalar_features: Sequence[np.ndarray],
    decisions: list[list[SplitDecision]],
) -> None:
    import torch

    frame_groups = island_frame_features[start:end]
    scalar_groups = island_scalar_features[start:end]
    counts = [int(np.asarray(frames).shape[0]) for frames in frame_groups]
    max_count = max(counts, default=0)
    if max_count <= 0:
        return
    frame_mean = np.asarray(planner.normalization["frame_mean"], dtype=np.float32)
    frame_std = np.maximum(
        np.asarray(planner.normalization["frame_std"], dtype=np.float32), 1e-6
    )
    scalar_mean = np.asarray(planner.normalization["scalar_mean"], dtype=np.float32)
    scalar_std = np.maximum(
        np.asarray(planner.normalization["scalar_std"], dtype=np.float32), 1e-6
    )
    exemplar = counts.index(max_count)
    first = np.asarray(frame_groups[exemplar], dtype=np.float32)
    bins, frame_dim = int(first.shape[1]), int(first.shape[2])
    scalar_dim = int(np.asarray(scalar_groups[exemplar]).shape[1])
    frames = np.zeros((len(counts), max_count, bins, frame_dim), dtype=np.float32)
    scalars = np.zeros((len(counts), max_count, scalar_dim), dtype=np.float32)
    mask = np.zeros((len(counts), max_count), dtype=np.int64)
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        frames[index, :count] = (
            np.asarray(frame_groups[index], dtype=np.float32) - frame_mean
        ) / frame_std
        scalars[index, :count] = (
            np.asarray(scalar_groups[index], dtype=np.float32) - scalar_mean
        ) / scalar_std
        mask[index, :count] = 1
    with torch.inference_mode():
        outputs = planner.model(
            torch.from_numpy(frames).to(planner.device),
            torch.from_numpy(scalars).to(planner.device),
            torch.from_numpy(mask).to(planner.device),
        )
        probabilities = torch.softmax(outputs["label"], dim=-1).detach().cpu().numpy()
    if probabilities.shape[-1] != len(SEMANTIC_SPLIT_TRAINING_LABELS):
        raise RuntimeError("Acoustic Split v4 model emitted a non-binary output head")
    for island_index, count in enumerate(counts):
        rows: list[SplitDecision] = []
        for position in range(count):
            row = probabilities[island_index, position]
            label_index = int(np.argmax(row))
            rows.append(
                SplitDecision(
                    label=SEMANTIC_SPLIT_TRAINING_LABELS[label_index],
                    p_cut=float(row[0]),
                    p_continue=float(row[1]),
                    p_unsure=0.0,
                )
            )
        decisions[start + island_index] = rows


def aggregate_cut_event_runs(
    *,
    candidate_times_s: Sequence[float],
    decisions: Sequence[SplitDecision],
    event_id_prefix: str = "split-v4",
) -> list[SplitEvent]:
    """Collapse each consecutive argmax=cut run into one acoustic event."""

    if len(candidate_times_s) != len(decisions):
        raise ValueError("candidate time/decision counts must match")
    events: list[SplitEvent] = []
    position = 0
    while position < len(decisions):
        if decisions[position].label != "cut":
            position += 1
            continue
        start = position
        while position + 1 < len(decisions) and decisions[position + 1].label == "cut":
            position += 1
        end = position
        representative = max(
            range(start, end + 1), key=lambda index: decisions[index].p_cut
        )
        events.append(
            SplitEvent(
                event_id=f"{event_id_prefix}-event-{len(events):03d}",
                candidate_start_index=start,
                candidate_end_index=end,
                representative_index=representative,
                representative_time_s=float(candidate_times_s[representative]),
                p_cut=float(decisions[representative].p_cut),
            )
        )
        position += 1
    return events


def load_acoustic_split_v4_planner(
    path: str | Path,
    *,
    device: str = "auto",
    expected_ptm_repo_id: str | None = None,
) -> AcousticSplitV4Planner:
    import torch

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SEMANTIC_SPLIT_V4_SCHEMA:
        raise ValueError(
            f"unsupported Acoustic Split Model schema: {payload.get('schema')!r}; "
            f"expected {SEMANTIC_SPLIT_V4_SCHEMA!r}"
        )
    if payload.get("model_arch") != SEMANTIC_SPLIT_V4_MODEL_ARCH:
        raise ValueError(
            "Acoustic Split v4 model_arch must be "
            f"{SEMANTIC_SPLIT_V4_MODEL_ARCH!r}"
        )
    metadata = dict(payload.get("metadata") or {})
    require_boundary_contract_id(
        metadata.get("boundary_serialization_contract_id")
    )
    if tuple(metadata.get("training_labels") or ()) != SEMANTIC_SPLIT_TRAINING_LABELS:
        raise ValueError("Acoustic Split v4 training_labels must be cut/continue")
    if tuple(metadata.get("excluded_training_labels") or ()) != ("unsure",):
        raise ValueError("Acoustic Split v4 must exclude unsure from training")
    model_config = dict(payload.get("model_config") or {})
    if int(model_config.get("num_classes") or 0) != 2:
        raise ValueError("Acoustic Split v4 requires a binary output head")
    model = IslandCandidateSequenceNetwork(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    actual_device = _model_device(device)
    model.to(actual_device).eval()
    if expected_ptm_repo_id is not None:
        validate_checkpoint_repo_id(
            metadata.get("ptm_repo_id"), expected_ptm_repo_id,
            checkpoint_kind="Acoustic Split Model", metadata_key="metadata.ptm_repo_id",
        )
    return AcousticSplitV4Planner(
        path=str(checkpoint_path), sha256=_sha256(checkpoint_path), model=model,
        model_config=model_config, feature_config=dict(payload.get("feature_config") or {}),
        normalization=dict(payload.get("normalization") or {}), metadata=metadata,
        device=str(actual_device),
    )


def build_acoustic_split_v4_checkpoint(
    *, model: Any, model_config: dict[str, Any], feature_config: dict[str, Any],
    normalization: dict[str, Any], metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SEMANTIC_SPLIT_V4_SCHEMA,
        "model_arch": SEMANTIC_SPLIT_V4_MODEL_ARCH,
        "model_config": {**model_config, "num_classes": 2},
        "feature_config": {**feature_config, "schema": SEMANTIC_SPLIT_FEATURE_SCHEMA},
        "normalization": dict(normalization),
        "decision_config": dict(SEMANTIC_SPLIT_V4_DECISION),
        "metadata": {
            **metadata,
            "boundary_serialization_contract_id": (
                ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            ),
            "artifact": dict(SEMANTIC_SPLIT_V4_ARTIFACT),
            "runtime_adapter": SEMANTIC_SPLIT_V4_RUNTIME_ADAPTER,
            "canonical_labels": list(SEMANTIC_SPLIT_LABELS),
            "training_labels": list(SEMANTIC_SPLIT_TRAINING_LABELS),
            "excluded_training_labels": ["unsure"],
        },
        "model_state_dict": model.state_dict(),
    }


def _model_device(requested: str):
    import torch

    value = str(requested or "auto").strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        value = "cpu"
    return torch.device(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
