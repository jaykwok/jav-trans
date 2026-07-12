from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from asr.backends.qwen import validate_checkpoint_repo_id
from boundary.backbones import Mamba2TemporalEncoder


SEMANTIC_SPLIT_LABELS = ("cut", "continue", "unsure")
SEMANTIC_SPLIT_FEATURE_SCHEMA = "semantic_split_candidate_features_v1"

SEMANTIC_SPLIT_V2_SCHEMA = "semantic_split_verifier_v2"
SEMANTIC_SPLIT_V2_MODEL_ARCH = "island_candidate_sequence_v1"
SEMANTIC_SPLIT_V2_RUNTIME_ADAPTER = "island_candidate_sequence_cut_v1"
SEMANTIC_SPLIT_STRUCTURAL_ROLES = (
    "none",
    "speech_to_speech",
    "speech_to_noise",
    "noise_to_speech",
)
SEMANTIC_SPLIT_OMNI_AUX_NAMES = ("left_complete", "right_complete", "merged_better")
SEMANTIC_SPLIT_V2_ARTIFACT = {
    "name": "semantic_split_model",
    "display_name": "Semantic Split Model",
    "version": "v2",
    "pipeline_stage": 3,
    "pipeline_role": "cut_continue_unsure_decision",
}
SEMANTIC_SPLIT_V2_DEFAULT_DECISION = {
    "short_core_max_s": 6.0,
    "short_core_cut_threshold": 0.90,
    "normal_cut_threshold": 0.75,
    "min_chunk_after_split_s": 1.2,
    "duration_pressure_enabled": False,
    "duration_pressure_log_median": 0.0,
    "duration_pressure_log_mad": 0.0,
    "duration_pressure_z": 0.0,
    "duration_pressure_floor": 0.50,
}


class IslandCandidateSequenceNetwork:
    """v2 island-level candidate-sequence network factory.

    Encodes every candidate with a per-candidate frame stack
    (``frame_proj`` -> bidirectional Mamba2 over bins -> mean pool + scalar arm)
    and then runs a second
    bidirectional Mamba2 over the ordered candidates of one speech island and
    emits per-candidate heads:

    - ``gate``: deployment-aligned binary cut logit
    - ``label``: cut / continue / unsure auxiliary logits
    - ``role``: structural-role auxiliary logits (none / speech_to_speech /
      speech_to_noise / noise_to_speech)
    - ``omni``: left_complete / right_complete / merged_better auxiliary logits
    - ``offset``: regression of (truth cut time - candidate time) in seconds,
      supervised only on rows with precise truth (hardmix); Omni rows mask
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
        aux_label_dim: int = 3,
        structural_role_dim: int = 4,
        omni_aux_dim: int = 3,
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
        if aux_label_dim != len(SEMANTIC_SPLIT_LABELS):
            raise ValueError("island split network requires aux_label_dim=3")
        if structural_role_dim != len(SEMANTIC_SPLIT_STRUCTURAL_ROLES):
            raise ValueError("island split network requires structural_role_dim=4")
        if omni_aux_dim != len(SEMANTIC_SPLIT_OMNI_AUX_NAMES):
            raise ValueError("island split network requires omni_aux_dim=3")
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
                self.gate_head = nn.Linear(hidden_size, 1)
                self.label_head = nn.Linear(hidden_size, aux_label_dim)
                self.role_head = nn.Linear(hidden_size, structural_role_dim)
                self.omni_head = nn.Linear(hidden_size, omni_aux_dim)
                self.offset_head = nn.Linear(hidden_size, 1)

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
                    "gate": self.gate_head(joint).squeeze(-1),
                    "label": self.label_head(joint),
                    "role": self.role_head(joint),
                    "omni": self.omni_head(joint),
                    "offset": self.offset_head(joint).squeeze(-1),
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
class SemanticSplitIslandVerifier:
    """v2 verifier: joint decisions over the ordered candidates of one island."""

    path: str
    sha256: str
    model: Any
    model_config: dict[str, Any]
    feature_config: dict[str, Any]
    normalization: dict[str, Any]
    decision_config: dict[str, Any]
    metadata: dict[str, Any]
    device: str

    def signature(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_SPLIT_V2_SCHEMA,
            "model_arch": SEMANTIC_SPLIT_V2_MODEL_ARCH,
            "runtime_adapter": SEMANTIC_SPLIT_V2_RUNTIME_ADAPTER,
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
    ) -> list[list[SplitDecision]]:
        import torch

        if len(island_frame_features) != len(island_scalar_features):
            raise ValueError("island frame/scalar feature counts must match")
        counts = [int(np.asarray(frames).shape[0]) for frames in island_frame_features]
        max_count = max(counts, default=0)
        if max_count <= 0:
            return [[] for _count in counts]
        frame_mean = np.asarray(self.normalization["frame_mean"], dtype=np.float32)
        frame_std = np.maximum(
            np.asarray(self.normalization["frame_std"], dtype=np.float32), 1e-6
        )
        scalar_mean = np.asarray(self.normalization["scalar_mean"], dtype=np.float32)
        scalar_std = np.maximum(
            np.asarray(self.normalization["scalar_std"], dtype=np.float32), 1e-6
        )
        first = np.asarray(island_frame_features[counts.index(max_count)], dtype=np.float32)
        bins, frame_dim = int(first.shape[1]), int(first.shape[2])
        scalar_dim = int(np.asarray(island_scalar_features[counts.index(max_count)]).shape[1])
        frames = np.zeros((len(counts), max_count, bins, frame_dim), dtype=np.float32)
        scalars = np.zeros((len(counts), max_count, scalar_dim), dtype=np.float32)
        mask = np.zeros((len(counts), max_count), dtype=np.int64)
        for index, count in enumerate(counts):
            if count <= 0:
                continue
            island_frames = np.asarray(island_frame_features[index], dtype=np.float32)
            island_scalars = np.asarray(island_scalar_features[index], dtype=np.float32)
            frames[index, :count] = (island_frames - frame_mean) / frame_std
            scalars[index, :count] = (island_scalars - scalar_mean) / scalar_std
            mask[index, :count] = 1
        with torch.inference_mode():
            outputs = self.model(
                torch.from_numpy(frames).to(self.device),
                torch.from_numpy(scalars).to(self.device),
                torch.from_numpy(mask).to(self.device),
            )
            gate = torch.sigmoid(outputs["gate"]).detach().cpu().numpy()
            aux = (
                torch.softmax(outputs["label"], dim=-1).detach().cpu().numpy()
            )
            role = (
                torch.softmax(outputs["role"], dim=-1).detach().cpu().numpy()
            )
            offset = outputs["offset"].detach().cpu().numpy()
        decisions: list[list[SplitDecision]] = []
        for index, count in enumerate(counts):
            rows: list[SplitDecision] = []
            for position in range(count):
                p_gate = float(gate[index, position])
                _, a_continue, a_unsure = (
                    float(aux[index, position, 0]),
                    float(aux[index, position, 1]),
                    float(aux[index, position, 2]),
                )
                residual = max(0.0, 1.0 - p_gate)
                denominator = max(a_continue + a_unsure, 1e-6)
                if p_gate >= 0.5:
                    label = "cut"
                elif a_continue >= a_unsure:
                    label = "continue"
                else:
                    label = "unsure"
                role_index = int(np.argmax(role[index, position]))
                rows.append(
                    SplitDecision(
                        label=label,
                        p_cut=p_gate,
                        p_continue=residual * a_continue / denominator,
                        p_unsure=residual * a_unsure / denominator,
                        role=SEMANTIC_SPLIT_STRUCTURAL_ROLES[role_index],
                        p_role=float(role[index, position, role_index]),
                        offset_s=float(offset[index, position]),
                    )
                )
            decisions.append(rows)
        return decisions


def load_semantic_split_feature_config(path: str | Path) -> dict[str, Any]:
    """Read only feature_config from a split checkpoint, without building the
    model or moving weights to a device (cheap enough for the boundary-cache
    signature path)."""

    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return dict(payload.get("feature_config") or {})


def load_semantic_split_verifier(
    path: str | Path,
    *,
    device: str = "auto",
    expected_ptm_repo_id: str | None = None,
) -> "SemanticSplitIslandVerifier":
    import torch

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SEMANTIC_SPLIT_V2_SCHEMA:
        raise ValueError(
            f"unsupported Semantic Split Model schema: {payload.get('schema')!r}; "
            f"expected {SEMANTIC_SPLIT_V2_SCHEMA!r}"
        )
    return _load_island_verifier(
        payload,
        checkpoint_path=checkpoint_path,
        device=device,
        expected_ptm_repo_id=expected_ptm_repo_id,
    )


def _load_island_verifier(
    payload: dict[str, Any],
    *,
    checkpoint_path: Path,
    device: str,
    expected_ptm_repo_id: str | None,
) -> "SemanticSplitIslandVerifier":
    if payload.get("model_arch") != SEMANTIC_SPLIT_V2_MODEL_ARCH:
        raise ValueError(
            f"Semantic Split v2 model must use {SEMANTIC_SPLIT_V2_MODEL_ARCH!r}"
        )
    metadata = dict(payload.get("metadata") or {})
    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Semantic Split v2 metadata.artifact is required")
    for key, expected in SEMANTIC_SPLIT_V2_ARTIFACT.items():
        if artifact.get(key) != expected:
            raise ValueError(
                f"Semantic Split v2 metadata.artifact.{key} must be {expected!r}"
            )
    if metadata.get("runtime_adapter") != SEMANTIC_SPLIT_V2_RUNTIME_ADAPTER:
        raise ValueError(
            "Semantic Split v2 metadata.runtime_adapter must be "
            f"{SEMANTIC_SPLIT_V2_RUNTIME_ADAPTER!r}"
        )
    if tuple(metadata.get("labels") or ()) != SEMANTIC_SPLIT_LABELS:
        raise ValueError(f"Semantic Split v2 labels must be {SEMANTIC_SPLIT_LABELS!r}")
    if tuple(metadata.get("structural_roles") or ()) != SEMANTIC_SPLIT_STRUCTURAL_ROLES:
        raise ValueError(
            "Semantic Split v2 structural_roles must be "
            f"{SEMANTIC_SPLIT_STRUCTURAL_ROLES!r}"
        )
    feature_config = dict(payload.get("feature_config") or {})
    if feature_config.get("schema") != SEMANTIC_SPLIT_FEATURE_SCHEMA:
        raise ValueError(
            f"Semantic Split v2 feature schema must be {SEMANTIC_SPLIT_FEATURE_SCHEMA!r}"
        )
    model_config = dict(payload.get("model_config") or {})
    model = IslandCandidateSequenceNetwork(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    actual_device = _model_device(device)
    model.to(actual_device).eval()
    if expected_ptm_repo_id is not None:
        validate_checkpoint_repo_id(
            metadata.get("ptm_repo_id"),
            expected_ptm_repo_id,
            checkpoint_kind="Semantic Split Model",
            metadata_key="metadata.ptm_repo_id",
        )
    decision_config = {
        **SEMANTIC_SPLIT_V2_DEFAULT_DECISION,
        **dict(payload.get("decision_config") or {}),
    }
    return SemanticSplitIslandVerifier(
        path=str(checkpoint_path),
        sha256=_sha256(checkpoint_path),
        model=model,
        model_config=model_config,
        feature_config=feature_config,
        normalization=dict(payload.get("normalization") or {}),
        decision_config=decision_config,
        metadata=metadata,
        device=str(actual_device),
    )


def build_semantic_split_island_checkpoint(
    *,
    model: Any,
    model_config: dict[str, Any],
    feature_config: dict[str, Any],
    normalization: dict[str, Any],
    metadata: dict[str, Any],
    decision_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        **SEMANTIC_SPLIT_V2_ARTIFACT,
        **dict(metadata.get("artifact") or {}),
    }
    return {
        "schema": SEMANTIC_SPLIT_V2_SCHEMA,
        "model_arch": SEMANTIC_SPLIT_V2_MODEL_ARCH,
        "model_config": dict(model_config),
        "feature_config": {
            **feature_config,
            "schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
        },
        "normalization": dict(normalization),
        "decision_config": {
            **SEMANTIC_SPLIT_V2_DEFAULT_DECISION,
            **dict(decision_config or {}),
        },
        "metadata": {
            **metadata,
            "artifact": artifact,
            "runtime_adapter": SEMANTIC_SPLIT_V2_RUNTIME_ADAPTER,
            "labels": list(SEMANTIC_SPLIT_LABELS),
            "structural_roles": list(SEMANTIC_SPLIT_STRUCTURAL_ROLES),
            "omni_aux_names": list(SEMANTIC_SPLIT_OMNI_AUX_NAMES),
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
