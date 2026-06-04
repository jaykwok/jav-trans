from __future__ import annotations

import torch
from torch import nn

TRANSFORMERS_MAMBA2_BACKBONE = "transformers.Mamba2Model"


class TinyMamba2BoundaryBackbone(nn.Module):
    """Small pure-PyTorch Mamba2 wrapper for Windows-friendly boundary research."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        state_size: int = 32,
        num_heads: int = 8,
        head_dim: int = 64,
        n_groups: int = 4,
        chunk_size: int = 64,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if hidden_size * 2 != num_heads * head_dim:
            raise ValueError("hidden_size * expand must equal num_heads * head_dim")
        self.bidirectional = bool(bidirectional)
        self.proj = nn.Linear(input_dim, hidden_size)

        from transformers import Mamba2Config, Mamba2Model

        config = Mamba2Config(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            state_size=state_size,
            expand=2,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            conv_kernel=4,
            chunk_size=chunk_size,
            use_cache=False,
        )
        self.forward_model = Mamba2Model(config)
        self.backward_model = Mamba2Model(config) if self.bidirectional else None
        out_dim = hidden_size * (2 if self.bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)

    @property
    def output_dim(self) -> int:
        return int(self.norm.normalized_shape[0])

    def forward(
        self,
        features: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, dim]")
        hidden = self.proj(features)
        forward = self.forward_model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        if self.backward_model is None:
            return self.norm(forward)
        reversed_hidden = torch.flip(hidden, dims=[1])
        reversed_mask = None if attention_mask is None else torch.flip(attention_mask, dims=[1])
        backward = self.backward_model(
            inputs_embeds=reversed_hidden,
            attention_mask=reversed_mask,
            use_cache=False,
        ).last_hidden_state
        backward = torch.flip(backward, dims=[1])
        return self.norm(torch.cat([forward, backward], dim=-1))


class BoundarySequenceClassifier(nn.Module):
    """Mamba2 frame/gap sequence classifier shared by training and runtime loading."""

    def __init__(
        self,
        *,
        input_dim: int,
        backbone: str = TRANSFORMERS_MAMBA2_BACKBONE,
        hidden_size: int = 128,
        num_layers: int = 2,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        self.backbone_name = normalize_boundary_backbone(backbone)
        if "num_heads" not in backbone_kwargs:
            backbone_kwargs["num_heads"] = 4
        if "head_dim" not in backbone_kwargs:
            num_heads = int(backbone_kwargs["num_heads"])
            if (hidden_size * 2) % num_heads != 0:
                raise ValueError("hidden_size * expand must be divisible by num_heads")
            backbone_kwargs["head_dim"] = (hidden_size * 2) // num_heads
        self.model_config = {
            "input_dim": input_dim,
            "backbone": self.backbone_name,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            **backbone_kwargs,
        }
        self.backbone = TinyMamba2BoundaryBackbone(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **backbone_kwargs,
        )
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.backbone(features, attention_mask=attention_mask)
        return self.head(hidden).squeeze(-1)


def normalize_boundary_backbone(backbone: str) -> str:
    normalized = (backbone or "").strip()
    if normalized == TRANSFORMERS_MAMBA2_BACKBONE:
        return TRANSFORMERS_MAMBA2_BACKBONE
    raise ValueError(
        "unsupported boundary refiner backbone: "
        f"{backbone!r}; only {TRANSFORMERS_MAMBA2_BACKBONE} is supported"
    )
