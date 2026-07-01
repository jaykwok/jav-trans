from __future__ import annotations

import logging

import torch
from torch import nn

TRANSFORMERS_MAMBA2_BACKBONE = "transformers.Mamba2Model"


class _Mamba2FastPathWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "The fast path is not available" not in record.getMessage()


def _install_mamba2_warning_filter() -> None:
    logger = logging.getLogger("transformers.models.mamba2.modeling_mamba2")
    if not any(isinstance(item, _Mamba2FastPathWarningFilter) for item in logger.filters):
        logger.addFilter(_Mamba2FastPathWarningFilter())


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
            raise ValueError("hidden_size * 2 must equal num_heads * head_dim")
        self.bidirectional = bool(bidirectional)
        self.proj = nn.Linear(input_dim, hidden_size)

        _install_mamba2_warning_filter()
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


class Mamba2TemporalEncoder(nn.Module):
    """Mamba2 temporal encoder for already-projected frame features."""

    def __init__(
        self,
        *,
        hidden_size: int = 128,
        num_layers: int = 2,
        state_size: int = 32,
        num_heads: int = 4,
        head_dim: int = 64,
        n_groups: int = 2,
        conv_kernel: int = 4,
        chunk_size: int = 8,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if hidden_size * 2 != num_heads * head_dim:
            raise ValueError("hidden_size * 2 must equal num_heads * head_dim")
        self.bidirectional = bool(bidirectional)

        _install_mamba2_warning_filter()
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
            conv_kernel=conv_kernel,
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
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, time, dim]")
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


class SplitBoundaryAdapter(nn.Module):
    """Local temporal adapter for split-boundary frame features."""

    def __init__(self, hidden_size: int, kernel_size: int = 5) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_size)
        self.dwconv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_size,
        )
        self.pwconv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, dim]")
        residual = features
        hidden = self.norm(features).transpose(1, 2)
        hidden = self.dwconv(hidden)
        hidden = self.act(hidden)
        hidden = self.pwconv(hidden).transpose(1, 2)
        return residual + hidden


def compute_temporal_diff_features(
    hidden: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [h_t, h_t - h_{t-1}, h_{t+1} - h_t] with padding-safe diffs."""

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, time, dim]")
    prev_diff = torch.zeros_like(hidden)
    next_diff = torch.zeros_like(hidden)
    prev_diff[:, 1:] = hidden[:, 1:] - hidden[:, :-1]
    next_diff[:, :-1] = hidden[:, 1:] - hidden[:, :-1]
    if attention_mask is not None:
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError("attention_mask must have shape [batch, time]")
        mask = attention_mask.bool()
        valid_prev = torch.zeros_like(mask)
        valid_next = torch.zeros_like(mask)
        valid_prev[:, 1:] = mask[:, 1:] & mask[:, :-1]
        valid_next[:, :-1] = mask[:, 1:] & mask[:, :-1]
        dtype = hidden.dtype
        hidden = hidden * mask.unsqueeze(-1).to(dtype)
        prev_diff = prev_diff * valid_prev.unsqueeze(-1).to(dtype)
        next_diff = next_diff * valid_next.unsqueeze(-1).to(dtype)
    return torch.cat([hidden, prev_diff, next_diff], dim=-1)


class DualBranchDiffBoundarySequenceClassifier(nn.Module):
    """Speech/split scorer with separate temporal branches and split diffs."""

    def __init__(
        self,
        *,
        input_dim: int,
        backbone: str = TRANSFORMERS_MAMBA2_BACKBONE,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 2,
        model_arch: str = "v7-dual-branch-diff",
        split_adapter_kernel_size: int = 5,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim != 2:
            raise ValueError("dual-branch scorer requires output_dim=2")
        self.backbone_name = normalize_boundary_backbone(backbone)
        if "num_heads" not in backbone_kwargs:
            backbone_kwargs["num_heads"] = 4
        if "head_dim" not in backbone_kwargs:
            num_heads = int(backbone_kwargs["num_heads"])
            if (hidden_size * 2) % num_heads != 0:
                raise ValueError("hidden_size * expand must be divisible by num_heads")
            backbone_kwargs["head_dim"] = (hidden_size * 2) // num_heads
        backbone_kwargs.setdefault("state_size", 32)
        backbone_kwargs.setdefault("n_groups", 2)
        backbone_kwargs.setdefault("chunk_size", 8)
        backbone_kwargs.setdefault("conv_kernel", 4)
        backbone_kwargs.setdefault("bidirectional", True)
        self.model_arch = str(model_arch)
        self.model_config = {
            "model_arch": self.model_arch,
            "input_dim": input_dim,
            "backbone": self.backbone_name,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_dim": output_dim,
            "split_adapter_kernel_size": int(split_adapter_kernel_size),
            **backbone_kwargs,
        }
        self.speech_proj = nn.Linear(input_dim, hidden_size)
        self.split_proj = nn.Linear(input_dim, hidden_size)
        self.split_adapter = SplitBoundaryAdapter(
            hidden_size=hidden_size,
            kernel_size=int(split_adapter_kernel_size),
        )
        encoder_kwargs = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            **backbone_kwargs,
        }
        self.speech_backbone = Mamba2TemporalEncoder(**encoder_kwargs)
        self.split_backbone = Mamba2TemporalEncoder(**encoder_kwargs)
        self.speech_norm = nn.LayerNorm(self.speech_backbone.output_dim)
        self.split_norm = nn.LayerNorm(self.split_backbone.output_dim * 3)
        self.speech_head = nn.Linear(self.speech_backbone.output_dim, 1)
        self.split_head = nn.Linear(self.split_backbone.output_dim * 3, 1)
        self.output_dim = int(output_dim)

    def forward(
        self,
        features: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, dim]")
        speech_hidden = self.speech_backbone(
            self.speech_proj(features),
            attention_mask=attention_mask,
        )
        split_hidden = self.split_backbone(
            self.split_adapter(self.split_proj(features)),
            attention_mask=attention_mask,
        )
        split_features = compute_temporal_diff_features(
            split_hidden,
            attention_mask=attention_mask,
        )
        speech_logit = self.speech_head(self.speech_norm(speech_hidden))
        split_logit = self.split_head(self.split_norm(split_features))
        logits = torch.cat([speech_logit, split_logit], dim=-1)
        if attention_mask is not None:
            logits = logits * attention_mask.unsqueeze(-1).to(logits.dtype)
        return logits


class BoundarySequenceClassifier(nn.Module):
    """Mamba2 frame/gap sequence classifier shared by training and runtime loading."""

    def __init__(
        self,
        *,
        input_dim: int,
        backbone: str = TRANSFORMERS_MAMBA2_BACKBONE,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
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
            "output_dim": output_dim,
            **backbone_kwargs,
        }
        self.backbone = TinyMamba2BoundaryBackbone(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **backbone_kwargs,
        )
        self.output_dim = int(output_dim)
        self.head = nn.Linear(self.backbone.output_dim, self.output_dim)

    def forward(
        self,
        features: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.backbone(features, attention_mask=attention_mask)
        logits = self.head(hidden)
        if self.output_dim == 1:
            return logits.squeeze(-1)
        return logits


def normalize_boundary_backbone(backbone: str) -> str:
    normalized = (backbone or "").strip()
    if normalized == TRANSFORMERS_MAMBA2_BACKBONE:
        return TRANSFORMERS_MAMBA2_BACKBONE
    raise ValueError(
        "unsupported boundary refiner backbone: "
        f"{backbone!r}; only {TRANSFORMERS_MAMBA2_BACKBONE} is supported"
    )
