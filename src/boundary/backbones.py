from __future__ import annotations

import logging

import torch
from torch import nn

TRANSFORMERS_MAMBA2_BACKBONE = "transformers.Mamba2Model"
MAMBA2_STATEFUL_INFERENCE_CHUNK_FRAMES = 1024


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
        valid_prefix_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if hidden_size * 2 != num_heads * head_dim:
            raise ValueError("hidden_size * 2 must equal num_heads * head_dim")
        self.bidirectional = bool(bidirectional)
        self.valid_prefix_bidirectional = bool(valid_prefix_bidirectional)
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
        if self.valid_prefix_bidirectional and attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        forward = self.forward_model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        if self.backward_model is None:
            return self.norm(forward)
        if self.valid_prefix_bidirectional and attention_mask is not None:
            reversed_hidden = self._reverse_valid_prefix(hidden, attention_mask)
            reversed_mask = attention_mask
        else:
            reversed_hidden = torch.flip(hidden, dims=[1])
            reversed_mask = None if attention_mask is None else torch.flip(attention_mask, dims=[1])
        backward = self.backward_model(
            inputs_embeds=reversed_hidden,
            attention_mask=reversed_mask,
            use_cache=False,
        ).last_hidden_state
        if self.valid_prefix_bidirectional and attention_mask is not None:
            backward = self._reverse_valid_prefix(backward, attention_mask)
        else:
            backward = torch.flip(backward, dims=[1])
        output = self.norm(torch.cat([forward, backward], dim=-1))
        if self.valid_prefix_bidirectional and attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(output.dtype)
        return output

    @staticmethod
    def _reverse_valid_prefix(tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length = tuple(attention_mask.shape)
        positions = torch.arange(length, device=tensor.device).reshape(1, length)
        valid_lengths = attention_mask.to(dtype=torch.long).sum(dim=1, keepdim=True)
        reverse_positions = torch.where(
            positions < valid_lengths,
            valid_lengths - 1 - positions,
            positions,
        )
        gather_index = reverse_positions.reshape(batch, length, 1).expand_as(tensor)
        return torch.gather(tensor, dim=1, index=gather_index)


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
        inference_chunk_size: int = MAMBA2_STATEFUL_INFERENCE_CHUNK_FRAMES,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if hidden_size * 2 != num_heads * head_dim:
            raise ValueError("hidden_size * 2 must equal num_heads * head_dim")
        if inference_chunk_size < 0:
            raise ValueError("inference_chunk_size must be non-negative")
        self.bidirectional = bool(bidirectional)
        self.inference_chunk_size = int(inference_chunk_size)

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
        forward = self._run_direction(
            self.forward_model,
            hidden,
            attention_mask=attention_mask,
        )
        if self.backward_model is None:
            return self.norm(forward)
        reversed_hidden = torch.flip(hidden, dims=[1])
        reversed_mask = None if attention_mask is None else torch.flip(attention_mask, dims=[1])
        backward = self._run_direction(
            self.backward_model,
            reversed_hidden,
            attention_mask=reversed_mask,
        )
        backward = torch.flip(backward, dims=[1])
        return self.norm(torch.cat([forward, backward], dim=-1))

    def _run_direction(
        self,
        model: nn.Module,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one causal direction with exact recurrent state across chunks.

        The pure-PyTorch Mamba2 path materializes very large temporary tensors
        for long sequences. During evaluation, carry the model's convolution
        and recurrent cache across bounded chunks. This is the same causal scan
        as one full call; no frames or context are discarded. Training keeps the
        original full-sequence graph.
        """

        chunk_size = self.inference_chunk_size
        if self.training or chunk_size <= 0 or hidden.shape[1] <= chunk_size:
            return model(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state

        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=model.config)
        outputs: list[torch.Tensor] = []
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(hidden.shape[1], start + chunk_size)
            chunk_mask = (
                None if attention_mask is None else attention_mask[:, start:end]
            )
            result = model(
                inputs_embeds=hidden[:, start:end],
                attention_mask=chunk_mask,
                cache_params=cache,
                use_cache=True,
            )
            cache = result.cache_params
            outputs.append(result.last_hidden_state)
        return torch.cat(outputs, dim=1)


class SpeechIslandSequenceClassifier(nn.Module):
    """Shared frame encoder used by single-head and dual-head boundary models."""

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
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
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
        backbone_kwargs.setdefault("state_size", 32)
        backbone_kwargs.setdefault("n_groups", 2)
        backbone_kwargs.setdefault("chunk_size", 8)
        backbone_kwargs.setdefault("conv_kernel", 4)
        backbone_kwargs.setdefault("bidirectional", True)
        self.model_config = {
            "input_dim": input_dim,
            "backbone": self.backbone_name,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_dim": output_dim,
            **backbone_kwargs,
        }
        self.proj = nn.Linear(input_dim, hidden_size)
        self.backbone = Mamba2TemporalEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            **backbone_kwargs,
        )
        self.norm = nn.LayerNorm(self.backbone.output_dim)
        self.head = nn.Linear(self.backbone.output_dim, output_dim)

    def forward(
        self,
        features: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.backbone(
            self.proj(features),
            attention_mask=attention_mask,
        )
        logits = self.head(self.norm(hidden))
        if attention_mask is not None:
            logits = logits * attention_mask.unsqueeze(-1).to(logits.dtype)
        return logits


def normalize_boundary_backbone(backbone: str) -> str:
    normalized = (backbone or "").strip()
    if normalized == TRANSFORMERS_MAMBA2_BACKBONE:
        return TRANSFORMERS_MAMBA2_BACKBONE
    raise ValueError(
        "unsupported boundary refiner backbone: "
        f"{backbone!r}; only {TRANSFORMERS_MAMBA2_BACKBONE} is supported"
    )
