"""CueQC Mamba v4 binary keep/drop classifier.

Four feature arms fused into a 2-way display decision:

* ASR encoder features  [B, T, asr_dim]   → Bi-Mamba2 → masked mean+max pool
* ASR token trace       [B, L, token_dim] → Bi-Mamba2 → masked mean+max pool
* decoder aggregate     [B, decoder_dim]  → LayerNorm → MLP
* structured metadata   [B, structured_dim] → LayerNorm → MLP

Concat → fusion MLP → display logits [B, 2]. Label convention: ``0 = drop``,
``1 = keep``; ``logits[:, 0]`` is drop, ``logits[:, 1]`` is keep.

Pooling is masked mean + max (NOT last hidden) — candidate lengths vary, so the
last valid frame would be padding-contaminated and display quality depends on
the whole segment, not the final frame.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _install_mamba2_warning_filter() -> None:
    import warnings

    warnings.filterwarnings("ignore", message=".*Mamba2.*")
    warnings.filterwarnings("ignore", message=".*supports.*generation.*")


def masked_mean_max_pool(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean + max pooling over the time dimension.

    ``seq``:  [B, T, D]  (post-Mamba hidden states)
    ``mask``: [B, T]     (1 = valid, 0 = padding)
    Returns:  [B, 2D]    (mean ⊕ max)
    """
    valid = mask.unsqueeze(-1).to(seq.dtype)  # [B, T, 1]
    lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B, 1]
    summed = (seq * valid).sum(dim=1)  # [B, D]
    mean = summed / lengths

    # Masked max: set padding positions to a very negative value before max.
    neg_inf = torch.finfo(seq.dtype).min
    masked_seq = seq.masked_fill(valid < 0.5, neg_inf)
    maxv = masked_seq.max(dim=1).values  # [B, D]
    return torch.cat([mean, maxv], dim=-1)  # [B, 2D]


class _BiMamba2Arm(nn.Module):
    """Linear → forward+backward Mamba2 → LayerNorm over a sequence arm."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        state_size: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        if hidden_size * 2 != num_heads * head_dim:
            raise ValueError("hidden_size * 2 must equal num_heads * head_dim")
        self.proj = nn.Linear(input_dim, hidden_size)

        from transformers import Mamba2Config, Mamba2Model

        mamba_config = Mamba2Config(
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
        self.forward_net = Mamba2Model(mamba_config)
        self.backward_net = Mamba2Model(mamba_config)
        # Bidirectional Mamba2 → 2 * hidden_size output channels.
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.out_dim = hidden_size * 2

    def _run(self, net, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return net(inputs_embeds=hidden, attention_mask=mask, use_cache=False).last_hidden_state

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(seq)
        fwd = self._run(self.forward_net, hidden, mask)
        rev_hidden = torch.flip(hidden, dims=[1])
        rev_mask = torch.flip(mask, dims=[1])
        bwd = self._run(self.backward_net, rev_hidden, rev_mask)
        bwd = torch.flip(bwd, dims=[1])
        return self.norm(torch.cat([fwd, bwd], dim=-1))  # [B, T, 2*hidden]


class _MlpArm(nn.Module):
    """LayerNorm → Linear(→64) → ReLU → Dropout for fixed-dim feature arms."""

    def __init__(self, *, input_dim: int, output_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class CueQCMambaV4Binary(nn.Module):
    """CueQC v4 binary classifier (4 arms -> fusion -> [B, 2])."""

    def __init__(
        self,
        *,
        asr_dim: int,
        token_dim: int,
        decoder_dim: int,
        structured_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        state_size: int = 32,
        num_heads: int = 4,
        head_dim: int = 64,
        n_groups: int = 4,
        chunk_size: int = 64,
        mlp_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _install_mamba2_warning_filter()

        self.asr_dim = int(asr_dim)
        self.token_dim = int(token_dim)
        self.decoder_dim = int(decoder_dim)
        self.structured_dim = int(structured_dim)
        self.hidden_size = int(hidden_size)

        self.asr_arm = _BiMamba2Arm(
            input_dim=self.asr_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            state_size=state_size,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            chunk_size=chunk_size,
        )
        self.token_arm = _BiMamba2Arm(
            input_dim=self.token_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            state_size=state_size,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            chunk_size=chunk_size,
        )
        self.decoder_arm = _MlpArm(input_dim=self.decoder_dim, output_dim=mlp_dim, dropout=dropout)
        self.structured_arm = _MlpArm(input_dim=self.structured_dim, output_dim=mlp_dim, dropout=dropout)

        # After masked mean+max pooling, each Mamba arm yields 2 * (2*hidden_size)
        # (the arm's LayerNorm output is 2*hidden; mean+max doubles it again).
        pooled_dim = self.asr_arm.out_dim * 2  # 4 * hidden_size
        token_pooled_dim = self.token_arm.out_dim * 2  # 4 * hidden_size
        fusion_in = pooled_dim + token_pooled_dim + self.decoder_arm.out_dim + self.structured_arm.out_dim

        self.fusion_norm = nn.LayerNorm(fusion_in)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

        self.model_config: dict[str, Any] = {
            "model_type": "CueQCMambaV4Binary",
            "asr_dim": self.asr_dim,
            "token_dim": self.token_dim,
            "decoder_dim": self.decoder_dim,
            "structured_dim": self.structured_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "state_size": state_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "n_groups": n_groups,
            "chunk_size": chunk_size,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
            "pooling": "masked_mean_max",
            "uses_bge": False,
            "text_embedding": "none",
        }

    def forward(
        self,
        *,
        asr_frames: torch.Tensor,
        asr_mask: torch.Tensor,
        token_trace: torch.Tensor,
        token_mask: torch.Tensor,
        decoder_stats: torch.Tensor,
        structured: torch.Tensor,
    ) -> torch.Tensor:
        """Return display logits [B, 2] (col 0 = drop, col 1 = keep)."""
        asr_seq = self.asr_arm(asr_frames, asr_mask)  # [B, T, 2H]
        asr_hidden = masked_mean_max_pool(asr_seq, asr_mask)  # [B, 4H]

        tok_seq = self.token_arm(token_trace, token_mask)  # [B, L, 2H]
        tok_hidden = masked_mean_max_pool(tok_seq, token_mask)  # [B, 4H]

        dec_hidden = self.decoder_arm(decoder_stats)  # [B, mlp_dim]
        struct_hidden = self.structured_arm(structured)  # [B, mlp_dim]

        fused = torch.cat([asr_hidden, tok_hidden, dec_hidden, struct_hidden], dim=-1)
        return self.fusion(self.fusion_norm(fused))  # [B, 2]
