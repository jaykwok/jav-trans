from __future__ import annotations


class AdditionFusionBiLSTM:
    def __new__(
        cls,
        *,
        whisper_dim: int,
        mfcc_dim: int = 40,
        fusion_dim: int = 256,
        hidden_dim: int = 192,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        import torch
        from torch import nn

        class _AdditionFusionBiLSTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.whisper_proj = nn.Linear(whisper_dim, fusion_dim)
                self.mfcc_proj = nn.Linear(mfcc_dim, fusion_dim)
                self.lstm = nn.LSTM(
                    input_size=fusion_dim,
                    hidden_size=hidden_dim,
                    num_layers=layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim * 2, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )

            def forward(self, whisper, mfcc):
                fused = self.whisper_proj(whisper) + self.mfcc_proj(mfcc)
                encoded, _state = self.lstm(fused)
                return self.head(encoded).squeeze(-1)

        return _AdditionFusionBiLSTM()


class AdditionFusionEndpointBiLSTM:
    def __new__(
        cls,
        *,
        whisper_dim: int,
        mfcc_dim: int = 40,
        fusion_dim: int = 256,
        hidden_dim: int = 192,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        from torch import nn

        class _AdditionFusionEndpointBiLSTM(nn.Module):
            output_names = ("speech", "start", "end", "cut")

            def __init__(self) -> None:
                super().__init__()
                self.whisper_proj = nn.Linear(whisper_dim, fusion_dim)
                self.mfcc_proj = nn.Linear(mfcc_dim, fusion_dim)
                self.lstm = nn.LSTM(
                    input_size=fusion_dim,
                    hidden_size=hidden_dim,
                    num_layers=layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim * 2, 96),
                    nn.ReLU(),
                    nn.Linear(96, 4),
                )

            def forward(self, whisper, mfcc):
                fused = self.whisper_proj(whisper) + self.mfcc_proj(mfcc)
                encoded, _state = self.lstm(fused)
                logits = self.head(encoded)
                return {
                    "speech": logits[..., 0],
                    "start": logits[..., 1],
                    "end": logits[..., 2],
                    "cut": logits[..., 3],
                }

        return _AdditionFusionEndpointBiLSTM()


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
