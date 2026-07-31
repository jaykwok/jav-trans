"""Encoder-only inference against the Qwen3-ASR audio tower.

The CTC alignment head reads this encoder's hidden states, and the chunker
reads the head's blank posteriors, so both the runtime and the offline
alignment tooling need one forward pass without any decoding. That is the whole
job of this module.

Frame rate is a contract, not a tuning knob: `qwen3_asr_audio_output_lengths`
mirrors the model's own downsampling, giving 13 encoder frames per second of
16 kHz audio (76.9 ms/frame). `src/asr/alignment.py` upsamples x2 on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asr.backends.qwen import QWEN_ASR_REPO_ID
from utils.model_paths import resolve_model_spec


@dataclass(frozen=True)
class EncoderFeatureConfig:
    repo_id: str = QWEN_ASR_REPO_ID
    device: str = "cuda"
    dtype: str = "float16"
    revision: str | None = None
    model_path: str = ""
    download: bool = True
    attention: str = "sdpa"
    language: str = "Japanese"


def torch_dtype_from_name(dtype: str):
    import torch

    normalized = (dtype or "").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


def qwen3_asr_audio_output_lengths(input_lengths: Any) -> Any:
    """Encoder frames produced for a given number of mel frames.

    Every 100 mel frames (1 second) becomes 13 encoder frames. Works on ints
    and on torch tensors, because callers need it in both places.
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class Qwen3AsrEncoder:
    """Loads the ASR model and exposes `get_audio_features` only.

    Decoding is ~178x the cost of this forward pass, which is what makes it
    affordable to encode every second of a file before deciding anything.
    """

    def __init__(self, config: EncoderFeatureConfig | None = None) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.config = config or EncoderFeatureConfig()
        self.repo_id = self.config.repo_id.strip()
        self.model_path = resolve_model_spec(
            self.config.model_path or None,
            self.repo_id,
            download=self.config.download,
            revision=self.config.revision,
        )
        if not self.config.download and self.model_path == self.repo_id:
            raise FileNotFoundError(
                f"{self.repo_id} is not available locally. Put the model under "
                f"models/{self.repo_id.replace('/', '-')} or pass --model-path, "
                "or rerun without --no-download."
            )
        self.torch_dtype = torch_dtype_from_name(self.config.dtype)
        want_cuda = self.config.device != "cpu"
        self.device = torch.device(
            self.config.device if not want_cuda or torch.cuda.is_available() else "cpu"
        )
        model_kwargs: dict[str, Any] = {
            "dtype": self.torch_dtype,
            "device_map": str(self.device),
        }
        if self.config.attention and self.config.attention != "sdpa":
            model_kwargs["attn_implementation"] = self.config.attention
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForMultimodalLM.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()

    def encode(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        return self.encode_batch([audio], sample_rate=sample_rate)[0]

    def encode_batch(
        self,
        audios: list[np.ndarray],
        *,
        sample_rate: int,
    ) -> list[np.ndarray]:
        import torch

        if not audios:
            return []
        if sample_rate != 16000:
            raise ValueError(
                f"Qwen3-ASR encoder expects 16kHz audio, got {sample_rate}"
            )
        inputs = self.processor.apply_transcription_request(
            audio=[np.asarray(audio, dtype=np.float32) for audio in audios],
            language=[self.config.language] * len(audios) if self.config.language else None,
        )
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                moved[key] = value
            elif key == "input_features":
                moved[key] = value.to(device=self.device, dtype=self.torch_dtype)
            else:
                moved[key] = value.to(device=self.device)

        input_lengths = moved["input_features_mask"].sum(dim=1)
        output_lengths = qwen3_asr_audio_output_lengths(input_lengths).detach().cpu().tolist()
        with torch.inference_mode():
            audio_features = self.model.get_audio_features(
                input_features=moved["input_features"],
                input_features_mask=moved["input_features_mask"],
            ).pooler_output
            hidden = audio_features.detach().float().cpu().numpy()

        result: list[np.ndarray] = []
        offset = 0
        for length in output_lengths:
            length_int = int(length)
            result.append(np.asarray(hidden[offset : offset + length_int], dtype=np.float32))
            offset += length_int
        return result

    def close(self) -> None:
        import torch

        self.model = None
        self.processor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
