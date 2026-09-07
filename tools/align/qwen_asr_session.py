#!/usr/bin/env python3
"""The three model calls every alignment tool makes, in one place.

`build_real_alignment_lines.py` and `measure_pregate_dropped_audio.py` carried
byte-identical `_move`/`transcribe` bodies and near-identical feature helpers.
They are the same operation - put one clip through the same Qwen3-ASR
processor+model on the same device - and the copies had already started to
drift in their comments.

Offline tooling only. Nothing under `src/` imports this, and it must stay that
way: it loads a research checkpoint eagerly and picks a device for a batch job,
neither of which the shipped pipeline may do on import. What it deliberately
does *not* absorb is which windows a tool measures, which checkpoint it scores,
or what it writes out - the experiment inputs and the schema tags stay in each
tool, because that is what makes one run comparable with another.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QwenAsrSession:
    """A loaded processor+model pair, plus the device they were placed on.

    `model_spec` is exposed so a tool can record what it actually ran against;
    a measurement without that is not reproducible.
    """

    processor: Any
    model: Any
    device: Any
    dtype: Any
    model_spec: str

    @classmethod
    def load(cls, *, download: bool = True, vram_cap: float = 0.95) -> "QwenAsrSession":
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        from asr.backends.qwen import active_qwen_asr_model_id, active_qwen_asr_model_path
        from utils.gpu_safety import apply_vram_safety_cap
        from utils.model_paths import resolve_model_spec

        apply_vram_safety_cap(vram_cap)
        model_spec = resolve_model_spec(
            active_qwen_asr_model_path() or None,
            active_qwen_asr_model_id(),
            download=download,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_spec)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_spec, dtype=dtype, device_map=str(device)
        )
        model.eval()
        return cls(
            processor=processor,
            model=model,
            device=device,
            dtype=dtype,
            model_spec=str(model_spec),
        )

    def move(self, clip: np.ndarray) -> dict:
        """One clip, prepared and placed on this session's device."""
        import torch

        inputs = self.processor.apply_transcription_request(audio=[clip], language=None)
        return {
            key: (
                value.to(device=self.device, dtype=self.dtype)
                if key == "input_features"
                else value.to(device=self.device)
            )
            if torch.is_tensor(value)
            else value
            for key, value in inputs.items()
        }

    def frame_features(self, clip: np.ndarray) -> np.ndarray:
        """Encoder frames for one clip, trimmed to its real length."""
        import torch

        from asr.encoder_features import qwen3_asr_audio_output_lengths

        moved = self.move(clip)
        with torch.inference_mode():
            features = self.model.get_audio_features(
                input_features=moved["input_features"],
                input_features_mask=moved["input_features_mask"],
            ).pooler_output
        frames = int(
            qwen3_asr_audio_output_lengths(moved["input_features_mask"].sum(dim=1))[0]
        )
        return features[:frames].detach().float().cpu().numpy()

    def transcribe(self, clip: np.ndarray, *, max_new_tokens: int) -> str:
        """The decode, parsed.

        Without `parse_output` the decode still carries the prompt template
        ("language Japanese<asr_text>"), ~25 characters that were never spoken;
        the aligner would place them in the audio and both the score and every
        timestamp after them would be wrong.
        """
        import torch

        moved = self.move(clip)
        with torch.inference_mode():
            generated = self.model.generate(
                **moved, max_new_tokens=max_new_tokens, do_sample=False
            )
        suffix = generated[:, moved["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(
            suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        parsed = self.processor.parse_output(decoded)
        if isinstance(parsed, dict):
            parsed = [parsed]
        return str(parsed[0].get("transcription") or "")
