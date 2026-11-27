"""Shared Qwen3-ASR model loading and batched audio-tower feature extraction.

Unifies the model-loading and encoder forward pass that previously lived in
three separate copies across ``asr.pipeline``, ``asr.local_backend``, and
``asr.encoder_features``.

Key guarantees:
  * Respects explicit ``device``, ``ASR_DTYPE``, and ``ASR_ATTENTION``.
  * Performs a single ``.detach().cpu().tolist()`` transfer for frame lengths
    instead of per-item GPU syncs.
  * Returns encoder ``pooler_output`` on-device alongside ``lengths`` so
    callers (chunking, CTC forced alignment) can feed the alignment head on
    GPU without a GPU->CPU->GPU roundtrip.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np


def detect_asr_dtype(device: str):
    """Resolve torch dtype for Qwen3-ASR from ``ASR_DTYPE`` and ``device``."""
    import torch

    dtype = os.getenv("ASR_DTYPE", "auto").strip().lower()
    if dtype in {"float32", "fp32"}:
        return torch.float32
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if str(device).startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def detect_asr_attention(device: str) -> str:
    """Resolve attention backend from ``ASR_ATTENTION`` and ``device``."""
    attention = os.getenv("ASR_ATTENTION", "auto").strip().lower()
    if attention and attention != "auto":
        return attention
    if not str(device).startswith("cuda"):
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_qwen_asr_model(
    *,
    model_spec: str,
    device: Any,
    dtype: Any,
    attention: str = "sdpa",
) -> tuple[Any, Any]:
    """Load Qwen3-ASR ``(model, processor)`` in eval mode on ``device``."""
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": str(device),
    }
    if attention and attention != "sdpa":
        model_kwargs["attn_implementation"] = attention

    processor = AutoProcessor.from_pretrained(model_spec)
    model = AutoModelForMultimodalLM.from_pretrained(model_spec, **model_kwargs)
    model.eval()
    return model, processor


def encode_audio_features_batch(
    model: Any,
    processor: Any,
    audios: Sequence[np.ndarray],
    *,
    device: Any,
    dtype: Any,
    language: str | None = None,
    stage_label: str = "encoding",
) -> tuple[Any, list[int]]:
    """Run the Qwen3-ASR audio tower on a batch of 16 kHz mono clips.

    Returns ``(pooler_output, lengths)`` where ``pooler_output`` remains on the
    model's device with shape ``(sum(lengths), D)`` and ``lengths`` holds each
    clip's frame count obtained via a single host transfer.
    """
    import torch

    from asr import encoder_features, qwen_native

    audio_list = [np.asarray(audio, dtype=np.float32) for audio in audios]
    if not audio_list:
        return None, []

    inputs = qwen_native.prepare_transcription_inputs(
        processor,
        audio=audio_list,
        language=language or None,
    )
    moved = qwen_native.move_processor_inputs(
        inputs,
        device=device,
        dtype=dtype,
    )
    with torch.inference_mode():
        features = model.get_audio_features(
            input_features=moved["input_features"],
            input_features_mask=moved["input_features_mask"],
        ).pooler_output

    output_lengths_raw = encoder_features.qwen3_asr_audio_output_lengths(
        moved["input_features_mask"].sum(dim=1)
    )
    if torch.is_tensor(output_lengths_raw):
        lengths = [int(value) for value in output_lengths_raw.detach().cpu().tolist()]
    else:
        lengths = [int(value) for value in output_lengths_raw]

    expected_frames = sum(lengths)
    actual_frames = int(features.shape[0])
    if expected_frames != actual_frames:
        raise RuntimeError(
            f"ASR encoder output length mismatch during {stage_label}: "
            f"expected_frames={expected_frames} actual_frames={actual_frames}"
        )
    return features, lengths


def split_encoded_features_numpy(
    features: Any,
    lengths: Sequence[int],
) -> list[np.ndarray]:
    """Transfer concatenated encoder ``pooler_output`` to per-clip float32 numpy arrays."""
    if not lengths:
        return []
    hidden = features.detach().float().cpu().numpy()
    expected_frames = sum(int(length) for length in lengths)
    if expected_frames != int(hidden.shape[0]):
        raise RuntimeError(
            "Qwen3-ASR encoder output frame count mismatch: "
            f"expected={expected_frames} actual={int(hidden.shape[0])}"
        )
    result: list[np.ndarray] = []
    offset = 0
    for length in lengths:
        length_int = int(length)
        result.append(np.asarray(hidden[offset : offset + length_int], dtype=np.float32))
        offset += length_int
    return result
