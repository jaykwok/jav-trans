"""Decode Hugging Face audio samples to the 16 kHz mono float32 the ASR wants.

`datasets` hands audio back in several shapes depending on the loader and the
column type, and the SFT dataset builder has to survive all of them.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


def normalize_audio_16k_mono(
    audio: np.ndarray | Sequence[float],
    sample_rate: int,
) -> tuple[np.ndarray, int]:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1, dtype=np.float32)
    samples = np.asarray(samples, dtype=np.float32)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if int(sample_rate) != 16000:
        from scipy import signal

        divisor = math.gcd(int(sample_rate), 16000)
        samples = signal.resample_poly(
            samples,
            16000 // divisor,
            int(sample_rate) // divisor,
        ).astype("float32", copy=False)
    return np.ascontiguousarray(samples, dtype=np.float32), 16000


def decode_audio_bytes_16k_mono(audio_bytes: bytes | bytearray) -> tuple[np.ndarray, int]:
    import io

    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    return normalize_audio_16k_mono(data, int(sample_rate))


def stable_hf_audio_id(*, dataset_name: str, split: str, index: int) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name).strip("_")
    split_key = re.sub(r"[^A-Za-z0-9._-]+", "_", split).strip("_")
    return f"{prefix}-{split_key}-{index:06d}"


def sample_hf_audio_16k_mono(example: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    audio_obj = example.get("ogg") or example.get("audio")
    if isinstance(audio_obj, (bytes, bytearray)):
        return decode_audio_bytes_16k_mono(audio_obj)
    if isinstance(audio_obj, Mapping):
        audio_bytes = audio_obj.get("bytes")
        if isinstance(audio_bytes, (bytes, bytearray)):
            return decode_audio_bytes_16k_mono(audio_bytes)
        array = audio_obj.get("array")
        sample_rate = int(audio_obj.get("sampling_rate") or 16000)
        if array is None:
            raise ValueError("audio sample has no bytes or array")
        return normalize_audio_16k_mono(np.asarray(array, dtype=np.float32), sample_rate)
    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        data = getattr(samples, "data")
        sample_rate = int(getattr(samples, "sample_rate"))
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        samples_array = np.asarray(data, dtype=np.float32)
        if samples_array.ndim == 2 and samples_array.shape[0] <= 8:
            samples_array = samples_array.mean(axis=0, dtype=np.float32)
        return normalize_audio_16k_mono(samples_array, sample_rate)
    raise ValueError("expected an 'ogg' or 'audio' field decoded by datasets")
