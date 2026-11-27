import math


def load_audio_16k_mono(audio_path: str):
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"Invalid audio sample rate ({sample_rate}) in {audio_path}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")
    if audio.ndim == 2:
        # soundfile.read always returns shape (frames, channels) for multi-channel audio.
        audio = np.mean(audio, axis=1, dtype=np.float32)
    elif audio.ndim != 1:
        raise ValueError(f"Unsupported audio array shape {audio.shape} in {audio_path}")
    if sample_rate != 16000:
        from scipy import signal

        divisor = math.gcd(sample_rate, 16000)
        audio = signal.resample_poly(
            audio,
            16000 // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(audio, dtype=np.float32), 16000
