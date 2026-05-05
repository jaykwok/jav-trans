import math


def load_audio_16k_mono(audio_path: str):
    import numpy as np
    import soundfile as sf
    from scipy import signal

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000:
        divisor = math.gcd(int(sample_rate), 16000)
        audio = signal.resample_poly(
            audio,
            16000 // divisor,
            int(sample_rate) // divisor,
        ).astype("float32", copy=False)
    return np.ascontiguousarray(audio, dtype="float32"), 16000
