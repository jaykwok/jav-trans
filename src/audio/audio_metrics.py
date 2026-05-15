from __future__ import annotations

import logging
import math

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def compute_rms_dbfs(audio_path: str, start_s: float, end_s: float) -> float:
    """Return the RMS energy of [start_s, end_s] in dBFS.

    Returns 0.0 on any error so the caller never drops the span on exception
    (0.0 dBFS is louder than any negative threshold).
    """
    try:
        info = sf.info(audio_path)
        sr = info.samplerate
        start_frame = max(0, int(start_s * sr))
        n_frames = max(1, int((end_s - start_s) * sr))
        data, _ = sf.read(
            audio_path,
            start=start_frame,
            frames=n_frames,
            dtype="float32",
            always_2d=False,
        )
        if data.ndim > 1:
            data = data.mean(axis=1)
        samples = data.astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return 20.0 * math.log10(max(rms, 1e-12))
    except Exception as exc:
        logger.debug("compute_rms_dbfs: %s [%.3f-%.3f] error: %s", audio_path, start_s, end_s, exc)
        return 0.0
