"""Single source of truth for loading speech audio.

Every consumer (dataset, EDA, training, inference, tests) loads audio through
this module so the training and serving representations can never drift apart.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from .dsp import resample

TARGET_SAMPLE_RATE = 16000


def load_mono(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio, mix to mono, and resample to ``target_sr``.

    Returns:
        ``(waveform, sample_rate)`` where ``waveform`` is ``float32`` in [-1, 1].
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = y.mean(axis=1)  # stereo -> mono (RAVDESS speech is stereo)
    if sr != target_sr:
        divisor = gcd(int(target_sr), int(sr))
        mono = resample(mono, int(target_sr) // divisor, int(sr) // divisor)
    return mono.astype(np.float32), int(target_sr)


__all__ = ["TARGET_SAMPLE_RATE", "load_mono"]
