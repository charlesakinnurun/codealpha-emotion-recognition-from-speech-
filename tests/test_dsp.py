"""Unit tests for the NumPy/SciPy DSP features (no dataset required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emotion_recognition.features.audio import load_mono
from emotion_recognition.features.dsp import (
    delta,
    hz_to_mel,
    log_mel_spectrogram,
    mel_filterbank,
    mel_to_hz,
    mfcc,
)
from tests.helpers import write_tone


def test_mel_scale_round_trip() -> None:
    freqs = np.array([0.0, 100.0, 440.0, 1000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(freqs)), freqs)


def test_mel_filterbank_shape_and_nonnegativity() -> None:
    fb = mel_filterbank(sr=16000, n_mels=64, n_fft=1024)
    assert fb.shape == (64, 513)
    assert (fb >= 0).all()
    assert (fb.sum(axis=1) > 0).all()  # every band has energy


def test_log_mel_shape_and_reference_level() -> None:
    sr = 16000
    y = np.sin(2.0 * np.pi * 440.0 * np.arange(sr) / sr).astype(np.float32)
    log_mel = log_mel_spectrogram(y, sr=sr, n_fft=1024, hop_length=256, n_mels=64, fmax=8000.0)
    assert log_mel.shape[0] == 64
    assert log_mel.shape[1] > 50
    # Reference-normalized: max bin should be ~0 dB.
    assert np.isclose(log_mel.max(), 0.0, atol=1e-6)


def test_low_tone_energy_peaks_in_low_mel_band() -> None:
    sr = 16000
    low = np.sin(2.0 * np.pi * 200.0 * np.arange(sr) / sr).astype(np.float32)
    log_mel = log_mel_spectrogram(low, sr=sr, n_mels=64, fmax=8000.0)
    peak_band = int(np.argmax(log_mel.mean(axis=1)))
    assert peak_band < 16  # 200 Hz should land in the lower quarter


def test_mfcc_shape() -> None:
    sr = 16000
    y = np.sin(2.0 * np.pi * 440.0 * np.arange(sr // 2) / sr).astype(np.float32)
    coeffs = mfcc(y, sr=sr, n_mfcc=13)
    assert coeffs.shape[0] == 13


def test_delta_requires_odd_width() -> None:
    with pytest.raises(ValueError, match="odd"):
        delta(np.zeros((5, 10)), width=8)


def test_delta_of_constant_is_zero() -> None:
    feats = np.full((13, 40), 3.7)
    assert np.allclose(delta(feats), 0.0, atol=1e-9)


def test_load_mono_resamples_and_mixes(tmp_path: Path) -> None:
    # 48 kHz stereo tone (matches RAVDESS format).
    sr = 48000
    stereo = np.stack(
        [
            0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(sr) / sr),
            0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(sr) / sr),
        ],
        axis=1,
    ).astype(np.float32)
    path = tmp_path / "tone.wav"
    import soundfile as sf

    sf.write(str(path), stereo, sr)

    wave, out_sr = load_mono(path, target_sr=16000)
    assert out_sr == 16000
    assert wave.ndim == 1  # mono
    assert abs(wave.size - 16000) <= 2  # ~1 second
    assert wave.dtype == np.float32


def test_write_tone_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "x.wav"
    write_tone(path, freq=440.0, duration=0.6)
    y, sr = load_mono(path, target_sr=22050)
    assert y.ndim == 1 and sr == 22050
    # A sine tone should have non-trivial energy.
    assert np.sqrt(np.mean(y**2)) > 0.1
