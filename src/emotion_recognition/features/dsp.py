"""Spectral feature extraction implemented with pure NumPy (+ SciPy.fft for DCT).

Why no ``scipy.signal`` / ``librosa``?

- ``librosa`` bundles ``numba`` and ``scipy.signal`` bundles several compiled
  Fortran extensions (``_dierckx``, ``_decomp_interpolative``) that OS-level
  application-control policies may block on some Windows hosts, making the
  whole dependency chain unusable there.
- A small, deterministic STFT/mel/resampler implemented on top of NumPy is
  all speech feature extraction for this project needs, and it runs anywhere.

The only SciPy import is ``scipy.fft`` (DCT for MFCCs), which is NumPy-backed
and loads cleanly on the restricted hosts.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dct

EPS = 1e-10


def _hann_periodic(n: int) -> np.ndarray:
    """Periodic Hann window (matches the common 'symmetric=False' convention)."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


def _design_lowpass(cutoff: float, numtaps: int, gain: float) -> np.ndarray:
    """Windowed-sinc low-pass FIR with a Kaiser window, numpy-only."""
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    h = gain * 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    h *= np.kaiser(numtaps, 8.6)
    return h / np.sum(h) * gain


def resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Resample a 1-D signal by the rational factor ``up / down``.

    Windowed-sinc polyphase resampler: factor out the ``up`` zero-insertion
    subfilters so each output sample costs one ``numtaps`` dot product instead
    of a full convolution on the zero-stuffed signal. The tapped coefficients
    are *exactly* the ones the naive filter + decimate approach would use, so
    this is a speed optimization, not a behavioral change.

    Raises:
        ValueError: if inputs are not positive integers or ``x`` is not 1-D.
    """
    if x.ndim != 1:
        raise ValueError("resample expects a 1-D signal.")
    if up <= 0 or down <= 0:
        raise ValueError(f"up/down must be positive integers, got {up}/{down}.")
    if up == down:
        return x.copy()

    cutoff = 1.0 / max(up, down)
    numtaps = min(512, max(32, 4 * max(up, down)))
    h = _design_lowpass(cutoff, numtaps, gain=up)

    # Sample n of the naive filter+decimate output reads:
    #   out[n] = sum_i x[i] * h[delay + n*down - i*up]
    # With phase = (delay + n*down) % up and base = (delay + n*down) // up
    # this equals sum_m x[base - m] * h[phase + m*up], i.e. one decimated
    # subfilter per phase, evaluated over a short window of x.
    delay = (numtaps - 1) // 2
    n_out = int(round(x.size * up / down))
    pos = delay + np.arange(n_out) * down
    base = pos // up
    phase = pos % up

    R = (numtaps - 1) // up + 1
    taps = np.arange(2 * R + 1)
    # Zero padding matches 'full' convolution edges. base spans [-0, N + 1]
    # samples, so reserve R zeros on the left and 2R+2 on the right.
    xp = np.pad(x, (R, 2 * R + 2))

    # Window of x around each output: xp[base + 2R - taps]
    win = xp[(base[:, None] + 2 * R - taps[None, :])]
    # Subfilter for each output: h[phase + (taps - R) * up], zero where undefined.
    hpad = np.concatenate([np.zeros(numtaps), h, np.zeros(numtaps + 3 * up)])
    sub = hpad[numtaps + (phase[:, None] - R * up + taps[None, :] * up)]

    y = np.einsum("rm,rm->r", win, sub)
    return y.astype(x.dtype) if np.issubdtype(x.dtype, np.floating) else y


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency in Hz to the perceptual mel scale (HTK formula)."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mels: np.ndarray | float) -> np.ndarray | float:
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(mels) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int, n_mels: int = 64, n_fft: int = 1024, fmin: float = 0.0, fmax: float = 8000.0
) -> np.ndarray:
    """Build a triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``.

    Each filter is a triangle ramping up to its center frequency and down to
    the next; bands are normalized by total energy so loudness is comparable.
    Results are cached per parameter set (the filterbank is config-invariant).
    """
    return _mel_filterbank_cached(int(sr), int(n_mels), int(n_fft), float(fmin), float(fmax))


@lru_cache(maxsize=16)
def _mel_filterbank_cached(
    sr: int, n_mels: int, n_fft: int, fmin: float, fmax: float
) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(float(hz_to_mel(fmin)), float(hz_to_mel(fmax)), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.clip(np.floor((n_fft + 1) * hz_points / float(sr)).astype(int), 0, n_freqs - 1)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if right <= left:
            continue
        filters[m - 1, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        filters[m - 1, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)

    areas = filters.sum(axis=1, keepdims=True)
    areas[areas == 0] = 1.0
    return filters / areas


def stft(y: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Framed short-time Fourier transform, numpy-only.

    Returns the complex spectrogram of shape ``(n_fft // 2 + 1, n_frames)``.
    The signal is edge-padded so every frame has full length.
    """
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive.")
    if y.size < n_fft:
        y = np.pad(np.asarray(y, dtype=float), (0, n_fft - y.size))

    window = _hann_periodic(n_fft)
    pad = n_fft // 2
    y = np.pad(y, (pad, pad))
    n_frames = 1 + (y.size - n_fft) // hop_length

    frames = np.empty((n_fft, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        frames[:, i] = y[start : start + n_fft] * window
    return np.fft.rfft(frames, axis=0)


def frame_signal(y: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Framed (windowed) time-domain signal, aligned with :func:`stft`.

    Returns a ``(n_fft, n_frames)`` matrix using the exact same padding and
    sample positions as the STFT, so time-domain and spectral features are
    frame-aligned by construction.
    """
    y = np.asarray(y, dtype=float)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    y = np.pad(y, (n_fft // 2, n_fft // 2))
    n_frames = 1 + (y.size - n_fft) // hop_length
    indices = np.arange(n_fft)[:, None] + hop_length * np.arange(n_frames)[None, :]
    return y[indices]


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """ZCR per frame: mean number of sign flips, shape ``(1, n_frames)``."""
    signs = np.signbit(frames)
    flips = np.abs(np.diff(signs.astype(np.int8), axis=0)).mean(axis=0)
    return flips[None, :]


def rms_energy(frames: np.ndarray) -> np.ndarray:
    """RMS energy per frame, shape ``(1, n_frames)``."""
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=0))[None, :]


def spectral_centroid(spec_mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Weighted mean frequency per frame, shape ``(1, n_frames)``."""
    total = spec_mag.sum(axis=0, keepdims=True)
    total[total == 0] = 1.0
    return (freqs[:, None] * spec_mag).sum(axis=0, keepdims=True) / total


def spectral_bandwidth(spec_mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Spectral spread around the centroid per frame, shape ``(1, n_frames)``."""
    centroid = spectral_centroid(spec_mag, freqs)
    total = spec_mag.sum(axis=0, keepdims=True)
    total[total == 0] = 1.0
    diff = (freqs[:, None] - centroid) ** 2
    spread = np.sqrt((diff * spec_mag).sum(axis=0, keepdims=True) / total)
    return spread


def spectral_rolloff(spec_mag: np.ndarray, freqs: np.ndarray, percent: float = 0.85) -> np.ndarray:
    """Frequency below which ``percent`` of the energy lies, per frame."""
    if not 0.0 < percent < 1.0:
        raise ValueError("percent must be in (0, 1).")
    total = spec_mag.sum(axis=0, keepdims=True)
    total[total == 0] = 1.0
    cum = np.cumsum(spec_mag, axis=0) / total
    # First frequency where the cumulative sum exceeds the target.
    rolloff_idx = np.argmax(cum >= percent, axis=0)
    return freqs[rolloff_idx][None, :]


def chroma(spec_mag: np.ndarray, sr: int, n_fft: int = 1024, n_chroma: int = 12) -> np.ndarray:
    """Pitch-class energy per frame, shape ``(n_chroma, n_frames)``.

    Each FFT bin is mapped to a semitone class via ``log2(f / 440)``; the
    magnitude of every bin is accumulated into its class.
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
    valid = freqs > 0.0
    classes = (np.round(12.0 * np.log2(freqs[valid] / 440.0)) % n_chroma).astype(int)

    weights = np.zeros((n_chroma, spec_mag.shape[0]), dtype=float)
    for k in range(n_chroma):
        weights[k, valid] = classes == k
    return weights @ spec_mag


def f0_autocorrelation(
    frames: np.ndarray,
    sr: int,
    fmin: float = 75.0,
    fmax: float = 400.0,
    rms_threshold: float = 0.01,
) -> np.ndarray:
    """Fundamental frequency per frame by normalized autocorrelation.

    Returns a ``(1, n_frames)`` array in Hz. Frames that are silent or lack a
    clear harmonic peak are marked ``0`` (unvoiced).
    """
    n_fft, n_frames = frames.shape
    lag_min = max(1, int(sr / fmax))
    lag_max = int(sr / fmin)
    if lag_max >= n_fft:
        raise ValueError("n_fft too small for the requested f0 range.")

    window = _hann_periodic(n_fft)
    f0 = np.zeros(n_frames)

    for i in range(n_frames):
        frame = frames[:, i] * window
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < rms_threshold:
            continue
        autocorr = np.correlate(frame, frame, mode="full")[n_fft - 1 : n_fft + lag_max]
        autocorr[autocorr[0] == 0] = 1.0
        norm = autocorr / autocorr[0]

        peak_lag = lag_min + int(np.argmax(norm[lag_min:]))
        if norm[peak_lag] < 0.4:  # weak periodicity -> unvoiced
            continue

        # Parabolic interpolation for sub-sample refinement.
        lo = max(1, peak_lag - 1)
        hi = min(len(norm) - 1, peak_lag + 1)
        a, b, c = norm[lo], norm[peak_lag], norm[hi]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            shift = 0.5 * (a - c) / denom
            peak_lag += int(np.round(np.clip(shift, -0.5, 0.5)))
        if peak_lag > 0:
            f0[i] = sr / peak_lag
    return f0[None, :]


def mel_db_from_power(
    power_spec: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    fmax: float = 8000.0,
    ref: float | None = None,
) -> np.ndarray:
    """Convert a power spectrogram to log-mel (dB), shape ``(n_mels, n_frames)``."""
    filters = mel_filterbank(sr=sr, n_mels=n_mels, n_fft=n_fft, fmin=0.0, fmax=fmax)
    mel = filters @ power_spec
    mel = np.maximum(mel, EPS)
    reference = ref if ref is not None else float(mel.max())
    if reference <= 0:  # fully silent input -> floor so log is finite
        reference = EPS
    return 10.0 * np.log10(mel / reference)


def log_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
    fmax: float = 8000.0,
    power: float = 2.0,
    ref: float | None = None,
) -> np.ndarray:
    """Compute a log-mel spectrogram of shape ``(n_mels, n_frames)``."""
    spec = stft(y, n_fft=n_fft, hop_length=hop_length)
    power_spec = np.abs(spec) ** power
    return mel_db_from_power(power_spec, sr=sr, n_mels=n_mels, n_fft=n_fft, fmax=fmax, ref=ref)


def mfcc_from_power(
    power_spec: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    n_mels: int = 40,
    fmax: float = 8000.0,
) -> np.ndarray:
    """MFCCs from a precomputed power spectrogram, shape ``(n_mfcc, n_frames)``.

    Lets the caller compute the STFT once and reuse it for both log-Mel and
    MFCC features instead of paying for two FFTs.
    """
    mel = mel_db_from_power(power_spec, sr=sr, n_mels=n_mels, n_fft=n_fft, fmax=fmax)
    return dct(mel, type=2, axis=0, norm="ortho")[:n_mfcc]


def mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 40,
    fmax: float = 8000.0,
) -> np.ndarray:
    """MFCC coefficients of shape ``(n_mfcc, n_frames)`` via DCT of log-mel."""
    power_spec = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2.0
    return mfcc_from_power(power_spec, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, n_mels=n_mels, fmax=fmax)


def delta(features: np.ndarray, width: int = 9) -> np.ndarray:
    """Compute local deltas of a feature matrix ``(features, frames)``.

    For MFCCs, delta captures the first temporal derivative (dynamics).
    """
    if width < 3 or width % 2 == 0:
        raise ValueError("width must be an odd integer >= 3")
    pad = width // 2
    n = features.shape[1]
    padded = np.pad(features, ((0, 0), (pad, pad)), mode="edge")

    weights = np.arange(1, pad + 1)
    denom = 2.0 * float(np.sum(weights**2))
    num = np.zeros_like(padded, dtype=float)
    for offset in weights:
        num[:, pad : pad + n] += offset * (
            padded[:, pad + offset : pad + offset + n] - padded[:, pad - offset : pad - offset + n]
        )
    return num[:, pad : pad + n] / denom


__all__ = [
    "EPS",
    "resample",
    "hz_to_mel",
    "mel_to_hz",
    "mel_filterbank",
    "stft",
    "frame_signal",
    "zero_crossing_rate",
    "rms_energy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "chroma",
    "f0_autocorrelation",
    "mel_db_from_power",
    "log_mel_spectrogram",
    "mfcc",
    "mfcc_from_power",
    "delta",
]
