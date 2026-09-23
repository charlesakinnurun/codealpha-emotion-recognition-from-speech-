"""Feature extraction: the single interface between audio and every model.

Training, inference, and tests all call :class:`FeatureExtractor`, guaranteeing
training/serving parity by construction. The extractor produces a
:class:`FeatureBundle`; deep models consume the log-mel or MFCC matrices, and
classical baselines consume the statistics-pooled vector built by
:func:`pool_utterance`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .audio import load_mono
from .dsp import (
    chroma,
    delta,
    f0_autocorrelation,
    frame_signal,
    mel_db_from_power,
    resample,
    rms_energy,
    spectral_bandwidth,
    spectral_centroid,
    spectral_rolloff,
    stft,
    zero_crossing_rate,
)
from .dsp import (
    mfcc_from_power as _mfcc_from_power,
)

DEFAULT_GROUPS = ("mfcc", "delta", "delta2", "spectral", "prosodic")

# Row counts of the non-MFCC bundle matrices. Kept as module constants so the
# classical-baseline feature layout (column slices per group) never drifts
# from what the extractor actually produces.
SPECTRAL_ROWS = 5 + 12  # zcr, rms, centroid, bandwidth, rolloff + chroma(12)
PITCH_ROWS = 1
PROSODIC_SIZE = 8  # 4 f0 stats + voiced ratio + 3 energy stats


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters for :class:`FeatureExtractor`.

    Keeping every DSP choice here (rather than scattered literals) is what
    makes experiments reproducible and the training/serving feature function
    a single, versionable artifact.
    """

    target_sr: int = 16000
    duration_s: float = 4.0
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    fmax: float = 8000.0
    n_mfcc: int = 40
    mfcc_n_mels: int = 40
    delta_width: int = 9
    f0_min: float = 75.0
    f0_max: float = 400.0
    f0_rms_threshold: float = 0.01
    rolloff_percent: float = 0.85
    # Normalizing RMS to a fixed target removes loudness, which itself is a
    # prosodic affect cue (angry is loud). Default False preserves loudness.
    normalize_rms: bool = False
    pool_stats: tuple[str, ...] = ("mean", "std")
    pool_groups: tuple[str, ...] = DEFAULT_GROUPS

    def n_samples(self) -> int:
        """Length of the fixed feature window in samples."""
        return int(self.target_sr * self.duration_s)

    def n_frames(self) -> int:
        """Number of frames the fixed window produces with the configured STFT."""
        return 1 + self.n_samples() // self.hop_length

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FeatureConfig:
        """Instantiate from a flat dict (e.g. a YAML section), ignoring unknown keys."""
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in allowed})


@dataclass
class FeatureBundle:
    """All feature views of one utterance, stacked along the frame axis.

    Matrices have shape ``(n_features, n_frames)``; ``prosodic`` and ``pitch``
    are provided separately since pitch is both a frame contour and an input
    to utterance-level prosodic statistics.
    """

    log_mel: np.ndarray
    mfcc: np.ndarray
    mfcc_delta: np.ndarray
    mfcc_delta2: np.ndarray
    spectral: np.ndarray
    pitch: np.ndarray
    prosodic: np.ndarray
    waveform: np.ndarray
    sample_rate: int

    @property
    def n_frames(self) -> int:
        return self.log_mel.shape[1]

    @property
    def mfcc_stack(self) -> np.ndarray:
        """MFCC + delta + delta-delta stacked, for sequence models."""
        return np.concatenate([self.mfcc, self.mfcc_delta, self.mfcc_delta2], axis=0)


def _build_prosodic_vector(
    pitch: np.ndarray, waveform: np.ndarray, cfg: FeatureConfig
) -> np.ndarray:
    """Utterance-level prosodic statistics (fixed length, independent of frames)."""
    voiced = pitch[pitch > 0]
    n_frames = pitch.size
    voiced_ratio = voiced.size / max(n_frames, 1)
    if voiced.size == 0:
        f0_stats = np.zeros((4,))
    else:
        f0_stats = np.array(
            [voiced.mean(), voiced.std(), np.median(voiced), (voiced.max() - voiced.min())]
        )
    amp = np.abs(waveform)
    rms_overall = float(np.sqrt(np.mean(waveform**2)))
    return np.concatenate(
        [f0_stats, [voiced_ratio, rms_overall, float(amp.mean()), float(amp.max())]]
    ).astype(np.float32)


class FeatureExtractor:
    """Deterministic audio -> features converter.

    Usage:
        extractor = FeatureExtractor(cfg)
        bundle = extractor.extract("/path/to/file.wav")   # files
        bundle = extractor.from_waveform(y, sr)           # in-memory waveforms
    """

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()

    # -- public interface ---------------------------------------------------
    def extract(self, audio_path: str | Path) -> FeatureBundle:
        """Load ``audio_path`` and extract features (training and serving path)."""
        y, sr = load_mono(audio_path, target_sr=self.cfg.target_sr)
        return self.from_waveform(y, sr)

    def from_waveform(self, y: np.ndarray, sr: int) -> FeatureBundle:
        """Extract features from an in-memory waveform, resampling to target_sr."""
        y = np.asarray(y, dtype=np.float32)
        if sr != self.cfg.target_sr:
            from math import gcd

            divisor = gcd(int(self.cfg.target_sr), int(sr))
            y = resample(y, int(self.cfg.target_sr) // divisor, int(sr) // divisor)
        y = self._fix_length(y)
        if self.cfg.normalize_rms:
            rms = float(np.sqrt(np.mean(y**2)))
            y = y / rms if rms > 1e-9 else y

        power = np.abs(stft(y, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length)) ** 2.0

        log_mel = mel_db_from_power(
            power,
            sr=self.cfg.target_sr,
            n_mels=self.cfg.n_mels,
            n_fft=self.cfg.n_fft,
            fmax=self.cfg.fmax,
            ref=float(power.max()),
        )
        mfcc = _mfcc_from_power(
            power,
            sr=sr,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft,
            n_mels=self.cfg.mfcc_n_mels,
            fmax=self.cfg.fmax,
        )
        d1 = delta(mfcc, width=self.cfg.delta_width)
        d2 = delta(d1, width=self.cfg.delta_width)

        frames = frame_signal(y, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length)
        freqs = np.fft.rfftfreq(self.cfg.n_fft, 1.0 / sr)

        spectral = self._spectral_stack(frames, power, freqs)
        pitch = f0_autocorrelation(
            frames,
            sr=sr,
            fmin=self.cfg.f0_min,
            fmax=self.cfg.f0_max,
            rms_threshold=self.cfg.f0_rms_threshold,
        )
        prosodic = _build_prosodic_vector(pitch[0], y, self.cfg)

        return FeatureBundle(
            log_mel=log_mel.astype(np.float32),
            mfcc=mfcc.astype(np.float32),
            mfcc_delta=d1.astype(np.float32),
            mfcc_delta2=d2.astype(np.float32),
            spectral=spectral.astype(np.float32),
            pitch=pitch.astype(np.float32),
            prosodic=prosodic,
            waveform=y,
            sample_rate=self.cfg.target_sr,
        )

    def pool_utterance(self, bundle: FeatureBundle) -> np.ndarray:
        """Statistics-pooled feature vector for a single utterance (classical ML).

        Applies ``pool_stats`` (default mean/std) per row of every group in
        ``pool_groups`` and concatenates them into one fixed-length vector.
        """
        return pool_utterance(bundle, self.cfg)

    # -- internals ------------------------------------------------------------
    def _fix_length(self, y: np.ndarray) -> np.ndarray:
        """Pad (right) or crop to the configured fixed window."""
        length = self.cfg.n_samples()
        if y.size < length:
            return np.pad(y, (0, length - y.size))
        return y[:length]

    def _spectral_stack(
        self, frames: np.ndarray, power: np.ndarray, freqs: np.ndarray
    ) -> np.ndarray:
        mag = np.sqrt(power)
        rows = [
            zero_crossing_rate(frames),
            rms_energy(frames),
            spectral_centroid(mag, freqs),
            spectral_bandwidth(mag, freqs),
            spectral_rolloff(mag, freqs, percent=self.cfg.rolloff_percent),
            chroma(mag, sr=self.cfg.target_sr, n_fft=self.cfg.n_fft),
        ]
        return np.concatenate(rows, axis=0)


def extract_features(audio_path: str | Path, cfg: FeatureConfig | None = None) -> FeatureBundle:
    """Convenience wrapper mirroring the project's documented interface.

    ``extract_features(path) -> FeatureBundle``
    """
    return FeatureExtractor(cfg).extract(audio_path)


# ---------------------------------------------------------------------------
# Statistics pooling (classical baselines consume this).
# ---------------------------------------------------------------------------
_GROUP_TO_ATTR = {
    "mfcc": "mfcc",
    "delta": "mfcc_delta",
    "delta2": "mfcc_delta2",
    "spectral": "spectral",
}


def _apply_pool_stats(matrix: np.ndarray, stats: tuple[str, ...]) -> np.ndarray:
    out: list[np.ndarray] = []
    for stat in stats:
        if stat == "mean":
            out.append(matrix.mean(axis=1))
        elif stat == "std":
            out.append(matrix.std(axis=1))
        elif stat == "median":
            out.append(np.median(matrix, axis=1))
        elif stat.startswith("q"):
            quantile = float(stat[1:]) / 100.0
            out.append(np.quantile(matrix, quantile, axis=1))
        else:
            raise ValueError(f"Unsupported pool stat: {stat!r}")
    return np.concatenate(out)


def pool_matrix(matrix: np.ndarray, stats: tuple[str, ...]) -> np.ndarray:
    """Pool a frame matrix ``(n_features, n_frames)`` into a fixed vector."""
    return _apply_pool_stats(matrix, stats)


def pool_utterance(bundle: FeatureBundle, cfg: FeatureConfig) -> np.ndarray:
    """Combine all configured groups into one pooled utterance vector.

    Groups: batch of frame matrices pooled with ``cfg.pool_stats``, plus the
    fixed-length ``prosodic`` vector appended unchanged.
    """
    parts: list[np.ndarray] = []
    for group in cfg.pool_groups:
        if group in _GROUP_TO_ATTR:
            matrix = getattr(bundle, _GROUP_TO_ATTR[group])
            parts.append(pool_matrix(matrix, cfg.pool_stats))
        elif group == "prosodic":
            parts.append(bundle.prosodic)
        elif group == "pitch":
            parts.append(pool_matrix(bundle.pitch, cfg.pool_stats))
        else:
            raise ValueError(f"Unknown feature group: {group!r}")
    return np.concatenate(parts).astype(np.float32)


_BASE_GROUP_ROWS = {
    "mfcc": None,  # filled from cfg.n_mfcc below
    "delta": None,
    "delta2": None,
    "spectral": SPECTRAL_ROWS,
    "pitch": PITCH_ROWS,
    "prosodic": PROSODIC_SIZE,
}


def group_column_sizes(cfg: FeatureConfig) -> list[tuple[str, int]]:
    """Column width of each group's chunk in the pooled vector, in order.

    Frame-matrix groups (mfcc/delta/delta2/spectral/pitch) are pooled with
    ``cfg.pool_stats``; the prosodic vector is appended unchanged.
    """
    widths: list[tuple[str, int]] = []
    for group in cfg.pool_groups:
        if group == "prosodic":
            widths.append((group, PROSODIC_SIZE))
        elif group in _BASE_GROUP_ROWS:
            rows = cfg.n_mfcc if group in ("mfcc", "delta", "delta2") else _BASE_GROUP_ROWS[group]
            widths.append((group, rows * len(cfg.pool_stats)))
        else:
            raise ValueError(f"Unknown feature group: {group!r}")
    return widths


def group_column_ranges(cfg: FeatureConfig) -> dict[str, tuple[int, int]]:
    """Mapping ``group -> (start, stop)`` column slice in the pooled vector."""
    ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for group, width in group_column_sizes(cfg):
        ranges[group] = (offset, offset + width)
        offset += width
    return ranges


def subset_columns(X: np.ndarray, cfg: FeatureConfig, groups: tuple[str, ...]) -> np.ndarray:
    """Select the subset of pooled columns belonging to ``groups``."""
    ranges = group_column_ranges(cfg)
    cols = np.concatenate([np.arange(*ranges[g]) for g in groups])
    return X[:, cols]


__all__ = [
    "DEFAULT_GROUPS",
    "SPECTRAL_ROWS",
    "PITCH_ROWS",
    "PROSODIC_SIZE",
    "FeatureConfig",
    "FeatureBundle",
    "FeatureExtractor",
    "extract_features",
    "pool_matrix",
    "pool_utterance",
    "group_column_sizes",
    "group_column_ranges",
    "subset_columns",
]
