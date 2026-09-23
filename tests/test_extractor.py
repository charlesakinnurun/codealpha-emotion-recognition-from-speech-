"""Tests for the feature extraction pipeline (no dataset required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emotion_recognition.features.audio import load_mono
from emotion_recognition.features.extractor import (
    FeatureBundle,
    FeatureConfig,
    FeatureExtractor,
    extract_features,
    pool_utterance,
)

SR = 16000
CFG = FeatureConfig()

HELPER_TONES = {}


def _tone(freq: float, seconds: float = 2.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_fixed_window_pads_short_clip() -> None:
    short = _tone(220.0, seconds=1.2)  # < 4 s
    ex = FeatureExtractor(CFG)
    bundle = ex.from_waveform(short, sr=SR)
    assert bundle.waveform.size == CFG.n_samples()
    assert bundle.n_frames == CFG.n_frames() == 251


def test_fixed_window_crops_long_clip() -> None:
    long_clip = _tone(220.0, seconds=6.0)
    bundle = FeatureExtractor(CFG).from_waveform(long_clip, sr=SR)
    assert bundle.waveform.size == CFG.n_samples()


def test_bundle_shapes() -> None:
    bundle = FeatureExtractor(CFG).from_waveform(_tone(220.0), sr=SR)
    assert bundle.log_mel.shape == (CFG.n_mels, 251)
    assert bundle.mfcc.shape == (CFG.n_mfcc, 251)
    assert bundle.mfcc_delta.shape == (CFG.n_mfcc, 251)
    assert bundle.mfcc_delta2.shape == (CFG.n_mfcc, 251)
    assert bundle.mfcc_stack.shape == (3 * CFG.n_mfcc, 251)
    assert bundle.spectral.shape == (17, 251)  # zcr, rms, centroid, bw, rolloff, chroma(12)
    assert bundle.pitch.shape == (1, 251)
    assert bundle.prosodic.shape == (8,)  # 4 f0 stats + voiced ratio + 3 energy stats
    assert bundle.sample_rate == SR


def test_extraction_is_deterministic() -> None:
    y = _tone(220.0)
    ex = FeatureExtractor(CFG)
    a = ex.from_waveform(y, sr=SR)
    b = ex.from_waveform(y, sr=SR)
    for attr in ("log_mel", "mfcc", "mfcc_delta", "mfcc_delta2", "spectral", "pitch", "prosodic"):
        assert np.array_equal(getattr(a, attr), getattr(b, attr)), attr


def test_file_and_waveform_paths_agree(tmp_path: Path) -> None:
    import soundfile as sf

    path = tmp_path / "tone.wav"
    sf.write(str(path), _tone(440.0, seconds=1.0), SR)

    ex = FeatureExtractor(CFG)
    from_file = ex.extract(path)
    y, sr = load_mono(path, target_sr=SR)
    from_wave = ex.from_waveform(y, sr)

    assert np.array_equal(from_file.log_mel, from_wave.log_mel)
    assert np.array_equal(from_file.mfcc, from_wave.mfcc)
    assert np.array_equal(from_file.spectral, from_wave.spectral)
    assert np.array_equal(from_file.prosodic, from_wave.prosodic)


def test_extract_features_convenience_returns_bundle(tmp_path: Path) -> None:
    import soundfile as sf

    path = tmp_path / "tone.wav"
    sf.write(str(path), _tone(220.0, seconds=1.0), SR)
    bundle = extract_features(path)
    assert isinstance(bundle, FeatureBundle)
    assert bundle.log_mel.shape[0] == CFG.n_mels


def test_pooled_vector_length_and_determinism() -> None:
    bundle = FeatureExtractor(CFG).from_waveform(_tone(220.0), sr=SR)
    v1 = pool_utterance(bundle, CFG)
    v2 = pool_utterance(bundle, CFG)
    assert np.array_equal(v1, v2)

    # mfcc+delta+delta2 (3*40 rows) * mean/std + spectral (17) * mean/std + prosodic(8)
    assert v1.shape == (3 * 40 * 2 + 17 * 2 + 8,) == (282,)


def test_pool_groups_change_vector_length() -> None:
    bundle = FeatureExtractor(CFG).from_waveform(_tone(220.0), sr=SR)
    mfcc_only = FeatureConfig(pool_groups=("mfcc",))
    spectral_only = FeatureConfig(pool_groups=("spectral",), pool_stats=("mean",))
    assert pool_utterance(bundle, mfcc_only).shape == (40 * 2,)
    assert pool_utterance(bundle, spectral_only).shape == (17,)


def test_fundamental_frequency_detected() -> None:
    bundle = FeatureExtractor(CFG).from_waveform(_tone(220.0), sr=SR)
    voiced = bundle.pitch[0]
    f0 = voiced[voiced > 0]
    assert f0.size > 50  # most frames voiced
    assert abs(float(f0.mean()) - 220.0) < 10.0


def test_silence_is_unvoiced() -> None:
    bundle = FeatureExtractor(CFG).from_waveform(np.zeros(SR * 2, dtype=np.float32), sr=SR)
    assert (bundle.pitch == 0).all()
    assert bundle.prosodic[4] == 0.0  # voiced ratio


def test_spectral_centroid_of_tone(tmp_path: Path) -> None:
    bundle = FeatureExtractor(CFG).from_waveform(_tone(1000.0), sr=SR)
    rms_row = bundle.spectral[1]
    centroid_row = bundle.spectral[2]
    voiced = rms_row > 0.05
    assert voiced.sum() > 100
    assert abs(float(centroid_row[voiced].mean()) - 1000.0) < 150.0


def test_config_from_dict_ignores_unknown_keys_and_computes_frames() -> None:
    cfg = FeatureConfig.from_dict({"n_mels": 32, "n_mfcc": 20, "not_a_field": 1})
    assert cfg.n_mels == 32
    assert cfg.n_mfcc == 20
    assert cfg.n_frames() == 251


def test_nondefault_config_propagates() -> None:
    cfg = FeatureConfig(n_mels=32, n_mfcc=20, duration_s=2.0)
    bundle = FeatureExtractor(cfg).from_waveform(_tone(220.0, seconds=2.5), sr=SR)
    assert bundle.log_mel.shape == (32, 1 + (32000 // 256))
    assert bundle.mfcc.shape == (20, 1 + (32000 // 256))
