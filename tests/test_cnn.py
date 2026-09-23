"""Unit tests for the EmotionCNN architecture.

These soft-skip when torch is unavailable (App-Control-blocked hosts) using an
explicit try/except, matching the sklearn pattern in ``test_baselines.py``.
"""

from __future__ import annotations

import pytest

try:
    import torch

    from emotion_recognition.models.cnn import CNNConfig, EmotionCNN
    from emotion_recognition.utils.reproducibility import set_seed
except Exception:  # missing, or blocked by an OS App-Control policy on this host
    pytest.skip("torch unavailable on this host", allow_module_level=True)

N_MELS = 64
N_FRAMES = 251
NUM_CLASSES = 8


@pytest.fixture()
def model():
    return EmotionCNN(CNNConfig(conv_channels=(8, 16), dropout=(0.0, 0.0), num_classes=NUM_CLASSES))


def test_forward_shape(model: EmotionCNN) -> None:
    x = torch.randn(2, 1, N_MELS, N_FRAMES)
    out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_feature_bottleneck_shape() -> None:
    model = EmotionCNN(CNNConfig(conv_channels=(16, 32, 64), dropout=(0.0, 0.0, 0.0)))
    x = torch.randn(4, 1, N_MELS, N_FRAMES)
    feats = model.features(x)
    assert feats.ndim == 4
    assert feats.shape[0] == 4
    assert feats.shape[1] == 64


def test_num_parameters_nonzero(model: EmotionCNN) -> None:
    n = sum(p.numel() for p in model.parameters())
    assert n > 0


def test_deterministic_forward() -> None:
    set_seed(42)
    model = EmotionCNN(CNNConfig(conv_channels=(8, 16)))
    model.eval()  # BatchNorm must be in eval mode for run-to-run equality
    x = torch.randn(3, 1, N_MELS, N_FRAMES)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
