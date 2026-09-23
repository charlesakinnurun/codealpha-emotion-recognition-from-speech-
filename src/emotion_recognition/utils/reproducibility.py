"""Reproducibility helpers (seeding across random / numpy / torch)."""

from __future__ import annotations

import os
import random

try:  # pragma: no cover - availability varies by host
    import numpy as np

    NUMPY_OK = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    NUMPY_OK = False

try:  # pragma: no cover - availability varies by host
    import torch

    TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_OK = False


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy (if present), and torch (if present) deterministically."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if NUMPY_OK and np is not None:
        np.random.seed(seed)
    if TORCH_OK and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CUDA host
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


__all__ = ["set_seed", "NUMPY_OK", "TORCH_OK"]
