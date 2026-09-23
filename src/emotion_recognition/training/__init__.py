"""Training package (PyTorch components).

Everything in this package soft-degrades when torch is unavailable: each module
guards its imports so ``emotion_recognition.training`` stays importable on
hosts where PyTorch is blocked by OS policy. Guard before use::

    from emotion_recognition.training.dataset import TORCH_OK
    if not TORCH_OK:
        ...
"""

from __future__ import annotations

from .dataset import TORCH_OK as TORCH_OK

__all__ = ["TORCH_OK"]
