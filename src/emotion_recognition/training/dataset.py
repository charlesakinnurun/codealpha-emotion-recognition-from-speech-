"""Dataset abstraction for training deep models on RAVDESS metadata rows.

``SpeechEmotionDataset`` materializes log-Mel tensors once at construction
(small corpus: ~1440 files => ~90 MB of float32 ``(64, 251)`` matrices), so
every epoch is a cheap memory lookup instead of re-running DSP.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..features.extractor import FeatureConfig, FeatureExtractor

try:  # pragma: no cover - availability varies by host
    import torch
    from torch.utils.data import Dataset as TorchDataset

    TORCH_OK = True
except Exception:  # pragma: no cover - missing or blocked by App Control
    torch = None  # type: ignore[assignment]
    TorchDataset = object  # type: ignore[misc, assignment]
    TORCH_OK = False

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor


class SpeechEmotionDataset(TorchDataset):  # type: ignore[misc]
    """Log-Mel feature dataset backed by metadata rows for a single split.

    Args:
        rows: Metadata rows (one per utterance) restricted to one split.
        label_map: Emotion name -> contiguous integer id used as the target.
        cfg: Feature configuration; must match the training/serving pipeline.
        audio_root: Optional prefix joined with relative ``file_path`` entries.
            Absolute paths (as produced by :func:`prepare_dataset`) are used
            as-is.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        label_map: dict[str, int],
        cfg: FeatureConfig | None = None,
        audio_root: str | Path | None = None,
    ) -> None:
        if not TORCH_OK or torch is None:
            raise ImportError(
                "PyTorch is required for SpeechEmotionDataset "
                "(blocked or not installed on this host; run in CI)."
            )
        if len(rows) == 0:
            raise ValueError("SpeechEmotionDataset received zero rows.")
        missing = set(rows["emotion"]) - set(label_map)
        if missing:
            raise ValueError(f"label_map is missing emotions present in rows: {sorted(missing)}")

        self.cfg = cfg or FeatureConfig()
        self.audio_root = Path(audio_root) if audio_root is not None else None
        self._paths: list[Path] = [self._resolve(p) for p in rows["file_path"].tolist()]
        self._labels: list[int] = [label_map[e] for e in rows["emotion"].tolist()]

        extractor = FeatureExtractor(self.cfg)
        self._spectrograms: list[np.ndarray] = [
            extractor.extract(path).log_mel.astype(np.float32, copy=False) for path in self._paths
        ]

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute() or self.audio_root is None:
            return path
        return self.audio_root / path

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Return ``(log_mel(1, n_mels, n_frames) float32, label int)``."""
        spec = self._spectrograms[index]
        tensor = torch.from_numpy(spec).unsqueeze(0)
        return tensor, self._labels[index]

    @property
    def labels(self) -> list[int]:
        """Integer labels in dataset order (for class-weight computation)."""
        return list(self._labels)

    @property
    def paths(self) -> list[Path]:
        """Audio file paths in dataset order (for error analysis later)."""
        return list(self._paths)


def build_label_map(emotions: Sequence[str]) -> dict[str, int]:
    """Deterministic emotion -> contiguous id mapping (alphabetical order)."""
    unique = sorted(set(emotions))
    if not unique:
        raise ValueError("build_label_map received an empty emotion sequence.")
    return {name: idx for idx, name in enumerate(unique)}


def split_rows(metadata: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of ``metadata`` belonging to one ``dataset_split`` value."""
    if "dataset_split" not in metadata.columns:
        raise ValueError("metadata is missing the 'dataset_split' column.")
    rows = metadata.loc[metadata["dataset_split"] == split]
    if rows.empty:
        raise ValueError(f"metadata has no rows for split {split!r}.")
    return rows.reset_index(drop=True)


__all__ = [
    "TORCH_OK",
    "SpeechEmotionDataset",
    "build_label_map",
    "split_rows",
    "torch",
]
