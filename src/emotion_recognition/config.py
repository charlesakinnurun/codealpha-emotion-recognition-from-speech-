"""Configuration loading for the data pipeline and experiments.

Configuration is stored as YAML and loaded into frozen dataclasses so that
every experiment/pipeline run is reproducible from a single file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .features.extractor import FeatureConfig


def _resolve_path(value: str | Path, base: Path | None = None) -> Path:
    """Resolve a path, expanding ``~`` and relativizing against ``base``.

    Relative paths inside a config file are interpreted relative to the
    *config file's* directory, so a config stays portable across checkouts.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = base if base is not None else Path.cwd()
    return (base / path).resolve()


@dataclass(frozen=True)
class DataConfig:
    """Configuration for the RAVDESS preparation pipeline.

    Attributes:
        data_dir: Root directory containing `Actor_XX/` subfolders.
        output_dir: Where `metadata.csv` and `quarantine.csv` are written.
        modality: RAVDESS modality code to keep (03 = audio-only).
        vocal_channel: RAVDESS vocal channel to keep (01 = speech).
        min_duration_s: Minimum acceptable utterance duration.
        max_duration_s: Maximum acceptable utterance duration.
        split_ratios: (train, val, test) speaker proportions, must sum to 1.
        seed: RNG seed for the deterministic speaker split.
    """

    data_dir: Path
    output_dir: Path = field(default=Path("data/processed"))
    modality: str = "03"
    vocal_channel: str = "01"
    min_duration_s: float = 0.5
    max_duration_s: float = 30.0
    split_ratios: tuple[float, float, float] = (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        """Build a :class:`DataConfig` from a YAML file.

        Relative paths in the file are resolved against the file's directory
        (after expanding ``~``). Expected shape:

        .. code-block:: yaml

            data:
              data_dir: data/raw/ravdess
              split_ratios: [0.667, 0.166, 0.167]
              seed: 42
        """
        config_path = Path(path).expanduser().resolve()
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        section = raw["data"]
        if not isinstance(section, dict):
            raise ValueError("Config file must contain a top-level 'data:' mapping.")

        base = config_path.parent
        data_dir = _resolve_path(section.pop("data_dir"), base)
        output_dir = _resolve_path(section.pop("output_dir", "data/processed"), base)
        ratios = section.pop("split_ratios", [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must contain exactly three values summing to 1.")
        return cls(data_dir=data_dir, output_dir=output_dir, split_ratios=tuple(ratios), **section)


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for the deep-model training script.

    Attributes:
        seed: Global RNG seed (set before building model/dataloaders).
        metadata: Path to ``metadata.csv`` produced by the prepare pipeline.
        output_dir: Root directory below which ``cnn_<timestamp>/`` run folders
            are written (checkpoints, history, config snapshot).
        features: Feature-extraction parameters shared with the serving path.
        model: Arbitrary model hyperparameters (interpreted by the script).
        batch_size / epochs / lr / weight_decay / patience / min_delta /
        num_workers / device / amp: run-level training settings.
    """

    metadata: Path
    seed: int = 42
    output_dir: Path = field(default=Path("reports/experiments"))
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    amp: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Build a :class:`TrainConfig` from a YAML file.

        Relative paths are resolved against the file's directory. Expected
        shape:

        .. code-block:: yaml

            seed: 42
            metadata: data/processed/metadata.csv
            output_dir: reports/experiments
            features:
              n_mels: 64
            model:
              conv_channels: [32, 64, 128]
            training:
              batch_size: 32
              epochs: 50
        """
        config_path = Path(path).expanduser().resolve()
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("TrainConfig YAML must contain a top-level mapping.")

        base = config_path.parent
        training = raw.pop("training", {}) or {}
        if not isinstance(training, dict):
            raise ValueError("The 'training:' section of a TrainConfig YAML must be a mapping.")

        metadata = _resolve_path(raw.pop("metadata"), base)
        output_dir = _resolve_path(raw.pop("output_dir", "reports/experiments"), base)
        features = FeatureConfig.from_dict(raw.pop("features", {}) or {})
        model = raw.pop("model", {}) or {}

        allowed = {f.name for f in fields(cls)}
        common = {k: v for k, v in {**training, **raw}.items() if k in allowed and k != "metadata"}
        return cls(
            metadata=metadata, output_dir=output_dir, features=features, model=dict(model), **common
        )


__all__ = ["DataConfig", "TrainConfig"]
