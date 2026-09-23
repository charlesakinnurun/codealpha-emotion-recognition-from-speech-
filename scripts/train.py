"""Train a deep model (EmotionCNN by default) on RAVDESS log-Mel features.

Selection is performed on the validation macro-F1 only; the test split is
never read by this script. Test evaluation is reserved for `evaluate.py`.

Usage:
    python scripts/train.py --config configs/cnn.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_recognition.config import TrainConfig  # noqa: E402
from emotion_recognition.models import cnn as cnn_module  # noqa: E402
from emotion_recognition.training.dataset import (  # noqa: E402
    SpeechEmotionDataset,
    build_label_map,
    split_rows,
)
from emotion_recognition.training.trainer import (  # noqa: E402
    Trainer,
    TrainerConfig,
    balanced_class_weights,
)
from emotion_recognition.utils.logging import get_logger  # noqa: E402
from emotion_recognition.utils.reproducibility import set_seed  # noqa: E402

LOGGER = get_logger("scripts.train")


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _model_from(tcfg: TrainConfig) -> cnn_module.EmotionCNN:
    spec = dict(tcfg.model)
    kind = str(spec.pop("kind", "cnn"))
    if kind != "cnn":
        raise ValueError(
            f"Unsupported model kind {kind!r}; this phase supports 'cnn' only "
            "(see Phase 6 for cnn_lstm)."
        )
    cnn_cfg = cnn_module.CNNConfig(
        num_classes=int(spec.pop("num_classes", 8)),
        conv_channels=tuple(int(c) for c in spec.pop("conv_channels", (32, 64, 128))),
        kernel_size=int(spec.pop("kernel_size", 3)),
        dropout=tuple(float(d) for d in spec.pop("dropout", (0.2, 0.3, 0.4))),
    )
    if spec:
        LOGGER.warning("Ignoring unknown model options: %s", sorted(spec))
    return cnn_module.EmotionCNN(cnn_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "cnn.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    args = parser.parse_args()

    if not args.config.is_file():
        LOGGER.error("Config file not found: %s", args.config)
        return 1
    tcfg = TrainConfig.from_yaml(args.config)
    if args.epochs is not None:
        tcfg = replace(tcfg, epochs=args.epochs)

    if not cnn_module.TORCH_OK:
        LOGGER.error("PyTorch is required to train (blocked or not installed on this host).")
        return 1
    if not tcfg.metadata.is_file():
        LOGGER.error("metadata.csv not found: %s (run `make prepare` first).", tcfg.metadata)
        return 1

    set_seed(tcfg.seed)
    metadata = pd.read_csv(tcfg.metadata)
    if "dataset_split" not in metadata.columns:
        LOGGER.error("metadata missing 'dataset_split'; run `make prepare` first.")
        return 1
    LOGGER.info("records by split: %s", metadata.groupby("dataset_split").size().to_dict())

    train_rows = split_rows(metadata, "train")
    val_rows = split_rows(metadata, "val")
    label_map = build_label_map(train_rows["emotion"].tolist())

    train_ds = SpeechEmotionDataset(train_rows, label_map, cfg=tcfg.features)
    val_ds = SpeechEmotionDataset(val_rows, label_map, cfg=tcfg.features)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        shuffle=True,
        num_workers=tcfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        num_workers=tcfg.num_workers,
    )

    weights = balanced_class_weights(train_ds.labels, len(label_map))
    LOGGER.info("class weights: %s", weights.tolist())

    model = _model_from(tcfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = tcfg.output_dir / f"cnn_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model,
        config=TrainerConfig(
            epochs=tcfg.epochs,
            lr=tcfg.lr,
            weight_decay=tcfg.weight_decay,
            patience=tcfg.patience,
            min_delta=tcfg.min_delta,
            output_dir=tcfg.output_dir,
            num_workers=tcfg.num_workers,
            device=tcfg.device,
            amp=tcfg.amp,
        ),
        run_dir=run_dir,
        label_map=label_map,
        seed=tcfg.seed,
    )
    history = trainer.fit(train_loader, val_loader, class_weights=weights)

    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": tcfg.seed,
                "metadata": str(tcfg.metadata),
                "features": tcfg.features.__dict__,
                "model": dict(tcfg.model),
                "training": {
                    "batch_size": tcfg.batch_size,
                    "epochs": tcfg.epochs,
                    "lr": tcfg.lr,
                    "weight_decay": tcfg.weight_decay,
                    "patience": tcfg.patience,
                    "min_delta": tcfg.min_delta,
                    "num_workers": tcfg.num_workers,
                    "device": tcfg.device,
                    "amp": tcfg.amp,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    result = {
        "run_dir": str(run_dir),
        "best_epoch": trainer.best_epoch,
        "best_val_macro_f1": trainer.best_val_macro_f1,
        "epochs_run": len(history),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info(
        "best epoch=%s val_macro_f1=%.4f -> %s",
        trainer.best_epoch,
        trainer.best_val_macro_f1,
        run_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
