"""Training utilities: trainer, early stopping, checkpointing.

Guard torch availability with ``emotion_recognition.training.dataset.TORCH_OK``
(or this module's re-export) before instantiating :class:`Trainer`.

Selection rule enforced by :class:`EarlyStopping` / :class:`ModelCheckpoint`:
the monitored metric is validation macro-F1 only. The test split is never
passed to this module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.metrics import accuracy, macro_f1, per_class_metrics
from ..utils.logging import get_logger

try:  # pragma: no cover - availability varies by host
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    TORCH_OK = True
except Exception:  # pragma: no cover - missing or blocked by App Control
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TORCH_OK = False

LOGGER = get_logger(__name__)

HistoryEntry = dict[str, float]


def balanced_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights (NumPy-only; avoids a sklearn import)."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # never divide by zero for absent classes
    weights = labels_arr.size / (num_classes * counts)
    return torch.from_numpy(weights.astype(np.float32))  # type: ignore[union-attr]


class EarlyStopping:
    """Stop when the monitored metric stops improving (mode: max or min)."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        warmup_epochs: int = 0,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        if patience < 0:
            raise ValueError("patience must be >= 0")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.best: float | None = None
        self.stale_epochs = 0
        self.stopped_epoch: int | None = None
        self.should_stop = False

    def step(self, value: float, epoch: int) -> bool:
        """Record ``value`` for ``epoch``; returns True if training should stop."""
        if epoch < self.warmup_epochs:
            return False
        improved = self._improved(value)
        if improved:
            self.best = float(value)
            self.stale_epochs = 0
            self.should_stop = False
            return False
        self.stale_epochs += 1
        if self.stale_epochs > self.patience:
            self.stopped_epoch = epoch
            self.should_stop = True
            return True
        return False

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta


class ModelCheckpoint:
    """Persist ``best.pt`` on metric improvement plus a ``last.pt`` every epoch."""

    def __init__(self, directory: str | Path, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best: float | None = None
        self.best_path = self.directory / "best.pt"
        self.last_path = self.directory / "last.pt"
        self.improved = False

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def save(
        self,
        payload: dict[str, Any],
        metric: float,
        path: Path,
    ) -> None:
        if TORCH_OK and torch is not None:
            torch.save(payload, path)
        else:  # pragma: no cover - guarded before reaching Trainer
            raise ImportError("PyTorch is required to save checkpoints.")

    def maybe_save_best(self, payload: dict[str, Any], metric: float) -> bool:
        """Save ``best.pt`` if ``metric`` improved; returns True on improvement."""
        self.improved = self._is_better(float(metric))
        if self.improved:
            self.best = float(metric)
            self.save(payload, float(metric), self.best_path)
        self.save(payload, float(metric), self.last_path)
        return self.improved


@dataclass
class TrainerConfig:
    """Run-level hyperparameters for :class:`Trainer`."""

    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-4
    label_smoothing: float = 0.0
    output_dir: str | Path = Path("reports/experiments")
    num_workers: int = 0
    device: str = "auto"
    amp: str = "auto"  # "auto" | "on" | "off"
    grad_clip_norm: float = 0.0  # 0 disables
    extra: dict[str, Any] = field(default_factory=dict)


class Trainer:
    """Train/validate loop with early stopping and checkpointing on val macro-F1.

    The test split must never be supplied: only ``train_loader`` and
    ``val_loader`` are accepted. Metrics are computed with the project's
    sklearn-free evaluation module, so reports run identically everywhere.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        run_dir: str | Path | None = None,
        label_map: dict[str, int] | None = None,
        seed: int = 42,
    ) -> None:
        if not TORCH_OK or torch is None or nn is None:
            raise ImportError(
                "PyTorch is required for Trainer (blocked or not installed on "
                "this host; run in CI)."
            )
        self.model = model
        self.config = config
        self.seed = seed
        self.label_map = label_map or {}

        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.use_amp = self._resolve_amp(config.amp)

        self.run_dir = Path(run_dir) if run_dir is not None else Path(config.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = ModelCheckpoint(self.run_dir, mode="max")
        self.early_stopping = EarlyStopping(
            patience=config.patience, min_delta=config.min_delta, mode="max"
        )
        self.history: list[HistoryEntry] = []
        self.best_epoch: int | None = None
        self.best_val_macro_f1: float = -1.0
        self._scaler = torch.amp.GradScaler(enabled=self.use_amp)

    # -- device / amp resolution ------------------------------------------
    def _resolve_device(self, spec: str) -> torch.device:
        if spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(spec)

    def _resolve_amp(self, spec: str) -> bool:
        if spec == "on":
            return True
        if spec == "off":
            return False
        return self.device.type == "cuda"

    # -- public API --------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor | None = None,
        on_epoch_end: Callable[[int, HistoryEntry], None] | None = None,
    ) -> list[HistoryEntry]:
        """Run the full training loop. Returns the per-epoch history list."""
        if TORCH_OK and torch is None:  # pragma: no cover - defensive
            raise ImportError("PyTorch is required for Trainer.fit().")

        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=self.config.label_smoothing,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=max(1, self.config.patience // 3)
        )

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(
                train_loader, criterion, optimizer=optimizer, epoch=epoch
            )
            val_metrics = self._run_epoch(val_loader, criterion, optimizer=None, epoch=epoch)

            current_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step(val_metrics["macro_f1"])

            entry: HistoryEntry = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": current_lr,
            }
            self.history.append(entry)

            improved = self.checkpoint.maybe_save_best(
                self._payload(epoch), val_metrics["macro_f1"]
            )
            if improved:
                self.best_epoch = epoch
                self.best_val_macro_f1 = float(val_metrics["macro_f1"])

            stop = self.early_stopping.step(val_metrics["macro_f1"], epoch)
            LOGGER.info(
                "epoch %d/%d train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f%s",
                epoch,
                self.config.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["macro_f1"],
                " *" if improved else "",
            )
            if on_epoch_end is not None:
                on_epoch_end(epoch, entry)
            if stop:
                LOGGER.info(
                    "early stopping at epoch %d (best val_macro_f1=%.4f @ epoch %s)",
                    epoch,
                    self.best_val_macro_f1,
                    self.best_epoch,
                )
                break

        self._write_history()
        return self.history

    @torch.no_grad()  # type: ignore[misc]
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        """Loss + accuracy + macro-F1 + per-class metrics (no gradients)."""
        criterion = nn.CrossEntropyLoss()
        return self._run_epoch(loader, criterion, optimizer=None, epoch=0, collect_preds=True)

    def load_best(self) -> int | None:
        """Load ``best.pt`` weights into the model; returns the checkpoint epoch."""
        if not self.checkpoint.best_path.is_file():
            return None
        payload = torch.load(self.checkpoint.best_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model_state_dict"])
        return int(payload["epoch"])

    # -- internals ---------------------------------------------------------
    def _payload(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": None,  # filled by _run_epoch callers via closure is overkill; kept minimal
            "label_map": self.label_map,
            "val_macro_f1": self.best_val_macro_f1,
            "config": {
                "epochs": self.config.epochs,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "patience": self.config.patience,
                "seed": self.seed,
            },
        }

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        collect_preds: bool = False,
    ) -> dict[str, Any]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        n_batches = 0
        all_true: list[int] = []
        all_pred: list[int] = []

        autocast_enabled = self.use_amp
        ctx = (
            torch.autocast(device_type=self.device.type, enabled=autocast_enabled)
            if TORCH_OK
            else _nullcontext()
        )

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
                with ctx:  # type: ignore[attr-defined]
                    logits = self.model(inputs)
                    loss = criterion(logits, targets)
                if self.use_amp:
                    self._scaler.scale(loss).backward()
                    if self.config.grad_clip_norm > 0.0:
                        self._scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip_norm
                        )
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    if self.config.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip_norm
                        )
                    optimizer.step()
            else:
                with torch.no_grad(), ctx:  # type: ignore[attr-defined]
                    logits = self.model(inputs)
                    loss = criterion(logits, targets)

            total_loss += float(loss.detach().item())
            n_batches += 1
            preds = logits.detach().argmax(dim=1)
            all_true.extend(targets.detach().cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

        result: dict[str, Any] = {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": accuracy(all_true, all_pred),
            "macro_f1": macro_f1(all_true, all_pred),
        }
        if collect_preds:
            result["per_class"] = per_class_metrics(all_true, all_pred)
            result["y_true"] = all_true
            result["y_pred"] = all_pred
        return result

    def _write_history(self) -> None:
        path = self.run_dir / "history.json"
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        LOGGER.info("wrote history -> %s", path)


class _nullcontext:  # minimal stand-in when torch is missing (defensive)
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


__all__ = [
    "TORCH_OK",
    "Trainer",
    "TrainerConfig",
    "EarlyStopping",
    "ModelCheckpoint",
    "balanced_class_weights",
]
