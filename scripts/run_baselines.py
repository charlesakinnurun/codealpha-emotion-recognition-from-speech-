"""Run the classical baselines: ablations (train CV) + one-shot val evaluation.

The test split is *never* touched unless ``--evaluate-test`` is passed, which
is reserved for the single, final, post-selection run.

Usage:
    python scripts/run_baselines.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_recognition.evaluation.metrics import accuracy, macro_f1  # noqa: E402
from emotion_recognition.features.extractor import (  # noqa: E402
    FeatureConfig,
    FeatureExtractor,
    pool_utterance,
    subset_columns,
)
from emotion_recognition.models.baseline import (  # noqa: E402
    DEFAULT_ABLATIONS,
    build_feature_matrix,
    make_model,
    run_ablations,
)
from emotion_recognition.utils.logging import get_logger  # noqa: E402

RAVDESS_DIR = REPO_ROOT / "data" / "raw" / "ravdess"


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=REPO_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--evaluate-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    log = get_logger(__name__)
    config_path = _resolve(args.config, REPO_ROOT)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    metadata_path = _resolve(raw["metadata"], config_path.parent)
    output_dir = _resolve(raw["output_dir"], config_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(raw.get("seed", 42))
    n_splits = int(raw.get("kfold_splits", 5))
    kinds = tuple(raw.get("model_kinds", ("logreg", "svm", "rf")))
    cfg = FeatureConfig.from_dict(raw.get("features", {}))

    import pandas as pd

    metadata = pd.read_csv(metadata_path)
    if "dataset_split" not in metadata.columns:
        log.error("metadata.csv missing 'dataset_split'; run `make prepare` first.")
        return 1
    log.info("records by split: %s", metadata.groupby("dataset_split").size().to_dict())

    def _rows(split: str) -> tuple[list[Path], list[str]]:
        sub = metadata.loc[metadata["dataset_split"] == split]
        return [RAVDESS_DIR / f for f in sub["file_name"]], sub["emotion"].tolist()

    log.info("extracting + pooling train features (single extractor pass)...")
    X_train, y_train = build_feature_matrix(*_rows("train"), cfg=cfg)
    log.info("train matrix: %s x %s", *X_train.shape)

    log.info("running %d-fold CV ablations on train...", n_splits)
    ablations = run_ablations(X_train, y_train, cfg, kinds=kinds, seed=seed, n_splits=n_splits)
    ablations = ablations.sort_values(["model", "macro_f1_mean"], ascending=[True, False])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_path = output_dir / f"ablations_{stamp}.csv"
    ablations.to_csv(ablation_path, index=False)
    print(ablations.to_string(index=False))
    log.info("ablation table written to %s", ablation_path)

    best = ablations.loc[ablations["macro_f1_mean"].idxmax()]
    log.info(
        "best on train CV: %s [%s] macro_f1=%.4f",
        best["model"],
        best["groups"],
        best["macro_f1_mean"],
    )

    groups = DEFAULT_ABLATIONS[str(best["groups"])]
    model = make_model(str(best["model"]), seed=seed)
    model.fit(subset_columns(X_train, cfg, groups), y_train)
    log.info("trained %s on %s training rows", best["model"], len(y_train))

    X_val_paths, y_val = _rows("val")
    extractor = FeatureExtractor(cfg)
    X_val = np_stack_pool(extractor, X_val_paths, cfg)
    y_pred_val = model.predict(subset_columns(X_val, cfg, groups))
    val_metrics = {
        "model": str(best["model"]),
        "groups": str(best["groups"]),
        "accuracy": accuracy(y_val, y_pred_val),
        "macro_f1": macro_f1(y_val, y_pred_val),
        "n_train": int(len(y_train)),
        "seed": seed,
    }
    metrics_path = output_dir / f"val_metrics_{stamp}.json"
    metrics_path.write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    log.info(
        "val (one-shot) macro_f1=%.4f accuracy=%.4f -> %s",
        val_metrics["macro_f1"],
        val_metrics["accuracy"],
        metrics_path,
    )

    if args.evaluate_test:  # reserved for the single, final, post-selection run
        X_test_paths, y_test = _rows("test")
        X_test = np_stack_pool(extractor, X_test_paths, cfg)
        final = {
            **val_metrics,
            "split": "test",
            "accuracy": accuracy(y_test, model.predict(subset_columns(X_test, cfg, groups))),
            "macro_f1": macro_f1(y_test, model.predict(subset_columns(X_test, cfg, groups))),
        }
        final_path = output_dir / f"test_metrics_{stamp}.json"
        final_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        log.info("test (final) macro_f1=%.4f -> %s", final["macro_f1"], final_path)

    return 0


def np_stack_pool(extractor: FeatureExtractor, paths: list[Path], cfg: FeatureConfig):
    """Extract + pool a list of files into one stacked matrix."""
    import numpy as np

    return np.stack([pool_utterance(extractor.extract(p), cfg) for p in paths])


if __name__ == "__main__":
    raise SystemExit(main())
