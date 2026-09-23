"""CLI entry point for the RAVDESS data preparation pipeline.

Usage:
    python scripts/prepare_data.py --config configs/data.yaml

Reads config from YAML, runs :func:`prepare_dataset`, and prints a compact
summary of the resulting splits and class distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emotion_recognition.config import DataConfig
from emotion_recognition.data.prepare import prepare_dataset, split_needs_balancing_check
from emotion_recognition.utils.logging import get_logger

LOGGER = get_logger("scripts.prepare_data")

pd.set_option("display.width", 140)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RAVDESS metadata and speaker splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to a YAML file containing a top-level 'data:' mapping.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        LOGGER.error("Config file not found: %s", args.config)
        return 1

    cfg = DataConfig.from_yaml(args.config)
    try:
        metadata, _quarantine = prepare_dataset(cfg)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return 1
    except ValueError as exc:
        LOGGER.error("Split validation failed: %s", exc)
        return 1

    split_needs_balancing_check(metadata)
    LOGGER.info("Total valid utterances: %d", len(metadata))
    LOGGER.info(
        "Class distribution (all splits):\n%s", metadata["emotion"].value_counts().to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
