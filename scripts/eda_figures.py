"""Regenerate EDA figures from the prepared metadata CSV.

Usage:
    python scripts/eda_figures.py --metadata data/processed/metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emotion_recognition.utils.figures import generate_eda_figures
from emotion_recognition.utils.logging import get_logger

LOGGER = get_logger("scripts.eda_figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA figures from metadata.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.csv")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="RAVDESS data root (needed for waveform/spectrogram examples). "
        "Defaults to the parent of the metadata file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory to write PNG figures into.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.metadata.is_file():
        LOGGER.error("Metadata file not found: %s", args.metadata)
        return 1

    metadata = pd.read_csv(args.metadata)
    data_dir = args.data_dir or args.metadata.parent
    try:
        outputs = generate_eda_figures(metadata, data_dir, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("EDA failed: %s", exc)
        return 1

    for path in outputs:
        LOGGER.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
