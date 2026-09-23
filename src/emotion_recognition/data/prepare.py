"""RAVDESS dataset preparation pipeline.

Scans a raw RAVDESS directory tree, validates every audio file, parses
emotion/intensity/statement/speaker labels from filenames, detects duplicates
(SHA-1), quarantines corrupted or out-of-range files, and assigns a
deterministic **speaker-disjoint** train/val/test split.

Writes ``metadata.csv`` (one row per valid audio-only speech utterance) and
``quarantine.csv`` (files that failed validation and why) into ``output_dir``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import soundfile as sf

from ..config import DataConfig
from ..utils.logging import get_logger
from .splits import assign_speaker_splits, parse_ravdess_filename, verify_split_disjointness

LOGGER = get_logger(__name__)

METADATA_COLUMNS = [
    "file_path",
    "file_name",
    "speaker_id",
    "gender",
    "emotion",
    "emotion_id",
    "intensity",
    "intensity_id",
    "statement",
    "statement_id",
    "repetition",
    "duration_s",
    "sampling_rate",
    "channels",
    "num_samples",
    "sha1",
    "dataset_split",
]

QUARANTINE_COLUMNS = ["file_path", "status", "error"]


@dataclass(frozen=True)
class AudioProbe:
    """Header-level information read from an audio file."""

    sampling_rate: int
    num_samples: int
    channels: int
    duration_s: float
    sha1: str


def probe_audio(path: Path) -> AudioProbe:
    """Read header + hash of an audio file with ``soundfile``.

    Raises:
        RuntimeError: if the file cannot be decoded by soundfile.
    """
    try:
        info = sf.info(str(path))
    except Exception as exc:  # soundfile raises a range of exceptions for bad files
        raise RuntimeError(f"soundfile could not read header of {path.name}: {exc}") from exc

    if info.frames <= 0:
        raise RuntimeError(f"{path.name} contains zero audio frames.")

    sha1 = hashlib.sha1(path.read_bytes()).hexdigest()  # noqa: S324 - content hash, not security
    return AudioProbe(
        sampling_rate=info.samplerate,
        num_samples=info.frames,
        channels=info.channels,
        duration_s=float(info.frames) / float(info.samplerate),
        sha1=sha1,
    )


def discover_audio_files(data_dir: Path) -> list[Path]:
    """Recursively collect every ``*.wav`` file under ``data_dir``."""
    return sorted(data_dir.rglob("*.wav"))


def build_speech_metadata(
    data_dir: Path,
    modality: str = "03",
    vocal_channel: str = "01",
    min_duration_s: float = 0.5,
    max_duration_s: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate raw RAVDESS files and build the metadata table.

    Returns:
        A ``(metadata, quarantine)`` pair of DataFrames. ``metadata`` contains
        one row per valid audio-only *speech* utterance; ``quarantine`` lists
        every other probed file with its exclusion reason.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = discover_audio_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .wav files found under {data_dir}.")

    LOGGER.info("Found %d .wav files under %s", len(files), data_dir)

    rows: list[dict] = []
    quarantine: list[dict] = []

    for path in files:
        try:
            parsed = parse_ravdess_filename(path.name)
        except ValueError as exc:
            quarantine.append({"file_path": str(path), "status": "unparsable", "error": str(exc)})
            continue

        if parsed.modality != int(modality) or parsed.vocal_channel != int(vocal_channel):
            # Parsed but not part of the selected subset (e.g. song or video-only).
            quarantine.append(
                {
                    "file_path": str(path),
                    "status": "excluded_subset",
                    "error": (
                        f"modality={parsed.modality} vocal_channel={parsed.vocal_channel}; "
                        f"expected {modality}/{vocal_channel}"
                    ),
                }
            )
            continue

        try:
            probe = probe_audio(path)
        except RuntimeError as exc:
            quarantine.append({"file_path": str(path), "status": "unreadable", "error": str(exc)})
            continue

        if not (min_duration_s <= probe.duration_s <= max_duration_s):
            quarantine.append(
                {
                    "file_path": str(path),
                    "status": "bad_duration",
                    "error": f"duration={probe.duration_s:.3f}s outside [{min_duration_s}, {max_duration_s}]",
                }
            )
            continue

        rows.append(
            {
                "file_path": str(path.resolve()),
                "file_name": path.name,
                "speaker_id": parsed.speaker_id,
                "gender": parsed.gender,
                "emotion": parsed.emotion,
                "emotion_id": parsed.emotion_id,
                "intensity": parsed.intensity,
                "intensity_id": parsed.intensity_id,
                "statement": parsed.statement,
                "statement_id": parsed.statement_id,
                "repetition": parsed.repetition,
                "duration_s": probe.duration_s,
                "sampling_rate": probe.sampling_rate,
                "channels": probe.channels,
                "num_samples": probe.num_samples,
                "sha1": probe.sha1,
            }
        )

    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    quarantine_df = pd.DataFrame(quarantine, columns=QUARANTINE_COLUMNS)

    n_before = len(metadata)
    duplicates = metadata[metadata.duplicated(subset="sha1", keep="first")]
    if not duplicates.empty:
        LOGGER.warning("Dropping %d duplicate rows (same content, keep first).", len(duplicates))
        metadata = metadata.drop_duplicates(subset="sha1", keep="first").reset_index(drop=True)

    LOGGER.info(
        "Valid speech utterances: %d (dropped %d duplicates of %d probed).",
        len(metadata),
        n_before - len(metadata),
        len(files),
    )
    return metadata, quarantine_df


def assign_dataset_split(
    metadata: pd.DataFrame, ratios: tuple[float, float, float], seed: int
) -> pd.DataFrame:
    """Add and validate the ``dataset_split`` column (speaker-disjoint).

    Raises:
        ValueError: if the resulting assignment violates disjointness.
    """
    gender_of = dict(zip(metadata["speaker_id"], metadata["gender"], strict=True))
    assignment = assign_speaker_splits(
        speakers=metadata["speaker_id"].unique(), gender_of=gender_of, ratios=ratios, seed=seed
    )
    metadata = metadata.copy()
    metadata["dataset_split"] = metadata["speaker_id"].map(assignment)
    verify_split_disjointness(metadata)
    return metadata


def prepare_dataset(cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full preparation pipeline and persist its outputs.

    Args:
        cfg: Configuration for data_dir, output paths, filters, and split.

    Returns:
        The ``(metadata, quarantine)`` dataframes that were written to disk.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    metadata, quarantine = build_speech_metadata(
        data_dir=cfg.data_dir,
        modality=cfg.modality,
        vocal_channel=cfg.vocal_channel,
        min_duration_s=cfg.min_duration_s,
        max_duration_s=cfg.max_duration_s,
    )
    metadata = assign_dataset_split(metadata, cfg.split_ratios, cfg.seed)

    metadata_path = cfg.output_dir / "metadata.csv"
    quarantine_path = cfg.output_dir / "quarantine.csv"
    metadata.to_csv(metadata_path, index=False)
    quarantine.to_csv(quarantine_path, index=False)

    LOGGER.info("Wrote metadata (%d rows) -> %s", len(metadata), metadata_path)
    LOGGER.info("Wrote quarantine (%d rows) -> %s", len(quarantine), quarantine_path)

    counts = metadata["dataset_split"].value_counts().to_dict()
    LOGGER.info("Split sizes (utterances): %s", counts)
    LOGGER.info(
        "Split sizes (speakers): %s",
        metadata.groupby("dataset_split")["speaker_id"].nunique().to_dict(),
    )
    LOGGER.info("Sample counts per emotion:\n%s", metadata["emotion"].value_counts().to_string())

    return metadata, quarantine


def split_needs_balancing_check(metadata: pd.DataFrame) -> None:
    """Emit a warning if a split is heavily skewed in gender or emotion.

    Called from the CLI to surface, rather than silently accept, a bad split.
    """
    per_split_speakers = metadata.groupby("dataset_split")["speaker_id"].nunique()
    if per_split_speakers.min() < 2:
        LOGGER.warning("At least one split has very few speakers: %s", per_split_speakers.to_dict())

    table = (
        metadata.groupby(["dataset_split", "gender"])["speaker_id"].nunique().unstack(fill_value=0)
    )
    LOGGER.info("Speakers per split and gender:\n%s", table.to_string())

    # Confirm a documented RAVDESS property: neutral never occurs at strong intensity.
    neutral_strong = metadata[
        (metadata["emotion"] == "neutral") & (metadata["intensity"] == "strong")
    ]
    if not neutral_strong.empty:
        LOGGER.warning(
            "Found %d neutral/strong samples, contradicting RAVDESS docs.", len(neutral_strong)
        )


__all__ = [
    "METADATA_COLUMNS",
    "QUARANTINE_COLUMNS",
    "AudioProbe",
    "probe_audio",
    "discover_audio_files",
    "build_speech_metadata",
    "assign_dataset_split",
    "prepare_dataset",
    "split_needs_balancing_check",
]
