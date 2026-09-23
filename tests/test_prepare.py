"""End-to-end tests for the dataset preparation pipeline (synthetic data only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from emotion_recognition.config import DataConfig
from emotion_recognition.data.prepare import (
    build_speech_metadata,
    prepare_dataset,
    probe_audio,
)
from tests.helpers import ACTOR_IDS, expected_speech_files

REQUIRED_COLUMNS = {
    "file_path",
    "file_name",
    "speaker_id",
    "gender",
    "emotion",
    "emotion_id",
    "intensity",
    "statement",
    "duration_s",
    "sampling_rate",
    "channels",
    "num_samples",
    "sha1",
    "dataset_split",
}


def test_metadata_schema_on_clean_tree(ravdess_tree: Path) -> None:
    metadata, quarantine = build_speech_metadata(ravdess_tree)

    assert metadata.empty is False
    assert metadata["sampling_rate"].nunique() == 1
    assert metadata["channels"].nunique() == 1
    assert (metadata["duration_s"] > 0).all()
    assert metadata["sha1"].apply(lambda h: len(h) == 40).all()
    assert quarantine.empty  # clean tree: nothing to quarantine

    # One speech file per (actor, emotion_id, intensity) combo.
    expected = expected_speech_files() * len(ACTOR_IDS)
    assert len(metadata) == expected

    # All 8 emotions present, including every target label.
    assert set(metadata["emotion"].unique()) == {
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    }


def test_prepare_dataset_assigns_speaker_disjoint_split(ravdess_tree: Path) -> None:
    cfg = DataConfig(data_dir=ravdess_tree, output_dir=ravdess_tree.parent / "processed")
    metadata, _ = prepare_dataset(cfg)

    out_csv = cfg.output_dir / "metadata.csv"
    assert out_csv.is_file()

    reread = pd.read_csv(out_csv)
    assert REQUIRED_COLUMNS.issubset(reread.columns)

    # Speaker-disjoint, gender-balanced assignment for 16 speakers (8M/8F):
    # per gender train=int(8*2/3)=5, val=int(8*1/6)=1, test=8-5-1=2.
    per_split_speakers = reread.groupby("dataset_split")["speaker_id"].nunique().to_dict()
    assert set(per_split_speakers) == {"train", "val", "test"}
    assert per_split_speakers["train"] == 10
    assert per_split_speakers["val"] == 2
    assert per_split_speakers["test"] == 4

    gender_counts = reread.groupby(["dataset_split", "gender"])["speaker_id"].nunique()
    assert dict(gender_counts) == {
        ("train", "female"): 5,
        ("train", "male"): 5,
        ("val", "female"): 1,
        ("val", "male"): 1,
        ("test", "female"): 2,
        ("test", "male"): 2,
    }


def test_corrupt_song_and_duplicate_are_quarantined_or_dropped(
    ravdess_tree_with_anomalies: Path,
) -> None:
    metadata, quarantine = build_speech_metadata(ravdess_tree_with_anomalies)
    statuses = quarantine.set_index("file_path")["status"].to_dict()

    corrupt = (ravdess_tree_with_anomalies / "Actor_01" / "03-01-05-01-02-01-01.wav").resolve()
    assert statuses.get(str(corrupt)) == "unreadable"

    song = (ravdess_tree_with_anomalies / "Actor_02" / "03-02-03-01-01-01-02.wav").resolve()
    assert statuses.get(str(song)) == "excluded_subset"

    dup_sha = metadata.loc[
        metadata["file_path"].str.contains("Actor_03")
        & (metadata["emotion_id"] == 5)
        & (metadata["statement_id"] == 1),
        "sha1",
    ].iloc[0]
    assert (metadata["sha1"] == dup_sha).sum() == 1


def test_probe_audio_return_shape(ravdess_tree: Path) -> None:
    first = next(ravdess_tree.rglob("*.wav"))
    probe = probe_audio(first)
    assert probe.sampling_rate > 0
    assert probe.num_samples > 0
    assert probe.duration_s == pytest.approx(probe.num_samples / probe.sampling_rate)
    assert len(probe.sha1) == 40


def test_data_config_from_yaml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "data.yaml"
    cfg_file.write_text(
        "data:\n"
        "  data_dir: data/raw/ravdess\n"
        "  output_dir: ~/out\n"
        "  split_ratios: [0.5, 0.25, 0.25]\n"
        "  seed: 7\n",
        encoding="utf-8",
    )
    cfg = DataConfig.from_yaml(cfg_file)
    assert cfg.data_dir == (tmp_path / "data/raw/ravdess").resolve()
    assert cfg.split_ratios == (0.5, 0.25, 0.25)
    assert cfg.seed == 7
    assert "out" in str(cfg.output_dir)


def test_data_config_rejects_bad_ratios(tmp_path: Path) -> None:
    cfg_file = tmp_path / "data.yaml"
    cfg_file.write_text("data:\n  data_dir: x\n  split_ratios: [0.5, 0.5]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split_ratios"):
        DataConfig.from_yaml(cfg_file)


def test_metadata_end_to_end_content(ravdess_tree: Path) -> None:
    """Spot-check decoded label columns against the RAVDESS codebook."""
    metadata, _ = build_speech_metadata(ravdess_tree)

    sample = metadata[(metadata["speaker_id"] == 12) & (metadata["emotion_id"] == 6)]
    assert (sample["gender"] == "female").all()
    assert (sample["emotion"] == "fearful").all()

    neutral = metadata[metadata["emotion_id"] == 1]
    assert (neutral["intensity"] == "normal").all()  # neutral has no strong variant
