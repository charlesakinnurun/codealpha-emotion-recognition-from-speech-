"""Unit tests for speaker-disjoint split assignment (anti-leakage core)."""

from __future__ import annotations

import pandas as pd
import pytest

from emotion_recognition.data.splits import (
    SPLITS,
    assign_speaker_splits,
    gender_from_actor,
    verify_split_disjointness,
)

DEFAULT_RATIOS = (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0)


def _speakers_up_to(n: int) -> range:
    return range(1, n + 1)


def _gender_of_within(speakers) -> dict[int, str]:
    return {s: gender_from_actor(s) for s in speakers}


def test_assignment_is_deterministic() -> None:
    speakers = _speakers_up_to(24)
    gender_of = _gender_of_within(speakers)
    first = assign_speaker_splits(speakers, gender_of, DEFAULT_RATIOS, seed=7)
    second = assign_speaker_splits(speakers, gender_of, DEFAULT_RATIOS, seed=7)
    assert first == second


def test_assignment_differs_across_seeds() -> None:
    speakers = _speakers_up_to(24)
    gender_of = _gender_of_within(speakers)
    a = assign_speaker_splits(speakers, gender_of, DEFAULT_RATIOS, seed=1)
    b = assign_speaker_splits(speakers, gender_of, DEFAULT_RATIOS, seed=2)
    assert a != b


def test_assignment_covers_every_speaker_once() -> None:
    speakers = _speakers_up_to(24)
    gender_of = _gender_of_within(speakers)
    assignment = assign_speaker_splits(speakers, gender_of, DEFAULT_RATIOS, seed=42)
    assert sorted(assignment.keys()) == sorted(speakers)
    assert set(assignment.values()) == set(SPLITS)


def test_assignment_balance_for_24_speakers() -> None:
    """With 24 speakers the 2/3-1/6-1/6 split gives 16/4/4, gender-balanced."""
    speakers = _speakers_up_to(24)
    gender_of = _gender_of_within(speakers)
    assignment = assign_speaker_splits(speakers, gender_of, DEFAULT_RATIOS, seed=42)

    per_split_gender = {split: {"male": 0, "female": 0} for split in SPLITS}
    for speaker, split in assignment.items():
        per_split_gender[split][gender_of[speaker]] += 1

    assert per_split_gender["train"] == {"male": 8, "female": 8}
    assert per_split_gender["val"] == {"male": 2, "female": 2}
    assert per_split_gender["test"] == {"male": 2, "female": 2}


def test_assignment_requires_val_split() -> None:
    """3 speakers per gender cannot form train+val+test; must raise."""
    speakers = _speakers_up_to(6)
    gender_of = _gender_of_within(speakers)
    with pytest.raises(ValueError, match="Too few"):
        assign_speaker_splits(speakers, gender_of, DEFAULT_RATIOS, seed=0)


def test_verify_disjointness_passes_on_valid_assignment() -> None:
    df = pd.DataFrame(
        {
            "speaker_id": [1, 1, 2, 3],
            "gender": ["male", "male", "female", "male"],
            "dataset_split": ["train", "train", "val", "test"],
        }
    )
    verify_split_disjointness(df)  # must not raise


def test_verify_disjointness_raises_on_speaker_overlap() -> None:
    df = pd.DataFrame(
        {
            "speaker_id": [1, 1, 2, 3],
            "gender": ["male", "male", "female", "male"],
            "dataset_split": ["train", "test", "val", "train"],
        }
    )
    with pytest.raises(ValueError, match="Speaker leakage"):
        verify_split_disjointness(df)


def test_verify_disjointness_raises_on_missing_column() -> None:
    with pytest.raises(ValueError, match="dataset_split"):
        verify_split_disjointness(pd.DataFrame({"speaker_id": [1, 2]}))


def test_verify_disjointness_raises_on_unknown_split_value() -> None:
    df = pd.DataFrame(
        {
            "speaker_id": [1, 2, 3],
            "gender": ["male", "female", "male"],
            "dataset_split": ["train", "val", "train"],
        }
    )
    df.loc[2, "dataset_split"] = "bogus"
    with pytest.raises(ValueError, match="Unknown split"):
        verify_split_disjointness(df)
