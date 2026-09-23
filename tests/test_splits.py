"""Unit tests for RAVDESS filename parsing and label encoding."""

from __future__ import annotations

import pytest

from emotion_recognition.data.splits import (
    EMOTION_IDS,
    gender_from_actor,
    parse_ravdess_filename,
)

VALID_NAMES = [
    (
        "03-01-01-01-01-01-24.wav",
        dict(
            modality=3,
            vocal_channel=1,
            emotion="neutral",
            emotion_id=1,
            intensity="normal",
            intensity_id=1,
            statement="kids",
            statement_id=1,
            repetition=1,
            actor=24,
            gender="female",
        ),
    ),
    (
        "03-01-06-01-02-01-12.wav",
        dict(
            modality=3,
            vocal_channel=1,
            emotion="fearful",
            emotion_id=6,
            intensity="normal",
            intensity_id=1,
            statement="dogs",
            statement_id=2,
            repetition=1,
            actor=12,
            gender="female",
        ),
    ),
    (
        "03-01-08-02-02-02-13.wav",
        dict(
            modality=3,
            vocal_channel=1,
            emotion="surprised",
            emotion_id=8,
            intensity="strong",
            intensity_id=2,
            statement="dogs",
            statement_id=2,
            repetition=2,
            actor=13,
            gender="male",
        ),
    ),
]


@pytest.mark.parametrize(("name", "expected"), VALID_NAMES)
def test_parse_valid_filename(name: str, expected: dict) -> None:
    parsed = parse_ravdess_filename(name)
    assert parsed.modality == expected["modality"]
    assert parsed.vocal_channel == expected["vocal_channel"]
    assert parsed.emotion == expected["emotion"]
    assert parsed.emotion_id == expected["emotion_id"]
    assert parsed.intensity == expected["intensity"]
    assert parsed.intensity_id == expected["intensity_id"]
    assert parsed.statement == expected["statement"]
    assert parsed.statement_id == expected["statement_id"]
    assert parsed.repetition == expected["repetition"]
    assert parsed.actor == expected["actor"]
    assert parsed.gender == expected["gender"]
    assert parsed.speaker_id == expected["actor"]


@pytest.mark.parametrize(
    "name",
    [
        "hello.wav",
        "03-01-06-01-02-01.wav",  # too few parts
        "03-01-06-01-02-01-12-99.wav",  # too many parts
        "03-01-99-01-02-01-12.wav",  # unknown emotion code
        "03-01-06-01-02-01-12.mp4",
        "03-01-06-01-02-01-WHAT.wav",
    ],
)
def test_parse_invalid_filename_raises(name: str) -> None:
    with pytest.raises(ValueError):
        parse_ravdess_filename(name)


def test_emotion_codebook_has_eight_classes() -> None:
    assert EMOTION_IDS == {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised",
    }


def test_gender_convention_odd_male_even_female() -> None:
    assert gender_from_actor(1) == "male"
    assert gender_from_actor(13) == "male"
    assert gender_from_actor(2) == "female"
    assert gender_from_actor(24) == "female"
