"""Reusable helpers for building a tiny synthetic RAVDESS tree.

Moves the heavy fixture-building logic out of ``conftest.py`` so tests can
import it directly without relying on pytest bootstrapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from emotion_recognition.data.splits import EMOTION_IDS, gender_from_actor

MODALITY = 3  # audio-only
CHANNEL = 1  # speech

# (emotion_id, intensity) combos per actor: neutral has no strong intensity.
EMOTION_INTENSITY_COMBOS: list[tuple[int, int]] = [(1, 1)] + [
    (emotion, intensity) for emotion in range(2, 9) for intensity in (1, 2)
]

# 16 actors -> 8 male (odd) / 8 female (even).
ACTOR_IDS = list(range(1, 17))


def filename_for(actor: int, emotion_id: int, intensity: int, statement: int) -> str:
    """Build a RAVDESS-style audio-only speech filename."""
    return (
        f"{MODALITY:02d}-{CHANNEL:02d}-{emotion_id:02d}-{intensity:02d}-"
        f"{statement:02d}-01-{actor:02d}.wav"
    )


def write_tone(path: Path, sr: int = 22050, duration: float = 0.6, freq: float = 220.0) -> None:
    """Write a short sine-tone WAV file at the requested sample rate."""
    time = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    waveform = 0.5 * np.sin(2.0 * np.pi * freq * time)
    sf.write(str(path), waveform, sr)


def _unique_freq(actor: int, emotion_id: int, intensity: int, statement: int) -> float:
    """Injectively map (actor, emotion, intensity, statement) to a distinct frequency.

    Every fixture file must have unique content so that legitimate recordings
    are never mistaken for duplicates by SHA-1 deduplication.
    """
    key = ((actor * 10 + emotion_id) * 10 + intensity) * 10 + statement
    return 150.0 + 0.05 * key  # max key 16822 -> ~991 Hz, well below Nyquist


def _make_actor_dir(data_dir: Path, actor: int) -> None:
    actor_dir = data_dir / f"Actor_{actor:02d}"
    actor_dir.mkdir(parents=True, exist_ok=True)
    for emotion_id, intensity in EMOTION_INTENSITY_COMBOS:
        statement = (emotion_id + intensity) % 2 + 1
        name = filename_for(actor, emotion_id, intensity, statement)
        write_tone(actor_dir / name, freq=_unique_freq(actor, emotion_id, intensity, statement))


def make_ravdess_tree(root: Path, with_anomalies: bool = False) -> Path:
    """Build a synthetic RAVDESS tree under ``root`` and return the data dir.

    With ``with_anomalies=True`` additional files are written:

    - one corrupt WAV (undecodable payload),
    - one song-channel file (excluded subset),
    - one content-duplicate of an existing utterance.
    """
    data_dir = root / "ravdess"
    for actor in ACTOR_IDS:
        _make_actor_dir(data_dir, actor)

    if with_anomalies:
        corrupt = data_dir / "Actor_01" / "03-01-05-01-02-01-01.wav"
        corrupt.write_bytes(b"RIFF\xff\xff\xff\xffgarbage-not-a-wav")

        song_dir = data_dir / "Actor_02"
        write_tone(song_dir / "03-02-03-01-01-01-02.wav")

        source = data_dir / "Actor_03" / filename_for(3, 5, 1, 1)
        duplicate = data_dir / "Actor_03" / filename_for(3, 5, 1, 2)
        duplicate.write_bytes(source.read_bytes())

    return data_dir


def expected_speech_files() -> int:
    """Valid speech files per actor (neutral x1 + 7 emotions x2 intensities)."""
    return len(EMOTION_INTENSITY_COMBOS)


def expected_speaker_genders() -> dict[int, str]:
    return {actor: gender_from_actor(actor) for actor in ACTOR_IDS}


EMOTION_LABELS = EMOTION_IDS


__all__ = [
    "MODALITY",
    "CHANNEL",
    "EMOTION_INTENSITY_COMBOS",
    "ACTOR_IDS",
    "EMOTION_LABELS",
    "filename_for",
    "write_tone",
    "make_ravdess_tree",
    "expected_speech_files",
    "expected_speaker_genders",
]
