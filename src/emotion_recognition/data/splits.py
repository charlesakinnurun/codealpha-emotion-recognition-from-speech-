"""RAVDESS filename parsing and speaker-disjoint dataset splitting.

Documented RAVDESS conventions (verified against the Zenodo record):

- Filename: ``MM-VC-EMOT-INT-STMT-REP-ACT.wav``
- Modality: 01 full-AV, 02 video-only, 03 audio-only
- Vocal channel: 01 speech, 02 song
- Emotion: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful,
  07 disgust, 08 surprised
- Intensity: 01 normal, 02 strong (no strong for *neutral*)
- Statement: 01 "Kids are talking by the door", 02 "Dogs are sitting by the door"
- Repetition: 01 / 02
- Actor: 01-24; odd = male, even = female
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import pandas as pd

FILENAME_RE = re.compile(
    r"^(?P<modality>\d{2})-(?P<vocal_channel>\d{2})-(?P<emotion_id>\d{2})-"
    r"(?P<intensity_id>\d{2})-(?P<statement_id>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})\.wav$"
)

EMOTION_IDS: dict[int, str] = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}
EMOTION_TO_ID: dict[str, int] = {name: code for code, name in EMOTION_IDS.items()}

INTENSITY_IDS: dict[int, str] = {1: "normal", 2: "strong"}
STATEMENT_IDS: dict[int, str] = {1: "kids", 2: "dogs"}

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class RavdessParsed:
    """Attributes decoded from a RAVDESS audio-only ``*.wav`` filename."""

    modality: int
    vocal_channel: int
    emotion_id: int
    intensity_id: int
    statement_id: int
    repetition: int
    actor: int
    emotion: str
    intensity: str
    statement: str
    gender: str

    @property
    def speaker_id(self) -> int:
        """RAVDESS actor id, used as the speaker group key."""
        return self.actor


def gender_from_actor(actor: int) -> str:
    """Return 'male'/'female' per RAVDESS: odd actor ids are male, even female."""
    return "male" if actor % 2 == 1 else "female"


def parse_ravdess_filename(name: str) -> RavdessParsed:
    """Parse a RAVDESS filename such as ``03-01-06-01-02-01-12.wav``.

    Raises:
        ValueError: if the name does not match the documented 7-part encoding.
    """
    match = FILENAME_RE.match(name.lower())
    if match is None:
        raise ValueError(f"Not a valid RAVDESS filename: {name!r}")

    actor = int(match.group("actor"))
    emotion_id = int(match.group("emotion_id"))
    if emotion_id not in EMOTION_IDS:
        raise ValueError(f"Unknown RAVDESS emotion code {emotion_id} in {name!r}")

    intensity_id = int(match.group("intensity_id"))
    statement_id = int(match.group("statement_id"))
    return RavdessParsed(
        modality=int(match.group("modality")),
        vocal_channel=int(match.group("vocal_channel")),
        emotion_id=emotion_id,
        intensity_id=intensity_id,
        statement_id=statement_id,
        repetition=int(match.group("repetition")),
        actor=actor,
        emotion=EMOTION_IDS[emotion_id],
        intensity=INTENSITY_IDS.get(intensity_id, f"unknown-{intensity_id}"),
        statement=STATEMENT_IDS.get(statement_id, f"unknown-{statement_id}"),
        gender=gender_from_actor(actor),
    )


def _ratio_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split ``total`` items into train/val/test by ``ratios`` (test takes the remainder)."""
    train = int(total * ratios[0])
    val = int(total * ratios[1])
    test = total - train - val
    return train, val, test


def assign_speaker_splits(
    speakers: pd.Series, gender_of: dict[int, str], ratios: tuple[float, float, float], seed: int
) -> dict[int, str]:
    """Deterministically assign each speaker to a split, balancing by gender.

    The dataset is small (24 speakers), so splitting is done at the *speaker*
    level. Speakers are shuffled within each gender group and then split by
    ``ratios``. This guarantees (a) no speaker appears in more than one split
    and (b) gender balance across train/val/test.

    Args:
        speakers: Unique speaker (actor) ids.
        gender_of: Mapping speaker id -> 'male' | 'female'.
        ratios: (train, val, test) proportions, summing to 1.
        seed: RNG seed; identical seeds produce identical assignments.

    Returns:
        Mapping speaker id -> one of 'train' | 'val' | 'test'.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {ratios}.")

    rng = random.Random(seed)
    assignment: dict[int, str] = {}

    for gender in ("male", "female"):
        group = sorted(s for s in speakers if gender_of[s] == gender)
        rng.shuffle(group)
        n = len(group)
        n_train, n_val, n_test = _ratio_counts(n, ratios)
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                f"Too few {gender} speakers ({n}) to form non-empty train/val/test "
                f"partitions with ratios {ratios}. Use more speakers or adjust ratios."
            )
        for speaker, split in zip(
            group,
            ["train"] * n_train + ["val"] * n_val + ["test"] * n_test,
            strict=True,
        ):
            assignment[speaker] = split

    return assignment


def verify_split_disjointness(df: pd.DataFrame) -> None:
    """Assert that no speaker appears in more than one split.

    This is the core anti-leakage check for the speaker-aware split.

    Raises:
        ValueError: if the split is invalid (missing column, unknown value,
            or speaker overlap between any two splits).
    """
    if "dataset_split" not in df.columns:
        raise ValueError("Metadata is missing the 'dataset_split' column.")
    if "speaker_id" not in df.columns:
        raise ValueError("Metadata is missing the 'speaker_id' column.")

    bad = set(df["dataset_split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"Unknown split values present: {sorted(bad)}")

    per_split = {s: set(df.loc[df["dataset_split"] == s, "speaker_id"]) for s in SPLITS}

    for split, speakers in per_split.items():
        if not speakers:
            raise ValueError(f"Split {split!r} contains no speakers.")

    a, b, c = per_split["train"], per_split["val"], per_split["test"]
    if a & b or a & c or b & c:
        overlap = sorted((a & b) | (a & c) | (b & c))
        raise ValueError(f"Speaker leakage detected between splits: {overlap}")

    total = len(a) + len(b) + len(c)
    if total != len(a | b | c):
        raise ValueError(
            "Coverage accounting inconsistent; every speaker must appear exactly once."
        )


__all__ = [
    "FILENAME_RE",
    "EMOTION_IDS",
    "EMOTION_TO_ID",
    "INTENSITY_IDS",
    "STATEMENT_IDS",
    "SPLITS",
    "RavdessParsed",
    "gender_from_actor",
    "parse_ravdess_filename",
    "assign_speaker_splits",
    "verify_split_disjointness",
]
