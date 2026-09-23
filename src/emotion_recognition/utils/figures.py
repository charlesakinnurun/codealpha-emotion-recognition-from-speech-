"""Plotting helpers for exploratory data analysis.

The EDA notebook and ``scripts/eda_figures.py`` share this module so figures are
produced once and reused; notebooks only call into reusable Python code.

Every figure answers a specific ML question, not just "what does the data
look like":

- class balance            -> is the label distribution imbalanced?
- duration distribution    -> is the fixed 4 s window appropriate?
- speaker distribution     -> is the speaker-disjoint split balanced?
- emotion x intensity      -> is there a confound we must be aware of?
- waveform / log-mel       -> what do inputs look like per emotion?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering (notebooks/CI/server-safe)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..features.audio import load_mono
from ..features.dsp import log_mel_spectrogram

EMOTION_ORDER = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

PALETTE = "Set2"

_MEL_KWARGS = dict(n_fft=1024, hop_length=256, n_mels=64, fmax=8000)
_LOG_MEL_FIG_CMAP = "magma"


def _style() -> None:
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def plot_class_balance(metadata: pd.DataFrame, dest: Path) -> Path:
    """Stacked counts per emotion and split: how imbalanced is each split?"""
    counts = (
        metadata.groupby(["dataset_split", "emotion"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=EMOTION_ORDER)
        .reindex(["train", "val", "test"])
    )
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, colormap=PALETTE)
    ax.set_xlabel("Dataset split")
    ax.set_ylabel("Number of utterances")
    ax.set_title("Class balance per split (RAVDESS speech)")
    ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    return dest


def plot_duration_distribution(metadata: pd.DataFrame, dest: Path) -> Path:
    """Duration per emotion; informs the fixed window length for feature extraction."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=metadata, x="emotion", y="duration_s", order=EMOTION_ORDER, ax=ax)
    ax.axhline(
        metadata["duration_s"].mean(),
        color="crimson",
        ls="--",
        lw=1.2,
        label=f"mean = {metadata['duration_s'].mean():.2f}s",
    )
    ax.axhline(4.0, color="k", ls="--", lw=1.0, label="feature window (4 s)")
    ax.set_title("Utterance duration by emotion")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Duration (seconds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    return dest


def plot_speaker_distribution(metadata: pd.DataFrame, dest: Path) -> Path:
    """Unique speakers per split and gender: is the split balanced across genders?"""
    speakers = (
        metadata.groupby(["dataset_split", "gender"])["speaker_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(["train", "val", "test"])
    )
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    speakers.plot(kind="bar", ax=ax)
    ax.set_title("Unique speakers per split (speaker-disjoint by design)")
    ax.set_xlabel("Dataset split")
    ax.set_ylabel("Number of speakers")
    ax.legend(title="Gender")
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    return dest


def plot_emotion_intensity_confound(metadata: pd.DataFrame, dest: Path) -> Path:
    """Emotion x intensity counts.

    RAVDESS only records *neutral* at normal intensity; knowing this affects how
    we interpret high-intensity-class results later.
    """
    table = pd.crosstab(metadata["emotion"], metadata["intensity"]).reindex(EMOTION_ORDER)
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Emotion x intensity (utterance counts)")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Emotion")
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    return dest


def _load_example(path: str) -> tuple[np.ndarray, int]:
    return load_mono(path)


def plot_audio_examples(
    metadata: pd.DataFrame, data_dir: Path, dest: Path, emotions: tuple[str, ...] = EMOTION_ORDER
) -> Path:
    """Waveform + log-Mel for one example per emotion (visual input sanity check)."""
    _style()
    n_rows = len(emotions)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.2 * n_rows))

    selected = metadata.groupby("emotion").first()
    for row, emotion in enumerate(emotions):
        sample = selected.loc[emotion]
        candidates = [data_dir / sample["file_name"], Path(sample["file_path"])]
        path = next((p for p in candidates if p.is_file()), candidates[1])
        wave, sr = _load_example(str(path))

        ax_w, ax_mel = axes[row]
        # Waveform.
        time = np.arange(wave.size) / sr
        ax_w.plot(time, wave, lw=0.5, alpha=0.8)
        ax_w.set_xlim(0, time[-1] if time.size else 1.0)
        ax_w.set_title(f"{emotion} — waveform ({sample['duration_s']:.2f}s)")
        ax_w.set_xlabel("time (s)")
        ax_w.set_ylabel("amplitude")

        # Log-Mel spectrogram.
        log_mel = log_mel_spectrogram(wave, sr=sr, **_MEL_KWARGS)
        img = ax_mel.imshow(
            log_mel,
            aspect="auto",
            origin="lower",
            cmap=_LOG_MEL_FIG_CMAP,
            interpolation="nearest",
        )
        ax_mel.set_title(f"{emotion} — log-mel")
        ax_mel.set_xlabel("frame (~16 ms)")
        ax_mel.set_ylabel("mel bin")
        fig.colorbar(img, ax=ax_mel, fraction=0.03, pad=0.02)

    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    return dest


def generate_eda_figures(metadata: pd.DataFrame, data_dir: Path, output_dir: Path) -> list[Path]:
    """Render all EDA figures into ``output_dir`` and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_class_balance(metadata, output_dir / "class_balance.png"),
        plot_duration_distribution(metadata, output_dir / "duration_distribution.png"),
        plot_speaker_distribution(metadata, output_dir / "speaker_distribution.png"),
        plot_emotion_intensity_confound(metadata, output_dir / "emotion_intensity.png"),
        plot_audio_examples(metadata, data_dir, output_dir / "audio_examples.png"),
    ]
    return outputs


__all__ = [
    "EMOTION_ORDER",
    "plot_class_balance",
    "plot_duration_distribution",
    "plot_speaker_distribution",
    "plot_emotion_intensity_confound",
    "plot_audio_examples",
    "generate_eda_figures",
]
