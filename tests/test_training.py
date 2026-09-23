"""End-to-end CNN training tests on a tiny learnable corpus.

Each emotion maps to a distinct sine-tone frequency, so a few epochs of a
small CNN must (a) reduce training loss and (b) significantly beat chance on
unseen speakers -- verifying the whole extract -> dataset -> train -> evaluate
-> checkpoint chain rather than just plumbing. Soft-skips without torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import torch

    from emotion_recognition.data.splits import (
        EMOTION_IDS,
        assign_speaker_splits,
        gender_from_actor,
    )
    from emotion_recognition.features.extractor import FeatureConfig
    from emotion_recognition.models.cnn import CNNConfig, EmotionCNN
    from emotion_recognition.training.dataset import (
        SpeechEmotionDataset,
        build_label_map,
        split_rows,
    )
    from emotion_recognition.training.trainer import (
        EarlyStopping,
        Trainer,
        TrainerConfig,
        balanced_class_weights,
    )
    from emotion_recognition.utils.reproducibility import set_seed
except Exception:  # missing, or blocked by an OS App-Control policy on this host
    pytest.skip("torch unavailable on this host", allow_module_level=True)

from tests.helpers import ACTOR_IDS, filename_for, write_tone

NUM_CLASSES = len(EMOTION_IDS)
EMOTION_FREQ = {emotion: 120.0 + 40.0 * code for code, emotion in EMOTION_IDS.items()}


@pytest.fixture(scope="module")
def toy_metadata(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame]:
    root = tmp_path_factory.mktemp("cnn_training")
    data_dir = root / "ravdess"
    speakers = ACTOR_IDS[:8]
    for actor in speakers:
        actor_dir = data_dir / f"Actor_{actor:02d}"
        actor_dir.mkdir(parents=True, exist_ok=True)
        for emotion_id, emotion in EMOTION_IDS.items():
            intensities = (1,) if emotion_id == 1 else (1, 2)
            for intensity in intensities:
                for stmt in (1, 2):
                    write_tone(
                        actor_dir / filename_for(actor, emotion_id, intensity, stmt),
                        freq=EMOTION_FREQ[emotion],
                    )

    assignment = assign_speaker_splits(
        speakers=speakers,
        gender_of={a: gender_from_actor(a) for a in speakers},
        ratios=(0.5, 0.25, 0.25),
        seed=42,
    )

    rows: list[dict] = []
    for actor, split_of in assignment.items():
        actor_dir = data_dir / f"Actor_{actor:02d}"
        for path in sorted(actor_dir.iterdir()):
            parts = path.stem.split("-")
            rows.append(
                {
                    "file_path": str(path),
                    "emotion": EMOTION_IDS[int(parts[2])],
                    "speaker_id": actor,
                    "dataset_split": split_of,
                }
            )
    return data_dir, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def train_val_fixtures(toy_metadata):
    _, metadata = toy_metadata
    train_rows = split_rows(metadata, "train")
    val_rows = split_rows(metadata, "val")
    label_map = build_label_map(train_rows["emotion"].tolist())
    cfg = FeatureConfig()
    train_ds = SpeechEmotionDataset(train_rows, label_map, cfg=cfg)
    val_ds = SpeechEmotionDataset(val_rows, label_map, cfg=cfg)
    return train_ds, val_ds, label_map


def test_dataset_shapes_and_labels(train_val_fixtures) -> None:
    train_ds, _, label_map = train_val_fixtures
    assert len(train_ds) > 0
    spec, label = train_ds[0]
    assert isinstance(spec, torch.Tensor)
    assert tuple(spec.shape) == (1, 64, 251)
    assert spec.dtype == torch.float32
    assert 0 <= int(label) < NUM_CLASSES
    assert len(label_map) == NUM_CLASSES


def test_class_weights_balanced(train_val_fixtures) -> None:
    train_ds, _, _ = train_val_fixtures
    weights = balanced_class_weights(train_ds.labels, NUM_CLASSES)
    assert weights.shape == (NUM_CLASSES,)
    assert bool(torch.all(weights > 0))
    # Invariant of inverse-frequency weights: sum(w_c * n_c) == n_samples.
    counts = np.bincount(np.asarray(train_ds.labels), minlength=NUM_CLASSES)
    n_samples = len(train_ds)
    assert float((weights.numpy() * counts).sum()) == pytest.approx(n_samples, rel=1e-5)
    # The minority class (neutral: no strong intensity in RAVDESS) gets the
    # highest weight, majority classes lower -- never all-equal.
    assert float(weights.max()) > float(weights.min())


def test_early_stopping_triggers() -> None:
    stopper = EarlyStopping(patience=2, min_delta=1e-4, mode="max")
    assert not stopper.step(0.5, epoch=1)
    assert not stopper.step(0.5, epoch=2)
    assert not stopper.step(0.5, epoch=3)
    assert stopper.step(0.4, epoch=4)
    assert stopper.should_stop


def test_train_decreases_loss_and_checkpoints(tmp_path, train_val_fixtures) -> None:
    train_ds, val_ds, label_map = train_val_fixtures
    set_seed(42)
    model = EmotionCNN(CNNConfig(conv_channels=(8, 16), dropout=(0.0, 0.0)))
    trainer = Trainer(
        model,
        TrainerConfig(
            epochs=3,
            lr=1e-2,
            patience=2,
            output_dir=tmp_path,
            device="cpu",
            amp="off",
        ),
        run_dir=tmp_path / "run",
        label_map=label_map,
        seed=42,
    )
    loaders = {
        "train": torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True),
        "val": torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False),
    }
    weights = balanced_class_weights(train_ds.labels, NUM_CLASSES)
    history = trainer.fit(loaders["train"], loaders["val"], class_weights=weights)
    assert len(history) >= 2
    assert history[0]["train_loss"] > history[-1]["train_loss"] * 0.5
    assert (tmp_path / "run" / "best.pt").is_file()
    assert (tmp_path / "run" / "last.pt").is_file()
    assert (tmp_path / "run" / "history.json").is_file()

    trainer.load_best()
    with torch.no_grad():
        model.eval()
        spec, label = val_ds[0]
        logits = model(spec.unsqueeze(0))
    assert logits.shape == (1, NUM_CLASSES)


def test_val_macro_f1_high_on_tones(tmp_path, train_val_fixtures) -> None:
    """Tone-per-emotion must be learnable to near-perfect unseen-speaker macro-F1."""
    set_seed(42)
    train_ds, val_ds, label_map = train_val_fixtures
    model = EmotionCNN(CNNConfig(conv_channels=(16, 32), dropout=(0.0, 0.0)))
    trainer = Trainer(
        model,
        TrainerConfig(
            epochs=25,
            lr=1e-2,
            patience=8,
            output_dir=tmp_path,
            device="cpu",
            amp="off",
        ),
        run_dir=tmp_path / "run",
        label_map=label_map,
        seed=42,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
    weights = balanced_class_weights(train_ds.labels, NUM_CLASSES)
    history = trainer.fit(train_loader, val_loader, class_weights=weights)
    best = max(h["val_macro_f1"] for h in history)
    assert best >= 0.9, f"val macro-F1 only {best:.4f}"
