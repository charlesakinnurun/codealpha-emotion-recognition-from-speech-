"""End-to-end classical-baseline tests on a toy learnable corpus.

scikit-learn is blocked by OS application-control policy on some Windows
hosts, so these tests skip locally and only run where sklearn imports (CI).
The toy corpus maps each emotion to a distinct pure tone, so a pooled-vector
classifier must learn emotion->frequency effortlessly -> the tests verify the
whole extract -> pool -> train -> evaluate chain, not just plumbing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import sklearn  # noqa: F401
except Exception:  # missing, or blocked by an OS App-Control policy on this host
    pytest.skip("scikit-learn unavailable on this host", allow_module_level=True)

from emotion_recognition.data.splits import EMOTION_IDS, assign_speaker_splits, gender_from_actor
from emotion_recognition.evaluation.metrics import accuracy, macro_f1
from emotion_recognition.features.extractor import (
    FeatureConfig,
    group_column_ranges,
    subset_columns,
)
from emotion_recognition.models.baseline import (
    DEFAULT_ABLATIONS,
    build_feature_matrix,
    evaluate_model,
    make_model,
    run_ablations,
)
from tests.helpers import ACTOR_IDS, filename_for, write_tone

CFG = FeatureConfig()
EMOTION_FREQ = {emotion: 120.0 + 40.0 * code for code, emotion in EMOTION_IDS.items()}


def _make_toy_corpus(root: Path) -> tuple[Path, dict[int, str]]:
    data_dir = root / "ravdess"
    speakers = ACTOR_IDS[:8]
    for actor in speakers:
        actor_dir = data_dir / f"Actor_{actor:02d}"
        actor_dir.mkdir(parents=True, exist_ok=True)
        for emotion_id, emotion in EMOTION_IDS.items():
            for intensity in (1, 2):
                for statement in (1, 2):
                    write_tone(
                        actor_dir / filename_for(actor, emotion_id, intensity, statement),
                        freq=EMOTION_FREQ[emotion],
                    )
    assignment = assign_speaker_splits(
        speakers=speakers,
        gender_of={a: gender_from_actor(a) for a in speakers},
        ratios=(0.5, 0.25, 0.25),
        seed=42,
    )
    return data_dir, assignment


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("baselines")
    data_dir, assignment = _make_toy_corpus(root)

    def _paths(split: str) -> tuple[list[Path], list[str]]:
        paths, labels = [], []
        for actor, split_of in assignment.items():
            if split_of != split:
                continue
            actor_dir = data_dir / f"Actor_{actor:02d}"
            for path in sorted(actor_dir.iterdir()):
                paths.append(path)
                parts = path.stem.split("-")
                labels.append(EMOTION_IDS[int(parts[2])])
        return paths, labels

    train_paths, train_labels = _paths("train")
    val_paths, val_labels = _paths("val")
    X_train, y_train = build_feature_matrix(train_paths, train_labels, cfg=CFG)
    X_val, y_val = build_feature_matrix(val_paths, val_labels, cfg=CFG)
    return X_train, y_train, X_val, y_val


def test_build_feature_matrix_shapes_and_determinism(corpus) -> None:
    X_train, y_train, _, _ = corpus
    assert X_train.shape == (X_train.shape[0], 282)
    assert X_train.dtype == np.float32
    assert X_train.shape[0] > 0 and X_train.shape[0] == len(y_train)


def test_column_layout_consistent_with_matrix(corpus) -> None:
    X_train, _, _, _ = corpus
    ranges = group_column_ranges(CFG)
    assert [ranges[g] for g in CFG.pool_groups]
    assert max(r[1] for r in ranges.values()) == X_train.shape[1]
    mfcc_only = subset_columns(X_train, CFG, ("mfcc",))
    assert mfcc_only.shape == (X_train.shape[0], CFG.n_mfcc * 2)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_logreg_separates_tones_on_unseen_speakers(corpus) -> None:
    X_train, y_train, X_val, y_val = corpus
    model = make_model("logreg", seed=42)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_val, y_val)
    assert metrics["macro_f1"] >= 0.9
    assert metrics["accuracy"] >= 0.9


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_all_model_kinds_fit_and_predict(corpus) -> None:
    X_train, y_train, X_val, y_val = corpus
    for kind in ("logreg", "svm", "rf"):
        model = make_model(kind, seed=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        assert preds.shape == (len(y_val),)
        assert macro_f1(y_val, preds) >= 0.9, kind


def test_run_ablations_structure_and_determinism(corpus) -> None:
    X_train, y_train, _, _ = corpus
    a1 = run_ablations(X_train, y_train, CFG, kinds=("logreg",), seed=42, n_splits=3)
    a2 = run_ablations(X_train, y_train, CFG, kinds=("logreg",), seed=42, n_splits=3)
    assert list(a1.columns) == ["model", "groups", "n_features", "macro_f1_mean", "macro_f1_std"]
    assert len(a1) == len(DEFAULT_ABLATIONS)
    assert (a1["macro_f1_mean"] == a2["macro_f1_mean"]).all()
    assert ((a1["macro_f1_mean"] >= 0.0) & (a1["macro_f1_mean"] <= 1.0)).all()
    full = a1.loc[a1["groups"] == "full"]
    assert not full.empty
    assert full.iloc[0]["n_features"] == 282


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_ablations_select_best_and_generalize(corpus) -> None:
    X_train, y_train, X_val, y_val = corpus
    ablations = run_ablations(X_train, y_train, CFG, kinds=("logreg", "svm"), seed=42, n_splits=3)
    best = ablations.loc[ablations["macro_f1_mean"].idxmax()]

    model = make_model(str(best["model"]), seed=42)
    if best["groups"] == "full":
        model.fit(X_train, y_train)
    else:
        model.fit(subset_columns(X_train, CFG, DEFAULT_ABLATIONS[str(best["groups"])]), y_train)

    X_val_sub = (
        X_val
        if best["groups"] == "full"
        else subset_columns(X_val, CFG, DEFAULT_ABLATIONS[str(best["groups"])])
    )
    assert macro_f1(y_val, model.predict(X_val_sub)) >= 0.9
    assert accuracy(y_val, model.predict(X_val_sub)) >= 0.9
