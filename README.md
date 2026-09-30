
# Speech Emotion Recognition (RAVDESS)

End-to-end Speech Emotion Recognition (SER) system that classifies the emotional content
of a spoken utterance into one of eight RAVDESS emotion categories
(neutral, calm, happy, sad, angry, fearful, disgust, surprised).

The pipeline covers the full loop: raw RAVDESS audio loading, on-the-fly DSP feature
extraction (log-Mel spectrograms and classical utterance vectors), a speaker-disjoint
data split, classical scikit-learn baselines with feature-group ablations, a PyTorch CNN
trained on log-Mel spectrograms with early stopping and checkpointing on validation
macro-F1, and a dependency-light evaluation harness that runs everywhere (including CI).

## Reference

This project implements the classical (MFCC/spectral/prosodic) and deep (log-Mel CNN)
approaches documented across the codebase and the EDA / feature-engineering notebooks.
A README is the primary entry point: see [Project structure](#project-structure) for where
each piece lives, and [Quick start](#quick-start) for the commands.

## Repository scan

| Layer | Where it lives | Status |
| --- | --- | --- |
| Audio loading | `src/emotion_recognition/features/audio.py` | Implemented, tested |
| DSP (STFT, mel, MFCC, f0, spectral) | `src/emotion_recognition/features/dsp.py` | Implemented, tested (pure NumPy) |
| Feature extraction + pooling | `src/emotion_recognition/features/extractor.py` | Implemented, tested |
| Data preparation + splits | `src/emotion_recognition/data/{prepare,splits}.py` | Implemented, tested |
| Evaluation metrics | `src/emotion_recognition/evaluation/metrics.py` | Implemented, tested |
| Classical baselines | `src/emotion_recognition/models/baseline.py` | Implemented (sklearn, lazy import) |
| Deep model | `src/emotion_recognition/models/cnn.py` | Implemented (PyTorch, guarded import) |
| Training loop | `src/emotion_recognition/training/{dataset,trainer}.py` | Implemented |
| CLI entry points | `scripts/{prepare_data,eda_figures,run_baselines,train}.py` | Implemented |
| Reports / figures | `reports/figures/`, `reports/experiments/` | Generated output |
| Config | `configs/{data,baseline,cnn}.yaml` | Single-file reproducible runs |
| CI | `.github/workflows/ci.yml` | Linux + CPU-torch + macOS |

Notable design constraints, verified in code:

- Raw and derived data are **never committed** (see `.gitignore`): download RAVDESS yourself
  and run the preparation pipeline.
- The DSP chain is implemented from scratch on top of NumPy (with `scipy.fft` for the DCT
  used in MFCCs) instead of `librosa`/`scipy.signal`. `scikit-learn` and `torch` are imported
  lazily or guarded so the package stays importable on hosts where OS application-control
  policies block their compiled extensions (see `emotion_recognition/models/__init__.py` and
  `features/dsp.py` docstrings).
- Everything (training, inference, tests) loads audio through one code path
  (`features/audio.py`), guaranteeing training/serving parity by construction.

## Pipeline

```
data/raw/ravdess/Actor_XX/*.wav
        │  prepare_data.py --config configs/data.yaml
        ▼
metadata.csv (validated, deduped, labelled)  quarantine.csv (rejected files)
        │  worker scripts read metadata.csv
        ▼
┌──────────────────────────────┬────────────────────────────────┐
│ scripts/run_baselines.py     │ scripts/train.py               │
│  extract → stats-pooled vec  │  extract → log-Mel spectrogram │
│  (282-d, mean/std per group) │  (64 mel bins × 251 frames)    │
│  sklearn CV ablations        │  EmotionCNN (GroupNorm)        │
│  one-shot val evaluation     │  early stop on val macro-F1    │
└──────────────────────────────┴────────────────────────────────┘
        │                                │
        ▼                                ▼
reports/experiments/           reports/experiments/cnn_<timestamp>/
  ablations_<ts>.csv             best.pt, last.pt, history.json,
  val_metrics_<ts>.json          config_snapshot.yaml, label_map.json,
  (test_metrics_<ts>.json)       result.json
```

Every experiment run is fully described by a single YAML file
(`configs/data.yaml`, `configs/baseline.yaml`, `configs/cnn.yaml`); relative paths inside a
config are resolved against the config file's directory so configs stay portable across
checkouts.

## Why the DSP is implemented from scratch

`requirements.txt` and the `dsp.py` module docstring document the reasoning:

- `librosa` bundles `numba`, and `scipy.signal` bundles compiled Fortran extensions that OS
  application-control policies may block on some Windows hosts.
- A small, deterministic STFT / mel-filterbank / MFCC / resampler written on top of NumPy is
  all speech feature extraction this project needs, and it runs anywhere.
- The only SciPy import is `scipy.fft` (the DCT used for MFCCs), which is NumPy-backed and
  loads cleanly on restricted hosts.

The resampler is a windowed-sinc polyphase resampler (factor out the `up` subfilters so each
output sample costs one `numtaps` dot product; the coefficients are exactly the ones the
naive filter+decimate approach would use). The mel filterbank is triangular, energy-normalized
per filter, and cached per parameter set. The STFT uses a periodic Hann window and edge-padding.
Fundamental frequency is estimated per frame by normalized autocorrelation with parabolic
interpolation, with frames marked unvoiced (`0 Hz`) when silent or below a periodicity gate.

## Features

`extractor.py` is the single interface between audio and every model. One
`FeatureExtractor` produces a `FeatureBundle`; deep models consume the log-Mel matrix and
classical baselines consume the statistics-pooled vector.

### Default configuration (`FeatureConfig`, from `configs/baseline.yaml` / `cnn.yaml`)

| Parameter | Value |
| --- | --- |
| `target_sr` | 16000 Hz |
| `duration_s` | 4.0 s (fixed window, right-pad / crop) |
| `n_fft` / `hop_length` | 1024 / 256 (~16 ms frames) |
| `n_mels` | 64 |
| `fmax` | 8000 Hz |
| `n_mfcc` | 40 |
| `delta_width` | 9 |
| `f0_min` / `f0_max` | 75 / 400 Hz |
| `rolloff_percent` | 0.85 |
| `normalize_rms` | false (preserves loudness, a prosodic affect cue) |

A 4 s window at 16 kHz with a 256-sample hop produces **251 frames**.

### Feature views per utterance

| Feature | Shape | Consumers |
| --- | --- | --- |
| log-Mel spectrogram | `(64, 251)` | CNN (deep models) |
| MFCC | `(40, 251)` | sequence models, baselines |
| MFCC delta / delta-delta | `(40, 251)` each | sequence models, baselines |
| Spectral stack | `(17, 251)` | baselines |
| Pitch contour | `(1, 251)` | prosodic stats |
| Prosodic statistics | `(8,)` fixed | baselines |

The spectral stack (`SPECTRAL_ROWS = 17`) is zero-crossing rate, RMS energy, spectral
centroid, spectral bandwidth, spectral rolloff, plus a 12-bin pitch-class chroma.

### Statistics pooling (classical baselines)

Frame matrices are collapsed per group with `pool_stats = (mean, std)`, and the fixed-length
prosodic vector is appended unchanged (`pool_utterance`). The default pooled vector is
**282 dimensions**:

```
mfcc 40×2 + delta 40×2 + delta2 40×2 + spectral 17×2 + prosodic 8 = 282
```

`subset_columns` / `group_column_ranges` select which group's columns an ablation uses, so
the feature layout can never drift from what the extractor actually produces.

## Dataset

### RAVDESS (primary)

**Ryerson Audio-Visual Database of Emotional Speech and Song** — speech channel only
(`audio.py` loads stereo and mixes to mono). See `data/README.md` for the authoritative
acquisition and licensing notes.

- 24 professional actors (12 female, 12 male), 8 emotions, 2 statements, 2 intensity levels.
- 16-bit / 48 kHz `*.wav`; the speech audio-only subset is **1,440 files** (60 trials × 24
  actors).
- License: **CC BY-NC-SA 4.0**; citation: Livingstone & Russo (2018), PLoS ONE 13(5):
  e0196391. Download from Zenodo: https://zenodo.org/records/1188976
- Filename encoding: `03-01-EMOT-INT-STMT-REP-ACT.wav` (modality `03` = audio-only,
  vocal channel `01` = speech; emotion codes 01–08; intensity `01` normal / `02` strong with
  **no strong variant for neutral**; odd actor ids = male, even = female).

### Preparation pipeline (`scripts/prepare_data.py`)

Validates every audio file (readable header, non-zero frames, duration within
`[0.5, 30.0] s`), parses labels from filenames, drops SHA-1 duplicates, quarantines rejected
files with reasons, and assigns a deterministic **speaker-disjoint** split. Outputs:

- `data/processed/metadata.csv` — 17 columns: `file_path, file_name, speaker_id, gender,
  emotion, emotion_id, intensity, intensity_id, statement, statement_id, repetition,
  duration_s, sampling_rate, channels, num_samples, sha1, dataset_split`.
- `data/processed/quarantine.csv` — `file_path, status, error`.

### Split design

Splitting happens at the **speaker** level (the dataset has only 24 speakers): within each
gender group, speakers are shuffled by a seeded RNG and split by ratios
`train / val / test = 0.667 / 0.166 / 0.167`. This guarantees (a) no speaker appears in more
than one split (the core anti-leakage check, enforced by `verify_split_disjointness`) and
(b) gender balance across splits. It also means the effective batch size is small — which is
why the CNN uses GroupNorm (see below).

## Models

### Classical baselines (`models/baseline.py`, scikit-learn)

StandardScaler + classifier pipelines over the 282-d pooled vectors, each run across seven
feature ablations (`full`, `mfcc_stack`, `spectral`, `prosodic`, `mfcc`, `delta`, `delta2`):

| Kind | Estimator |
| --- | --- |
| `logreg` | LogisticRegression(C=1.0, max_iter=2000) |
| `svm` | SVC(C=1.0, rbf, gamma=scale, class_weight=balanced) |
| `rf` | RandomForestClassifier(300 trees, min_samples_leaf=2, class_weight=balanced) |

sklearn is imported lazily so the module (and the package) imports on hosts where its
compiled extensions are blocked.

### Deep model (`models/cnn.py`, PyTorch)

`EmotionCNN` is a small 2D CNN over log-Mel spectrograms, channels-first `(B, 1, 64, 251)`:

- Three `Conv2d(3×3, padding=1)` blocks with channels `32 → 64 → 128`, each followed by
  `GroupNorm`, ReLU, `MaxPool2d(2, 2)`, and `Dropout2d` (`0.2 / 0.3 / 0.4`).
- **GroupNorm instead of BatchNorm**: the speaker-disjoint splits leave only a handful of
  batches per epoch, so BatchNorm's running statistics are too noisy and cause validation-loss
  spikes that corrupt early-stopping selection. GroupNorm normalizes per-sample per-group and
  is batch-size independent (documented rationale in the module docstring, with measured
  behavior on the synthetic test fixture).
- Final pooling is over **time only** (`AdaptiveAvgPool2d((freq_keep, 1))`), preserving the
  frequency axis so pitch/formant placement survives, then a flatten + linear head to
  `num_classes = 8`.

The module is import-safe when PyTorch is unavailable: the top-level import only succeeds if
`torch` imports, and callers guard with `TORCH_OK`.

## Training

`scripts/train.py --config configs/cnn.yaml` (or `make train`) runs the full loop:

1. Reads metadata.csv, builds the label map from train emotions (alphabetical order), and
   materializes the train/val `SpeechEmotionDataset`s once at construction (log-Mel tensors,
   ~90 MB float32 for the corpus) so each epoch is a cheap memory lookup.
2. Sets the seed globally (`set_seed`), builds dataloaders, and computes inverse-frequency
   `balanced_class_weights`.
3. Trains with AdamW (`lr=1e-3`, `weight_decay=1e-4`), a `ReduceLROnPlateau` scheduler
   (`factor=0.5`), optional AMP (`device: auto`, `amp: auto` resolves to CUDA-or-CPU /
   CUDA-only), and cross-entropy with class weights.
4. **Selection is on validation macro-F1 only** — the test split is never read by training.
   `ModelCheckpoint` writes `best.pt` on improvement and `last.pt` every epoch;
   `EarlyStopping` (patience 10, min_delta 1e-4) stops when val macro-F1 plateaus.

Each run writes a timestamped `cnn_<timestamp>/` directory under `reports/experiments/`
with `best.pt`, `last.pt`, `history.json`, `config_snapshot.yaml`, `label_map.json`, and
`result.json` (run dir, best epoch, best val macro-F1, epochs run).

## Evaluation

Metrics live in `evaluation/metrics.py` and are deliberately **NumPy-only** (accuracy,
per-class precision/recall/F1/support, macro-F1), so evaluation and model selection run
identically everywhere — including CI hosts that cannot load sklearn. `fit` never drops a
class: classes absent from ground truth contribute F1 0 without raising.

Evaluation workflow:

1. `scripts/run_baselines.py --config configs/baseline.yaml` — stratified 5-fold CV ablations
   on the train split (writes `ablations_<ts>.csv`), then refits the best configuration on all
   train rows and reports a one-shot validation evaluation (writes `val_metrics_<ts>.json`).
2. `scripts/train.py` selects the CNN epoch on validation macro-F1 (see Training).
3. `--evaluate-test` on `run_baselines.py` is reserved for the single, final, post-selection
   run on the test split (`test_metrics_<ts>.json`).

**Evaluation results are not currently included in the repository.** No experiment outputs
are committed (they are gitignored under `reports/experiments/`), and training was not run to
completion as part of this repository's history.

## Inference

**No serving/inference code is currently included in this repository.** There is no `api/`
or `app/` module and no `scripts/evaluate.py` / `scripts/predict.py` (the Makefile declares
`evaluate`, `predict`, `api`, and `app` targets, but the scripts they invoke do not exist
yet). FastAPI / uvicorn and Streamlit are declared in `requirements.txt` as dependencies for
the planned API and demo layers.

The inference contract is nevertheless fixed today and cannot drift:

- Audio is loaded through `emotion_recognition.features.audio.load_mono` (stereo→mono,
  resample to 16 kHz), which is the same code path training uses.
- `FeatureExtractor` is "the single interface between audio and every model" — training,
  inference, and tests all call it, and the identical `FeatureConfig` must be reused to avoid
  training/serving skew (noted explicitly in `configs/cnn.yaml`).
- The CNN checkpoint embeds the `label_map` so predictions map back to emotion names.

## Quick start

```bash
# Setup (Windows: replaces .venv/Scripts/python.exe if you use a Unix venv layout)
make install        # create .venv + runtime deps (requirements.txt)
make dev-install    # editable package + dev deps (pytest, ruff)

# Data (download Audio_Speech_Actors_01-24.zip from Zenodo first — see data/README.md)
make prepare        # data/raw/ravdess -> data/processed/{metadata,quarantine}.csv

# EDA figures -> reports/figures/
make eda

# Classical baselines (CV ablations + one-shot val eval)
make baseline

# Deep model training (validate on val macro-F1 only)
make train ARGS="--config configs/cnn.yaml"

# Quality
make test
make lint
```

Direct invocations are covered at the top of each script's docstring, and the two notebooks
(`notebooks/01_eda.ipynb`, `notebooks/02_feature_engineering.ipynb`) demonstrate the
figures and feature verify-shapes workflow. CI (`.github/workflows/ci.yml`) runs the full
test suite on ubuntu (runtime deps), on a second job with CPU `torch`/`torchaudio` (which
exercises the CNN tests), and on macOS; lint is `ruff check` + `ruff format --check` over
`src`, `tests`, and `scripts`, and tests pin `PYTHONHASHSEED=0`.

```bash
pip install -e ".[dev]"   # if you prefer not to use the Makefile
pytest                    # -q, testpaths=tests, warnings-as-errors
```

## Project structure

```
.
├── .github/workflows/ci.yml     # 3-job CI matrix (lint + tests)
├── configs/                     # data.yaml, baseline.yaml, cnn.yaml (single-file runs)
├── data/README.md               # RAVDESS acquisition, license, cache, filename spec
├── notebooks/
│   ├── 01_eda.ipynb             # class balance, durations, speakers, examples
│   └── 02_feature_engineering.ipynb  # shapes, pooling, per-emotion spectrograms
├── scripts/
│   ├── prepare_data.py          # raw RAVDESS -> metadata.csv / quarantine.csv
│   ├── eda_figures.py           # metadata.csv -> reports/figures/*.png
│   ├── run_baselines.py         # sklearn CV ablations + one-shot val eval
│   └── train.py                 # EmotionCNN training on log-Mel
├── src/emotion_recognition/
│   ├── config.py                # DataConfig / TrainConfig from YAML
│   ├── data/                    # prepare.py, splits.py (speaker-disjoint)
│   ├── evaluation/metrics.py    # NumPy-only accuracy / macro-F1 / per-class
│   ├── features/                # audio.py, dsp.py, extractor.py (+ pooling)
│   ├── models/                  # baseline.py (sklearn), cnn.py (EmotionCNN)
│   ├── training/                # dataset.py, trainer.py (checkpoint/early-stop)
│   └── utils/                   # logging.py, reproducibility.py, figures.py
├── tests/                       # pytest suite (synthetic RAVDESS tree, no download)
├── Makefile                     # install/prepare/eda/baseline/train/test/lint/clean
├── pyproject.toml               # packaging, ruff, pytest config
└── requirements.txt             # runtime deps (see comments for the DSP rationale)
```

`models/` is intentionally not part of the package's public import surface: importing
sklearn from `__init__` would make the whole package unimportable on hosts that block its
compiled extensions, so `emotion_recognition.models.baseline` is imported explicitly by the
scripts instead.

## Tests

The test suite (`tests/`, run by `make test`) builds a synthetic in-memory RAVDESS tree
(`tests/helpers.py`) with unique per-file sine tones so duplicate detection, filtering, and
the speaker-disjoint split are testable without downloading the real dataset. It exercises
the filename parser, splitting, audio probing, DSP/extractor/pooling shapes, metrics, and
(unconditionally when torch is present) the CNN and trainer.