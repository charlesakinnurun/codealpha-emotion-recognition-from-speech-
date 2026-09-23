# Data

Raw datasets are **never committed** to this repository. The pipeline expects you to
download them yourself and place them under `data/raw/`.

## RAVDESS (primary dataset)

**Ryerson Audio-Visual Database of Emotional Speech and Song** (RAVDESS), speech channel only.

- 24 professional actors (12 female, 12 male), 8 emotions, 2 statements,
  2 intensity levels, audio-only `*.wav` files at 16-bit / 48 kHz.
- Speech audio-only subset: **1,440 files** (60 trials x 24 actors).
- License: **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike).
- Citing: Livingstone & Russo (2018), PLoS ONE 13(5): e0196391.
  These projects should also link to the Zenodo record.

### Download

1. Get `Audio_Speech_Actors_01-24.zip` from the Zenodo record:
   https://zenodo.org/records/1188976
   (the file is ~215 MB; there is also a Kaggle mirror).
2. Unzip so that the directory layout matches the expected structure:

   ```
   data/raw/ravdess/
   ├── Actor_01/
   │   ├── 03-01-01-01-01-01-01.wav
   │   ├── 03-01-01-01-01-02-01.wav
   │   └── ...
   ├── Actor_02/
   └── ...
   ```

3. Run the preparation pipeline:

   ```bash
   make prepare
   # or
   python scripts/prepare_data.py --config configs/data.yaml
   ```

This validates every file, parses labels from filenames, detects duplicates
(SHA-1), quarantines corrupted audio, and writes:

- `data/processed/metadata.csv` — one row per valid speech utterance including
  `file_path, speaker_id, gender, emotion, emotion_id, intensity, statement,
  duration_s, sampling_rate, channels, sha1, dataset_split`.
- `data/processed/quarantine.csv` — files that failed validation and their reason.

### Filename encoding (RAVDESS)

`03-01-EMOTION-INTENSITY-STATEMENT-REPETITION-ACTOR.wav`

| Field | Codes |
| --- | --- |
| Modality | `03` = audio-only (we keep only these) |
| Vocal channel | `01` = speech (we keep only these; `02` = song is excluded) |
| Emotion | `01` neutral, `02` calm, `03` happy, `04` sad, `05` angry, `06` fearful, `07` disgust, `08` surprised |
| Intensity | `01` normal, `02` strong (neutral has no strong variant) |
| Statement | `01` "Kids are talking by the door", `02` "Dogs are sitting by the door" |
| Repetition | `01` / `02` |
| Actor | `01`–`24`; odd = male, even = female |

## TESS (optional, future cross-corpus work)

Toronto Emotional Speech Set (TESS): 2 speakers, 7 emotions, ~2,800 files.
Not used in the core pipeline (label set differs and 2 speakers preclude a
speaker-independent split); may be added later as an external evaluation set.