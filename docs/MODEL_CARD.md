# Model Card -- Smart Stethoscope Heart Sound Classifier

## Summary

| | |
|---|---|
| Model file | `heart_model_v9_int8.tflite` |
| Task | 4-class audio classification: `normal`, `murmur`, `extrahls`, `artifact` |
| Architecture | Convolutional Neural Network (CNN) over Mel-spectrogram images |
| Framework | TensorFlow -> TensorFlow Lite, INT8 post-training quantization |
| Input | Mel-spectrogram, 64 mel bands x ~47 time frames x 1 channel |
| Training data | PASCAL "Classifying Heart Sounds" Challenge dataset (`set_a` + `set_b`, ~579 recordings) via Kaggle -- see `DATASET_CARD.md` |
| Target hardware | Raspberry Pi (ARM), via `ai_edge_litert` / `tflite_runtime` / `tensorflow.lite` fallback chain |
| Iterations | 9+ trained versions before settling on v9 |

## Intended use

A **screening aid** to flag heart sounds that likely contain a murmur, an
extra heart sound (S3/S4), or a recording artifact, so that a human
(clinician, nurse, or trained operator) can decide whether to escalate to
a cardiologist. It is meant to run on cheap, portable hardware (a
Raspberry Pi + digital stethoscope mic) in settings where a full
echocardiogram isn't the first triage step.

## Out-of-scope / not intended for

- Standalone diagnosis without human review.
- Any regulated clinical deployment -- this model has not been validated
  against a clinical-grade dataset, has not gone through any regulatory
  review (FDA/CE/etc.), and was built as an academic capstone project.
- Populations not represented in the training data (see Dataset Card --
  the PASCAL dataset's demographic coverage is not documented in detail
  by the original challenge organizers).

## Audio pipeline (must match exactly at inference time)

```
Capture @ 44100 Hz (8s buffer)
  -> Downsample to 2000 Hz
  -> Butterworth bandpass 20-950 Hz (4th order) + 50 Hz notch
  -> Noise gate (drop frames below 30% of peak RMS)
  -> Best-window selection: pick the 3-second slice with the most
     REGULAR heartbeat pattern (peak-interval regularity), not just the
     loudest slice -- this avoids picking a cough or stethoscope-friction
     burst as "the interesting part"
  -> Normalize to [-1, 1]
  -> Mel-spectrogram: 64 mel bands, n_fft=512, hop_length=128
```

If you retrain or fine-tune this model, the sample rate, mel band count,
FFT size and hop length are all baked into the trained weights -- change
any of them and you must retrain, not just re-run inference. See
`src/smart_stethoscope/config.py`, which now reads these values from an
optional `heart_config_v9.pkl` so a differently-configured retrain doesn't
require editing source.

## Decision rule: murmur-first (not plain argmax)

```python
murmur_p = probs[MURMUR_IDX]
if murmur_p >= MURMUR_THRESHOLD:      # default 0.50
    pred = "murmur"
else:
    pred = argmax(probs)
```

This is a **deliberate safety choice**, arrived at via a threshold sweep
across the validation set rather than by retraining from scratch. It
trades some overall accuracy for a much lower false-negative rate on the
single class where missing a positive is most dangerous.

## Reported evaluation (validation set, threshold-sweep result)

| Metric | Value |
|---|---|
| Murmur recall | 100% |
| Overall accuracy | 75% |

**Read this carefully:** 100% recall was reached by choosing a
conservative decision threshold, which pushes some non-murmur recordings
into the "murmur" bucket (lower precision) in exchange for never missing
a real murmur in this validation set. This is the right tradeoff for a
*screening* tool, but it means the murmur probability score should be
read as "the model isn't confident this is safe to ignore," not as "the
model is confident this is a murmur." There is no held-out *test* set
result reported separately from the validation set used for threshold
selection -- if you build on this project, adding a genuinely held-out
test split is one of the highest-value next steps (see the repo Roadmap
in `README.md`).

## Known failure modes (found during development)

- **Feature mismatch crashes accuracy, silently.** An earlier deployment
  attempt fed MFCC features to a model trained on Mel-spectrograms; every
  prediction collapsed to `artifact`. There was no error, just wrong
  output -- worth calling out because it's a very easy mistake to
  reintroduce if the feature-extraction code and the model are ever
  edited independently of each other.
- Full VGGish embeddings (288 MB) were evaluated and rejected as too
  large/slow for Raspberry Pi deployment; the current model does not use
  transfer learning from VGGish.
- The `artifact` class is a catch-all for "recording quality was too poor
  to classify" (movement, poor coupling, ambient noise) -- it does not
  correspond to a specific cardiac finding and should never be reported
  to a patient as one.

## Retraining

Training notebooks are not included in this repository (they were run in
Google Colab against the raw dataset). If you retrain: use the corrected
label files in `data/set_a_fixed.csv` and `data/set_b_fixed.csv` (see
`DATASET_LABEL_FIX.md`) rather than the original Kaggle CSVs, which
contain filename/label mismatches that this project discovered and fixed.
