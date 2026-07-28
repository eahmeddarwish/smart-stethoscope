# Dataset Card -- Heartbeat Sounds (PASCAL Classifying Heart Sounds Challenge)

## Source

- **Name:** Classifying Heart Sounds Challenge (2011/2012), PASCAL CHSC
- **Mirror used:** ["Heartbeat Sounds" on Kaggle](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)
- **License:** Per the original PASCAL challenge terms / the Kaggle
  dataset page. **This repository does not redistribute the audio.** See
  "What this repo does and does not include" below.

## Composition

| Subset | Recordings | Source |
|---|---|---|
| `set_a` | 176 | iPhone / iStethoscope Pro app (public contributions, noisier, real-world) |
| `set_b` | 656 | Clinical DigiScope digital stethoscope (cleaner, hospital-recorded) |

Classes present: `normal`, `murmur`, `extrastole`, `artifact` (`set_a`),
plus `extrahls` and unlabeled test recordings depending on subset. This
project's model targets `normal` / `murmur` / `extrahls` / `artifact`.

## What this repo does and does not include

- **Included:** `data/set_a_fixed.csv` and `data/set_b_fixed.csv` --
  corrected label/filename mapping files. These contain only filenames
  and class labels, no audio.
- **Not included:** the `.wav` audio files themselves. Download them
  directly from the Kaggle dataset page linked above (free, requires a
  Kaggle account) and place them under
  `datasets/Heartbeats/set_a/` and `datasets/Heartbeats/set_b/`
  (or wherever `STETHOSCOPE_BASE_DIR` points -- see `README.md`).

## Known data-quality issue: filename/label corruption in the published CSVs

While preparing this project for public release, we found that the
`set_a.csv` and, more severely, `set_b.csv` label files shipped with the
Kaggle mirror do not reliably match the actual on-disk filenames -- a
systematic prefix-corruption bug, not a training bug. Full details,
before/after examples, and the corrected mapping methodology are in
[`DATASET_LABEL_FIX.md`](./DATASET_LABEL_FIX.md).

If you're using this dataset for your own project (independent of this
stethoscope model), we'd encourage cross-checking your label file against
the actual filenames on disk using the same suffix-matching approach
before trusting a `fname` -> `label` join.
