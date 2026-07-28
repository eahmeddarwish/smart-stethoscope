# Dataset Label/Filename Fix -- Heartbeat Sounds (PASCAL CHSC)

**TL;DR:** the `set_a.csv` and `set_b.csv` label files distributed with
the popular ["Heartbeat Sounds" Kaggle
mirror](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds) of
the PASCAL Classifying Heart Sounds Challenge do not reliably match the
actual audio filenames on disk. We reverse-engineered the corruption
pattern, rebuilt a correct `fname -> label` mapping for all 832 rows
(176 in set_a, 656 in set_b), and verified every corrected entry against
the real files on disk. The corrected mapping files are published here as
`data/set_a_fixed.csv` and `data/set_b_fixed.csv` -- **no audio is
redistributed**, only the corrected filename/label table, which anyone
can join against their own copy of the dataset.

## Why this matters

If you join `set_a.csv`/`set_b.csv` to the actual `.wav` files by
filename (the obvious, standard thing to do), a large fraction of rows in
`set_b` simply fail to match -- and a smaller but nonzero fraction of
`set_a` rows silently match nothing, so any naive pipeline either crashes
or (worse) silently drops labeled examples without anyone noticing.

## What's broken, exactly

### `set_a.csv` (176 rows)

52 of 176 rows are missing an `Aunlabelledtest` prefix that the actual
on-disk filename carries. Example:

| CSV `fname` | Actual file on disk |
|---|---|
| `set_a/201108222226.wav` | `set_a/Aunlabelledtest__201108222226.wav` |

### `set_b.csv` (656 rows) -- a compound, three-part corruption

1. **Phantom `Btraining_` prefix.** Every training-set row in the CSV has
   a `Btraining_` prefix that does not exist on the actual file.
2. **Inconsistent underscore convention between "plain" and "noisy"
   variants.** The plain recording uses a double underscore
   (`murmur__171_...D.wav`); the noisy version of the *same* recording
   uses a single underscore before a `noisy<label>` tag
   (`murmur_noisymurmur_171_...D.wav`). A handful of CSV rows are
   doubly-corrupted, concatenating both patterns:
   `Btraining_murmur_Btraining_noisymurmur_171_1307971016233_D.wav`.
3. **Unlabeled rows use the wrong prefix convention**, e.g. the CSV has
   `Bunlabelledtest_154_...D.wav` (single underscore) while the real file
   is `Bunlabelledtest__154_...D.wav` (double underscore) -- and several
   of these numeric IDs are shared with an unrelated, differently-labeled
   recording of the same approximate timestamp (e.g. one `normal`-labeled
   file and one unlabeled test file legitimately share the same numeric
   suffix, because they're two separate recordings taken close together).

Examples:

| CSV `fname` | Actual file on disk | Label |
|---|---|---|
| `Btraining_murmur_171_1307971016233_D.wav` | `murmur__171_1307971016233_D.wav` | murmur |
| `Btraining_murmur_Btraining_noisymurmur_171_1307971016233_D.wav` | `murmur_noisymurmur_171_1307971016233_D.wav` | murmur |
| `Bunlabelledtest_154_1306935608852_D.wav` | `Bunlabelledtest__154_1306935608852_D.wav` | (none) |
| `Btraining_normal_154_1306935608852_D.wav` | `normal__154_1306935608852_D.wav` | normal |

## Method used to fix it

A naive "strip the known bad prefix" approach was tried first and only
recovered 312/656 `set_b` rows -- it doesn't handle the underscore
inconsistency or the doubly-corrupted rows. The approach that actually
worked:

1. **Suffix-based matching.** Extract the numeric/timestamp suffix common
   to both the CSV name and the real filename with a prefix-agnostic
   regex (everything from the first digit onward), and build a lookup of
   real files keyed by that suffix. This alone resolved 638/656 `set_b`
   rows and all but 0 of the `set_a` rows needing it.
2. **Tie-breaking for suffix collisions.** Where a suffix matches more
   than one real file (plain vs. noisy variant, or unlabeled vs. labeled
   duplicate), disambiguate using:
   - whether the CSV row's raw filename contains `noisy` -> prefer the
     on-disk candidate that also contains `noisy`, else prefer the one
     that doesn't;
   - whether the CSV label is empty -> prefer the on-disk candidate whose
     name contains `unlabelledtest`.
3. Every resulting mapping was **verified to point at a file that
   actually exists on disk**, and every disk file was checked for being
   referenced by at least one row (zero orphans in the final result).

## Result

| | Total rows | Resolved | Orphan files (unreferenced) |
|---|---|---|---|
| `set_a.csv` | 176 | **176 (100%)** | 0 |
| `set_b.csv` | 656 | **656 (100%)** | 0 |

## How to use the fix

`data/set_a_fixed.csv` and `data/set_b_fixed.csv` each contain:

```
dataset, fname_original, fname_fixed, label, sublabel
```

`fname_original` is exactly what the published Kaggle CSV said;
`fname_fixed` is the path that actually resolves to a real file once you
download the dataset yourself. Join on `fname_fixed`, not on the
original Kaggle `fname` column.

## What we are and are not publishing

We are publishing the corrected *mapping* (filenames + labels only). We
are **not** redistributing the PASCAL/Kaggle audio itself, since it is
covered by the original challenge's own terms -- download it directly
from Kaggle and apply the corrected mapping locally.

---

## بالعربي (ملخص)

اكتشفنا إن ملفات التسميات (`set_a.csv` و `set_b.csv`) المرفقة مع نسخة
Kaggle الشهيرة من داتاسيت "Heartbeat Sounds" فيها أخطاء منهجية في أسماء
الملفات: بادئات وهمية، وعدم اتساق في الخط تحت السفلي بين النسخة العادية
والنسخة المشوشة "noisy"، وتضارب في بعض الأرقام التسلسلية بين ملفات بدون
تصنيف وملفات عليها تصنيف. صلّحنا الاتنين بالكامل (176/176 و656/656) وتأكدنا
برمجياً إن كل صف بيشاور على الملف الصحيح فعلاً على القرص. نشرنا هنا جدول
التصحيح فقط (أسماء ملفات + تصنيفات) من غير ما نعيد نشر أي صوت، لأن حقوق
الداتاسيت الأصلي محفوظة لـ PASCAL/Kaggle.
