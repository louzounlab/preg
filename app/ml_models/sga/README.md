# SGA — birth-weight centile for gestational age

Artifacts
- `sga_model.pkl` — a pickled `xgboost.core.Booster` (`reg:squarederror`, 100 trees,
  44 features).
- `sga_feature_medians.csv` — the training-set median of every one of those 44
  columns, indexed by column name.

What it returns

This is a **regression**, not a risk classifier. The booster emits the newborn's
birth weight normalised for the gestational week of birth — a z-score, observed
range roughly `-5.8` to `+3.0`. The adapter converts it to the familiar 0–100
birth-weight centile with the normal CDF and reports both numbers.

Missing data

The features are fed to the model **unnormalised** (no mean/std scaling), so a
partial form only needs the medians to be completed: any field the caller leaves
blank falls back to its value in `sga_feature_medians.csv`, and the adapter
returns the list of fields that did so in `defaults_used` (plus the raw column
names in `defaults_used_fields`).

Two pairs of columns are **not** median-filled, because they are not independent
of each other and defaulting one would contradict the other:

- `BMI` is derived from `height` and `weight_22` when those are supplied.
- `last.bwcent` and `last.bwzscore` are the same quantity in two units, and the
  booster reads both. A caller usually knows only one, so the adapter derives
  the other through the normal CDF. Before this was added, supplying only the
  centile moved the result by up to 22 centile points against supplying both.

The booster has **no missing-value handling at all** — across the 463 splits on
the four previous-delivery columns, not one has a default direction distinct
from its yes-branch, which is the signature of a training frame with no NaNs.
Passing `NaN` is therefore not a supported input path. The practical consequence
is that *a nulliparous woman cannot be represented*: leaving the previous-delivery
fields blank asserts an earlier delivery at 40 weeks on the 50th centile, and
`last.bwcent` is the model's strongest single feature by gain. The adapter reports
those fields in `defaults_used` and the page raises a visible warning, but the
correct fill value is unknown from the shipped artifacts. **Worth confirming with
the model's author what the training frame held in `interval`, `last.ga`,
`last.bwcent` and `last.bwzscore` for first pregnancies.**

Out-of-range input

The booster scores whatever it is given — an EFW mistyped in kilograms (0.478)
returns a confident "severe SGA, 0th centile". Values outside `PLAUSIBLE_RANGES`
are still scored, but come back in a `warnings` list so the JSON API, which never
sees the form's own `min`/`max`, still hears about a slipped decimal point.

Feature notes
- `ga22` / `efw22` / `u22` — gestational age (weeks), estimated fetal weight (g)
  and uterine artery PI (MoM) at the ~22-week scan. The `u22` median of exactly
  `1.0` is what identifies it as a multiple-of-median rather than a raw PI.
- `BMI` = `weight_22 / (height / 100) ** 2` in the training data, so the adapter
  derives it when the form supplies height and weight but not BMI.
- The categorical features are one-hot encoded with the prefixes `race_`,
  `conception__`, `smoking_`, `sle_`, `dm_`, `chr_`, `fh_`, `prev.pe_`,
  `Previous_death_`, `Previous_IUD_`.
- Three columns are trailing-whitespace duplicates carried over from the training
  frame: `race_Black_`, `dm_Type_1_DM_`, `chr_Chronic_hypertension_`. The adapter
  never sets them, it only zeroes them along with the rest of their group.
  `race_Black_` is inert (zero cover, never split on), but **`dm_Type_1_DM_` is
  not**: it is the 10th strongest feature by gain (187, 14 splits) and is split
  on at nodes covering ~4.7x more training weight than `dm_Type_1_DM`, the column
  the form actually sets. The two encodings carry very different effects — from
  the same baseline, `dm_Type_1_DM=1` gives z +0.751 while `dm_Type_1_DM_=1`
  gives z +1.421 — so the form always picks the weaker of the two and understates
  Type 1 diabetes. This looks like a dirty categorical (`"Type 1 DM"` vs
  `"Type 1 DM "`) split across two one-hot columns during training. Fixing it
  properly means asking the model's author which encoding was intended; it
  affects well under 1% of users and the direction stays correct, so the adapter
  leaves it alone for now rather than silently changing results for that group.

Wiring
- Adapter: `ml_models/adapters/sga.py`
- Page: `templates/sga.html` at `/sga` (form posts to `/process_sga_form`)
- JSON: `POST /api/predict/sga`
