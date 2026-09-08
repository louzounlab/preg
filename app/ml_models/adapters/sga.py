"""SGA (small-for-gestational-age) model adapter.

Wraps ``ml_models/sga/sga_model.pkl`` — an XGBoost ``reg:squarederror``
booster over 44 features that scores a mid-trimester (~22 week) scan plus
maternal history.

This model is *not* a risk classifier like the rest of the catalog: it is a
regression that returns a single continuous number, the newborn's
birth-weight normalised for gestational age at delivery (a z-score, observed
range about -5.8 to +3.0). The matching birth-weight centile is derived from
it with the normal CDF, so the UI can show the familiar 0-100 percentile.

Inputs are used unnormalised — no mean/std scaling — so completing a partial
form only needs the training-set column medians shipped alongside the model in
``sga_feature_medians.csv``. Anything the caller leaves blank falls back to
its median, and the fields that did so are reported back in ``defaults_used``.

Two exceptions to that fallback, because the columns concerned are not
independent of one another: BMI is derived from height and weight when those
are supplied, and ``last.bwcent`` / ``last.bwzscore`` — the same quantity in
two units — are derived from each other. Median-filling either half of a
dependent pair would contradict the half the caller did supply.
"""
from functools import lru_cache
from math import erf, sqrt
from os.path import join
from statistics import NormalDist
import pickle

import pandas as pd
import xgboost as xgb


MODEL_FILE = "sga_model.pkl"
MEDIANS_FILE = "sga_feature_medians.csv"

# One-hot groups: submitted field name -> column prefix. The form posts the
# target column name itself (e.g. race=race_White); every other column sharing
# the prefix is zeroed. The training frame carries a few trailing-whitespace
# duplicates ("race_Black_", "dm_Type_1_DM_", "chr_Chronic_hypertension_") —
# those are never selected, only zeroed.
GROUP_PREFIXES = {
    "race": "race_",
    "conception": "conception__",
    "smoking": "smoking_",
    "sle": "sle_",
    "dm": "dm_",
    "chr": "chr_",
    "fh": "fh_",
    "prev.pe": "prev.pe_",
    "Previous_death": "Previous_death_",
    "Previous_IUD": "Previous_IUD_",
}

# Human-readable names, used only for the "filled from median" notice.
FIELD_LABELS = {
    "ga22": "Gestational age at scan",
    "efw22": "Estimated fetal weight at scan",
    "u22": "Uterine artery PI (MoM)",
    "BMI": "BMI",
    "age": "Maternal age",
    "height": "Maternal height",
    "weight_22": "Maternal weight at scan",
    "interval": "Interval since previous pregnancy",
    "last.ga": "GA at previous delivery",
    "last.bwzscore": "Previous birth-weight z-score",
    "last.bwcent": "Previous birth-weight centile",
    "race": "Racial origin",
    "conception": "Method of conception",
    "smoking": "Smoking",
    "sle": "SLE / APS",
    "dm": "Diabetes mellitus",
    "chr": "Chronic hypertension",
    "fh": "Family history of PE",
    "prev.pe": "Previous preeclampsia",
    "Previous_death": "Previous fetal/neonatal death",
    "Previous_IUD": "Previous intrauterine death",
}

# Plausible input ranges. Values outside these are still scored — the booster
# has no opinion about them — but they are reported back in ``warnings`` so a
# caller that skipped the browser's own validation (the JSON API) still hears
# about a mistyped unit or a slipped decimal point.
PLAUSIBLE_RANGES = {
    "ga22": (19.0, 24.0),
    "efw22": (100.0, 1500.0),
    "u22": (0.3, 4.0),
    "age": (12.0, 60.0),
    "height": (120.0, 210.0),
    "weight_22": (35.0, 200.0),
    "BMI": (12.0, 70.0),
    "interval": (0.0, 30.0),
    "last.ga": (20.0, 43.0),
    "last.bwcent": (0.0, 100.0),
    "last.bwzscore": (-6.0, 6.0),
}

# Birth-weight centile cut-offs (customary SGA / LGA definitions).
SEVERE_SGA_CENTILE = 3.0
SGA_CENTILE = 10.0
LGA_CENTILE = 90.0


def _to_float(value, default=None):
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unwrap(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


_NORMAL = NormalDist()


def _z_to_centile(z: float) -> float:
    """Normal CDF, as a 0-100 percentile."""
    return 100.0 * 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _centile_to_z(centile: float) -> float:
    """Inverse of :func:`_z_to_centile`. Clamped just inside 0 and 100, which
    have no finite z."""
    return _NORMAL.inv_cdf(min(max(centile, 0.05), 99.95) / 100.0)


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    with open(model_path, "rb") as f:
        booster = pickle.load(f)
    return booster, tuple(booster.feature_names)


@lru_cache(maxsize=1)
def _load_medians(medians_path: str) -> dict:
    series = pd.read_csv(medians_path, index_col=0).iloc[:, 0]
    return {str(k): float(v) for k, v in series.items()}


@lru_cache(maxsize=1)
def _column_layout(feature_names: tuple):
    """Split the model columns into one-hot groups and plain numeric fields."""
    groups = {}
    grouped = set()
    for field, prefix in GROUP_PREFIXES.items():
        columns = tuple(c for c in feature_names if c.startswith(prefix))
        groups[field] = columns
        grouped.update(columns)
    numeric = tuple(c for c in feature_names if c not in grouped)
    return groups, numeric


def _apply_numeric(payload: dict, row: dict, numeric_columns, defaults_used: list):
    for column in numeric_columns:
        value = _to_float(payload.get(column))
        if value is None:
            defaults_used.append(column)
        else:
            row[column] = value


def _apply_derived_bmi(payload: dict, row: dict, defaults_used: list):
    """BMI is weight / height^2 in the training data; derive it when the form
    supplies the two measurements but leaves BMI blank."""
    if _to_float(payload.get("BMI")) is not None:
        return
    height = _to_float(payload.get("height"))
    weight = _to_float(payload.get("weight_22"))
    if height and weight and height > 0:
        row["BMI"] = weight / (height / 100.0) ** 2
        if "BMI" in defaults_used:
            defaults_used.remove("BMI")


def _apply_derived_birthweight(payload: dict, row: dict, defaults_used: list):
    """``last.bwcent`` and ``last.bwzscore`` are one quantity in two units and
    the booster reads both columns. Median-filling one while the caller supplied
    the other hands the model a contradiction — a previous baby on the 97th
    centile with a z-score of 0.0 — and the two features cancel. Callers
    routinely know only one of the pair, so derive the other rather than
    defaulting it."""
    centile = _to_float(payload.get("last.bwcent"))
    zscore = _to_float(payload.get("last.bwzscore"))

    if centile is not None and zscore is None:
        row["last.bwzscore"] = _centile_to_z(centile)
        if "last.bwzscore" in defaults_used:
            defaults_used.remove("last.bwzscore")
    elif zscore is not None and centile is None:
        row["last.bwcent"] = _z_to_centile(zscore)
        if "last.bwcent" in defaults_used:
            defaults_used.remove("last.bwcent")


def _collect_warnings(payload: dict) -> list:
    """Flag supplied values that fall outside their plausible range."""
    notes = []
    for field, (low, high) in PLAUSIBLE_RANGES.items():
        value = _to_float(payload.get(field))
        if value is not None and not low <= value <= high:
            label = FIELD_LABELS.get(field, field)
            notes.append(f"{label}: {value:g} is outside the expected range {low:g}-{high:g}.")
    return notes


def _apply_groups(payload: dict, row: dict, groups, defaults_used: list):
    for field, columns in groups.items():
        if not columns:
            continue

        # Preferred form: a single select posting the target column name.
        choice = _unwrap(payload.get(field))
        if choice in columns:
            for column in columns:
                row[column] = 0.0
            row[choice] = 1.0
            continue

        # API convenience: raw one-hot columns passed straight through.
        supplied = {c: _to_float(payload.get(c)) for c in columns}
        if any(v is not None for v in supplied.values()):
            for column, value in supplied.items():
                row[column] = 0.0 if value is None else value
            continue

        defaults_used.append(field)


def _classify(centile: float) -> tuple:
    if centile < SEVERE_SGA_CENTILE:
        return "severe-sga", "Severe SGA (below 3rd centile)"
    if centile < SGA_CENTILE:
        return "sga", "SGA (below 10th centile)"
    if centile > LGA_CENTILE:
        return "lga", "LGA (above 90th centile)"
    return "normal", "Appropriate for gestational age"


def predict(payload: dict, submodule_root: str) -> dict:
    """Estimate the birth-weight centile for gestational age.

    Returns a dict with the raw regression output (``zscore``), the centile
    derived from it, a coarse SGA/LGA classification, the fields that were
    completed from the training medians (``defaults_used`` for display,
    ``defaults_used_fields`` for callers that need to branch on them), and any
    supplied values that fell outside their plausible range (``warnings``).
    """
    booster, feature_names = _load_model(join(submodule_root, MODEL_FILE))
    medians = _load_medians(join(submodule_root, MEDIANS_FILE))
    groups, numeric_columns = _column_layout(feature_names)

    row = {column: medians.get(column, 0.0) for column in feature_names}
    defaults_used = []

    _apply_numeric(payload, row, numeric_columns, defaults_used)
    _apply_derived_bmi(payload, row, defaults_used)
    _apply_derived_birthweight(payload, row, defaults_used)
    _apply_groups(payload, row, groups, defaults_used)

    frame = pd.DataFrame([row], columns=list(feature_names))
    zscore = float(booster.predict(xgb.DMatrix(frame))[0])
    centile = _z_to_centile(zscore)
    category, category_label = _classify(centile)

    return {
        "zscore": round(zscore, 3),
        "centile": round(centile, 1),
        "category": category,
        "category_label": category_label,
        "defaults_used": [FIELD_LABELS.get(f, f) for f in defaults_used],
        "defaults_used_fields": defaults_used,
        "warnings": _collect_warnings(payload),
    }
