"""Preterm-birth (delivery before 34 weeks) model adapter.

Wraps the XGBoost artifacts in ``ml_models/preterm_birth/models``. Each pickle
holds ``{"model", "feature_columns", "target"}``. The adapter is intentionally
generic: it reads ``feature_columns`` and ``target`` from each pickle, so when
the full/real model is delivered it can be dropped into the same folder (same
pickle shape) without touching this code.
"""
from functools import lru_cache
from glob import glob
from os.path import basename, join

import pandas as pd
import pickle


# The two targets currently shipped, with display metadata for the UI. Any
# extra/renamed targets found in the models dir still work — they just fall
# back to the raw target name.
TARGET_LABELS = {
    "Deliverybefore34w": "Delivery before 34 weeks",
    "Spontaneousdeliverybefore34w": "Spontaneous delivery before 34 weeks",
}

# Sensible fall-backs for fields left blank in the form. Binary risk factors
# default to 0 (absent); the rest use representative values.
FEATURE_DEFAULTS = {
    "Number of previous Cesarean Deliveries": 0,
    "Maternal Height": 165.0,
    "Pre-gestational Weight": 70.0,
    "Pre-gestational BMI": 25.7,
    "Number of previous Ectopic Pregnancies": 0,
    "Coagulation defects": 0,
    "Number of previous miscarriages or abortions": 0,
    "Maternal age": 31.0,
    "Hypothyroidism": 0,
    "Diabetes Mellitus type 1": 0,
    "Diabetes Mellitus type 2": 0,
    "Number of Living Children": 0,
    "Nulliparity": 0,
    "Chronic Hypertension": 0,
}


def _to_float(value, default):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=8)
def _load_model(model_path: str):
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], tuple(saved["feature_columns"]), saved["target"]


@lru_cache(maxsize=4)
def _model_paths(model_dir: str):
    return tuple(sorted(glob(join(model_dir, "*.pkl"))))


def _build_features(payload: dict, feature_columns) -> dict:
    """Map form fields (named exactly after the model's columns) to values,
    falling back to FEATURE_DEFAULTS for anything missing or blank."""
    features = {}
    for col in feature_columns:
        features[col] = _to_float(payload.get(col), FEATURE_DEFAULTS.get(col, 0.0))
    return features


def predict(payload: dict, submodule_root: str) -> list:
    """Run every preterm-birth model and return a list of result dicts.

    Each item: {target, label, probability, prediction_class} where
    ``probability`` is the percent risk of the positive class (class 1).
    """
    model_dir = join(submodule_root, "models")
    results = []

    for model_path in _model_paths(model_dir):
        model, feature_columns, target = _load_model(model_path)

        features = _build_features(payload, feature_columns)
        X = pd.DataFrame([features])
        # Match the training preprocessing (no-op while all inputs are numeric,
        # but kept so categorical columns in a future model behave correctly).
        X = pd.get_dummies(X, drop_first=True)
        X = X.reindex(columns=list(feature_columns), fill_value=0)

        prob_class_1 = float(model.predict_proba(X)[0][1])

        results.append({
            "target": target,
            "label": TARGET_LABELS.get(target, target),
            "probability": round(prob_class_1 * 100, 2),
            "prediction_class": int(prob_class_1 >= 0.5),
            "source": basename(model_path),
        })

    return results
