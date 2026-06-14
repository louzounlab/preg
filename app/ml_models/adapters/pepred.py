"""PEPRED (Preeclampsia prediction) model adapter."""
from datetime import date, datetime
from functools import lru_cache
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml_models.pepred_minimal.run_model import PE_Model, processing


_NUMERIC_DEFAULTS = {
    "age": 31.0,
    "wt": 70.4,
    "ht": 164.9,
    "interval": 3.0,
    "last.ga": 40.0,
    "ga": 89.0,
    "pappa": 2.7,
    "plgf": 35.0,
    "utpi": 1.65,
    "map": 86.5,
}


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_or(value, default):
    if value is None or value == "":
        return default
    return str(value)


@lru_cache(maxsize=1)
def _load_features(submodule_root: str) -> pd.DataFrame:
    return pd.read_csv(join(submodule_root, "Features.csv"))


@lru_cache(maxsize=4)
def _load_model(submodule_root: str, binary_case: int, is_full: bool):
    full_or_par = "full" if is_full else "partial"
    saved_model_path = join(submodule_root, "models", f"{full_or_par}_data_case{binary_case}.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    model_dict = torch.load(saved_model_path, map_location=map_location, weights_only=False)

    model = PE_Model(is_full).to(device)
    model.load_state_dict(model_dict["model_state_dict"])
    model.eval()
    return model, device


@lru_cache(maxsize=4)
def _load_train_distribution(submodule_root: str, binary_case: int, is_full: bool):
    full_or_par = "full" if is_full else "partial"
    scores_path = join(submodule_root, "models", f"{full_or_par}_case{binary_case}.csv")
    labels_path = join(submodule_root, "models", "train_labels.csv")

    df = pd.read_csv(scores_path)
    df.columns = ["Score"]
    tag = pd.read_csv(labels_path)
    df = df.sort_values(by=["Score"])
    tag = tag.reindex(df.index)
    scores = np.array(df.iloc[:, -1])
    labels = np.array(tag.iloc[:, int(binary_case) - 1])
    return scores, labels


def _build_features(payload: dict):
    """Port of pepred_minimal/site_service.py form-parsing logic."""
    features = {}

    # Required numeric fields with defaults
    for key in ("age", "wt", "interval", "last.ga"):
        v = _to_float(payload.get(key))
        features[key] = _NUMERIC_DEFAULTS[key] if v is None else v

    # Height (with cm/foot conversion)
    ht_measure = _str_or(payload.get("ht.measure"), "cm")
    raw_ht = payload.get("ht")
    if ht_measure == "foot" and raw_ht and "'" in str(raw_ht):
        parts = str(raw_ht).split("'")
        features["ht"] = float(parts[0]) * 30.48 + float(parts[1]) * 2.54
    else:
        v = _to_float(raw_ht)
        features["ht"] = _NUMERIC_DEFAULTS["ht"] if v is None else v

    # Weight unit
    wt_measure = _str_or(payload.get("wt.measure"), "kg")
    if wt_measure == "pound":
        features["wt"] = features["wt"] * 0.4535

    # Categorical fields
    features["race"] = _str_or(payload.get("race"), "White")
    features["conception"] = _str_or(payload.get("conception"), "Spontaneous")
    features["smoking"] = _str_or(payload.get("smoking"), "No")
    features["FH_PE_grandmother"] = _str_or(payload.get("FH_PE_grandmother"), "No")
    features["Chronic_hypertension"] = _str_or(payload.get("Chronic_hypertension"), "No")
    features["Diabetes"] = _str_or(payload.get("Diabetes"), "No")
    features["SLE"] = _str_or(payload.get("SLE"), "No")
    features["Previous_PE"] = _str_or(payload.get("Previous_PE"), "Nullip")

    test_date = _str_or(payload.get("test-date"), "")
    is_full = bool(test_date)

    if is_full:
        ga_measure = _str_or(payload.get("ga.measure"), "days")
        v = _to_float(payload.get("ga"))
        ga_val = _NUMERIC_DEFAULTS["ga"] if v is None else v
        if ga_measure == "weeks":
            ga_val = ga_val * 7

        for key in ("pappa", "plgf", "utpi", "map"):
            v = _to_float(payload.get(key))
            features[key] = _NUMERIC_DEFAULTS[key] if v is None else v

        features["plgf.machine"] = _str_or(payload.get("plgf.machine"), "Delfia")

        # Adjust ga by elapsed days since the test date
        start = datetime.strptime(test_date.replace("-", "/"), "%Y/%m/%d")
        today = date.today()
        end = datetime.strptime(today.strftime("%d/%m/%Y"), "%d/%m/%Y")
        features["ga"] = ga_val - (end - start).days

    return features, is_full


def _predict_score(features: dict, binary_case: int, is_full: bool, submodule_root: str) -> float:
    features_data = _load_features(submodule_root)
    vec = processing(features, features_data, is_full)
    tensor = torch.FloatTensor(vec)

    model, device = _load_model(submodule_root, binary_case, is_full)
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        tensor = tensor.to(device)
        pred = model(tensor)
        pred = sigmoid(pred).item()
    return pred


def _score_to_percentile_and_risk(score: float, binary_case: int, is_full: bool, submodule_root: str):
    scores, labels = _load_train_distribution(submodule_root, binary_case, is_full)

    percentile = float(len(scores[scores < score]) / len(scores) * 100)

    close_samples = np.argpartition(np.abs(scores - score), 5)[:5]
    window = int(0.02 * len(scores))
    risks = []
    for sample in close_samples:
        lo = max(0, int(sample - window))
        hi = min(len(labels), int(sample + window))
        risks.append(np.mean(labels[lo:hi]))
    risk = float(np.mean(risks) * 100)

    return percentile, risk


def predict(payload: dict, submodule_root: str) -> dict:
    """Run PEPRED model and return percentile + risk for the UI."""
    binary_case = int(_str_or(payload.get("model"), "1"))
    if binary_case not in (1, 3):
        binary_case = 1

    features, is_full = _build_features(payload)
    score = _predict_score(features, binary_case, is_full, submodule_root)
    percentile, risk = _score_to_percentile_and_risk(score, binary_case, is_full, submodule_root)

    return {
        "score": round(score, 3),
        "percentile": round(percentile, 2),
        "risk": round(risk, 2),
        "is_full": is_full,
        "binary_case": binary_case,
    }
