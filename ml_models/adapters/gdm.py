"""GDM model adapter (compute-only)."""
from functools import lru_cache
from os.path import join
import pickle

import pandas as pd


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unwrap(value):
    if isinstance(value, list): 
        return value[0] if value else None
    return value


def _divide_array(arr, num_subarrays):
    n = len(arr)
    subarray_size = n // num_subarrays
    remainder = n % num_subarrays
    subarrays = []
    start_index = 0

    for i in range(num_subarrays):
        end_index = start_index + subarray_size + (1 if i < remainder else 0)
        subarrays.append(arr[start_index:end_index])
        start_index = end_index

    return subarrays


@lru_cache(maxsize=1)
def _load_models(model_dir: str):
    model_names = ("lgbm_0.pkl", "lgbm_1.pkl", "lgbm_2.pkl", "lgbm_3.pkl")
    models = []
    for model_name in model_names:
        with open(join(model_dir, model_name), "rb") as file:
            models.append(pickle.load(file))
    return tuple(models)


def _calculate_risk(predict_prob, step, model_dir):
    pred_file_path = join(model_dir, f"GDM_{step}.csv")
    list_pred_real = []

    try:
        with open(pred_file_path) as pred_file:
            for line in pred_file:
                line = line.strip().split(",")
                if len(line) < 2:
                    continue
                list_pred_real.append([float(line[0]), float(line[1])])
    except FileNotFoundError:
        return predict_prob

    if not list_pred_real:
        return predict_prob

    sorted_list_pred_real = sorted(list_pred_real, key=lambda x: x[1])
    sorted_list_pred_real[0][1] = 0
    sorted_list_pred_real[-1][1] = 1.0

    for bin_data in _divide_array(sorted_list_pred_real, 20):
        if not bin_data:
            continue
        if predict_prob < bin_data[-1][1]:
            values = [val[0] for val in bin_data]
            return sum(values) / len(values)

    return predict_prob


def predict(payload: dict, submodule_root: str) -> list:
    """Compute GDM risks from the upstream twin-pregnancy model artifacts."""
    mean_std_path = join(submodule_root, "static", "mean_std_gdm.csv")
    mean_std_df = pd.read_csv(mean_std_path)
    mean_dict = dict(zip(mean_std_df["label"], mean_std_df["Mean"]))

    data_dict = {}
    for label in mean_dict.keys():
        value = _unwrap(payload.get(label))
        if value in (None, ""):
            data_dict[label] = _to_float(mean_dict.get(label, 0.0))
        else:
            data_dict[label] = _to_float(value, mean_dict.get(label, 0.0))

    data_df = pd.DataFrame({key: [value] for key, value in data_dict.items()})

    model_dir = join(submodule_root, "models_gdm")
    pred_results_dir = join(submodule_root, "predicted_results_gdm")
    model_from_step = [0, 1, 2, 3]
    risks = []

    for i, model in enumerate(_load_models(model_dir)):
        if i >= len(model_from_step):
            break

        data_df_ = data_df.copy()
        feature_names = model.feature_name()

        missing_features = [feature for feature in feature_names if feature not in data_df_.columns]
        for feature in missing_features:
            data_df_[feature] = 0.0

        data_df_ = data_df_[feature_names]
        probability = float(model.predict(data_df_)[0])
        risk = _calculate_risk(probability, model_from_step[i], pred_results_dir)
        risks.append(risk)

    return [f"{round(float(risk) * 100, 2)}%" for risk in risks]