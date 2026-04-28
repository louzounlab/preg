"""GDM/Twin-FWE compute-only adapter."""
import pickle
from os.path import join
from functools import lru_cache
import pandas as pd


@lru_cache(maxsize=1)
def _load_models(model_dir: str) -> tuple:
    """Load the GDM models."""
    models_names = ("lgbm_0.pkl", "lgbm_1.pkl", "lgbm_2.pkl", "lgbm_3.pkl")
    loaded_models = []
    for model_name in models_names:
        try:
            with open(join(model_dir, model_name), "rb") as f:
                loaded_models.append(pickle.load(f))
        except FileNotFoundError:
            pass
    return tuple(loaded_models)


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


def _calculate_risk(predict_prob, step, model_dir):
    """Map probability to risk using GDM prediction results."""
    pred_file = join(model_dir, f"GDM_{step}.csv")
    list_pred_real = []
    try:
        with open(pred_file) as f:
            for line in f:
                value, prob = line.strip().split(",")
                list_pred_real.append([float(value), float(prob)])
    except FileNotFoundError:
        return predict_prob
    
    if not list_pred_real:
        return predict_prob
    
    sorted_list = sorted(list_pred_real, key=lambda x: x[1])
    sorted_list[0][1] = 0
    sorted_list[-1][1] = 1.0
    
    for bin_data in _divide_array(sorted_list, 20):
        if predict_prob < bin_data[-1][1]:
            vals = [v[0] for v in bin_data]
            return sum(vals) / len(vals)
    
    return predict_prob


def predict(payload: dict, submodule_root: str) -> list:
    """Compute GDM risk prediction."""
    # Load mean/std
    mean_std_df = pd.read_csv(join(submodule_root, "static", "mean_std_gdm.csv"))
    mean_dict = dict(zip(mean_std_df["label"], mean_std_df["Mean"]))
    
    # Build data dict
    data_dict = {}
    for label in mean_dict.keys():
        value = payload.get(label)
        if value in (None, ""):
            data_dict[label] = float(mean_dict[label])
        else:
            try:
                data_dict[label] = float(value)
            except (TypeError, ValueError):
                data_dict[label] = float(mean_dict[label])
    
    # Create DataFrame
    data_df = pd.DataFrame({k: [v] for k, v in data_dict.items()})
    
    # Predict (no normalization for GDM models)
    models = _load_models(join(submodule_root, "models_gdm"))
    pred_results_dir = join(submodule_root, "predicted_results_gdm")
    risks = []
    
    for step, model in enumerate(models):
        data_df_ = data_df.copy()
        feature_names = model.feature_name()
        for feature in feature_names:
            if feature not in data_df_.columns:
                data_df_[feature] = 0.0
        
        prob = float(model.predict(data_df_[feature_names])[0])
        risks.append(_calculate_risk(prob, step, pred_results_dir))
    
    return [f"{round(r * 100, 2)}%" for r in risks]
