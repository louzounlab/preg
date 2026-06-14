import pickle
import pandas as pd


def predict_risk(sample_dict, model_path):
    """
    Run inference on a single sample.

    Returns:
        target               : target name used during training
        prediction_class     : predicted class (0 or 1)
        probability_class_0  : probability of class 0
        probability_class_1  : probability of class 1

    Interpretation:
        Deliverybefore34w:
            class 1 = delivery before 34 weeks

        Spontaneousdeliverybefore34w:
            class 1 = spontaneous delivery before 34 weeks
    """

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    feature_columns = saved["feature_columns"]
    target = saved["target"]

    # Convert sample to DataFrame
    X = pd.DataFrame([sample_dict])

    # Apply the same preprocessing used during training
    X = pd.get_dummies(X, drop_first=True)

    # Ensure the feature order matches the training data
    X = X.reindex(columns=feature_columns, fill_value=0)

    probabilities = model.predict_proba(X)[0]

    return {
        "target": target,
        "prediction_class": int(probabilities[1] >= 0.5),
        "probability_class_0": float(probabilities[0]),
        "probability_class_1": float(probabilities[1]),
    }


# Example usage

sample = {
    "Number of previous Cesarean Deliveries": 1,
    "Maternal Height": 165,
    "Pre-gestational Weight": 70,
    "Pre-gestational BMI": 25.7,
    "Number of previous Ectopic Pregnancies": 0,
    "Coagulation defects": 0,
    "Number of previous miscarriages or abortions": 1,
    "Maternal age": 32,
    "Hypothyroidism": 0,
    "Diabetes Mellitus type 1": 0,
    "Diabetes Mellitus type 2": 0,
    "Number of Living Children": 2,
    "Nulliparity": 0,
    "Chronic Hypertension": 0
}

result = predict_risk(
    sample,
    "models/xgb_Deliverybefore34w.pkl"
)

print(f"Target: {result['target']}")
print(f"Predicted class: {result['prediction_class']}")
print(f"P(class=0): {result['probability_class_0']:.4f}")
print(f"P(class=1): {result['probability_class_1']:.4f}")