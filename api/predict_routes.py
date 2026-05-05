from pathlib import Path
from flask import Blueprint, jsonify, request

from ml_models.adapters.twin_pe import predict as predict_pe
from ml_models.adapters.twin_fwe import predict as predict_fwe

api = Blueprint('api', __name__, url_prefix='/api')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = str(PROJECT_ROOT / "static")


@api.route('/predict/twin-pe', methods=['POST'])
def predict_twin_pe():
    payload = request.get_json(silent=True) or {}
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twins_pe")
    risks = predict_pe(payload, submodule_root)
    return jsonify({"success": True, "model": "twin-pe", "risks": risks})


@api.route('/predict/twin-fwe', methods=['POST'])
def predict_twin_fwe():
    payload = request.get_json(silent=True) or {}
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twin_fwe")
    ctx = predict_fwe(payload, submodule_root, static_root=STATIC_ROOT)
    return jsonify({
        "success": True,
        "model": "twin-fwe",
        "percentages": ctx["percentage_dict"],
        "zscores": ctx["zscore_dict"],
        "discordance_index": ctx["discordance_index"],
        "highlight_index": ctx["highlight_index"],
        "trend_line": ctx["trend_line"],
        "gaussians": ctx["gaussians"],
        "percentages_csv": ctx["percentages_df"],
        "zscores_csv": ctx["zscores_df"],
    })
