from pathlib import Path
from flask import Blueprint, jsonify, request

from auth.decorators import login_required
from ml_models.adapters.twin_pe import predict as predict_pe
from ml_models.adapters.twin_efw import predict as predict_efw
from ml_models.adapters.preterm import predict as predict_preterm

api = Blueprint('api', __name__, url_prefix='/api')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = str(PROJECT_ROOT / "static")


@api.route('/predict/preterm', methods=['POST'])
@login_required
def predict_preterm_route():
    payload = request.get_json(silent=True) or {}
    submodule_root = str(PROJECT_ROOT / "ml_models" / "preterm_birth")
    results = predict_preterm(payload, submodule_root)
    return jsonify({"success": True, "model": "preterm", "results": results})


@api.route('/predict/twin-pe', methods=['POST'])
@login_required
def predict_twin_pe():
    payload = request.get_json(silent=True) or {}
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twins_pe")
    risks = predict_pe(payload, submodule_root)
    return jsonify({"success": True, "model": "twin-pe", "risks": risks})


@api.route('/predict/twin-efw', methods=['POST'])
@login_required
def predict_twin_efw():
    payload = request.get_json(silent=True) or {}
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twin_efw")
    ctx = predict_efw(payload, submodule_root, static_root=STATIC_ROOT)
    return jsonify({
        "success": True,
        "model": "twin-efw",
        "percentages": ctx["percentage_dict"],
        "zscores": ctx["zscore_dict"],
        "discordance_index": ctx["discordance_index"],
        "highlight_index": ctx["highlight_index"],
        "trend_line": ctx["trend_line"],
        "gaussians": ctx["gaussians"],
        "percentages_csv": ctx["percentages_df"],
        "zscores_csv": ctx["zscores_df"],
    })
