from flask import Blueprint, jsonify, request

from ml_models.service import predict_model

api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/v1/predict/<model_slug>', methods=['POST'])
def predict_model_route(model_slug: str):
    payload = request.get_json(silent=True) or {}

    try:
        result = predict_model(model_slug, payload)
    except KeyError:
        return jsonify({"success": False, "error": f"Unknown model '{model_slug}'"}), 404

    return jsonify(result)


@api.route('/predict/twin-pe', methods=['POST'])
def predict_twin_pe():
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_model('twin-pe', payload)
    except KeyError:
        return jsonify({"success": False, "error": "Unknown model 'twin-pe'"}), 404
    return jsonify(result)


@api.route('/predict/twin-fwe', methods=['POST'])
def predict_twin_fwe():
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_model('twin-fwe', payload)
    except KeyError:
        return jsonify({"success": False, "error": "Unknown model 'twin-fwe'"}), 404
    return jsonify(result)
