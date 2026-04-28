from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from ml_models.registry import PROJECT_ROOT, ModelSpec, get_model_spec, list_model_specs


def _ensure_submodule_path(spec: ModelSpec) -> Path:
    submodule_path = (PROJECT_ROOT / spec.submodule_path).resolve()
    if submodule_path.exists():
        submodule_path_str = str(submodule_path)
        if submodule_path_str not in sys.path:
            sys.path.insert(0, submodule_path_str)
    return submodule_path


def _load_adapter(spec: ModelSpec):
    _ensure_submodule_path(spec)
    try:
        return import_module(spec.adapter_module)
    except ModuleNotFoundError:
        return None


def list_public_models() -> list[ModelSpec]:
    return list_model_specs()


def predict_model(model_slug: str, payload: dict[str, Any]) -> dict[str, Any]:
    spec = get_model_spec(model_slug)
    adapter = _load_adapter(spec)

    if adapter is not None and hasattr(adapter, "predict"):
        return adapter.predict(payload, spec)

    received_payload = {field.name: payload.get(field.name) for field in spec.fields}
    return {
        "success": True,
        "model": spec.slug,
        "title": spec.title,
        "source": "placeholder",
        "github_url": spec.github_url,
        "data": spec.demo_response,
        "received": received_payload,
        "note": "The GitHub submodule path is registered, but no adapter module is wired yet.",
    }
