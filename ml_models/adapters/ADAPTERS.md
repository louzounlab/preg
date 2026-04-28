Adapter guidelines for researchers

Goal
- Make it trivial for a researcher to add a model from an upstream repo into this site.
- Keep adapter modules compute-only (no Flask app code, no template rendering).

Where to put adapters
- Place adapter modules under: `ml_models/adapters/` 
Required public API

- Adapter MUST expose one of these two functions (prefer option A):

Option A (recommended, used by UI routes in this project)
- def predict(payload: dict, submodule_root: str) -> list[str]
  - `payload`: flat dict of form values (strings) → adapter should coerce to floats/ints as needed.
  - `submodule_root`: filesystem path to the cloned upstream repo (string).
  - Return: list of formatted percentages (e.g. ["12.34%", "5.67%", "0.00%"]).

Option B (compatible with the service layer)
- def predict(payload: dict, spec: ModelSpec) -> dict
  - `payload`: flat dict of form values
  - `spec`: `ModelSpec` object (from `ml_models.registry`) containing `submodule_path` and metadata
  - Return: full response dict with keys:
    - success (bool)
    - model (slug)
    - title (string)
    - source (string)
    - data (dict) — should include at least `risks` (list of formatted percentages) and `risk_score` (float 0..1)

Implementation pattern (recommended)
1. Copy the original upstream computational helper functions into the adapter module. Remove any Flask app code (routes, render_template, request). Keep only helpers that compute.
2. Add an lru_cache-backed loader for model artifacts (pickles) so models are loaded only once per process:

```py
from functools import lru_cache
from os.path import join
import pickle

@lru_cache(maxsize=1)
def _load_models(model_dir: str):
    models = []
    for name in ("lgbm_0.pkl","lgbm_1.pkl","lgbm_2.pkl"):
        with open(join(model_dir, name), "rb") as f:
            models.append(pickle.load(f))
    return tuple(models)
```

3. Keep all helper functions compute-only (pure Python) and place them above the `predict` function in the adapter. Examples: normalization, map-to-risk mapping, percentiles, plotting helpers (if needed) — but plotting should not run during web requests.

4. File/CSV paths
- Use the provided `submodule_root` (or `spec.submodule_path`) to load `static/mean_std_*.csv`, `models_*` directories, and `predicted_results_*` files. Avoid hard-coded relative paths.

5. Input handling
- Coerce values using a small helper:

```py
def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default
```

- Accept both field name styles if the upstream uses different keys (e.g. `map` vs `Trim_1_MAP`). Map aliases inside the adapter.

6. Output
- For the minimal pattern return a list of formatted percentage strings (Option A). Let the UI template render them.
- If you implement Option B, return the full dict described above to be compatible with `ml_models.service.predict_model()`.

7. Testing
- Add a small unit-test that imports the adapter and calls `predict()` with a small payload and `submodule_root` set to the local clone path. Example test snippet:

```py
from ml_models.adapters.twin_pe import predict
risks = predict({'Trim_1_MAP':'100','Trim_1_PLGF':'50'}, '/path/to/ml_models/twins_pe')
assert isinstance(risks, list)
```

8. Performance and safety
- Avoid reloading pickles per request (use caching).
- Avoid executing plotting or heavy I/O during request handling; precompute artifacts or run offline.
- Avoid importing Flask or registering routes inside adapter modules.

9. Example adapter header (minimal)

```py
# ml_models/adapters/twin_pe.py
import pandas as pd
from functools import lru_cache
from os.path import join
import pickle

@lru_cache(maxsize=1)
def _load_models(model_dir):
    ...

# helper functions copied from upstream (compute-only)

def predict(payload: dict, submodule_root: str) -> list:
    # read mean/std
    # coerce payload values
    # normalize
    # call models
    # map probabilities -> risks
    return ["12.34%", "4.56%", "1.23%"]
```

10. Converting an upstream app.py
- Copy these parts into the adapter:
  - model loading
  - helper functions (divide_array, calculate_risk, gaussian, etc.)
  - any CSV parsing and normalization logic
- Remove or adapt these parts:
  - Flask app creation and `@app.route` handlers
  - `render_template`, `request.form` usage
  - direct calls to `plt.show()` or saving images during request

11. Backwards compatibility
- If you want the adapter to be compatible with both `views` (which call `predict(payload, submodule_root)`) and `ml_models.service.predict_model()` (which expects `predict(payload, spec)`), implement a thin wrapper:

```py
from ml_models.registry import ModelSpec

def predict(payload, spec_or_root):
    if isinstance(spec_or_root, ModelSpec):
        submodule_root = str(Path(__file__).resolve().parents[2] / spec_or_root.submodule_path)
    else:
        submodule_root = spec_or_root
    return predict_compute(payload, submodule_root)


Flexible outputs (important)
---------------------------------
Different models predict different things — probabilities, continuous values, images, time-series, etc. Adapters should be flexible about the `data` they return while still exposing a small stable surface for the UI.

Recommended conventions:

- Keep the adapter `predict` function compute-only and return a serializable object (dict/list) — avoid returning Flask Responses or rendered HTML.
- Preferred shape (most flexible): return a dict with at least these keys when using the service-style API (Option B):

  {
    "success": True,
    "model": "slug",
    "title": "Human title",
    "source": "upstream-repo-or-artifact",
    "data": { ... }    # Arbitrary payload-specific content
  }

  - `data` may contain any keys the UI needs. Examples:
    - classification: `{ "risks": ["12.3%","4.5%"], "risk_score": 0.123 }`
    - regression: `{ "estimate": 3.456, "unit": "kg" }`
    - time-series: `{ "times": ["2026-01-01",...], "values": [0.1, 0.2, ...] }`
    - plots/files: `{ "plots": [{"name":"gauss.png","path":"/static/tmp/gauss.png"}], "summary":"..." }`

- Minimal UI-compatible shape (used by current `views` routes): a list of formatted strings (e.g. percentages) is acceptable for simple cases — but prefer the dict form for anything non-trivial.

Examples
--------

1) Classification adapter (service-style):

```py
def predict(payload, spec):
    probs = [0.12, 0.05, 0.01]
    return {
        "success": True,
        "model": spec.slug,
        "title": spec.title,
        "source": "upstream",
        "data": {
            "risk_score": probs[0],
            "risks": [f"{p*100:.2f}%" for p in probs]
        }
    }
```

2) Regression adapter (estimate):

```py
def predict(payload, submodule_root):
    estimate = model.predict(X)[0]
    return {
        "success": True,
        "model": "efw",
        "title": "Estimated fetal weight",
        "source": "twin_fwe",
        "data": {"estimate": float(estimate), "unit":"g"}
    }
```

3) Adapter producing images/plots:

```py
def predict(payload, submodule_root):
    out_path = generate_plot(payload, submodule_root)
    return {"success": True, "data": {"plots": [{"name":"gauss.png","path": out_path}]}}
```

UI integration notes
--------------------
- The `views` layer should read the adapter output and render appropriately. If `data` contains `plots`, the view can embed image links; if `data` contains `risks`, it can display percentages; if it contains `estimate`, show the numeric result.
- For backward-compatibility, adapters can provide a short `display_values` helper in the module, but the recommended pattern is to keep all presentation decisions in the `views` templates.

Template adapter file
----------------------
- If you want, I can add `ml_models/adapters/TEMPLATE.py` with both API styles implemented as skeletons and a small unit-test example. It will make onboarding even easier for researchers.

```

