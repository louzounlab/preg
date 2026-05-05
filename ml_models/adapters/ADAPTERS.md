Adapter guidelines for researchers

Goal
- Make it trivial for a researcher to add a model from an upstream repo into this site.
- Keep adapter modules compute-only (no Flask app code, no template rendering).

Where to put adapters
- Place adapter modules under: `ml_models/adapters/`

Required public API

Adapter MUST expose:

  def predict(payload: dict, submodule_root: str) -> list[str]
    - `payload`: flat dict of form values (strings) -> adapter should coerce to floats/ints as needed.
    - `submodule_root`: filesystem path to the cloned upstream repo (string).
    - Return: list of formatted percentages (e.g. ["12.34%", "5.67%", "0.00%"]).

Adding a new model (3 steps)

1. Drop the upstream repo at `ml_models/<your_model>/` (clone or submodule).
2. Add an adapter at `ml_models/adapters/<your_model>.py` exporting `predict(payload, submodule_root)`.
3. Wire two routes in `views/ui_routes.py`:
   - GET to render `templates/<your_model>.html`
   - POST `/process_<your_model>_form` that calls the adapter and re-renders the template with `risks=...`.
   Optionally add a JSON route to `api/predict_routes.py` mirroring the others.

Implementation pattern

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

3. Keep all helper functions compute-only (pure Python) and place them above the `predict` function in the adapter. Examples: normalization, map-to-risk mapping, percentiles. Plotting should not run during web requests.

4. File/CSV paths
- Use the provided `submodule_root` to load `static/mean_std_*.csv`, `models_*` directories, and `predicted_results_*` files. Avoid hard-coded relative paths.

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
- Return a list of formatted percentage strings. The UI template renders them.
- If you need richer output (estimates, images, time-series), return a dict and update the consuming view/template to read it.

7. Testing
- Add a small unit-test that imports the adapter and calls `predict()` with a small payload and `submodule_root` set to the local clone path. Example:

```py
from ml_models.adapters.twin_pe import predict
risks = predict({'Trim_1_MAP':'100','Trim_1_PLGF':'50'}, '/path/to/ml_models/twins_pe')
assert isinstance(risks, list)
```

8. Performance and safety
- Avoid reloading pickles per request (use caching).
- Avoid executing plotting or heavy I/O during request handling; precompute artifacts or run offline.
- Avoid importing Flask or registering routes inside adapter modules.

Converting an upstream app.py
- Copy these parts into the adapter:
  - model loading
  - helper functions (divide_array, calculate_risk, gaussian, etc.)
  - any CSV parsing and normalization logic
- Remove or adapt these parts:
  - Flask app creation and `@app.route` handlers
  - `render_template`, `request.form` usage
  - direct calls to `plt.show()` or saving images during request
