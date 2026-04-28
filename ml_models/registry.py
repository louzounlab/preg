from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelField:
    name: str
    label: str
    kind: str = "number"
    step: str | None = "any"
    placeholder: str = ""
    options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    title: str
    subtitle: str
    description: str
    accent_start: str
    accent_end: str
    github_url: str
    submodule_path: str
    template_name: str
    api_route_label: str
    adapter_module: str
    order: int
    fields: tuple[ModelField, ...]
    demo_response: dict[str, object]


PROJECT_ROOT = Path(__file__).resolve().parent.parent


MODEL_CATALOG: dict[str, ModelSpec] = {
    "twin-pe": ModelSpec(
        slug="twin-pe",
        title="PE Twins Prediction",
        subtitle="GitHub-linked preeclampsia risk model for twin pregnancies",
        description=(
            "Connected to the Louzoun Lab twin preeclampsia repository so the research "
            "code can be updated independently from this platform."
        ),
        accent_start="#ff6b6b",
        accent_end="#ee5a52",
        github_url="https://github.com/louzounlab/twins_pe",
        submodule_path="ml_models/twins_pe",
        template_name="twin-pe.html",
        api_route_label="twin-pe",
        adapter_module="ml_models.adapters.twin_pe",
        order=1,
        fields=(
            ModelField(name="map", label="MAP", placeholder="Mean arterial pressure"),
            ModelField(name="plgf", label="PLGF", placeholder="Placental growth factor"),
            ModelField(name="cffdna", label="cffDNA", placeholder="Cell-free fetal DNA percentage"),
            ModelField(
                name="b",
                label="B",
                kind="select",
                options=(("", "Select"), ("0", "0"), ("1", "1")),
            ),
            ModelField(
                name="dm",
                label="DM",
                kind="select",
                options=(("", "Select"), ("0", "0"), ("1", "1")),
            ),
            ModelField(
                name="nulliparous",
                label="Nulliparous",
                kind="select",
                options=(("", "Select"), ("0", "0"), ("1", "1")),
            ),
        ),
        demo_response={
            "risk_score": 0.75,
            "status": "High Risk",
            "summary": "Placeholder risk until the linked submodule adapter is swapped in.",
        },
    ),
    "twin-fwe": ModelSpec(
        slug="twin-fwe",
        title="Twin-FWE",
        subtitle="GitHub-linked fetal weight estimation and discordance model",
        description=(
            "Mounted from the Louzoun Lab Twin-FWE repository so longitudinal model "
            "updates can be pulled through Git rather than copied into the site."
        ),
        accent_start="#2ec4b6",
        accent_end="#1b7f78",
        github_url="https://github.com/louzounlab/Twin-FWE",
        submodule_path="ml_models/twin_fwe",
        template_name="twin-fwe.html",
        api_route_label="gdm",
        adapter_module="ml_models.adapters.twin_fwe",
        order=2,
        fields=(
            ModelField(
                name="mcda",
                label="Chorionicity",
                kind="select",
                options=(("1", "MCDA"), ("0", "DCDA")),
            ),
            ModelField(name="week", label="Gestational Week", placeholder="Week of measurement"),
            ModelField(name="day", label="Day", placeholder="Day within the week", step="1"),
            ModelField(name="efw1", label="Twin 1 EFW", placeholder="Estimated fetal weight for twin 1"),
            ModelField(name="efw2", label="Twin 2 EFW", placeholder="Estimated fetal weight for twin 2"),
        ),
        demo_response={
            "discordance_pct": 12.4,
            "status": "Within expected range",
            "summary": "Placeholder discordance result until the linked submodule adapter is connected.",
        },
    ),
}


def get_model_spec(model_slug: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[model_slug]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{model_slug}'") from exc


def list_model_specs() -> list[ModelSpec]:
    return sorted(MODEL_CATALOG.values(), key=lambda spec: spec.order)
