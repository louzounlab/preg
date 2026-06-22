from pathlib import Path
from flask import Blueprint, render_template, request, abort

from auth.decorators import login_required
from ml_models.adapters.gdm import predict as predict_gdm
from ml_models.adapters.twin_pe import predict as predict_pe
from ml_models.adapters.twin_efw import predict as predict_efw, adjust_trend as adjust_efw_trend
from ml_models.adapters.pepred import predict as predict_pepred
from ml_models.adapters.preterm import predict as predict_preterm

ui = Blueprint('ui', __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = str(PROJECT_ROOT / "static")

# Example runs, one entry per model. Each model has its own example page so the
# walk-throughs are not all mixed together on a single page.
EXAMPLES = {
    "pepred": {
        "name": "PE prediction",
        "css": "model-pepred",
        "endpoint": "ui.pepred",
        "description": "Preeclampsia risk from maternal history and 1st trimester biomarkers.",
        "scenarios": [
            {
                "title": "High-risk example",
                "form": "examples/PEPRED/high_risk_form.png",
                "result": "examples/PEPRED/high_risk_result.png",
            },
        ],
    },
    "pe-twins": {
        "name": "PE Twins Prediction",
        "css": "model-pe",
        "endpoint": "ui.pe_twins",
        "description": "Preeclampsia risk in twin pregnancies.",
        "scenarios": [
            {
                "title": "Low-risk example",
                "form": "examples/PE_TWINS/low_risk_form.png",
                "result": "examples/PE_TWINS/low_risk_results.png",
            },
            {
                "title": "High-risk example",
                "form": "examples/PE_TWINS/high_risk_form.png",
                "result": "examples/PE_TWINS/high_risk_results.png",
            },
        ],
    },
    "twin-gdm": {
        "name": "Twin GDM Prediction",
        "css": "model-gdm",
        "endpoint": "ui.gdm",
        "description": "Gestational diabetes mellitus risk in twin pregnancies.",
        "scenarios": [
            {
                "title": "Low-risk example",
                "form": "examples/gdm/low_risk_form.png",
                "result": "examples/gdm/low_risk_results.png",
            },
            {
                "title": "High-risk example",
                "form": "examples/gdm/high_risk_form.png",
                "result": "examples/gdm/high_risk_results.png",
            },
        ],
    },
    "twin-efw": {
        "name": "Twin-EFW",
        "css": "model-twin-efw",
        "endpoint": "ui.twin_efw",
        "description": "Twin fetal weight estimation and discordance prediction.",
        "scenarios": [
            {
                "title": "Both twins severely growth restricted",
                "form": "examples/TWIN_EFW/both_twins_severly_growth_form.png",
                "result": "examples/TWIN_EFW/both_twins_severly_growth_result.png",
            },
            {
                "title": "Severe discordance",
                "form": "examples/TWIN_EFW/Severe_discordance_form.png",
                "result": "examples/TWIN_EFW/Severe_discordance_result.png",
            },
        ],
    },
}


@ui.route('/')
@ui.route('/Home')
def home():
    return render_template('index.html')


@ui.route('/about')
@ui.route('/About')
def about():
    return render_template('about.html')


@ui.route('/glossary')
@ui.route('/Glossary')
def glossary():
    return render_template('glossary.html')


@ui.route('/example')
@ui.route('/Example')
def example():
    return render_template('examples_index.html', models=EXAMPLES)


@ui.route('/example/<model>')
@ui.route('/Example/<model>')
def example_model(model):
    data = EXAMPLES.get(model)
    if data is None:
        abort(404)
    return render_template('example.html', model=data)


@ui.route('/PE_Twins')
@ui.route('/twin-pe')
@login_required
def pe_twins():
    return render_template('twin-pe.html', risks=[])


@ui.route('/GDM')
@ui.route('/gdm')
@login_required
def gdm():
    return render_template('gdm.html', active='GDM', risks=[])


@ui.route('/twin-efw')
@login_required
def twin_efw():
    return render_template('twin-efw.html', data={}, percentage_dict={}, zscore_dict={}, last_row=4)


@ui.route('/PEPRED')
@ui.route('/pepred')
@login_required
def pepred():
    return render_template('pepred.html', results=False)


@ui.route('/preterm')
@ui.route('/PRETERM')
@login_required
def preterm():
    return render_template('preterm.html', results=[])


@ui.route('/process_preterm_form', methods=['POST', 'GET'])
@login_required
def process_preterm_form():
    submodule_root = str(PROJECT_ROOT / "ml_models" / "preterm_birth")
    try:
        results = predict_preterm(dict(request.form), submodule_root)
        return render_template('preterm.html', results=results, data=dict(request.form))
    except Exception as exc:
        return render_template('preterm.html', results=[], error=str(exc))


@ui.route('/process_pe_form', methods=['POST', 'GET'])
@login_required
def process_pe_form():
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twins_pe")
    try:
        risks = predict_pe(dict(request.form), submodule_root)
        return render_template('twin-pe.html', risks=risks)
    except Exception as exc:
        return render_template('twin-pe.html', risks=[], error=str(exc))


@ui.route('/process_gdm_form', methods=['POST', 'GET'])
@login_required
def process_gdm_form():
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twins_pe")
    try:
        risks = predict_gdm(dict(request.form), submodule_root)
        return render_template('gdm.html', active='GDM', risks=risks)
    except Exception as exc:
        return render_template('gdm.html', active='GDM', risks=[], error=str(exc))


@ui.route('/process_efw_form', methods=['POST', 'GET'])
@login_required
def process_efw_form():
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twin_efw")
    form_data = request.form.to_dict()
    last_row = int(form_data.get('last_row', 4) or 4)
    try:
        ctx = predict_efw(form_data, submodule_root, static_root=STATIC_ROOT)
        return render_template('twin-efw.html', **ctx)
    except Exception as exc:
        return render_template(
            'twin-efw.html',
            data=form_data,
            percentage_dict={},
            zscore_dict={},
            discordance_index={},
            highlight_index={},
            last_row=last_row,
            error=str(exc),
        )


@ui.route('/process_pepred_form', methods=['POST', 'GET'])
@login_required
def process_pepred_form():
    submodule_root = str(PROJECT_ROOT / "ml_models" / "pepred_minimal")
    try:
        ctx = predict_pepred(dict(request.form), submodule_root)
        return render_template('pepred.html', results=True, **ctx)
    except Exception as exc:
        return render_template('pepred.html', results=False, error=str(exc))


@ui.route('/adjust_trend', methods=['POST', 'GET'])
@login_required
def adjust_trend():
    trend_data_path = request.form.get("trend_data")
    extended_by = int(request.form.get("range") or 1)
    last_row = int(request.form.get("last_row", 4) or 4)
    try:
        ctx = adjust_efw_trend(trend_data_path, extended_by=extended_by, static_root=STATIC_ROOT)
        return render_template('twin-efw.html', **ctx)
    except Exception as exc:
        return render_template(
            'twin-efw.html',
            data={},
            percentage_dict={},
            zscore_dict={},
            discordance_index={},
            highlight_index={},
            last_row=last_row,
            error=str(exc),
        )
