import sys
from pathlib import Path
from flask import Blueprint, abort, redirect, render_template, request, url_for

from ml_models.registry import get_model_spec, list_model_specs

from ml_models.adapters.twin_pe import predict as predict_pe
from ml_models.adapters.twin_fwe import predict as predict_fwe

# create a blueprint for UI routes
ui = Blueprint('ui', __name__)


@ui.context_processor
def inject_models():
    return {"available_models": list_model_specs()}

@ui.route('/')
def home():
    return render_template('index.html')


@ui.route('/Home')
def home_alias():
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
    return redirect(url_for('ui.home'))


@ui.route('/PE_Twins')
@ui.route('/twin-pe')
def pe_twins():
    return render_template('twin-pe.html', risks=[])

@ui.route('/models/<model_slug>')
def model_page(model_slug: str):
    try:
        spec = get_model_spec(model_slug)
    except KeyError:
        abort(404)

    return render_template(spec.template_name, model=spec)


@ui.route('/GDM')
@ui.route('/twin-fwe')
def gdm():
    return redirect(url_for('ui.model_page', model_slug='twin-fwe'))


@ui.route('/process_pe_form', methods=['POST', 'GET'])
def process_pe_form():
    """Process PE form using minimal adapter."""
    from pathlib import Path
    try:
        submodule_root = str(Path(__file__).resolve().parents[1] / "ml_models" / "twins_pe")
        payload = dict(request.form)
        risks = predict_pe(payload, submodule_root)
        return render_template('twin-pe.html', risks=risks)
    except Exception as exc:
        return render_template('twin-pe.html', risks=[], error=str(exc))


@ui.route('/process_gdm_form', methods=['POST', 'GET'])
def process_gdm_form():
    """Process GDM form using minimal adapter."""
    from pathlib import Path
    try:
        submodule_root = str(Path(__file__).resolve().parents[1] / "ml_models" / "twin_fwe")
        payload = dict(request.form)
        risks = predict_fwe(payload, submodule_root)
        return render_template('gdm.html', risks=risks)
    except Exception as exc:
        return render_template('gdm.html', risks=[], error=str(exc))