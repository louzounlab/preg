from pathlib import Path
from flask import Blueprint, redirect, render_template, request, url_for

from ml_models.adapters.gdm import predict as predict_gdm
from ml_models.adapters.twin_pe import predict as predict_pe
from ml_models.adapters.twin_fwe import predict as predict_fwe, adjust_trend as adjust_fwe_trend
from ml_models.adapters.pepred import predict as predict_pepred

ui = Blueprint('ui', __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = str(PROJECT_ROOT / "static")


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
    return redirect(url_for('ui.home'))


@ui.route('/PE_Twins')
@ui.route('/twin-pe')
def pe_twins():
    return render_template('twin-pe.html', risks=[])


@ui.route('/GDM')
@ui.route('/gdm')
def gdm():
    return render_template('gdm.html', active='GDM', risks=[])


@ui.route('/twin-fwe')
def twin_fwe():
    return render_template('twin-fwe.html', data={}, percentage_dict={}, zscore_dict={}, last_row=4)


@ui.route('/PEPRED')
@ui.route('/pepred')
def pepred():
    return render_template('pepred.html', results=False)


@ui.route('/process_pe_form', methods=['POST', 'GET'])
def process_pe_form():
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twins_pe")
    try:
        risks = predict_pe(dict(request.form), submodule_root)
        return render_template('twin-pe.html', risks=risks)
    except Exception as exc:
        return render_template('twin-pe.html', risks=[], error=str(exc))


@ui.route('/process_gdm_form', methods=['POST', 'GET'])
def process_gdm_form():
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twins_pe")
    try:
        risks = predict_gdm(dict(request.form), submodule_root)
        return render_template('gdm.html', active='GDM', risks=risks)
    except Exception as exc:
        return render_template('gdm.html', active='GDM', risks=[], error=str(exc))


@ui.route('/process_fwe_form', methods=['POST', 'GET'])
def process_fwe_form():
    submodule_root = str(PROJECT_ROOT / "ml_models" / "twin_fwe")
    form_data = request.form.to_dict()
    last_row = int(form_data.get('last_row', 4) or 4)
    try:
        ctx = predict_fwe(form_data, submodule_root, static_root=STATIC_ROOT)
        return render_template('twin-fwe.html', **ctx)
    except Exception as exc:
        return render_template(
            'twin-fwe.html',
            data=form_data,
            percentage_dict={},
            zscore_dict={},
            discordance_index={},
            highlight_index={},
            last_row=last_row,
            error=str(exc),
        )


@ui.route('/process_pepred_form', methods=['POST', 'GET'])
def process_pepred_form():
    submodule_root = str(PROJECT_ROOT / "ml_models" / "pepred_minimal")
    try:
        ctx = predict_pepred(dict(request.form), submodule_root)
        return render_template('pepred.html', results=True, **ctx)
    except Exception as exc:
        return render_template('pepred.html', results=False, error=str(exc))


@ui.route('/adjust_trend', methods=['POST', 'GET'])
def adjust_trend():
    trend_data_path = request.form.get("trend_data")
    extended_by = int(request.form.get("range") or 1)
    last_row = int(request.form.get("last_row", 4) or 4)
    try:
        ctx = adjust_fwe_trend(trend_data_path, extended_by=extended_by)
        return render_template('twin-fwe.html', **ctx)
    except Exception as exc:
        return render_template(
            'twin-fwe.html',
            data={},
            percentage_dict={},
            zscore_dict={},
            discordance_index={},
            highlight_index={},
            last_row=last_row,
            error=str(exc),
        )
