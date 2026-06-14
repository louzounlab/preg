import os
import pickle
from zipfile import ZipFile
from os.path import basename
import shutil
import torch
import numpy as np
import  run_model
import pandas as pd
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, send_file
)
from datetime import datetime, timedelta
from datetime import date
import uuid
from pathlib import Path
from datetime import datetime

bp = Blueprint('/', __name__, url_prefix='/')

input_dir = "static/input/"
output_dir = "static/output/"
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

##
# load the models premanently if wanted
##

def get_float(input):
    if not input:
        return None
    else:
        try:
            return float(input)
        except:
            raise -1

@bp.route('/Home', methods=('GET', 'POST'))
def home_page():
    error = None

    if request.method == 'POST':
        features_dictonary = {}
        is_full = 0
        try:
            model = int(str(request.form.get('model')))
            features_dictonary['age'] = get_float(request.form['age'])
            if features_dictonary['age'] == None:
                features_dictonary['age'] = 31
            features_dictonary['ht'] = None
            features_dictonary['wt'] = get_float(request.form['wt'])
            if features_dictonary['wt'] == None:
                features_dictonary['wt'] = 70.4
            features_dictonary['interval'] = get_float(request.form['interval'])
            if features_dictonary['interval'] == None:
                features_dictonary['interval'] = 3
            features_dictonary['last.ga'] = get_float(request.form['last.ga'])
            if features_dictonary['last.ga'] == None:
                features_dictonary['last.ga'] = 40
            ht_measure = str(request.form.get('ht.measure'))
            wt_measure = str(request.form.get('wt.measure'))
            ga_measure = str(request.form.get('ga.measure'))
            if ht_measure == 'foot':
                features_dictonary['ht']= float(str(request.form['ht']).split('\'')[0]) * 30.48 +float(str(request.form['ht']).split('\'')[1])  * 2.54
            else:
                features_dictonary['ht'] = get_float(request.form['ht'])
                if features_dictonary['ht'] == None:
                    features_dictonary['ht'] = 164.9
            if wt_measure  == 'pound':
                features_dictonary['wt']=features_dictonary['wt'] * 0.4535

            features_dictonary['race'] =  str(request.form.get('race'))
            features_dictonary['conception'] =  str(request.form.get('conception'))
            features_dictonary['smoking'] =  str(request.form.get('smoking'))
            features_dictonary['FH_PE_grandmother'] =  str(request.form.get('FH_PE_grandmother'))
            features_dictonary['Chronic_hypertension'] =str(request.form.get('Chronic_hypertension'))
            features_dictonary['Diabetes'] =  str(request.form.get('Diabetes'))
            features_dictonary['SLE'] =  str(request.form.get('SLE'))
            features_dictonary['Previous_PE'] =  str(request.form.get('Previous_PE'))

            test_date =  str(request.form.get('test-date'))
            if test_date != '':
                features_dictonary['ga'] = get_float(request.form['ga'])
                if features_dictonary['ga'] == None:
                    features_dictonary['ga'] = 89
                if ga_measure == 'weeks':
                    features_dictonary['ga'] = features_dictonary['ga'] * 7
                features_dictonary['pappa'] = get_float(request.form['pappa'])
                if features_dictonary['pappa'] == None:
                    features_dictonary['pappa'] = 2.7
                features_dictonary['plgf'] = get_float(request.form['plgf'])
                if features_dictonary['plgf'] == None:
                    features_dictonary['plgf'] = 35
                features_dictonary['utpi'] = get_float(request.form['utpi'])
                if features_dictonary['utpi'] == None:
                    features_dictonary['utpi'] = 1.65
                features_dictonary['map'] = get_float(request.form['map'])
                if features_dictonary['map'] == None:
                    features_dictonary['map'] = 86.5
                features_dictonary['plgf.machine'] = str(request.form.get('plgf.machine'))

                start = datetime.strptime(test_date.replace('-','/'), "%Y/%m/%d")
                today = date.today()
                end = datetime.strptime(today.strftime("%d/%m/%Y"), "%d/%m/%Y")
                features_dictonary['ga'] =  features_dictonary['ga'] - ((end - start)).days
                is_full = 1

            error = ""
        except:
            if test_date == '':
                error = 'please insert test date'
            else:
                error = 'Not a numeric input'

        if not error:
            result = run_model.main(features_dictonary, model, is_full)
        else:
            flash(error)

        if error:
            if 'not_site' in request.form:
                return {'percentile':error,'risk':error}
            return render_template('Home_new.html', error_message=error, results=False)
        else:
            if is_full:
                df = pd.read_csv('models/full_case'+str(model)+'.csv')
            else:
                df = pd.read_csv('models/partial_case'+str(model)+'.csv')

            df.columns = ['Score']
            tag = pd.read_csv('models/train_labels.csv')
            df = df.sort_values(by=['Score'])
            tag = tag.reindex(df.index)
            scores = np.array(df.iloc[:, -1])
            precentege = len(np.array(scores)[np.array(scores) < result]) / len(scores) * 100
            close_samples = np.argpartition(np.abs(np.array(df.iloc[:, -1] - result)), 5)[:5]
            risks = []
            for sample in close_samples:
                risks.append(
                    np.mean(np.array(tag.iloc[:,int(model)-1][int(sample - 0.02 * df.shape[0]):int(sample + 0.02 * df.shape[0])])))
            risk = np.mean(risks) * 100

            ##saving the id in our database
            id = uuid.uuid1()
            database_dict_path = Path("dict.pkl")
            if database_dict_path.is_file():
                with open('dict.pkl', 'rb') as f:
                    database_dict = pickle.load(f)
            else:
                database_dict = {}
            features_dictonary['score']=round(result, 3)
            features_dictonary['precentege'] = round(precentege, 2)
            features_dictonary['risk'] = round(risk, 2)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            features_dictonary['time'] = dt_string
            features_dictonary['ip'] = request.remote_addr
            database_dict[str(id)]=features_dictonary
            with open('dict.pkl', 'wb') as f:
                pickle.dump(database_dict, f)

            if 'not_site' in request.form:
                return {'percentile': round(precentege, 2), 'risk': round(risk, 2)}
            return render_template('Home_new.html', error_message='', results=True, score=round(result, 3),
                                   precentege=round(precentege, 2), risk=round(risk, 2),id=str(id))


    return render_template('Home_new.html', error_message="", results=False)


@bp.route('/Help')
def help_page():
    return render_template('Help.html')


@bp.route('/Example')
def example_page():
    return render_template('Example.html')


@bp.route('/About')
def about_page():
    return render_template('About.html')
