import os
from os.path import isfile, join
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import pandas as pd


def list_models(local_dir):
    """Return the list of models in a given directory."""
    mypath = os.getcwd() + local_dir
    model_list = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    list = ['Select a model'] + model_list
    return list


def read_data(path):
    """
    Read data from a given path as a dataframe.

    Data must be CSV with no missing values.
    """
    dataset = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
    return dataset


def load_data(file):
    """Load data file from web page."""
    if file.filename == '':
        raise ImportError('Please select a data file!')
        # return 'Data file!!'
    # ensure we choose a data file, save the uploaded file to static/data/ folder
    file.save('static/data/' + secure_filename(file.filename))
    # After uploading a file, change the msg on index page to
    # 'Successfully uploaded a file!'
    data_dir = os.getcwd() + '/static/data/' + secure_filename(file.filename)
    dataset = read_data(data_dir)
    return dataset


def choose_model(select):
    """Make sure we choose a model."""
    try:
        model = load_model('Flask_webapp/models/' + str(select))
    except Exception:
        raise ImportError("Please choose a model!")
        # print(colored("Choose a model!", 'red'))
        # return 'SOMETHING'
    return model
