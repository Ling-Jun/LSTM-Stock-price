import pandas as pd
import numpy as np
from io import BytesIO
import base64
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from os.path import isfile, join
from termcolor import colored
from matplotlib import rcParams
# without this line, savefig() will cut off the xlabel and ylabel
rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')


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


# def load_data(file):
#     """Load data file from web page."""
#     # ensure we choose a data file
#     try:
#         if request.method == 'POST':
#             # save the uploaded file to static/data/ folder
#             file.save('static/data/' + secure_filename(file.filename))
#             # After uploading a file, change the msg on index page to
#             # 'Successfully uploaded a file!'
#             data_dir = os.getcwd() + '/static/data/' + secure_filename(file.filename)
#             dataset = read_data(data_dir)
#     except Exception:
#         # mention specific exceptions whenever possible instead of using a bare except:
#         return "Please choose a data file!"
#     return dataset
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


def create_test_set(dataset, test_start='2017', col=1):
    """
    Create test_set from dataset.

    No need to call this functions, if the test_set is the uploaded file.
    test_start specifies the beggining of test_set.
    col specifies the column of data we analyze.
    """
    test_set = dataset[test_start:].iloc[:, col:col + 1].values  # np array
    return test_set


def create_test_input(dataset, scaler, input_start='2017', lookback=60, col=1):
    """
    Generate a test input dataset from a given dataset.

    Use this function only if the given dataset contains data for training the models.
    lookback -> the number of historical values we need to predict one future value.
    """
    column_header_list = list(dataset.columns.values)
    target_col = column_header_list[col]
    length_of_test_output = len(dataset[target_col][input_start:])  # 251
    # the row number below which is the test_input dataframe
    start_of_test_input = len(dataset[target_col][:]) - length_of_test_output - lookback  # 2709
    test_input_ungrouped = dataset[target_col][start_of_test_input:].values.reshape(-1, 1)
    # total number of rows for test_input: 251+60 =311
    test_input = [test_input_ungrouped[i - lookback:i, 0] for i in range(lookback, len(test_input_ungrouped))]
    test_input = np.array(test_input)
    # --------------------------------------------
    test_input = scaler.transform(test_input)
    # --------------------------------------------
    test_input = np.reshape(test_input, (test_input.shape[0], test_input.shape[1], 1))
    # X_test.shape=(251, 60, 1)
    return test_input


# def choose_model(select):
#     # ensures we choose a model
#     try:
#         model = load_model('models/' + str(select))
#     except Exception:
#         return "Please choose a model!"
#     return model
def choose_model(select):
    # ensures we choose a model
    try:
        model = load_model('models/' + str(select))
    except Exception:
        # raise ImportError("Please choose a model!")
        # print(colored("Choose a model!", 'red'))
        return 'SOMETHING'
    return model


def predict(input, model, scaler):
    """Predict the price."""
    # model = load_model(model)
    prediction = model.predict(input)
    prediction = scaler.inverse_transform(prediction)
    return prediction


def return_rmse(target, prediction):
    """Return the RMSE between prediction and target."""
    rmse = math.sqrt(mean_squared_error(target, prediction))
    return rmse


def plot_predictions(target, predicted, ticker=""):
    """
    Plot prediction and target prices.

    The function takes 4 parameters: test data, predicted data,
    nameURL - the name of image to savefig
    ticker  - if we want to specify the stock ticker
    """
    img = BytesIO()
    # added plt.figure() to initiate a new figure every time we plot
    plt.figure()
    plt.plot(target, color='red', label='Real Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Stock Price')
    plt.title(str(ticker) + ' Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(img)
    # plt.autoscale()
    # this line also allows us to savefig() with xlabel and ylabel
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url
