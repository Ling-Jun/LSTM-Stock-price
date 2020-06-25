"""
Created on Thu June 14, 2020.

@author: zhoul
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from io import BytesIO
import base64
plt.style.use('fivethirtyeight')


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


def predict(input, model, scaler):
    """Predict the price."""
    # model = load_model(model)
    prediction = model.predict(input)
    prediction = scaler.inverse_transform(prediction)
    return prediction


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


def return_rmse(target, prediction):
    """Return the RMSE between prediction and target."""
    rmse = math.sqrt(mean_squared_error(target, prediction))
    # print("The root mean squared error is {}.".format(rmse))
    return rmse
