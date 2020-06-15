import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import os
from matplotlib import rcParams
# without this line, savefig() will cut off the xlabel and ylabel
rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')


def test_inputs(data, test_set, lookback=60):
    # Make sure the first 60 (lookback) entires of test set have 60 previous values
    test_inputs = data["High"][len(data["High"][:]) - len(test_set) - lookback:].values.reshape(-1,1)
    # total length 251+60 =311
    # shape into 1 column as indicated by '1', with an UNKNOWN row number indicated by '-1' COMPATIBLE with the original list
    return test_inputs


def test_prep(data, test_set, lookback=60):
    X_test = [data[i - lookback:i, 0] for i in range(lookback, lookback + len(test_set))]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # X_test.shape=(251, 60, 1)
    return X_test


# takes 4 parameters: test data, predicted data,
# nameURL-the name of image to savefig
# ticker -if we want to specify the stock ticker
def plot_predictions(test, predicted, nameURL, ticker=""):
    # added plt.figure() to initiate a new figure every time we plot
    plt.figure()
    plt.plot(test, color='red', label='Real Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Stock Price')
    plt.title(str(ticker) + ' Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(nameURL)
    # plt.autoscale()
    # this line also allows us to savefig() with xlabel and ylabel
    plt.close()


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    # print("The root mean squared error is {}.".format(rmse))
    return rmse
