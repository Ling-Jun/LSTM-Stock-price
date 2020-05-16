# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:40:05 2020

@author: zhoul
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import math
import train


def test_inputs(lookback, data):
    # Make sure the first 60 (lookback) entires of test set have 60 previous values
    test_inputs = data["High"][len(data["High"][:])-len(test_set) - lookback:].values.reshape(-1,1)# total length 251+60 =311
    #shape into 1 column as indicated by '1', with an UNKNOWN row number indicated by '-1' COMPATIBLE with the original list
    test_inputs = train.sc.fit_transform(test_inputs) # test_inputs.shape =(311, 1), ndarray
    return test_inputs


def test_prep(lookback,data):
    X_test = [data[i-lookback:i,0] for i in range(lookback,lookback+len(test_set))]; X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))#X_test.shape=(251, 60, 1)
    return X_test

# Some functions to help out with
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real IBM Stock Price')
    plt.plot(predicted, color='blue',label='Predicted IBM Stock Price')
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.show()

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Do something.")
    #optional arguments are ID-ed by the - prefix, and the remaining arguments are assumed to be positional
    parser.add_argument("--path", "-filepath")
    args = parser.parse_args(sys.argv[1:])
    dataset=train.read_data(args.path)

    training_set, test_set=train.check_data(dataset)
    test_inputs=test_inputs(60, dataset)
    X_test=test_prep(60,test_inputs) # (251, 60, 1)
    train.sc.fit_transform(training_set)

    model=load_model('my_model.h5')
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = train.sc.inverse_transform(predicted_stock_price)
    plot_predictions(test_set,predicted_stock_price)
    return_rmse(test_set,predicted_stock_price)
