# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:40:05 2020

@author: zhoul
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('fivethirtyeight')
#import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout
#import math
from sklearn.metrics import mean_squared_error
import train


def test_inputs(lookback, data):
    # Make sure the first 60 (lookback) entires of test set have 60 previous values
    test_inputs = data["High"][len(data["High"][:])-len(test_set) - lookback:].values.reshape(-1,1)# total length 251+60 =311
    #shape into 1 column as indicated by '1', with an UNKNOWN row number indicated by '-1' COMPATIBLE with the original list
    test_inputs = train.sc.transform(test_inputs) # test_inputs.shape =(311, 1), ndarray
    return test_inputs


def test_prep(lookback,data):
    X_test = [data[i-lookback:i,0] for i in range(lookback,lookback+len(test_set))]; X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))#X_test.shape=(251, 60, 1)
    return X_test







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Do something.")
    #optional arguments are ID-ed by the - prefix, and the remaining arguments are assumed to be positional
    parser.add_argument("--path", "-file_path")
    args = parser.parse_args(sys.argv[1:])
    dataset=train.read_data(args.path)

    test_set=train.check_data(dataset)[1]
    test_inputs=test_inputs(60, dataset)
    X_test=test_prep(60,test_inputs) # (251, 60, 1)

    predicted_stock_price = train.regressor.predict(X_test)
    predicted_stock_price = train.sc.inverse_transform(predicted_stock_price)
    plot_predictions(test_set,predicted_stock_price)
    return_rmse(test_set,predicted_stock_price)
