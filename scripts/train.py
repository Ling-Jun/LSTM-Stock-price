# -*- coding: utf-8 -*-
"""
Created on Thu June 14, 2020.

@author: zhoul
"""
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# get raw data from path='https://raw.githubusercontent.com/Ling-Jun/LSTM-Stock-price/master/IBM_2006-01-01_to_2018-01-01.csv'


def read_data(path):
    """
    Read data from a given path as a dataframe.

    Data must be CSV with no missing values.
    """
    dataset = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
    return dataset


def train_test_split(dataset, train_end='2016', test_start='2017', col=1):
    """
    Split data into training_set and test_set, returns two np.arrays.

    train_end, test_start are two strings that specify the year of the last
    training dataset and the year of the first test data. col specifies the
    column of data we analyze.
    """
    training_set = dataset[:train_end].iloc[:, col:col + 1].values  # np array
    test_set = dataset[test_start:].iloc[:, col:col + 1].values  # np array
    return training_set, test_set


def train_test_split_plot(dataset, ticker='', train_end='2016', test_start='2017', col=1):
    """Plot the training and testing data together."""
    column_header_list = list(dataset.columns.values)
    dataset[column_header_list[col]][:train_end].plot(figsize=(16, 4), legend=True)
    dataset[column_header_list[col]][test_start:].plot(figsize=(16, 4), legend=True)
    # the following two lines will create two SEPARATE plots instead one.
    # dataset.iloc[:, col:col + 1][:train_end].plot(figsize=(16, 4), legend=True)
    # dataset.iloc[:, col:col + 1][test_start:].plot(figsize=(16, 4), legend=True)
    plt.legend(['Training set (Before 2017)', 'Test set (2017 and beyond)'])
    plt.title(ticker + ' Stock Price')
    plt.show()


def scale_data(range=(0, 1)):
    """Scales the data to a default range of (0, 1), range can be changed."""
    sc = MinMaxScaler(feature_range=range)
    return sc


def input_output_split(data_set, lookback=60):
    """
    Split the data_set into input and output for preprocessing.

    lookback specifies number of historical values needed to predict 1 value.
    X - the input of a function, y - the output of a function.
    data_set can be either training_set or test_set.
    """
    data_set_scaled = scale_data().fit_transform(data_set)
    X = [data_set_scaled[i - lookback:i, 0] for i in range(lookback, len(data_set))]
    y = [data_set_scaled[i, 0] for i in range(lookback, len(data_set))]
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # X_train.shape=(2709 , 60 , 1)
    return X, y


# def X_y_reshape(single_y_size, x, y):
# #To predict 2 consecutive prices, sing_y_size=2, so that each element in y now contains 2 numbers.
#     if float(single_y_size).is_integer():
#         x=[x[i] for i in range(0,len(y)-len(y)%single_y_size, single_y_size)]
#         y=[y[i:i+single_y_size] for i in range(0,len(y)-len(y)%single_y_size,single_y_size)]
#     else:
#         print('Need an integer!')
#     return np.array(x), np.array(y)

def build(input):
    """Build the LSTM model."""
    regressor = Sequential()
    # =======================================================================================================================
    # return_sequences: Whether to return the last output. in the output sequence, or the full sequence. Default: False.
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(input.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # =======================================================================================================================
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # =======================================================================================================================
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # =======================================================================================================================
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    # =======================================================================================================================
    regressor.add(Dense(units=1))
    # =======================================================================================================================
    regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
    print(regressor.summary())
    return regressor


def train(input, output, regressor, batch_size=50, epochs=30):
    """Train the model and display the training preprocess."""
    history = regressor.fit(input, output, batch_size, epochs)
    print(history)
    return regressor


def save_model(regressor, ticker='', epochs='UNKNOWN'):
    """Save the trained models, with default unkown epochs."""
    regressor.save('model_{}_{}_epochs.h5'.format(ticker, epochs))


def cli_args():
    """Read arguments from CLI."""
    parser = argparse.ArgumentParser(description="Do something.")
    # optional arguments are ID-ed by the - prefix, and the remaining arguments
    # are assumed to be positional
    parser.add_argument("--path", "-path")
    parser.add_argument("--epochs", "-epoch")
    parser.add_argument("--ticker", "-ticker")
    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == '__main__':
    args = cli_args()
    dataset = read_data(args.path)
    print('Tail of the dataset is: \n\n {}:'.format(dataset.tail()))
    training_set, test_set = train_test_split(dataset)
    train_test_split_plot(dataset)

    train_input, train_output = input_output_split(training_set)

    regressor = build(train_input)
    regressor = train(train_input, train_output, regressor, epochs=int(args.epochs), batch_size=50)
    save_model(regressor, args.ticker, args.epochs)
