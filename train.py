"""
Created on Thu May 14 11:38:50 2020

@author: zhoul
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
#import math
#from sklearn.metrics import mean_squared_error


# get raw data from path
#path='https://raw.githubusercontent.com/Ling-Jun/LSTM-Stock-price/master/IBM_2006-01-01_to_2018-01-01.csv'

def read_data(path):
    dataset = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
    print('Tail of the dataset is: \n\n {}:'.format(dataset.tail()))
    return dataset


def main(args):
    parser = argparse.ArgumentParser(description="Do something.")
    #optional arguments are ID-ed by the - prefix, and the remaining arguments are assumed to be positional
    parser.add_argument("--path", "-file_path")
    args = parser.parse_args(args)
    dataset=read_data(args.path)
    return dataset


if __name__ == '__main__':
    dataset=main(sys.argv[1:])







#check data
def check_data(dataset):
    training_set = dataset[:'2016'].iloc[:,1:2].values # it is a np array
    test_set = dataset['2017':].iloc[:,1:2].values # it is a np array
    # dataset[:'2016'].iloc[:,0:2].isnull().sum() #no missing value

    dataset["High"][:'2016'].plot(figsize=(16,4),legend=True)
    dataset["High"]['2017':].plot(figsize=(16,4),legend=True)
    plt.legend(['Training set (Before 2017)','Test set (2017 and beyond)'])
    plt.title('IBM stock price');plt.show()
    return training_set, test_set

training_set, test_set=check_data(dataset)


# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1)); training_set_scaled = sc.fit_transform(training_set)

def X_y_split(lookback): # Use the previous 60 (lookback) values to predict 1 output
    X_train=[training_set_scaled[i-lookback:i,0] for i in range(lookback, len(training_set))]
    y_train=[training_set_scaled[i,0] for i in range(lookback, len(training_set))]
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train

X_train, y_train=X_y_split(60) #X_train.shape=(2709, 60); y_train.shape=(2709, 1)


def X_y_reshape(single_y_size, x, y): 
#To predict 2 consecutive prices, sing_y_size=2, so that each element in y now contains 2 numbers.
    if float(single_y_size).is_integer():
        x=[x[i] for i in range(0,len(y)-len(y)%single_y_size, single_y_size)]
        y=[y[i:i+single_y_size] for i in range(0,len(y)-len(y)%single_y_size,single_y_size)]
    else:
        print('Need an integer!')
    return np.array(x), np.array(y)

X_train, y_train=X_y_reshape(1,X_train, y_train)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1)) # X_train.shape=(2709 , 60 , 1)
# 1st, 2nd, 3rd dimensions: 
# the number of samples. time-steps fed to a sequence, features('price').


def build(X, y, batch_size, epoch):
    # The LSTM architecture
    regressor = Sequential()
    #=======================================================================================================================
    # return_sequences: Whether to return the last output. in the output sequence, or the full sequence. Default: False.
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1],1))); regressor.add(Dropout(0.2))
    #=======================================================================================================================
    regressor.add(LSTM(units=50, return_sequences=True));regressor.add(Dropout(0.2))
    #=======================================================================================================================
    regressor.add(LSTM(units=50, return_sequences=True));regressor.add(Dropout(0.2))
    #=======================================================================================================================
    regressor.add(LSTM(units=50));regressor.add(Dropout(0.2))
    # The output layer
    regressor.add(Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
    print(regressor.summary())
    history=regressor.fit(X,y,epochs=epoch, batch_size=batch_size)
    print(history)
    return regressor

regressor= build(X_train, y_train, batch_size=50, epoch=10)
