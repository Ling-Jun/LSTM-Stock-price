import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import math
from sklearn.metrics import mean_squared_error
import os
from matplotlib import rcParams
# without this line, savefig() will cut off the xlabel and ylabel
rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')



def read_data(path):
    dataset = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
    return dataset

def train_test_split(dataset, trainend='2016', teststart='2017', target_col=1):
    training_set = dataset[:trainend].iloc[:,target_col:target_col+1].values # it is a np array
    test_set = dataset[teststart:].iloc[:,target_col:target_col+1].values # it is a np array
    # dataset[:'2016'].iloc[:,0:2].isnull().sum() #no missing value
    return training_set, test_set

sc = MinMaxScaler(feature_range=(0,1))

def X_y_split(lookback): # Use the previous 60 (lookback) values to predict 1 output
    X_train=[training_set_scaled[i-lookback:i,0] for i in range(lookback, len(training_set))]
    y_train=[training_set_scaled[i,0] for i in range(lookback, len(training_set))]
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train

# non-default argument must be in front of default argument
def X_y_reshape(x, y, single_y_size = 1):
#To predict 2 consecutive prices, single_y_size=2, so that each element in y now contains 2 numbers.
    if float(single_y_size).is_integer():
        x=[x[i] for i in range(0,len(y)-len(y)%single_y_size, single_y_size)]
        y=[y[i:i+single_y_size] for i in range(0,len(y)-len(y)%single_y_size,single_y_size)]
    else:
        print('Need an integer!')
    return np.array(x), np.array(y)

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
    #=======================================================================================================================
    regressor.add(Dense(units=1))
    #=======================================================================================================================
    regressor.compile(optimizer='rmsprop',loss='mean_squared_error'); print(regressor.summary())
    history=regressor.fit(X,y,epochs=epoch, batch_size=batch_size)
    print(history)
    return regressor


# there's a 3rd argument test_set comparing to older version of predict.py
def test_inputs(lookback, data, test_set):
    # Make sure the first 60 (lookback) entires of test set have 60 previous values
    test_inputs = data["High"][len(data["High"][:])-len(test_set) - lookback:].values.reshape(-1,1)# total length 251+60 =311
    #shape into 1 column as indicated by '1', with an UNKNOWN row number indicated by '-1' COMPATIBLE with the original list
    return test_inputs
# there's a 3rd argument test_set comparing to older version of predict.py
def test_prep(lookback,data, test_set):
    X_test = [data[i-lookback:i,0] for i in range(lookback, lookback+len(test_set))]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))#X_test.shape=(251, 60, 1)
    return X_test


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)
# ctime refers to the last metadata change for specified path in UNIX,
# while in Windows it refers to path creation time. So in Windows it DOESN't refer
# to the newest file in the directory


# takes 4 parameters: test data, predicted data, nameURL-the name of image to savefig
# ticker-if we want to specify the stock ticker
def plot_predictions(test, predicted, nameURL, ticker=""):
# added plt.figure() to initiate a new figure every time we plot
    plt.figure()
    plt.plot(test, color='red',label='Real Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Stock Price')
    plt.title(str(ticker) +' Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    # changed from plt.show() to plt.savefig()
    plt.savefig(nameURL)
    # savefig() doesn't save the xlabel, ylabel?
    # plt.autoscale()
    # this line also allows us to savefig() with xlabel and ylabel
    plt.close()


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    # print("The root mean squared error is {}.".format(rmse))
    return rmse
