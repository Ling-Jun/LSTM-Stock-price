"""
Created on Thu May 14 11:40:05 2020

@author: zhoul
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('fivethirtyeight')
from keras.models import load_model
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import train
from matplotlib import rcParams
# without this line, savefig() will cut off the xlabel and ylabel
rcParams.update({'figure.autolayout': True})

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


def plot_predictions(test, predicted, nameURL):
    plt.plot(test, color='red',label='Real Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
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

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Do something.")
#     #optional arguments are ID-ed by the - prefix, and the remaining arguments are assumed to be positional
#     parser.add_argument("--path", "-file_path"); parser.add_argument("--model", "-model")
#     args = parser.parse_args(sys.argv[1:])
#     dataset=train.read_data(args.path)
#     training_set, test_set = train.train_test_split(dataset)
#     train.sc.fit(training_set)
#
#     test_inputs = test_inputs(60, dataset)
#     test_inputs = train.sc.transform(test_inputs)
#     X_test = test_prep(60, test_inputs) # (251, 60, 1)
#
#     model=load_model(args.model)
#     predicted_stock_price = model.predict(X_test)
#     predicted_stock_price = train.sc.inverse_transform(predicted_stock_price)
#     return_rmse(test_set,predicted_stock_price)
#     plot_predictions(test_set,predicted_stock_price)
