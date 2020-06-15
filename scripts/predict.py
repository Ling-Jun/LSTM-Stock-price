"""
Created on Thu June 14, 2020.

@author: zhoul
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import math
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import train
from pickle import load
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


# def scale_data(train_path):
#     """Adjust the scaler according to training data."""
#     dataset = train.read_data(train_path)
#     training_set, test_set = train.train_test_split(dataset)
#     train.scale_data().fit(training_set)


def predict(input, model, scaler):
    """Predict the price."""
    model = load_model(model)
    prediction = model.predict(input)
    prediction = scaler.inverse_transform(prediction)
    return prediction


def plot_predictions(test, prediction, ticker=''):
    """Plot predicted & real price."""
    plt.plot(test, color='red', label='Real Price')
    plt.plot(prediction, color='blue', label='Predicted Price')
    plt.title(ticker + ' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def return_rmse(target, prediction):
    """Return the RMSE between prediction and target."""
    rmse = math.sqrt(mean_squared_error(target, prediction))
    print("The root mean squared error is {}.".format(rmse))


def parse_CLI_args():
    """Parse CLI arguments for predict.py."""
    parser = argparse.ArgumentParser(description="Do something.")
    parser.add_argument("--path", "-path")
    parser.add_argument("--model", "-model")
    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == '__main__':
    args = parse_CLI_args()
    dataset = train.read_data(args.path)
    _, test_set = train.train_test_split(dataset)
    scaler = load(open('scaler.pkl', 'rb'))
    test_input = create_test_input(dataset, scaler)
    prediction = predict(test_input, args.model, scaler)
    return_rmse(test_set, prediction)
    plot_predictions(test_set, prediction)
