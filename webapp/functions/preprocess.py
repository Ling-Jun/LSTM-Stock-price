import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
# without this line, savefig() will cut off the xlabel and ylabel
rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')


def read_data(path):
    dataset = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
    return dataset


def train_test_split(dataset, trainend='2016', teststart='2017', target_col=1):
    # the two lines below give np array
    training_set = dataset[:trainend].iloc[:, target_col:target_col + 1].values
    test_set = dataset[teststart:].iloc[:, target_col:target_col + 1].values
    # dataset[:'2016'].iloc[:,0:2].isnull().sum() #no missing value
    return training_set, test_set


sc = MinMaxScaler(feature_range=(0, 1))


def training_set_scaled(training_set):
    return sc.fit_transform(training_set)


def X_y_split(lookback, training_set):
    # Use the previous 60 (lookback) values to predict 1 output
    X_train = [training_set_scaled[i - lookback:i, 0] for i in range(lookback, len(training_set))]
    y_train = [training_set_scaled[i, 0] for i in range(lookback, len(training_set))]
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train


# non-default argument must be in front of default argument
def X_y_reshape(x, y, single_y_size=1):
    # To predict 2 consecutive prices, single_y_size=2, so that each element in y now contains 2 numbers.
    if float(single_y_size).is_integer():
        x = [x[i] for i in range(0, len(y) - len(y) % single_y_size, single_y_size)]
        y = [y[i:i + single_y_size] for i in range(0, len(y) - len(y) % single_y_size, single_y_size)]
    else:
        print('Need an integer!')
    return np.array(x), np.array(y)
