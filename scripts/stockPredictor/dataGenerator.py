"""Create a dataframe of historical price from a ticker."""
import yfinance as yf
import argparse
# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# set seed for reproducibility
np.random.seed(0)


def stock_info(ticker):
    """Return stock information."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return info


def data_fetch(ticker):
    """Create price dataframe from a ticker."""
    stock = yf.Ticker(ticker)
    # get historical market data
    hist = stock.history(period="max")
    # hist = hist.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
    return hist


def data_clean(dataframe):
    """Check if the dataframe is full, all data has the right format."""
    cols = list(dataframe.columns.values)
    cols_to_drop = set(cols) - {'Open', 'High', 'Low', 'Close'}
    cols_to_drop = list(cols_to_drop)
    dataframe.drop(cols_to_drop, axis=1, inplace=True)
    dataframe = dataframe.interpolate(method='linear', limit_direction='forward')
    dataframe = dataframe.interpolate(method='linear', limit_direction='backward')
    dataframe.index = pd.to_datetime(dataframe.index)
    return dataframe


def data_year_range(dataframe):
    """Return the beginning and ending years from index."""
    index_year = dataframe.index.year.tolist()
    # remove duplicated elements
    index_year = list(dict.fromkeys(index_year))
    start_year = index_year[0]
    end_year = index_year[-1]
    return start_year, end_year


def data_split_point(start, end, ratio=0.1):
    """
    Calculate the time point to split data into train & test.

    ratio: the percentage of test data in the entire dataset.
    this function doesn't handle the case where start and end are very
    close.
    """
    split_point = int(end) - (int(end) - int(start)) * ratio
    split_point = int(round(split_point, 0))
    return split_point


# def parse_CLI_args():
#     """Parse CLI arguments for predict.py."""
#     parser = argparse.ArgumentParser(description="Generate price data.")
#     parser.add_argument("--ticker", "-ticker", "-t")
#     args = parser.parse_args()
#     #  can call parse_args() with no arguments and it will still work
#     return args
#
#
# if __name__ == '__main__':
#     args = parse_CLI_args()
#     ticker = args.ticker
#     price_history = data_fetch(ticker)
#     price_history = data_clean(price_history)
#     start, end = data_year_range(price_history)
#     split_point = data_split_point(start, end)
#     training_set, test_set = train_test_split(price_history, train_end=str(split_point - 1), test_start=str(split_point))
#     train_test_split_plot(price_history, train_end=str(split_point - 1), test_start=str(split_point))
    # print(price_history.head(500))
    # print(price_history.index.year)
    # print('start:', start, '\n', 'end:', end)
    # print(data_split_point(start, end))
    # print(round(2.9, 0))
