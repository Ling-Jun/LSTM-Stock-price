"""Create a dataframe of historical price from a ticker."""
import yfinance as yf
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
    if ticker == '':
        raise ImportError('Please select specify a stock ticker!')
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
