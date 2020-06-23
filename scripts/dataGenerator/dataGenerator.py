"""Create a dataframe of historical price from a ticker."""
import yfinance as yf
import argparse
import os
import numpy as np
# import pandas as pd
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
    # this returns None, since performance is done on the original dataframe
    # missing_values = dataframe.isnull().sum()
    # dataframe = dataframe.fillna(method='pad')
    # dataframe = dataframe.fillna(method='bfill', inplace=True)
    dataframe = dataframe.interpolate(method='linear', limit_direction='forward')
    dataframe = dataframe.interpolate(method='linear', limit_direction='backward')
    return dataframe


def data_to_CSV(df, ticker):
    """Save data in a CSV file."""
    cwd = os.getcwd()
    path = cwd + '/' + str(ticker) + '_price.csv'
    df.to_csv(str(path), index=True, header=True)


def parse_CLI_args():
    """Parse CLI arguments for predict.py."""
    parser = argparse.ArgumentParser(description="Generate price data.")
    parser.add_argument("--ticker", "-ticker", "-t")
    args = parser.parse_args()
    #  can call parse_args() with no arguments and it will still work
    return args


# no data for HWO, where does yfiance get its data?
# know if the current ticker is still listed, which exchange?
if __name__ == '__main__':
    args = parse_CLI_args()
    ticker = args.ticker
    price_history = data_fetch(ticker)
    price_history = data_clean(price_history)
    # print(price_history.head(20))
    # print('=================================================')
    # info = stock_info(ticker)
    # s = pd.Series(info, index=info.keys())
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     # display full pd series without truncation
    #     print(s)
    data_to_CSV(price_history, ticker)
