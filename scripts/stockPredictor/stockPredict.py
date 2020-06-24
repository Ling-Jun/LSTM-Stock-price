"""Create a dataframe of historical price from a ticker."""
import argparse
# import os
from pickle import load
import numpy as np
# import pandas as pd
import dataGenerator
import train
import predict
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# set seed for reproducibility
np.random.seed(0)


def parse_CLI_args():
    """Parse CLI arguments for predict.py."""
    parser = argparse.ArgumentParser(description="Generate price data.")
    parser.add_argument("--ticker", "-ticker", "-t")
    parser.add_argument("--train", "-train")
    parser.add_argument("--epochs", "-epoch", "-e")
    parser.add_argument("--batch_size", "-batch_size", "-b")
    parser.add_argument("--model", "-model", "-m")
    args = parser.parse_args()
    #  can call parse_args() with no arguments and it will still work
    if args.epochs is None:
        # add default values
        args.epochs = 30
    if args.batch_size is None:
        args.batch_size = 50
    return args


if __name__ == '__main__':
    args = parse_CLI_args()
    ticker = args.ticker
    price_history = dataGenerator.data_fetch(ticker)
    price_history = dataGenerator.data_clean(price_history)
    start, end = dataGenerator.data_year_range(price_history)
    split_point = dataGenerator.data_split_point(start, end)
    training_set, test_set = train.train_test_split(price_history, train_end=str(split_point - 1), test_start=str(split_point))

    if args.train == 'y':
        train.train_test_split_plot(price_history, train_end=str(split_point - 1), test_start=str(split_point))
        train_input, train_output = train.input_output_split(training_set)
        regressor = train.build(train_input)
        regressor = train.train(train_input, train_output, regressor, epochs=int(args.epochs), batch_size=int(args.batch_size))
        train.save_model(regressor, args.ticker, args.epochs)
        # _, test_set = train.train_test_split(dataset)
    elif args.train == 'n':
        scaler = load(open('scaler.pkl', 'rb'))
        test_input = predict.create_test_input(price_history, scaler, input_start=str(split_point))
        prediction = predict.predict(test_input, args.model, scaler)
        predict.return_rmse(test_set, prediction)
        predict.plot_predictions(test_set, prediction)
    else:
        print('-train parameter needs to be either "y" or "n".')
