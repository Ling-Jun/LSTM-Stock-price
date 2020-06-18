# Web-App for LSTM powered stock prediction

## Development
Set up environment

* Setup up virtual environment with Anaconda, or VENV

>
    $ conda create --name env_name
>
    $ conda activate env_name

* Install Python 3.7.6

* Install dependencies with pip:

>
    $ pip install -r requirements.txt

## Dataset
Example datasets are given [here] (https://github.com/Ling-Jun/LSTM-Stock-price/tree/master/time%20series%20data).
The data must be arranged in the following manners:

* CSV format
* Index column is Date
* There are several stock prices: Open, Close, High, Low, etc. Each price occupies
  a column. The webapp by default takes the 2nd column.
* The webapp by default takes price history until the end of year 2016 as the training data, and uses data from the beginning of year 2017 as test data. They can be changed from the .py files.
