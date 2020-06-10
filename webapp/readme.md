# Web-App for LSTM powered stock prediction

## The data must be arranged in the following manners:
* CSV format
* Index column is Date
* There are several stock prices: Open, Close, High, Low, etc. Each price occupies
  a column. The webapp by default takes the 2nd column.
* The webapp by default takes price history until the end of year 2016 as the training data, and uses data from the beginning of year 2017 as test data.


# Development
## Set up enviornment
* Setup up virtual environment with Anaconda
** $ conda create --name env_name
** $ conda activate env_name
* Install dependencies:
** $ pip install -r requirements.txt
