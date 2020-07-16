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

`$ pip install -r requirements.txt`

* TO use the App: navigate to root folder, run/test on local computer:

`$ python Flask_webapp/app.py`




## Folder structure
```
├── .flake8
├── .gitignore
├── Flask_webapp
│   ├── app.py
│   ├── data_preparation_objects
│   │   └── scaler.pkl
│   ├── models
│   │   ├── model_IBM_2_epochs.h5
│   │   └── model_IBM_60_epochs.h5
│   ├── static
│   │   └── js
│   │       ├── stock_info.js
│   │       └── stock_predict.js
│   └── templates
│       ├── index.html
│       ├── price_predict.html
│       └── show_prediction.html
├── LSTM_predictor
│   ├── setup.py
│   └── stockPredictor
│       ├── dataGenerator.py
│       ├── predict.py
│       ├── preload.py
│       └── train.py
├── README.md
├── requirements.txt
└── runtime.txt
```

## Dataset
* We now allow users to provide a simple stock ticker, the price data will be fetched from Yahoo finance.



## Background Information on Long Short-Term Memory(LSTM)
Long short-term memory (LSTM) unit is a building unit for layers of a recurrent neural network (RNN). A **RNN** composed of LSTM units is often called an LSTM network. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell is responsible for "remembering" values over arbitrary time intervals; hence the word "memory" in LSTM. Each of the three gates can be thought of as a "conventional" artificial neuron, as in a multi-layer (or feedforward) neural network.

An LSTM is well-suited to classify, process and predict time series given time lags of unknown size and duration between important events. LSTMs were developed to deal with the exploding and vanishing gradient problem when training traditional RNNs.
<img src="https://cdn-images-1.medium.com/max/1600/0*LyfY3Mow9eCYlj7o.">
Source: [Medium](https://codeburst.io/generating-text-using-an-lstm-network-no-libraries-2dff88a3968). <br>


## Components of LSTMs
* Forget Gate “f” ( a neural network with sigmoid)
* Candidate layer “C"(a NN with Tanh)
* Input Gate “I” ( a NN with sigmoid )
* Output Gate “O”( a NN with sigmoid)
* Hidden state “H” ( a vector )
* Memory state “C” ( a vector)

* Inputs to the LSTM cell at any step are X<sub>t</sub> (current input) , H<sub>t-1</sub> (previous hidden state ) and C<sub>t-1</sub> (previous memory state).  
* Outputs from the LSTM cell are H<sub>t</sub> (current hidden state ) and C<sub>t</sub> (current memory state)

## Technical Details

* requirements.txt specifies the dependencies of this webapp, it is generated with [pipreqs](https://github.com/bndr/pipreqs) package.
* .gitignore specifies which files and folders to ignore when pushing to Git remote, i.e. Github.
* .flake8 specifies which linting errors to ignore when using flake8 package. Linting errors are the conventions and practice of best formatting the source code, i.e. two empty lines are recommended from the last line of code to the definition of a new function.
* runtime.txt: by default, new Python applications on Heroku use the Python runtime indicated in Specifying a Python version.
* Procfile specifes how to deploy the webapp on [Heroku](https://devcenter.heroku.com/articles/procfile)
* LSTM_predictor is a custom package written by the author to realize customized functions for the webapp. It is [installed locally](https://github.com/Ling-Jun/example-local-package) by adding the line `-e ./LSTM_predictor` in requirements.txt file.

## Deploy
* must have gunicorn package specified in requirements.txt file
* must ensure that you are pushing the branch with your code to heroku master. So instead of:

`$ git push heroku master`

You would do something like:

`$git push heroku (current-branch):master$
