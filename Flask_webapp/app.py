"""Python version 3.7.6 ."""
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import matplotlib
from pickle import load
from stockPredictor import predict, train, preload, dataGenerator

plt.style.use('fivethirtyeight')
matplotlib.use('Agg')

app = Flask(__name__)

onlyfiles = preload.list_models('/Flask_webapp/models')
string = ""


@app.route('/', methods=['GET', 'POST'])
def index():
    # ticker = request.form.get('ticker')
    # stock_info = dataGenerator.stock_info(ticker)
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    ticker = request.form['ticker']
    if ticker:
        info = dataGenerator.stock_info(ticker)
        # think of better ways to present the stock info!
        return jsonify(info)

    return jsonify({'error': 'Invalid ticker!'})


@app.route('/price_predict', methods=['GET', 'POST'])
def pred():
    # relative path of models folder
    onlyfiles = preload.list_models('/Flask_webapp/models')
    # send ML models to index.html page, no value is taken from html pages
    return render_template('price_predict.html', modelfiles=onlyfiles, string=string)


@app.route('/show_prediction', methods=['POST'])
def show_pred():
    # get the form from stock_predict.js
    ticker = request.form['ticker']
    select = request.form['model']
    if ticker and (select != 'Select a model'):
        # do sth
        return jsonify({'ticker': ticker, 'model': select})
    return jsonify({'error': 'Invalid ticker or model selection!'})

    # try:
    #     dataset = dataGenerator.data_fetch(ticker)
    # # except ImportError:
    # #     return "Please add a ticker!!"
    # except Exception:
    #     # return "Invalid ticker!!"
    #     return render_template('price_predict.html', modelfiles=onlyfiles, string="ENTER A VALID TICKER!")
    # # process the dataset
    # dataset = dataGenerator.data_clean(dataset)
    # start, end = dataGenerator.data_year_range(dataset)
    # split_point = dataGenerator.data_split_point(start, end, ratio=0.1)
    # _, test_target = train.train_test_split(dataset, train_end=str(split_point - 1), test_start=str(split_point))
    #
    # # load models and data transfomation objects
    # try:
    #     model = preload.choose_model(select)
    # except ImportError:
    #     # return "Please choose a model!!"
    #     return render_template('price_predict.html', modelfiles=onlyfiles, string="PLEASE CHOOSE A MODEL!")
    # scaler = load(open('Flask_webapp/data_preparation_objects/scaler.pkl', 'rb'))
    # # specify the right path
    # test_input = predict.create_test_input(dataset, scaler, input_start=str(split_point))
    #
    # # predict
    # prediction = predict.predict(test_input, model, scaler)
    # rmse = predict.return_rmse(test_target, prediction)
    # # read ticker from ticker textbox,
    # plot_url = predict.plot_predictions(test_target, prediction, ticker)
    # return render_template('show_prediction.html', model=select, rmse=rmse, img=plot_url)


if __name__ == '__main__':
    app.run(debug=True)
