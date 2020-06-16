# python version 3.7.6
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import matplotlib

from pickle import load
import predict

plt.style.use('fivethirtyeight')
matplotlib.use('Agg')
"""
By default matplotlib uses TK gui toolkit, when you're rendering an image without
using the toolkit (i.e. into a file or a string-base64), matplotlib still instantiates
a window that doesn't get displayed, causing all kinds of problems. In order to
avoid that, you should use an Agg backend.
https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
When using Matplotlib versions older than 3.1, it is necessary to explicitly
instantiate an Agg canvas
"""
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    # relative path of models folder
    onlyfiles = predict.list_models('/models')
    # send ML models to index.html page, no value is taken from html pages
    return render_template('index.html', modelfiles=onlyfiles)


@app.route('/predict', methods=['POST', 'GET'])
def pred():
    # if request.method == 'POST':
    file = request.files['file']
    dataset = predict.load_data(file)
    scaler = load(open('scaler.pkl', 'rb'))
    test_input = predict.create_test_input(dataset, scaler)
    select = request.form.get('comp_select')
    model = predict.choose_model(select)
    prediction = predict.predict(test_input, model, scaler)
    test_set = predict.create_test_set(dataset)
    rmse = predict.return_rmse(test_set, prediction)
    # read ticker from ticker textbox,
    ticker = request.form.get('ticker')
    plot_url = predict.plot_predictions(test_set, prediction, ticker)
    return render_template('predict.html', model=select, rmse=rmse, img=plot_url)
    # image for 30 epochs is still not showing xlabel and ylabel


if __name__ == '__main__':
    app.run(debug=True)
