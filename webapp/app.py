from flask import Flask, render_template, request, url_for
import matplotlib.pyplot as plt;plt.style.use('fivethirtyeight')
# this line is different from Trial_6, it has to be this way for the code to work
from tensorflow.keras.models import load_model # NOT: from keras.models import load_model
from io import BytesIO #
import base64
import numpy as np
import pandas as pd
import train
import pred
# from predict import test_inputs, but this wouldn't work because we have
# model.predict() where predict is a function, so the computer recognizes it
# as a function instead of a package name
import math
from sklearn.metrics import mean_squared_error


app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html', models=[{'name':''}, {'name':'model_epoch_60.h5'}, {'name':'model_epoch_30.h5'}])



@app.route('/predict',methods=['POST', 'GET'])
def predict():
    file=request.values['file']
    select = request.form.get('comp_select')

    dataset = train.read_data(file)
    training_set, test_set = train.train_test_split(dataset)
    train.sc.fit(training_set)

    test_inputs = pred.test_inputs(60, dataset, test_set)
    test_inputs = train.sc.transform(test_inputs)
    X_test = pred.test_prep(60, test_inputs, test_set)   # (251, 60, 1)

    model = load_model('models/' + str(select))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = train.sc.inverse_transform(predicted_stock_price)

    rmse = pred.return_rmse(test_set, predicted_stock_price)
    pred.plot_predictions(test_set, predicted_stock_price, nameURL = 'static/'+ str(select).replace('.h5',"")+'.png')
    # the image URL to pass on to predict.html, it has to be in a 'static' folder
    imgname = 'static/'+ str(select).replace('.h5',"")+'.png'
    return render_template('predict.html', model=select, rmse=rmse, imgname = imgname)
    # image not showing!


if __name__=='__main__':
    app.run(debug=True)
