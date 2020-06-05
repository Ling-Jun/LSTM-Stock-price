from flask import Flask, render_template, request, url_for, session
import matplotlib.pyplot as plt; plt.style.use('fivethirtyeight')
# this line is different from Trial_6, it has to be this way for the code to work
from tensorflow.keras.models import load_model
# NOT: from keras.models import load_model
from io import BytesIO
import base64
import func
import os
from os.path import isfile, join
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def index():
    # relative path of models folder
    mypath = os.getcwd()+'\models'
    onlyfiles = ['Select a model'] + [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    # send ML models to index.html page, no value is taken from html pages
    return render_template('index.html', modelfiles = onlyfiles)



@app.route('/predict',methods=['POST', 'GET'])
def predict():
    # ensure we choose a data file
    try:
        if request.method == 'POST':
            f = request.files['file']
            # save the uploaded file to static/data/ folder
            f.save('static/data/'+secure_filename(f.filename))
            # After uploading a file, change the msg on index page to
            # 'Successfully uploaded a file!'
            dataset = func.read_data(os.getcwd()+'/static/data/'+secure_filename(f.filename))
    except:
        return "Please choose a data file!"


    training_set, test_set = func.train_test_split(dataset)
    func.sc.fit(training_set)

    test_inputs = func.test_inputs(60, dataset, test_set)
    test_inputs = func.sc.transform(test_inputs)
    X_test = func.test_prep(60, test_inputs, test_set)   # (251, 60, 1)

    # ensures we choose a model
    try:
        select = request.form.get('comp_select')
        model = load_model('models/' + str(select))
    except:
        return "Please choose a model!"

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = func.sc.inverse_transform(predicted_stock_price)

    rmse = func.return_rmse(test_set, predicted_stock_price)

    img = BytesIO()
    # read ticker from ticker textbox,
    ticker = request.form.get('ticker')
    # what's the use of get()?
    func.plot_predictions(test_set, predicted_stock_price, img, ticker)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')




    return render_template('predict.html', model=select, rmse=rmse,
                            img = plot_url)
    # image for 30 epochs is still not showing xlabel and ylabel


if __name__=='__main__':
    app.run(debug=True)
