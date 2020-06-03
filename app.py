import os
from flask import Flask, request, render_template, jsonify, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy

regressor = load_model()

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
# db = SQLAlchemy(app)

@app.route('/', methods=['POST','GET'])
def index():
	return render_template('index.html')


# @app.route('/train', methods=['GET','POST'])
# def train():
#     if request.method == 'GET':
#         render_template('train.html')
#     else:
#         pass

@app.route('/train')
def train():
    if request.method == 'POST':
		if 'album' not in request.files:
			return 'No file attached', 400

		raw = request.files['raw_data']
		# TODO: check file type and format etc

		genre, pred = regressor.classify_one(raw)

		return jsonify({
			'genre': genre
		})

    return render_template('train.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    raw =request.files['raw_data']
    Predictions=regressor.predict()
	return render_template('predict.html')


if __name__ == '__main__':
	app.run(debug=True)
