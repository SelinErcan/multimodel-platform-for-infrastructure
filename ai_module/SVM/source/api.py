import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import keras
from data_helper import prepare_data
from sklearn.metrics import accuracy_score
import requests
from werkzeug.utils import secure_filename
import os
import glob

app = Flask(__name__)

@app.route('/do_prediction/<model_name>/',methods=['GET', 'POST'])
def do_predicttion(model_name):

    file1 = request.files['file1']
    file1.save('tmp/' + file1.filename)
    
    model = pickle.load(open('model_' + model_name + '.h5', 'rb'))
    x_test, y_test = prepare_data(file_name = 'tmp/' + file1.filename)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)*100

    files = glob.glob('tmp/*')
    for f in files:
        os.remove(f)

    return jsonify([accuracy])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5005"), debug=True)
