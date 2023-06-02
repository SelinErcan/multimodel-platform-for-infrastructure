import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import keras
from data_helper import prepare_test_data
from metrics import root_mean_squared_error, competitive_score, predict
import requests
from werkzeug.utils import secure_filename
import os
import glob

app = Flask(__name__)

@app.route('/do_prediction/<model_name>/',methods=['GET', 'POST'])
def do_predicttion(model_name):

    file1 = request.files['file1']
    file1.save('tmp/' + file1.filename)
    file2 = request.files['file2']
    file2.save('tmp/' + file2.filename)
    
    model = keras.models.load_model('model_' + model_name + '.h5', custom_objects={'root_mean_squared_error': root_mean_squared_error, 'competitive_score': competitive_score} )
    x_test, y_test = prepare_test_data(test_file = 'tmp/' + file1.filename, label_file='tmp/' + file2.filename)
    rmse_score, compatitive_score = predict(model, x_test, y_test) 

    files = glob.glob('tmp/*')
    for f in files:
        os.remove(f)

    return jsonify([rmse_score, compatitive_score])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5002"), debug=True)
