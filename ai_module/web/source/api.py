"""
Get data on demand
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/show_predict',methods=['GET', 'POST'])
def show_predict():
    
    parameters = [x for x in request.form.values()]
    print("Selected model parameters: ", parameters)
    analysis_type = parameters[0]
    model_name = parameters[1]
    data_name = parameters[2]
    
    if model_name in ["dcnn", "dnn", "lstm"]:
        try:
            r = requests.get('https://ingenis-datahub.ngrok.io/get_data/' +  'test_' + data_name + '/')
            with open('data/test_' + data_name +'.txt', 'w') as f:
                f.write(r.text)
        except Exception as e: 
            return str(e)
        try:
            r = requests.get('https://ingenis-datahub.ngrok.io/get_data/' + 'RUL_' + data_name + '/')
            with open('data/RUL_' + data_name + '.txt', 'w') as f:
                f.write(r.text)
        except:
            return "2."

        files = {'file1': open('data/test_' + data_name +'.txt', 'rb'), 'file2': open('data/RUL_' + data_name + '.txt', 'rb')}

        if model_name == "dcnn":
            scores = requests.post('http://dcnn:5001/do_prediction/' + model_name +'/', files=files)
        if model_name == "dnn":
            scores = requests.post('http://dnn:5002/do_prediction/' + model_name +'/', files=files)
        if model_name == "lstm":
            scores = requests.post('http://lstm:5003/do_prediction/' + model_name +'/', files=files)

        scores = scores.text.replace("\n", "").replace(" ", "").replace("[", "").replace("]", "")
        prediction_text = 'Prediction result - RMSE: {} Compatitive: {}'.format(scores.split(",")[0], scores.split(",")[1])

    elif model_name in ["lstm_ae", "svm"]:
        try:
            r = requests.get('https://ingenis-datahub.ngrok.io/get_data/' + data_name + '/')
            with open('data/' + data_name +'.txt', 'w') as f:
                f.write(r.text)
        except Exception as e:
            return str(e)

        files = {'file1': open('data/' + data_name +'.txt', 'rb')}

        if model_name == "lstm_ae":
            score = requests.post('http://lstm_ae:5004/do_prediction/' + model_name +'/', files=files)
            prediction_text = 'Prediction result - MSE: {}'.format(score.text.replace("\n", "").replace(" ", "").replace("[", "").replace("]", ""))
        if model_name == "svm":
            score = requests.post('http://svm:5005/do_prediction/' + model_name +'/', files=files)
            prediction_text = 'Prediction result - Accuracy: %{}'.format(score.text.replace("\n", "").replace(" ", "").replace("[", "").replace("]", ""))
    
    else:
        prediction_text = "There is no model with this name."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
