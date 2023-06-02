"""
Send data on demand
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import glob
import json

app = Flask(__name__)

@app.route('/show_avaliable_data',methods=['GET'])
def show_avaliable_data():
    return jsonify(glob.glob("data/*.txt"))

@app.route('/get_data/<data_name>/',methods=['GET'])
def get_data(data_name):
    f = open('data/' + data_name + ".txt","r")
    return f.read()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5009"), debug=True)
