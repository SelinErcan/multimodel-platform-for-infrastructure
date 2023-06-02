import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.metrics import RootMeanSquaredError
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import pickle
from sklearn.metrics import accuracy_score
from data_helper import prepare_data

def svm_model(x_train, y_train):

    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    svm.fit(x_train)
    return svm

if __name__ == "__main__":
    x_train, y_train = prepare_data()
    model_svm = svm_model(x_train, y_train)
    pickle.dump(model_svm, open('model_svm.h5', 'wb'))