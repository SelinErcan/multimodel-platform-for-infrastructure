import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
import keras
from keras.layers import Dense, Conv1D, LSTM, Flatten, Dropout, Activation, TimeDistributed, RepeatVector
from keras.initializers import glorot_uniform # Xavier normal initializer
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from data_helper import prepare_data

def lstm_ae_model(x_train):
    # define model
    model_ae = Sequential()
    model_ae.add(LSTM(400, activation='relu', input_shape=(1, x_train.shape[2]), return_sequences=True))
    model_ae.add(LSTM(200, activation='relu', return_sequences=False))
    model_ae.add(RepeatVector(1))
    model_ae.add(LSTM(200, activation='relu', return_sequences=True))
    model_ae.add(LSTM(400, activation='relu', return_sequences=True))
    model_ae.add(TimeDistributed(Dense(x_train.shape[2])))
    model_ae.summary()

    adam = keras.optimizers.Adam(learning_rate=0.001)
    model_ae.compile(optimizer=adam, loss='mse')

    history_ae = model_ae.fit(x_train, x_train, epochs=30)
    print(history_ae.history.keys())
    print(model_ae.evaluate(x_train, x_train, verbose=2))

    return model_ae, history_ae

if __name__ == "__main__":
    x_train = prepare_data()
    model_lstm, history_lstm = lstm_ae_model(x_train)
    model_lstm.save('model_lstm_ae.h5')

