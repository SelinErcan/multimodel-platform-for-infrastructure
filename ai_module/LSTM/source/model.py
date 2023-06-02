from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
import keras
from keras.layers import Dense, Conv1D, LSTM, SimpleRNN, Flatten, Dropout, Activation
from keras import backend as K
from matplotlib import pyplot as plt
import math
import numpy as np
from metrics import root_mean_squared_error, competitive_score, predict
from data_helper import prepare_train_data, prepare_test_data

def lstm_model(x_train, y_train, sequence_length=30):

    model_lstm = Sequential()
    model_lstm.add(LSTM(10, activation='tanh',kernel_initializer=keras.initializers.glorot_normal(), input_shape=(sequence_length, x_train.shape[2]), return_sequences=True))
    model_lstm.add(Dropout(0.5))
    model_lstm.add(LSTM(10, activation='tanh',kernel_initializer=keras.initializers.glorot_normal(), return_sequences=True))
    model_lstm.add(Dropout(0.5))
    model_lstm.add(LSTM(10, activation='tanh',kernel_initializer=keras.initializers.glorot_normal(), return_sequences=True))
    model_lstm.add(Dropout(0.5))
    model_lstm.add(LSTM(10, activation='tanh',kernel_initializer=keras.initializers.glorot_normal(), return_sequences=True))
    model_lstm.add(Dropout(0.5))
    model_lstm.add(LSTM(1, activation='tanh',kernel_initializer=keras.initializers.glorot_normal(), return_sequences=True))
    model_lstm.add(Flatten())
    model_lstm.add(Dense(100, activation='tanh', kernel_initializer=keras.initializers.glorot_normal()))
    model_lstm.add(Dense(y_train.shape[1]))
    model_lstm.summary()

    #compile model using accuracy to measure model performance
    model_lstm.compile(optimizer='adam', loss= root_mean_squared_error, metrics=[root_mean_squared_error,competitive_score]) 

    #train the model
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=200, min_lr=0.0001)
    history_lstm = model_lstm.fit(x_train, y_train, batch_size=512, validation_split=0.05, epochs=10, callbacks=[reduce_lr])
    return model_lstm, history_lstm

if __name__ == "__main__":
    x_train, y_train = prepare_train_data()
    model_lstm, history_lstm = lstm_model(x_train, y_train)
    model_lstm.save('model_lstm.h5')
    # x_test, y_test = prepare_test_data()
    # rmse_score, compatitive_score = predict(model_lstm, x_test, y_test)
    # print(rmse_score, compatitive_score)