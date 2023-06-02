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

def dcnn_model(x_train, y_train, sequence_length=30):

    model_dcnn = Sequential()#add model layers
    model_dcnn.add(Conv1D(10, kernel_size=10, activation='tanh', kernel_initializer=keras.initializers.glorot_normal(), padding="same", input_shape=(sequence_length, x_train.shape[2])))
    model_dcnn.add(Dropout(0.5))
    model_dcnn.add(Conv1D(10, kernel_size=10, activation='tanh', kernel_initializer=keras.initializers.glorot_normal(), padding="same"))
    model_dcnn.add(Dropout(0.5))
    model_dcnn.add(Conv1D(10, kernel_size=10, activation='tanh', kernel_initializer=keras.initializers.glorot_normal(), padding="same"))
    model_dcnn.add(Dropout(0.5))
    model_dcnn.add(Conv1D(10, kernel_size=10, activation='tanh', kernel_initializer=keras.initializers.glorot_normal(), padding="same"))
    model_dcnn.add(Dropout(0.5))
    model_dcnn.add(Conv1D(1, kernel_size=3, activation='tanh', kernel_initializer=keras.initializers.glorot_normal(), padding="same"))
    model_dcnn.add(Flatten())
    model_dcnn.add(Dense(100, activation='tanh', kernel_initializer=keras.initializers.glorot_normal()))
    model_dcnn.add(Dense(y_train.shape[1]))
    print(model_dcnn.summary())  

    #compile model using accuracy to measure model performance
    model_dcnn.compile(optimizer='adam', loss= root_mean_squared_error, metrics=[root_mean_squared_error,competitive_score]) 

    #train the model
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=200, min_lr=0.0001)
    history_dcnn = model_dcnn.fit(x_train, y_train, batch_size=512, validation_split=0.05, epochs=10, callbacks=[reduce_lr])
    return model_dcnn, history_dcnn

if __name__ == "__main__":
    x_train, y_train = prepare_train_data()
    model_dcnn, history_dcnn = dcnn_model(x_train, y_train)
    model_dcnn.save('model_dcnn.h5')
    # x_test, y_test = prepare_test_data()
    # rmse_score, compatitive_score = predict(model_dcnn, x_test, y_test)
    # print(rmse_score, compatitive_score)