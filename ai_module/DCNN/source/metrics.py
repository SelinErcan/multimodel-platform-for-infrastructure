from keras import backend as K
import numpy as np

def root_mean_squared_error(y_true, y_pred):
    """
    Loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def competitive_score(y_true,y_pred):
    """
    Error Function for Competitive Data
    """
    y_error = y_pred - y_true
    bool_idx = K.greater(y_error, 0)
    loss1 = K.exp(-1*y_error/13) - 1 # greater 0
    loss2 = K.exp(y_error/10) - 1    # less 0
    loss = K.switch(bool_idx, loss2, loss1)
    return K.sum(loss)

def predict(model, x_test, y_test):
    """
    Do prediction on model
    """
    score = model.evaluate(x_test, y_test, verbose=2)
    print('\nRMSE: {}'.format(score[1]))
    print('\nScore: {}'.format(score[2]))
    # y_pred = model.predict(seq_array_test_last)
    return score[1], score[2]