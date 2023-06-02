import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)
from matplotlib import pyplot
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
import json

def add_columns():
    """
    Add all avaliable feature names for dataset
    """
    columns = ['unit_number', 'time_in_cycles']
    for i in range(1,4):
        columns = columns + ['operational_setting_' + str(i)] 
    for i in range(1,24):
        columns = columns + ['sensor_measurement_' + str(i)] 
    #print(columns)
    print("Column length: ",len(columns))
    return columns
    
def lifes_cycles(dataset):
  """
  Look max, min, mean life in cycles for dataset
  """
  cycles_limit=[dataset.groupby('unit_number').max()['time_in_cycles']]
  cycles_limit = np.asarray(cycles_limit)
  print("Cycles Limits >> ", cycles_limit)
  print("Max Life >> ",np.max(cycles_limit))
  print("Mean Life >> ",np.mean(cycles_limit))
  print("Min Life >> ",np.min(cycles_limit))

def drop_features(dataset):
    """
    Create a list to drop some features which has null or constant values
    """

    dataset_np = dataset.to_numpy()

    # drop features that are constant 
    drop_list = [2,3,4]
    for i in range(5,28):
      unique_count=len(np.unique(dataset_np[:,i]))
      if unique_count<5 or np.isnan(dataset_np[:,i]).all():
          drop_list.append(i)
    print("Removed column size: ",len(drop_list))

    return drop_list

def prepare_data(data, factor = 0):
    """
    Create Rul labels using time_in_cycles feature.
    """
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)
    
    return df[df['time_in_cycles'] > factor]

def preprocess(dataset,drop_list, scaler):
    """
    Drop features according to drop_list, normalize sensors and create new dataframe
    """
    columns = dataset.columns
    dataset_np = dataset.to_numpy()
    dataset_dropped = np.delete(dataset_np, drop_list, axis=1) 
    print("New shape of train data: ",dataset_dropped.shape)

    # Min-max normalization(StandardScaler() will normalize the features i.e. each column of X)
    dataset_normalized=scaler.transform(dataset_dropped[:,2:-1])  

    # combine unit_name and times_in_cycles features with normalized data
    dataset_df = pd.concat([pd.DataFrame(data=dataset_dropped[:,:2]), pd.DataFrame(data=dataset_normalized)], axis=1, sort=False)
    dataset_df['RUL']=dataset_dropped[:,-1]

    # get new column names for new df
    new_columns = [columns[idx_col] for idx_col in range(len(columns)) if idx_col not in drop_list]
    dataset_df.columns=new_columns

    return dataset_df

def gen_sequence(id_df, seq_length, seq_cols):
    """
    Function to reshape features into (samples, time steps, features)
    Only sequences that meet the window-length are considered, no padding is used. 
    """
    # for one unit, put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, r_early):
    """ 
    Function to generate labels
    Only sequences that meet the window-length are considered, no padding is used. 
    After this, r-early is applied
    """
    
    y = id_df[['RUL']].values
    num_elements = y.shape[0]

    y=y[seq_length:num_elements, :]

    # use R-early
    y[y > r_early] = r_early
    return y

def time_window_processing(data_df, sequence_length = 30):
    """
    Generate sequence data and their labels
    """
    # pick the feature columns 
    sequence_cols = list(data_df.columns)
    # drop RUL column
    sequence_cols = sequence_cols[2:-1]
    print("Columns: ", sequence_cols)

    with open('assets/columns.json', 'w') as jsonfile:
        json.dump(sequence_cols, jsonfile)

    # generator for the sequences
    # transform each id of the train dataset in a sequence
    seq_gen = (list(gen_sequence(data_df[data_df['unit_number']==id], sequence_length, sequence_cols)) 
               for id in data_df['unit_number'].unique())

    # generate sequences and convert to numpy array
    x = np.concatenate(list(seq_gen)).astype(np.float32)
    print("Train size: ",x.shape)

    # generate labels
    label_gen = [gen_labels(data_df[data_df['unit_number']==id], sequence_length, 125) 
                 for id in data_df['unit_number'].unique()]
    y = np.concatenate(label_gen).astype(np.float32)
    print(y.shape)
    return x,y

def prepare_train_data(train_file = 'assets/train_FD001.txt'):

    train_dataset = pd.read_csv(train_file, sep=" ", header=None)
    train_dataset.columns = add_columns()
    train_dataset=prepare_data(train_dataset)

    drop_list = drop_features(train_dataset)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(np.delete(train_dataset.to_numpy(), drop_list, axis=1)[:,2:-1])
    train_df = preprocess(train_dataset, drop_list, scaler)
    x_train, y_train = time_window_processing(train_df)
    return x_train, y_train

def prepare_test_data(test_file = 'assets/test_FD001.txt', label_file='assets/RUL_FD001.txt', sequence_length =30):

    test_dataset = pd.read_csv(test_file, sep=" ", header=None)
    labels = pd.read_csv(label_file, sep=" ", header=None)
    f = open('assets/columns.json',) 
    sequence_cols = list(json.load(f)) 
  
    test_dataset.columns = add_columns()
    test_dataset=prepare_data(test_dataset)
    drop_list = drop_features(test_dataset)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(np.delete(test_dataset.to_numpy(), drop_list, axis=1)[:,2:-1])
    test_df = preprocess(test_dataset, drop_list, scaler)

    y_test = labels[0].to_numpy()
    y_test = np.asarray(y_test.reshape(y_test.shape[0],-1)).astype(float)
    print(y_test.shape)
    # We pick the last sequence for each id in the test data
    x_test = []
    delete_ids = []
    for id in test_df['unit_number'].unique():
        if len(test_df[test_df['unit_number']==id]) >= sequence_length:
            x_test.append(test_df[test_df['unit_number']==id][sequence_cols].values[-sequence_length:])
        else:
            delete_ids.append(id)
    y_test = np.delete(y_test, delete_ids)
    x_test = np.asarray(x_test).astype(np.float32)
    print(x_test.shape)
    # test metrics
    return x_test, y_test

        