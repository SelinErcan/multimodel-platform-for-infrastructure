import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import MinMaxScaler

cycle_count_in_a_day = int(60/15*24)
cycle_count_in_week = int(cycle_count_in_a_day*7)
total_week_count = int(math.ceil(35040/cycle_count_in_week))-2
cycles_for_weekdays = int(cycle_count_in_a_day*5)
cycles_for_weekends = int(cycle_count_in_a_day*2)
print("Cycle count for a day:", cycle_count_in_a_day)
print("Cycle count for one week:", cycle_count_in_week) 
print("Week count for dataset: ", total_week_count)
print("Cycles count weekdays/5 days: ", cycles_for_weekdays)
print("Cycles count weekends/2 days: ", cycles_for_weekends)

train_week_size = 37
test_week_size = 14

def create_week_info_columns(dataset):
    week=1

    # delete weeks that are complete
    dataset=dataset[cycles_for_weekdays:-cycle_count_in_a_day*3]

    # for other weeks
    for i in range(0, len(dataset['power_comsumption']), 672):
        dataset['week'][i:i+672] = week
        dataset['weekend'][i:i+cycles_for_weekdays] = 0
        dataset['weekend'][i+cycles_for_weekdays:i+672] = 1
        for j in range(1,8):
            dataset['day'][i+cycle_count_in_a_day*(j-1):i+cycle_count_in_a_day*j] = j

        week = week +1
    return dataset

def plot_cycles_for_weeks(dataset, week):
    plt.plot(dataset[(dataset['week']==week)&(dataset['weekend']==0)]['power_comsumption'], label="weekdays")
    plt.plot(dataset[(dataset['week']==week)&(dataset['weekend']==1)]['power_comsumption'], color='green', label="weekend")
    plt.xlabel('cycles (in every 15 min)')
    plt.ylabel('power consumption')
    plt.legend(loc="upper left")
    plt.title("Week " + str(week))
    plt.show()

def reshape(dataset_np):
    x_reshaped=[]
    for i in np.unique(dataset_np[:, 1]):
        temp_x = np.asarray([dataset_np[dataset_np[:, 1]==i][:,0]]).astype(np.float32)
        temp_y = np.asarray([dataset_np[dataset_np[:, 1]==i][:,-1]]).astype(np.float32)
        if x_reshaped==[]:
            x_reshaped = temp_x
        else:
            x_reshaped = np.concatenate((x_reshaped, temp_x), axis=0)
    return x_reshaped

def prepare_data(file_name = 'assets/power_demand.txt'):

    dataset = pd.read_csv(file_name, sep=" ", header=None)
    dataset.columns = ['power_comsumption']
    dataset['week'] = ""
    dataset['weekend'] =""
    dataset['day'] =""
    dataset['label'] = ""

    dataset = create_week_info_columns(dataset)

    print("Week 2 data: ", dataset[cycle_count_in_a_day*5:cycle_count_in_a_day*5+cycle_count_in_week])

    # for i in range (1, total_week_count+1):
    #     plot_cycles_for_weeks(dataset, i)

    """Fill label(anomaly) column"""

    anomalies = [(12,5), (13, 1), (17, 3), (18,1), (18,4), (20,1), (39,6), (42,6), (42,7), (51,4), (51,5)]

    for week, day in anomalies:
        dataset.loc[((dataset["week"]==week)&(dataset["day"]==day)), 'label'] = 1
    dataset.loc[dataset["label"]=="", 'label'] = 0

    dataset[dataset['label']==1]

    pca = PCA(n_components=1)
    pca.fit(dataset[['power_comsumption','weekend']])
    dataset['power_comsumption'] = pca.transform(dataset[['power_comsumption','weekend']])

    min_scaler = MinMaxScaler(feature_range=(0,1))
    min_scaler.fit(dataset[['power_comsumption']].to_numpy())
    dataset[['power_comsumption']] = min_scaler.transform(dataset[['power_comsumption']].to_numpy())
    
    X = reshape(np.asarray(dataset))
    X = X.reshape((X.shape[0],1, X.shape[1]))
    
    return X
