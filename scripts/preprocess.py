# #Pre-processing of data

import pandas as pd
from statistics import mean
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

path_data = 'C:/Users/okiem/OneDrive/Desktop/The_Project_Folder/Presumed_Open_Data_Science_Challenge/data/task0'

# **Import Data**
df_demand = pd.read_csv(f'{path_data}/demand_train_set0.csv')
df_demand.head()
df_pv = pd.read_csv(f'{path_data}/pv_train_set0.csv')
df_pv.head()
df_weather = pd.read_csv(f'{path_data}/weather_train_set0.csv')
df_weather.head()
df_submit = pd.read_csv(f'{path_data}/teamname_set0.csv')
df_submit.tail()

# Check if 'datetime' columns are equal
df_demand['datetime'].equals(df_pv['datetime'])

# Join dfs together
df_pv_demand = df_pv.merge(right=df_demand,
                           on='datetime',
                           how='left')
df_pv_demand.head(100)

# Create a weather set that matches the date range of pv_demand set
df_weather.set_index('datetime',
                     inplace=True)
df_weather_set0 = df_weather.loc['2017-11-03 00:00:00':'2018-07-22 23:00:00']
df_weather_set0.reset_index(inplace=True)
df_weather_set0.head()

# Join pv_demand set with weather_set0 set to complete the model dataframe
df_pv_demand_weather = df_pv_demand.merge(right=df_weather_set0,
                                          on='datetime',
                                          how='outer')


# ------------------------------------------------------------------------------
def impute_weather(lst_cols):
    '''Imputes the missing data for the weather paramters using the mean of the
    timesteps before and after. '''
    for c in range(0, len(lst_cols)):
        for i in range(0, len(df_pv_demand_weather)-1):
            if pd.isnull(df_pv_demand_weather[lst_cols[c]].iloc[i]):
                df_pv_demand_weather.loc[i, lst_cols[c]] = mean([df_pv_demand_weather.loc[i-1, lst_cols[c]], df_pv_demand_weather.loc[i+1, lst_cols[c]]])
# ------------------------------------------------------------------------------


# Impute NaN values for weather features
impute_weather(df_pv_demand_weather.columns.values[5:])

# This block of code inserst a new column that contains the k_index(1-48) for each day.
# It slices the HH:MM:SS of the datetime value and creates a dictionary with the
# extracted HH:MM:SS string as the key and the k_index(1-48) as the value. A list
# is created from the dictionary to be inserted into the dataframe, then the to_datetime
# column is converted into a date index.
df_pv_demand_weather.loc[0:47, 'datetime'].values
lst_times = [t[11:] for t in df_pv_demand_weather.loc[0:47, 'datetime'].values]
lst_k = list(range(1, 49))
dct_time_k = dict(zip(lst_times, lst_k))
lst_k_index = [dct_time_k.get(t[11:], None) for t in df_pv_demand_weather['datetime']]
df_pv_demand_weather.insert(1, 'k_index', lst_k_index)
df_pv_demand_weather['datetime'] = pd.to_datetime(df_pv_demand_weather['datetime'])
df_pv_demand_weather.set_index('datetime',
                               inplace=True)

df_pv_demand_weather.dropna(axis=0, how='any', inplace=True)

path = 'C:/Users/okiem/OneDrive/Desktop/The_Project_Folder/Presumed_Open_Data_Science_Challenge/pod-challenge/pickles'
df_pv_demand_weather.to_pickle(f'{path}/df_set0_dropna.pkl')
