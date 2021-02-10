# #Pre-processing of data

import pandas as pd
from statistics import mean
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

path_data = os.path.abspath(os.path.join(os.getcwd(), 'data/task0'))

# **Import Data**
df_demand = pd.read_csv(f'{path_data}/demand_train_set0.csv')
df_demand.tail()
df_pv = pd.read_csv(f'{path_data}/pv_train_set0.csv')
df_pv.head()
df_weather = pd.read_csv(f'{path_data}/weather_train_set0.csv')
df_weather.head()
df_submit = pd.read_csv(f'{path_data}/teamname_set0.csv')
df_submit.head()

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

df_weather_submit = df_weather.loc['2018-07-23 00:00:00':'2018-07-29 23:30:00']
df_weather_submit.reset_index(inplace=True)
df_weather_submit.head()

# Join pv_demand set with weather_set0 set to complete the model dataframe
df_pv_demand_weather = df_pv_demand.merge(right=df_weather_set0,
                                          on='datetime',
                                          how='outer')

df_submit_weather = df_submit.merge(right=df_weather_submit,
                                          on='datetime',
                                          how='outer')

# ------------------------------------------------------------------------------
def impute_weather(lst_cols, df):
    '''Imputes the missing data for the weather paramters using the mean of the
    timesteps before and after. '''
    for c in range(0, len(lst_cols)):
        for i in range(0, len(df)-1):
            if pd.isnull(df[lst_cols[c]].iloc[i]):
                df.loc[i, lst_cols[c]] = mean([df.loc[i-1, lst_cols[c]], df.loc[i+1, lst_cols[c]]])
        df[lst_cols[c]].iloc[len(df)-1] = df[lst_cols[c]].iloc[len(df)-2]

# ------------------------------------------------------------------------------

df_submit_weather.head()

# Impute NaN values for weather features
impute_weather(df_pv_demand_weather.columns.values[5:], df_pv_demand_weather)
impute_weather(df_submit_weather.columns.values[2:], df_submit_weather)
df_submit_weather.tail()

# ------------------------------------------------------------------------------
def calc_mean(i, df, param=None):
    '''Calculates mean for certain solar and temp columns from weather station and returns value'''
    if param == 'solar':
        return mean([df.loc[i, 'solar_location1'], df.loc[i, 'solar_location2'], df.loc[i, 'solar_location3'], df.loc[i, 'solar_location4'], df.loc[i, 'solar_location5'], df.loc[i, 'solar_location6']])
    if param == 'temp':
        return mean([df.loc[i, 'temp_location1'], df.loc[i, 'temp_location2'], df.loc[i, 'temp_location5'], df.loc[i, 'temp_location6']])
# ------------------------------------------------------------------------------


# Create and insert solar and temp mean columns for weather data
lst_solar_mean = [calc_mean(i, df_pv_demand_weather, param='solar') for i in range(0, len(df_pv_demand_weather))]
lst_temp_mean = [calc_mean(i, df_pv_demand_weather, param='temp') for i in range(0, len(df_pv_demand_weather))]
df_pv_demand_weather.insert(5, 'temp_mean1256', lst_temp_mean)
df_pv_demand_weather.insert(6, 'solar_mean123456', lst_solar_mean)

# do it for the submit dataframe
lst_solar_mean = [calc_mean(i, df_submit_weather, param='solar') for i in range(0, len(df_submit_weather))]
lst_temp_mean = [calc_mean(i, df_submit_weather, param='temp') for i in range(0, len(df_submit_weather))]
df_submit_weather.insert(5, 'temp_mean1256', lst_temp_mean)
df_submit_weather.insert(6, 'solar_mean123456', lst_solar_mean)




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

# do it for the submit dataframe
df_submit_weather.loc[0:47, 'datetime'].values
lst_times = [t[11:] for t in df_submit_weather.loc[0:47, 'datetime'].values]
lst_k = list(range(1, 49))
dct_time_k = dict(zip(lst_times, lst_k))
lst_k_index = [dct_time_k.get(t[11:], None) for t in df_submit_weather['datetime']]
df_submit_weather.insert(1, 'k_index', lst_k_index)
df_submit_weather['datetime'] = pd.to_datetime(df_submit_weather['datetime'])
df_submit_weather.set_index('datetime',
                               inplace=True)

# Engineer a few features breaking up the month, day of month and day of week
# to investigate seasonal trends.
df_pv_demand_weather.insert(0, 'day_of_week', df_pv_demand_weather.index.dayofweek)
df_pv_demand_weather.insert(0, 'day_of_month', df_pv_demand_weather.index.day)
df_pv_demand_weather.insert(0, 'month', df_pv_demand_weather.index.month)

# do it for submit dataframe
df_submit_weather.insert(0, 'day_of_week', df_submit_weather.index.dayofweek)
df_submit_weather.insert(0, 'day_of_month', df_submit_weather.index.day)
df_submit_weather.insert(0, 'month', df_submit_weather.index.month)

# Remove outage event from data
index = df_pv_demand_weather[(df_pv_demand_weather['demand_MW'] >= 6.0)|(df_pv_demand_weather['demand_MW'] <= 1.2)].index
df_pv_demand_weather_nooutage = df_pv_demand_weather.drop(index)
df_pv_demand_weather_nooutage.drop('panel_temp_C', axis=1, inplace=True)
# interpolate missing data left over
df_pv_demand_weather_nooutage['irradiance_Wm-2'].interpolate(method='time', inplace=True)
df_pv_demand_weather_nooutage['pv_power_mw'].interpolate(method='time', inplace=True)

# final dataframe with panel temp dropped to avoid that missing data, and outage outliers removed
df_pv_demand_weather_nooutage = df_pv_demand_weather_nooutage[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'pv_power_mw', 'demand_MW']]

# final dataframe with all NaN values rows removed and outage event kept in data
# drop all missing values rows present in the raw data
df_pv_demand_weather.dropna(axis=0, how='any', inplace=True)
df_pv_demand_weather = df_pv_demand_weather[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'pv_power_mw', 'demand_MW']]

# final dataframe with task0 week weather data in it ready for prediction
df_submit_weather = df_submit_weather[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'charge_MW']]

df_pv_demand_weather.info()
df_submit_weather.info()
df_pv_demand_weather_nooutage.info()

# dataframe with all NaN values rows removed and outage event kept in data
df_pv_demand_weather.to_pickle(f'{path}/df_pv_demand_weather_dropna.pkl')
# dataframe with panel temp dropped to avoid that missing data, and outage outliers removed
df_pv_demand_weather_nooutage.to_pickle(f'{path}/df_pv_demand_weather_nooutage.pkl')
# dataframe with task0 week weather data in it ready for prediction
df_submit_weather.to_pickle(f'{path}/df_submit_task0.pkl')
