#!/usr/bin/env python
# coding: utf-8

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
pd.set_option('display.max_columns', 500)

class PodChallenge():
    '''
    Args:


    Attributes:
    '''

    def __init__(self):
        os.chdir(os.path.dirname(os.getcwd()))
        self.path = os.getcwd()

    def import_task_data(self, set):
        ''' Reads in csv files into DataFrames from the specific task week data directory

        Args:
            set(str): an integer number representing which task week to be used

        Returns:
            DataFrames for each dataset
        '''
        self.df_demand = pd.read_csv(f'{self.path}/data/task{set}/demand_train_set{set}.csv')
        self.df_pv = pd.read_csv(f'{self.path}/data/task{set}/pv_train_set{set}.csv')
        self.df_weather = pd.read_csv(f'{self.path}/data/task{set}/weather_train_set{set}.csv')
        self.df_submit = pd.read_csv(f'{self.path}/data/task{set}/teamname_set{set}.csv')

        return self.df_demand, self.df_pv, self.df_weather, self.df_submit

    def merge_dataframes(self):
        '''Creates two combined dataframes from raw task week data.

        Returns:
            df_train: dataframe containing all weather data, pv_power data, and demand data
                      given for the task week. There will be missing weather data at this point,
                      becuase the weather data is hourly.
            df_taskweek: dataframe containing all the wather data that can be used for
                         prediction of the wekk of interest.

        '''
        # Check 'datetime' columns are equal in demand and pv dfs, this must be true
        assert self.df_demand['datetime'].equals(self.df_pv['datetime']), "Demand and PV train sets have different timestamps"

        # Merge the demand and pv_power dataset together into one dataframe
        self.df_train = self.df_pv.merge(right=self.df_demand,
                                         on='datetime',
                                         how='left')

        # Create a weather sets that matches the date range of pv_demand set and submit set
        # Set DatetimeIndex
        self.df_weather.set_index('datetime',
                                  inplace=True)

        # Create weather dataframe with matching timestamps with training/testing data
        start_date = self.df_train.loc[0, 'datetime']
        end_date = self.df_train.loc[len(self.df_train)-1, 'datetime']
        df_weather_train = self.df_weather.loc[start_date:end_date]

        # Create weather dataframe with matching timestamps with task week submission data
        start_date = self.df_submit.loc[0, 'datetime']
        end_date = self.df_submit.loc[len(self.df_submit)-1, 'datetime']
        df_weather_taskweek = self.df_weather.loc[start_date:end_date]

        # Join pv_demand set with weather_set0 set to complete the model dataframe
        self.df_train = self.df_train.merge(right=df_weather_train,
                                            on='datetime',
                                            how='outer')
        self.df_taskweek = self.df_submit.merge(right=df_weather_taskweek,
                                                  on='datetime',
                                                  how='outer')

        return self.df_train, self.df_taskweek

    def impute_weather(self, lst_cols, df):
        '''Imputes the missing data for the weather paramters using the mean of the
        timesteps before and after.

        Args:
            lst_cols(list): list of weather columns to impute
            df: dataframe to which to pply the transformation
        Returns:
            df: transformed df
        '''
        for c in range(0, len(lst_cols)):
            for i in range(0, len(df)-1):
                if pd.isnull(df[lst_cols[c]].iloc[i]):
                    df.loc[i, lst_cols[c]] = mean([df.loc[i-1, lst_cols[c]], df.loc[i+1, lst_cols[c]]])
            df[lst_cols[c]].iloc[len(df)-1] = df[lst_cols[c]].iloc[len(df)-2]

        return df

    def fill_weather_data(self):
        '''Calls the impute_weather() function for each dataframe'''

        self.df_train = self.impute_weather(self.df_train.columns.values[5:], self.df_train)
        self.df_taskweek = self.impute_weather(self.df_taskweek.columns.values[2:], self.df_taskweek)

    def calc_mean(self, i, df, param=None):
        '''Calculates mean for certain solar and temp columns from weather station and returns value

        Args:
            i(int): index values
            df(DataFrame): dataframe to perform tranformation on
            param(str): 'solar' or 'temp', which parameter to perform calculation on

        Retruns:
            The mean value

        '''
        if param == 'solar':
            return mean([df.loc[i, 'solar_location1'], df.loc[i, 'solar_location2'], df.loc[i, 'solar_location3'], df.loc[i, 'solar_location4'], df.loc[i, 'solar_location5'], df.loc[i, 'solar_location6']])
        if param == 'temp':
            return mean([df.loc[i, 'temp_location1'], df.loc[i, 'temp_location2'], df.loc[i, 'temp_location5'], df.loc[i, 'temp_location6']])

    def insert_weather_means(self):
        '''Inserts columns with the mean values of the solar and temperature weather
        data from the six given weather sites. Currently, the solar mean uses all
        sites while the temperature uses the four site surrounding the solar panel.
        '''
        # Create and insert solar and temp mean columns for train data
        lst_solar_mean = [self.calc_mean(i, self.df_train, param='solar') for i in range(0, len(self.df_train))]
        lst_temp_mean = [self.calc_mean(i, self.df_train, param='temp') for i in range(0, len(self.df_train))]

        self.df_train.insert(5, 'temp_mean1256', lst_temp_mean)
        self.df_train.insert(6, 'solar_mean123456', lst_solar_mean)

        # do it for the submit dataframe
        lst_solar_mean = [self.calc_mean(i, self.df_taskweek, param='solar') for i in range(0, len(self.df_taskweek))]
        lst_temp_mean = [self.calc_mean(i, self.df_taskweek, param='temp') for i in range(0, len(self.df_taskweek))]

        self.df_taskweek.insert(5, 'temp_mean1256', lst_temp_mean)
        self.df_taskweek.insert(6, 'solar_mean123456', lst_solar_mean)

    def create_k_index(self, df):
        '''This block of code inserst a new column that contains the k_index(1-48) for each day.

        It slices the HH:MM:SS of the datetime value and creates a dictionary with the
        extracted HH:MM:SS string as the key and the k_index(1-48) as the value. A list
        is created from the dictionary to be inserted into the dataframe, then the to_datetime
        column is converted into a date index.

        Args:
            df(DataFrame): dataframe to perform transformation on
        Returns:
            Transformed dataframe

        '''
        df.loc[0:47, 'datetime'].values
        lst_times = [t[11:] for t in df.loc[0:47, 'datetime'].values]
        lst_k = list(range(1, 49))
        dct_time_k = dict(zip(lst_times, lst_k))
        lst_k_index = [dct_time_k.get(t[11:], None) for t in df['datetime']]
        df.insert(1, 'k_index', lst_k_index)
        df['datetime'] = pd.to_datetime(df['datetime'])

        return df

    def insert_k_index(self):
        '''Calls function to create k_index clomuns for each dataframe'''

        self.df_train = self.create_k_index(self.df_train)
        self.df_taskweek = self.create_k_index(self.df_taskweek)

    def increase_time_lod(self, df):
        '''Inserts columns with increased level of detail for time:
           -day of week
           -day of month
           -month

        Args:
            df(DataFrame): dataframe to perform transformation on
        Returns:
            Transformed dataframe
        '''
        df.set_index('datetime', inplace=True)
        df.insert(0, 'day_of_week', df.index.dayofweek)
        df.insert(0, 'day_of_month', df.index.day)
        df.insert(0, 'month', df.index.month)
        df.reset_index(inplace=True)

        return df

    def insert_time_cols(self):
        '''Calls function to increase the level od detail of time'''

        self.df_train = self.increase_time_lod(self.df_train)
        self.df_taskweek = self.increase_time_lod(self.df_taskweek)

    def drop_missing_values(self):
        '''Drops all rows with missing values'''

        self.df_train.dropna(axis=0, how='any', inplace=True)
        self.df_taskweek.dropna(axis=0, how='any', inplace=True)

    def rfr_model_train(self):
        '''

        '''

        self.df_train_dm = self.df_train[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'demand_MW']]
        self.df_train_pv = self.df_train[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'pv_power_mw']]

        # Process dataset

        # make sure the dataset is sorted
        self.df_train_dm.sort_index(inplace=True)
        self.df_train_pv.sort_index(inplace=True)

        # Create independent and dependent variable matrices
        X_dm = self.df_train_dm.iloc[:, :-1].values
        y_dm = self.df_train_dm.iloc[:, -1].values
        X_pv = self.df_train_pv.iloc[:, :-1].values
        y_pv = self.df_train_pv.iloc[:, -1].values

        # #Encode Categorical Data IF Needed
        # #One Hot Encoder
        # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3] )], remainder='passthrough')
        # X = np.array(ct.fit_transform(X))
        # X_svr = np.array(ct.fit_transform(X_svr))

        # Split datasets without shuffling
        X_dm_train, X_dm_test, y_dm_train, self.y_dm_test = train_test_split(X_dm, y_dm,
                                                                             test_size=336,
                                                                             shuffle=False)
        X_pv_train, X_pv_test, y_pv_train, self.y_pv_test = train_test_split(X_pv, y_pv,
                                                                             test_size=336,
                                                                             shuffle=False)

        # Random Forest Regression Models
        rfr_dm = RandomForestRegressor(n_estimators=850)
        rfr_dm.fit(X_dm_train, y_dm_train)
        rfr_pv = RandomForestRegressor(n_estimators=100)
        rfr_pv.fit(X_pv_train, y_pv_train)

        # Random Forest Regression Prediction
        self.y_pred_rfr_dm = rfr_dm.predict(X_dm)
        self.y_pred_rfr_pv = rfr_pv.predict(X_pv)

    def evaluate_model(self):
        '''

        '''

        print('                               R_Squared           Adjusted R_Squared    Mean Squared Error')
        print(f'Random Forest Regression Demand:     {r2_score(self.y_dm_test, self.y_pred_rfr_dm)}   {1-(1-r2_score(self.y_dm_test, self.y_pred_rfr_dm))*((len(self.y_dm_test)-1)/(len(self.y_dm_test)-len(self.y_dm_test[0])-1))}    {mean_squared_error(self.y_dm_test, self.y_pred_rfr_dm)}')
        print(f'Random Forest Regression Power:     {r2_score(self.y_pv_test, self.y_pred_rfr_pv)}   {1-(1-r2_score(self.y_pv_test, self.y_pred_rfr_pv))*((len(self.y_pv_test)-1)/(len(self.y_pv_test)-len(self.y_pv_test[0])-1))}    {mean_squared_error(self.self.y_pv_test, self.y_pred_rfr_pv)}')
