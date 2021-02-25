#!/usr/bin/env python
# coding: utf-8

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
import pickle
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import KNNImputer
pd.set_option('display.max_columns', 500)

class PodChallenge():
    '''
    Args:


    Attributes:
    '''

    def __init__(self):
        self.clf = {}
        self.y_pred = {}
        self.y_test = {}

    def import_task_data(self, set):
        ''' Reads in csv files into DataFrames from the specific task week data directory

        Args:
            set(str): an integer number representing which task week to be used

        Returns:
            DataFrames for each dataset
        '''
        self.set = set
        self.df_demand = pd.read_csv(f'../data/task{set}/demand_train_set{set}.csv')
        print(f'Import complete: demand_train_set{set}.csv')
        self.df_pv = pd.read_csv(f'../data/task{set}/pv_train_set{set}.csv')
        print(f'Import complete: pv_train_set{set}.csv')
        self.df_weather = pd.read_csv(f'../data/task{set}/weather_train_set{set}.csv')
        print(f'Import complete: weather_train_set{set}.csv')
        self.df_submit = pd.read_csv(f'../data/task{set}/teamname_set{set}.csv')
        print(f'Import complete: teamname_set{set}.csvdemand_train_set{set}.csv')

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

        print('Completed: Traind and tast_week dataframes merged with weather data')

        self.df_train.sort_index(inplace=True)

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
            # df[lst_cols[c]].where(isinstance(df[lst_cols[c]], int), other=(df[lst_cols[c]].fillna(method='ffill') + df[lst_cols[c]].fillna(method='bfill'))/2)

        return df

    def fill_weather_data(self):
        '''Calls the impute_weather() function for each dataframe'''

        self.df_train = self.impute_weather(self.df_train.columns.values[5:], self.df_train)
        self.df_taskweek = self.impute_weather(self.df_taskweek.columns.values[2:], self.df_taskweek)
        print('Complete: fill missing weather data')

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

        print('Completed: weather data mean columns calculated and inserted')

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

        print('Completed: k_index column created and inserted')

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

        return df

    def insert_time_cols(self):
        '''Calls function to increase the level of detail of time'''

        self.df_train = self.increase_time_lod(self.df_train)
        self.df_taskweek = self.increase_time_lod(self.df_taskweek)

        print('Completed: time level of detail increased')

    def remove_outage(self):
        '''Remove the outage from the data set. After EDA, the demand went to 0
        and then abnormally rebounded to demands greater than 6 MW during the following
        time period: '2018-05-08 08:00:00':'2018-05-11 01:00:00'. The demand data
        during this time period will be removed entirely, so the KNN imputer can
        replace the data will reasonable data.
        '''
        self.df_train['demand_MW']['2018-05-08 08:00:00':'2018-05-11 01:00:00'] = None

    def impute_knn(self):
        '''Uses k-Nearest Neighbors to impute missing values in columns:
        - panel_temp
        - pv_power
        - irradiance_Wm-2
        These columns have, so far, been the columns with missing data.
        '''
        datetime_index = self.df_train.index.copy() # must make copy of datetime index to insert
        imputer_knn = KNNImputer(n_neighbors=5,
                                 weights='uniform',
                                 metric='nan_euclidean')
        X = self.df_train.values
        imputer_knn.fit(X)
        X_trans = imputer_knn.transform(X)
        self.df_train = pd.DataFrame(X_trans,
                                     columns=self.df_train.columns)
        self.df_train.set_index(datetime_index, inplace=True)

        print(f'Completed: missing values imputed using KNN model. \n Total missing values: {self.df_train.isna().sum().sum()}')

    def drop_missing_values(self):
        '''Drops all rows with missing values'''

        self.df_train.dropna(axis=0, how='any', inplace=True)
        self.df_taskweek.dropna(axis=0, how='any', inplace=True)

        print('Completed: missing values rows dropped from data')

    def create_pickles(self):
        '''Creates pickle files for easy access to dataframes
        In testing, needs to be more robust to catch different type of DFs being
        created.
        '''
        # self.df_train.to_pickle(f'../pickles/task{self.set}/df_train_dropna.pkl')
        # self.df_taskweek.to_pickle(f'../pickles/task{self.set}/df_taskweek.pkl')

    def rfr_model_irradiance(self):
        '''

        '''
        self.df_train_ir = self.df_train[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'irradiance_Wm-2']]

        # Process dataset

        # make sure the dataset is sorted
        self.df_train_ir.sort_index(inplace=True)

        # Create independent and dependent variable matrices
        X_ir = self.df_train_ir.iloc[:, :-1].values
        y_ir = self.df_train_ir.iloc[:, -1].values

        # Split datasets without shuffling
        self.X_ir_train, self.X_ir_test, self.y_ir_train, self.y_ir_test = train_test_split(X_ir, y_ir,
                                                                             test_size=336,
                                                                             shuffle=False)

        rfr_ir = RandomForestRegressor(n_estimators=100)
        rfr_ir.fit(self.X_ir_train, self.y_ir_train)

        self.y_pred_rfr_ir = rfr_ir.predict(self.X_ir_test)

        print('Completed: trained and predicted rfr for irradiance')

    def rfr_model_paneltemp(self):
        '''

        '''
        self.df_train_pt = self.df_train[['month', 'day_of_week', 'k_index', 'irradiance_Wm-2', 'temp_mean1256', 'solar_mean123456', 'panel_temp_C']]
        self.df_train_pt.loc[len(self.df_train_pt)-336:len(self.df_train_pt)+1, 'irradiance_Wm-2'] = self.y_pred_rfr_ir
        # for i in range(len(self.df_train_pt)-336, len(self.df_train_pt)+1):
        #     for j in range(0, len(self.y_pred_rfr_ir)):
        #         self.df_train_pt.loc[i, 'irradiance_Wm-2'] = self.y_pred_rfr_ir[j]

        # Process dataset

        # make sure the dataset is sorted
        self.df_train_pt.sort_index(inplace=True)

        # Create independent and dependent variable matrices
        X_pt = self.df_train_pt.iloc[:, :-1].values
        y_pt = self.df_train_pt.iloc[:, -1].values

        # Split datasets without shuffling
        self.X_pt_train, self.X_pt_test, self.y_pt_train, self.y_pt_test = train_test_split(X_pt, y_pt,
                                                                             test_size=336,
                                                                             shuffle=False)

        rfr_pt = RandomForestRegressor(n_estimators=20)
        rfr_pt.fit(self.X_pt_train, self.y_pt_train)

        self.y_pred_rfr_pt = rfr_pt.predict(self.X_pt_test)

        print('Completed: trained and predicted rfr for panel temp')

    def load_model(self, model_name):
        try:
            with open(f'../data/task{self.set}/{model_name}.pkl', 'rb') as file:
                return pickle.load(file)
        except IOError:
            print("No existing model saved, moving onto training")
            return None

    def rfr_model_train(self, y_var, features=None, n_estimators=100, load=False):
        '''

        '''
        # set of default features
        features = features or ['month', 'day_of_week', 'k_index',
                        'temp_mean1256', 'solar_mean123456']

        self.clf[y_var] = self.load_model(f'rfr_dm_n_{n_estimators}')
        df_train = self.df_train[features + [y_var]] # 'demand_MW' or 'pv_power_mw'

        # Process dataset
        # make sure the dataset is sorted

        # Create independent and dependent variable matrices
        X = self.df_train[features].values
        y = self.df_train[y_var].values

        # #Encode Categorical Data IF Needed
        # #One Hot Encoder
        # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3] )], remainder='passthrough')
        # X = np.array(ct.fit_transform(X))
        # X_svr = np.array(ct.fit_transform(X_svr))

        # Split datasets without shuffling
        # Test on the last week of data before task week
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size=336, shuffle=False)
        self.X_test = X_test
        self.y_test[y_var] = y_test
        if not self.clf[y_var]:
            # Random Forest Regression Models
            print(f"Training Random Forest Regressor for classifier on {y_var}")
            self.clf[y_var] = RandomForestRegressor(n_estimators=n_estimators,
                                           criterion='mae',
                                           verbose=(os.getenv('LOG_LEVEL') or 0)
                                           )
            self.clf[y_var].fit(X_train, y_train)
            # Random Forest Regression Prediction
            pkl_filename = f'../data/task{self.set}/rfr_{y_var}_n_{n_estimators}.pkl'
            with open(pkl_filename, 'wb') as file:
                pickle.dump(self.clf[y_var], file)

        self.y_pred[y_var] = self.clf[y_var].predict(X_test)


    def evaluate_model(self, y_var):
        '''

        '''
        print('\t\t\t\tR_Squared \tAdjusted R_Squared \tMean Squared Error')
        print(f'Random Forest Regression {y_var}:\t \
            {r2_score(self.y_test[y_var], self.y_pred[y_var])}\t \
            {1-(1-r2_score(self.y_test[y_var], self.y_pred[y_var]))*((len(self.y_test[y_var]) - 1)/(len(self.y_test[y_var])-len(self.X_test[0])-1))} \t\
            {mean_squared_error(self.y_test[y_var], self.y_pred[y_var])}')

        '''


    def plot_pred(self):
        '''

        '''
        fig, ax = plt.subplots(nrows=3, figsize=(40,25))
        ax1 = sns.lineplot(x=range(1, len(self.y_test['demand_MW'])+1),
                           y=self.y_test['demand_MW'],
                           color='black',
                           ax=ax[0],
                           label='Actual Demand')
        ax1 = sns.lineplot(x=range(1, len(self.y_pred['demand_MW'])+1),
                           y=self.y_pred['demand_MW'],
                           color='red',
                           ax=ax[0],
                           label='Predicted Demand')
        ax2 = sns.lineplot(x=range(1, len(self.y_test['pv_power_mw'])+1),
                           y=self.y_test['pv_power_mw'],
                           color='black',
                           ax=ax[1],
                           label='Actual PV_Power')
        ax2 = sns.lineplot(x=range(1, len(self.y_pred['pv_power_mw'])+1),
                           y=self.y_pred['pv_power_mw'],
                           color='orange',
                           ax=ax[1],
                           label='Predicted PV_Power')
        # ax3 = sns.lineplot(x=range(1, len(self.y_dm_test)+1),
        #                    y=self.y_pred_rfr_dm,
        #                    color='red',
        #                    ax=ax[2],
        #                    label='Predicted Demand')
        # ax3 = sns.lineplot(x=range(1, len(self.y_dm_test)+1),
        #                    y=self.y_pred_rfr_pv,
        #                    color='orange',
        #                    ax=ax[2],
        #                    label='Predicted PV_Power')
        plt.show()
