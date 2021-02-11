#!/usr/bin/env python
# coding: utf-8

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os
pd.set_option('display.max_columns', 500)


# Importing dataset
path = os.getcwd()
path = 'C:/Users/okiem/github/pod-challenge'
# Dataframe that drops all missing data and keeps outage event
dataset_1 = pd.read_pickle(f'{path}/pickles/df_pv_demand_weather_dropna.pkl')
# Dataframe that removes outage event outliers then imputes all missing values
dataset_2 = pd.read_pickle(f'{path}/pickles/df_pv_demand_weather_nooutage.pkl')

# Create pv_power and demand prediction datasets, picking one of the above
# Select dataset_1 or dataset_2
selection = 2 # 1 or 2 --!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if selection ==1:
    dataset_demand = dataset_1[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'demand_MW']]
    dataset_pvpower = dataset_1[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'pv_power_mw']]
if selection == 2:
    dataset_demand = dataset_2[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'demand_MW']]
    dataset_pvpower = dataset_2[['month', 'k_index', 'solar_mean123456', 'pv_power_mw']]

# Process dataset

# make sure the dataset is sorted
dataset_demand.sort_index(inplace=True)
dataset_pvpower.sort_index(inplace=True)

# Create independent and dependent variable matrices
X_dm = dataset_demand.iloc[:, :-1].values
y_dm = dataset_demand.iloc[:, -1].values
X_pv = dataset_pvpower.iloc[:, :-1].values
y_pv = dataset_pvpower.iloc[:, -1].values

# #Encode Categorical Data
# #One Hot Encoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3] )], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# X_svr = np.array(ct.fit_transform(X_svr))

# Split datasets without shuffling
X_dm_train, X_dm_test, y_dm_train, y_dm_test = train_test_split(X_dm, y_dm, test_size=336, shuffle=False)
X_pv_train, X_pv_test, y_pv_train, y_pv_test = train_test_split(X_pv, y_pv, test_size=336, shuffle=False)

# Random Forest Regression Models
rfr_dm = RandomForestRegressor(n_estimators=1000)
rfr_dm.fit(X_dm_train, y_dm_train)
rfr_pv = RandomForestRegressor(n_estimators=100)
rfr_pv.fit(X_pv_train, y_pv_train)

# Random Forest Regression Prediction
y_pred_rfr_dm = rfr_dm.predict(X_dm_test)
y_pred_rfr_pv = rfr_pv.predict(X_pv_test)

# Evaluating the results of each model
print('                               R_Squared           Adjusted R_Squared    Mean Squared Error')
print(f'Random Forest Regression Demand:     {r2_score(y_dm_test, y_pred_rfr_dm)}   {1-(1-r2_score(y_dm_test, y_pred_rfr_dm))*((len(X_dm_test)-1)/(len(X_dm_test)-len(X_dm_test[0])-1))}    {mean_squared_error(y_dm_test, y_pred_rfr_dm)}')
print(f'Random Forest Regression Power:     {r2_score(y_pv_test, y_pred_rfr_pv)}   {1-(1-r2_score(y_pv_test, y_pred_rfr_pv))*((len(X_pv_test)-1)/(len(X_pv_test)-len(X_pv_test[0])-1))}    {mean_squared_error(y_pv_test, y_pred_rfr_pv)}')

# Plot the comparisons
fig, ax = plt.subplots(nrows=3, figsize=(40,25))
ax1 = sns.lineplot(x=range(1, len(y_dm_test)+1),
                   y=y_dm_test,
                   color='black',
                   ax=ax[0],
                   label='Actual Demand')
ax1 = sns.lineplot(x=range(1, len(y_dm_test)+1),
                   y=y_pred_rfr_dm,
                   color='red',
                   ax=ax[0],
                   label='Predicted Demand')
ax2 = sns.lineplot(x=range(1, len(y_dm_test)+1),
                   y=y_pv_test,
                   color='black',
                   ax=ax[1],
                   label='Actual PV_Power')
ax2 = sns.lineplot(x=range(1, len(y_dm_test)+1),
                   y=y_pred_rfr_pv,
                   color='orange',
                   ax=ax[1],
                   label='Predicted PV_Power')
ax3 = sns.lineplot(x=range(1, len(y_dm_test)+1),
                   y=y_pred_rfr_dm,
                   color='red',
                   ax=ax[2],
                   label='Predicted Demand')
ax3 = sns.lineplot(x=range(1, len(y_dm_test)+1),
                   y=y_pred_rfr_pv,
                   color='orange',
                   ax=ax[2],
                   label='Predicted PV_Power')
plt.savefig('C:/Users/okiem/OneDrive/Desktop/The_Project_Folder/Presumed_Open_Data_Science_Challenge/plot.png')


# Create datasheet for battery schedule optimization problem
optimize_datasheet = dataset_2.iloc[len(dataset_2)-336:]
optimize_datasheet.drop(['month', 'day_of_week', 'temp_mean1256', 'solar_mean123456'], axis=1, inplace=True)
optimize_datasheet.insert(3, 'PV_Power_Predicted', y_pred_rfr_pv)
optimize_datasheet.insert(4, 'Demand_Predicted', y_pred_rfr_dm)
optimize_datasheet.columns = ['k_index', 'PV_Power_Actual', 'Demand_Actual', 'PV_Power_Predicted', 'Demand_Predicted']
