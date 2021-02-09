#!/usr/bin/env python
# coding: utf-8

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# Importing dataset
path_data = 'C:/Users/okiem/OneDrive/Desktop/The_Project_Folder/Presumed_Open_Data_Science_Challenge/data/task0'
path = 'C:/Users/okiem/OneDrive/Desktop/The_Project_Folder/Presumed_Open_Data_Science_Challenge/pod-challenge'
dataset_submit = pd.read_pickle(f'{path}/pickles/df_submit_dropna.pkl')
dataset = pd.read_pickle(f'{path}/pickles/df_set0_dropna.pkl')
dataset_fullmodel = dataset + dataset_submit

# Process dataset

# Move demand_mw to last column in dataset
new_col = dataset.pop('demand_MW')
dataset.insert(21, 'demand_MW', new_col)

# make diferent datasets to run model on
dataset = dataset[['month', 'day_of_week', 'k_index', 'temp_mean1256', 'solar_mean123456', 'demand_MW']]
# dataset.drop(['day_of_month', 'irradiance_Wm-2', 'pv_power_mw', 'panel_temp_C', 'temp_mean1256', 'solar_mean123456'], axis=1, inplace=True)

# make sure the dataset is sorted
dataset.sort_index(inplace=True)

# Create independent and dependent variable matrices
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_svr = dataset.iloc[:, :-1].values
y_svr = dataset.iloc[:, -1].values
y_svr = y_svr.reshape(len(y_svr),1)

# #Encode Categorical Data
# #One Hot Encoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3] )], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# X_svr = np.array(ct.fit_transform(X_svr))

# Split dataset without shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# Splitting sets for SVR
X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(X_svr, y_svr, test_size=0.2, shuffle=False)

# Feature Scaling for SVR
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_svr = sc_X.fit_transform(X_train_svr)
y_train_svr = sc_y.fit_transform(y_train_svr)

# Multiple Linear Regression
regressor_mlr = LinearRegression()
regressor_mlr.fit(X_train, y_train)

# Polynomial Linear Regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
regressor_plr = LinearRegression()
regressor_plr.fit(X_poly, y_train)

# Support Vector Regression
regressor_svr = SVR(kernel='rbf')
regressor_svr.fit(X_train_svr, y_train_svr)

# Decision Tree Regression
regressor_dtr = DecisionTreeRegressor(random_state=0)
regressor_dtr.fit(X_train, y_train)

# Random Forest Regression
regressor_rfr = RandomForestRegressor(n_estimators=900, random_state=0)
regressor_rfr.fit(X_train, y_train)

# XGBoost Regression
regressor_xgb = XGBRegressor()
regressor_xgb.fit(X_train, y_train)

# Multiple Linear Regression Prediction
y_pred_mlr = regressor_mlr.predict(X_test)
# Polynomial Linear Regression Prediction
y_pred_plr = regressor_plr.predict(poly_reg.transform(X_test))
# Support Vector Regression Prediction
y_pred_svr = sc_y.inverse_transform(regressor_svr.predict(sc_X.transform(X_test_svr)))
# Decision Tree Regression Prediction
y_pred_dtr = regressor_dtr.predict(X_test)
# Random Forest Regression Prediction
y_pred_rfr = regressor_rfr.predict(X_test)
# XGBoost Regression Prediction
y_pred_xgb = regressor_xgb.predict(X_test)


# Evaluating the results of each model
print('                               R_Squared           Adjusted R_Squared\n')
print(f'Multiple Linear Regression:   {r2_score(y_test, y_pred_mlr)}   {1-(1-r2_score(y_test, y_pred_mlr))*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))}')
print(f'Polynomial Linear Regression: {r2_score(y_test, y_pred_plr)}   {1-(1-r2_score(y_test, y_pred_plr))*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))}')
print(f'Support Vector Regression:    {r2_score(y_test, y_pred_svr)}   {1-(1-r2_score(y_test, y_pred_svr))*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))}')
print(f'Decision Tree Regression:     {r2_score(y_test, y_pred_dtr)}   {1-(1-r2_score(y_test, y_pred_dtr))*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))}')
print(f'Random Forest Regression:     {r2_score(y_test, y_pred_rfr)}   {1-(1-r2_score(y_test, y_pred_rfr))*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))}')
print(f'XGBoost Regression:     {r2_score(y_test, y_pred_xgb)}   {1-(1-r2_score(y_test, y_pred_xgb))*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))}')


plt.figure(figsize=(16,5))
plt.plot(range(0,len(y_test)), y_test, color='dimgray', linewidth=.5)
plt.plot(range(0,len(y_pred_rfr)), y_pred_rfr, color='blue')
# plt.plot(range(0,len(y_pred_svr[2450:])), y_pred_svr[2450:], color='darkblue')
# plt.plot(range(0,len(y_pred_dtr[2450:])), y_pred_dtr[2450:], color='skyblue')
plt.show

dataset.insert(6, 'demand_predicted_mw', None)
len(y_pred_rfr)

dataset.info()
