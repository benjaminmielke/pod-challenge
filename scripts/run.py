#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import seaborn as sns
from challenge_week import PodChallenge

task_no = 1
task_week = PodChallenge()
task_week.import_task_data(task_no)
task_week.merge_dataframes()
task_week.fill_weather_data()
task_week.insert_weather_means()
task_week.insert_k_index()
task_week.insert_time_cols()
task_week.remove_outage()
task_week.impute_knn()
# task_week.rfr_model_irradiance()
# task_week.rfr_model_paneltemp()
task_week.rfr_model_train('demand_MW', n_estimators=100, load=True)
task_week.rfr_model_train('pv_power_mw', n_estimators=10, load=True)
task_week.y_pred['demand_MW']
task_week.df_train.index[-1]
task_week.evaluate_model('demand_MW')
task_week.evaluate_model('pv_power_mw')
# TODO: Save model to csv

# Calculate and visualize taskweek schedule
df_pred = pd.read_csv(f'../data/task{task_no}/task{task_no}_predictions.csv')
df_pred.columns
task_week.calculate_battery_schedule(
    df_pred['Demand_Prediction'], df_pred['PV_Power_Prediction'])
task_week.format_submit()


plt, ax = task_week.plot_taskweek()
ax2 = sns.lineplot(x=task_week.df_taskweek.index,
                   y=df_pred['Demand_Prediction'],
                   color='blue',
                   ax=ax[0],
                   label='Predicted demand')
ax1 = sns.lineplot(x=task_week.df_taskweek.index,
                   y=df_pred['PV_Power_Prediction'],
                   color='purple',
                   ax=ax[0],
                   label='Predicted PV output')
plt.show()

# Double check charging periods don't overlap
task_week.df_taskweek['discharge'].iloc[30:48]
task_week.df_taskweek['charge'].iloc[30:48]
