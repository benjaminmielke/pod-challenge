#!/usr/bin/env python
# coding: utf-8
import os

from challenge_week import PodChallenge


task_week = PodChallenge()
task_week.import_task_data('1')
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

task_week.evaluate_model('demand_MW')
task_week.evaluate_model('pv_power_mw')

task_week.plot_pred()
