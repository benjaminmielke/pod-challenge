#!/usr/bin/env python
# coding: utf-8
import os
os.chdir('C:/Users/okiem/github/pod-challenge/scripts')


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
# task_week.drop_missing_values()
# task_week.rfr_model_irradiance()
# task_week.rfr_model_paneltemp()
task_week.rfr_model_train()
task_week.evaluate_model()
task_week.plot_pred()
