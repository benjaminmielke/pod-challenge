#!/usr/bin/env python
# coding: utf-8
import os
os.chdir('C:/Users/okiem/github/pod-challenge/scripts')


from challenge_week import PodChallenge


tw = PodChallenge()
tw.import_task_data('1')
tw.merge_dataframes()
tw.fill_weather_data()
tw.insert_weather_means()
tw.insert_k_index()
tw.insert_time_cols()
tw.remove_outage()
tw.impute_knn()
# tw.drop_missing_values()
# tw.rfr_model_irradiance()
# tw.rfr_model_paneltemp()
tw.rfr_model_train()
tw.evaluate_model()
tw.plot_pred()
