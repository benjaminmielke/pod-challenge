# #Exploratory Data Analysis

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

# import dataframe pickle
path = os.getcwd()
df_pv_demand_weather = pd.read_pickle(f'{path}/pickles/df_set0_dropna.pkl')


df_pv_demand_weather.tail()


fig, ax = plt.subplots(2, 3, figsize=(30,15))
fig.suptitle('Time Series Trends of Demand Compared to Daily Trends of Relavent Features', fontsize=20)
ax1 = sns.lineplot(x='k_index',
                   y='irradiance_Wm-2',
                   data=df_pv_demand_weather,
                   color='orange',
                   ax=ax[1][0],
                   label='Irradiance_Panel')
ax1 = sns.lineplot(x='k_index',
                   y='solar_mean123456',
                   data=df_pv_demand_weather,
                   color='gold',
                   ax=ax[1][0],
                   label='Irradiance_Sun')

ax1.set_title('Irradiance of Panel vs. Mean Irradiance of Weather Stations',
              fontsize=12)
ax1.set_xlabel('K_Index(k is each half hour of day)',
               fontsize=10)
ax1.set_ylabel('Solar Iraddiance(Wm-2)',
               fontsize=10,
               color='black')
ax1.tick_params(axis='y')
ax1.set(xlim=(0, 48))
ax2 = ax1.twinx()
ax2.set_ylabel('Demand(MW)',
               fontsize=10,
               color='red')
ax2 = sns.lineplot(x='k_index',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   label='Demand')
ax2.tick_params(axis='y',
                color='red')
ax1.axvline(32, 0, 1,
            color='blue')
ax1.axvline(42, 0, 1,
            color='blue')
ax1.axvline(1, 0, 1,
            color='green')
ax1.axvline(31, 0, 1,
            color='green')
ax1.legend(loc='upper left')
# -----------------------------
ax3 = sns.lineplot(x='k_index',
                   y='panel_temp_C',
                   data=df_pv_demand_weather,
                   color='darkred',
                   ax=ax[1][1],
                   label='Panel_Temp')
ax3 = sns.lineplot(x='k_index',
                   y='temp_mean1256',
                   data=df_pv_demand_weather,
                   color='orangered',
                   ax=ax[1][1],
                   label='Temp_Sun')
ax3.set_title('Temperature of Panel vs. Mean Temperature of Weather Stations 1,2,5,6',
              fontsize=12)
ax3.set_xlabel('K_Index(k is each half hour of day)',
               fontsize=10)
ax3.set_ylabel('Temperature(C)',
               fontsize=10,
               color='black')
ax3.tick_params(axis='y')
ax3.set(xlim=(0, 48))
ax4 = ax3.twinx()
ax4.set_ylabel('Demand(MW)',
               fontsize=10,
               color='red')
ax4 = sns.lineplot(x='k_index',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   label='Demand')
ax4.tick_params(axis='y',
                color='red')
ax3.axvline(32, 0, 1,
            color='blue')
ax3.axvline(42, 0, 1,
            color='blue')
ax3.axvline(1, 0, 1,
            color='green')
ax3.axvline(31, 0, 1,
            color='green')
ax3.legend(loc='upper left')
# -----------------------------
ax5 = sns.lineplot(x='k_index',
                   y='pv_power_mw',
                   data=df_pv_demand_weather,
                   color='darkseagreen',
                   ax=ax[1][2],
                   label='PV_Power')
ax5 = sns.lineplot(x='k_index',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   ax=ax[1][2],
                   label='Demand')
ax5.set_title('PV Power and Demand Compared to Irradiance From Sun',
              fontsize=12)
ax5.set_xlabel('K_Index(k is each half hour of day)',
               fontsize=10)
ax5.set_ylabel('Solar Iraddiance(Wm-2)',
               fontsize=10,
               color='black')
ax5.tick_params(axis='y')
ax5.set(xlim=(0, 48))
ax6 = ax5.twinx()
ax6.set_ylabel('Sun Irradiance(Wm-2)',
               fontsize=10,
               color='red')
ax6 = sns.lineplot(x='k_index',
                   y='solar_mean123456',
                   data=df_pv_demand_weather,
                   color='gold',
                   label='Irransiance_Sun')
ax6.tick_params(axis='y',
                color='red')
ax5.axvline(32, 0, 1,
            color='blue')
ax5.axvline(42, 0, 1,
            color='blue')
ax5.axvline(1, 0, 1,
            color='green')
ax5.axvline(31, 0, 1,
            color='green')
ax5.legend(loc='upper left')
# -----------------------------
ax7 = sns.lineplot(x='datetime',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   ax=ax[0][0],
                   label='Demand')
ax7.set_title('Demand Time Series',
              fontsize=12)
ax7.set_xlabel('Date',
               fontsize=10)
ax7.set_ylabel('Demand(MW)',
               fontsize=10,
               color='black')
ax7.tick_params(axis='y')
ax7.legend(loc='upper left')
# -----------------------------
ax9 = sns.lineplot(x='month',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   ax=ax[0][1],
                   label='Demand')
ax9.set_title('Demand Monthly Seasonality',
              fontsize=12)
ax9.set_xlabel('Month',
               fontsize=10)
ax9.set_ylabel('Demand(MW)',
               fontsize=10,
               color='black')
ax9.tick_params(axis='y')
ax9.legend(loc='upper left')
# -----------------------------
ax11 = sns.lineplot(x='day_of_week',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   ax=ax[0][2],
                   label='Demand')
ax11.set_title('Demand Day-Of-Week Trend',
              fontsize=12)
ax11.set_xlabel('Day of Week(0 is Monday)',
               fontsize=10)
ax11.set_ylabel('Demand(MW)',
               fontsize=10,
               color='black')
ax11.tick_params(axis='y')
plt.savefig(f'{path}/figs/demand_trend_2by3.png')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


correlation = df_pv_demand_weather.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(correlation, dtype=bool))
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(18, 16))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.savefig(f'{path}/figs/correlation_matrix.png')

# ------------------------------------------------------------------------------
