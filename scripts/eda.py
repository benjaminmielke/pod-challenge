ax[0][1]ax[0][1]# #Exploratory Data Analysis

import pandas as pd
from statistics import mean
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

path_data = 'C:/Users/okiem/OneDrive/Desktop/The_Project_Folder/Presumed_Open_Data_Science_Challenge/data/task0'

# **Import Data**
df_demand = pd.read_csv(f'{path_data}/demand_train_set0.csv')
df_demand.head()
df_pv = pd.read_csv(f'{path_data}/pv_train_set0.csv')
df_pv.head()
df_weather = pd.read_csv(f'{path_data}/weather_train_set0.csv')
df_weather.head()
df_submit = pd.read_csv(f'{path_data}/teamname_set0.csv')
df_submit.tail()

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

# Join pv_demand set with weather_set0 set to complete the model dataframe
df_pv_demand_weather = df_pv_demand.merge(right=df_weather_set0,
                                          on='datetime',
                                          how='outer')


# ------------------------------------------------------------------------------
def impute_weather(lst_cols):
    '''Imputes the missing data for the weather paramters using the mean of the
    timesteps before and after. '''
    for c in range(0, len(lst_cols)):
        for i in range(0, len(df_pv_demand_weather)-1):
            if pd.isnull(df_pv_demand_weather[lst_cols[c]].iloc[i]):
                df_pv_demand_weather.loc[i, lst_cols[c]] = mean([df_pv_demand_weather.loc[i-1, lst_cols[c]], df_pv_demand_weather.loc[i+1, lst_cols[c]]])
# ------------------------------------------------------------------------------


# Impute NaN values for weather features
impute_weather(df_pv_demand_weather.columns.values[5:])

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

df_pv_demand_weather.tail(100)

# ==============================================================================
# ==============================================================================

fig, ax = plt.subplots(3, 2, figsize=(20,14))
ax1 = sns.lineplot(x='k_index',
                   y='irradiance_Wm-2',
                   data=df_pv_demand_weather,
                   color='orange',
                   ax=ax[0][0],
                   label='Irradiance_Panel')
ax1 = sns.lineplot(x='k_index',
                   y='solar_location3',
                   data=df_pv_demand_weather,
                   color='yellow',
                   ax=ax[0][0],
                   label='Irradiance_Sun')

ax1.set_title('Solar Weather Station 3',
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
                   y='irradiance_Wm-2',
                   data=df_pv_demand_weather,
                   color='orange',
                   ax=ax[0][1],
                   label='Irradiance_Panel')
ax3 = sns.lineplot(x='k_index',
                   y='solar_location4',
                   data=df_pv_demand_weather,
                   color='yellow',
                   ax=ax[0][1],
                   label='Irradiance_Sun')
ax3.set_title('Solar Weather Station 4',
              fontsize=12)
ax3.set_xlabel('K_Index(k is each half hour of day)',
               fontsize=10)
ax3.set_ylabel('Solar Iraddiance(Wm-2)',
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
                   y='irradiance_Wm-2',
                   data=df_pv_demand_weather,
                   color='orange',
                   ax=ax[1][0],
                   label='Irradiance_Panel')
ax5 = sns.lineplot(x='k_index',
                   y='solar_location1',
                   data=df_pv_demand_weather,
                   color='yellow',
                   ax=ax[1][0],
                   label='Irradiance_Sun')
ax5.set_title('Solar Weather Station 1',
              fontsize=12)
ax5.set_xlabel('K_Index(k is each half hour of day)',
               fontsize=10)
ax5.set_ylabel('Solar Iraddiance(Wm-2)',
               fontsize=10,
               color='black')
ax5.tick_params(axis='y')
ax5.set(xlim=(0, 48))
ax6 = ax5.twinx()
ax6.set_ylabel('Demand(MW)',
               fontsize=10,
               color='red')
ax6 = sns.lineplot(x='k_index',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   label='Demand')
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
ax7 = sns.lineplot(x='k_index',
                   y='irradiance_Wm-2',
                   data=df_pv_demand_weather,
                   color='orange',
                   ax=ax[1][1],
                   label='Irradiance_Panel')
ax7 = sns.lineplot(x='k_index',
                   y='solar_location2',
                   data=df_pv_demand_weather,
                   color='yellow',
                   ax=ax[1][1],
                   label='Irradiance_Sun')
ax7.set_title('Solar Weather Station 2',
              fontsize=12)
ax7.set_xlabel('K_Index(k is each half hour of day)',
               fontsize=10)
ax7.set_ylabel('Solar Iraddiance(Wm-2)',
               fontsize=10,
               color='black')
ax7.tick_params(axis='y')
ax7.set(xlim=(0, 48))
ax8 = ax7.twinx()
ax8.set_ylabel('Demand(MW)',
               fontsize=10,
               color='red')
ax8 = sns.lineplot(x='k_index',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   label='Demand')
ax8.tick_params(axis='y',
                color='red')
ax7.axvline(32, 0, 1,
            color='blue')
ax7.axvline(42, 0, 1,
            color='blue')
ax7.axvline(1, 0, 1,
            color='green')
ax7.axvline(31, 0, 1,
            color='green')
ax7.legend(loc='upper left')
# -----------------------------
ax9 = sns.lineplot(x='k_index',
                   y='irradiance_Wm-2',
                   data=df_pv_demand_weather,
                   color='orange',
                   ax=ax[2][0],
                   label='Irradiance_Panel')
ax9 = sns.lineplot(x='k_index',
                   y='solar_location5',
                   data=df_pv_demand_weather,
                   color='yellow',
                   ax=ax[2][0],
                   label='Irradiance_Sun')
ax9.set_title('Solar Weather Station 5',
              fontsize=12)
ax9.set_xlabel('K_Index(k is each half hour of day)',
               fontsize=10)
ax9.set_ylabel('Solar Iraddiance(Wm-2)',
               fontsize=10,
               color='black')
ax9.tick_params(axis='y')
ax9.set(xlim=(0, 48))
ax10 = ax9.twinx()
ax10.set_ylabel('Demand(MW)',
               fontsize=10,
               color='red')
ax10 = sns.lineplot(x='k_index',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   label='Demand')
ax10.tick_params(axis='y',
                color='red')
ax9.axvline(32, 0, 1,
            color='blue')
ax9.axvline(42, 0, 1,
            color='blue')
ax9.axvline(1, 0, 1,
            color='green')
ax9.axvline(31, 0, 1,
            color='green')
ax9.legend(loc='upper left')
# -----------------------------
ax11 = sns.lineplot(x='k_index',
                   y='irradiance_Wm-2',
                   data=df_pv_demand_weather,
                   color='orange',
                   ax=ax[2][1],
                   label='Irradiance_Panel')
ax11 = sns.lineplot(x='k_index',
                   y='solar_location6',
                   data=df_pv_demand_weather,
                   color='yellow',
                   ax=ax[2][1],
                   label='Irradiance_Sun')
ax11.set_title('Solar Weather Station 6',
              fontsize=12)
ax11.set_xlabel('K_Index(k is each half hour of day)',
               fontsize=10)
ax11.set_ylabel('Solar Iraddiance(Wm-2)',
               fontsize=10,
               color='black')
ax11.tick_params(axis='y')
ax11.set(xlim=(0, 48))
ax12 = ax11.twinx()
ax12.set_ylabel('Demand(MW)',
               fontsize=10,
               color='red')
ax12 = sns.lineplot(x='k_index',
                   y='demand_MW',
                   data=df_pv_demand_weather,
                   color='red',
                   label='Demand')
ax12.tick_params(axis='y',
                color='red')
ax11.axvline(32, 0, 1,
            color='blue')
ax11.axvline(42, 0, 1,
            color='blue')
ax11.axvline(1, 0, 1,
            color='green')
ax11.axvline(31, 0, 1,
            color='green')
ax11.legend(loc='upper left')
plt.show()

# ------------------------------------------------------------------------------

correlation = df_pv_demand_weather.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(correlation, dtype=bool))
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

# ------------------------------------------------------------------------------
