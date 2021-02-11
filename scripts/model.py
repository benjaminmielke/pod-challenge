import numpy as np
import pandas as pd
from operator import itemgetter
import os
from matplotlib import pyplot as plt


def charge_policy(irradiance, solar_mw, capacity):
    """
    Returns the charging schedule for the batteries given predicted
    solar power curve
    Args:
        solar_mw (list): predicted solar output over the charging period
    """
    # Want to charge to 6MWh, simple way to do that should be to charge
    # proportionally based on how much irradiance is expected

    c = np.sum(solar_mw)/capacity
    # Note that c < 1, in which case we'll draw proportionally more power
    # from the grid when solar is expected to be high then when it is low
    return np.array(solar_mw)/c


def discharge_policy(charge, D):
    """
    Returns optimal discharge schedule for a given demand profile
    to reduce peak load
    Args:
        charge (float): charge of battery at beginning of discharge
        L (list[float]): forecasted demand
    Returns:
        discharge_profile: list
    """
    charge = 2*charge  # Change units to MWh/2
    D_sorted = np.array(sorted(list(enumerate(D)), key=itemgetter(1)))

    candidate_solns = [(np.sum(D_sorted[i:, 1]) - charge)/len(D[i:])
                       for i in range(len(D_sorted))]
    bounds = [[D_sorted[i][1], D_sorted[i + 1][1]]
              for i in range(len(D_sorted) - 1)]
    bounds.insert(0, [0, bounds[0][0]])
    for i, soln in enumerate(candidate_solns):
        if soln >= bounds[i][0] and soln <= bounds[i][1]:
            return list(np.clip(np.array(D) - soln, 0, None))
    return None


dataset_1 = pd.read_pickle(f'{path}/pickles/df_pv_demand_weather_dropna.pkl')
df_demand = pd.read_csv(os.path.join(
    os.getcwd(), 'scripts/task0_rfr_pred.csv')
# df_demand = pd.read_csv('predictions.csv')
df_demand['d_index'] = np.array(df_demand.index + 1) // 48 + 1
# Want to ideally turn this into a multi-index
df_demand
df_demand['Demand_Predicted(MW)'] = df_demand['Demand_Predicted']
df_demand['k_index'] = np.mod((np.array(df_demand.index)), 48) + 1
df_demand
peak_indices_mask = (df_demand['k_index'] >= 32) & (df_demand['k_index'] < 43)
demand_pred_peak_hours = df_demand[peak_indices_mask]['Demand_Predicted(MW)']

df_demand['discharge'] = 0
for d in range(1, 8):
    battery_capacity = 6  # MWh
    print(list(demand_pred_peak_hours[(df_demand['d_index'] == d)]))
    discharge_schedule = discharge_policy(battery_capacity, list(
        demand_pred_peak_hours[(df_demand['d_index'] == d)]))
    # discharge_schedule = pd.Series(discharge_schedule, index=[])
    df_demand.loc[peak_indices_mask & (
        df_demand['d_index'] == d), 'discharge'] = discharge_schedule

df_demand['discharge']
df_demand['grid_supply'] = pd.Series(df_demand['Demand_Predicted(MW)'].to_numpy() - df_demand['discharge'].to_numpy(), index=df_demand.index)
df_demand['grid_supply'].iloc[32:42]
df_demand['Demand_Predicted(MW)'].iloc[32:42]
df_demand[['Demand_Predicted(MW)', 'discharge', 'grid_supply']].plot()

# Check that we use up the battery
assert(np.abs(np.sum(discharge_schedule)/2 - battery_capacity) < 1e-4)
all(np.array(discharge_schedule) < 2.5)
