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
    capacity = 2*capacity  # MWh to MW-half-hours
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


path = os.getcwd()
dataset_1 = pd.read_pickle(f'{path}/pickles/df_pv_demand_weather_dropna.pkl')
df_pred = pd.read_csv(os.path.join(
    os.getcwd(), 'scripts/task0_rfr_pred.csv'))

# df_pred = pd.read_csv('predictions.csv')
df_pred['d_index'] = (np.array(df_pred.index + 1) // 48) + 1
# Want to ideally turn this into a multi-index
df_pred
df_pred['Demand_Predicted(MW)'] = df_pred['Demand_Predicted']
df_pred['k_index'] = np.mod((np.array(df_pred.index)), 48) + 1
df_pred.columns
peak_indices_mask = (df_pred['k_index'] >= 32) & (df_pred['k_index'] < 43)
charge_indices_mask = (df_pred['k_index'] < 32)
demand_pred_peak_hours = df_pred[peak_indices_mask]['Demand_Predicted(MW)']
df_charge_hours = df_pred[charge_indices_mask]
df_pred['discharge'] = 0.0
df_pred['charge'] = 0.0
for d in range(1, 8):
    battery_capacity = 6  # MWh
    print(list(demand_pred_peak_hours[(df_pred['d_index'] == d)]))
    charge_schedule = charge_policy(
        None, list(df_charge_hours[(df_charge_hours['d_index'] == d)]['PVPower_Predicted']), battery_capacity)
    discharge_schedule = discharge_policy(battery_capacity, list(
        demand_pred_peak_hours[(df_pred['d_index'] == d)]))
    df_pred.loc[charge_indices_mask & (
        df_pred['d_index'] == d), 'charge'] = charge_schedule
    df_pred.loc[peak_indices_mask & (
        df_pred['d_index'] == d), 'discharge'] = discharge_schedule
    assert(all(np.array(discharge_schedule) < 2.5))
    assert(all(np.array(charge_schedule) < 2.5))
    assert(np.abs(np.sum(charge_schedule)/2 - battery_capacity) < 1e-4)
    assert(np.abs(np.sum(discharge_schedule)/2 - battery_capacity) < 1e-4)

df_pred['discharge']
df_pred['grid_supply'] = pd.Series(df_pred['Demand_Predicted(MW)'].to_numpy(
) - df_pred['discharge'].to_numpy(), index=df_pred.index)
df_pred['grid_supply'].iloc[32:42]
df_pred['Demand_Predicted(MW)'].iloc[32:42]

df_pred['charge_MW'] = df_pred['charge'] - df_pred['discharge']

df_pred[['Demand_Predicted(MW)', 'charge', 'discharge', 'grid_supply', 'charge_MW']].plot()
df_pred[['Demand_Predicted(MW)', 'grid_supply', 'charge_MW']].plot()
df_pred.columns
df_pred['charge'].iloc[0:40]
df_pred[['datetime', 'charge_MW']].to_csv('lightening_voltage_set0.csv', index=False)
# Check that we use up the battery
