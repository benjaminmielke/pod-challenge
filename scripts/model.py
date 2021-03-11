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
    capacity = 2 * capacity  # MWh to MW-half-hours
    c = np.sum(solar_mw) / capacity
    # Note that c < 1, in which case we'll draw proportionally more power
    # from the grid when solar is expected to be high then when it is low
    return np.array(solar_mw) / c


def charge_by_quantile(solar_mw_low, solar_mw_med, capacity, charge_threshold=2.5):
    """
    Returns the charging schedule for batteries given a lower,
    more assured amount of expected solar power, and then a mid or expected
    solar power amount
    Args:
        solar_mw_low (list): pick a lower quantile prediction
        solar_mw_med (list): either 50% quantile or output of some other
            predictive model which estimates median or expected value
    """
    capacity = 2 * capacity  # MWh to MW-half-hours
    solar_mw_low = np.array(solar_mw_low)
    solar_mw_med = np.array(solar_mw_med)
    # First, allocate as much charge as possible to the solar_mw_low
    total_min_solar = np.sum(solar_mw_low)
    schedule = None
    if total_min_solar > capacity:
        schedule = charge_policy(None, solar_mw_low, capacity/2)
    else:
        low_capture = solar_mw_low
        remaining_available = solar_mw_med - solar_mw_low
        remaining_capacity = capacity - total_min_solar
        med_capture = charge_policy(None, remaining_available, remaining_capacity/2)
        schedule = low_capture + med_capture

    # Now we want to deal with all periods that have charges > threshold
    excess = 1.0
    while excess > 1.0e-2:
        excess = np.sum(schedule[schedule - np.full(schedule.shape, 2.5) > 0] - 2.5)
        capped_schedule = np.where(schedule > 2.5, 2.5, schedule)
        # hack - set available power to 0 where we've hitting charging threshold
        remaining_available = np.where(schedule > 2.5, 0, remaining_available)
        if excess > 1.0e-2:
            reallocate_remainder = charge_policy(None, remaining_available, excess/2)
            capped_schedule
        schedule = capped_schedule
    return schedule





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
    charge = 2 * charge  # Change units to MWh/2
    D_sorted = np.array(sorted(list(enumerate(D)), key=itemgetter(1)))

    candidate_solns = [(np.sum(D_sorted[i:, 1]) - charge) / len(D[i:])
                       for i in range(len(D_sorted))]
    bounds = [[D_sorted[i][1], D_sorted[i + 1][1]]
              for i in range(len(D_sorted) - 1)]
    bounds.insert(0, [0, bounds[0][0]])
    for i, soln in enumerate(candidate_solns):
        if soln >= bounds[i][0] and soln <= bounds[i][1]:
            return list(np.clip(np.array(D) - soln, 0, None))
    return None
