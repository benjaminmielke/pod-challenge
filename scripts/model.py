import numpy as np
from operator import itemgetter


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
    D_sorted = np.array(sorted(list(enumerate(D)), key=itemgetter(1)))

    candidate_solns = [(np.sum(D_sorted[i:, 1]) - charge)/len(D[i:])
                       for i in range(len(D_sorted))]
    bounds = [[D_sorted[i][1], D_sorted[i + 1][1]]
              for i in range(len(D_sorted) - 1)]
    bounds.insert(0, [0, 1])
    for i, soln in enumerate(candidate_solns):
        if soln >= bounds[i][0] and soln <= bounds[i][1]:
            return list(np.clip(np.array(D) - soln, 0, None))
    return None
