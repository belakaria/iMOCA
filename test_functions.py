import math
from copy import deepcopy

import numpy as np


def get_benchmark_functions(names="branin,Currin"):
    """
    Returns a list of benchmark functions.
    """
    names = names.split(",")
    available_functions = {
        "branin": cmfbranin,
        "Currin": cmfCurrin,
        # Add other functions here as needed
    }
    functions = []
    for name in names:
        if name in available_functions:
            functions.append(available_functions[name])
        else:
            raise ValueError(f"Function '{name}' is not available.")
    return functions


def get_cost_functions(names="branin,Currin"):
    """
    Returns a list of cost functions.
    """
    names = names.split(",")
    available_cost_functions = {
        "branin": branin_cost,
        "Currin": currin_cost,
        # Add other cost functions here as needed
    }
    cost_functions = []
    for name in names:
        if name in available_cost_functions:
            cost_functions.append(available_cost_functions[name])
        else:
            raise ValueError(f"Cost function '{name}' is not available.")
    return cost_functions


def cmfCurrin(x1, d, f1):
    x = deepcopy(x1)
    f = deepcopy(f1)
    f = (f + 1) / 20
    if x[1] == 0:
        x[1] = 1e-100
    return -1 * float(
        (
            (1 - 0.1 * (1 - f) - math.exp(-0.5 * (1 / x[1])))
            * (
                (2300 * pow(x[0], 3) + 1900 * x[0] * x[0] + 2092 * x[0] + 60)
                / (100 * pow(x[0], 3) + 500 * x[0] * x[0] + 4 * x[0] + 20)
            )
        )
    )


def cmfbranin(x1, d, f1):
    x = deepcopy(x1)
    f = deepcopy(f1)
    f = (f + 1) / 20
    x[0] = 15 * x[0] - 5
    x[1] = 15 * x[1]
    return -1 * float(
        np.square(
            x[1]
            - (5.1 / (4 * np.square(math.pi)) - 0.01 * (1 - f)) * np.square(x[0])
            + (5 / math.pi - 0.1 * (1 - f)) * x[0]
            - 6
        )
        + 10 * (1 - (1.0 / (8 * math.pi) + 0.05 * (1 - f))) * np.cos(x[0])
        + 10
    )


def branin_cost(z):
    return 0.05 + pow(z, 6.5)


def currin_cost(z):
    return 0.05 + pow(z, 2)
