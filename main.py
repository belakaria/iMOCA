# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:57:25 2019

@author: Syrine Belakaria
"""
import sys

import numpy as np
from platypus import NSGAII, Problem, Real
from scipy.optimize import minimize as scipyminimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import mfmes as MFBO
import mfmodel as MFGP
from test_functions import get_benchmark_functions, get_cost_functions

######################Algorithm input##############################

args = sys.argv[1:]
function_names = args[0]
functions = get_benchmark_functions("branin,Currin")
cost_functions = get_cost_functions("branin,Currin")
d = int(args[1])
seed = int(args[2])
intial_number = int(args[3])
total_iterations = int(args[4])
sample_number = int(args[5])
fidelity_grid_size = int(args[6])  # number of fidelities
approx = args[7]  # 'TG' for True Gaussian, 'EG' for Empirical Gaussian
paths = "."
np.random.seed(seed)


fir_num = 1
# defining the cost for each fidelity
# for example if we have 3 fidelities where the first fidelity cost 1 min and the last cost 10 min, the cost vector should be
# cost=[1,3,10]
cost = [
    [
        cost_functions[i](z / fidelity_grid_size)
        for z in range(1, fidelity_grid_size + 1)
    ]
    for i in range(len(functions))
]
referencePoint = [1e5] * len(functions)
bound = [0, 1]
Fun_bounds = [bound] * d

# Initialization
M = [len(i) for i in cost]
temp0 = list(np.random.choice(fidelity_grid_size - 1, fir_num, replace=False)) + [
    fidelity_grid_size - 1
]
temps0 = np.zeros(fidelity_grid_size)
for i in temp0:
    temps0[i] = 1

fidelity_iter = [temps0 for i in range(len(M))]
total_cost = sum(
    [
        sum(
            [
                (float(cost[i][m]) / cost[i][M[i] - 1]) * fidelity_iter[i][m]
                for m in range(M[i])
            ]
        )
        for i in range(len(M))
    ]
)
allcosts = [total_cost]

x = np.random.uniform(bound[0], bound[1], size=(1000, d))
candidate_x = [np.asarray([]) for i in range(len(functions))]
temp1 = list(np.random.randint(0, np.size(x, 0), fir_num + 1))

for i in range(len(functions)):
    candidate_x[i] = np.c_[temp0 * np.ones(1), x[temp1]]


# Create data from functions
y = [[] for i in range(len(functions))]
for xi in candidate_x[i]:
    for i in range(len(functions)):
        y[i].append(functions[i](xi[1:], d, xi[0]))
# Kernel configuration
kernel_f = 1.0 * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
kernel_z = 1.0 * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
new_kernel = MFGP.MTGPKernel(kernel_f, kernel_z)

###################GP Initialisation##########################
GPs = []
MFMES = []
GP_index = []
func_samples = []
acq_funcs = []
xValues = []
for i in range(len(functions)):
    GPs.append(GaussianProcessRegressor(kernel=new_kernel, n_restarts_optimizer=5))
    MFMES.append(0)
    func_samples.append([])
    acq_funcs.append([])
experiment_num = 0
cost_input_output = open(str(experiment_num) + approx + "_input_output.txt", "a")


print("total_cost:", total_cost)


for i in range(len(y[0])):
    cost_input_output.write(
        str(np.array([candidate_x[k][i][0] for k in range(len(functions))]))
    )
    cost_input_output.write(str(candidate_x[0][i][1:]))
    cost_input_output.write(
        str(np.array([y[k][i] for k in range(len(functions))])) + "\n"
    )


def compute_beta(iter_num, d, GP):
    l = GP.kernel.kernel_f.get_params()["k2__length_scale"]
    beta = 0.125 * d * np.log(2 * l * iter_num + 1)
    return beta


for j in range(1, total_iterations):
    beta = [compute_beta(j, d, GPs[i]) for i in range(len(functions))]
    for i in range(len(functions)):
        if j % 20 != 0:
            GPs[i].fit(candidate_x[i], y[i])
        else:
            GPs[i].fit(candidate_x[i], y[i])
            MFGP.optimize_scale(
                GPs[i].kernel,
                GPs[i].X_train_,
                GPs[i].y_train_,
                beta=1e6,
                error_opt=True,
            )
        # Acquisition function calculation
    for i in range(len(functions)):
        MFMES[i] = MFBO.MultiFidelityMaxvalueEntroySearch(
            M[i],
            cost[i],
            size=1,
            beta=1e3,
            RegressionModel=GPs[i],
            approximation=approx,
            sampling_num=sample_number,
        )
        MFMES[i].Sampling_RFM()
    max_samples = []
    for i in range(sample_number):
        for k in range(len(functions)):
            MFMES[k].weigh_sampling()
        cheap_pareto_front = []

        def schaffer(xi):
            global beta
            y = [
                MFMES[i].f_regression(np.asarray([[M[i] - 1] + xi]))[0][0]
                for i in range(len(GPs))
            ]
            return y

        problem = Problem(d, len(functions))
        problem.types[:] = Real(bound[0], bound[1])
        problem.function = schaffer
        algorithm = NSGAII(problem)
        algorithm.run(1500)
        cheap_pareto_front = [
            list(solution.objectives) for solution in algorithm.result
        ]
        maxoffunctions = [-1 * min(f) for f in list(zip(*cheap_pareto_front))]
        max_samples.append(maxoffunctions)
    max_samples = list(zip(*max_samples))

    x_best_fidelities = 0

    def mesmo_acq(x):
        global x_best_fidelities, beta

        for i in range(len(functions)):
            for m in range(M[i]):
                if m == 0:
                    x_ = np.c_[m, np.array([x])]
                else:
                    x_ = np.r_[x_, np.c_[m, np.array([x])]]
            means, stds = GPs[i].predict(x_, return_std=True)
            acq_funcs[i] = MFMES[i].calc_acq(x_, np.asarray(max_samples[i]))

            ################# z condition
            q = 1 / (1 + d + 2)
            for m in range(M[i] - 1):
                xi = (
                    np.linalg.norm((M[i] - 1 - m) / (M[i] - 1))
                    / GPs[i].kernel.kernel_z.get_params()["k2__length_scale"]
                )
                gamma = xi * np.power(cost[i][m] / cost[i][M[i] - 1], q)
                c = 1
                if (stds[m] < c * gamma) or (
                    xi
                    < beta[i]
                    * (1 / GPs[i].kernel.kernel_z.get_params()["k2__length_scale"])
                ):
                    acq_funcs[i][m] = None

        size = 1
        result = np.zeros((size, len(functions) + 1))
        for k in range(size):
            temp = []
            for i in range(len(functions)):
                temp.append([acq_funcs[i][k + m * size] for m in range(M[i])])
            indecies = list(zip(*[range(len(x)) for x in temp]))
            #########option 1 for cost calculation
            values_costs = [
                sum(
                    [
                        float(cost[i][m]) / cost[i][M[i] - 1]
                        for i, m in zip(range(len(functions)), index)
                    ]
                )
                for index in indecies
            ]
            values = [
                float(sum(AF)) / i for AF, i in zip(list(zip(*temp)), values_costs)
            ]
            result[k][0] = max(np.asarray(values)[np.invert(np.isnan(values))])
            max_index = values.index(result[k][0])
            for i in range(len(functions)):
                result[k][i + 1] = indecies[max_index][i]
        x_best_index = np.argmax(list(zip(*result))[0])
        x_best_fidelities = result[x_best_index][1:]
        return -1 * result[x_best_index][0]

    # l-bfgs-b
    x_tries = np.random.uniform(bound[0], bound[1], size=(1000, d))
    y_tries = [mesmo_acq(x) for x in x_tries]
    sorted_indecies = np.argsort(y_tries)
    i = 0
    x_best = x_tries[sorted_indecies[i]]
    mesmo_acq(x_best)

    x_full = np.concatenate((x_best_fidelities, x_best))
    while any((x_full == x1).all() for x1 in xValues):
        i = i + 1
        x_best = x_tries[sorted_indecies[i]]
        mesmo_acq(x_best)
        x_full = np.concatenate((x_best_fidelities, x_best))
    y_best = y_tries[sorted_indecies[i]]
    x_seed = list(np.random.uniform(low=bound[0], high=bound[1], size=(10, d)))
    for x_try in x_seed:
        result = scipyminimize(
            mesmo_acq,
            x0=np.asarray(x_try).reshape(1, -1),
            method="L-BFGS-B",
            bounds=Fun_bounds,
        )
        if not result.success:
            continue
        mesmo_acq(result.x)
        x_full = np.concatenate((x_best_fidelities, result.x))
        if (result.fun <= y_best) and (
            not (any((x_full == x1).all() for x1 in xValues))
        ):
            x_best = result.x
            y_best = result.fun
    mesmo_acq(x_best)
    x_full = np.concatenate((x_best_fidelities, x_best))
    xValues.append(x_full)

    for i in range(len(functions)):
        new_x = np.concatenate((np.array([x_best_fidelities[i]]), x_best))
        candidate_x[i] = np.r_[candidate_x[i], np.array([new_x])]
        y[i].append(functions[i](new_x[1:], d, new_x[0]))
        print("new_input", new_x)
        total_cost += float(cost[i][int(x_best_fidelities[i])]) / cost[i][M[i] - 1]
        fidelity_iter[i][int(x_best_fidelities[i])] += 1

    cost_input_output.write(
        str(x_best_fidelities)
        + str(x_best)
        + str(np.array([y[i][-1] for i in range(len(functions))]))
        + "\n"
    )
    cost_input_output.close()

    print("total_cost: ", total_cost)
    cost_input_output = open(str(experiment_num) + approx + "_input_output.txt", "a")
    allcosts.append(total_cost)

cost_input_output.close()
