import numpy as np
from scipy.stats import chi2
import sys


#  generating one simulation based on uniform distribution
def generate(theta, size_of_data):
    cat_1 = (1 - theta) ** 2  # probability of -1
    cat_2 = 2 * theta * (1 - theta)  # probability of 0
    data = np.random.uniform(0, 1, size_of_data)
    data[data < cat_1] = -1
    data[(cat_1 < data) & (data <= cat_1 + cat_2)] = 0
    data[(cat_1 + cat_2 < data) & (data <= 1)] = 1
    return data


#  deriving the MLE estimates of the simulation data: (2*n1+n0)/2*n
def max_l_estimator(simulation_matrix):
    ones = np.where(simulation_matrix == 1)
    zeros = np.where(simulation_matrix == 0)
    for_estimates = np.zeros(simulation_matrix.shape)
    for_estimates[zeros] = 1
    for_estimates[ones] = 2
    estimates = np.mean(for_estimates, axis=0) / 2
    return estimates


#  fisher information
def fisher_inf(theta):
    return 2 / (theta * (1 - theta) + sys.float_info.min)  # sys.float_info.min added to avoid division by zero


#  test with the null
def test1(estimates, null, data_size, significance=0.95):
    t1 = data_size * (estimates - null) ** 2 * fisher_inf(null)
    return np.mean(t1 > chi2.ppf(q=significance, df=1))


#  test with the MLE estimates
def test2(estimates, null, data_size, significance=0.95):
    t2 = data_size * (estimates - null) ** 2 * np.vectorize(fisher_inf)(estimates)
    return np.mean(t2 > chi2.ppf(q=significance, df=1))


data_size = 100
simulation_number = 10000
theta0 = 0.2
theta_alt1 = 0.1
theta_alt2 = 0.3

#  simulation data, where H0: theta = 0.2 is true
simulation_matrix_0 = np.zeros((data_size, simulation_number))
for simulation in range(simulation_number):
    simulation_matrix_0[:, simulation] = generate(theta0, data_size)

#  simulation data, where H1: theta = 0.1 is true
simulation_matrix_alt1 = np.zeros((data_size, simulation_number))
for simulation in range(simulation_number):
    simulation_matrix_alt1[:, simulation] = generate(theta_alt1, data_size)

#  simulation data, where H1: theta = 0.3 is true
simulation_matrix_alt2 = np.zeros((data_size, simulation_number))
for simulation in range(simulation_number):
    simulation_matrix_alt2[:, simulation] = generate(theta_alt2, data_size)

est_0 = max_l_estimator(simulation_matrix_0)
est_alt1 = max_l_estimator(simulation_matrix_alt1)
est_alt2 = max_l_estimator(simulation_matrix_alt2)

#  type 1 error rates
print(test1(est_0, theta0, data_size))
print(test2(est_0, theta0, data_size))

#  type 2 error rates for H1: theta = 0.1
print(1 - test1(est_alt1, theta0, data_size))
print(1 - test2(est_alt1, theta0, data_size))

#  type 2 error rates for H2: theta = 0.3
print(1 - test1(est_alt2, theta0, data_size))
print(1 - test2(est_alt2, theta0, data_size))
