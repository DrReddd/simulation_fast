import numpy as np
from scipy.stats import chi2
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm


#  generating one simulation based on uniform distribution
def generate(theta, size_of_data):
    cat_1 = (1 - theta) ** 2  # probability of -1
    cat_2 = 2 * theta * (1 - theta)  # probability of 0
    data = np.random.uniform(0, 1, size_of_data)
    data[data <= cat_1] = -1
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
    if theta * (1 - theta) == 0:
        return 2 / (theta * (1 - theta) + sys.float_info.min * 3)  # sys.float_info.min added to avoid division by zero
    else:
        return 2 / (theta * (1 - theta))


#  test with the null
def test1(estimates, null, data_size, significance):
    significance = 1 - significance
    t1 = data_size * (estimates - null) ** 2 * fisher_inf(null)
    return np.mean(t1 > chi2.ppf(q=significance, df=1))


#  test with the MLE estimates
def test2(estimates, null, data_size, significance):
    significance = 1 - significance
    t2 = data_size * (estimates - null) ** 2 * np.vectorize(fisher_inf)(estimates)
    return np.mean(t2 > chi2.ppf(q=significance, df=1))


data_size = 100
simulation_number = 10000
theta0 = 0.2
theta_alt1 = 0.1
theta_alt2 = 0.3
significance_level = 0.1

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

print(f"\n Error rates with sample size of {data_size}, and number of simulations {simulation_number}: \n")
#  type 1 error rates
print(f" Type 1 error under first test (H0: theta = {theta0}): {test1(est_0, theta0, data_size, significance_level)}")
print(f" Type 1 error under second test (H0: theta = {theta0}): {test2(est_0, theta0, data_size, significance_level)} \n")

#  type 2 error rates for H1: theta = 0.1
print(f" Type 2 error under first test (H1: theta = {theta_alt1}): {1 - test1(est_alt1, theta0, data_size, significance_level)}")
print(f" Type 2 error under second test (H1: theta = {theta_alt1}): {1 - test2(est_alt1, theta0, data_size, significance_level)} \n")

#  type 2 error rates for H2: theta = 0.3
print(f" Type 2 error under first test (H1: theta = {theta_alt2}): {1 - test1(est_alt2, theta0, data_size, significance_level)}")
print(f" Type 2 error under second test (H1: theta = {theta_alt2}): {1 - test2(est_alt2, theta0, data_size, significance_level)} \n")

#  END OF SIMULATION FROM THE RECITATION

#  plot type 2 errors as a function of sample size
sample_size_min = 5
sample_size_max = 200
data_sizes = np.arange(sample_size_min, sample_size_max, 1)
simulation_number = 2000

#  type 1 errors for the two test
t11 = []
t12 = []
#  type 2 errors for the two test under H1
t211 = []
t212 = []
#  type 2 errors for the two test under alternative H1
t221 = []
t222 = []

print(f"Generating data and testing for different sample sizes ({sample_size_min} to {sample_size_max}), with "
      f"{simulation_number} simulations for a given sample size:")

with tqdm(total=sample_size_max - sample_size_min) as pbar:
    for count, data_size in enumerate(data_sizes):
        pbar.update(1)

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
        t11.append(test1(est_0, theta0, data_size, significance_level))
        t12.append(test2(est_0, theta0, data_size, significance_level))
        t211.append(1 - test1(est_alt1, theta0, data_size, significance_level))
        t212.append(1 - test2(est_alt1, theta0, data_size, significance_level))
        t221.append(1 - test1(est_alt2, theta0, data_size, significance_level))
        t222.append(1 - test2(est_alt2, theta0, data_size, significance_level))

fig, ax = plt.subplots()
pl_t11, = plt.plot(data_sizes, t11, ".-", alpha=0.3, label=f"H0: {theta0}, Test1")
pl_t12, = plt.plot(data_sizes, t12, ".-", alpha=0.3, label=f"H0: {theta0}, Test2")
pl_asymp, = plt.plot(data_sizes, [significance_level for i in data_sizes], "--", alpha=0.5, color="black", label="Asymptotic")
legend1 = plt.legend(handles=[pl_t11, pl_t12, pl_asymp], loc=7)
plt.gca().add_artist(legend1)
plt.text(0.05, significance_level+0.05, "Type 1 error rates", alpha=0.8, transform=ax.transAxes)
pl_t211, = plt.plot(data_sizes, t211, ".-", label=f"H1: {theta_alt1}, Test1")
pl_t212, = plt.plot(data_sizes, t212, ".-", label=f"H1: {theta_alt1}, Test2")
pl_t221, = plt.plot(data_sizes, t221, ".-", label=f"H1: {theta_alt2}, Test1")
pl_t222, = plt.plot(data_sizes, t222, ".-", label=f"H1: {theta_alt2}, Test2")
plt.legend(handles=[pl_t211, pl_t212, pl_t221, pl_t222], loc="upper right")
plt.xlabel("Sample size")
plt.ylabel("Type 2 error rate")
plt.title("Type II errors as a function of sample size")
plt.show()
