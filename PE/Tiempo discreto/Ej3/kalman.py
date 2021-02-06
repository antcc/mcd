import numpy as np
from scipy.linalg import solve
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(42)

def kf_predict_step(A, B, Q, u, xpred, P):
    """ Perform one step of the prediction phase. """

    xbar = A @ xpred + B @ u
    Pbar = A @ P @ A.T + Q

    return xbar, Pbar

def kf_update_step(C, Pbar, R, x, v, xbar):
    """ Perform one step of the update phase. """

    n = C.shape[1]
    K = solve(C @ Pbar @ C.T + R, C @ Pbar).T
    z = C @ x + v.rvs()
    xpred = xbar + K @ (z - C @ xbar)
    P = (np.diag(np.ones(n)) - K @ C) @ Pbar

    return K, xpred, P

def kf(A, B, C, Q, R, xs, us, xbar_0, Pbar_0, max_t):
    """ Perform 'max_t' iterations of the Kalman filter.
          - A, B: hidden process dynamics matrices.
          - C: observable process dynamics matrix.
          - xs: vector of actual hidden process values to simulate z_t.
          - us: vector of user-defined inputs.
          - xbar_0: prior of first value.
          - Pbar_0: prior of first covariance matrix.
          - Q: white noise covariance matrix for hidden process.
          - R: white noise covariance matrix for observable process. """

    n = A.shape[0]
    v = multivariate_normal(None, R, allow_singular = True)

    # A priori estimations
    Pbar = Pbar_0
    xbar = xbar_0

    # Array to store predictions
    xpreds = np.zeros((max_t, n))

    # Loop through iterations
    for t in range(max_t):
        K, xpred, P = kf_update_step(C, Pbar, R, xs[t], v, xbar)
        xbar, Pbar = kf_predict_step(A, B, Q, us[t], xpred, P)
        xpreds[t] = xpred

    return xpreds

def dynamical_system(A, B, us, Q, x0, max_t):
    """ Compute 'max_t' steps of the process defined by
            x_0 = x0,
            x_{t+1} = Ax_t + Bu_t + w_t,
        where w_t is a white Gaussian noise with covariance matrix Q. """

    n = A.shape[0]
    xs = np.full((max_t + 1, n), x0)
    w = multivariate_normal(None, Q, allow_singular = True)

    for t in range(1, max_t + 1):
        xs[t] = A @ xs[t - 1] + B @ us[t - 1] + w.rvs()

    return xs[1:]

# Number of iterations
max_t = 100
ts = np.arange(max_t)

# Dynamics matrices
A = np.array([[0.15, 0.15, 0.70, -0.20],
             [0.15, 0.15, -0.20, 0.70],
             [0.70, -0.20, 0.15, 0.15],
             [-0.20, 0.70, 0.15, -0.15]])
B = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
C = np.array([[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]])
n = A.shape[0]
q = C.shape[0]

# Input signals are always 1 (irrelevant)
us = np.ones(max_t).reshape(-1, 1)

# White noise covariance matrices for hidden and observable states
sigma_w = 10.0
sigma_v = 1.0
Q = np.diag(np.full(n, sigma_w ** 2))
R = np.diag(np.full(q, sigma_v ** 2))

# Initial estimates
xbar_0 = np.zeros(n)
Pbar_0 = np.diag(np.full(n, sigma_w ** 2))

# Simulate system to get a sample
xs = dynamical_system(A, B, us, Q, xbar_0, max_t)

# Perform simulations of Kalman filters
runs = int(sigma_v * 10)
xpreds_runs = np.zeros((runs, max_t, n))
for i in range(runs):
    xpreds_runs[i] = kf(A, B, C, Q, R, xs, us, xbar_0, Pbar_0, max_t)

# Compute mean values
xpreds = np.mean(xpreds_runs, axis = 0)
errors = [np.linalg.norm(xs[t] - xpreds_runs[:, t], axis = 1) ** 2 for t in ts]
mean_error = np.mean(errors, axis = 1)
std_error = np.std(errors, axis = 1) / np.sqrt(runs)

# Visualize predictions vs real values
fig, axs = plt.subplots(2, 2)
plt.suptitle(r"$\sigma_v =$" + f"{sigma_v}", y = 0.95)
for i in range(2):
    for j in range(2):
        axs[i, j].plot(xs[:, 2 * i + j], label = r"Real values $x_t$")
        axs[i, j].plot(xpreds[:, 2 * i + j],
                       label = r"Mean predicted values $\hat x_t$")
        axs[i, j].legend()
        axs[i, j].set_title(f"Component {2 * i + j + 1}")
        axs[i, j].set_xlabel("Time")
plt.show()

# Visualize mean MSE +- std
plt.plot(ts, mean_error, label = f"Mean of MSE estimations for {runs} runs")
plt.fill_between(ts, np.maximum(mean_error - std_error, 0),
                 mean_error + std_error,
                 color = 'b', alpha = 0.1,
                 label = r"$\pm$ Standard deviation (clipped)")
plt.xlabel("Time")
plt.title(r"$\sigma_v =$" + f"{sigma_v}")
plt.legend()
plt.show()
