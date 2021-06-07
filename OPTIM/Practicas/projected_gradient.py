import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

np.random.seed(42)
np.set_printoptions(precision=4)


def projected_gradient(X, y, w0, rho, lr=1, max_iter=100, eps=1e-4):
    w_ext = w0
    w = w_ext[1:]
    b = w_ext[0]
    n = X.shape[0]
    ones = np.ones((1, n))
    X_red = X[:, 1:]

    for k in range(max_iter):
        x_new = w + lr*(X_red.T@(y - X@w_ext))/n
        w_new_ext = np.hstack((
            b + lr*(ones@(y - X@w_ext))/n,
            rho*x_new/max(rho, np.linalg.norm(x_new))
        ))

        if np.linalg.norm(w_ext - w_new_ext) < eps:
            break

        w_ext = w_new_ext
        w = w_ext[1:]
        b = w_ext[0]

    return w_new_ext, k + 1


def mse(w, X, y):
    return 0.5*np.linalg.norm(y - X@w)**2/X.shape[0]


def constrain(w, rho):
    return rho - np.linalg.norm(w[1:])


N = 500
d = 10
b = -20
sigma = 10.0
plot = False
weights = np.random.randn(d)
data, y, coef = make_regression(
    n_samples=N, n_features=d,
    n_informative=d,
    bias=b, noise=sigma,
    coef=True, random_state=42)
X = np.hstack((
    np.ones(N).reshape(-1, 1),
    data
))
w0 = np.hstack((b, weights))
rho = 10

# Scipy's constrained optimization
reg = {'type': 'ineq', 'fun': lambda w: constrain(w, rho)}
w_optim = minimize(lambda w: mse(w, X, y), w0, constraints=reg).x

# Projected gradient
w, k = projected_gradient(X, y, w0, rho)

if d == 2 and plot:
    # Plot constrained regression line
    xmin, xmax = np.min(data), np.max(data)
    ymin, ymax = np.min(y), np.max(y)
    delta_x = (xmax - xmin)/10
    delta_y = (ymax - ymin)/10
    x = np.array([xmin - delta_x, xmax + delta_x])
    plt.scatter(data, y)
    plt.plot(x, w[0] + x*w[1], color="green")
    plt.show()

print("True solution:", coef)
print("Scipy's minimize solution:", w_optim)
print("Projected gradient solution:", w)
print("[PG] Iterations:", k)
