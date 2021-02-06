import numpy as np
from scipy.stats import norm, uniform
from time import time

def experiment():
    """ Simulate the experiment of generating a random point on
        the square and evaluating whether it falls inside the circle. """

    # Generate a uniform sample on [0,1]x[0,1]
    x = uniform.rvs()
    y = uniform.rvs()

    # Test whether it falls inside the circle
    return 1 if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.25 else 0

def mc(alpha, eps, max_iter = np.Inf):
    """ Approximate π by a Monte-Carlo method, up to a tolerance of 'eps'
        with a confidence of 100(1-'alpha')%. """

    n = 1  # Number of points
    s = 0  # Sample standard deviation
    sn = experiment()  # Sample mean
    z_alpha = norm.ppf(1 - alpha / 2)

    while (n <= max_iter):
        # Perform a new instance of the experiment
        xi = experiment()
        n += 1

        # Update s and sn reutilizing the previous ones
        s = s + (n - 1) * ((xi - sn) ** 2) / n
        sn = sn + (xi - sn) / n

        # Check stopping condition
        if s != 0:
            eps_n = 2 * z_alpha * (s / (n - 1)) / np.sqrt(n)
            if eps_n < eps:
                break

    return 4 * sn, n

np.random.seed(42)

# Perform simulation and show results
start = time()
print("Running simulation...")
pi, n = mc(alpha = 0.01, eps = 5e-4)
elapsed = time() - start

print("\n-- Approximation of π with 3 significant digits --")
print(f"π ~= {pi:.5f}")
print(f"Number of points used: {n}")
print(f"Elapsed time: {elapsed:.3f}s")
print(f"Absolute error: {np.abs(pi - np.pi):.5f}")

R = 10
print(f"\nRunning {R} independent simulations...")
elapsed = []
ns = []
ts = []
errs = []
success = 0
for i in range(0, R):
    start = time()
    pi, n = mc(alpha = 0.01, eps = 5e-4)
    ts.append(time() - start)
    ns.append(n)
    errs.append(np.abs(pi - np.pi))
    pi_digits = str(pi)
    real_pi_digits = str(np.pi)
    if pi_digits[:5] == real_pi_digits[:5]:
        success += 1

print(f"Number of runs that succeeded: {success}/{R}")
print(f"Number of points used on average: {int(np.mean(ns))}")
print(f"Mean elapsed time: {np.mean(ts):.3f}s")
print(f"Mean absolute error: {np.mean(errs):.5f}")
