import numpy as np

# Transition matrix
P = np.array([[0.5, 0.5, 0, 0, 0, 0],
              [0, 0.5, 0.5, 0, 0, 0],
              [0.25, 0, 0.25, 0.25, 0.25, 0],
              [0.25, 0, 0, 0.5, 0.25, 0],
              [0, 0, 0, 0, 0.5, 0.5],
              [0, 0, 0, 0.25, 0.5, 0.25]])

# Initial distributions
l1 = np.array([1, 0, 0, 0, 0, 0])
l2 = np.array([0, 0, 1, 0, 0, 0])
l3 = np.array([0, 0, 0, 0, 0, 1])

# Desired future distributions
t1 = 2
t2 = 10
t3 = 100

# Print distributions p(t) up to 6 significant digits
np.set_printoptions(precision = 6, suppress = True)
for l in [l1, l2, l3]:
    print(f"Initial distribution: {l}")
    for t in [t1, t2, t3]:
        pt = l @ np.linalg.matrix_power(P, t)
        print(f"  p({t}) = {pt}")

# Print limiting transition matrix
print("\nLimiting transition matrix:")
l, v = np.linalg.eig(P)
l[np.abs(l) < 1] = 0
d = np.diag(l)
p_inf = np.real_if_close(v @ d @ np.linalg.inv(v))
print(p_inf)
