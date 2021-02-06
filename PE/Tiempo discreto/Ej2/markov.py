import numpy as np
from scipy.stats import uniform
from scipy.optimize import nnls
import matplotlib.pyplot as plt

#
# MARKOV CHAIN DEFINITION
#

class MarkovChain:
    """
    This class represents a Markov chain via an initial state and a transition
    matrix. It keeps track of the probability of being in a certain state at
    any given time.
    """

    def __init__(self, P, m_ini):
        """ The state of the chain is represented by:
              - The total number of states, 'n'.
              - A transition matrix, 'P', of dimensions n x n.
              - The current state, 'm', indexed in [0, n-1].
              - The initial state, 'm_ini'.
              - The number of time instants that have passed, 'n_steps'.
              - An array that stores how many times each state has been visited,
                'state_count'. """

        self.P = P
        self.n = len(P[0])
        self.m_ini = m_ini
        self.m = m_ini
        self.n_steps = 1
        self.states_count = self.n * [0]
        self.states_count[self.m_ini] = 1  # We start at m_ini

    def theoretical_distribution(self, t):
        """ Return the theoretical distribution of the chain after t instants,
            starting from m_ini. """

        init_dist = [0] * self.n
        init_dist[self.m_ini] = 1

        return init_dist @ np.linalg.matrix_power(self.P, t)

    def step(self):
        """ Simulate a single step in the chain and return the new state. """

        # Throw an uniform number u in [0,1] and move to the first state
        # such that cumsum(P[m]) >= u
        u = uniform.rvs()
        self.m = np.where(np.cumsum(self.P[self.m]) >= u)[0][0]
        self.n_steps += 1
        self.states_count[self.m] += 1

        return self.m

    def steps(self, t):
        """ Perform T steps in the chain and return the new state. """

        for _ in range(t):
            self.step()

        return self.m

#
# MARKOV CHAIN SIMULATION
#

def theoretical_probs(P, m_ini, t):
    """ Compute the theoretical probability of being in each state
        after t iterations in the chain (P, m_ini). """

    mc = MarkovChain(P, m_ini)

    return mc.theoretical_distribution(t)

def empirical_probs(P, m_ini, t, n):
    """ Estimate the probability of being in each state after t
        iterations by performing n independent simulations of t
        steps on the chain (P, m_ini). """

    n_states = len(P[0])
    states_visited = np.zeros(n_states)
    for i in range(n):
        MC = MarkovChain(P, m_ini)
        states_visited[MC.steps(t)] += 1

    return states_visited / np.sum(states_visited)

def theoretical_h(P, m):
    """ Compute the theoretical hitting probability of m on the chain (P). """

    n = len(P[0])
    A = -P
    A += np.diag(np.ones(n))

    b = np.zeros(n)
    b[m] = 1
    A[m] = b

    return nnls(A, b)[0]

def empirical_h(P, m_ini, m, t, n):
    """ Compute the empirical hitting probability of m with n independent
        executions of t iterations of the chain (P, m_ini). """

    if m_ini == m:
        return 1.0

    hits = 0
    for i in range(n):
        MC = MarkovChain(P, m_ini)
        MC.steps(t)
        if MC.states_count[m] > 0:
            hits += 1

    return hits / n

def theoretical_k(P, m):
    """ Compute the theoretical expected hitting time of m on the chain (P),
       given that they are all finite. """

    n = len(P[0])
    A = -P
    A += np.diag(np.ones(n))

    b = np.ones(n)
    b[m] = 0
    A[m] = np.logical_not(b)

    return nnls(A, b)[0]

def empirical_k(P, m_ini, m, t, n):
    """ Compute the empirical expected hitting time of m with n independent
        executions of t iterations of the chain (P, m_ini). """

    if m_ini == m:
        return 0.0

    hit_time = [0.0] * n
    for i in range(n):
        MC = MarkovChain(P, m_ini)
        for tt in range(t):
            if MC.step() == m:
                hit_time[i] = tt + 1
                break

    return np.Infinity if 0.0 in hit_time else np.mean(hit_time)

def plot_first_hitting_time(P, m_ini, m, tmax=100):
    """ Plot the (theoretical) graph of P[H_{m_ini}^{m} = t]. """

    n = len(P[0])
    idx = list(np.arange(m)) + list(np.arange(m + 1, n))
    g = np.zeros(tmax + 1)
    f = P[idx, m]
    g[0] = 0
    g[1] = f[m_ini]  # first step

    # Compute recurrence relation
    for t in range(2, tmax + 1):
        f = [P[i, idx] @ f for i in idx]
        g[t] = f[m_ini]

    # Plot graph
    plt.plot(g)
    plt.ylabel("g(t)")
    plt.xlabel("Tiempo (discreto)")
    plt.show()

#
# ANALYSIS OF A SPECIFIC MARKOV CHAIN
#

# Seed for reproducibility
np.random.seed(42)

# Transition matrix
p = 0.3
q = 0.0
P1 = np.array([
    [1 - p, p, 0.0, 0.0, 0.0, 0.0],                  # 0
    [0.0, 1 - p, p / 2.0, 0.0, p / 2.0, 0.0],        # 1
    [1 / 4.0, 0.0, 1 / 4.0, 1 / 4.0, 1 / 4.0, 0.0],  # 2
    [q, 0.0, 0.0, 0.9 - q, 0.1, 0.0],                # 3
    [0.0, 0.0, 0.0, 0.0, 1 / 2.0, 1 / 2.0],          # 4
    [0.0, 0.0, 0.0, 1 / 4.0, 1 / 2.0, 1 / 4.0]])     # 5

q = 0.1
P2 = np.array([
    [1 - p, p, 0.0, 0.0, 0.0, 0.0],                  # 0
    [0.0, 1 - p, p / 2.0, 0.0, p / 2.0, 0.0],        # 1
    [1 / 4.0, 0.0, 1 / 4.0, 1 / 4.0, 1 / 4.0, 0.0],  # 2
    [q, 0.0, 0.0, 0.9 - q, 0.1, 0.0],                # 3
    [0.0, 0.0, 0.0, 0.0, 1 / 2.0, 1 / 2.0],          # 4
    [0.0, 0.0, 0.0, 1 / 4.0, 1 / 2.0, 1 / 4.0]])     # 5

# Number of independent simulations and steps in each one
n = 1000
t = 1000

print("THEORETICAL PROBABILITIES\n")
print("--- Probability of each state after t steps ---")
th1_1 = theoretical_probs(P1, 0, 1)
th2_1 = theoretical_probs(P1, 0, 2)
print("With q = 0.0:")
print("  After 1 step=", th1_1)
print("  After 2 steps=", th2_1)
th1_2 = theoretical_probs(P2, 0, 1)
th2_2 = theoretical_probs(P2, 0, 2)
print("With q = 0.1:")
print("  After 1 step=", th1_2)
print("  After 2 steps=", th2_2)

print("--- Hitting probability ---")
th_h02_1 = theoretical_h(P1, 2)[0]
th_h05_1 = theoretical_h(P1, 5)[0]
th_h42_1 = theoretical_h(P1, 2)[4]
print("With q = 0.0:")
print("  h02 =", th_h02_1)
print("  h05 =", th_h05_1)
print("  h42 =", th_h42_1)
th_h02_2 = theoretical_h(P2, 2)[0]
th_h05_2 = theoretical_h(P2, 5)[0]
th_h42_2 = theoretical_h(P2, 2)[4]
print("With q = 0.1:")
print("  h02 =", th_h02_2)
print("  h05 =", th_h05_2)
print("  h42 =", th_h42_2)

print("--- Expected hitting time ---")
print("With q = 0.0:")
print("  k02 = inf")
print("  k42 = inf")
th_k02_2 = theoretical_k(P2, 2)[0]
th_k42_2 = theoretical_k(P2, 2)[4]
print("With q = 0.1:")
print("  k02 =", th_k02_2)
print("  k42 =", th_k42_2)

print("\nEMPIRICAL PROBABILITIES")
print(f"Simulating {n} independent chains up to {t} steps...\n")

print("--- Probability of each state after t steps ---")
emp1_1 = empirical_probs(P1, 0, 1, n)
emp2_1 = empirical_probs(P1, 0, 2, n)
print("With q = 0.0:")
print("  After 1 step=", emp1_1)
print("  After 2 steps=", emp2_1)
emp1_2 = empirical_probs(P2, 0, 1, n)
emp2_2 = empirical_probs(P2, 0, 2, n)
print("With q = 0.1:")
print("  After 1 step=", emp1_2)
print("  After 2 steps=", emp2_2)

print("--- Hitting probability ---")
emp_h02_1 = empirical_h(P1, 0, 2, t, n)
emp_h05_1 = empirical_h(P1, 0, 5, t, n)
print("With q = 0.0:")
print("  h02 =", emp_h02_1)
print("  h05 =", emp_h05_1)
emp_h02_2 = empirical_h(P2, 0, 2, t, n)
emp_h05_2 = empirical_h(P2, 0, 5, t, n)
print("With q = 0.1:")
print("  h02 =", emp_h02_2)
print("  h05 =", emp_h05_2)

print("--- Expected hitting time ---")
emp_k02_1 = empirical_k(P1, 0, 2, t, n)
emp_k42_1 = empirical_k(P1, 4, 2, t, n)
print("With q = 0.0:")
print("  k02 =", emp_k02_1)
print("  k42 =", emp_k42_1)
emp_k02_2 = empirical_k(P2, 0, 2, t, n)
emp_k42_2 = empirical_k(P2, 4, 2, t, n)
print("With q = 0.1:")
print("  k02 =", emp_k02_2)
print("  k42 =", emp_k42_2)

print("\nGRAPH OF HITTING TIME")
print("Showing theoretical graph of P[H_0^4 = t] for q = 0.1...")
plot_first_hitting_time(P2, 0, 4)
