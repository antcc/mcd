import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
from tqdm import tqdm

class GamblersRuin:
    """ This class represents the infinite Markov Chain of associated with
        the game gambler's ruin. """

    def __init__(self, p, m_ini):
        """ The infinite chain is represented by the probability of winning
            a round ('p') and the initial money ('m_ini'). """

        self.p = p
        self.m_ini = m_ini
        self.m = m_ini

    def step(self):
        """ Simulate a single step in the chain and return the new state. """

        if self.m == 0:
            return 0

        u = uniform.rvs()
        self.m = self.m + 1 if u <= self.p else self.m - 1

        return self.m

def steps_to_ruin(m_ini, t):
    """ Compute the steps performed until going bankrupt, with up to t iterations
        of the chain starting at m_ini. If the bankrupt state fails to be reached,
        the total number of iterations of the chain is returned. """

    if m_ini == 0:
        return 0.0

    MC = GamblersRuin(0.5, m_ini)
    for tt in range(1, t + 1):
        if MC.step() == 0:
            return tt
    return t

#
# SIMULATION
#

np.random.seed(1)
N = 20
M = 50
ms = np.arange(1, M + 1)
T = int(1e5)

ruin_time = np.zeros((M, N))
for m in tqdm(ms):
    ruin_time[m - 1] = [steps_to_ruin(m, T) for _ in range(N)]

mean_rt = np.mean(ruin_time, axis = 1)
sd_rt = np.std(ruin_time, axis = 1)

fig, axs = plt.subplots(1, 2)
axs[0].plot(ms, mean_rt, lw = 2)
axs[0].set_xlabel("Initial money (€)")
axs[0].set_ylabel("Expected steps until ruin")
axs[1].plot(ms, sd_rt ** 2, lw = 2)
axs[1].set_xlabel("Initial money (€)")
axs[1].set_ylabel("Variance of expected steps until ruin")
plt.show()

plt.plot(ms, mean_rt, lw = 2, label = "Expected steps until ruin")
plt.fill_between(ms, np.maximum(mean_rt - sd_rt, 0), mean_rt + sd_rt, color = 'b',
                 alpha=.1, label = "+- Standard deviation (clipped)")
plt.xlabel("Initial money (€)")
plt.legend(loc = 'upper left')
plt.show()
