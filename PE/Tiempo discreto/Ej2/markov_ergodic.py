#
#  Project: Demos for the Stochastic Systems class
#  File:    proc.py
#  Vers.    1.0
#  Date:    11/11/2020
#
#
#  Simple program to check the probability transition of a Markov
#  model. It uses the transition matrix and the initial distribution
#  (always a "delta", that is, we always start from a specific state)
#  to compute the probability of occupancy at time t, and at the same
#  time we use a set of chains to simulate the transition and estimate
#  the probability distribution to chech whether the two converge.
#

import intrf
import sys
from random import uniform

#
# Transition probabilities of the Markov model
#
p = 0.3  # basic parameters for the transition
q = 0.0  # probability: they are based on these values

#         0      1       2       3       4       5
PT = [[  1-p,    p,    0.0,    0.0,    0.0,    0.0], # 0
      [  0.0,  1-p,  p/2.0,    0.0,   p/2.0,   0.0], # 1
      [1/4.0,  0.0,  1/4.0,   1/4.0,  1/4.0,   0.0], # 2
      [    q,  0.0,    0.0,   0.9-q,    0.1,   0.0], # 3
      [  0.0,  0.0,    0.0,     0.0,  1/2.0, 1/2.0], # 4
      [  0.0,  0.0,    0.0,    1/4.0, 1/2.0, 1/4.0]  #5
      ]


#
# This class implements a Markov chain. Upon creation, a transition
# matrix and an initial state are given. The chain may be executed and
# it keeps track of the probability that at time t it is in state m.
#
class Markov:

    #
    # Initialization function
    #
    def __init__(self, PT, m_ini):
        self.PT = PT
        self.nstate = len(PT[0])
        self.state = m_ini
        self.steps = 1      # Number of steps executed so far (for the
                            # computation of averages)
        self.count = self.nstate*[0]  # count of the times a state was visited
        self.count[self.state] = 1

    #
    # Given a state and the transition probability matrix PT, determine a
    # new state with the probabilities prescribed by the matrix. Returns
    # the value of the new state.
    #
    def _transition(self, m):
        p = self.PT[m]
        u = uniform(0.0, 1.0)
        a = 0
        for k in range(len(p)):
            a = a + p[k]
            if u < a:
                return k
        return len(p)-1

    #
    # Makes a step of the Markov chain: generates a new state, updates
    # the current state, the step counter and the state counter.
    #
    # Returns the new state of the chain
    #
    def step(self):
        mnew = self._transition(self.state)
        self.state = mnew
        self.steps = self.steps+1
        self.count[self.state] = self.count[self.state] + 1
        return mnew

    #
    # Returns the probability of occupations of each one of the states
    # of the chain
    #
    def stats(self):
        s = sum(self.count)
        return [float(x)/float(s) for x in self.count]


#
# Runs a set of NC chains (with transition PT and initial state m_ini)
# for a total of T steps and at the end collects a collection
# statistics of the states in which the chains are.
#
def set_run(NC, PT, m_ini, T):
    chains = [Markov(PT, m_ini) for _ in range(NC)]
    state_no = len(PT[0])
    state = NC*[0]
    for t in range(T):
        for k in range(NC):
            state[k] = chains[k].step()

    count = state_no*[0.0]
    for k in range(NC):
        count[state[k]] = count[state[k]] + 1
    return [float(c)/float(NC) for c in count]


#
# Multiplies on the left a vector by a matrix. Returns the result
#
def mult(v, P):
    n = len(v)
    r = n*[0.0]
    for j in range(n):
        r[j] = 0
        for i in range(n):
            r[j] = r[j] + v[i]*P[i][j]
    return r

#
# Componentwise square error between two arrays
#
def err_sq(a, b):
    return [(x-y)*(x-y) for (x, y) in zip(a,b)]


#
# Plot a point in the plit schema (coordinate y is in [0,1]) with a
# given color
#
ymax = 0.05    # max value displayed
def plot(t, y, col):
    r = int(Pr1+Pnr-Pnr*y/ymax)
    r = max(Pr1,r)
    intrf.set_fore(col)
    intrf.print_at(r, Pc1+t, "x")
    intrf.set_fore(0)


#
#  Script begins here
#

n_s = len(PT[0])
ensemble_size = 1000
initial_state = 0

#
# Initial probability distribution
#
ini_prob = n_s*[0.0]
ini_prob[initial_state] = 1.0


ensemble_prob = ini_prob[:]
theor_prob = ini_prob[:]
ergod_prob = ini_prob[:]
cur_time = 0
delta_t = 10

#
# Errors
#
err_1 = n_s*[0.0]
err_2 = n_s*[0.0]

#
# Data for the plot
#
Pr1 = 1    # first row
Pc1 = 40   # first column
Pnr = 40  # number of rows
Pnc = 100  # number of columns


#
# Draw the left interface (the one with the numbers)
#
intrf.set_fore(0)
intrf.set_back(9)
intrf.clear()

intrf.print_at(1, 1, "T: ")
intrf.print_at(1, 4, "%5d" % cur_time)

intrf.print_at(1, 21, "Dt: ")
intrf.print_at(1, 25, "%5d" % delta_t)

intrf.set_fore(1)
intrf.print_at(3, 1, "Ensemble")
intrf.print_at(3, 11, "Theory")
intrf.print_at(3, 21, "Ergodic")
intrf.set_fore(0)

for k in range(n_s):
    intrf.print_at(4+k, 1, "%6.4f" % ensemble_prob[k])
    intrf.print_at(4+k, 11, "%6.4f" % theor_prob[k])
    intrf.print_at(4+k, 21, "%6.4f" % ergod_prob[k])

intrf.set_fore(2)
intrf.print_at(4+n_s, 4, "|")
intrf.print_at(4+n_s+1, 4, "+->")
intrf.print_at(4+n_s, 3+11, "|")
intrf.print_at(4+n_s+1, 3+9, "<-+->")
intrf.print_at(4+n_s, 3+21, "|")
intrf.print_at(4+n_s+1, 3+19, "<-+")
intrf.set_fore(0)

for k in range(n_s):
    intrf.set_fore(1)
    intrf.print_at(4+n_s+1+k, 7, "%5.3f" % err_1[k])
    intrf.set_fore(4)
    intrf.print_at(4+n_s+1+k, 17, "%5.3f" % err_2[k])
intrf.set_fore(0)

intrf.print_at(4+2*n_s+1, 7, "-----")
intrf.print_at(4+2*n_s+1, 17, "-----")

intrf.set_fore(4)
intrf.print_at(4+2*n_s+2, 7, "%5.3f/100" % 0.0)
intrf.print_at(4+2*n_s+2, 17,"%5.3f" % 0.0)
intrf.set_fore(0)


#
# Draw the schema for the plot
#
for r in range(Pr1, Pr1+Pnr):
    intrf.print_at(r, Pc1, "|")
for c in range(Pc1, Pc1+Pnc):
    intrf.print_at(Pr1+Pnr, c, "-")
intrf.print_at(Pr1,Pc1-6, "%5.3f" % ymax)


me = Markov(PT, initial_state)   # model for the ergodic probability


pc = 0
while True:
    sys.stdin.readline()
    last_time = cur_time
    cur_time = cur_time + delta_t
    pc = pc + 1
    intrf.print_at(1, 4, "%5d" % cur_time)

    #
    # Theoretical
    #
    for t in range(delta_t):
        theor_prob = mult(theor_prob, PT)

    #
    # Ensemble
    #
    ensemble_prob = set_run(ensemble_size, PT, initial_state, cur_time)

    #
    # Ergodic
    #
    for t in range(delta_t):
        me.step()
    ergod_prob = me.stats()

    for k in range(n_s):
        intrf.print_at(4+k, 1, "%6.4f" % ensemble_prob[k])
        intrf.print_at(4+k, 11, "%6.4f" % theor_prob[k])
        intrf.print_at(4+k, 21, "%6.4f" % ergod_prob[k])

    err_1 = err_sq(ensemble_prob, theor_prob)
    err_2 = err_sq(ergod_prob, theor_prob)


    for k in range(n_s):
        intrf.set_fore(1)
        intrf.print_at(4+n_s+1+k, 7, "%5.3f" % (100*err_1[k]))
        intrf.set_fore(4)
        intrf.print_at(4+n_s+1+k, 17, "%5.3f" % err_2[k])
    intrf.set_fore(0)

    intrf.set_fore(4)
    intrf.print_at(4+2*n_s+2, 7, "%5.3f/100" % (100*sum(err_1)))
    intrf.print_at(4+2*n_s+2, 17,"%5.3f" % (sum(err_2)))
    intrf.set_fore(0)

    plot(pc, sum(err_1), 1)
    plot(pc, sum(err_2), 4)

#
#m = Markov(PT, 0)
#
#for t in range(100):
#    m.step()
#
#print m.stats()
#
#print set_run(200, PT, 0, 100)
#
#l = ini_prob[:]
#for t in range(100):
#    l = mult(l, PT)
#
#print l
