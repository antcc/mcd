# -*- coding: utf-8 -*-
"""
@author: Alberto Suárez
         Luis Antonio Ortega Andrés
         Antonio Coín Castro
"""

import numpy as np


def ode_euler(t0, x0, T, a, N):
    """Numerical integration of an ODE using the Euler scheme.

        x(t0) = x0
        dx(t) = a(t, x(t))*dt    [ODE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t,x(t)) that characterizes the drift term
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T].
    X: numpy.ndarray of shape (N+1,)
        Vector composed of the values of the approximated
        solution trajectory of the equation at t.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> def a(t, x): return 1.3*x
    >>> t0, x0 = 0.0, 100.0
    >>> T = 2.0
    >>> N = 1000
    >>> t, X = sde.ode_euler(t0, x0, T, a, N)
    >>> plt.plot(t, X)
    """
    dT = T/N  # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros(N + 1)

    # Initial condition
    X[0] = x0

    # Integration of the ODE
    for n in range(N):
        X[n + 1] = X[n] + a(t[n], X[n])*dT

    return t, X


def euler_maruyana(t0, x0, T, a, b, M, N):
    """Numerical integration of an SDE using the stochastic Euler scheme.

        x(t0) = x0
        dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)    [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t,x(t)) that characterizes the drift term
    b :
        Function b(t,x(t)) that characterizes the diffusion term
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T].
    X: numpy.ndarray of shape (M, N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values
        of the process at t.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> t, S = sde.euler_maruyana(t0, S0, T, a, b, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Euler scheme)')

    """
    dT = T/N  # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros((M, N + 1))

    # Initial condition
    X[:, 0] = np.full(M, x0)

    # Integration of the SDE
    for n in range(N):
        dW = np.random.randn(M)
        X[:, n + 1] = (X[:, n] + a(t[n], X[:, n])*dT
                       + b(t[n], X[:, n])*np.sqrt(dT)*dW)

    return t, X


def milstein(t0, x0, T, a, b, db_dx, M, N):
    """Numerical integration of an SDE using the stochastic Milstein scheme.

        x(t0) = x0
        dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)    [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t, x(t)) that characterizes the drift term
    b :
        Function b(t, x(t)) that characterizes the diffusion term
    db_dx:
        Derivative wrt the second argument of b(t, x)
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T].
    X: numpy.ndarray of shape (M, N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> def db_dSt(t, St): return sigma
    >>> t, S = sde.milstein(t0, S0, T, a, b, db_dSt, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Milstein scheme)')
    """
    dT = T/N  # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros((M, N + 1))

    # Initial condition
    X[:, 0] = np.full(M, x0)

    # Integration of the SDE
    for n in range(N):
        dW = np.random.randn(M)
        X[:, n + 1] = (X[:, n] + a(t[n], X[:, n])*dT
                       + b(t[n], X[:, n])*np.sqrt(dT)*dW
                       + 0.5*b(t[n], X[:, n])*db_dx(t[n], X[:, n])
                       * (dW**2 - 1)*dT)

    return t, X


def simulate_jump_process(t0, T, simulator_arrival_times, simulator_jumps, M):
    """Simulation of jump process.

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    T : float
        Length of the simulation interval [t0, t0+T]
    simulator_arrival_times: callable with arguments (t0,T)
        Function that returns a list of M arrays of arrival times in [t0, t0+T]
    simulator_jumps: callable with argument N
        Function that returns a list of M arrays with the sizes of the jumps
    M: int
        Number of trajectories in the simulation

    Returns
    -------
    times_of_jumps: list [list [float]] with M elements
        Simulation consisting of M trajectories.
        Each trajectory is a row list composed of jump times.
    sizes_of_jumps: list [list [float]] with M elements
        Simulation consisting of M trajectories.
        Each trajectory is a row list composed of the
        sizes of the jumps at times_of_jumps.

    Example
    -------
    >>> import arrival_process_simulation as arrival
    >>> lambda_rate = 0.5
    >>> def simulator_arrival_times(t0, T):
    ...     return arrival.simulate_poisson(t0, t0 + T, lambda_rate, M=1)[0]
    >>> def simulator_jump_sizes(N): return 0.2 + 0.5*np.random.randn(N)
    >>> def simulator_jump_process(t0, T, M):
    ...     return sde.simulate_jump_process(t0, T,
    ...       simulator_arrival_times, simulator_jump_sizes, M)
    >>> t0, T, M = 0.0, 2000.0, 3
    >>> times_of_jumps, sizes_of_jumps = simulator_jump_process(t0, T, M)
    """
    times_of_jumps = [[] for _ in range(M)]
    sizes_of_jumps = [[] for _ in range(M)]

    for m in range(M):
        times_of_jumps[m] = simulator_arrival_times(t0, T)
        max_jumps = len(times_of_jumps[m])
        sizes_of_jumps[m] = simulator_jumps(max_jumps)

    return times_of_jumps, sizes_of_jumps


def euler_jump_diffusion(t0, x0, T, a, b, c, simulator_jump_process, M, N):
    """Simulation of jump diffusion process.

        x(t0) = x0
        dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t) + c(t, x(t-)) dJ(t)

        [Itô SDE with a jump term]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a : Function a(t,x(t)) that characterizes the drift term
    b : Function b(t,x(t)) that characterizes the diffusion term
    c : Function c(t,x(t)) that characterizes the jump term
    simulator_jump_process: Function that returns times and sizes of jumps
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t
    times_of_jumps: list [list [float]] with M elements
        Simulation consisting of M trajectories.
        Each trajectory is a row list composed of jump times.
    sizes_of_jumps: list [list [float]] with M elements
        Simulation consisting of M trajectories.
        Each trajectory is a row list composed of the
        sizes of the jumps at times_of_jumps.

    Example
    -------
    >>> import arrival_process_simulation as arrival
    >>> import stochastic_plots as stoch
    >>> lambda_rate = 0.5
    >>> def simulator_arrival_times(t0, T):
    ...     return arrival.simulate_poisson(t0, t0 + T, lambda_rate, M=1)[0]
    >>> def simulator_jump_sizes(N): return 0.2 + 0.5*np.random.randn(N)
    >>> def simulator_jump_process(t0, T, M):
    ...     return sde.simulate_jump_process(t0, T,
    ...       simulator_arrival_times, simulator_jump_sizes, M)
    >>> t0, x0, T, N, M = 0, 10.0, 2.0, 1000, 500
    >>> def a(t, x): return 5.0*x/x0
    >>> def b(t, x): return 3.0*x/x0
    >>> def c(t, x): return 10.0*x/x0
    >>> t, X_jump, _, _ = sde.euler_jump_diffusion(t0, x0, T, a, b, c,
    ...     simulator_jump_process, M, N)
    >>> stoch.plot_trajectories(t, X_jump, fig_num=10, max_trajectories=30)
    """
    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros((M, N + 1))

    # Initial condition
    X[:, 0] = np.full(M, x0)

    # Simulate jump process
    times_of_jumps, sizes_of_jumps = simulator_jump_process(t0, T, M)

    def compute_jumps_effect(X_prev_m, tn, taus, Ys):
        """Compute the aggregated effect for the mth simulation of all the jumps in [t_n, t_n+1].

        Parameters
        ----------
        tn : float
           Time at step n.
        X_prev_m : float
           Estimation at time tn for the mth simulation.
        taus : list [float]
            List of jump times for the mth simulation in [tn, tn+1].
        Ys : list [float]
            List of jump sizes for each jump in taus.

        Returns
        -------
        t_prev_m : float
            Time of last jump in [tn, tn+1]
        X_prev_m : float
            Estimation at time t_prev_m for the mth simulation.
        """
        t_prev_m = tn
        for (tau, Y) in zip(taus, Ys):
            dW = np.random.randn()
            dT_jump = tau - t_prev_m
            diffusion = (X_prev_m + a(t_prev_m, X_prev_m)*dT_jump
                         + b(t_prev_m, X_prev_m)*np.sqrt(dT_jump)*dW)
            t_prev_m = tau
            X_prev_m = diffusion + c(tau, diffusion)*Y

        return t_prev_m, X_prev_m

    # Integration of the SDE with jumps
    for n in range(N):
        # Select only the jumps between t_n and t_n+1 for each simulation
        jumps_mask = np.array([(tau > t[n]) & (tau < t[n + 1])
                               for tau in times_of_jumps], dtype=object)

        # Select only the simulations where there is at least one jump between t_n and t_n+1
        any_jump_mask = [jumps.any() for jumps in jumps_mask]

        # Compute possible jump effects
        t_prev = np.full(M, t[n])
        X_prev = X[:, n]
        for m in np.where(any_jump_mask)[0]:
            t_prev[m], X_prev[m] = compute_jumps_effect(X_prev[m], t[n],
                                                        times_of_jumps[m][jumps_mask[m]],
                                                        sizes_of_jumps[m][jumps_mask[m]])

        # Compute approximation at time t_n+1
        dW = np.random.randn(M)
        dT_jump = t[n + 1] - t_prev
        X[:, n + 1] = (X_prev + a(t_prev, X_prev)*dT_jump
                       + b(t_prev, X_prev)*np.sqrt(dT_jump)*dW)

    return t, X, times_of_jumps, sizes_of_jumps
