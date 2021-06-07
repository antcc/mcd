# -*- coding: utf-8 -*-

"""
Collection of search algorithms for unconstrained optimization.

Authors:  Antonio Coín Castro
          Luis Antonio Ortega Andrés
"""

from typing import Callable, Tuple, List
import numpy as np


def dichotomic_search(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    epsilon: float,
    uncertainty_length: float,
    max_iter: int = 50,
    verbose: bool = False,
) -> Tuple[float, float, int, List[Tuple[float, float]]]:
    """
    Perform dichotomic search for a function on a given interval.

    Arguments
    ---------
    f : callable
        Function to minimize. Must be strictly quasiconvex in [lower, upper].
    lower : float
        Lower end of initial interval.
    upper : float
        Upper end of initial interval.
    epsilon: float
        Half length of the uncertainty intervals (lamb, mu).
    uncertainty_length : float
        Maximum length of final interval containing the minimum value.
    max_iter : int
        Maximum number of iterations.
    verbose : bool
        Whether to print information after each iteration.

    Returns
    -------
    lower : float
        Lower end of final interval.
    upper : float
        Upper end of final interval.
    it : int
        Number of iterations.
    evol : List[Tuple[float, float]]
        Temporal evolution of the uncertainty intervals.

    Example
    -------
    >>> def f(x): return x**2 - 2
    >>> a1, b1 = -5, 5
    >>> length = 0.001
    >>> eps = length/10.
    >>> lower, upper, _, _ = dichotomic_search(f, a1, b1, eps, length)
    >>> print(f"[{lower:.4f}, {upper:.4f}]")
    [-0.0001, 0.0007]
    """
    evol = []
    converged = False

    # Loop through iterations
    for it in range(max_iter):
        # Save evolution for later usage
        evol.append((lower, upper))

        if verbose:
            print(f"Iteration {it}, uncertainty interval = "
                  f"[{lower:.4f}, {upper:.4f}]")

        # Check stopping condition
        if upper - lower < uncertainty_length:
            converged = True
            break

        # Define prospective new bounds
        lamb = (lower + upper)/2. - epsilon
        mu = (lower + upper)/2. + epsilon

        # Update current uncertainty interval
        if f(lamb) < f(mu):
            upper = mu
        else:
            lower = lamb

    if not converged:
        print("[dichotomic_search] Convergence failed.")

    return lower, upper, it, evol


def golden_search(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    uncertainty_length: float,
    max_iter: int = 50,
    verbose: bool = False,
) -> Tuple[float, float, int, List[Tuple[float, float]]]:
    """
    Perform golden-section search for a function on a given interval.

    Arguments
    ---------
    f : callable
        Function to minimize. Must be strictly quasiconvex in [lower, upper].
    lower : float
        Lower end of initial interval.
    upper : float
        Upper end of initial interval.
    uncertainty_length : float
        Maximum length of final interval containing the minimum value.
    max_iter : int
        Maximum number of iterations.
    verbose : bool
        Whether to print information after each iteration.

    Returns
    -------
    lower : float
        Lower end of final interval.
    upper : float
        Upper end of final interval.
    it : int
        Number of iterations.
    evol : List[Tuple[float, float]]
        Temporal evolution of the uncertainty intervals.

    Example
    -------
    >>> def f(x): return x**2 - 2
    >>> a1, b1 = -5, 5
    >>> length = 0.001
    >>> eps = length/10.
    >>> lower, upper, _, _ = golden_search(f, a1, b1, length)
    >>> print(f"[{lower:.4f}, {upper:.4f}]")
    [-0.0004, 0.0002]
    """
    evol = []
    converged = False
    ratio = 0.618

    # Initial prospective bounds
    lamb = lower + (1 - ratio)*(upper - lower)
    mu = lower + ratio*(upper - lower)

    # Loop through iterations
    for it in range(max_iter):
        # Save evolution for later usage
        evol.append((lower, upper))

        if verbose:
            print(f"Iteration {it}, uncertainty interval = "
                  f"[{lower:.4f}, {upper:.4f}]")

        # Check stopping condition
        if upper - lower < uncertainty_length:
            converged = True
            break

        # Update current uncertainty interval and prospective bounds
        if f(lamb) > f(mu):
            lower = lamb
            lamb = mu
            mu = lower + ratio*(upper - lower)
        else:
            upper = mu
            mu = lamb
            lamb = lower + (1 - ratio)*(upper - lower)

    if not converged:
        print("[golden_search] Convergence failed.")

    return lower, upper, it, evol


def newton(
    grad: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 1.0,
    epsilon: float = 1e-5,
    max_iter: int = 50,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Approximate minimum solution by Newton's multivariate method.

    Compute the quadratic approximation of f(x_k) and find the next
    point by the formula

            x_k+1 = x_k - lr * (hessian(x_k))^-1 * grad(x_k)

    Continue until ||x_k+1 - x_k|| < epsilon or until the maximum
    number of iterations is reached.

    Parameters
    ----------
    grad : callable
        Gradient of the objective function.
    hessian : callable
        Hessian matrix of the objective function
    x0 : np.ndarray
        Initial guess for the solution.
    lr : float
        Learning rate for the relaxed method.
    epsilon : float
        Tolerance for stopping condition.
    max_iter : integer
        Maximum number of iterations.

    Raises
    ------
    LinAlgError
        If the Hessian matrix is not invertible at a point.

    Returns
    -------
    x : np.ndarray
        Optimal solution found.
    it : int
        Number of iterations.
    evol : np.ndarray
        Temporal evolution of the approximations.

    Examples
    --------
    >>> def f(x): return x[0]**2 + x[1]**2
    >>> def df(x): return 2*x
    >>> def Hf(x): return 2*np.eye(2)
    >>> sol, _, _ = newton(df, Hf, np.array([1.0, 3.0]))
    >>> print(sol)
    [0. 0.]
    """
    x_prev = x0.copy()
    converged = False
    evol = [x0]

    for it in range(max_iter):
        H = hessian(x_prev)

        # Invert Hessian
        try:
            H_inv = np.linalg.solve(H, np.eye(H.shape[0]))
        except np.linalg.LinAlgError:
            print(
                f"[newton] Hessian matrix at {x_prev} is singular.")
            return None

        # Update approximation
        x = x_prev - lr*grad(x_prev)@H_inv

        # Check stopping condition
        if (np.linalg.norm(x - x_prev) < epsilon):
            converged = True
            break

        # Advance iteration
        x_prev = x
        evol.append(x_prev.copy())

    if not converged:
        print("[newton] Convergence failed.")

    return x, it, np.array(evol)


if __name__ == "__main__":
    import doctest
    doctest.testmod()  # Test cases
