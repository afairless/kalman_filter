#! /usr/bin/env python3

import numpy as np
from typing import Iterable, Sequence
from scipy.linalg import inv as scipy_inv


def kalman_update_univariate(
    x: float, p: float, z: float, r: float) -> tuple[float, float]:
    """
    Calculates one Kalman filter update step for univariate state

    'x' - state estimate
    'p' - state variance estimate
    'z' - measurement / observation / data point
    'r' - measurement variance 
    'k' - Kalman gain
    """

    k = p / (p + r)
    x_next = x + k * (z - x)
    p_next = (1 - k) * p

    return x_next, p_next


def kalman_updates_sequence_univariate(
    x: list[float], p: list[float], z: Iterable[float], r: float, 
    ) -> tuple[list[float], list[float]]:
    """
    Calculates a sequence of Kalman filter updates for univariate state
    """

    for z_i in z:
        x_i, p_i = kalman_update_univariate(x[-1], p[-1], z_i, r)
        x.append(x_i)
        p.append(p_i)

    return x, p


def kalman_update_multivariate(
    x0: np.ndarray, P0: np.ndarray, 
    F: np.ndarray, Q: np.ndarray, 
    B: np.ndarray, u: np.ndarray, 
    H: np.ndarray, R: np.ndarray, 
    z: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates one Kalman filter update step for multivariate state

    Input parameters:
        'x0' - state estimate
        'P0' - state covariance estimate
        'F' - process model / state transition matrix
        'Q' - process noise
        'B' - control input model / control function
        'u' - control input
        'H' - measurement function
        'R' - measurement noise 
        'z' - measurement / observation / data point

    Intermediate variables:
        'x1' - predicted state at next time step
        'P1' - predicted state covariance at next time step
        'S' - system uncertainty / innovation covariance / predicted state 
            covariance projected into measurement space
        'K' - Kalman gain / scaling factor
        'y' - residual between predicted state and measurement in measurement 
            space

    Output variables:
        'x2' - updated state estimate
        'P2' - updated state covariance estimate

    Adapted from:
        https://rlabbe.github.io/Kalman-and-Bayesian-Filters-in-Python/
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
        Kalman and Bayesian Filters in Python
        Roger R Labbe Jr
        May 23, 2020

        Chapter 6 Multivariate Kalman Filters
        6.9 The Kalman Filter Equations
        6.9.3 An Example not using FilterPy

        PDF version, page 211:
        https://drive.google.com/open?id=0By_SW19c1BfhSVFzNHc0SjduNzg
    """

    # predict
    x1 = F @ x0 + B @ u
    P1 = F @ P0 @ F.T + Q

    # update
    S = H @ P1 @ H.T + R
    K = P1 @ H.T @ scipy_inv(S)
    y = z - H @ x1
    x2 = x1 + K @ y
    P2 = P1 - K @ H @ P1

    return x2, P2


def kalman_updates_sequence_multivariate(
    x0: np.ndarray, P0: np.ndarray, 
    F: np.ndarray, Q: np.ndarray, 
    B: np.ndarray, u: np.ndarray, 
    H: np.ndarray, R: np.ndarray, 
    z_vec: Sequence[float]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Calculates a sequence of Kalman filter updates for multivariate state
    """

    xs = [x0]
    cov = [P0]
    for z_i in z_vec:
        x, P = kalman_update_multivariate(
            xs[-1], cov[-1], F, Q, B, u, H, R, z_i)
        xs.append(x)
        cov.append(P)

    return xs, cov


def main():

    print('This is the main function.')



if __name__ == '__main__':
    main()
