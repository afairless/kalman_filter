#! /usr/bin/env python3

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from typing import Iterable, Sequence, Callable
from scipy.linalg import inv as scipy_inv
from pathlib import Path


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


def kalman_update_multivariate_reannotated(
    a_t: np.ndarray, P_t: np.ndarray, 
    T: np.ndarray, Q: np.ndarray, 
    B: np.ndarray, u: np.ndarray, 
    Z: np.ndarray, H: np.ndarray, 
    y_t: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates one Kalman filter update step for multivariate state

    The variable names in this function have been changed to match Durbin and 
        Koopman (2012) 
    In the variable/parameter descriptions below, the first name matches 
        nomenclature in most of the control research literature, including 
        Labbe (2020), and in the function 'kalman_update_multivariate' above;
        the second name corresponds to Durbin and Koopman (2012)

    Input parameters:
        'x0' / 'a_t' - state estimate
        'P0' / 'P_t' - state covariance estimate
        'F' / 'T' - process model / state transition matrix
        'Q' - process noise / state disturbance covariance matrix
        'B' / omitted - control input model / control function
        'u' / omitted - control input
        'H' / 'Z' - measurement function / design matrix
        'R' / 'H'- measurement noise / observation disturbance covariance matrix
        'z' / 'y_t' - measurement / observation / data point
        omitted / 'R' - selection matrix

    Intermediate variables:
        'x1' / 'a1' - predicted state at next time step
        'P1' - predicted state covariance at next time step
        'S' / 'F' - system uncertainty / innovation covariance / predicted state 
            covariance projected into measurement space
        'K' - Kalman gain / scaling factor
        'y' / 'v_t' - residual between predicted state and measurement in 
            measurement space

    Output variables:
        'x2' / 'a_t1' - updated state estimate
        'P2' / 'P_t1'- updated state covariance estimate

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

    Re-annotated to match:
        Time Series Analysis by State Space Methods, 2nd Edition
        J. Durbin and S.J. Koopman
        Oxford University Press, 2012
        ISBN: 978-0-19-964117-8

    Some terminology taken from:
        https://www.chadfulton.com/files/fulton_statsmodels_2017_v1.pdf
        Estimating time series models by state space methods in Python: 
            Statsmodels
        Chad Fulton
        2017
    """

    # the selection matrix 'R' is often but not always the identity matrix
    #   Durbin and Koopman (2012), pages 43-44
    # I am restricting 'R' to the identity matrix and excluding it from the 
    #   input parameters to show where it fits into calculations without 
    #   changing behavior of the function, compared to 
    #   'kalman_update_multivariate'
    R = np.eye(T.shape[0], Q.shape[0])
    assert np.all(Q == R @ Q @ R.T)

    # predict
    a1 = T @ a_t #+ B @ u
    P1 = T @ P_t @ T.T + R @ Q @ R.T

    # update
    F = Z @ P1 @ Z.T + H
    K = P1 @ Z.T @ scipy_inv(F)
    v_t = y_t - Z @ a1
    a_t1 = a1 + K @ v_t
    P_t1 = P1 - K @ Z @ P1

    return a_t1, P_t1 


def kalman_updates_sequence_multivariate(
    x0: np.ndarray, P0: np.ndarray, 
    F: np.ndarray, Q: np.ndarray, 
    B: np.ndarray, u: np.ndarray, 
    H: np.ndarray, R: np.ndarray, 
    z_vec: Sequence[float],
    kalman_function: Callable) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Calculates a sequence of Kalman filter updates for multivariate state
    """

    xs = [x0]
    cov = [P0]
    for z_i in z_vec:
        x, P = kalman_function(xs[-1], cov[-1], F, Q, B, u, H, R, z_i)
        xs.append(x)
        cov.append(P)

    return xs, cov


def plot_kalman_results(
    data_x_time: np.ndarray, data_y: np.ndarray, 
    state: np.ndarray, state_variance: np.ndarray,
    output_path: Path):
    """
    Plot results from Kalman filter pass:
        1) filter state and observed data vs. time
        2) filter variance vs. time
        3) filter prediction errors vs. time

    'data_x_time' - observed data time points
    'data_y_time' - observed data y values
    'state' - univariate filter state (filtered or smooth)
    'state_variance' - univariate filter state variance
    'output_path' - directory in which to save plots
    """

    filename = 'filter_state_with_data.png'
    filepath = output_path / filename
    fig = plt.scatter(data_x_time, data_y)
    fig = plt.plot(data_x_time, state, color='black')
    fig = plt.title('Filter state with observed data')
    plt.savefig(filepath)
    plt.clf()
    plt.close()

    filename = 'filter_variance.png'
    filepath = output_path / filename
    fig = plt.plot(data_x_time, state_variance)
    fig = plt.title('Filter variance')
    plt.savefig(filepath)
    plt.clf()
    plt.close()

    filename = 'prediction_errors.png'
    filepath = output_path / filename
    fig = plt.plot(data_x_time, data_y - state)
    fig = plt.hlines(0, data_x_time.min(), data_x_time.max(), colors='black')
    fig = plt.title('Filter prediction errors')
    plt.savefig(filepath)
    plt.clf()
    plt.close()


def durbin_koopman_figure_2_1_multivariate():
    """
    Replicates results from a Kalman filter pass over the classic Nile River 
        data set as presented in Durbin and Koopman (2012) Figure 2.1, page 16

    Reference:
        Time Series Analysis by State Space Methods, 2nd Edition
        J. Durbin and S.J. Koopman
        Oxford University Press, 2012
        ISBN: 978-0-19-964117-8
    """

    df = sm.datasets.nile.load().data
    assert isinstance(df, pd.DataFrame)
    y = df['volume']
    t = df['year'].values

    # filter initialization from Durbin and Koopman (2012)
    a = [0.]
    P = [1e7]
    sig_e = 15099
    sig_n = 1469.1

    result_x, result_p = kalman_updates_sequence_multivariate(
        x0=np.array(np.array(a[0]).reshape(1, 1)),
        P0=np.array(np.array(P[0]).reshape(1, 1)),
        F=np.array([1.]).reshape(1, 1),
        Q=np.array([sig_n]).reshape(1, 1),
        B=np.array([0.]).reshape(1, 1),
        u=np.array([0.]).reshape(1, 1),
        H=np.array([1.]).reshape(1, 1),
        R=np.array(sig_e).reshape(1, 1),
        z_vec=y,
        kalman_function=kalman_update_multivariate)

    result_x_arr = np.stack(result_x).reshape(-1)
    result_p_arr = np.stack(result_p).reshape(-1)

    output_path = Path.cwd() / 'output'
    plot_kalman_results(t, y, result_x_arr[1:], result_p_arr[1:], output_path)


def main():

    durbin_koopman_figure_2_1_multivariate()


if __name__ == '__main__':
    main()
