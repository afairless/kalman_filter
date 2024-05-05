 
import pytest
import numpy as np

from src.project import (
    kalman_updates_sequence_univariate,
    kalman_updates_sequence_multivariate,
    )


def esme_kalman_input_output():
    """
    Example from:

        http://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies
        Kalman Filter For Dummies
        Bilgin Esme
        March 2009     
    """

    x_input = [0.]     # state vector with initial state at zero
    p_input = [1.]     # state variance vector with initial state at one
    r_input = 0.1      # standard deviation of measurement noise

    z_input = [0.39, 0.50, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45]

    x_output = [
        x_input[0], 
        0.355, 0.424, 0.442, 0.405, 0.375, 
        0.366, 0.362, 0.377, 0.380, 0.387]
    p_output = [
        p_input[0], 
        0.091, 0.048, 0.032, 0.024, 0.020, 
        0.016, 0.014, 0.012, 0.011, 0.010]

    return x_input, p_input, r_input, z_input, x_output, p_output


def test_esme_example_univariate():
    """
    Test Esme example data with univariate Kalman filter
    """

    x, p, r, z, correct_x, correct_p = esme_kalman_input_output()

    result_x, result_p = kalman_updates_sequence_univariate(x, p, z, r)

    assert all([e[0] == round(e[1], 3) for e in zip(correct_x, result_x)])
    assert all([e[0] == round(e[1], 3) for e in zip(correct_p, result_p)])


def test_esme_example_multivariate():
    """
    Test Esme example data with multivariate Kalman filter
    """

    x0, p0, r, z, correct_x, correct_p = esme_kalman_input_output()

    result_x, result_p = kalman_updates_sequence_multivariate(
        x0=np.array(np.array(x0[0]).reshape(1, 1)),
        P0=np.array(np.array(p0[0]).reshape(1, 1)),
        F=np.array([1.]).reshape(1, 1),
        Q=np.array([0.]).reshape(1, 1),
        B=np.array([0.]).reshape(1, 1),
        u=np.array([0.]).reshape(1, 1),
        H=np.array([1.]).reshape(1, 1),
        R=np.array(r).reshape(1, 1),
        z_vec=z)

    assert all([e[0] == np.round(e[1], 3)[0][0] for e in zip(correct_x, result_x)])
    assert all([e[0] == np.round(e[1], 3)[0][0] for e in zip(correct_p, result_p)])


