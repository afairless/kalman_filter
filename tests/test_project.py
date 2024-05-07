 
import numpy as np

from src.project import (
    kalman_updates_sequence_univariate,
    kalman_update_multivariate,
    kalman_update_multivariate_reannotated,
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
        z_vec=z,
        kalman_function=kalman_update_multivariate)

    assert all([e[0] == np.round(e[1], 3)[0][0] for e in zip(correct_x, result_x)])
    assert all([e[0] == np.round(e[1], 3)[0][0] for e in zip(correct_p, result_p)])


def test_esme_example_multivariate_reannotated():
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
        z_vec=z,
        kalman_function=kalman_update_multivariate_reannotated)

    assert all([e[0] == np.round(e[1], 3)[0][0] for e in zip(correct_x, result_x)])
    assert all([e[0] == np.round(e[1], 3)[0][0] for e in zip(correct_p, result_p)])


def labbe_chapter_6_kalman_input_output_01():
    """
    Example from:

        https://rlabbe.github.io/Kalman-and-Bayesian-Filters-in-Python/
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
        Kalman and Bayesian Filters in Python
        Roger R Labbe Jr
        May 23, 2020

        Chapter 6 Multivariate Kalman Filters
        6.12 Adjusting the Filter

        PDF version, page 217-218:
        https://drive.google.com/open?id=0By_SW19c1BfhSVFzNHc0SjduNzg
    """

    # trk = [ 
    #     0.941,  1.639,  2.385,  3.456,  4.307,  5.385,  6.391,
    #     7.467,  8.464,  9.358, 10.234, 11.271, 12.223, 13.132,
    #    13.931, 14.893, 15.549, 16.601, 17.672, 18.672, 19.628,
    #    20.364, 21.571, 22.658, 23.541, 24.682, 25.629, 26.69 ,
    #    27.601, 28.631, 29.572, 30.566, 31.278, 32.182, 33.256,
    #    34.164, 35.11 , 35.954, 36.578, 37.676, 38.65 , 39.766,
    #    40.709, 41.903, 42.837, 43.891, 44.92 , 45.674, 46.728,
    #    47.894]

    x_input = [0., 0.]

    # unclear whether this is intended initialization of P, because it comes
    #   from an earlier, completed filtering run, but it is the one that is
    #   input into the example
    p_input = np.array([[2.222, 0.279], [0.279, 0.075]])
    r_input = 225
    q_input = 200

    z_input = [
         0.097,  26.243, -10.241, -15.223,  -9.328,  39.768, -10.378,
        -1.475,  26.089,   9.494,   7.888,  -3.561,   8.68 ,  -4.682,
        11.629,  48.364,  17.24 ,  36.996,   5.009,  26.808,  31.193,
        46.332,  16.536,  23.378,  24.856,  18.966,  24.512,  45.866,
        35.227,   0.752,  27.587,  35.456,  31.972,  10.59 ,  44.285,
        46.8  ,  36.107,  59.722,  35.206,   7.174,  37.491,  58.49 ,
        19.941,  60.171,  48.101,  52.385,  66.021,  61.287,  43.471,
        12.74 ]

    x_output = np.array([
        x_input, 
        [  0.018,   0.035],
        [ 17.095,  14.066],
        [  0.83 ,  -5.591],
        [-12.43 , -10.508],
        [-12.912,  -3.996],
        [ 25.001,  23.25 ],
        [  4.861,  -4.931],
        [ -1.11 ,  -5.607],
        [ 17.563,  10.16 ],
        [ 14.231,   1.399],
        [  9.9  ,  -2.322],
        [ -0.666,  -7.676],
        [  4.256,   0.505],
        [ -2.228,  -4.033],
        [  6.98 ,   4.565],
        [ 38.795,  22.261],
        [ 28.627,   1.202],
        [ 35.133,   4.647],
        [ 14.045, -12.065],
        [ 20.356,  -0.132],
        [ 28.342,   5.14 ],
        [ 42.992,  11.316],
        [ 26.352,  -6.838],
        [ 22.373,  -4.982],
        [ 22.917,  -1.394],
        [ 19.63 ,  -2.623],
        [ 22.562,   0.984],
        [ 40.065,  11.712],
        [ 39.528,   3.757],
        [ 11.805, -16.685],
        [ 19.149,  -1.081],
        [ 30.938,   7.276],
        [ 33.594,   4.276],
        [ 17.68 ,  -8.835],
        [ 35.075,   8.199],
        [ 45.884,   9.894],
        [ 41.219,   0.44 ],
        [ 55.027,   9.121],
        [ 42.728,  -4.789],
        [ 15.169, -19.575],
        [ 26.603,   0.562],
        [ 50.349,  15.617],
        [ 31.902,  -6.504],
        [ 51.134,  10.209],
        [ 51.542,   3.844],
        [ 53.165,   2.402],
        [ 63.304,   7.426],
        [ 63.741,   2.887],
        [ 49.489,  -8.243],
        [ 20.148, -21.944]])

    p_output = np.array([
        p_input,
        [[ 42.8  ,  81.264],
         [ 81.264, 163.83 ]],
        [[146.409, 120.539],
         [120.539, 178.953]],
        [[164.835, 106.824],
         [106.824, 189.286]],
        [[164.93 , 105.752],
         [105.752, 203.11 ]],
        [[165.758, 107.653],
         [107.653, 207.487]],
        [[166.376, 108.166],
         [108.166, 207.915]],
        [[166.516, 108.151],
         [108.151, 207.916]],
        [[166.524, 108.134],
         [108.134, 207.956]],
        [[166.525, 108.138],
         [108.138, 207.978]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]],
        [[166.527, 108.141],
         [108.141, 207.982]]])

    return x_input, p_input, r_input, q_input, z_input, x_output, p_output


def labbe_chapter_6_kalman_input_output_02():
    """
    Example from:

        https://rlabbe.github.io/Kalman-and-Bayesian-Filters-in-Python/
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
        Kalman and Bayesian Filters in Python
        Roger R Labbe Jr
        May 23, 2020

        Chapter 6 Multivariate Kalman Filters
        6.12 Adjusting the Filter

        PDF version, page 217-218:
        https://drive.google.com/open?id=0By_SW19c1BfhSVFzNHc0SjduNzg
    """

    # trk = [ 
    #     0.941,  1.639,  2.385,  3.456,  4.307,  5.385,  6.391,
    #     7.467,  8.464,  9.358, 10.234, 11.271, 12.223, 13.132,
    #    13.931, 14.893, 15.549, 16.601, 17.672, 18.672, 19.628,
    #    20.364, 21.571, 22.658, 23.541, 24.682, 25.629, 26.69 ,
    #    27.601, 28.631, 29.572, 30.566, 31.278, 32.182, 33.256,
    #    34.164, 35.11 , 35.954, 36.578, 37.676, 38.65 , 39.766,
    #    40.709, 41.903, 42.837, 43.891, 44.92 , 45.674, 46.728,
    #    47.894]

    x_input = [0., 0.]

    # unclear whether this is intended initialization of P, because it comes
    #   from an earlier, completed filtering run, but it is the one that is
    #   input into the example
    p_input = np.array([[2.222, 0.279], [0.279, 0.075]])
    r_input = 225
    q_input = 0.02

    z_input = [
         0.097,  26.243, -10.241, -15.223,  -9.328,  39.768, -10.378,
        -1.475,  26.089,   9.494,   7.888,  -3.561,   8.68 ,  -4.682,
        11.629,  48.364,  17.24 ,  36.996,   5.009,  26.808,  31.193,
        46.332,  16.536,  23.378,  24.856,  18.966,  24.512,  45.866,
        35.227,   0.752,  27.587,  35.456,  31.972,  10.59 ,  44.285,
        46.8  ,  36.107,  59.722,  35.206,   7.174,  37.491,  58.49 ,
        19.941,  60.171,  48.101,  52.385,  66.021,  61.287,  43.471,
        12.74 ]

    x_output = np.array([
        x_input, 
        [ 0.001,  0.   ],
        [ 0.419,  0.053],
        [ 0.257,  0.026],
        [-0.105, -0.021],
        [-0.411, -0.055],
        [ 1.045,  0.117],
        [ 0.644,  0.061],
        [ 0.59 ,  0.049],
        [ 2.183,  0.204],
        [ 2.876,  0.251],
        [ 3.492,  0.285],
        [ 3.159,  0.23 ],
        [ 3.871,  0.271],
        [ 3.281,  0.199],
        [ 4.321,  0.267],
        [ 9.309,  0.638],
        [10.763,  0.701],
        [14.405,  0.922],
        [14.111,  0.832],
        [16.367,  0.935],
        [18.994,  1.056],
        [23.283,  1.284],
        [23.571,  1.214],
        [24.61 ,  1.202],
        [25.693,  1.194],
        [25.892,  1.126],
        [26.702,  1.104],
        [30.083,  1.259],
        [31.833,  1.293],
        [29.037,  1.015],
        [29.741,  0.994],
        [31.332,  1.035],
        [32.316,  1.031],
        [30.468,  0.836],
        [32.947,  0.947],
        [35.53 ,  1.059],
        [36.527,  1.054],
        [40.39 ,  1.246],
        [40.82 ,  1.19 ],
        [37.585,  0.888],
        [38.348,  0.879],
        [41.678,  1.047],
        [39.824,  0.848],
        [43.157,  1.019],
        [44.676,  1.053],
        [46.579,  1.111],
        [50.03 ,  1.272],
        [52.578,  1.359],
        [52.599,  1.267],
        [48.607,  0.906]])

    p_output = np.array([
        p_input,
        [[ 2.824,  0.359],
         [ 0.359,  0.094]],
        [[ 3.583,  0.456],
         [ 0.456,  0.113]],
        [[ 4.52 ,  0.567],
         [ 0.567,  0.132]],
        [[ 5.646,  0.691],
         [ 0.691,  0.15 ]],
        [[ 6.961,  0.824],
         [ 0.824,  0.166]],
        [[ 8.451,  0.963],
         [ 0.963,  0.182]],
        [[10.091,  1.104],
         [ 1.104,  0.196]],
        [[11.841,  1.241],
         [ 1.241,  0.209]],
        [[13.656,  1.372],
         [ 1.372,  0.22 ]],
        [[15.48 ,  1.492],
         [ 1.492,  0.23 ]],
        [[17.264,  1.599],
         [ 1.599,  0.237]],
        [[18.959,  1.69 ],
         [ 1.69 ,  0.244]],
        [[20.528,  1.767],
         [ 1.767,  0.248]],
        [[21.943,  1.827],
         [ 1.827,  0.252]],
        [[23.19 ,  1.874],
         [ 1.874,  0.254]],
        [[24.264,  1.908],
         [ 1.908,  0.256]],
        [[25.171,  1.931],
         [ 1.931,  0.258]],
        [[25.92 ,  1.945],
         [ 1.945,  0.259]],
        [[26.528,  1.953],
         [ 1.953,  0.259]],
        [[27.013,  1.955],
         [ 1.955,  0.26 ]],
        [[27.392,  1.955],
         [ 1.955,  0.261]],
        [[27.683,  1.952],
         [ 1.952,  0.261]],
        [[27.903,  1.947],
         [ 1.947,  0.262]],
        [[28.065,  1.943],
         [ 1.943,  0.263]],
        [[28.183,  1.938],
         [ 1.938,  0.264]],
        [[28.267,  1.934],
         [ 1.934,  0.265]],
        [[28.326,  1.931],
         [ 1.931,  0.266]],
        [[28.367,  1.929],
         [ 1.929,  0.267]],
        [[28.396,  1.928],
         [ 1.928,  0.268]],
        [[28.416,  1.927],
         [ 1.927,  0.269]],
        [[28.432,  1.928],
         [ 1.928,  0.27 ]],
        [[28.446,  1.929],
         [ 1.929,  0.271]],
        [[28.459,  1.931],
         [ 1.931,  0.272]],
        [[28.473,  1.933],
         [ 1.933,  0.273]],
        [[28.488,  1.936],
         [ 1.936,  0.274]],
        [[28.504,  1.939],
         [ 1.939,  0.275]],
        [[28.522,  1.942],
         [ 1.942,  0.276]],
        [[28.541,  1.946],
         [ 1.946,  0.277]],
        [[28.562,  1.949],
         [ 1.949,  0.277]],
        [[28.583,  1.952],
         [ 1.952,  0.278]],
        [[28.605,  1.956],
         [ 1.956,  0.279]],
        [[28.627,  1.959],
         [ 1.959,  0.279]],
        [[28.648,  1.962],
         [ 1.962,  0.279]],
        [[28.669,  1.964],
         [ 1.964,  0.28 ]],
        [[28.69 ,  1.967],
         [ 1.967,  0.28 ]],
        [[28.709,  1.969],
         [ 1.969,  0.28 ]],
        [[28.727,  1.971],
         [ 1.971,  0.281]],
        [[28.744,  1.972],
         [ 1.972,  0.281]],
        [[28.759,  1.974],
         [ 1.974,  0.281]],
        [[28.774,  1.975],
         [ 1.975,  0.281]]])

    return x_input, p_input, r_input, q_input, z_input, x_output, p_output


def test_labbe_chapter_6_example_1_multivariate():
    """
    Test Labbe Chapter 6 example #1 data with multivariate Kalman filter
    """

    x0, p0, r, q, z, correct_x, correct_p = (
        labbe_chapter_6_kalman_input_output_01())

    result_x, result_p = kalman_updates_sequence_multivariate(
        x0=np.array(np.array(x0).reshape(2, 1)),
        P0=p0,
        F=np.array([[1., 1.], [0., 1.]]),
        Q=np.array([[50., 100.], [100., q]]),
        B=np.array([0.]).reshape(1, 1),
        u=np.array([0.]).reshape(1, 1),
        H=np.array([1., 0.]).reshape(1, 2),
        R=np.array(r).reshape(1, 1),
        z_vec=z,
        kalman_function=kalman_update_multivariate)

    result_x_arr = np.stack(result_x).reshape(-1, 2)
    assert np.allclose(result_x_arr, correct_x, atol=1e-3)

    result_p_arr = np.stack(result_p)
    assert np.allclose(result_p_arr, correct_p, atol=1e-3)


def test_labbe_chapter_6_example_2_multivariate():
    """
    Test Labbe Chapter 6 example #2 data with multivariate Kalman filter
    """

    x0, p0, r, q, z, correct_x, correct_p = (
        labbe_chapter_6_kalman_input_output_02())

    result_x, result_p = kalman_updates_sequence_multivariate(
        x0=np.array(np.array(x0).reshape(2, 1)),
        P0=p0,
        F=np.array([[1., 1.], [0., 1.]]),
        Q=np.array([[0.005, 0.01], [0.01, q]]),
        B=np.array([0.]).reshape(1, 1),
        u=np.array([0.]).reshape(1, 1),
        H=np.array([1., 0.]).reshape(1, 2),
        R=np.array(r).reshape(1, 1),
        z_vec=z,
        kalman_function=kalman_update_multivariate)

    result_x_arr = np.stack(result_x).reshape(-1, 2)
    assert np.allclose(result_x_arr, correct_x, atol=1e-2)

    result_p_arr = np.stack(result_p)
    assert np.allclose(result_p_arr, correct_p, atol=1e-1)


def test_labbe_chapter_6_example_1_multivariate_reannotated():
    """
    Test Labbe Chapter 6 example #1 data with multivariate Kalman filter
    """

    x0, p0, r, q, z, correct_x, correct_p = (
        labbe_chapter_6_kalman_input_output_01())

    result_x, result_p = kalman_updates_sequence_multivariate(
        x0=np.array(np.array(x0).reshape(2, 1)),
        P0=p0,
        F=np.array([[1., 1.], [0., 1.]]),
        Q=np.array([[50., 100.], [100., q]]),
        B=np.array([0.]).reshape(1, 1),
        u=np.array([0.]).reshape(1, 1),
        H=np.array([1., 0.]).reshape(1, 2),
        R=np.array(r).reshape(1, 1),
        z_vec=z,
        kalman_function=kalman_update_multivariate_reannotated)

    result_x_arr = np.stack(result_x).reshape(-1, 2)
    assert np.allclose(result_x_arr, correct_x, atol=1e-3)

    result_p_arr = np.stack(result_p)
    assert np.allclose(result_p_arr, correct_p, atol=1e-3)


def test_labbe_chapter_6_example_2_multivariate_reannotated():
    """
    Test Labbe Chapter 6 example #2 data with multivariate Kalman filter
    """

    x0, p0, r, q, z, correct_x, correct_p = (
        labbe_chapter_6_kalman_input_output_02())

    result_x, result_p = kalman_updates_sequence_multivariate(
        x0=np.array(np.array(x0).reshape(2, 1)),
        P0=p0,
        F=np.array([[1., 1.], [0., 1.]]),
        Q=np.array([[0.005, 0.01], [0.01, q]]),
        B=np.array([0.]).reshape(1, 1),
        u=np.array([0.]).reshape(1, 1),
        H=np.array([1., 0.]).reshape(1, 2),
        R=np.array(r).reshape(1, 1),
        z_vec=z,
        kalman_function=kalman_update_multivariate_reannotated)

    result_x_arr = np.stack(result_x).reshape(-1, 2)
    assert np.allclose(result_x_arr, correct_x, atol=1e-2)

    result_p_arr = np.stack(result_p)
    assert np.allclose(result_p_arr, correct_p, atol=1e-1)


