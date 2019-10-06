import numpy as np


def mean_squared_error(y, t):
    return np.sum((y - t) ** 2) / y.shape[0]


def least_error_square(x, t):
    pinvA = np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T)
    m, b = np.matmul(pinvA, t)
    return m, b