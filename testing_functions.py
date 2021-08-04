import numpy as np


def sphere_function(x):
    """
    Classical sphere function $y(x) = \sum_{i=1}^n x_{i}^{2}
    :param x: input vector of length n
    :return: the value of sphere function y(x)
    """
    return np.sum(np.asarray(x) ** 2)


def rastrigin_function(x):
    """
    Rastrigin function given as $y(x) = A * n + \sum_{i=1}^n(x_i^2 - A * cos(2 * \pi * x_i))$
    :param x: input vector of length n
    :return: the value of Rastrigin function y(x)
    """
    A = 10
    return A * len(x) + np.sum(np.asarray(x) ** 2 - A * np.cos(2 * np.pi * np.asarray(x)))

