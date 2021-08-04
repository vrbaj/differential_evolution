import numpy as np


def sphere_function(x):
    """
    Classical sphere function $y(x) = \sum_{i=1}^n x_{i}^{2}
    :param x: input list of length n
    :return: the value of sphere function y(x)
    """
    return np.sum(np.asarray(x) ** 2)


def rastrigin_function(x):
    """
    Rastrigin function given as $y(x) = A * n + \sum_{i=1}^n(x_i^2 - A * cos(2 * \pi * x_i))$
    with global minimum y(0,....0) = 0
    :param x: input list of length n
    :return: the value of Rastrigin function y(x)
    """
    A = 10
    return A * len(x) + np.sum(np.asarray(x) ** 2 - A * np.cos(2 * np.pi * np.asarray(x)))


def beale_function(x):
    """
    Beale function given as $f(x,y) = (1.5 - x - xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$
    with global minimum f(3, 0.5) = 0
    :param x: input list [x, y]
    :return: the value of Beale function f(x,y)
    """
    return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + \
           (2.625 - x[0] + x[0] * x[1] ** 3) ** 2


def booth_function(x):
    """
    Booth function given as $f(x,y) = (x + 2y - 7)^2 + (2 * x + y - 5)^2$
    :param x: input list [x, y]
    :return: the value of Booth function f(x,y)
    """
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] +x[1] - 5) ** 2
