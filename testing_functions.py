import numpy as np


def sphere_function(x):
    """
    Classical sphere function $y(x) = \sum_{i=1}^n x_{i}^{2}
    with global minimum $y(0,...,0) = 0$
    :param x: input list of length n
    :return: the value of sphere function y(x)
    """
    return np.sum(np.asarray(x) ** 2)


def rastrigin_function(x):
    """
    Rastrigin function given as $y(x) = A * n + \sum_{i=1}^n(x_i^2 - A * cos(2 * \pi * x_i))$
    with global minimum $y(0,...,0) = 0$
    :param x: input list of length n
    :return: the value of Rastrigin function y(x)
    """
    A = 10
    return A * len(x) + np.sum(np.asarray(x) ** 2 - A * np.cos(2 * np.pi * np.asarray(x)))


def beale_function(x):
    """
    Beale function given as $f(x,y) = (1.5 - x - xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$
    with global minimum $f(3, 0.5) = 0$
    :param x: input list [x, y]
    :return: the value of Beale function f(x,y)
    """
    return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + \
           (2.625 - x[0] + x[0] * x[1] ** 3) ** 2


def booth_function(x):
    """
    Booth function given as $f(x,y) = (x + 2y - 7)^2 + (2 * x + y - 5)^2$
    with global minimum $f(1, 3) = 0$
    :param x: input list [x, y]
    :return: the value of Booth function f(x,y)
    """
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def matyas_function(x):
    """
    Matyas function given as $f(x,y) = 0.26 * (x^2 + y^2) - 0.48 * x * y$
    with global minimum $f(0, 0) = 0$
    :param x: input list [x, y]
    :return: the value of Matyas function f(x,y)
    """
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


def himmelblaus_function(x):
    """
    Himmelblaus function given as $f(x,y)=(x^2 + y - 11)^2 + (x + y^2 - 7)^2$
    with global minimums:
    $f(3,2) = 0$
    $f(-2.805118, 3.131312) = 0$
    $f(-3.779310, -3.283186) = 0$
    $f(3.584428, -1.848126) = 0$
    :param x: input list [x, y]
    :return: the value of Himmelblaus function f(x,y)
    """
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def bukin_function(x):
    """
    Bukin function given as $f(x,y)=100\sqrt{\lvert|y-0.01x^2 \rvert|} + 0.01\lvert| x + 10 \rvert|$
    with global minimum $f(-10,1)=0$
    :param x: input list [x, y]
    :return: value of Bukin function f(x,y)
    """
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0])) + 0.01 * np.abs(x[0] + 10)
