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


def mccormick_function(x):
    """
    McCormick function given as $f(x,y)=sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1$
    with global minimum $f(-0.54719,-1.54719)=-1.9133$
    :param x: input list [x, y]
    :return: value of McCormick function $f(x,y)$
    """
    return np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1


def three_hump_camel_function(x):
    """
    Three-hump camel function given as $f(x,y)=2x^2-1.05x^4+\frac{x^6}{6}+xy+y^2$
    with global minimum $f(0,0)=0$
    :param x: input list [x, y]
    :return: value of three-hump camel function $f(x,y)$
    """
    return 2 * x[0] - 1.05 * x[0] ** 4 + 1 / 6 * x[0] ** 6 + x[0] * x[1] + x[1] ** 2


def ackley_function(x):
    """
    Ackley function given as $-20exp\[-0.2\sqrt{0.5(x^2+y^2)}\]-exp\[0.5(cos\pi x + cos \pi y) \]+e+20$
    with global minimum $f(0,0)=0$
    :param x: input list [x, y]
    :return: value of Ackley function $f(x,y)$"
    """
    return -20 * np.exp(-0.2 * np.sqrt(x[0] ** 2 + x[1] ** 2)) \
           - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20


def goldstein_price_function(x):
    """
    Goldstein-Price function given as $f(x,y)=[1 + (x+y+1)^2*(19-14x+3x^3-14y+6xy+3y^2)][30+(2x-3y)^2*(18-32x+12x^2+48y
    -36xy+27y^2)]$
    with global minimum $f(0,-1)=3$
    :param x: input list [x, y]
    :return: value of Goldstein-Price function $f(x,y)$
    """
    return (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] +
                                          3 * x[1] ** 2)) * (30 + (2 * x[0] - 3 * x[1]) ** 2 *
                                                             (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1]
                                                              - 36 * x[0] * x[1] + 27 * x[1] ** 2))


def levi_function(x):
    """
    Levi function n.13 defined as $f(x,y)=sin^2(3\pi x)+(x-1)^2(1+sin^2(3\pi y))+(y-1)^2(1+sin^2 2\pi y)$
    with global minimum $f(1,1)=0$
    :param x: input list [x, y]
    :return: value of Levi function n.13 $f(x,y)$
    """
    return np.sin(3 * np.pi * x[0]) ** 2 + (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2)\
                                         + (1 - x[1]) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)

