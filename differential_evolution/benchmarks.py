"""Benchmark objective functions."""

from __future__ import annotations

import math
import pickle
from pathlib import Path


def sphere_function(x: list[float]) -> float:
    """Sphere benchmark function."""
    return sum(value * value for value in x)


def rastrigin_function(x: list[float]) -> float:
    a_value = 10.0
    return a_value * len(x) + sum(
        value * value - a_value * math.cos(2.0 * math.pi * value) for value in x
    )


def beale_function(x: list[float]) -> float:
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


def booth_function(x: list[float]) -> float:
    return (x[0] + 2.0 * x[1] - 7.0) ** 2 + (2.0 * x[0] + x[1] - 5.0) ** 2


def matyas_function(x: list[float]) -> float:
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


def himmelblau_function(x: list[float]) -> float:
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2


def bukin_function(x: list[float]) -> float:
    return 100.0 * math.sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10.0)


def mccormick_function(x: list[float]) -> float:
    return math.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1.0


def three_hump_camel_function(x: list[float]) -> float:
    return 2.0 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6.0 + x[0] * x[1] + x[1] ** 2


def ackley_function(x: list[float]) -> float:
    """Two-dimensional Ackley benchmark with global minimum at ``[0, 0]``."""
    first_term = -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
    second_term = -math.exp(0.5 * (math.cos(2.0 * math.pi * x[0]) + math.cos(2.0 * math.pi * x[1])))
    return first_term + second_term + math.e + 20.0


def goldstein_price_function(x: list[float]) -> float:
    first = 1.0 + (x[0] + x[1] + 1.0) ** 2 * (
        19.0
        - 14.0 * x[0]
        + 3.0 * x[0] ** 2
        - 14.0 * x[1]
        + 6.0 * x[0] * x[1]
        + 3.0 * x[1] ** 2
    )
    second = 30.0 + (2.0 * x[0] - 3.0 * x[1]) ** 2 * (
        18.0
        - 32.0 * x[0]
        + 12.0 * x[0] ** 2
        + 48.0 * x[1]
        - 36.0 * x[0] * x[1]
        + 27.0 * x[1] ** 2
    )
    return first * second


def levi_function(x: list[float]) -> float:
    return (
        math.sin(3.0 * math.pi * x[0]) ** 2
        + (x[0] - 1.0) ** 2 * (1.0 + math.sin(3.0 * math.pi * x[1]) ** 2)
        + (x[1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * x[1]) ** 2)
    )


def easom_function(x: list[float]) -> float:
    return -math.cos(x[0]) * math.cos(x[1]) * math.exp(-((x[0] - math.pi) ** 2 + (x[1] - math.pi) ** 2))


def eggholder_function(x: list[float]) -> float:
    return -(
        (x[1] + 47.0) * math.sin(math.sqrt(abs(x[0] / 2.0 + x[1] + 47.0)))
        + x[0] * math.sin(math.sqrt(abs(x[0] - x[1] - 47.0)))
    )


def schaffer_n2_function(x: list[float]) -> float:
    """Two-dimensional Schaffer N.2 benchmark with global minimum at ``[0, 0]``."""
    numerator = math.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5
    denominator = (1.0 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    return 0.5 + numerator / denominator


def gpd_ll_function(x: list[float], sample_path: str | Path = "gpd_sample") -> float:
    scale = x[0]
    shape = x[1]
    theta = shape / scale
    with Path(sample_path).open("rb") as handle:
        data = pickle.load(handle)
    log_sum = 0.0
    for item in data:
        log_sum += math.log(theta / shape) - (1.0 + 1.0 / shape) * math.log(1.0 + theta * item)
    return -log_sum
