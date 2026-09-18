"""Benchmark objective functions."""

from __future__ import annotations

import math
from functools import lru_cache
from collections.abc import Iterable
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
    if len(x) != 2:
        raise ValueError("beale_function requires exactly 2 dimensions.")
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


def booth_function(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("booth_function requires exactly 2 dimensions.")
    return (x[0] + 2.0 * x[1] - 7.0) ** 2 + (2.0 * x[0] + x[1] - 5.0) ** 2


def matyas_function(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("matyas_function requires exactly 2 dimensions.")
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


def himmelblau_function(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("himmelblau_function requires exactly 2 dimensions.")
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2


def bukin_function(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("bukin_function requires exactly 2 dimensions.")
    return 100.0 * math.sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10.0)


def mccormick_function(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("mccormick_function requires exactly 2 dimensions.")
    return math.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1.0


def three_hump_camel_function(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("three_hump_camel_function requires exactly 2 dimensions.")
    return (
        2.0 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6.0 + x[0] * x[1] + x[1] ** 2
    )


def ackley_function(x: list[float]) -> float:
    """Two-dimensional Ackley benchmark with global minimum at ``[0, 0]``."""
    if len(x) != 2:
        raise ValueError("ackley_function requires exactly 2 dimensions.")
    first_term = -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
    second_term = -math.exp(
        0.5 * (math.cos(2.0 * math.pi * x[0]) + math.cos(2.0 * math.pi * x[1]))
    )
    return first_term + second_term + math.e + 20.0


def goldstein_price_function(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("goldstein_price_function requires exactly 2 dimensions.")
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
    if len(x) != 2:
        raise ValueError("levi_function requires exactly 2 dimensions.")
    return (
        math.sin(3.0 * math.pi * x[0]) ** 2
        + (x[0] - 1.0) ** 2 * (1.0 + math.sin(3.0 * math.pi * x[1]) ** 2)
        + (x[1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * x[1]) ** 2)
    )


def easom_function(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("easom_function requires exactly 2 dimensions.")
    return (
        -math.cos(x[0])
        * math.cos(x[1])
        * math.exp(-((x[0] - math.pi) ** 2 + (x[1] - math.pi) ** 2))
    )


def eggholder_function(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("eggholder_function requires exactly 2 dimensions.")
    return -(
        (x[1] + 47.0) * math.sin(math.sqrt(abs(x[0] / 2.0 + x[1] + 47.0)))
        + x[0] * math.sin(math.sqrt(abs(x[0] - x[1] - 47.0)))
    )


def schaffer_n2_function(x: list[float]) -> float:
    """Two-dimensional Schaffer N.2 benchmark with global minimum at ``[0, 0]``."""
    if len(x) != 2:
        raise ValueError("schaffer_n2_function requires exactly 2 dimensions.")
    numerator = math.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5
    denominator = (1.0 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    return 0.5 + numerator / denominator


def gpd_negative_log_likelihood(data: Iterable[float]):
    """Build a generalized Pareto objective, loading observations once.

    Observations must be finite and non-negative. The fitted parameters are
    [scale, shape]; invalid parameter support returns positive infinity.
    """
    observations = tuple(float(value) for value in data)
    if not observations or any(
        not math.isfinite(value) or value < 0 for value in observations
    ):
        raise ValueError("GPD observations must be non-empty, finite and non-negative.")

    def objective(x):
        if len(x) != 2:
            raise ValueError("GPD requires scale and shape.")
        scale, shape = x
        if not math.isfinite(scale) or not math.isfinite(shape) or scale <= 0:
            return math.inf
        if shape == 0:
            return len(observations) * math.log(scale) + sum(observations) / scale
        terms = [shape * value / scale for value in observations]
        if any(value <= -1 for value in terms):
            return math.inf
        return len(observations) * math.log(scale) + (1 + 1 / shape) * sum(
            math.log1p(value) for value in terms
        )

    return objective


@lru_cache(maxsize=16)
def _gpd_from_text(sample_path):
    return gpd_negative_log_likelihood(
        float(token)
        for token in Path(sample_path).read_text().replace(",", " ").split()
    )


def gpd_ll_function(
    x: list[float], sample_path: str | Path = "gpd_sample.csv"
) -> float:
    """Compatibility wrapper for a cached comma/whitespace-delimited text sample.

    Pickle samples are no longer supported. Prefer gpd_negative_log_likelihood
    with observations loaded explicitly by the caller.
    """
    return _gpd_from_text(str(Path(sample_path).resolve()))(x)
