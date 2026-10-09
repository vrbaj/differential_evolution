"""Overflow-safe coordinate arithmetic for finite search bounds."""

import math
from collections.abc import Sequence
from fractions import Fraction

from .protocols import RandomSource


def require_integer(value: object, name: str) -> None:
    """Reject fractional counts and booleans before they reach loops or budgets."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def mean(values: Sequence[float], *, total: float | None = None) -> float:
    """Preserve ordinary arithmetic, avoiding overflow of a finite-value sum."""
    if total is None:
        total = sum(values)
    if math.isfinite(total) or not all(math.isfinite(value) for value in values):
        return total / len(values)
    return float(sum(Fraction(value) for value in values) / len(values))


def sum_differences(base: float, *terms: tuple[float, float, float]) -> float:
    """Compute base + sum(scale * (left - right)), with an overflow fallback.

    Ordinary coordinates retain their floating-point operation order. Exact
    arithmetic recovers finite results lost to intermediate overflow. A result
    outside float range becomes signed infinity for boundary repair.
    """
    value = base
    for scale, left, right in terms:
        value += scale * (left - right)
    if math.isfinite(value):
        return value
    if not math.isfinite(base) or not all(
        math.isfinite(number) for term in terms for number in term
    ):
        return value
    exact = Fraction(base) + sum(
        Fraction(scale) * (Fraction(left) - Fraction(right))
        for scale, left, right in terms
    )
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


def interpolate(lower: float, upper: float, fraction: float) -> float:
    width = upper - lower
    if math.isfinite(width):
        value = lower + fraction * width
    else:
        value = (1.0 - fraction) * lower + fraction * upper
    return min(max(lower, upper), max(min(lower, upper), value))


def uniform(lower: float, upper: float, rng: RandomSource) -> float:
    if math.isfinite(upper - lower):
        return rng.uniform(lower, upper)
    return interpolate(lower, upper, rng.random())


def midpoint(left: float, right: float) -> float:
    total = left + right
    return total / 2 if math.isfinite(total) else left / 2 + right / 2


def opposite(value: float, lower: float, upper: float) -> float:
    total = lower + upper
    return total - value if math.isfinite(total) else lower + (upper - value)
