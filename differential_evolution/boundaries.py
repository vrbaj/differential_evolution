"""Boundary handling strategies."""

from __future__ import annotations

from .protocols import RandomSource
from ._numeric import midpoint, uniform

import math
from fractions import Fraction

from typing import Protocol


Bounds = list[tuple[float, float]]


class BoundaryHandler(Protocol):
    """Apply a boundary policy to a trial vector."""

    def __call__(
        self,
        trial_vector: list[float],
        bounds: Bounds,
        rng: RandomSource,
    ) -> list[float]:
        """Return a bounded trial vector."""


class NoBoundaryHandler:
    """Leave the trial vector unchanged."""

    def __call__(
        self,
        trial_vector: list[float],
        bounds: Bounds,
        rng: RandomSource,
    ) -> list[float]:
        return list(trial_vector)


class ClipBoundaryHandler:
    """Clip each coordinate to its interval bounds."""

    def __call__(
        self,
        trial_vector: list[float],
        bounds: Bounds,
        rng: RandomSource,
    ) -> list[float]:
        return [
            min(max(value, lower), upper)
            for value, (lower, upper) in zip(trial_vector, bounds, strict=True)
        ]


class RandomResetBoundaryHandler:
    """Reset violating coordinates uniformly within their bounds."""

    def __call__(
        self,
        trial_vector: list[float],
        bounds: Bounds,
        rng: RandomSource,
    ) -> list[float]:
        bounded = []
        for value, (lower, upper) in zip(trial_vector, bounds, strict=True):
            if lower <= value <= upper:
                bounded.append(value)
            else:
                bounded.append(uniform(lower, upper, rng))
        return bounded


class MidpointBoundaryHandler:
    """Repair violations halfway between the target coordinate and the bound."""

    def repair(self, trial_vector, target_vector, bounds, rng):
        return [
            midpoint(lower, parent)
            if value < lower
            else midpoint(upper, parent)
            if value > upper
            else value
            for value, parent, (lower, upper) in zip(
                trial_vector, target_vector, bounds, strict=True
            )
        ]

    def __call__(self, trial_vector, bounds, rng):
        raise ValueError(
            "Midpoint repair requires a target; call repair(trial, target, bounds, rng)."
        )


class ReflectionBoundaryHandler:
    """Reflect coordinates repeatedly into their intervals."""

    def __call__(self, trial_vector, bounds, rng):
        result = []
        for value, (lower, upper) in zip(trial_vector, bounds, strict=True):
            if not math.isfinite(value):
                raise ValueError("Reflection requires finite trial coordinates.")
            if lower <= value <= upper:
                result.append(value)
                continue
            width = upper - lower
            if width == 0:
                result.append(lower)
            elif not all(math.isfinite(number) for number in
                         (width, 2 * width, value - lower)):
                # Exact arithmetic only for intervals exceeding float range.
                exact_lower = Fraction(lower)
                exact_width = Fraction(upper) - exact_lower
                period = 2 * exact_width
                offset = (Fraction(value) - exact_lower) % period
                result.append(float(exact_lower + min(offset, period - offset)))
            else:
                offset = (value - lower) % (2 * width)
                reflected = lower + min(offset, 2 * width - offset)
                result.append(min(upper, max(lower, reflected)))
        return result


class ParentAwareBoundaryHandler(Protocol):
    """Optional repair hook; optimizers prefer it over the legacy call signature."""

    def repair(
        self,
        trial_vector: list[float],
        target_vector: list[float],
        bounds: Bounds,
        rng: RandomSource,
    ) -> list[float]: ...
