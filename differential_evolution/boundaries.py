"""Boundary handling strategies."""

from __future__ import annotations

from .protocols import RandomSource

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
                bounded.append(rng.uniform(lower, upper))
        return bounded


class MidpointBoundaryHandler:
    """Repair violations halfway between the target coordinate and the bound."""

    def repair(self, trial_vector, target_vector, bounds, rng):
        return [
            (lower + parent) / 2
            if value < lower
            else (upper + parent) / 2
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
            width = upper - lower
            if width == 0:
                result.append(lower)
            else:
                offset = (value - lower) % (2 * width)
                result.append(lower + min(offset, 2 * width - offset))
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
