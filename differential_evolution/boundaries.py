"""Boundary handling strategies."""

from __future__ import annotations

from typing import Protocol


Bounds = list[tuple[float, float]]


class BoundaryHandler(Protocol):
    """Apply a boundary policy to a trial vector."""

    def __call__(
        self,
        trial_vector: list[float],
        bounds: Bounds,
        rng: object,
    ) -> list[float]:
        """Return a bounded trial vector."""


class NoBoundaryHandler:
    """Leave the trial vector unchanged."""

    def __call__(
        self,
        trial_vector: list[float],
        bounds: Bounds,
        rng: object,
    ) -> list[float]:
        return list(trial_vector)


class ClipBoundaryHandler:
    """Clip each coordinate to its interval bounds."""

    def __call__(
        self,
        trial_vector: list[float],
        bounds: Bounds,
        rng: object,
    ) -> list[float]:
        return [
            min(max(value, lower), upper)
            for value, (lower, upper) in zip(trial_vector, bounds)
        ]


class RandomResetBoundaryHandler:
    """Reset violating coordinates uniformly within their bounds."""

    def __call__(
        self,
        trial_vector: list[float],
        bounds: Bounds,
        rng: object,
    ) -> list[float]:
        bounded = []
        for value, (lower, upper) in zip(trial_vector, bounds):
            if lower <= value <= upper:
                bounded.append(value)
            else:
                bounded.append(rng.uniform(lower, upper))
        return bounded
