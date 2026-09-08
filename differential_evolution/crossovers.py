"""Crossover operators for Differential Evolution."""

from __future__ import annotations

from dataclasses import fields
from dataclasses import dataclass
from typing import Protocol


class CrossoverOperator(Protocol):
    """Combine a target vector and donor vector into a trial vector."""

    def __call__(
        self,
        target_vector: list[float],
        donor_vector: list[float],
        rng: object,
    ) -> list[float]:
        """Return a trial vector."""


@dataclass(frozen=True)
class IdentityCrossover:
    """Return the donor vector unchanged.

    Useful for DE variants whose conventional definition does not include a
    separate crossover step after mutation.
    """

    def __call__(
        self,
        target_vector: list[float],
        donor_vector: list[float],
        rng: object,
    ) -> list[float]:
        return list(donor_vector)

    def initialize(self, population_size: int, rng: object) -> None:
        return None

    def commit(self, target_index: int, accepted: bool) -> None:
        return None


class BaseCrossover:
    """Shared helper for crossover operators."""

    def initialize(self, population_size: int, rng: object) -> None:
        self._active_target_index = 0
        for controller in self._rate_controllers():
            controller.initialize(population_size, rng)

    def commit(self, target_index: int, accepted: bool) -> None:
        for controller in self._rate_controllers():
            controller.commit(target_index, accepted)

    def resize(self, kept_indices: list[int]) -> None:
        for controller in self._rate_controllers():
            if hasattr(controller, "resize"):
                controller.resize(kept_indices)

    def set_target_index(self, target_index: int) -> None:
        self._active_target_index = target_index

    def _resolve_rate(self, value: float | object, rng: object) -> float:
        if hasattr(value, "propose"):
            return float(value.propose(self._active_target_index, rng))
        return float(value)

    def _rate_controllers(self) -> list[object]:
        seen: set[int] = set()
        controllers = []
        for field_definition in fields(self):
            value = getattr(self, field_definition.name)
            if hasattr(value, "initialize") and hasattr(value, "propose") and hasattr(value, "commit"):
                if id(value) not in seen:
                    seen.add(id(value))
                    controllers.append(value)
        return controllers


@dataclass
class BinomialCrossover(BaseCrossover):
    """Binomial DE crossover with the usual forced donor index.

    One uniformly sampled coordinate is always inherited from the donor so that
    at least one donor component survives even when ``crossover_rate`` is
    zero.
    """

    crossover_rate: float | object

    def __post_init__(self) -> None:
        if not hasattr(self.crossover_rate, "propose") and not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")

    def __call__(
        self,
        target_vector: list[float],
        donor_vector: list[float],
        rng: object,
    ) -> list[float]:
        if len(target_vector) != len(donor_vector):
            raise ValueError("target_vector and donor_vector must have the same length.")
        if not target_vector:
            raise ValueError("Vectors must be non-empty.")

        rate = self._resolve_rate(self.crossover_rate, rng)
        forced_index = rng.randrange(len(target_vector))
        trial = []
        for index, (target_value, donor_value) in enumerate(zip(target_vector, donor_vector)):
            if index == forced_index or rng.random() < rate:
                trial.append(donor_value)
            else:
                trial.append(target_value)
        return trial


@dataclass
class ExponentialCrossover(BaseCrossover):
    """Exponential DE crossover with cyclic wrap-around.

    A contiguous donor segment of length at least one is copied, extending
    while fresh uniform draws remain below ``crossover_rate`` and wrapping
    cyclically at the dimension boundary.
    """

    crossover_rate: float | object

    def __post_init__(self) -> None:
        if not hasattr(self.crossover_rate, "propose") and not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")

    def __call__(
        self,
        target_vector: list[float],
        donor_vector: list[float],
        rng: object,
    ) -> list[float]:
        if len(target_vector) != len(donor_vector):
            raise ValueError("target_vector and donor_vector must have the same length.")
        if not target_vector:
            raise ValueError("Vectors must be non-empty.")

        rate = self._resolve_rate(self.crossover_rate, rng)
        dimension = len(target_vector)
        start = rng.randrange(dimension)
        length = 1
        while length < dimension and rng.random() < rate:
            length += 1

        trial = list(target_vector)
        for offset in range(length):
            index = (start + offset) % dimension
            trial[index] = donor_vector[index]
        return trial
