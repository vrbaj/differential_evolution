"""Crossover-rate controllers for Differential Evolution."""

from __future__ import annotations

from .protocols import RandomSource

from dataclasses import dataclass, field
from typing import Protocol


class CrossoverRateController(Protocol):
    """Propose one rate per target and retain it only after accepted selection."""

    def initialize(self, population_size: int, rng: RandomSource) -> None: ...
    def propose(self, target_index: int, rng: RandomSource) -> float: ...
    def commit(self, target_index: int, accepted: bool) -> None: ...
    def resize(self, kept_indices: list[int]) -> None: ...


@dataclass(frozen=True)
class ConstantCrossoverRate:
    """A fixed crossover rate."""

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("ConstantCrossoverRate requires 0 <= value <= 1.")

    def initialize(self, population_size: int, rng: RandomSource) -> None:
        return None

    def propose(self, target_index: int, rng: RandomSource) -> float:
        return self.value

    def commit(self, target_index: int, accepted: bool) -> None:
        return None

    def resize(self, kept_indices: list[int]) -> None:
        return None


@dataclass
class AdaptiveCrossoverRate:
    """jDE-style self-adapting crossover rate.

    Each individual carries its own ``CR_i``. Before crossover for target
    ``i``, a new proposal is drawn with probability ``tau`` from
    ``Uniform(lower, upper)``; otherwise the current value is reused. The
    proposal survives only if the associated trial vector wins selection.
    """

    initial: float = 0.9
    tau: float = 0.1
    lower: float = 0.0
    upper: float = 1.0
    _values: list[float] = field(default_factory=list, init=False, repr=False)
    _pending: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise ValueError("AdaptiveCrossoverRate requires 0 <= lower <= upper <= 1.")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError("initial must lie within [lower, upper].")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be between 0 and 1.")

    def initialize(self, population_size: int, rng: RandomSource) -> None:
        self._values = [self.initial for _ in range(population_size)]
        self._pending.clear()

    def propose(self, target_index: int, rng: RandomSource) -> float:
        if target_index in self._pending:
            return self._pending[target_index]
        if not self._values:
            self.initialize(target_index + 1, rng)
        value = self._values[target_index]
        if rng.random() < self.tau:
            value = rng.uniform(self.lower, self.upper)
        self._pending[target_index] = value
        return value

    def commit(self, target_index: int, accepted: bool) -> None:
        if target_index not in self._pending:
            return None
        if accepted:
            self._values[target_index] = self._pending[target_index]
        del self._pending[target_index]
        return None

    def values(self) -> list[float]:
        """Return a copy of the current per-individual crossover rates."""

        return list(self._values)

    def resize(self, kept_indices: list[int]) -> None:
        self._values = [self._values[index] for index in kept_indices]
        self._pending.clear()
