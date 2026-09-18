"""Scale-factor controllers for Differential Evolution mutation."""

from __future__ import annotations

from .protocols import RandomSource

from dataclasses import dataclass, field
from typing import Protocol

from .mutation import MutationContext


class ScaleFactorController(Protocol):
    """Produce scale factors and optionally adapt them across generations."""

    def initialize(self, population_size: int, rng: RandomSource) -> None:
        """Prepare any per-population state."""

    def propose(self, context: MutationContext) -> float:
        """Return the scale factor to use for the current target."""

    def commit(self, target_index: int, accepted: bool) -> None:
        """Update internal state after one-to-one selection."""

    def resize(self, kept_indices: list[int]) -> None:
        """Drop state for removed population members."""


@dataclass(frozen=True)
class ConstantScaleFactor:
    """A fixed differential weight."""

    value: float

    def initialize(self, population_size: int, rng: RandomSource) -> None:
        return None

    def propose(self, context: MutationContext) -> float:
        return self.value

    def commit(self, target_index: int, accepted: bool) -> None:
        return None

    def resize(self, kept_indices: list[int]) -> None:
        return None


@dataclass
class RandomizedScaleFactor:
    """Uniform dither for the scale factor.

    A new ``F`` is drawn independently for each target mutation from
    ``Uniform(lower, upper)``.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (0.0 < self.lower <= self.upper):
            raise ValueError("RandomizedScaleFactor requires 0 < lower <= upper.")

    def initialize(self, population_size: int, rng: RandomSource) -> None:
        return None

    def propose(self, context: MutationContext) -> float:
        return context.rng.uniform(self.lower, self.upper)

    def commit(self, target_index: int, accepted: bool) -> None:
        return None

    def resize(self, kept_indices: list[int]) -> None:
        return None


@dataclass
class AdaptiveScaleFactor:
    """jDE-style self-adapting scale factor.

    Each population member has its own scale factor ``F_i``. Before
    mutating target ``i``, a new proposal is drawn with probability
    ``tau`` from ``Uniform(lower, upper)``; otherwise the current value
    is reused. The proposal survives only if the associated trial vector
    wins selection.
    """

    initial: float = 0.5
    tau: float = 0.1
    lower: float = 0.1
    upper: float = 1.0
    _values: list[float] = field(default_factory=list, init=False, repr=False)
    _pending: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.lower <= self.upper:
            raise ValueError("AdaptiveScaleFactor requires 0 < lower <= upper.")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError("initial must lie within [lower, upper].")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be between 0 and 1.")

    def initialize(self, population_size: int, rng: RandomSource) -> None:
        self._values = [self.initial for _ in range(population_size)]
        self._pending.clear()

    def propose(self, context: MutationContext) -> float:
        target_index = context.target_index
        if target_index in self._pending:
            return self._pending[target_index]

        if not self._values:
            self.initialize(len(context.population), context.rng)
        value = self._values[target_index]
        if context.rng.random() < self.tau:
            value = context.rng.uniform(self.lower, self.upper)
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
        """Return a copy of the current per-individual scale factors."""

        return list(self._values)

    def resize(self, kept_indices: list[int]) -> None:
        self._values = [self._values[index] for index in kept_indices]
        self._pending.clear()
