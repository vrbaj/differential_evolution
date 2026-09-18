"""Population-size reduction schedules for Differential Evolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PopulationScheduleState:
    """Current optimizer state visible to population schedules."""

    generation: int
    max_generations: int
    evaluations: int
    max_evaluations: int | None
    initial_population_size: int
    current_population_size: int
    min_population_size: int


class PopulationSchedule(Protocol):
    """Return the desired population size after a generation."""

    @property
    def min_population_size(self) -> int: ...

    def target_size(self, state: PopulationScheduleState) -> int:
        """Return the desired next population size."""


def _normalized_progress(state: PopulationScheduleState) -> float:
    if state.max_evaluations is not None and state.max_evaluations > 0:
        return min(1.0, state.evaluations / state.max_evaluations)
    if state.max_generations <= 0:
        return 1.0
    return min(1.0, state.generation / state.max_generations)


@dataclass(frozen=True)
class LinearPopulationReduction:
    """L-SHADE-style linear population size reduction.

    When ``max_evaluations`` is available, this follows the L-SHADE idea
    of shrinking linearly with the fraction of used function evaluations.
    Otherwise, it falls back to generation-based progress.
    """

    min_population_size: int

    def __post_init__(self) -> None:
        if self.min_population_size < 1:
            raise ValueError("min_population_size must be positive.")

    def target_size(self, state: PopulationScheduleState) -> int:
        progress = _normalized_progress(state)
        target = round(
            state.initial_population_size
            + (self.min_population_size - state.initial_population_size) * progress
        )
        return max(self.min_population_size, min(state.current_population_size, target))


@dataclass(frozen=True)
class HyperbolicTangentPopulationReduction:
    """Smooth nonlinear population reduction using a normalized tanh curve.

    This is a project-specific schedule inspired by tanh-based nonlinear
    population reduction papers. Population size decreases slowly early
    in the run and more aggressively later.
    """

    min_population_size: int
    start: float = -3.0
    end: float = 0.0

    def __post_init__(self) -> None:
        if self.min_population_size < 1:
            raise ValueError("min_population_size must be positive.")
        if self.end <= self.start:
            raise ValueError("end must be greater than start.")

    def target_size(self, state: PopulationScheduleState) -> int:
        progress = _normalized_progress(state)
        mapped = self.start + (self.end - self.start) * progress
        tanh_start = math.tanh(self.start)
        tanh_end = math.tanh(self.end)
        normalized = (math.tanh(mapped) - tanh_start) / (tanh_end - tanh_start)
        target = round(
            state.initial_population_size
            + (self.min_population_size - state.initial_population_size) * normalized
        )
        return max(self.min_population_size, min(state.current_population_size, target))
