"""Main Differential Evolution optimizer."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from .diversity import resolve_diversity_measures
from .boundaries import Bounds, NoBoundaryHandler
from .crossovers import IdentityCrossover
from .history import GenerationSnapshot
from .initializers import RandomInitializer
from .mutation import MutationContext
from .population_schedules import PopulationScheduleState
from .result import OptimizeResult


ObjectiveFunction = Callable[[list[float]], float]


@dataclass
class DifferentialEvolution:
    """Composable Differential Evolution optimizer.

    Parameters
    ----------
    objective
        Objective function to minimize. It receives one candidate vector and
        must return one scalar fitness value. Non-finite return values are
        treated as ``math.inf``.
    bounds
        Coordinate-wise search bounds as ``(lower, upper)`` pairs.
    population_size
        Number of individuals in the population. This is only a global lower
        bound; specific mutation operators may require larger populations.
    mutation
        Callable receiving :class:`~differential_evolution.MutationContext`
        and returning one donor vector of length ``D``.
    crossover
        Callable combining the target and donor vectors into one trial vector.
    max_generations
        Maximum number of completed generations.
    initializer
        Population initializer receiving ``(population_size, bounds, objective,
        rng)``.
    boundary_handler
        Trial-vector repair policy applied after crossover.
    diversity_measures
        ``None``, one diversity specification, or a list of specifications.
    population_schedule
        Optional population-size schedule applied after each generation.
    max_evaluations
        Optional objective-evaluation budget.
    record_snapshots
        Whether to record per-generation snapshots in the result object.
    seed
        Seed for an internal ``random.Random`` instance.
    rng
        Explicit random-generator-like object. Pass either ``seed`` or
        ``rng``, not both.

    Notes
    -----
    Selection is strict one-to-one minimization: a trial replaces its target
    only if its objective value is strictly smaller.
    """

    objective: ObjectiveFunction
    bounds: list[tuple[float, float]]
    population_size: int
    mutation: object
    crossover: object = field(default_factory=IdentityCrossover)
    max_generations: int = 100
    initializer: object = field(default_factory=RandomInitializer)
    boundary_handler: object = field(default_factory=NoBoundaryHandler)
    diversity_measures: object = None
    population_schedule: object | None = None
    max_evaluations: int | None = None
    record_snapshots: bool = False
    seed: int | None = None
    rng: object | None = None

    def __post_init__(self) -> None:
        if self.population_size < 3:
            raise ValueError("population_size must be at least 3.")
        if self.max_generations < 0:
            raise ValueError("max_generations must be non-negative.")
        if not self.bounds:
            raise ValueError("bounds must be non-empty.")
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive when provided.")
        for lower, upper in self.bounds:
            if lower > upper:
                raise ValueError("Each bound must satisfy lower <= upper.")
        if self.seed is not None and self.rng is not None:
            raise ValueError("Pass either seed or rng, not both.")

        self._rng = self.rng if self.rng is not None else random.Random(self.seed)
        if hasattr(self.mutation, "initialize"):
            self.mutation.initialize(self.population_size, self._rng)
        if hasattr(self.crossover, "initialize"):
            self.crossover.initialize(self.population_size, self._rng)
        self._diversity_measures = resolve_diversity_measures(self.diversity_measures)
        self._initial_population_size = self.population_size
        self.population: list[list[float]] = []
        self.fitness: list[float] = []
        self.best_history: list[list[float]] = []
        self.best_fitness_history: list[float] = []
        self.diversity_history: dict[str, list[float]] = {
            measure.name: [] for measure in self._diversity_measures
        }
        self.population_size_history: list[int] = []
        self.snapshots: list[GenerationSnapshot] = []
        self.nfev = 0
        self.nit = 0

    def initialize(self) -> None:
        """Create and evaluate the initial population.

        Some initializers rank candidate points by objective value before the
        final population is chosen. In those cases, initialization consumes
        extra objective evaluations before the retained population is evaluated
        again by the optimizer.
        """

        self.population = self.initializer(
            self.population_size,
            self.bounds,
            self._evaluate_vector,
            self._rng,
        )
        self._validate_population(self.population)
        self.fitness = [self._evaluate_vector(individual) for individual in self.population]
        self._record_best()
        self._record_diversity(previous_population=None)
        self.population_size_history.append(self.population_size)
        self._record_snapshot()

    def run(self) -> OptimizeResult:
        """Run the optimizer and return an optimization result.

        Returns
        -------
        OptimizeResult
            Final state of the optimization run, including histories and
            optional snapshots.
        """

        if not self.population:
            self.initialize()

        while self.nit < self.max_generations:
            if self.max_evaluations is not None and self.nfev >= self.max_evaluations:
                break
            processed = self._step()
            if processed == 0:
                break

        best_index = self._best_index(self.fitness)
        return OptimizeResult(
            x=list(self.population[best_index]),
            fun=self.fitness[best_index],
            nit=self.nit,
            nfev=self.nfev,
            success=True,
            message="Optimization finished after reaching max_generations.",
            population=[list(individual) for individual in self.population],
            fitness=list(self.fitness),
            best_history=[list(vector) for vector in self.best_history],
            best_fitness_history=list(self.best_fitness_history),
            diversity_history={name: list(values) for name, values in self.diversity_history.items()},
            population_size_history=list(self.population_size_history),
            snapshots=list(self.snapshots),
            metadata=self._result_metadata(),
        )

    def _step(self) -> int:
        population_snapshot = [list(individual) for individual in self.population]
        fitness_snapshot = list(self.fitness)
        best_index = self._best_index(fitness_snapshot)
        best_vector = list(population_snapshot[best_index])

        next_population = [list(individual) for individual in population_snapshot]
        next_fitness = list(fitness_snapshot)
        processed_targets = 0

        for target_index, target_vector in enumerate(population_snapshot):
            if self.max_evaluations is not None and self.nfev >= self.max_evaluations:
                break
            context = MutationContext(
                population=population_snapshot,
                fitness=fitness_snapshot,
                target_index=target_index,
                best_index=best_index,
                best_vector=best_vector,
                bounds=self.bounds,
                rng=self._rng,
            )
            if hasattr(self.crossover, "set_target_index"):
                self.crossover.set_target_index(target_index)
            donor_vector = self.mutation(context)
            trial_vector = self.crossover(target_vector, donor_vector, self._rng)
            bounded_trial = self.boundary_handler(trial_vector, self.bounds, self._rng)
            trial_fitness = self._evaluate_vector(bounded_trial)
            accepted = trial_fitness < fitness_snapshot[target_index]
            if accepted:
                next_population[target_index] = bounded_trial
                next_fitness[target_index] = trial_fitness
            if hasattr(self.mutation, "commit"):
                self.mutation.commit(target_index, accepted)
            if hasattr(self.crossover, "commit"):
                self.crossover.commit(target_index, accepted)
            processed_targets += 1

        if processed_targets == 0:
            return 0

        self.population = next_population
        self.fitness = next_fitness
        self.population_size = len(self.population)
        self.nit += 1
        kept_indices = self._apply_population_schedule()
        self._record_best()
        previous_population = population_snapshot
        if kept_indices is not None:
            previous_population = [population_snapshot[index] for index in kept_indices]
        self._record_diversity(previous_population=previous_population)
        self.population_size_history.append(self.population_size)
        self._record_snapshot()
        return processed_targets

    def _record_best(self) -> None:
        best_index = self._best_index(self.fitness)
        self.best_history.append(list(self.population[best_index]))
        self.best_fitness_history.append(self.fitness[best_index])

    def _record_diversity(self, previous_population: list[list[float]] | None) -> None:
        for measure in self._diversity_measures:
            self.diversity_history[measure.name].append(
                measure(
                    population=self.population,
                    bounds=self.bounds,
                    previous_population=previous_population,
                )
            )

    def _record_snapshot(self) -> None:
        if not self.record_snapshots:
            return None
        best_index = self._best_index(self.fitness)
        self.snapshots.append(
            GenerationSnapshot(
                generation=self.nit,
                evaluations=self.nfev,
                population_size=self.population_size,
                best_vector=list(self.population[best_index]),
                best_fitness=self.fitness[best_index],
                population=[list(individual) for individual in self.population],
                fitness=list(self.fitness),
                diversity={
                    name: values[-1]
                    for name, values in self.diversity_history.items()
                    if values
                },
                extra=self._snapshot_extra(),
            )
        )
        return None

    def _snapshot_extra(self) -> dict[str, object]:
        return {}

    def _result_metadata(self) -> dict[str, object]:
        return {"algorithm": self.__class__.__name__}

    def _apply_population_schedule(self) -> list[int] | None:
        if self.population_schedule is None:
            return None
        target_size = self.population_schedule.target_size(
            PopulationScheduleState(
                generation=self.nit,
                max_generations=self.max_generations,
                evaluations=self.nfev,
                max_evaluations=self.max_evaluations,
                initial_population_size=self._initial_population_size,
                current_population_size=self.population_size,
                min_population_size=self.population_schedule.min_population_size,
            )
        )
        if target_size >= self.population_size:
            return None
        kept_indices = sorted(
            range(self.population_size),
            key=lambda index: self.fitness[index],
        )[:target_size]
        self.population = [self.population[index] for index in kept_indices]
        self.fitness = [self.fitness[index] for index in kept_indices]
        self.population_size = target_size
        if hasattr(self.mutation, "resize"):
            self.mutation.resize(kept_indices)
        if hasattr(self.crossover, "resize"):
            self.crossover.resize(kept_indices)
        return kept_indices

    def _evaluate_vector(self, vector: list[float]) -> float:
        if len(vector) != len(self.bounds):
            raise ValueError("Objective input dimension does not match bounds.")
        value = float(self.objective(list(vector)))
        self.nfev += 1
        if math.isnan(value) or math.isinf(value):
            return math.inf
        return value

    @staticmethod
    def _best_index(fitness: list[float]) -> int:
        return min(range(len(fitness)), key=lambda index: fitness[index])

    def _validate_population(self, population: list[list[float]]) -> None:
        if len(population) != self.population_size:
            raise ValueError("Initializer returned the wrong population size.")
        for individual in population:
            if len(individual) != len(self.bounds):
                raise ValueError("Initializer returned an individual with the wrong dimension.")
