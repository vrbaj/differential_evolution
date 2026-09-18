"""Main Differential Evolution optimizer."""

from __future__ import annotations

from .protocols import RandomSource

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

from .diversity import (
    DiversityMeasure,
    resolve_diversity_measures,
    _pairwise_distances,
    PopulationDiameter,
    AveragePairwiseDistance,
    AverageDistanceAroundAllIndividuals,
)
from collections.abc import Iterable
from .boundaries import BoundaryHandler, ClipBoundaryHandler
from .crossovers import CrossoverOperator, IdentityCrossover
from .history import GenerationSnapshot
from .initializers import PopulationInitializer, RandomInitializer
from .mutation import MutationContext, MutationOperator
from .population_schedules import PopulationSchedule, PopulationScheduleState
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
        treated as ``math.inf`` except negative infinity, which is preserved.
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
    mutation: MutationOperator
    crossover: CrossoverOperator = field(default_factory=IdentityCrossover)
    max_generations: int = 100
    initializer: PopulationInitializer = field(default_factory=RandomInitializer)
    boundary_handler: BoundaryHandler = field(default_factory=ClipBoundaryHandler)
    diversity_measures: (
        str | DiversityMeasure | Iterable[str | DiversityMeasure] | None
    ) = None
    population_schedule: PopulationSchedule | None = None
    max_evaluations: int | None = None
    record_snapshots: bool = False
    snapshot_interval: int = 1
    objective_errors: str = "raise"
    seed: int | None = None
    rng: RandomSource | None = None

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
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError("Each bound must satisfy lower <= upper.")
        if self.seed is not None and self.rng is not None:
            raise ValueError("Pass either seed or rng, not both.")

        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive.")
        if self.objective_errors not in ("raise", "penalize"):
            raise ValueError("objective_errors must be raise or penalize.")
        required = getattr(self.mutation, "required_population_size", 1)
        if self.population_size < required:
            raise ValueError(
                f"{type(self.mutation).__name__} requires population_size at least {required}."
            )
        if (
            self.population_schedule is not None
            and self.population_schedule.min_population_size < required
        ):
            raise ValueError(
                f"Population schedule must retain at least {required} individuals."
            )
        if (
            self.max_evaluations is not None
            and self.max_evaluations < self.population_size
        ):
            raise ValueError("max_evaluations must cover the initial population.")
        self.mutation, self.crossover = deepcopy((self.mutation, self.crossover))
        self._has_run = False
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
        extra objective evaluations; retained candidate fitness is reused.
        Initialization raises before exceeding max_evaluations if the budget
        cannot cover the initializer's candidate pool.
        """

        if self.population:
            raise RuntimeError(
                "This optimizer is already initialized; create a new instance to restart."
            )
        evaluated = {}

        def evaluate_initial(vector):
            key = tuple(vector)
            evaluated[key] = self._evaluate_vector(vector)
            return evaluated[key]

        self.population = self.initializer(
            self.population_size,
            self.bounds,
            evaluate_initial,
            self._rng,
        )
        self._validate_population(self.population)
        self.fitness = [
            evaluated[tuple(individual)]
            if tuple(individual) in evaluated
            else self._evaluate_vector(individual)
            for individual in self.population
        ]
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

        if self._has_run:
            raise RuntimeError(
                "run() has already completed; create a new optimizer to restart."
            )
        if not self.population:
            self.initialize()

        while self.nit < self.max_generations:
            if self.max_evaluations is not None and self.nfev >= self.max_evaluations:
                break
            processed = self._step()
            if processed == 0:
                break

        self._has_run = True
        best_index = self._best_index(self.fitness)
        reason = (
            "max_evaluations"
            if self.max_evaluations is not None and self.nfev >= self.max_evaluations
            else "max_generations"
        )
        finite_solution = self.fitness[best_index] < math.inf
        return OptimizeResult(
            x=list(self.population[best_index]),
            fun=self.fitness[best_index],
            nit=self.nit,
            nfev=self.nfev,
            success=finite_solution,
            message=f"Optimization finished after reaching {reason}."
            if finite_solution
            else f"No feasible solution found before {reason}.",
            population=[list(individual) for individual in self.population],
            fitness=list(self.fitness),
            best_history=[list(vector) for vector in self.best_history],
            best_fitness_history=list(self.best_fitness_history),
            diversity_history={
                name: list(values) for name, values in self.diversity_history.items()
            },
            population_size_history=list(self.population_size_history),
            snapshots=list(self.snapshots),
            metadata=self._result_metadata(),
        )

    @property
    def current_population_size(self) -> int:
        """Current population size; population_size remains the initial configuration."""
        return len(self.population) if self.population else self.population_size

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
            bounded_trial = self._repair_trial(trial_vector, target_vector)
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
        self.nit += 1
        kept_indices = self._apply_population_schedule()
        self._record_best()
        previous_population = population_snapshot
        if kept_indices is not None:
            previous_population = [population_snapshot[index] for index in kept_indices]
        self._record_diversity(previous_population=previous_population)
        self.population_size_history.append(self.current_population_size)
        self._record_snapshot()
        return processed_targets

    def _repair_trial(self, trial_vector, target_vector):
        if len(trial_vector) != len(self.bounds):
            raise ValueError("Trial dimension does not match bounds.")
        if hasattr(self.boundary_handler, "repair"):
            return self.boundary_handler.repair(
                trial_vector, target_vector, self.bounds, self._rng
            )
        return self.boundary_handler(trial_vector, self.bounds, self._rng)

    def _record_best(self) -> None:
        best_index = self._best_index(self.fitness)
        self.best_history.append(list(self.population[best_index]))
        self.best_fitness_history.append(self.fitness[best_index])

    def _record_diversity(self, previous_population: list[list[float]] | None) -> None:
        shared = {}
        pairwise_types = (
            PopulationDiameter,
            AveragePairwiseDistance,
            AverageDistanceAroundAllIndividuals,
        )
        if any(type(measure) in pairwise_types for measure in self._diversity_measures):
            distances = _pairwise_distances(self.population)
            total = sum(distances)
            count = len(self.population)
            shared = {
                PopulationDiameter: max(distances, default=0.0),
                AveragePairwiseDistance: total / len(distances) if distances else 0.0,
                AverageDistanceAroundAllIndividuals: 2 * total / (count * count),
            }
        for measure in self._diversity_measures:
            value = shared.get(type(measure))
            if value is None:
                value = measure(
                    population=self.population,
                    bounds=self.bounds,
                    previous_population=previous_population,
                )
            self.diversity_history[measure.name].append(value)

    def _record_snapshot(self) -> None:
        if not self.record_snapshots or self.nit % self.snapshot_interval:
            return None
        best_index = self._best_index(self.fitness)
        self.snapshots.append(
            GenerationSnapshot(
                generation=self.nit,
                evaluations=self.nfev,
                population_size=self.current_population_size,
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
        return {
            "algorithm": self.__class__.__name__,
            "mutation_fallbacks": getattr(self.mutation, "fallback_count", 0),
            "termination_reason": "max_evaluations"
            if self.max_evaluations is not None and self.nfev >= self.max_evaluations
            else "max_generations",
        }

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
                current_population_size=self.current_population_size,
                min_population_size=self.population_schedule.min_population_size,
            )
        )
        if target_size >= self.current_population_size:
            return None
        kept_indices = sorted(
            range(self.current_population_size),
            key=lambda index: self.fitness[index],
        )[:target_size]
        self.population = [self.population[index] for index in kept_indices]
        self.fitness = [self.fitness[index] for index in kept_indices]
        if hasattr(self.mutation, "resize"):
            self.mutation.resize(kept_indices)
        if hasattr(self.crossover, "resize"):
            self.crossover.resize(kept_indices)
        return kept_indices

    def _evaluate_vector(self, vector: list[float]) -> float:
        if len(vector) != len(self.bounds):
            raise ValueError("Objective input dimension does not match bounds.")
        if self.max_evaluations is not None and self.nfev >= self.max_evaluations:
            raise ValueError(
                "max_evaluations exhausted during initialization; increase the budget for this initializer."
            )
        self.nfev += 1
        try:
            value = float(self.objective(list(vector)))
        except (ValueError, ArithmeticError):
            if self.objective_errors == "raise":
                raise
            return math.inf
        if math.isnan(value):
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
                raise ValueError(
                    "Initializer returned an individual with the wrong dimension."
                )
