"""Compatibility layer for the original public API."""

from __future__ import annotations

import warnings

from .boundaries import NoBoundaryHandler
from .crossovers import BinomialCrossover
from .diversity import resolve_diversity_measure
from .initializers import (
    OppositionInitializer,
    QuasiOppositionInitializer,
    RandomInitializer,
    SobolInitializer,
    TentInitializer,
)
from .mutation import (
    Best1,
    Best2,
    CurrentToBest1,
    CurrentToBest2,
    CurrentToRand1,
    CurrentToRand2,
    Rand1,
    Rand2,
)
from .optimizer import DifferentialEvolution as ModernDifferentialEvolution


def _resolve_initializer(name: str) -> object:
    mapping = {
        "random": RandomInitializer(),
        "obl": OppositionInitializer(),
        "tent": TentInitializer(),
        "qobl": QuasiOppositionInitializer(),
        "sobol": SobolInitializer(),
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown population initialization algorithm: {name!r}"
        ) from error


def _resolve_mutation(strategy: str, mutation: list[float]) -> object:
    if strategy == "DE/rand/1":
        return Rand1(scale=mutation[0])
    if strategy == "DE/rand/2":
        return Rand2(scale=mutation[0])
    if strategy == "DE/best/1":
        return Best1(scale=mutation[0])
    if strategy == "DE/best/2":
        return Best2(scale=mutation[0])
    if strategy == "DE/current-to-best/1":
        return CurrentToBest1(scale=mutation[0], difference_scale=mutation[1])
    if strategy == "DE/current-to-best/2":
        return CurrentToBest2(scale=mutation[0], difference_scale=mutation[1])
    if strategy == "DE/current-to-rand/1":
        return CurrentToRand1(scale=mutation[0], difference_scale=mutation[1])
    if strategy == "DE/current-to-rand/2":
        return CurrentToRand2(scale=mutation[0], difference_scale=mutation[1])
    raise ValueError(f"Unknown strategy: {strategy!r}")


class DifferentialEvolution:
    """Backward-compatible wrapper around the refactored optimizer."""

    def __init__(
        self,
        cost_function,
        bounds,
        max_iterations,
        population_size,
        mutation,
        crossover,
        strategy,
        population_initialization_algorithm,
        seed=None,
    ):
        warnings.warn(
            "The legacy DifferentialEvolution constructor from main.py is deprecated. "
            "Use differential_evolution.DifferentialEvolution with explicit components instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized_bounds = [tuple(bound) for bound in bounds]
        self._optimizer = ModernDifferentialEvolution(
            objective=cost_function,
            bounds=normalized_bounds,
            population_size=population_size,
            mutation=_resolve_mutation(strategy, mutation),
            crossover=BinomialCrossover(crossover_rate=crossover),
            max_generations=max_iterations,
            initializer=_resolve_initializer(population_initialization_algorithm),
            boundary_handler=NoBoundaryHandler(),
            seed=seed,
        )
        self.cost_function = cost_function
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation = mutation
        self.crossover = crossover
        self.bounds = bounds
        self.strategy = strategy
        self.generation = 0
        self.generation_best_individual = None
        self.generation_best_individual_idx = None
        self.best_individual_history = []
        self.generation_fitness = []
        self.initial_population = population_initialization_algorithm
        self.population = []

    def initialize(self):
        self._optimizer.initialize()
        self.population = [
            list(individual) for individual in self._optimizer.population
        ]
        self.generation_fitness = list(self._optimizer.fitness)
        self._sync_best()

    def evolve(self):
        if not self._optimizer.population:
            self.initialize()
        result = self._optimizer.run()
        self.population = [list(individual) for individual in result.population]
        self.generation = result.nit
        self.generation_fitness = list(result.fitness)
        self.best_individual_history = [list(vector) for vector in result.best_history]
        self._sync_best()

    def get_best(self):
        if not self._optimizer.population:
            raise ValueError("Population has not been initialized.")
        self._sync_best()
        return list(self.generation_best_individual)

    def filter_history(self, dimension):
        return [solution[dimension] for solution in self.best_individual_history]

    def measure_diversity(self, measure):
        if measure != "std-fitness":
            diversity_measure = resolve_diversity_measure(measure)
            previous_population = None
            if (
                diversity_measure.name == "population_coherence"
                and diversity_measure.name in self._optimizer.diversity_history
            ):
                return self._optimizer.diversity_history[diversity_measure.name][-1]
            return diversity_measure(
                self.population,
                [tuple(bound) for bound in self.bounds],
                previous_population,
            )
        if not self.generation_fitness:
            raise ValueError("No generation fitness values are available.")
        mean = sum(self.generation_fitness) / len(self.generation_fitness)
        variance = sum((value - mean) ** 2 for value in self.generation_fitness) / len(
            self.generation_fitness
        )
        return variance**0.5

    def _sync_best(self):
        best_index = min(
            range(len(self._optimizer.fitness)),
            key=lambda index: self._optimizer.fitness[index],
        )
        self.generation_best_individual_idx = best_index
        self.generation_best_individual = list(self._optimizer.population[best_index])
