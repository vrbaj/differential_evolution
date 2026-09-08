"""SHADE and L-SHADE optimizers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .optimizer import DifferentialEvolution
from .population_schedules import LinearPopulationReduction


@dataclass
class SHADE(DifferentialEvolution):
    """Success-History based Adaptive Differential Evolution.

    This implementation uses the ``current-to-pbest/1/bin`` structure with an
    external archive and success-history memories for ``F`` and ``CR``.
    """

    mutation: object = field(init=False, default=None, repr=False)
    crossover: object = field(init=False, default=None, repr=False)
    memory_size: int = 6
    p_best_rate: float = 0.2
    archive_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("SHADE requires population_size at least 4.")
        if self.memory_size < 1:
            raise ValueError("memory_size must be positive.")
        if not 0.0 < self.p_best_rate <= 1.0:
            raise ValueError("p_best_rate must be in (0, 1].")
        if self.archive_rate < 0.0:
            raise ValueError("archive_rate must be non-negative.")
        if self.population_schedule is not None and self.population_schedule.min_population_size < 4:
            raise ValueError("SHADE-compatible population schedules require min_population_size at least 4.")
        super().__post_init__()
        self.archive: list[list[float]] = []
        self.memory_f: list[float] = [0.5] * self.memory_size
        self.memory_cr: list[float | None] = [0.5] * self.memory_size
        self.memory_index = 0

    def initialize(self) -> None:
        self.archive = []
        self.memory_f = [0.5] * self.memory_size
        self.memory_cr = [0.5] * self.memory_size
        self.memory_index = 0
        super().initialize()

    def _step(self) -> int:
        population_snapshot = [list(individual) for individual in self.population]
        fitness_snapshot = list(self.fitness)
        next_population = [list(individual) for individual in population_snapshot]
        next_fitness = list(fitness_snapshot)
        successful_f: list[float] = []
        successful_cr: list[float] = []
        improvements: list[float] = []
        processed_targets = 0

        for target_index, target_vector in enumerate(population_snapshot):
            if self.max_evaluations is not None and self.nfev >= self.max_evaluations:
                break

            memory_slot = self._rng.randrange(self.memory_size)
            scale_factor = self._sample_scale_factor(memory_slot)
            crossover_rate = self._sample_crossover_rate(memory_slot)
            p_best_index = self._sample_p_best_index(target_index, fitness_snapshot)
            r1_index = self._sample_r1_index(target_index, p_best_index)
            r2_vector = self._sample_r2_vector(target_index, p_best_index, r1_index)

            donor_vector = [
                target_value
                + scale_factor * (population_snapshot[p_best_index][dimension] - target_value)
                + scale_factor * (population_snapshot[r1_index][dimension] - r2_vector[dimension])
                for dimension, target_value in enumerate(target_vector)
            ]
            trial_vector = self._binomial_crossover(target_vector, donor_vector, crossover_rate)
            bounded_trial = self.boundary_handler(trial_vector, self.bounds, self._rng)
            trial_fitness = self._evaluate_vector(bounded_trial)
            improvement = fitness_snapshot[target_index] - trial_fitness
            accepted = improvement > 0.0

            if accepted:
                next_population[target_index] = bounded_trial
                next_fitness[target_index] = trial_fitness
                successful_f.append(scale_factor)
                successful_cr.append(crossover_rate)
                improvements.append(improvement)
                self.archive.append(list(target_vector))
            processed_targets += 1

        if processed_targets == 0:
            return 0

        self.population = next_population
        self.fitness = next_fitness
        self.population_size = len(self.population)
        self.nit += 1
        self._update_memory(successful_f, successful_cr, improvements)
        kept_indices = self._apply_population_schedule()
        self._resize_archive()
        self._record_best()
        previous_population = population_snapshot
        if kept_indices is not None:
            previous_population = [population_snapshot[index] for index in kept_indices]
        self._record_diversity(previous_population=previous_population)
        self.population_size_history.append(self.population_size)
        self._record_snapshot()
        return processed_targets

    def _sample_scale_factor(self, memory_slot: int) -> float:
        location = self.memory_f[memory_slot]
        scale_factor = -1.0
        while scale_factor <= 0.0:
            scale_factor = location + 0.1 * math.tan(math.pi * (self._rng.random() - 0.5))
        return min(scale_factor, 1.0)

    def _sample_crossover_rate(self, memory_slot: int) -> float:
        location = self.memory_cr[memory_slot]
        if location is None:
            return 0.0
        return min(1.0, max(0.0, self._rng.gauss(location, 0.1)))

    def _sample_p_best_index(self, target_index: int, fitness: list[float]) -> int:
        population_size = len(fitness)
        p_min = 2.0 / population_size
        p_value = self._rng.uniform(p_min, max(p_min, self.p_best_rate))
        top_count = max(2, math.ceil(p_value * population_size))
        sorted_indices = sorted(range(population_size), key=lambda index: fitness[index])
        top_indices = sorted_indices[:top_count]
        filtered = [index for index in top_indices if index != target_index]
        if filtered:
            return self._rng.sample(filtered, 1)[0]
        return self._rng.sample(top_indices, 1)[0]

    def _sample_r1_index(self, target_index: int, p_best_index: int) -> int:
        candidates = [
            index
            for index in range(self.population_size)
            if index != target_index and index != p_best_index
        ]
        return self._rng.sample(candidates, 1)[0]

    def _sample_r2_vector(self, target_index: int, p_best_index: int, r1_index: int) -> list[float]:
        candidates: list[tuple[str, int]] = [
            ("population", index)
            for index in range(self.population_size)
            if index not in {target_index, p_best_index, r1_index}
        ]
        candidates.extend(("archive", index) for index in range(len(self.archive)))
        source, index = self._rng.sample(candidates, 1)[0]
        if source == "population":
            return list(self.population[index])
        return list(self.archive[index])

    def _binomial_crossover(
        self,
        target_vector: list[float],
        donor_vector: list[float],
        crossover_rate: float,
    ) -> list[float]:
        forced_index = self._rng.randrange(len(target_vector))
        return [
            donor_value if index == forced_index or self._rng.random() < crossover_rate else target_value
            for index, (target_value, donor_value) in enumerate(zip(target_vector, donor_vector))
        ]

    def _update_memory(
        self,
        successful_f: list[float],
        successful_cr: list[float],
        improvements: list[float],
    ) -> None:
        if not improvements:
            return None
        total_improvement = sum(improvements)
        weights = [improvement / total_improvement for improvement in improvements]
        weighted_f_numerator = sum(weight * value * value for weight, value in zip(weights, successful_f))
        weighted_f_denominator = sum(weight * value for weight, value in zip(weights, successful_f))
        self.memory_f[self.memory_index] = weighted_f_numerator / weighted_f_denominator

        if self.memory_cr[self.memory_index] is not None:
            if max(successful_cr, default=0.0) == 0.0:
                self.memory_cr[self.memory_index] = None
            else:
                self.memory_cr[self.memory_index] = sum(
                    weight * value for weight, value in zip(weights, successful_cr)
                )
        self.memory_index = (self.memory_index + 1) % self.memory_size
        return None

    def _resize_archive(self) -> None:
        archive_limit = round(self.archive_rate * self.population_size)
        if archive_limit <= 0:
            self.archive = []
            return None
        while len(self.archive) > archive_limit:
            remove_index = self._rng.randrange(len(self.archive))
            del self.archive[remove_index]
        return None

    def _snapshot_extra(self) -> dict[str, object]:
        return {
            "archive": [list(vector) for vector in self.archive],
            "memory_f": list(self.memory_f),
            "memory_cr": list(self.memory_cr),
            "memory_index": self.memory_index,
        }

    def _result_metadata(self) -> dict[str, object]:
        return {
            "algorithm": self.__class__.__name__,
            "memory_size": self.memory_size,
            "p_best_rate": self.p_best_rate,
            "archive_rate": self.archive_rate,
        }


@dataclass
class LSHADE(SHADE):
    """L-SHADE: SHADE with linear population size reduction.

    If no explicit population schedule is supplied, the class installs
    :class:`~differential_evolution.LinearPopulationReduction`.
    """

    min_population_size: int = 4
    archive_rate: float = 2.6

    def __post_init__(self) -> None:
        if self.min_population_size < 4:
            raise ValueError("L-SHADE requires min_population_size at least 4.")
        if self.population_schedule is None:
            self.population_schedule = LinearPopulationReduction(
                min_population_size=self.min_population_size
            )
        super().__post_init__()

    def _result_metadata(self) -> dict[str, object]:
        metadata = super()._result_metadata()
        metadata["min_population_size"] = self.min_population_size
        return metadata
