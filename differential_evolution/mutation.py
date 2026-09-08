"""Mutation strategies and index sampling for Differential Evolution."""

from __future__ import annotations

import math
from dataclasses import fields
from dataclasses import dataclass


Population = list[list[float]]


def sample_distinct_indices(
    population_size: int,
    count: int,
    rng: object,
    excluded: tuple[int, ...] = (),
) -> list[int]:
    """Sample distinct population indices while excluding reserved indices.

    Parameters
    ----------
    population_size
        Number of currently available population members.
    count
        Number of distinct indices to draw.
    rng
        Random-generator-like object exposing ``sample(population, k)``.
    excluded
        Indices that must not appear in the result.

    Returns
    -------
    list[int]
        ``count`` mutually distinct indices.

    Raises
    ------
    ValueError
        If too few eligible indices remain.
    """

    available = [index for index in range(population_size) if index not in set(excluded)]
    if count > len(available):
        raise ValueError(
            "Population too small for the requested mutation strategy and exclusions."
        )
    return list(rng.sample(available, count))


@dataclass(frozen=True)
class MutationContext:
    """Immutable information available to a mutation strategy.

    The fields expose the population snapshot, aligned fitness values, the
    current target index, the current best individual, coordinate bounds, and
    the shared random-generator-like object. Mutation operators should treat
    the stored population as read-only and return a new donor vector.
    """

    population: Population
    fitness: list[float]
    target_index: int
    best_index: int
    best_vector: list[float]
    bounds: list[tuple[float, float]]
    rng: object


class BaseMutation:
    """Shared helper for mutation strategies."""

    def _require_dimension(self, context: MutationContext) -> None:
        if not context.population:
            raise ValueError("Population must be initialized before mutation.")

    def initialize(self, population_size: int, rng: object) -> None:
        for controller in self._scale_controllers():
            controller.initialize(population_size, rng)

    def commit(self, target_index: int, accepted: bool) -> None:
        for controller in self._scale_controllers():
            controller.commit(target_index, accepted)

    def resize(self, kept_indices: list[int]) -> None:
        for controller in self._scale_controllers():
            if hasattr(controller, "resize"):
                controller.resize(kept_indices)

    def _resolve_scale(self, value: float | object, context: MutationContext) -> float:
        if hasattr(value, "propose"):
            return float(value.propose(context))
        return float(value)

    def _scale_controllers(self) -> list[object]:
        seen: set[int] = set()
        controllers = []
        for field_definition in fields(self):
            value = getattr(self, field_definition.name)
            if hasattr(value, "initialize") and hasattr(value, "propose") and hasattr(value, "commit"):
                if id(value) not in seen:
                    seen.add(id(value))
                    controllers.append(value)
        return controllers


@dataclass(frozen=True)
class Rand1(BaseMutation):
    """DE/rand/1 donor mutation."""

    scale: float | object

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        scale = self._resolve_scale(self.scale, context)
        r1, r2, r3 = sample_distinct_indices(
            len(context.population),
            3,
            context.rng,
            excluded=(context.target_index,),
        )
        return [
            context.population[r1][dimension]
            + scale
            * (context.population[r2][dimension] - context.population[r3][dimension])
            for dimension in range(len(context.population[context.target_index]))
        ]


@dataclass(frozen=True)
class Rand2(BaseMutation):
    """DE/rand/2 donor mutation."""

    scale: float | object

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        scale = self._resolve_scale(self.scale, context)
        r1, r2, r3, r4, r5 = sample_distinct_indices(
            len(context.population),
            5,
            context.rng,
            excluded=(context.target_index,),
        )
        return [
            context.population[r1][dimension]
            + scale
            * (context.population[r2][dimension] - context.population[r3][dimension])
            + scale
            * (context.population[r4][dimension] - context.population[r5][dimension])
            for dimension in range(len(context.population[context.target_index]))
        ]


@dataclass(frozen=True)
class Best1(BaseMutation):
    """DE/best/1 donor mutation."""

    scale: float | object

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        scale = self._resolve_scale(self.scale, context)
        r1, r2 = sample_distinct_indices(
            len(context.population),
            2,
            context.rng,
            excluded=(context.target_index, context.best_index),
        )
        return [
            context.best_vector[dimension]
            + scale
            * (context.population[r1][dimension] - context.population[r2][dimension])
            for dimension in range(len(context.population[context.target_index]))
        ]


@dataclass(frozen=True)
class Best2(BaseMutation):
    """DE/best/2 donor mutation."""

    scale: float | object

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        scale = self._resolve_scale(self.scale, context)
        r1, r2, r3, r4 = sample_distinct_indices(
            len(context.population),
            4,
            context.rng,
            excluded=(context.target_index, context.best_index),
        )
        return [
            context.best_vector[dimension]
            + scale
            * (context.population[r1][dimension] - context.population[r2][dimension])
            + scale
            * (context.population[r3][dimension] - context.population[r4][dimension])
            for dimension in range(len(context.population[context.target_index]))
        ]


@dataclass(frozen=True)
class CurrentToBest1(BaseMutation):
    """DE/current-to-best/1 with an optional separate difference scale."""

    scale: float | object
    difference_scale: float | object | None = None

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        scale = self._resolve_scale(self.scale, context)
        difference_scale_value = self.scale if self.difference_scale is None else self.difference_scale
        difference_scale = self._resolve_scale(difference_scale_value, context)
        r1, r2 = sample_distinct_indices(
            len(context.population),
            2,
            context.rng,
            excluded=(context.target_index, context.best_index),
        )
        target_vector = context.population[context.target_index]
        return [
            target_vector[dimension]
            + scale * (context.best_vector[dimension] - target_vector[dimension])
            + difference_scale
            * (context.population[r1][dimension] - context.population[r2][dimension])
            for dimension in range(len(target_vector))
        ]


@dataclass(frozen=True)
class CurrentToBest2(BaseMutation):
    """DE/current-to-best/2 with an optional separate difference scale."""

    scale: float | object
    difference_scale: float | object | None = None

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        scale = self._resolve_scale(self.scale, context)
        difference_scale_value = self.scale if self.difference_scale is None else self.difference_scale
        difference_scale = self._resolve_scale(difference_scale_value, context)
        r1, r2, r3, r4 = sample_distinct_indices(
            len(context.population),
            4,
            context.rng,
            excluded=(context.target_index, context.best_index),
        )
        target_vector = context.population[context.target_index]
        return [
            target_vector[dimension]
            + scale * (context.best_vector[dimension] - target_vector[dimension])
            + difference_scale
            * (context.population[r1][dimension] - context.population[r2][dimension])
            + difference_scale
            * (context.population[r3][dimension] - context.population[r4][dimension])
            for dimension in range(len(target_vector))
        ]


@dataclass(frozen=True)
class CurrentToRand1(BaseMutation):
    """DE/current-to-rand/1 donor mutation."""

    scale: float | object
    difference_scale: float | object | None = None

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        scale = self._resolve_scale(self.scale, context)
        difference_scale_value = self.scale if self.difference_scale is None else self.difference_scale
        difference_scale = self._resolve_scale(difference_scale_value, context)
        r1, r2, r3 = sample_distinct_indices(
            len(context.population),
            3,
            context.rng,
            excluded=(context.target_index,),
        )
        target_vector = context.population[context.target_index]
        return [
            target_vector[dimension]
            + scale * (context.population[r1][dimension] - target_vector[dimension])
            + difference_scale
            * (context.population[r2][dimension] - context.population[r3][dimension])
            for dimension in range(len(target_vector))
        ]


@dataclass(frozen=True)
class CurrentToRand2(BaseMutation):
    """DE/current-to-rand/2 donor mutation."""

    scale: float | object
    difference_scale: float | object | None = None

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        scale = self._resolve_scale(self.scale, context)
        difference_scale_value = self.scale if self.difference_scale is None else self.difference_scale
        difference_scale = self._resolve_scale(difference_scale_value, context)
        r1, r2, r3, r4, r5 = sample_distinct_indices(
            len(context.population),
            5,
            context.rng,
            excluded=(context.target_index,),
        )
        target_vector = context.population[context.target_index]
        return [
            target_vector[dimension]
            + scale * (context.population[r1][dimension] - target_vector[dimension])
            + difference_scale
            * (context.population[r2][dimension] - context.population[r3][dimension])
            + difference_scale
            * (context.population[r4][dimension] - context.population[r5][dimension])
            for dimension in range(len(target_vector))
        ]


@dataclass(frozen=True)
class TrigonometricMutation(BaseMutation):
    """Fan-Lampinen trigonometric mutation with DE/rand/1 fallback.

    With probability ``probability`` this applies the trigonometric
    mutation operator from Fan and Lampinen (2003). Otherwise it applies
    the usual ``DE/rand/1`` donor formula using the same sampled indices.
    """

    probability: float
    scale: float | object

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        r1, r2, r3 = sample_distinct_indices(
            len(context.population),
            3,
            context.rng,
            excluded=(context.target_index,),
        )
        if context.rng.random() <= self.probability:
            return self._trigonometric_donor(context, r1, r2, r3)
        return self._rand1_donor(context, r1, r2, r3)

    def _rand1_donor(
        self,
        context: MutationContext,
        r1: int,
        r2: int,
        r3: int,
    ) -> list[float]:
        scale = self._resolve_scale(self.scale, context)
        return [
            context.population[r1][dimension]
            + scale
            * (context.population[r2][dimension] - context.population[r3][dimension])
            for dimension in range(len(context.population[context.target_index]))
        ]

    def _trigonometric_donor(
        self,
        context: MutationContext,
        r1: int,
        r2: int,
        r3: int,
    ) -> list[float]:
        selected_fitness = [
            context.fitness[r1],
            context.fitness[r2],
            context.fitness[r3],
        ]
        absolute_sum = sum(abs(value) for value in selected_fitness)
        if not math.isfinite(absolute_sum):
            return self._rand1_donor(context, r1, r2, r3)
        if absolute_sum == 0.0:
            weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        else:
            weights = [abs(value) / absolute_sum for value in selected_fitness]

        vector_1 = context.population[r1]
        vector_2 = context.population[r2]
        vector_3 = context.population[r3]
        mean_vector = [
            (value_1 + value_2 + value_3) / 3.0
            for value_1, value_2, value_3 in zip(vector_1, vector_2, vector_3)
        ]
        p1, p2, p3 = weights
        return [
            mean_value
            + (p2 - p1) * (value_1 - value_2)
            + (p3 - p2) * (value_2 - value_3)
            + (p1 - p3) * (value_3 - value_1)
            for mean_value, value_1, value_2, value_3 in zip(
                mean_vector,
                vector_1,
                vector_2,
                vector_3,
            )
        ]


@dataclass(frozen=True)
class DirectedMutation(BaseMutation):
    """Fan-Lampinen directed mutation with DE/rand/1 fallback.

    The sampled triple is ordered by objective value. The best sampled vector
    becomes the base vector and the two worse vectors define the directed
    extrapolation terms. If the canonical coefficients are undefined because
    the worse objective values are non-positive or non-finite, the operator
    falls back to ``DE/rand/1`` using the original sampled order.
    """

    fallback_scale: float | object = 1.0

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        sampled = sample_distinct_indices(
            len(context.population),
            3,
            context.rng,
            excluded=(context.target_index,),
        )
        ordered = sorted(sampled, key=lambda index: context.fitness[index])
        best_index, worse_index_1, worse_index_2 = ordered

        best_fitness = context.fitness[best_index]
        worse_fitness_1 = context.fitness[worse_index_1]
        worse_fitness_2 = context.fitness[worse_index_2]

        if not all(
            math.isfinite(value) and value > 0.0
            for value in (best_fitness, worse_fitness_1, worse_fitness_2)
        ):
            return self._rand1_fallback(context, sampled)

        best_vector = context.population[best_index]
        worse_vector_1 = context.population[worse_index_1]
        worse_vector_2 = context.population[worse_index_2]

        coefficient_1 = (1.0 - best_fitness) / worse_fitness_1
        coefficient_2 = (1.0 - best_fitness) / worse_fitness_2

        return [
            best_value
            + coefficient_1 * (best_value - worse_value_1)
            + coefficient_2 * (best_value - worse_value_2)
            for best_value, worse_value_1, worse_value_2 in zip(
                best_vector,
                worse_vector_1,
                worse_vector_2,
            )
        ]

    def _rand1_fallback(
        self,
        context: MutationContext,
        sampled: list[int],
    ) -> list[float]:
        r1, r2, r3 = sampled
        scale = self._resolve_scale(self.fallback_scale, context)
        return [
            context.population[r1][dimension]
            + scale
            * (context.population[r2][dimension] - context.population[r3][dimension])
            for dimension in range(len(context.population[context.target_index]))
        ]


@dataclass(frozen=True)
class NeighborhoodSearchMutation(BaseMutation):
    """Neighborhood Search Differential Evolution mutation.

    This implements the NSDE mutation described by Yang, He, and Yao:

    ``v_i = x_r1 + d_i * N(mu, sigma)`` with probability ``gaussian_probability``
    and ``v_i = x_r1 + d_i * Cauchy(0, cauchy_scale)`` otherwise, where
    ``d_i = x_r2 - x_r3``.
    """

    gaussian_probability: float = 0.5
    gaussian_mean: float = 0.5
    gaussian_stddev: float = 0.5
    cauchy_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gaussian_probability <= 1.0:
            raise ValueError("gaussian_probability must be between 0 and 1.")
        if self.gaussian_stddev < 0.0:
            raise ValueError("gaussian_stddev must be non-negative.")
        if self.cauchy_scale <= 0.0:
            raise ValueError("cauchy_scale must be positive.")

    def __call__(self, context: MutationContext) -> list[float]:
        self._require_dimension(context)
        r1, r2, r3 = sample_distinct_indices(
            len(context.population),
            3,
            context.rng,
            excluded=(context.target_index,),
        )
        factor = self._sample_factor(context.rng)
        return [
            context.population[r1][dimension]
            + factor
            * (context.population[r2][dimension] - context.population[r3][dimension])
            for dimension in range(len(context.population[context.target_index]))
        ]

    def _sample_factor(self, rng: object) -> float:
        if rng.random() < self.gaussian_probability:
            return rng.gauss(self.gaussian_mean, self.gaussian_stddev)
        return self._sample_cauchy(rng)

    def _sample_cauchy(self, rng: object) -> float:
        uniform = rng.random()
        while uniform <= 0.0 or uniform >= 1.0:
            uniform = rng.random()
        return self.cauchy_scale * math.tan(math.pi * (uniform - 0.5))
