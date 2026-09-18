"""Population diversity measures for Differential Evolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast
from collections.abc import Iterable


Population = list[list[float]]
Bounds = list[tuple[float, float]]


class DiversityMeasure(Protocol):
    """Compute one scalar diversity statistic for a population."""

    name: str

    def __call__(
        self,
        population: Population,
        bounds: Bounds,
        previous_population: Population | None = None,
    ) -> float:
        """Return the diversity value for the current population."""


def _validate_population(population: Population) -> None:
    if not population:
        raise ValueError(
            "Population diversity measures require a non-empty population."
        )


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right, strict=True)
        )
    )


def _population_center(population: Population) -> list[float]:
    dimension = len(population[0])
    return [
        sum(individual[coordinate] for individual in population) / len(population)
        for coordinate in range(dimension)
    ]


def _pairwise_distances(population: Population) -> list[float]:
    distances: list[float] = []
    for left_index in range(len(population)):
        distances.extend(
            _euclidean_distance(population[left_index], population[right_index])
            for right_index in range(left_index + 1, len(population))
        )
    return distances


@dataclass(frozen=True)
class PopulationDiameter:
    """Maximum pairwise Euclidean distance in the population."""

    name: str = "population_diameter"

    def __call__(
        self,
        population: Population,
        bounds: Bounds,
        previous_population: Population | None = None,
    ) -> float:
        _validate_population(population)
        distances = _pairwise_distances(population)
        return max(distances, default=0.0)


@dataclass(frozen=True)
class PopulationRadius:
    """Maximum Euclidean distance from the population center."""

    name: str = "population_radius"

    def __call__(
        self,
        population: Population,
        bounds: Bounds,
        previous_population: Population | None = None,
    ) -> float:
        _validate_population(population)
        center = _population_center(population)
        return max(_euclidean_distance(individual, center) for individual in population)


@dataclass(frozen=True)
class AverageDistanceAroundPopulationCenter:
    """Mean Euclidean distance from the population center."""

    name: str = "average_distance_around_population_center"

    def __call__(
        self,
        population: Population,
        bounds: Bounds,
        previous_population: Population | None = None,
    ) -> float:
        _validate_population(population)
        center = _population_center(population)
        return sum(
            _euclidean_distance(individual, center) for individual in population
        ) / len(population)


@dataclass(frozen=True)
class AverageDistanceAroundAllIndividuals:
    """Mean distance around every population member, including self-zero terms."""

    name: str = "average_distance_around_all_individuals"

    def __call__(
        self,
        population: Population,
        bounds: Bounds,
        previous_population: Population | None = None,
    ) -> float:
        _validate_population(population)
        population_size = len(population)
        total = 0.0
        for center_individual in population:
            total += (
                sum(
                    _euclidean_distance(center_individual, other_individual)
                    for other_individual in population
                )
                / population_size
            )
        return total / population_size


@dataclass(frozen=True)
class AveragePairwiseDistance:
    """Average Euclidean distance over unordered population pairs."""

    name: str = "average_pairwise_distance"

    def __call__(
        self,
        population: Population,
        bounds: Bounds,
        previous_population: Population | None = None,
    ) -> float:
        _validate_population(population)
        distances = _pairwise_distances(population)
        if not distances:
            return 0.0
        return sum(distances) / len(distances)


@dataclass(frozen=True)
class PopulationCoherence:
    """Ratio of center movement to average individual movement."""

    name: str = "population_coherence"

    def __call__(
        self,
        population: Population,
        bounds: Bounds,
        previous_population: Population | None = None,
    ) -> float:
        _validate_population(population)
        if previous_population is None:
            return 0.0
        if len(previous_population) != len(population):
            raise ValueError("Population coherence requires populations of equal size.")

        current_center = _population_center(population)
        previous_center = _population_center(previous_population)
        center_step = _euclidean_distance(current_center, previous_center)
        average_individual_step = sum(
            _euclidean_distance(current, previous)
            for current, previous in zip(population, previous_population, strict=True)
        ) / len(population)
        if average_individual_step == 0.0:
            return 0.0
        return center_step / average_individual_step


@dataclass(frozen=True)
class DimensionalVariance:
    """Average per-dimension variance around the population center."""

    name: str = "dimensional_variance"

    def __call__(
        self,
        population: Population,
        bounds: Bounds,
        previous_population: Population | None = None,
    ) -> float:
        _validate_population(population)
        center = _population_center(population)
        dimension = len(center)
        return (
            sum(
                sum(
                    (individual[coordinate] - center[coordinate]) ** 2
                    for individual in population
                )
                / len(population)
                for coordinate in range(dimension)
            )
            / dimension
        )


@dataclass(frozen=True)
class AggregatedDistribution:
    """Mean marginal index of dispersion over normalized coordinate histograms.

    This is a practical project-specific aggregation statistic:
    each coordinate is normalized into ``[0, 1]``, binned into
    ``ceil(sqrt(population_size))`` bins, and the variance-to-mean ratio
    of the resulting 1-D occupancy counts is averaged across dimensions.
    Values above 1 indicate clumped occupancy, values below 1 indicate
    more even occupancy.
    """

    name: str = "aggregated_distribution"

    def __call__(
        self,
        population: Population,
        bounds: Bounds,
        previous_population: Population | None = None,
    ) -> float:
        _validate_population(population)
        population_size = len(population)
        bin_count = max(2, math.ceil(math.sqrt(population_size)))
        dimensional_indices = []
        for coordinate, (lower, upper) in enumerate(bounds):
            if upper == lower:
                dimensional_indices.append(0.0)
                continue
            counts = [0] * bin_count
            for individual in population:
                normalized = (individual[coordinate] - lower) / (upper - lower)
                clipped = min(max(normalized, 0.0), 1.0)
                bin_index = min(bin_count - 1, int(clipped * bin_count))
                counts[bin_index] += 1
            mean_count = sum(counts) / bin_count
            variance = sum((count - mean_count) ** 2 for count in counts) / bin_count
            dimensional_indices.append(
                0.0 if mean_count == 0.0 else variance / mean_count
            )
        return sum(dimensional_indices) / len(dimensional_indices)


_DIVERSITY_MEASURES = {
    "population_diameter": PopulationDiameter,
    "diameter": PopulationDiameter,
    "population_radius": PopulationRadius,
    "radius": PopulationRadius,
    "average_distance_around_population_center": AverageDistanceAroundPopulationCenter,
    "average_distance_population_center": AverageDistanceAroundPopulationCenter,
    "average_distance_around_all_individuals": AverageDistanceAroundAllIndividuals,
    "average_distance_all_individuals": AverageDistanceAroundAllIndividuals,
    "population_coherence": PopulationCoherence,
    "coherence": PopulationCoherence,
    "dimensional_variance": DimensionalVariance,
    "aggregated_distribution": AggregatedDistribution,
    "average_pairwise_distance": AveragePairwiseDistance,
    "pairwise_distance": AveragePairwiseDistance,
}


def resolve_diversity_measure(
    specification: str | DiversityMeasure,
) -> DiversityMeasure:
    """Normalize a user diversity-measure specification."""

    if callable(specification) and hasattr(specification, "name"):
        return cast(DiversityMeasure, specification)
    if not isinstance(specification, str):
        raise TypeError(
            "Diversity measures must be named strings or callables with a name attribute."
        )
    try:
        return cast(DiversityMeasure, _DIVERSITY_MEASURES[specification]())
    except KeyError as error:
        raise ValueError(f"Unknown diversity measure: {specification!r}") from error


def resolve_diversity_measures(
    specifications: str | DiversityMeasure | Iterable[str | DiversityMeasure] | None,
) -> list[DiversityMeasure]:
    """Resolve zero, one, or many diversity-measure specifications."""

    if specifications is None:
        return []
    if isinstance(specifications, str) and specifications == "none":
        return []
    if isinstance(specifications, (str,)) or hasattr(specifications, "name"):
        return [
            resolve_diversity_measure(cast("str | DiversityMeasure", specifications))
        ]
    measures = [
        resolve_diversity_measure(specification)
        for specification in cast(Iterable[str | DiversityMeasure], specifications)
    ]
    names = [measure.name for measure in measures]
    if len(names) != len(set(names)):
        raise ValueError("Diversity measure names must be unique (including aliases).")
    return measures
