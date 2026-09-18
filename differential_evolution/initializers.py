"""Population initialization strategies."""

from __future__ import annotations

from .protocols import RandomSource

from dataclasses import dataclass
from typing import Callable, Protocol


Bounds = list[tuple[float, float]]
ObjectiveFunction = Callable[[list[float]], float]

_SOBOL_BIT_COUNT = 30
_SOBOL_MAX_DIMENSION = 40
_SOBOL_PRIMITIVE_POLYNOMIALS = [
    1,
    3,
    7,
    11,
    13,
    19,
    25,
    37,
    59,
    47,
    61,
    55,
    41,
    67,
    97,
    91,
    109,
    103,
    115,
    131,
    193,
    137,
    145,
    143,
    241,
    157,
    185,
    167,
    229,
    171,
    213,
    191,
    253,
    203,
    211,
    239,
    247,
    285,
    369,
    299,
]
_SOBOL_DEGREES = [
    0,
    1,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
]
_SOBOL_V_INIT = [
    [
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    [
        0,
        0,
        1,
        3,
        1,
        3,
        1,
        3,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
    ],
    [
        0,
        0,
        0,
        7,
        5,
        1,
        3,
        3,
        7,
        5,
        5,
        7,
        7,
        1,
        3,
        3,
        7,
        5,
        1,
        1,
        5,
        3,
        3,
        1,
        7,
        5,
        1,
        3,
        3,
        7,
        5,
        1,
        1,
        5,
        7,
        7,
        5,
        1,
        3,
        3,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        1,
        7,
        9,
        13,
        11,
        1,
        3,
        7,
        9,
        5,
        13,
        13,
        11,
        3,
        15,
        5,
        3,
        15,
        7,
        9,
        13,
        9,
        1,
        11,
        7,
        5,
        15,
        1,
        15,
        11,
        5,
        3,
        1,
        7,
        9,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        9,
        3,
        27,
        15,
        29,
        21,
        23,
        19,
        11,
        25,
        7,
        13,
        17,
        1,
        25,
        29,
        3,
        31,
        11,
        5,
        23,
        27,
        19,
        21,
        5,
        1,
        17,
        13,
        7,
        15,
        9,
        31,
        9,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        37,
        33,
        7,
        5,
        11,
        39,
        63,
        27,
        17,
        15,
        23,
        29,
        3,
        21,
        13,
        31,
        25,
        9,
        49,
        33,
        19,
        29,
        11,
        19,
        27,
        15,
        25,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        13,
        33,
        115,
        41,
        79,
        17,
        29,
        119,
        75,
        73,
        105,
        7,
        59,
        65,
        21,
        3,
        113,
        61,
        89,
        45,
        107,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        7,
        23,
        39,
    ],
]


class PopulationInitializer(Protocol):
    """Create an initial population."""

    def __call__(
        self,
        population_size: int,
        bounds: Bounds,
        objective: ObjectiveFunction,
        rng: RandomSource,
    ) -> list[list[float]]:
        """Return population_size vectors within bounds."""


def _random_vector(bounds: Bounds, rng: RandomSource) -> list[float]:
    return [rng.uniform(lower, upper) for lower, upper in bounds]


@dataclass(frozen=True)
class RandomInitializer:
    """Independent uniform sampling inside bounds."""

    def __call__(
        self,
        population_size: int,
        bounds: Bounds,
        objective: ObjectiveFunction,
        rng: RandomSource,
    ) -> list[list[float]]:
        return [_random_vector(bounds, rng) for _ in range(population_size)]


@dataclass(frozen=True)
class TentInitializer:
    """Tent-map initialization with independently seeded coordinate chains.

    Chains are refreshed every 32 steps to avoid binary floating-point collapse.
    """

    def __call__(
        self,
        population_size: int,
        bounds: Bounds,
        objective: ObjectiveFunction,
        rng: RandomSource,
    ) -> list[list[float]]:
        states = [rng.random() for _ in bounds]
        population = []
        for _ in range(population_size):
            individual = []
            for dimension, (lower, upper) in enumerate(bounds):
                # Refresh each chain before finite precision can collapse it.
                x_value = states[dimension]
                if _ % 32 == 0 or x_value in (0.0, 0.5, 1.0):
                    x_value = rng.random()
                x_value = 2.0 * x_value if x_value < 0.5 else 2.0 * (1.0 - x_value)
                if x_value in (0.0, 1.0):
                    x_value = rng.uniform(0.25, 0.75)
                states[dimension] = x_value
                individual.append(lower + x_value * (upper - lower))
            population.append(individual)
        return population


@dataclass(frozen=True)
class OppositionInitializer:
    """Opposition-based learning initializer.

    The initializer samples random vectors, forms their exact opposites with
    respect to the bounds, evaluates the doubled pool, and returns the best
    ``population_size`` candidates.
    """

    def __call__(
        self,
        population_size: int,
        bounds: Bounds,
        objective: ObjectiveFunction,
        rng: RandomSource,
    ) -> list[list[float]]:
        expanded = []
        for _ in range(population_size):
            vector = []
            opposite = []
            for lower, upper in bounds:
                value = rng.uniform(lower, upper)
                vector.append(value)
                opposite.append(lower + upper - value)
            expanded.append(vector)
            expanded.append(opposite)
        expanded.sort(key=objective)
        return [list(vector) for vector in expanded[:population_size]]


@dataclass(frozen=True)
class QuasiOppositionInitializer:
    """Quasi-opposition-based learning initializer.

    The initializer samples random vectors, forms one quasi-opposite vector for
    each sample using the coordinate-wise midpoint, evaluates the doubled pool,
    and returns the best ``population_size`` candidates.
    """

    def __call__(
        self,
        population_size: int,
        bounds: Bounds,
        objective: ObjectiveFunction,
        rng: RandomSource,
    ) -> list[list[float]]:
        expanded = []
        for _ in range(population_size):
            vector = []
            quasi_opposite = []
            for lower, upper in bounds:
                value = rng.uniform(lower, upper)
                midpoint = 0.5 * (lower + upper)
                opposite = lower + upper - value
                vector.append(value)
                quasi_opposite.append(midpoint + (opposite - midpoint) * rng.random())
            expanded.append(vector)
            expanded.append(quasi_opposite)
        expanded.sort(key=objective)
        return [list(vector) for vector in expanded[:population_size]]


def _least_significant_zero_bit(index: int) -> int:
    position = 1
    while index & 1:
        position += 1
        index >>= 1
    return position


def _build_sobol_directions(dimension: int) -> tuple[list[list[int]], float]:
    if not 1 <= dimension <= _SOBOL_MAX_DIMENSION:
        raise ValueError(
            f"Sobol initialization supports dimensions 1 through {_SOBOL_MAX_DIMENSION}."
        )

    directions = [[0] * dimension for _ in range(_SOBOL_BIT_COUNT)]

    for bit_index in range(_SOBOL_BIT_COUNT):
        directions[bit_index][0] = 1

    for dim_index in range(1, dimension):
        degree = _SOBOL_DEGREES[dim_index]
        polynomial = _SOBOL_PRIMITIVE_POLYNOMIALS[dim_index]
        include = [0] * degree
        for bit_index in range(degree - 1, -1, -1):
            include[bit_index] = polynomial & 1
            polynomial >>= 1

        for bit_index in range(degree):
            directions[bit_index][dim_index] = _SOBOL_V_INIT[bit_index][dim_index]

        for bit_index in range(degree, _SOBOL_BIT_COUNT):
            new_direction = directions[bit_index - degree][dim_index]
            scale = 1
            for offset in range(degree):
                scale *= 2
                if include[offset]:
                    new_direction ^= (
                        scale * directions[bit_index - offset - 1][dim_index]
                    )
            directions[bit_index][dim_index] = new_direction

    denominator_factor = 1
    for bit_index in range(_SOBOL_BIT_COUNT - 2, -1, -1):
        denominator_factor *= 2
        for dim_index in range(dimension):
            directions[bit_index][dim_index] *= denominator_factor

    denominator_inverse = 1.0 / (2.0 * denominator_factor)
    return directions, denominator_inverse


def _sobol_unit_points(population_size: int, dimension: int) -> list[list[float]]:
    directions, denominator_inverse = _build_sobol_directions(dimension)
    numerators = [0] * dimension
    points: list[list[float]] = []

    for index in range(population_size):
        if index == 0:
            points.append([0.0] * dimension)
            continue

        zero_bit = _least_significant_zero_bit(index - 1)
        if zero_bit > _SOBOL_BIT_COUNT:
            raise ValueError(
                f"Sobol initialization supports at most {2**_SOBOL_BIT_COUNT} points."
            )
        direction_row = directions[zero_bit - 1]
        for dim_index in range(dimension):
            numerators[dim_index] ^= direction_row[dim_index]
        points.append([value * denominator_inverse for value in numerators])

    return points


@dataclass(frozen=True)
class SobolInitializer:
    """Randomly digitally shifted Sobol initialization using Bratley-Fox parameters.

    Notes
    -----
    The built-in direction-number tables currently support dimensions 1
    through 40 and at most ``2**30`` generated points.
    """

    scramble: bool = True

    def __call__(
        self,
        population_size: int,
        bounds: Bounds,
        objective: ObjectiveFunction,
        rng: RandomSource,
    ) -> list[list[float]]:
        if population_size < 0:
            raise ValueError("population_size must be non-negative.")
        unit_points = _sobol_unit_points(population_size, len(bounds))
        shifts = [
            rng.randrange(2**_SOBOL_BIT_COUNT) if self.scramble else 0 for _ in bounds
        ]
        unit_points = [
            [
                (int(value * 2**_SOBOL_BIT_COUNT) ^ shift) / 2**_SOBOL_BIT_COUNT
                for value, shift in zip(point, shifts, strict=True)
            ]
            for point in unit_points
        ]
        return [
            [
                lower + coordinate * (upper - lower)
                for coordinate, (lower, upper) in zip(point, bounds, strict=True)
            ]
            for point in unit_points
        ]
