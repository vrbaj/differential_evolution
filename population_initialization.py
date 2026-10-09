"""Backward-compatible population initialization helpers."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from differential_evolution.initializers import (
    OppositionInitializer,
    QuasiOppositionInitializer,
    RandomInitializer,
    SobolInitializer,
    TentInitializer,
)


def random_initialization(
    population_size: int,
    bounds: Sequence[Sequence[float]],
    seed: int | float | str | bytes | bytearray | None = None,
) -> list[list[float]]:
    initializer = RandomInitializer()
    return initializer(
        population_size,
        [(lower, upper) for lower, upper in bounds],
        lambda _: 0.0,
        random.Random(seed),
    )


def tent_initialization(
    population_size: int,
    bounds: Sequence[Sequence[float]],
    seed: int | float | str | bytes | bytearray | None = None,
) -> list[list[float]]:
    initializer = TentInitializer()
    return initializer(
        population_size,
        [(lower, upper) for lower, upper in bounds],
        lambda _: 0.0,
        random.Random(seed),
    )


def obl_initialization(
    population_size: int,
    bounds: Sequence[Sequence[float]],
    cost_function: Callable[[list[float]], float],
    seed: int | float | str | bytes | bytearray | None = None,
) -> list[list[float]]:
    initializer = OppositionInitializer()
    return initializer(
        population_size,
        [(lower, upper) for lower, upper in bounds],
        cost_function,
        random.Random(seed),
    )


def qobl_initialization(
    population_size: int,
    bounds: Sequence[Sequence[float]],
    cost_function: Callable[[list[float]], float],
    seed: int | float | str | bytes | bytearray | None = None,
) -> list[list[float]]:
    initializer = QuasiOppositionInitializer()
    return initializer(
        population_size,
        [(lower, upper) for lower, upper in bounds],
        cost_function,
        random.Random(seed),
    )


def sobol_initialization(
    population_size: int,
    bounds: Sequence[Sequence[float]],
) -> list[list[float]]:
    initializer = SobolInitializer()
    return initializer(
        population_size,
        [(lower, upper) for lower, upper in bounds],
        lambda _: 0.0,
        random.Random(0),
    )
