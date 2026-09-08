"""Backward-compatible population initialization helpers."""

from __future__ import annotations

import random

from differential_evolution.initializers import (
    OppositionInitializer,
    QuasiOppositionInitializer,
    RandomInitializer,
    SobolInitializer,
    TentInitializer,
)


def random_initialization(population_size, bounds, seed=None):
    initializer = RandomInitializer()
    return initializer(population_size, [tuple(bound) for bound in bounds], lambda _: 0.0, random.Random(seed))


def tent_initialization(population_size, bounds, seed=None):
    initializer = TentInitializer()
    return initializer(population_size, [tuple(bound) for bound in bounds], lambda _: 0.0, random.Random(seed))


def obl_initialization(population_size, bounds, cost_function, seed=None):
    initializer = OppositionInitializer()
    return initializer(
        population_size,
        [tuple(bound) for bound in bounds],
        cost_function,
        random.Random(seed),
    )


def qobl_initialization(population_size, bounds, cost_function, seed=None):
    initializer = QuasiOppositionInitializer()
    return initializer(
        population_size,
        [tuple(bound) for bound in bounds],
        cost_function,
        random.Random(seed),
    )


def sobol_initialization(population_size, bounds):
    initializer = SobolInitializer()
    return initializer(
        population_size,
        [tuple(bound) for bound in bounds],
        lambda _: 0.0,
        random.Random(0),
    )
