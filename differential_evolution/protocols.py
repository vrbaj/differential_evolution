"""Structural interfaces for random sources and optional component lifecycle hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

T = TypeVar("T")


class RandomSource(Protocol):
    """The random operations used by built-in components."""

    def random(self) -> float: ...
    def uniform(self, a: float, b: float) -> float: ...
    def randrange(self, stop: int, /) -> int: ...
    def sample(self, population: Sequence[T], k: int) -> list[T]: ...
    def gauss(self, mu: float, sigma: float) -> float: ...


class StatefulComponent(Protocol):
    """Optional hooks: initialize, then propose/commit per target, then resize.

    Plain callable mutation/crossover objects need not implement these hooks.
    Controllers must discard rejected proposals at commit and pending proposals
    at resize. A default CurrentTo* scale is proposed only once per target.
    """

    def initialize(self, population_size: int, rng: RandomSource) -> None: ...
    def commit(self, target_index: int, accepted: bool) -> None: ...
    def resize(self, kept_indices: list[int]) -> None: ...


class TargetAwareComponent(Protocol):
    """Optional crossover hook called before generating the target's trial."""

    def set_target_index(self, target_index: int) -> None: ...
