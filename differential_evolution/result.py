"""Optimization result containers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .history import GenerationSnapshot, save_json, save_pickle, snapshots_to_dicts


@dataclass(frozen=True)
class OptimizeResult:
    """Final state of a Differential Evolution optimization run.

    Stores the final best point, final population, optimization histories,
    optional snapshots, and algorithm-specific metadata.
    """

    x: list[float]
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str
    population: list[list[float]]
    fitness: list[float]
    best_history: list[list[float]]
    best_fitness_history: list[float]
    diversity_history: dict[str, list[float]]
    population_size_history: list[int]
    snapshots: list[GenerationSnapshot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain serializable dictionary."""

        result = asdict(self)
        result["snapshots"] = snapshots_to_dicts(self.snapshots)
        return result

    def save_json(self, path: str | Path) -> None:
        """Save the full result as JSON."""

        save_json(self.to_dict(), path)

    def save_pickle(self, path: str | Path) -> None:
        """Save the full result as a pickle file."""

        save_pickle(self, path)
