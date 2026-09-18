"""Per-generation run history containers."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationSnapshot:
    """Serializable snapshot of one optimizer state."""

    generation: int
    evaluations: int
    population_size: int
    best_vector: list[float]
    best_fitness: float
    population: list[list[float]] | None
    fitness: list[float] | None
    diversity: dict[str, float]
    extra: dict[str, Any]


def snapshots_to_dicts(snapshots: list[GenerationSnapshot]) -> list[dict[str, Any]]:
    """Convert snapshots to plain dictionaries."""

    return [asdict(snapshot) for snapshot in snapshots]


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON-serializable data."""

    def normalize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    Path(path).write_text(
        json.dumps(normalize(data), indent=2, allow_nan=False), encoding="utf-8"
    )


def save_pickle(data: Any, path: str | Path) -> None:
    """Save arbitrary Python data with pickle."""

    with Path(path).open("wb") as handle:
        pickle.dump(data, handle)
