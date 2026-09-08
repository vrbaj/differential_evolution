# Migration Guide

## New API

The refactored package uses explicit components:

```python
from differential_evolution import (
    BinomialCrossover,
    DifferentialEvolution,
    Rand1,
)

optimizer = DifferentialEvolution(
    objective=lambda x: x[0] ** 2 + x[1] ** 2,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    population_size=20,
    mutation=Rand1(scale=0.8),
    crossover=BinomialCrossover(crossover_rate=0.9),
    max_generations=100,
    seed=123,
)
result = optimizer.run()
```

## Legacy constructor

The old import path still works:

```python
from main import DifferentialEvolution
```

It now routes through a compatibility wrapper and emits a `DeprecationWarning`.

## Behavioral changes

- Binomial crossover was corrected to the canonical DE definition.
- Trial-vector generation now always uses a full generation snapshot.
- Randomness is explicit and reproducible through `seed` or a supplied RNG.
- Boundary handling is explicit instead of being an unimplemented comment.
- The `sobol` initializer now uses a correct deterministic Sobol sequence implementation for dimensions up to 40.

## Parameter mapping

- Old `max_iterations` maps to new `max_generations`.
- Old `mutation=[...]` maps to explicit mutation objects.
- Old scalar `crossover` maps to `BinomialCrossover(crossover_rate=...)`.
- Old `population_initialization_algorithm` strings map to initializer objects.

## Current-to-best and current-to-rand scales

The old implementation used two different mutation factors for these strategies. The new API keeps that behavior available:

```python
from differential_evolution import CurrentToBest1

mutation = CurrentToBest1(scale=0.8, difference_scale=0.5)
```

If `difference_scale` is omitted, the canonical single-factor form is used.

## Dynamic scale factors

The new API also supports reusable scale-factor controllers:

```python
from differential_evolution import RandomizedScaleFactor, Rand1

mutation = Rand1(scale=RandomizedScaleFactor(lower=0.5, upper=1.0))
```

For jDE-style self-adapting `F`, use `AdaptiveScaleFactor`.

To match Brest et al.'s self-adapting control-parameter scheme more closely, pair it with `AdaptiveCrossoverRate` or use `jde_rand_1_bin()`.

## Diversity tracking

The new optimizer can record zero, one, or several diversity measures during the run:

```python
optimizer = DifferentialEvolution(
    ...,
    diversity_measures=["population_diameter", "average_pairwise_distance"],
)
```

The recorded values are returned in `result.diversity_history`.

## Population reduction

The new optimizer supports optional population-size schedules:

```python
optimizer = DifferentialEvolution(
    ...,
    population_schedule=LinearPopulationReduction(min_population_size=4),
    max_evaluations=2000,
)
```

`LinearPopulationReduction` is the closest built-in analogue to L-SHADE's linear population size reduction. A smooth nonlinear alternative is available as `HyperbolicTangentPopulationReduction`.

## SHADE family

The package now includes dedicated `SHADE` and `LSHADE` optimizers for the full success-history adaptive algorithms. Use these when you want the canonical success-history memory mechanism, `current-to-pbest/1`, and the external archive rather than the more generic composable DE optimizer.

## Saving trajectories

Set `record_snapshots=True` and use `result.save_json(...)` or `result.save_pickle(...)` to preserve populations, diversity histories, and algorithm-specific state for later analysis.
