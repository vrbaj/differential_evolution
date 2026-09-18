# Differential Evolution

This repository now provides a small, explicit Differential Evolution package with independently composable mutation strategies and crossover operators.

## What changed

- The optimizer is now a real Python package: `differential_evolution/`
- Mutation and crossover are separate components
- Randomness is explicit and reproducible
- A deterministic test suite verifies formulas and DE invariants
- The original `main.py`, `population_initialization.py`, and `testing_functions.py` remain as compatibility shims
- Full `SHADE` and `LSHADE` optimizers are available for success-history adaptive DE

## Installation

The project is pure Python and has no mandatory runtime dependencies.

```bash
python3 -m pip install -e .
```

## Core concepts

- target vector: the current population member being evolved
- donor vector: the output of a mutation strategy
- trial vector: the result of applying a crossover operator to the target and donor

The optimizer orchestrates:

1. population initialization
2. donor-vector mutation
3. target/donor crossover
4. boundary handling
5. objective evaluation
6. greedy one-to-one selection

## Built-in mutation strategies

- `Rand1`
- `Rand2`
- `Best1`
- `Best2`
- `CurrentToBest1`
- `CurrentToBest2`
- `CurrentToRand1`
- `CurrentToRand2`
- `TrigonometricMutation`
- `DirectedMutation`
- `NeighborhoodSearchMutation`

## Built-in scale-factor controllers

- `ConstantScaleFactor`
- `RandomizedScaleFactor`
- `AdaptiveScaleFactor`

## Built-in crossover-rate controllers

- `ConstantCrossoverRate`
- `AdaptiveCrossoverRate`

These can be passed anywhere a mutation strategy expects `scale` or `difference_scale`.

```python
from differential_evolution import AdaptiveScaleFactor, Rand1

mutation = Rand1(scale=AdaptiveScaleFactor(initial=0.5, tau=0.1, lower=0.1, upper=0.9))
```

For the Brest et al. paper's full `jDE` control-parameter scheme, pair `AdaptiveScaleFactor` with `AdaptiveCrossoverRate`, or use the convenience helper:

```python
from differential_evolution import DifferentialEvolution, jde_rand_1_bin, sphere_function

components = jde_rand_1_bin()
optimizer = DifferentialEvolution(
    objective=sphere_function,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    population_size=20,
    mutation=components.mutation,
    crossover=components.crossover,
    max_generations=100,
    seed=123,
)
```

## Built-in crossover operators

- `BinomialCrossover`
- `ExponentialCrossover`
- `IdentityCrossover`

`IdentityCrossover` is useful for canonical current-to-rand style runs where no extra crossover is desired.

## Built-in diversity measures

- `PopulationDiameter`
- `PopulationRadius`
- `AverageDistanceAroundPopulationCenter`
- `AverageDistanceAroundAllIndividuals`
- `PopulationCoherence`
- `DimensionalVariance`
- `AggregatedDistribution`
- `AveragePairwiseDistance`

You can disable diversity tracking by leaving `diversity_measures=None`, or track several measures at once:

```python
from differential_evolution import (
    AveragePairwiseDistance,
    DifferentialEvolution,
    PopulationDiameter,
    Rand1,
    BinomialCrossover,
    sphere_function,
)

optimizer = DifferentialEvolution(
    objective=sphere_function,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    population_size=20,
    mutation=Rand1(scale=0.8),
    crossover=BinomialCrossover(crossover_rate=0.9),
    diversity_measures=[
        PopulationDiameter(),
        AveragePairwiseDistance(),
        "population_coherence",
    ],
    max_generations=50,
    seed=123,
)
result = optimizer.run()
print(result.diversity_history["population_diameter"])
```

`PopulationCoherence` compares the movement of the population center between consecutive generations with the average movement of individuals. Its first recorded value is `0.0` because there is no previous generation yet.

`AggregatedDistribution` is implemented here as a project-specific marginal dispersion statistic: each coordinate is normalized to `[0, 1]`, binned, and summarized by the variance-to-mean ratio of occupancy counts.

## Built-in initializers

- `RandomInitializer`
- `TentInitializer`
- `OppositionInitializer`
- `QuasiOppositionInitializer`
- `SobolInitializer`

The original Sobol implementation was wrong and has been replaced with a correct low-discrepancy initializer for dimensions up to 40. Details are in [ALGORITHM_AUDIT.md](ALGORITHM_AUDIT.md).

## Population reduction

The optimizer can optionally reduce population size during the run:

- `LinearPopulationReduction`
- `HyperbolicTangentPopulationReduction`

```python
from differential_evolution import (
    BinomialCrossover,
    DifferentialEvolution,
    LinearPopulationReduction,
    Rand1,
    sphere_function,
)

optimizer = DifferentialEvolution(
    objective=sphere_function,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    population_size=20,
    mutation=Rand1(scale=0.8),
    crossover=BinomialCrossover(crossover_rate=0.9),
    population_schedule=LinearPopulationReduction(min_population_size=4),
    max_generations=100,
    max_evaluations=2000,
    seed=123,
)
```

`LinearPopulationReduction` follows the L-SHADE idea of shrinking linearly with used function-evaluation budget when `max_evaluations` is provided. If `max_evaluations` is omitted, it falls back to generation progress. `HyperbolicTangentPopulationReduction` is a project-specific smooth nonlinear schedule.

When the population is reduced, the worst individuals by current fitness are removed. Users must choose a minimum population size that remains compatible with the mutation strategy in use.

## SHADE and L-SHADE

The package also includes dedicated optimizers for the full success-history adaptive algorithms:

- `SHADE`
- `LSHADE`

`SHADE` includes:

- `current-to-pbest/1`
- external archive
- Cauchy sampling of `F`
- Gaussian sampling of `CR`
- success-history memories `M_F` and `M_CR`

`LSHADE` extends `SHADE` with linear population size reduction.

```python
from differential_evolution import LSHADE, sphere_function

optimizer = LSHADE(
    objective=sphere_function,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    population_size=18,
    min_population_size=4,
    memory_size=6,
    max_generations=100,
    max_evaluations=2000,
    seed=123,
)
result = optimizer.run()
```

## Saving runs

For post-run investigation, enable `record_snapshots=True` and save the result:

```python
result = optimizer.run()
result.save_json("shade_run.json")
result.save_pickle("shade_run.pkl")
```

Each snapshot stores the generation number, evaluation count, best solution, population size, optional population and fitness arrays, diversity values, and algorithm-specific extra state. For `SHADE` and `LSHADE`, the extra state includes the external archive and the parameter memories.

## Example

```python
from differential_evolution import (
    BinomialCrossover,
    DifferentialEvolution,
    ExponentialCrossover,
    Rand1,
    Best1,
    sphere_function,
)

optimizer = DifferentialEvolution(
    objective=sphere_function,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    population_size=20,
    mutation=Rand1(scale=0.8),
    crossover=BinomialCrossover(crossover_rate=0.9),
    max_generations=50,
    seed=123,
)
result = optimizer.run()

optimizer_alt = DifferentialEvolution(
    objective=sphere_function,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    population_size=20,
    mutation=Best1(scale=0.5),
    crossover=ExponentialCrossover(crossover_rate=0.8),
    max_generations=50,
    seed=123,
)
result_alt = optimizer_alt.run()
```

## Custom components

Any callable with the right signature can be injected.

```python
class MyMutation:
    def __call__(self, context):
        target = context.population[context.target_index]
        return [value * 0.5 for value in target]


class MyCrossover:
    def __call__(self, target_vector, donor_vector, rng):
        trial = list(target_vector)
        trial[0] = donor_vector[0]
        return trial
```

## Reproducibility

Use `seed=` or pass a custom RNG object. The optimizer does not use module-level global random state.

## Compatibility

The original constructor remains available:

```python
from main import DifferentialEvolution
```

That path now emits a deprecation warning and forwards into the new implementation. Details are in [MIGRATION.md](MIGRATION.md).

## Verification

- algorithm audit: [ALGORITHM_AUDIT.md](ALGORITHM_AUDIT.md)
- migration notes: [MIGRATION.md](MIGRATION.md)
- tests: `python3 -m unittest discover -s tests -v`
- examples: `python3 -m examples.basic_usage` and `python3 -m examples.custom_components`

### Review corrections and compatibility

See [review fixes](docs/review_fixes.md) for changes and the disposition of the
reported findings. Plain DE now clips to bounds by default; SHADE/L-SHADE use
parent-aware midpoint repair. Mutation and crossover components are copied per
optimizer; inspect adapted state on `optimizer.mutation` and `optimizer.crossover`.
`run()` is single-use. `result.success` means a usable objective value was found,
not numerical convergence; `result.message` reports the exhausted limit.

Sobol initialization uses a seeded digital shift by default. Use
`SobolInitializer(scramble=False)` for the original unshifted sequence. All
fixed-dimensional benchmarks, including Ackley, require exactly two coordinates.
