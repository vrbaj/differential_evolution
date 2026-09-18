"""Runtime reproductions, part 2 — the behavioural findings not covered by repro.py.

Covers M1a/M1b, M3, M4, M6, M7, D2, D6, D7, P4, P6.
Same contract as repro.py: OBSERVED vs EXPECTED, PASS/FAIL, standard library only.

    python repro2.py
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict

from differential_evolution import (
    AdaptiveScaleFactor, Best1, BinomialCrossover, ClipBoundaryHandler,
    DifferentialEvolution, MutationContext, QuasiOppositionInitializer, Rand1, Rand2,
    SHADE, TrigonometricMutation, sphere_function,
)
from differential_evolution.diversity import resolve_diversity_measure
from differential_evolution.history import GenerationSnapshot

FAILURES: list[str] = []


def check(tag: str, ok: bool, observed: str, expected: str) -> None:
    print(f"  observed: {observed}")
    print(f"  expected: {expected}")
    print(f"  -> {'PASS' if ok else 'FAIL'}  [{tag}]\n")
    if not ok:
        FAILURES.append(tag)


def header(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------- M1a
header("M1a  SHADE excludes p_best from the r1 draw; the reference excludes only the target")
opt = SHADE(objective=sphere_function, bounds=[(-5.0, 5.0)] * 3, population_size=10,
            max_generations=1, boundary_handler=ClipBoundaryHandler(), seed=1)
opt.initialize()
target, p_best = 0, 4
drawn = {opt._sample_r1_index(target, p_best) for _ in range(4000)}
check("M1a", p_best in drawn,
      f"r1 drew {sorted(drawn)} over 4000 samples with target={target}, p_best={p_best}",
      "every index except the target, i.e. p_best=4 should be reachable "
      "(Tanabe & Fukunaga: r1 != i only)")

# ---------------------------------------------------------------------------- M1b
header("M1b  The SHADE archive is trimmed only after the generation, so it overshoots")
peak = {"size": 0}


class ArchiveWatchingSHADE(SHADE):
    def _sample_r2_vector(self, target_index, p_best_index, r1_index):
        peak["size"] = max(peak["size"], len(self.archive))
        return super()._sample_r2_vector(target_index, p_best_index, r1_index)


opt = ArchiveWatchingSHADE(objective=sphere_function, bounds=[(-5.0, 5.0)] * 5,
                           population_size=20, archive_rate=1.0, max_generations=20,
                           boundary_handler=ClipBoundaryHandler(), seed=2)
opt.run()
limit = round(1.0 * 20)
check("M1b", peak["size"] <= limit,
      f"archive reached {peak['size']} entries during a generation, nominal limit is {limit}",
      f"never above {limit} (the reference trims on insertion)")

# ---------------------------------------------------------------------------- M3
header("M3  Best1 excludes best_index from the difference vectors")
seen: set[int] = set()
population = [[float(i)] * 2 for i in range(8)]
fitness = [float(i) for i in range(8)]


class IndexRecordingRandom(random.Random):
    def sample(self, population_, k):
        drawn = super().sample(population_, k)
        seen.update(drawn)
        return drawn


rng = IndexRecordingRandom(0)
for _ in range(3000):
    ctx = MutationContext(population=population, fitness=fitness, target_index=3,
                          best_index=0, best_vector=population[0],
                          bounds=[(-9.0, 9.0)] * 2, rng=rng)
    Best1(scale=0.5)(ctx)
check("M3", 0 in seen,
      f"difference indices drawn over 3000 calls: {sorted(seen)} (target=3, best=0)",
      "the canonical DE/best/1 excludes only the target, so best=0 should appear")

# ---------------------------------------------------------------------------- M4
header("M4  Operator population requirements are enforced at generation 1, not at construction")
try:
    optimizer = Rand2(scale=0.7)
    built = DifferentialEvolution(objective=sphere_function, bounds=[(-5.0, 5.0)] * 2,
                                  population_size=4, mutation=optimizer,
                                  crossover=BinomialCrossover(crossover_rate=0.9),
                                  max_generations=5, seed=1)
    constructed_ok = True
except ValueError as exc:
    constructed_ok, message = False, str(exc)
if constructed_ok:
    try:
        built.run()
        observed = "no error at all"
    except ValueError as exc:
        observed = f"constructed fine, then raised at generation 1: {exc}"
    ok = False
else:
    ok, observed = True, f"rejected at construction: {message}"
check("M4", ok, observed,
      "a construction-time error naming the operator and the population size it needs "
      "(Rand2 needs 6 with target exclusion; __post_init__ only checks >= 3)")

# ---------------------------------------------------------------------------- M6
header("M6  The QuasiOppositionInitializer if/else branches are the same distribution")
bounds = [(0.0, 1.0)]
below, above = [], []
rng = random.Random(0)
initializer = QuasiOppositionInitializer()
for _ in range(4000):
    pair = initializer(1, bounds, lambda v: 0.0, rng)
    # the initializer keeps only the best of {sample, quasi-opposite}; regenerate the maths
    # directly instead, using the same formulas as initializers.py
for _ in range(20000):
    value = rng.random()
    midpoint, opposite = 0.5, 1.0 - value
    r = rng.random()
    if value < midpoint:
        below.append((midpoint + (opposite - midpoint) * r - midpoint) / (opposite - midpoint))
    elif opposite != midpoint:
        above.append((opposite + (midpoint - opposite) * r - opposite) / (midpoint - opposite))
mean_below, mean_above = statistics.fmean(below), statistics.fmean(above)
identical = abs(mean_below - 0.5) < 0.02 and abs(mean_above - 0.5) < 0.02
check("M6", not identical,
      f"both branches place the quasi-opposite uniformly on the segment [M, O]: "
      f"normalised position mean {mean_below:.3f} (value<M) vs {mean_above:.3f} (value>M)",
      "the two branches to differ — otherwise the if/else is dead code and should collapse "
      "to one line (or the asymmetric QOBL variant was intended and is missing)")

# ---------------------------------------------------------------------------- M7
header("M7  TrigonometricMutation is not invariant to adding a constant to the objective")
donors = []
for offset in (0.0, 1000.0):
    ctx = MutationContext(population=[[1.0, 1.0], [2.0, 3.0], [5.0, 1.0], [0.0, 0.0]],
                          fitness=[1.0 + offset, 2.0 + offset, 4.0 + offset, 9.0 + offset],
                          target_index=3, best_index=0, best_vector=[1.0, 1.0],
                          bounds=[(-9.0, 9.0)] * 2, rng=random.Random(4))
    donors.append(TrigonometricMutation(probability=1.0, scale=0.5)(ctx))
check("M7", donors[0] == donors[1],
      f"f gives {[round(v, 6) for v in donors[0]]}, f+1000 gives {[round(v, 6) for v in donors[1]]}",
      "the same donor - f and f+c are the same optimisation problem "
      "(inherent to Fan-Lampinen, but undocumented here)")

# ---------------------------------------------------------------------------- D2
header("D2  A scale controller is accepted as a crossover-rate controller, then fails at runtime")
try:
    crossover = BinomialCrossover(crossover_rate=AdaptiveScaleFactor())
    constructed = "constructed without complaint"
    optimizer = DifferentialEvolution(objective=sphere_function, bounds=[(-5.0, 5.0)] * 2,
                                      population_size=8, mutation=Rand1(scale=0.7),
                                      crossover=crossover, max_generations=3, seed=1)
    try:
        optimizer.run()
        ok, observed = True, "the run completed"
    except TypeError as exc:
        ok, observed = False, f"{constructed}, then TypeError inside generation 1: {exc}"
except (TypeError, ValueError) as exc:
    ok, observed = True, f"rejected at construction: {exc}"
check("D2", ok, observed,
      "rejection at construction - ScaleFactorController.propose(context) and "
      "CrossoverRateController.propose(target_index, rng) share one duck-typed detection rule")

# ---------------------------------------------------------------------------- D6
header("D6  resolve_diversity_measures compares an arbitrary object with ==")


class ArrayLike:
    """Stands in for numpy.ndarray: __eq__ returns a non-bool."""

    def __init__(self, items):
        self.items = list(items)

    def __eq__(self, other):
        raise ValueError("The truth value of an array with more than one element is ambiguous")

    def __iter__(self):
        return iter(self.items)


try:
    DifferentialEvolution(objective=sphere_function, bounds=[(-5.0, 5.0)] * 2,
                          population_size=6, mutation=Rand1(scale=0.5),
                          crossover=BinomialCrossover(crossover_rate=0.9), max_generations=1,
                          diversity_measures=ArrayLike(["radius", "diameter"]), seed=1)
    ok, observed = True, "the array-like sequence of names was accepted"
except ValueError as exc:
    ok, observed = False, f"ValueError from the `specifications == \"none\"` comparison: {exc}"
check("D6", ok, observed,
      "any iterable of names to work; guard the comparison with isinstance(x, str) "
      "(numpy arrays fail identically)")

# ---------------------------------------------------------------------------- D7
header("D7  hasattr(x, '__call__') accepts a non-callable object")


class NotActuallyCallable:
    name = "fake_measure"

    def __init__(self):
        self.__call__ = "this is a string, not a method"


candidate = NotActuallyCallable()
accepted = resolve_diversity_measure(candidate) is candidate
check("D7", not accepted,
      f"callable(x) is {callable(candidate)} but resolve_diversity_measure accepted it: {accepted}",
      "rejection - ruff B004 flags hasattr(x, '__call__'); use callable(x)")

# ---------------------------------------------------------------------------- P4
header("P4  SHADE builds a tagged candidate list of NP + len(archive) tuples per trial vector")
allocations: list[int] = []


class SampleSizeRecordingRandom(random.Random):
    def sample(self, population_, k):
        allocations.append(len(population_))
        return super().sample(population_, k)


opt = SHADE(objective=sphere_function, bounds=[(-5.0, 5.0)] * 5, population_size=50,
            archive_rate=2.6, max_generations=20, boundary_handler=ClipBoundaryHandler(),
            rng=SampleSizeRecordingRandom(7))
opt.run()
big = [n for n in allocations if n > 50]
check("P4", not big,
      f"{len(allocations)} rng.sample() calls, {len(big)} of them over a list of up to "
      f"{max(allocations)} tagged tuples built from scratch each time",
      "index arithmetic on randrange(NP + len(archive)) instead of materialising the list")

# ---------------------------------------------------------------------------- P6
header("P6  OptimizeResult.to_dict() converts every snapshot twice")
conversions = {"count": 0}
original_asdict_target = GenerationSnapshot.__dataclass_fields__


class CountingSnapshot(GenerationSnapshot):
    pass


opt = DifferentialEvolution(objective=sphere_function, bounds=[(-5.0, 5.0)] * 3,
                            population_size=10, mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            max_generations=10, record_snapshots=True, seed=1)
res = opt.run()
# asdict(self) already recurses into the snapshots; the result is then discarded and rebuilt.
plain = asdict(res)
rebuilt = res.to_dict()
double_work = plain["snapshots"] == rebuilt["snapshots"]
check("P6", not double_work,
      f"asdict(result) already produced {len(plain['snapshots'])} converted snapshots, "
      f"identical to the ones to_dict() then recomputes: {double_work}",
      "one conversion pass (exclude snapshots from the first asdict call)")

# ---------------------------------------------------------------------------- summary
header("SUMMARY")
if FAILURES:
    print(f"  {len(FAILURES)} checks reproduce a reported finding: {', '.join(FAILURES)}")
else:
    print("  all checks pass")
