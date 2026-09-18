"""Self-contained reproductions for the review of vrbaj/differential_evolution @ ed6ff80.

    conda create -n differential_evolution python=3.12 -y
    conda activate differential_evolution
    pip install -e .            # inside the repo checkout
    python repro.py

Only the standard library plus the package itself is required.
Every check prints OBSERVED vs EXPECTED and a PASS/FAIL verdict, so the same file can be
re-run after the fixes to confirm each defect is gone.
"""
from __future__ import annotations

import json
import math
import os
import random
import tempfile

import differential_evolution as de
from differential_evolution import (
    AdaptiveCrossoverRate, BinomialCrossover, ClipBoundaryHandler, CurrentToBest1,
    DifferentialEvolution, DirectedMutation, LSHADE, MutationContext, NoBoundaryHandler,
    OppositionInitializer, QuasiOppositionInitializer, Rand1, RandomInitializer,
    RandomizedScaleFactor, SHADE, SobolInitializer, TentInitializer, jde_rand_1_bin,
    sphere_function,
)
from differential_evolution.population_schedules import (
    HyperbolicTangentPopulationReduction, PopulationScheduleState,
)

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


# ---------------------------------------------------------------------------- C1
header("C1  TentInitializer collapses onto the lower bound")
pop = TentInitializer()(20, [(-5.0, 5.0)] * 10, lambda v: 0.0, random.Random(7))
flat = [c for ind in pop for c in ind]
at_lower = sum(1 for c in flat if c == -5.0)
distinct = len({tuple(p) for p in pop})
check("C1", at_lower == 0 and distinct == 20,
      f"{at_lower}/{len(flat)} coordinates exactly at the lower bound; "
      f"{distinct}/20 distinct individuals",
      "no coordinate pinned to the bound; 20 distinct individuals")

# ---------------------------------------------------------------------------- C2
header("C2  One inf fitness poisons the SHADE memories with NaN")
opt = SHADE(objective=lambda x: math.inf if x[0] < 0.0 else sum(v * v for v in x),
            bounds=[(-5.0, 5.0)] * 5, population_size=20, max_generations=30,
            boundary_handler=ClipBoundaryHandler(), seed=3)
res = opt.run()
nan_f = sum(1 for v in opt.memory_f if math.isnan(v))
nan_cr = sum(1 for v in opt.memory_cr if v is not None and math.isnan(v))
check("C2", nan_f == 0 and nan_cr == 0,
      f"memory_f has {nan_f}/6 NaN, memory_cr has {nan_cr}/6 NaN; success={res.success}",
      "no NaN in either memory")

# ---------------------------------------------------------------------------- C3
header("C3  2-D-only benchmarks silently ignore extra coordinates")
try:
    value = de.ackley_function([0.0, 0.0] + [7.0] * 8)
    ok, observed = False, f"ackley_function([0,0]+[7]*8) returned {value} with no error"
except ValueError as exc:
    ok, observed = True, f"raised ValueError: {exc}"
check("C3", ok, observed, "ValueError naming the required dimension")

# ---------------------------------------------------------------------------- C4
header("C4  Aliased diversity names corrupt diversity_history")
opt = DifferentialEvolution(objective=sphere_function, bounds=[(-5.0, 5.0)] * 2,
                            population_size=10, mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            max_generations=5,
                            diversity_measures=["radius", "population_radius"], seed=1)
res = opt.run()
recorded = len(res.diversity_history["population_radius"])
check("C4", recorded == res.nit + 1,
      f"len(history['population_radius']) == {recorded} for nit == {res.nit}",
      f"{res.nit + 1} values, or a duplicate-name error")

# ---------------------------------------------------------------------------- H1
header("H1  A legitimate -inf objective value is flipped to +inf")
res = DifferentialEvolution(objective=lambda x: -math.inf if abs(x[0]) < 0.5 else x[0] ** 2,
                            bounds=[(-5.0, 5.0)], population_size=10, mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            max_generations=20, seed=5).run()
check("H1", res.fun == -math.inf, f"best fun == {res.fun}", "-inf, the true minimum")

# ---------------------------------------------------------------------------- H2
header("H2  CurrentTo* draws two independent F values when difference_scale is None")
draws: list[float] = []


class TracingScaleFactor(RandomizedScaleFactor):
    def propose(self, context):
        value = super().propose(context)
        draws.append(value)
        return value


context = MutationContext(
    population=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    fitness=[4.0, 3.0, 2.0, 1.0, 0.0], target_index=0, best_index=4,
    best_vector=[4.0, 4.0], bounds=[(-9.0, 9.0)] * 2, rng=random.Random(1))
CurrentToBest1(scale=TracingScaleFactor(0.5, 1.0))(context)
check("H2", len(set(draws)) == 1,
      f"the controller produced {len(draws)} values: {[round(v, 6) for v in draws]}",
      "one value, reused for both difference terms")

# ---------------------------------------------------------------------------- H3
header("H3  success / message are constants")
res = DifferentialEvolution(objective=sphere_function, bounds=[(-5.0, 5.0)] * 2,
                            population_size=10, mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            max_generations=10_000, max_evaluations=100, seed=4).run()
check("H3", "evaluation" in res.message.lower(),
      f"stopped on max_evaluations (nfev={res.nfev}, nit={res.nit}); message == {res.message!r}",
      "a message naming the evaluation budget as the stopping reason")

# ---------------------------------------------------------------------------- H4
header("H4  max_evaluations is ignored during initialization")
res = DifferentialEvolution(objective=sphere_function, bounds=[(-5.0, 5.0)] * 2,
                            population_size=50, mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            max_generations=10, max_evaluations=5, seed=1).run()
check("H4", res.nfev <= 5, f"nfev == {res.nfev} for a budget of 5",
      "nfev <= 5, or a clear error at construction")

# ---------------------------------------------------------------------------- H5
header("H5  An exception in the objective destroys the run")
state = {"calls": 0}


def flaky(x):
    state["calls"] += 1
    if state["calls"] == 40:
        raise ZeroDivisionError("simulated model failure")
    return sum(v * v for v in x)


opt = DifferentialEvolution(objective=flaky, bounds=[(-5.0, 5.0)] * 2, population_size=10,
                            mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            max_generations=50, seed=3)
try:
    opt.run()
    ok, observed = True, "run() completed"
except ZeroDivisionError as exc:
    ok, observed = False, (f"run() propagated {type(exc).__name__}; {opt.nfev} evaluations "
                           "lost and no result returned")
check("H5", ok, observed, "a configurable on_error policy, or a partial OptimizeResult")

# ---------------------------------------------------------------------------- H6
header("H6  DirectedMutation coefficients scale like 1/f")
magnitudes = []
for scale in (1.0, 1e-3, 1e-6):
    ctx = MutationContext(population=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                          fitness=[10 * scale, 4 * scale, 2 * scale, 1 * scale],
                          target_index=0, best_index=3, best_vector=[4.0, 4.0],
                          bounds=[(-5.0, 5.0)] * 2, rng=random.Random(0))
    magnitudes.append(max(abs(v) for v in DirectedMutation()(ctx)))
check("H6", max(magnitudes) < 1e3,
      "max |donor| for f ~ 1e0 / 1e-3 / 1e-6 == " + " / ".join(f"{m:.4g}" for m in magnitudes),
      "donor magnitude independent of the absolute scale of the objective")

# ---------------------------------------------------------------------------- H7
header("H7  SobolInitializer ignores the RNG")
box = [(-5.0, 5.0)] * 3
first = SobolInitializer()(8, box, lambda v: 0.0, random.Random(1))
second = SobolInitializer()(8, box, lambda v: 0.0, random.Random(999_999))
check("H7", first != second,
      f"two different seeds give identical populations: {first == second}; "
      f"first point == {first[0]}",
      "different seeds give different scrambled populations, and no point on the box corner")

# ---------------------------------------------------------------------------- H8
header("H8  Constructing a second optimizer wipes the first one's adaptation state")
components = jde_rand_1_bin(tau_f=1.0, tau_cr=1.0)
shared = dict(objective=sphere_function, bounds=[(-5.0, 5.0)] * 2, population_size=6,
              mutation=components.mutation, crossover=components.crossover, max_generations=5)
running = DifferentialEvolution(**shared, seed=1)
running.initialize()
for _ in range(3):
    running._step()
before = list(components.mutation.scale.values())
DifferentialEvolution(**shared, seed=2)
after = list(components.mutation.scale.values())
check("H8", before == after,
      f"F before == {[round(v, 3) for v in before]}, after == {[round(v, 3) for v in after]}",
      "an unfinished optimizer keeps its own adaptation state")

# ---------------------------------------------------------------------------- H9
header("H9  The default boundary handler does not enforce bounds")
res = DifferentialEvolution(objective=lambda x: sum((v - 20.0) ** 2 for v in x),
                            bounds=[(-5.0, 5.0)] * 5, population_size=30,
                            mutation=Rand1(scale=0.8),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            boundary_handler=NoBoundaryHandler(),
                            max_generations=200, seed=1).run()
outside = sum(1 for v in res.x if not -5.0 <= v <= 5.0)
check("H9", outside == 0,
      f"{outside}/5 coordinates outside the declared box; x == {[round(v, 2) for v in res.x]}",
      "no coordinate outside, or a warning that bounds are advisory by default")

# ---------------------------------------------------------------------------- H10
header("H10  save_json writes non-RFC-8259 JSON")
res = DifferentialEvolution(objective=lambda x: math.inf if x[0] > 0 else sum(v * v for v in x),
                            bounds=[(-5.0, 5.0)] * 2, population_size=8,
                            mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            max_generations=3, seed=2).run()
path = os.path.join(tempfile.mkdtemp(), "result.json")
res.save_json(path)
with open(path, encoding="utf-8") as handle:
    text = handle.read()


def reject_constant(token):
    raise ValueError(f"non-standard token {token!r}")


try:
    json.loads(text, parse_constant=reject_constant)
    ok, observed = True, "a strict JSON parser accepted the file"
except ValueError as exc:
    ok, observed = False, f"a strict JSON parser rejected the file: {exc}"
check("H10", ok, observed, "valid RFC-8259 JSON (non-finite values as null or a sentinel)")

# ---------------------------------------------------------------------------- M5
header("M5  HyperbolicTangentPopulationReduction contradicts its docstring")
schedule = HyperbolicTangentPopulationReduction(min_population_size=10)
current = 100
trace = {}
for generation in range(0, 101, 10):
    current = schedule.target_size(PopulationScheduleState(
        generation=generation, max_generations=100, evaluations=0, max_evaluations=None,
        initial_population_size=100, current_population_size=current, min_population_size=10))
    trace[generation / 100] = current
done_by_half = (100 - trace[0.5]) / 90
check("M5", done_by_half < 0.5,
      f"{done_by_half:.0%} of the reduction is already complete at half the run; trace == {trace}",
      "under 50%, per the docstring 'slowly early in the run and more aggressively later'")

# ---------------------------------------------------------------------------- M8
header("M8  run() twice is a silent no-op")
opt = DifferentialEvolution(objective=sphere_function, bounds=[(-5.0, 5.0)] * 2,
                            population_size=10, mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            max_generations=20, seed=1)
one, two = opt.run(), opt.run()
check("M8", (one.nit, one.nfev) != (two.nit, two.nfev),
      f"the second run() returned the same nit={two.nit} nfev={two.nfev}, success={two.success}",
      "an error, a reset, or an explicit resume API")

header("M8b  population_size, a user-supplied argument, is mutated in place")
opt = LSHADE(objective=sphere_function, bounds=[(-5.0, 5.0)] * 2, population_size=40,
             max_generations=50, min_population_size=4, seed=1)
opt.run()
check("M8b", opt.population_size == 40,
      f"opt.population_size == {opt.population_size} after run(), constructed with 40",
      "40 - configuration and mutable run state should not share a field")

# ---------------------------------------------------------------------------- M9
header("M9  Initialization evaluation accounting differs between initializers")
counts = {}
for initializer, label in ((RandomInitializer(), "RandomInitializer"),
                           (SobolInitializer(), "SobolInitializer"),
                           (TentInitializer(), "TentInitializer"),
                           (OppositionInitializer(), "OppositionInitializer"),
                           (QuasiOppositionInitializer(), "QuasiOppositionInitializer")):
    counts[label] = DifferentialEvolution(
        objective=sphere_function, bounds=[(-5.0, 5.0)] * 2, population_size=20,
        mutation=Rand1(scale=0.7), crossover=BinomialCrossover(crossover_rate=0.9),
        initializer=initializer, max_generations=10, seed=1).run().nfev
check("M9", len(set(counts.values())) == 1,
      f"nfev after 10 generations with NP=20: {counts}",
      "one value for every initializer, or documented per-initializer accounting")

# ---------------------------------------------------------------------------- D5
header("D5  Crossover objects are unusable outside the optimizer")
try:
    BinomialCrossover(crossover_rate=AdaptiveCrossoverRate())(
        [1.0, 2.0], [3.0, 4.0], random.Random(0))
    ok, observed = True, "the direct call succeeded"
except AttributeError as exc:
    ok, observed = False, f"AttributeError: {exc}"
check("D5", ok, observed, "a class-level default for _active_target_index")

# ---------------------------------------------------------------------------- summary
header("SUMMARY")
if FAILURES:
    print(f"  {len(FAILURES)} checks reproduce a reported defect: {', '.join(FAILURES)}")
else:
    print("  all checks pass - every reported defect appears to be fixed")
