"""Second-pass reproductions for the review of vrbaj/differential_evolution @ ed6ff80.

    conda create -n differential_evolution python=3.12 -y
    conda activate differential_evolution
    pip install -e .            # inside the repo checkout
    python verify_extra.py

Covers the four findings added by the independent second pass (M1d, M1e, M1f, M10) plus the
benchmark-formula sweep that backs the "the formulas are fine" note in C3 and section 9.

The two deviations settled from the literature in the second pass -- L-SHADE's fixed p = 0.11
and its weighted Lehmer mean for M_CR -- are not asserted here, because they are facts about
the papers rather than observable behaviour. The relevant equations are quoted in REVIEW.md
M2 and in ISSUES.md issue 13:

    SHADE 2013    Eq. (17) M_CR = mean_WA(S_CR)   Eq. (18) M_F = mean_WL(S_F)
                  Eq. (20) p_i = rand[2/NP, 0.2]
                  the word "terminal" does not occur anywhere in the paper
    L-SHADE 2014  Eq. (3)  x_pbest from the top N x p members, p fixed (0.11 for D = 30)
                  Eq. (4)  v_j = (x_min_j + x_j,i,G)/2   midpoint repair
                  Eq. (6)  x_{i,G+1} = u_{i,G} if f(u_{i,G}) <= f(x_{i,G})
                  Eq. (7)  mean_WL, where "S refers to either S_CR or S_F"
                  Alg. 1   M_CR gets the absorbing terminal value when max(S_CR) = 0

Only the standard library plus the package itself is required. Every check prints OBSERVED vs
EXPECTED and a PASS/FAIL verdict, so the same file can be re-run after the fixes to confirm
each deviation is gone.
"""
from __future__ import annotations

import inspect
import math

import differential_evolution as de
from differential_evolution import (
    ClipBoundaryHandler,
    LSHADE,
    SHADE,
    sphere_function,
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


def fresh_shade(dimensions: int = 3, population_size: int = 10, seed: int = 1) -> SHADE:
    optimizer = SHADE(
        objective=sphere_function,
        bounds=[(-5.0, 5.0)] * dimensions,
        population_size=population_size,
        max_generations=1,
        boundary_handler=ClipBoundaryHandler(),
        seed=seed,
    )
    optimizer.initialize()
    return optimizer


# --------------------------------------------------------------------------- M1d
header("M1d  p_best selection excludes the target; the reference does not")
# L-SHADE 2014, after Eq. (3): "individual x_pbest,G is randomly selected from the top
# N x p members in generation G" -- no exclusion of i.  Only r1 and r2 are constrained
# ("randomly selected from [1, N] such that they differ from each other as well as i").
optimizer = fresh_shade(dimensions=2, seed=2)
ranked_fitness = list(range(10))  # index 0 is the best individual
drawn = {optimizer._sample_p_best_index(0, ranked_fitness) for _ in range(5000)}
check(
    "M1d",
    0 in drawn,
    f"target = 0, which is also the best individual. p_best indices drawn over 5000 "
    f"samples: {sorted(drawn)}",
    "index 0 reachable, i.e. x_pbest == x_i allowed, so the first difference term may vanish",
)

# --------------------------------------------------------------------------- M1e
header("M1e  Selection is strict '<'; both reference papers use f(u) <= f(x)")
source = inspect.getsource(SHADE._step)
strict = "improvement > 0.0" in source


def staircase(vector: list[float]) -> float:
    """Piecewise-constant objective, so exact ties between parent and trial are common."""
    return float(sum(math.floor(abs(value)) for value in vector))


flat_counts = []
for seed in (1, 2):
    optimizer = SHADE(
        objective=staircase,
        bounds=[(-5.0, 5.0)] * 10,
        population_size=30,
        max_generations=60,
        boundary_handler=ClipBoundaryHandler(),
        seed=seed,
    )
    history = optimizer.run().best_fitness_history
    flat_counts.append(sum(1 for a, b in zip(history, history[1:]) if a == b))
check(
    "M1e",
    not strict,
    f"SHADE._step accepts on 'improvement > 0.0' (strict): {strict}; on a piecewise-constant "
    f"objective {flat_counts} of 60 generations make no progress",
    "acceptance on f(trial) <= f(target), per Eq. (6) in both papers, so tied trials survive "
    "and their parents enter the archive",
)

# --------------------------------------------------------------------------- M1f
header("M1f  SHADE carries the terminal CR value, an L-SHADE-only mechanism")
# L-SHADE 2014 Alg. 1 lines 2-3:  if M_CR,k = _|_ or max(S_CR) = 0 then M_CR,k+1 = _|_
# SHADE 2013 Eq. (17) is an unconditional weighted arithmetic mean; the paper has no
# absorbing state and never uses the word "terminal".
optimizer = fresh_shade(seed=1)
optimizer._update_memory([0.5, 0.5], [0.0, 0.0], [1.0, 2.0])
terminal_slots = sum(1 for value in optimizer.memory_cr if value is None)
check(
    "M1f",
    terminal_slots == 0,
    f"after one generation whose successes all used CR = 0, memory_cr = "
    f"{optimizer.memory_cr} ({terminal_slots} permanently disabled slot(s))",
    "no absorbing state in SHADE: the slot should hold mean_WA(S_CR); the terminal value "
    "belongs to LSHADE only",
)

# --------------------------------------------------------------------------- M10
header("M10  Default LSHADE() applies LPSR on generation progress, not on evaluations")
# L-SHADE 2014 Eq. (10): N_{G+1} = round[((N_min - N_init)/MAX_NFE) * NFE + N_init].
# LSHADE installs LinearPopulationReduction by default, but max_evaluations defaults to
# None, so _normalized_progress falls back to generation / max_generations.
optimizer = LSHADE(
    objective=sphere_function,
    bounds=[(-5.0, 5.0)] * 10,
    population_size=100,
    max_generations=50,
    boundary_handler=ClipBoundaryHandler(),
    seed=1,
)
result = optimizer.run()
reduced_without_budget = (
    optimizer.max_evaluations is None
    and result.population_size_history[-1] <= optimizer.min_population_size
)
check(
    "M10",
    not reduced_without_budget,
    f"max_evaluations = {optimizer.max_evaluations}; NP every 5 generations = "
    f"{result.population_size_history[::5]}; total evaluations = {result.nfev}",
    "LPSR requires an evaluation budget: either demand max_evaluations for LSHADE, or warn "
    "that generation-based reduction is a project-specific variant",
)

# ----------------------------------------------------------------------- C3 / s.9
header("C3+  Are the benchmark formulas themselves correct?")
KNOWN_OPTIMA = {
    "sphere_function": ([0.0, 0.0], 0.0),
    "rastrigin_function": ([0.0, 0.0], 0.0),
    "beale_function": ([3.0, 0.5], 0.0),
    "booth_function": ([1.0, 3.0], 0.0),
    "matyas_function": ([0.0, 0.0], 0.0),
    "himmelblau_function": ([3.0, 2.0], 0.0),
    "bukin_function": ([-10.0, 1.0], 0.0),
    "mccormick_function": ([-0.54719, -1.54719], -1.9133),
    "three_hump_camel_function": ([0.0, 0.0], 0.0),
    "ackley_function": ([0.0, 0.0], 0.0),
    "goldstein_price_function": ([0.0, -1.0], 3.0),
    "levi_function": ([1.0, 1.0], 0.0),
    "easom_function": ([math.pi, math.pi], -1.0),
    "eggholder_function": ([512.0, 404.2319], -959.6407),
    "schaffer_n2_function": ([0.0, 0.0], 0.0),
}
mismatches = []
for name, (point, expected) in KNOWN_OPTIMA.items():
    value = getattr(de, name)(point)
    if abs(value - expected) >= 1e-3:
        mismatches.append(f"{name}: got {value!r}, published optimum {expected!r}")
check(
    "C3+",
    not mismatches,
    f"{len(KNOWN_OPTIMA)} functions evaluated at their published global optima, "
    f"{len(mismatches)} mismatches" + (f": {mismatches}" if mismatches else ""),
    "zero mismatches -- confirming C3 is a missing dimension guard, not wrong mathematics",
)

# ------------------------------------------------------------------------ summary
header("SUMMARY")
if FAILURES:
    print(f"  {len(FAILURES)} checks reproduce a reported deviation: {', '.join(FAILURES)}")
else:
    print("  all checks pass - every reported deviation appears to be resolved")
print()
print("  Not asserted here (facts about the papers, not about the code):")
print("    * LSHADE should use a fixed p = 0.11   -- L-SHADE 2014 Eq. (3) + parameter table")
print("    * LSHADE should use mean_WL for M_CR   -- L-SHADE 2014 Eq. (7)")
print("    * LSHADE needs midpoint bound repair   -- L-SHADE 2014 Eq. (4), blocked by D3")
print("  See REVIEW.md M2 and ISSUES.md issue 13.")
