# Ready-to-paste GitHub issues

Thirteen self-contained issue bodies for the highest-impact findings, so they can be filed
individually rather than as one wall of text. Suggested labels in brackets.

These 13 are a subset. The full report covers **53 numbered findings** — see `REVIEW.md`,
and `REVIEW.md` §11 for a table stating exactly how each one was verified.

Every snippet below is also runnable as a batch: `python repro.py` covers issues 1-10,
`python repro2.py` covers the algorithm-fidelity findings not filed here,
`python audit.py` covers issues 11-12 plus the packaging and API-surface checks, and
`python verify_extra.py` covers issue 13 and the benchmark-formula sweep in issue 3. Each
prints `observed / expected / PASS|FAIL`, so they double as an acceptance checklist after
fixing.

> **Second-pass note.** A second independent agent re-cloned the same commit on a different
> machine and Python build, re-ran every script, re-read the source, and pulled the SHADE
> (CEC 2013) and L-SHADE (CEC 2014) papers. All 30 runtime assertions and all 13 repository
> checks reproduced. Two things changed: the `DirectedMutation` verdict in issue 10 was
> softened — a candidate primary source was found, but it could not be read, so whether its
> equation matches the implementation is still an open question for you — and issue 13 was
> added, because the SHADE/L-SHADE fidelity questions are now settled against the papers
> rather than merely suspected.

---

## Issue 1 — `TentInitializer` collapses the population onto the lower bound after ~53 coordinates
`[bug] [critical] [initializers]`

**Summary.** The tent map is iterated in double precision, once per emitted coordinate.
`x ← 2x` shifts the mantissa left, so after ~53 iterations the state is *exactly* `0.0`,
which is a fixed point of the map. Every coordinate emitted after that equals `lower`.

**Reproduce**

```python
import random
from differential_evolution import TentInitializer

pop = TentInitializer()(20, [(-5.0, 5.0)] * 10, lambda v: 0.0, random.Random(7))
print(pop[0])                                  # fine
print(pop[6])                                  # [-5.0] * 10
print(len({tuple(p) for p in pop}))            # 7   (out of 20)
flat = [c for ind in pop for c in ind]
print(sum(1 for c in flat if c == -5.0), "/", len(flat))   # 147 / 200
```

Across seeds 0–4 the chain reaches exact `0.0` at iteration 53 or 54, every time. Any run
with `population_size * len(bounds) > ~53` starts from a degenerate population where most
difference vectors are zero. No error, no warning — the run just silently under-performs.

**Two related problems in the same block**
* One scalar chain is threaded through the whole population, so individual *k+1* is a
  deterministic function of individual *k*. The docstring calls this "low-correlation
  initialization"; it is maximally correlated.
* A starting value of exactly `0.5` (or a chain reaching `1.0`) kills the map on iteration 1.

**Suggested fix.** Iterate the map on integers in `[0, 2**k)` rather than on floats, use an
independent chain per dimension (or per individual), and reject the fixed points. Regression
test: `len({tuple(p) for p in init(50, [(-5,5)]*30, ...)}) == 50`.

**Note.** `coverage` shows `initializers.py:124-135` (this function's body) is executed by
**zero** tests.

---

## Issue 2 — A single `inf` fitness poisons the SHADE/L-SHADE memories with `NaN` and stops the search
`[bug] [critical] [shade]`

**Summary.** `SHADE._update_memory` weights successful `F`/`CR` by objective improvement:

```python
total_improvement = sum(improvements)
weights = [improvement / total_improvement for improvement in improvements]
```

`_evaluate_vector` maps both `inf` and `NaN` objective values to `math.inf`, so a parent with
`f = inf` gives `improvement = inf`, hence `total_improvement = inf` and every weight is
`inf/inf = NaN`. `memory_f[k]` becomes `NaN`.

`_sample_scale_factor` then does `while scale_factor <= 0.0:` — and `NaN <= 0.0` is `False`,
so it exits immediately and returns `min(NaN, 1.0) == NaN`. Donors become all-`NaN`, trials
evaluate to `inf`, nothing is ever accepted again. With `memory_size = 6` all six slots are
poisoned within six generations, after which the optimizer burns the entire remaining budget
doing nothing — and still returns `success=True`.

**Reproduce**

```python
import math
from differential_evolution import SHADE, ClipBoundaryHandler

f = lambda x: math.inf if x[0] < 0.0 else sum(v * v for v in x)   # infeasible half-space
opt = SHADE(objective=f, bounds=[(-5., 5.)] * 5, population_size=20,
            max_generations=30, boundary_handler=ClipBoundaryHandler(), seed=3)
r = opt.run()

print(opt.memory_f)    # [nan, nan, nan, nan, nan, nan]
print(opt.memory_cr)   # [nan, nan, nan, nan, nan, nan]
print(r.success, "|", r.message)   # True | Optimization finished after reaching max_generations.
print([round(v, 3) for v in r.best_fitness_history])
# improves for 7 generations, then flat for the remaining 24
```

**Why it matters.** Returning `inf` (or `NaN`) for infeasible points is *the* standard way to
express constraints to a box-constrained global optimizer. This is not an exotic edge case.

**Suggested fix.**
1. Drop non-finite improvements from `S_F`/`S_CR`, or clamp with
   `improvement = min(f_parent, BIG) - min(f_trial, BIG)`.
2. Defensively guard `_sample_scale_factor` / `_sample_crossover_rate` with
   `math.isfinite(location)` so a poisoned memory can never reach a donor.
3. Add a test asserting no `NaN` ever appears in `memory_f` / `memory_cr` for an objective
   that returns `inf` on part of the domain.

---

## Issue 3 — The 2-D-only benchmark functions silently accept *n*-dimensional input
`[bug] [critical] [benchmarks]`

**Summary.** `beale`, `booth`, `matyas`, `himmelblau`, `bukin`, `mccormick`,
`three_hump_camel`, `ackley`, `goldstein_price`, `levi`, `easom`, `eggholder` and
`schaffer_n2` index `x[0]` and `x[1]` and ignore every further coordinate.

```python
import differential_evolution as de
de.ackley_function([0, 0] + [7.0] * 8)   # 0.0  -> reported as the global optimum
```

A full run:

```python
opt = DifferentialEvolution(objective=de.ackley_function, bounds=[(-5., 5.)] * 10,
                            population_size=30, mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            boundary_handler=ClipBoundaryHandler(),
                            max_generations=200, seed=5)
r = opt.run()
print(r.fun)                              # 0.0
print([round(v, 3) for v in r.x])
# [0.0, 0.0, -5.0, 5.0, -2.557, 5.0, 0.245, -2.172, 2.088, -1.61]
```

Anyone benchmarking DE variants on "10-D Ackley" gets `0.0` for every variant. For a library
whose purpose is comparing DE variants, this produces confidently wrong published numbers.

In the other direction the same functions raise a bare `IndexError: list index out of range`
for 1-D input.

**The formulas themselves are fine — this is only a missing guard.** All 15 exported
benchmarks were evaluated at their published global optima (`beale` at (3, 0.5), `booth` at
(1, 3), `himmelblau` at (3, 2), `bukin` at (−10, 1), `mccormick` −1.9132, `goldstein_price`
3.0 at (0, −1), `easom` −1 at (π, π), `eggholder` −959.6407 at (512, 404.2319), `levi`
Lévy N.13, and the rest): **zero mismatches**. So this is a cheap fix, not a rewrite.

**Suggested fix.** Raise `ValueError(f"... is defined for exactly 2 dimensions, got {len(x)}")`
in every fixed-dimension benchmark. Consider promoting `ackley` to its standard *n*-D form —
it currently sits in `__all__` next to the *n*-D `sphere` and `rastrigin`, which is what makes
the trap so easy to fall into. Ship the known optimum and recommended box alongside each
function and assert them in tests.

---

## Issue 4 — Aliased diversity measure names silently corrupt `diversity_history`
`[bug] [diversity]`

**Summary.** `_DIVERSITY_MEASURES` maps 14 names onto 8 classes (`"radius"` and
`"population_radius"` are the same class). The history dict is keyed by `measure.name`, which
de-duplicates, but `_record_diversity` iterates the **list** of resolved measures and appends
to the shared key.

```python
opt = DifferentialEvolution(objective=sphere_function, bounds=[(-5., 5.)] * 2,
                            population_size=10, mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            max_generations=5,
                            diversity_measures=["radius", "population_radius"], seed=1)
r = opt.run()
print(r.nit)                                            # 5
print(len(r.diversity_history["population_radius"]))    # 12, expected 6
print([round(v, 4) for v in r.diversity_history["population_radius"]])
# [6.0764, 6.0764, 6.2148, 6.2148, 3.8367, 3.8367, ...]
```

The same happens for two *different* custom measures that share a `name`, in which case two
series are interleaved into one key and cannot be separated.

**Suggested fix.** Raise on duplicate `measure.name` in `resolve_diversity_measures`, or key
the history by position and report names separately.

---

## Issue 5 — `result.success` and `result.message` are constants
`[bug] [api]`

`run()` always returns `success=True` and
`"Optimization finished after reaching max_generations."`, regardless of why it stopped.

```python
opt = DifferentialEvolution(..., max_generations=10_000, max_evaluations=100, seed=4)
r = opt.run()
print(r.nfev, r.nit)          # 100 9
print(r.success, r.message)   # True 'Optimization finished after reaching max_generations.'
```

Related: calling `run()` a second time is a silent no-op that still returns
`success=True` with the same `nit`/`nfev`.

Any code branching on `result.success` — the scipy convention this API imitates — is being
misled. Suggest a `status` enum plus a matching message covering `max_generations`,
`max_evaluations`, degenerate population, and future tolerance/stagnation criteria.

---

## Issue 6 — `CurrentTo*` operators draw two independent `F` values when `difference_scale=None`
`[bug] [mutation]`

`CurrentToBest1/2` and `CurrentToRand1/2` do:

```python
difference_scale_value = self.scale if self.difference_scale is None else self.difference_scale
difference_scale = self._resolve_scale(difference_scale_value, context)
```

`_resolve_scale` calls `controller.propose(...)` a second time. With `RandomizedScaleFactor`
that is a fresh draw, so the documented "reuse `scale`" default silently uses two different
`F` values in one donor formula (and consumes two RNG draws):

```
CurrentToBest1(scale=RandomizedScaleFactor(0.5, 1.0)), difference_scale=None
   propose() -> 0.567182
   propose() -> 0.923717
```

With `AdaptiveScaleFactor` the internal `_pending` cache masks it and both terms agree. **The
same operator behaves differently depending on which controller class is plugged in.**

**Suggested fix.** Resolve `scale` once and reuse the value when `difference_scale is None`.
Separately, make the `_pending` "one proposal per target per generation" caching an explicit,
documented part of the controller contract instead of an emergent side effect.

---

## Issue 7 — The default `NoBoundaryHandler` silently violates the declared `bounds`
`[bug] [api] [defaults]`

`bounds` is a required argument, which every user reads as a constraint. With the default
handler it constrains nothing:

```python
f = lambda x: sum((v - 20.0) ** 2 for v in x)   # unconstrained minimum at x = 20
box = [(-5., 5.)] * 5

# NoBoundaryHandler (library default) -> best x = [20.0, 20.0, 20.0, 20.0, 20.0]
# ClipBoundaryHandler                 -> best x = [ 5.0,  5.0,  5.0,  5.0,  5.0]
```

Five of five coordinates outside the declared box, `success=True`, no warning.

**Suggested fix.** Default to `ClipBoundaryHandler` (matching scipy, pymoo and the DE
literature), or warn once at construction when `NoBoundaryHandler` is combined with finite
bounds.

---

## Issue 8 — `SobolInitializer` ignores the RNG, so "independent runs" share one starting population
`[bug] [initializers] [reproducibility]`

```python
a = SobolInitializer()(8, box, obj, random.Random(1))
b = SobolInitializer()(8, box, obj, random.Random(999999))
a == b      # True
a[0]        # [-5.0, -5.0, -5.0]  -- the lower corner of the box
```

Sobol is deterministic and unscrambled, and `rng` is discarded. For a library whose purpose
is statistics over *N* independent runs, this invalidates the statistics: the runs are not
independent samples. The first point always lands exactly on the lower corner of the box.

**Suggested fix.** Owen or random-digit scrambling seeded from `rng` (this also removes the
corner point); at minimum a Cranley–Patterson random shift plus skipping index 0. Document
the 40-dimension and `2**30`-point limits.

**Not a bug (verified).** The underlying sequence itself is correct. Cross-checked against
`scipy.stats.qmc.Sobol`: it differs from scipy for D ≥ 3 only because scipy uses Joe–Kuo
direction numbers and this uses Bratley–Fox. The 1-D projections of the first 2^m points hit
every dyadic bin exactly once for all dimensions up to 40, and on 2-D adjacent-pair balance
at D=40 it has 19 failing pairs vs scipy's 22. The polynomial/degree tables match
Bratley–Fox exactly.

---

## Issue 9 — Constructing a second optimizer wipes the first one's adaptation state
`[bug] [api] [thread-safety]`

`DifferentialEvolution.__post_init__` calls `mutation.initialize(...)`, which resets the jDE
per-individual `F`/`CR` vectors. Reusing component objects across optimizers — which
`jde_rand_1_bin()` returning a reusable `JDEComponents` bundle actively invites — destroys
state mid-run:

```python
comp = jde_rand_1_bin(tau_f=1.0, tau_cr=1.0)
o1 = DifferentialEvolution(..., mutation=comp.mutation, crossover=comp.crossover, seed=1)
o1.initialize()
for _ in range(3): o1._step()
print([round(v, 3) for v in comp.mutation.scale.values()])
# [0.594, 0.123, 0.5, 0.346, 0.817, 0.978]

o2 = DifferentialEvolution(..., mutation=comp.mutation, crossover=comp.crossover, seed=2)
print([round(v, 3) for v in comp.mutation.scale.values()])
# [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]     <- o1 has not finished
```

This also makes the library thread-hostile: two optimizers run concurrently over shared
components corrupt each other's adaptation.

**Suggested fix.** Deep-copy components in `__post_init__`, or move per-run state into a
run-scoped object. Either way, document the ownership rule prominently.

---

## Issue 10 — `DirectedMutation`: scale-dependent coefficients, and silently disabled on any objective that can go ≤ 0
`[bug] [mutation] [documentation]`

**(a) The coefficient is dimensional.** `coefficient = (1 - f_best) / f_worse` puts a raw
objective value in a coefficient, so it depends on the objective's units *and* on an
arbitrary additive constant:

```
f values ~ 1e+0 : donor = [4, 4]
f values ~ 1e-3 : donor = [1003, 1003]
f values ~ 1e-6 : donor = [1e+06, 1e+06]
```

As a run converges (`f → 0`) the coefficient → `1/f` and the donor is flung an unbounded
distance away.

**(b) It silently is not directed mutation on half the shipped benchmarks.** The guard
`math.isfinite(v) and v > 0.0` fails for any individual with `f <= 0`, so the operator falls
back to DE/rand/1 with `F = 1.0`. That covers `easom` (min −1), `eggholder` (min −959.64),
`mccormick` (min −1.91) and `goldstein_price` — four of the library's own benchmarks.
Nothing in `OptimizeResult` records that the requested operator never ran.

**(c) The citation points at the wrong paper — but the right paper does exist.**
`ALGORITHM_AUDIT.md` cites Fan & Lampinen 2003 (DOI `10.1023/A:1024653025686`) for
`DirectedMutation`. That DOI is the *trigonometric* mutation paper, already cited for
`TrigonometricMutation` directly above, and two different operators sharing one DOI will be
spotted immediately in peer review.

Fan & Lampinen do appear to have a separate directed-mutation paper — *"A directed mutation
operation for the differential evolution algorithm"* — and there is also Fan, Lampinen &
Dulikravich, *"Improvements to Mutation Donor Formulation of Differential Evolution"*,
EUROGEN 2003. So the name and attribution in `mutation.py` are probably not invented.

**This is a lead, not a verified citation.** Neither candidate is indexed in Crossref and
neither full text was reachable, so it has *not* been confirmed that the paper's equation is
the `(1 − f_b)/f_w` form implemented here — and secondary sources describe Fan & Lampinen's
directed mutation as combining the standard and trigonometric operators, which does not
obviously match. `docs/algorithm_reference.rst` already says the "exact paper mapping should be
treated as requiring manual review", which is the same gap seen from the inside.

**Suggested fix.** Obtain the original and compare its equation with `mutation.py`. If it
matches, correct the DOI and keep the name; if it does not, rename the operator and present it
as a project-specific variant. Then fix (a) and (b) independently of the outcome: give the
coefficient a normalised, shift-invariant form — e.g. based on fitness *ranks*, or on
`(f_w − f_b) / (f_max − f_min)` — and record a fallback counter in the result metadata so a
silent fallback is at least observable.

---

## Issue 11 — Publication blockers: PyPI name taken, no LICENSE, README claims that fail after `pip install`
`[blocker] [packaging]`

**(a) The PyPI name is taken.**

```
pypi.org/pypi/differential-evolution  ->  v1.12.0
    "Differential Evolution Algorithm with OpenMDAO Driver"
```

`differential_evolution` normalises to the same name, so the project cannot publish under it.
Worth deciding early, because the name propagates into the docs, the DOI and the citation.
(The *import* name can stay `differential_evolution`; only the distribution name must change.)

**(b) There is no LICENSE file.** `pyproject.toml` declares `license = { text = "MIT" }` but
no `LICENSE`/`LICENSE.txt`/`COPYING` exists and the built wheel ships none. GitHub and PyPI
both show "no license", and JOSS/SoftwareX reject on this alone. Also, `{ text = ... }` is
deprecated under PEP 639 for `setuptools>=77` — use `license = "MIT"` plus
`license-files = ["LICENSE"]`.

**(c) The README's compatibility claim is false for installed users.** README says
`main.py`, `population_initialization.py` and `testing_functions.py` "remain as compatibility
shims", but `[tool.setuptools.packages.find] include = ["differential_evolution*"]` excludes
them. Verified against the built wheel — it contains only `differential_evolution/*.py`.
So `from main import DifferentialEvolution`, the documented migration path and the subject of
`MIGRATION.md`, works from a git checkout and `ImportError`s after `pip install`.

**(d) Absolute local paths in the published README.** Six links of the form
`[ALGORITHM_AUDIT.md](/home/honza/PycharmProjects/differential_evolution/ALGORITHM_AUDIT.md)`
in `README.md` and `ALGORITHM_AUDIT.md`. Broken on GitHub, and they go verbatim into the PyPI
long description.

**(e) Also missing:** `.github/workflows/` (no CI at all), `CITATION.cff`, `CHANGELOG.md`,
`CONTRIBUTING.md`, `py.typed`, `classifiers`, `keywords`, `[project.urls]`, author email, and
a `test` extra (the suite needs pytest but nothing declares it). `requires-python = ">=3.11"`
is asserted although nothing in the code needs 3.11.

---

## Issue 12 — `gpd_ll_function` unpickles a file on every objective evaluation
`[bug] [security] [benchmarks]`

```python
def gpd_ll_function(x, sample_path="gpd_sample"):
    ...
    with Path(sample_path).open("rb") as handle:
        data = pickle.load(handle)      # on EVERY evaluation
```

Three problems:

1. It re-opens and unpickles a file for every single objective call — catastrophic inside an
   optimizer loop.
2. `pickle.load` on a caller-supplied path is arbitrary code execution.
3. The default file `gpd_sample` does not exist in the repository, so the function cannot run
   at all. It is also absent from `__init__.__all__`, reachable only through the unpackaged
   `testing_functions.py` shim — i.e. effectively dead code.

Its `math.log` calls also have no domain guard, so it raises for `shape <= 0`. Combined with
the default `NoBoundaryHandler` (Issue 7) and the absence of any exception policy in
`_evaluate_vector`, that reliably kills runs with no partial result:

```
run() raised ZeroDivisionError: simulated model failure
evaluations completed before the failure: 39 | generations kept: 2
```

**Suggested fix.** Delete `gpd_ll_function`, or turn it into a class that loads the sample
once in `__init__` from a documented non-pickle format (`.npy`, CSV). Separately, give
`DifferentialEvolution` an `on_error` policy (`"raise"` / `"penalize"` / `"stop"`) and return
the partial `OptimizeResult` when a run is interrupted.

---

## Issue 13 — `SHADE` and `LSHADE` deviate from Tanabe & Fukunaga in five places, and the two classes cannot differ
`[bug] [shade] [fidelity]`

**Summary.** `LSHADE` inherits `_step`, `_sample_p_best_index` and `_update_memory` from
`SHADE` verbatim, so the two classes are forced to share every parameter-adaptation rule. The
papers do not share them. All five points below were checked against the primary sources —
R. Tanabe and A. Fukunaga, *Success-History Based Parameter Adaptation for Differential
Evolution*, CEC 2013 (**SHADE 2013**), and *Improving the search performance of SHADE using
linear population size reduction*, CEC 2014 (**L-SHADE 2014**).

**(a) `M_CR` uses the weighted arithmetic mean; L-SHADE requires the weighted Lehmer mean.**
SHADE 2013 splits the two rules — Eq. (17) `M_CR = mean_WA(S_CR)`, Eq. (18)
`M_F = mean_WL(S_F)`. L-SHADE 2014 collapses them into one Eq. (7) and says explicitly that
"`S` refers to either `S_CR` **or** `S_F`". The implementation is right for `SHADE` and wrong
for `LSHADE`, and because `_update_memory` is inherited, exactly one of them has to be wrong.

**(b) `p` is randomised; L-SHADE fixes it at 0.11.** The class uses SHADE 2013 Eq. (20),
`p_i = rand[2/NP, 0.2]`. L-SHADE 2014 Eq. (3) makes `p` a fixed hyper-parameter — "the top
`N × p` members" — swept over `{0.05, 0.06, …, 0.15}` in the paper's parameter table with
0.11 reported for `D = 30`.

**(c) `SHADE` carries the terminal `M_CR` value, which is an L-SHADE-only mechanism.**
`_update_memory` sets `memory_cr[k] = None` when `max(S_CR) == 0` and never updates that slot
again — L-SHADE 2014's absorbing `⊥` state (Alg. 1, lines 2-3). The word "terminal" appears
**zero** times in SHADE 2013. So the misattribution runs in both directions.

```python
opt = SHADE(objective=sphere_function, bounds=[(-5., 5.)] * 3, population_size=10,
            max_generations=1, boundary_handler=ClipBoundaryHandler(), seed=1)
opt.initialize()
opt._update_memory([0.5, 0.5], [0.0, 0.0], [1.0, 2.0])
print(opt.memory_cr)     # [None, 0.5, 0.5, 0.5, 0.5, 0.5]  <- L-SHADE's absorbing state
```

**(d) Three exclusion deviations in the index sampling.** Both papers say only: "The indices
`r1`, `r2` are randomly selected from `[1, N]` such that they differ from each other as well
as `i`", and `x_pbest` is "randomly selected from the top `N × p` members" with no exclusion.
The implementation additionally excludes `p_best_index` from `r1`; excludes `p_best_index`,
`r1` and the target from the population half of `P ∪ A` for `r2`; and excludes the target
from the p-best set itself:

```python
# target = 0, which is also the best individual, NP = 10
{opt._sample_p_best_index(0, list(range(10))) for _ in range(5000)}   # {1} — index 0 unreachable
```

**(e) Selection is strict; the papers accept ties.** `accepted = improvement > 0.0` means
`f(u) < f(x)`. Eq. (6) is identical in both papers and non-strict:
`x_{i,G+1} = u_{i,G}` if `f(u_{i,G}) ≤ f(x_{i,G})`. Ties are rejected and the tied parent is
not archived, so the population cannot drift across plateaus. Note that Tanabe treats survival
and success as two different tests (`≤` for survival, `<` for recording into `S_F`/`S_CR`);
the library collapses them into one.

**Also, the default `LSHADE()` is not L-SHADE.** LPSR is defined on the evaluation budget,
`N_{G+1} = round[((N_min − N_init)/MAX_NFE) · NFE + N_init]`, but `max_evaluations` defaults
to `None`, so `LinearPopulationReduction` falls back to generation progress:

```
LSHADE(population_size=100, max_generations=50, seed=1)   # max_evaluations is None
NP every 5 generations : [100, 90, 81, 71, 62, 52, 42, 33, 23, 14, 4]
total evaluations spent: 2748
```

Because the population shrinks, generations are not equal-cost, so the two schedules are not
reparametrisations of each other.

**Why it matters.** The planned paper will claim faithful jDE / SHADE / L-SHADE
implementations. A referee who knows these algorithms will check exactly these five rules, and
`docs/discrepancies.rst` currently records none of them.

**Suggested fix.** Override `_sample_p_best_index` and `_update_memory` in `LSHADE` (fixed
`p = 0.11`, `mean_WL` for `M_CR`); move the terminal-`M_CR` mechanism out of `SHADE` into
`LSHADE`; either align the exclusions and the `≤` selection with the papers or record each
deviation in `discrepancies.rst` with its rationale; require `max_evaluations` for `LSHADE`
(or warn when absent). The midpoint bound repair, L-SHADE 2014 Eq. (4)
`v_j = (x_min_j + x_j,i,G)/2`, needs the `BoundaryHandler` protocol widened first — see
`REVIEW.md` D3.

**Reproduce:** `python verify_extra.py`.
