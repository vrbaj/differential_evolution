# Independent adversarial review — `vrbaj/differential_evolution`

**Commit reviewed:** `ed6ff80` ("Initial commit: current differential evolution implementation")
**Reviewer:** Claude Opus 5 (Claude Code), no relation to the Codex refactor
**Environment:** fresh conda env `differential_evolution`, Python 3.12.14, `pip install -e .`,
Windows 11. Cross-checks against `scipy 1.x` (`scipy.stats.qmc.Sobol`,
`scipy.optimize.differential_evolution`), `numpy`, `pytest`, `ruff 0.16`, `mypy 2.3`, `coverage 7.16`.

**Revision note — second independent pass.** After this report was written, a second
independent agent re-cloned `ed6ff80` on a different machine and Python build (3.12.7),
re-ran all three scripts, re-read the source module by module, and pulled the two primary
sources — Tanabe & Fukunaga, *Success-History Based Parameter Adaptation for Differential
Evolution*, CEC 2013 (**SHADE 2013**), and *Improving the search performance of SHADE using
linear population size reduction*, CEC 2014 (**L-SHADE 2014**) — to settle the three
literature questions §11 flagged as uncertain.

* **All 30 runtime assertions and all 13 repository checks reproduced.** So did the numbers:
  64 tests, 88 % coverage (`initializers.py` 68 %, same uncovered ranges), mypy 56 errors in
  9 files, ruff 22 findings with the same composition, the PyPI name taken at v1.12.0, and
  the performance measurements within run-to-run variance (8.6× for P1, 19.7× for P3,
  16.6 µs/evaluation of overhead).
* **Two of the three uncertain literature claims were confirmed** from the papers (M2).
  The third — the `DirectedMutation` citation — was only *partly* advanced: a candidate
  source was identified but never read, so it is still open. See H6.
* **Five findings were added** — M1d, M1e, M1f, M10, and a benchmark-formula check that came
  out clean (C3, §9). They are marked **[2nd pass]** and are reproduced by
  `verify_extra.py`.
* **One fix suggestion was wrong** and has been corrected in place (M5).

---

The brief was "be vicious", so this report leads with what is broken and buries the
compliments at the end. Everything below was **executed**, not eyeballed; each finding
carries a reproduction you can paste into a REPL. Line references are to the reviewed commit.

---

## 0. Executive summary

The refactor is real: the architecture is clean, the component boundaries are sensible, the
docs build warning-free under `-W`, and L-SHADE genuinely converges to ~1e-48 on a 10-D sphere.
That is a much better starting point than most hobby optimizer code.

But the library is **not** ready for PyPI or for a software-journal submission, for three
independent reasons:

1. **Four defects silently produce wrong scientific results** — not crashes, not warnings.
   A user gets a number, publishes it, and it is meaningless. Details in §1.
2. **The PyPI name `differential-evolution` is already taken** (v1.12.0, "Differential
   Evolution Algorithm with OpenMDAO Driver"), and there is **no LICENSE file** in the
   repository despite `license = { text = "MIT" }`. Both are hard blockers for the
   publication plan. §7.
3. **The paper's central claim would be "more functionality and customisation than the two
   existing DE libraries"**, but the extension points are duck-typed `object` (56 mypy errors),
   the declared `Protocol`s are decorative, there is no CI, no benchmark suite, no
   reference-value regression tests, and the pure-Python core costs ~17-23 µs of framework
   overhead per objective evaluation. A referee will attack all four. §5, §6, §8.

**Finding count and evidence status.** 53 numbered findings (C1-4, H1-10, M1-10, D1-8, P1-6, T1-4, K1-8), 63 if the sub-items inside M1, D8, K6 and K8 are counted separately. Of these, **35 are reproduced by the accompanying scripts** as pass/fail assertions (`repro.py` 19, `repro2.py` 11, `verify_extra.py` 5) and **14 more by the repository/packaging audit** (`audit.py`); 6 are measurements with no pass/fail threshold (P1-P6, printed by `audit.py`); the remainder are literature-fidelity or absence claims. Every claim's evidence is itemised in §11, and the three that the first pass could not settle from the literature were resolved in the second pass — see the revision note above.

---

## 1. Critical — silently wrong results

### C1. `TentInitializer` collapses the entire population onto the lower bound

`initializers.py:120-137`. The tent map is iterated **once per emitted coordinate** in
double precision. `x ← 2x` is a left bit-shift of the mantissa; after ~53 iterations the
state is *exactly* `0.0`, and `0.0` is a fixed point of the map. Every coordinate emitted
after that is `lower`.

```python
from differential_evolution import TentInitializer
import random
pop = TentInitializer()(20, [(-5.0, 5.0)]*10, lambda v: 0.0, random.Random(7))
print(pop[6])    # [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]
print(len({tuple(p) for p in pop}))   # 7  (13 of 20 individuals are the same point)
```

Measured: for `NP=20, D=10` (200 coordinates), **147/200 coordinates are exactly `-5.0`** and
only 7 of 20 individuals are distinct. Across seeds 0–4 the state hits exact zero at
iteration 53 or 54, every time. Any run with `NP × D > ~53` — i.e. essentially every real
run — starts from a degenerate population in which most difference vectors are the zero
vector. DE cannot recover from that; it does not crash, it just quietly under-performs.

Two further problems in the same 15 lines:

* One scalar chain is threaded through the whole population, so individual *k+1* is a
  deterministic function of individual *k*. The docstring calls this "low-correlation
  initialization"; it is the opposite. Chaotic-initialisation papers use one chain per
  dimension, or per individual.
* If `rng.random()` returns exactly `0.5` (or the chain reaches `1.0`), the map dies on
  iteration 1.

**Fix:** iterate the map in a domain where it does not degenerate (e.g. keep the state as an
integer in `[0, 2^k)` and use the integer tent map), reseed per dimension, and reject the
fixed points `{0, 1}` and the pre-images of `0.5`. Add a regression test asserting
`len(set(map(tuple, pop))) == NP` for `NP=50, D=30`.

---

### C2. One `inf` fitness poisons the whole SHADE/L-SHADE memory with `NaN`

`shade.py:167-192`. `_update_memory` computes

```python
total_improvement = sum(improvements)
weights = [improvement / total_improvement for improvement in improvements]
```

`improvements[k] = f(parent) - f(trial)`. If any parent has `f = inf` — which is exactly what
`optimizer.py:_evaluate_vector` returns for `inf` **and for `NaN`**, i.e. the standard way to
express "infeasible" or "the simulation diverged" — then that improvement is `inf`,
`total_improvement` is `inf`, and every weight becomes `inf/inf = NaN`.

`memory_f[k]` becomes `NaN`. Then `_sample_scale_factor` does
`while scale_factor <= 0.0:` — and `NaN <= 0.0` is `False`, so the loop exits immediately
and returns `min(NaN, 1.0) = NaN`. Every donor vector from that memory slot is all-`NaN`,
every trial evaluates to `inf`, and nothing is ever accepted again. With `memory_size=6`
the poison spreads to all six slots in six generations, after which **the optimizer burns
the remaining budget doing literally nothing**, and still reports `success=True`.

```python
import math
from differential_evolution import SHADE, ClipBoundaryHandler
f = lambda x: math.inf if x[0] < 0.0 else sum(v*v for v in x)   # an infeasible half-space
opt = SHADE(objective=f, bounds=[(-5.,5.)]*5, population_size=20,
            max_generations=30, boundary_handler=ClipBoundaryHandler(), seed=3)
r = opt.run()
print(opt.memory_f)   # [nan, nan, nan, nan, nan, nan]
print(opt.memory_cr)  # [nan, nan, nan, nan, nan, nan]
print(r.success, r.message)   # True  'Optimization finished after reaching max_generations.'
```

The observed `best_fitness_history` improves for exactly 7 generations and is then flat for
the remaining 24 — the signature of total memory death.

This is the single most damaging bug in the library, because penalty-method constraint
handling (`return inf` outside the feasible region) is the *normal* way people use a
box-constrained global optimizer.

**Fix:** clamp the improvement weights. The reference implementation works on finite
differences only; the standard remedy is
`improvement = min(f_parent, LARGE) - min(f_trial, LARGE)` or to drop non-finite
improvements from `S_F`/`S_CR` entirely. Additionally, guard `_sample_scale_factor` with an
explicit `math.isfinite` check so a `NaN` memory can never leak into a donor, and add a
`NaN`/`inf` assertion to the SHADE tests.

---

### C3. Thirteen 2-D benchmarks silently accept *n*-dimensional input

`benchmarks.py`. `beale`, `booth`, `matyas`, `himmelblau`, `bukin`, `mccormick`,
`three_hump_camel`, `ackley`, `goldstein_price`, `levi`, `easom`, `eggholder`,
`schaffer_n2` all index `x[0]` and `x[1]` and **ignore everything else**.

```python
import differential_evolution as de
de.ackley_function([0, 0] + [7.0]*8)   # -> 0.0   "global optimum"
```

A full run makes it concrete:

```
10-D "Ackley" run  ->  fun = 0.0
returned x = [0.0, 0.0, -5.0, 5.0, -2.557, 5.0, 0.245, -2.172, 2.088, -1.61]
```

Machine-checked: **13 of the 15 exported benchmark functions return an identical value for 2-D and 5-D input**, i.e. silently discard the extra coordinates (only `sphere_function` and `rastrigin_function` are genuinely *n*-D). A user benchmarking DE variants on "10-D Ackley" gets `0.0` for every variant and concludes they are all perfect. For a library whose selling point is *benchmarking* DE variants, this
is a correctness trap of the worst kind: it produces a plausible number.

In the other direction the same functions raise a bare `IndexError: list index out of range`
for 1-D input, with no indication that the function is 2-D only.

**[2nd pass] The formulas themselves are all correct.** All 15 exported benchmarks were
evaluated at their published global optima — `sphere`, `rastrigin`, `beale` (3, 0.5),
`booth` (1, 3), `matyas`, `himmelblau` (3, 2), `bukin` (−10, 1), `mccormick` (−1.9132),
`three_hump_camel`, `ackley`, `goldstein_price` (3.0 at (0, −1)), `levi` (Lévy N.13),
`easom` (−1 at (π, π)), `eggholder` (−959.6407 at (512, 404.2319)) and `schaffer_n2` —
**zero mismatches**. `gpd_ll_function`'s GPD reparametrisation is algebraically correct too
(`log(θ/ξ) = −log σ` for `θ = ξ/σ`). That is good news for the cost of this
fix: C3 is a missing guard clause, not wrong mathematics.


**Fix:** every fixed-dimension benchmark should raise
`ValueError(f"{name} is defined for exactly 2 dimensions, got {len(x)}")`. Better: promote
`ackley` and `rastrigin`/`sphere` to their published *n*-D forms (Ackley's *n*-D definition
is standard and would remove the inconsistency that `rastrigin` and `sphere` are *n*-D while
`ackley` is not, despite all three being exported side by side). Ship the known global
optimum and recommended box as data next to each function, and assert them in tests.

---

### C4. Aliased or duplicate diversity measures silently corrupt the recorded history

`diversity.py:_DIVERSITY_MEASURES` maps 14 names onto 8 classes (`"radius"` and
`"population_radius"` are the same class). `optimizer.py:104-106` builds the history dict
keyed by `measure.name`, which de-duplicates; `_record_diversity` then iterates over the
**list** of resolved measures and appends to that shared key.

```python
opt = DifferentialEvolution(..., diversity_measures=["radius", "population_radius"], seed=1)
r = opt.run()
r.nit                                     # 5
len(r.diversity_history["population_radius"])   # 12, not 6
# [6.076, 6.076, 6.215, 6.215, 3.837, 3.837, ...]  each value duplicated
```

The series is silently twice as long as the generation count, with values interleaved. The
same happens for two *different* custom measures that happen to share a `name`, in which case
the two series are interleaved and indistinguishable. Nothing warns.

**Fix:** validate uniqueness of `measure.name` in `resolve_diversity_measures` and raise on
collision; or key the history by position and expose names separately.

---

## 2. High severity

### H1. A legitimate `-inf` objective value is flipped to `+inf`

`optimizer.py:311-317` returns `math.inf` when `math.isinf(value)` — including for `-inf`.
For minimisation, `-inf` is the *best possible* value; the optimizer scores it as the worst.

```python
opt = DifferentialEvolution(objective=lambda x: -math.inf if abs(x[0])<0.5 else x[0]**2, ...)
opt.run().fun     # 0.2529...  — the true optimum was found and then discarded
```

The docstring says "non-finite return values are treated as `math.inf`", so it is documented,
but documenting a sign error does not make it correct. Map `-inf` to `-inf` (or to
`-sys.float_info.max`), and keep the `+inf` / `NaN` → `+inf` convention.

### H2. `CurrentTo*` operators draw *two independent* scale factors when `difference_scale=None`

`mutation.py:213-215` (and the three sibling classes):

```python
difference_scale_value = self.scale if self.difference_scale is None else self.difference_scale
difference_scale = self._resolve_scale(difference_scale_value, context)
```

`_resolve_scale` calls `controller.propose(...)` again. With `RandomizedScaleFactor` that is
a fresh uniform draw, so the "reuse `scale`" default silently uses two different `F` values
inside one donor formula (and consumes two RNG draws, perturbing the stream):

```
CurrentToBest1(scale=RandomizedScaleFactor(0.5, 1.0)) with difference_scale=None
   propose() -> 0.567182
   propose() -> 0.923717
```

With `AdaptiveScaleFactor` the `_pending` cache masks it and both terms agree. **The same
operator therefore behaves differently depending on which controller class you plug in** —
the exact kind of hidden coupling a composable library exists to prevent. Worse, an operator
that legitimately *wants* two independent dithers cannot get them under jDE.

**Fix:** resolve `scale` once per call and reuse the value when `difference_scale is None`;
make the `_pending` caching an explicit, documented part of the controller contract rather
than an emergent property.

### H3. `success` and `message` are constants; there is no stopping-criterion reporting

`optimizer.py:157-160` always sets `success=True` and
`"Optimization finished after reaching max_generations."`, whatever actually happened.

```
stopped by max_evaluations:  nfev=100/100  nit=9
success = True   message = 'Optimization finished after reaching max_generations.'
```

Any code that branches on `result.success` — the scipy convention this API is clearly
imitating — is being lied to. Add a `status`/`message` set covering `max_generations`,
`max_evaluations`, degenerate population, and (once they exist) tolerance/stagnation
criteria.

### H4. `max_evaluations` is not honoured during initialization

The budget is checked per target inside `_step`, but `initialize()` evaluates the whole
population unconditionally.

```
budget = 5, population_size = 50  ->  nfev = 50   (10x over budget)
```

For `OppositionInitializer` it is 3×NP before generation 1 even starts. Check the budget in
`initialize()` too, or document that `max_evaluations` is a post-initialisation budget.

### H5. An exception in the objective destroys the whole run

`_evaluate_vector` catches `NaN`/`inf` but not exceptions. A single `ValueError` from
`math.log`, a failed FEM solve, or a network timeout aborts `run()` with no result object:

```
run() raised ZeroDivisionError: simulated model failure
evaluations completed before the failure: 39 | generations kept: 2
```

For a library aimed at expensive real-world objectives (the author's own `gpd_ll_function`
will raise `math.log` domain errors the moment the population leaves the feasible region —
which the default `NoBoundaryHandler` guarantees), this loses multi-hour runs. Offer an
`on_error` policy (`"raise"` / `"penalize"` / `"stop"`) and return the partial
`OptimizeResult` on interruption.

### H6. `DirectedMutation` is scale-dependent, explodes near the optimum, and is silently disabled on half the shipped benchmarks

`mutation.py:441-461`. `coefficient = (1 - f_best) / f_worse` puts a raw, *dimensional*
objective value in a coefficient:

```
f values ~ 1e+0 : donor = [4, 4]
f values ~ 1e-3 : donor = [1003, 1003]
f values ~ 1e-6 : donor = [1e+06, 1e+06]
```

As a run converges, `f → 0` and the coefficient → `1/f`, so the donor is flung an unbounded
distance from the population. The operator is also **not invariant to adding a constant to
the objective**, i.e. `f` and `f + c` — the same optimisation problem — give different
donors.

And the guard `math.isfinite(v) and v > 0.0` means that on any function that can take
non-positive values the operator **silently degrades to DE/rand/1 with F = 1.0**. That
includes `easom` (min −1), `eggholder` (min −959.64), `mccormick` (min −1.91) and
`goldstein_price`… i.e. four of the library's own benchmarks. Nothing in the result object
records that the requested operator never ran.

**The citation is wrong; a candidate source exists but is unread. [2nd pass correction.]**
`ALGORITHM_AUDIT.md` cites Fan & Lampinen 2003 (DOI `10.1023/A:1024653025686`) for
`DirectedMutation` — that is the *trigonometric* mutation paper, already cited for
`TrigonometricMutation` a few lines above, and a referee will catch a duplicated DOI on two
different operators immediately. The first pass concluded that no primary source could be
identified and suggested renaming the operator as project-specific; that was too pessimistic.
Fan & Lampinen do have a separate directed-mutation paper, *"A directed mutation operation for
the differential evolution algorithm"*, and there is also Fan, Lampinen & Dulikravich,
*"Improvements to Mutation Donor Formulation of Differential Evolution"*, EUROGEN 2003. So
the attribution in the repository is probably not invented.

**But this is a lead, not a verified citation, and it does not close the question.** The second
pass established only that a paper with that title by those authors exists. It did **not** read
the paper and did **not** verify that the paper's equation is the `(1 - f_b)/f_w` form
implemented in `mutation.py`. Neither candidate is indexed in Crossref, the full text was not
reachable, and secondary sources describe Fan & Lampinen's directed mutation as *combining the
standard and trigonometric mutation operators* — which does not obviously match the formula
here. Note also that `docs/algorithm_reference.rst` already says of this operator that "the
exact paper mapping should be treated as requiring manual review", so the repository itself
flags the same gap.

**What to do:** obtain the original and check the equation against `mutation.py` yourself. If
it matches, fix the DOI and keep the name. If it does not, the first pass's advice stands —
rename the operator and present it as a project-specific variant. Either way, do not cite the
directed-mutation paper on the strength of this review alone.

The scale-dependence (a) and silent-fallback (b) problems above are independent of the
citation and still need fixing either way: a normalised, shift-invariant coefficient based on
fitness *ranks*, or on `(f_w − f_b) / (f_max − f_min)`, plus a fallback counter in the result
metadata so a silent fallback is at least observable.

### H7. `SobolInitializer` ignores the RNG — every "independent run" starts from the same population

Sobol is deterministic and unscrambled, and the `rng` argument is discarded:

```python
SobolInitializer()(8, box, obj, random.Random(1)) == SobolInitializer()(8, box, obj, random.Random(999999))
# True
```

For a library whose purpose is producing statistics over *N* independent runs, this
invalidates the statistics: all *N* runs share an identical starting population, so the runs
are not independent samples and the variance estimate is wrong. The first point is
`(0,0,…,0)`, which maps to the **lower corner of the box** — a systematically bad and
identical individual in every single run.

**Fix:** implement Owen or random-digit scrambling seeded from `rng` (this also solves the
corner problem), or at minimum apply a random Cranley–Patterson shift, and skip the zero
point. Also raise a clear error above 40 dimensions instead of relying on the table size,
and document the `2**30`-point cap.

*(Credit where due: the underlying sequence itself is correct — see §9.)*

### H8. Sharing components between optimizers silently wipes adaptation state

`DifferentialEvolution.__post_init__` calls `mutation.initialize(...)`, which resets the jDE
`_values` vector. Constructing a *second* optimizer that reuses the same component objects
therefore destroys the *first* one's state mid-run:

```
o1's adapted F after 3 generations : [0.594, 0.123, 0.5, 0.346, 0.817, 0.978]
right after `o2 = DifferentialEvolution(...)`: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
```

Sharing components looks natural — `jde_rand_1_bin()` returns a `JDEComponents` bundle that
invites reuse across a benchmark sweep — and nothing warns. This also makes the library
**thread-hostile**: running two optimizers concurrently over shared components corrupts both.
Either deep-copy components in `__post_init__`, or store per-run state in a run-scoped object
keyed by the optimizer instance, and document the ownership rule loudly either way.

### H9. The default `boundary_handler=NoBoundaryHandler` silently violates the declared bounds

`bounds` is a **required** positional-ish argument, which every user will read as a
constraint. It is not:

```
NoBoundaryHandler (library default): best x = [20.0, 20.0, 20.0, 20.0, 20.0]   # box is (-5, 5)
ClipBoundaryHandler                : best x = [ 5.0,  5.0,  5.0,  5.0,  5.0]
```

5 of 5 coordinates outside the declared box, no warning, `success=True`. Make
`ClipBoundaryHandler` the default (matching scipy, pymoo and the DE literature), or at
minimum warn once at construction when `NoBoundaryHandler` is combined with finite bounds.

### H10. `OptimizeResult.save_json()` writes invalid JSON

`json.dumps` emits the non-standard `Infinity` token for `inf` fitness values, which any
`inf` in the population produces:

```
'Infinity' literal present in file: True
strict JSON parser rejects it: non-standard token 'Infinity'
```

The file is not RFC-8259 JSON. R's `jsonlite`, Go's `encoding/json`, and most JS parsers
reject it. Serialise non-finite values as `null` or as a sentinel string, and document the
choice.

---

## 3. Medium — algorithm fidelity vs. the cited references

These are the ones a reviewer of a SoftwareX / JOSS / *Journal of Open Source Software*-style
paper will grep for, because the paper will claim "faithful implementations of jDE, SHADE
and L-SHADE".

### M1. SHADE deviations from Tanabe & Fukunaga (2013)

Equation numbers below are from the primary sources, which were read directly (see the
revision note): **SHADE 2013** and **L-SHADE 2014**.

* **M1a.** `_sample_r1_index` (`shade.py:139-146`) excludes **`p_best_index`** as well as the
  target. Both papers say only: "The indices `r1`, `r2` are randomly selected from `[1, N]`
  such that they differ from each other as well as `i`." `p_best` is not excluded. The same
  applies to `_sample_r2_vector`, which additionally excludes `r1` *and* the target from the
  population half of `P ∪ A`.
* **M1b.** The archive is appended to inside the generation loop but `_resize_archive()` runs
  only **after** the generation, so the archive can transiently reach ~2× its nominal size and
  those extra entries are eligible as `r2` donors within the same generation. The reference
  trims on insertion: "Whenever the size of the archive exceeds `|A|`, randomly selected
  elements are deleted to make space for the newly inserted elements."
* **M1c.** `_sample_r2_vector` reads `self.population` rather than the loop's
  `population_snapshot`. They happen to be equal today because `self.population` is only
  rebound after the loop — but this is a landmine: any future in-loop update silently changes
  the algorithm.
* **M1d [2nd pass].** `_sample_p_best_index` also **excludes the target from the p-best set**:
  `filtered = [index for index in top_indices if index != target_index]`. The reference does
  not — "individual `x_pbest,G` is randomly selected from the top `N × p` members in
  generation `G`" — so `x_pbest = x_i` is allowed and the first difference term is allowed to
  vanish. This is a third exclusion deviation on top of M1a. Machine-checked: with `NP = 10`
  and the target also being the best individual, index `0` is never drawn in 5 000 samples.
* **M1e [2nd pass].** Selection is **strict**. `shade.py` uses `accepted = improvement > 0.0`,
  i.e. `f(u) < f(x)`. Eq. (6) is identical in both papers and is non-strict:
  `x_{i,G+1} = u_{i,G}` if `f(u_{i,G}) ≤ f(x_{i,G})`. Ties are therefore rejected and the tied
  parent is not archived, so the population cannot drift across plateaus. The optimizer
  docstring documents strict selection for plain DE, but `SHADE`/`LSHADE` inherit it and
  `docs/discrepancies.rst` does not mention it. Note that in Tanabe's formulation survival and
  success are two different tests — `≤` for survival, `<` for recording into `S_F`/`S_CR` —
  and the library collapses them into one.
* **M1f [2nd pass].** `SHADE` implements the **terminal `M_CR` value**, which is an
  **L-SHADE-only** mechanism. `_update_memory` sets `memory_cr[k] = None` when
  `max(S_CR) == 0` and then never updates that slot again — exactly L-SHADE 2014's absorbing
  `⊥` state (Alg. 1, lines 2-3). The word "terminal" appears **zero** times in SHADE 2013,
  whose memory update is Eq. (17) `M_CR = mean_WA(S_CR)`, with no absorbing state. So the
  misattribution runs in **both** directions: `LSHADE` is missing L-SHADE features (M2) while
  `SHADE` has one that does not belong to it.

None of M1a-M1f is listed in `docs/discrepancies.rst`.


### M2. `LSHADE` is L-SHADE-flavoured SHADE, not L-SHADE

**[2nd pass: both literature questions flagged here are now settled against the papers, and
both of the original suspicions were correct.]**

`LSHADE` inherits `_step`, `_sample_p_best_index` and `_update_memory` verbatim, so:

* **`p` is randomised where it should be fixed. Confirmed.** The class uses SHADE's
  `p_i = rand[2/NP, 0.2]` — SHADE 2013 Eq. (20), correct there. L-SHADE 2014 Eq. (3) makes `p`
  a fixed hyper-parameter ("individual `x_pbest,G` is randomly selected from the top `N × p`
  members"), tuned over `{0.05, 0.06, …, 0.15}` in the paper's parameter table with
  **`p = 0.11`** the reported setting for `D = 30`.
* **`M_CR` uses the weighted arithmetic mean where it should use the weighted Lehmer mean.
  Confirmed.** SHADE 2013 splits the two rules: Eq. (17) `M_CR = mean_WA(S_CR)` and Eq. (18)
  `M_F = mean_WL(S_F)`. L-SHADE 2014 collapses them into a single Eq. (7) and states
  explicitly that "`S` refers to either `S_CR` **or** `S_F`" — so **both** memories use
  `mean_WL`. The implementation is therefore right for `SHADE` and wrong for `LSHADE`; and
  because `_update_memory` is inherited verbatim, the two classes *cannot* differ, so exactly
  one of them has to be wrong.
* **The midpoint bound repair is missing and cannot be added as a component.** L-SHADE 2014
  Eq. (4), inherited from JADE, is `v_j = (x_min_j + x_j,i,G)/2` when `v_j < x_min_j` and
  `(x_max_j + x_j,i,G)/2` when `v_j > x_max_j`. It needs the parent vector, and the
  `BoundaryHandler` protocol only receives `(trial, bounds, rng)`. Without it, published
  L-SHADE numbers are not reproducible. This is a **protocol design problem, not a missing
  class** — see D3. (One detail that is *not* a problem: the paper repairs the donor `v`
  before crossover while the library repairs the trial `u` after it. For all three shipped
  handlers the two are equivalent, because only coordinates inherited from `v` can be out of
  bounds.)

`ALGORITHM_AUDIT.md` currently states the memory-update rule once, for both classes. Split
it, state the `p` difference explicitly, and record M1d-M1f in `discrepancies.rst`.


### M3. `Best*` and `CurrentToBest*` exclude `best_index` from the difference vectors

A defensible choice, but it is a deviation from the canonical definitions (which exclude only
`i`), it changes the search distribution, and it silently raises the minimum viable
population size. It is not listed in `docs/discrepancies.rst`.

### M4. Operator population requirements are enforced too late and are undocumented

`__post_init__` only checks `population_size >= 3`. `Rand2` needs 6, `Best2` needs 5 with the
exclusions above, `SHADE` needs 4. The failure arrives in generation 1:

```
Rand2 with NP=4 -> ValueError: Population too small for the requested mutation strategy and exclusions.
```

Give each operator a `required_population_size` property and validate it at construction,
naming the operator and the required size.

### M5. `HyperbolicTangentPopulationReduction`'s docstring contradicts its behaviour

Docstring: *"Population size decreases slowly early in the run and more aggressively later."*
Measured with `NP_init=100, NP_min=10`:

| progress | tanh NP | linear NP |
|---:|---:|---:|
| 0.2 | 93 | 82 |
| 0.3 | 68 | 73 |
| 0.4 | 31 | 64 |
| 0.5 | **14** | 55 |
| 0.7 | 10 | 37 |
| 1.0 | 10 | 10 |

**96 % of the reduction happens in the first half of the run, and the last 30 % is
completely flat.** The defaults `start=-3, end=6` are asymmetric and undocumented; with
`end=6`, `tanh` has saturated by progress ≈ 0.5. Either fix the defaults or fix the docstring.

**[2nd pass correction.]** The symmetric `start=-3, end=3` suggested in the first pass does
*not* give the documented shape either — it produces a symmetric sigmoid,
`[100, 99, 98, 93, 79, 55, 31, 17, 12, 11, 10]`, i.e. slow-fast-slow. To match the docstring's
"slowly early in the run and more aggressively later" the mapping has to stay on the convex
half of `tanh`: `start=-3, end=0` gives `[100, 100, 99, 98, 96, 92, 85, 75, 59, 36, 10]`.

For a paper, this schedule needs either a citation or an empirical justification — right now
it is an unexplained magic curve with the wrong description.

### M6. The `QuasiOppositionInitializer` branch is a no-op

`initializers.py:186-196`:

```python
if value < midpoint:  quasi = midpoint + (opposite - midpoint) * rng.random()
else:                 quasi = opposite + (midpoint - opposite) * rng.random()
```

Both branches sample uniformly on the segment `[M, O]`; the distributions are identical. The
`if` buys nothing but a reader's confusion (and one wasted comparison per coordinate).
Collapse to one line, or — if the intent was the asymmetric QOBL variant — fix it to match
the intent.

### M7. `TrigonometricMutation` is not shift-invariant, and this is undocumented

The Fan–Lampinen weights use `|f|`, so `f` and `f + c` give different donors:

```
f + 0    : donor = [-0.047619, 1.952381]
f + 1000 : donor = [ 2.660348, 1.667332]
```

This is inherent to the original operator, not a bug — but it means the operator's behaviour
depends on an arbitrary additive constant in the user's objective, and a library that
advertises it as a first-class strategy owes users a warning in the docstring.

### M8. `run()` called twice is a silent no-op that reports success

```
first run:  nit=20 nfev=210 fun=0.0002087
second run: nit=20 nfev=210 fun=0.0002087    <- no work done, success=True
```

Either raise, or reset, or expose an explicit `resume()`. Related: `population_size` is a
user-supplied dataclass field that the optimizer **mutates in place** — after an L-SHADE run,
`opt.population_size` is `4`, not the `40` the user passed. Configuration and state should
not share a field.

### M9. Initialisation evaluation accounting is inconsistent across initializers

Measured `nfev` after 10 generations with `NP=20` (expected 220):

| initializer | nfev |
|---|---:|
| `RandomInitializer` | 220 |
| `SobolInitializer` | 220 |
| `TentInitializer` | 220 |
| `OppositionInitializer` | **260** |
| `QuasiOppositionInitializer` | **260** |

OBL/QOBL spend 3×NP (2×NP to rank the doubled pool via `list.sort(key=objective)`, then NP
again because the optimizer re-evaluates the survivors). Two of those NP evaluations are pure
waste — the initializer already knows the values and simply throws them away. Comparing
initializers under a fixed `max_generations` budget is therefore not a fair comparison, and
this is exactly the comparison a paper about initialization strategies would make.

**Fix:** let the initializer protocol optionally return `(population, fitness)` so the
optimizer can skip re-evaluation; document the remaining 2×NP as the intrinsic cost of OBL.

---

### M10 [2nd pass]. The default `LSHADE()` applies LPSR on generation progress, not on evaluations

L-SHADE's linear population size reduction is defined on the *evaluation* budget:
`N_{G+1} = round[((N_min − N_init)/MAX_NFE) · NFE + N_init]`. `LSHADE.__post_init__` installs
`LinearPopulationReduction` by default, but `max_evaluations` defaults to `None`, so
`_normalized_progress` silently falls back to `generation / max_generations`.

```
LSHADE(population_size=100, max_generations=50, seed=1)   # max_evaluations is None
NP every 5 generations : [100, 90, 81, 71, 62, 52, 42, 33, 23, 14, 4]
total evaluations spent: 2748
```

So the out-of-the-box `LSHADE` shrinks to `N_min` after 50 generations having spent 2 748
evaluations, on a curve that is linear in the wrong variable. The two schedules are not
reparametrisations of each other: because the population is shrinking, generations are not
equal-cost. The fallback is documented in `LinearPopulationReduction`'s docstring, but not at
the `LSHADE` class, which is where the default actually takes effect — so the algorithm a
reader gets from `LSHADE(objective=..., bounds=..., population_size=..., max_generations=...)`
is not the published one.

**Fix:** require `max_evaluations` for `LSHADE` (or warn loudly when it is absent), and state
in the docs that LPSR without an evaluation budget is a project-specific variant.


## 4. Design and API

### D1. The `Protocol`s are decorative; everything is `object`

`ScaleFactorController`, `CrossoverOperator`, `BoundaryHandler`, `PopulationInitializer`,
`DiversityMeasure` and `PopulationSchedule` are all declared as `Protocol`s — and then
**never used in a single annotation**. `DifferentialEvolution` declares
`mutation: object`, `crossover: object`, `initializer: object`, `boundary_handler: object`,
`rng: object`, `diversity_measures: object`.

Consequence: `mypy --ignore-missing-imports differential_evolution` reports **56 errors in
9 files** (**63 in 10 files** once `--disallow-untyped-defs` is added), essentially all of the form
`"object" has no attribute "sample" / "random" / "propose" / "initialize"`. Users of the
library get zero IDE completion and zero type checking on the extension points that are the
library's entire value proposition.

**Fix:** annotate with the Protocols, mark them `@runtime_checkable` where useful, ship a
`py.typed` marker, and add `mypy` to CI. This is a half-day of work with a large payoff for
a paper that claims superior customisability.

### D2. Two incompatible `propose()` signatures share one duck-typed detection rule

`ScaleFactorController.propose(context)` vs `CrossoverRateController.propose(target_index, rng)`.
Both `BaseMutation._scale_controllers` and `BaseCrossover._rate_controllers` detect
controllers with the *same* test — `hasattr(initialize) and hasattr(propose) and hasattr(commit)`.
So `BinomialCrossover(crossover_rate=AdaptiveScaleFactor())` type-checks, constructs fine, and
blows up with a `TypeError` deep inside generation 1. Unify the signature (pass the context to
both) or make the two protocols structurally distinguishable and validate at construction.

### D3. The `BoundaryHandler` protocol cannot express the standard repair rules

`__call__(trial_vector, bounds, rng)` has no access to the parent/target vector, so the two
most important repair strategies in the DE literature are unimplementable as components:

* **midpoint / bounce-back** `x = (x_parent + bound) / 2` — used by the *reference* L-SHADE
  and by most CEC-competition entries;
* **reflection into the box**;
* projection toward the base vector.

Only `none`, `clip` and `random-reset` exist. For a library that wants to be *the* extensible
DE library, this is the most consequential missing extension point. Widen the protocol to
`(trial, target, bounds, rng)` before 1.0, because it is a breaking change afterwards.

### D4. The component lifecycle is only half-declared, and mutation has no protocol at all

The optimizer probes four optional hooks with `hasattr` — `initialize`, `commit`, `resize`,
`set_target_index` — scattered across `optimizer.py` and `shade.py`. Machine-checked against
the Protocol class bodies:

| Protocol | declared methods |
|---|---|
| `scales.ScaleFactorController` | `initialize`, `propose`, `commit`, `resize` |
| `crossovers.CrossoverOperator` | `__call__` only |
| `boundaries.BoundaryHandler` | `__call__` only |
| `initializers.PopulationInitializer` | `__call__` only |
| `diversity.DiversityMeasure` | `__call__` only |
| `population_schedules.PopulationSchedule` | `target_size` only |
| *the mutation operator* | **no Protocol exists** |

So `ScaleFactorController` does document its lifecycle properly — credit for that. But the
crossover protocol declares none of the four hooks the optimizer actually calls on crossover
objects, `set_target_index` is declared in **no** Protocol anywhere, and the single most
important extension point — the mutation operator — has no protocol at all, only the
`MutationContext` dataclass and an informal `BaseMutation` base class.

A user writing a custom stateful mutation has no way to discover the lifecycle except by
reading `optimizer._step`. Define a `MutationOperator` protocol plus a `StatefulComponent`
protocol with no-op defaults, and document the order
(`initialize → [set_target_index → propose → commit]* → resize`).

### D5. Components are unusable outside the optimizer

```python
BinomialCrossover(crossover_rate=AdaptiveCrossoverRate())([1.,2.], [3.,4.], random.Random(0))
# AttributeError: 'BinomialCrossover' object has no attribute '_active_target_index'
```

`_active_target_index` is created in `initialize()`, which only the optimizer calls. Give it
a class-level default of `0`.

### D6. `resolve_diversity_measures` compares an arbitrary object with `==`

`diversity.py:275`: `if specifications == "none":`. With a numpy array of names this raises
`ValueError: The truth value of an array with more than one element is ambiguous`. The magic
`"none"` string is also undocumented and inconsistent with `None`, and `["none"]` is not
handled. Use `isinstance(specifications, str) and specifications == "none"`, or drop the
magic string.

### D7. `hasattr(x, "__call__")` instead of `callable(x)`

`diversity.py:258` — flagged by ruff as `B004`; unreliable for objects with a `__call__`
attribute that is not a bound method.

### D8. Missing capabilities a "full-featured DE library" needs

The paper's argument is "the two existing DE libraries offer limited functionality". Referees
will check the list. Currently absent:

* **stopping criteria** beyond `max_generations` / `max_evaluations` — no tolerance,
  no stagnation counter, no target-value stop, no wall-clock limit. (The audit's "Remaining
  limitations" acknowledges this.)
* **callbacks / `yield`-based iteration** — no way to observe or intervene per generation
  without subclassing `_step`.
* **parallel / vectorised objective evaluation** — the single most requested feature for
  expensive objectives; `scipy` has `workers=`, `pymoo` has `Problem.evaluate` on matrices.
  The current one-vector-at-a-time protocol makes this impossible to add without an API break.
* **constraint handling** — no feasibility rules, no ε-constraint, no penalty helper.
* **maximisation** — minimisation only, so users hand-negate and then hit H1.
* **integer / categorical / mixed variables**.
* **restart strategies**, **island models**, **CoDE / EPSDE / SaDE strategy pools**.
* **JADE** — the direct ancestor of SHADE, and a conspicuous omission next to jDE and SHADE.

Pick the subset you actually want to claim, and design the protocols for it *now*
(especially batch evaluation and stopping criteria), because retrofitting them is a 1.0 break.

---

## 5. Performance

Pure Python, no numpy. That is a legitimate design choice (zero-dependency install), but it
must be measured and stated, because a referee will benchmark it.

**Measured, 20-D Rastrigin, 30 000 evaluations, 3 runs, same machine:**

| | wall clock | median best `f` |
|---|---:|---:|
| this library, DE/rand/1/bin | 0.57 s | 114.7 |
| `scipy.optimize.differential_evolution`, `rand1bin` | 0.26 s | 115.6 |

Same search quality, **2.2× the wall clock on a cheap objective**. Framework overhead
isolated (objective replaced by a constant): **17-23 µs per evaluation** for plain DE
(run-to-run variance), **~34 µs** for SHADE. For an objective costing < ~100 µs, the framework dominates.

### P1. `sample_distinct_indices` is the #1 hotspot and rebuilds a set per element

`mutation.py:44`:

```python
available = [index for index in range(population_size) if index not in set(excluded)]
```

`set(excluded)` is inside the comprehension, so it is constructed **once per candidate
index** — 100 set constructions per call at `NP=100`. It is the top entry in `tottime` for a
plain run (0.28 s of ~1.5 s). Rejection sampling instead of materialising the full index
list:

```
current implementation    6.9 - 7.1 us per call
rejection sampling        0.80 - 0.82 us per call    (8.4 - 8.9x faster)
```

### P2. `dataclasses.fields()` is called twice per trial vector

`BaseMutation.commit` and `BaseCrossover.commit` both call `_scale_controllers()` /
`_rate_controllers()`, which call `dataclasses.fields(self)` and re-run the `hasattr` triple
test on every field — **for every target, every generation**, even when `scale` is a plain
float. Measured: exactly 2.0 `dataclasses.fields()` calls per trial vector - 10 002 calls for 5 050
evaluations, 60 002 for a 30 000-evaluation run.
Resolve the controller list once in `initialize()` and cache it.

### P3. The diversity machinery costs up to 18× the entire optimisation

Same 30 000-evaluation run:

| `diversity_measures` | wall clock |
|---|---:|
| `None` | 0.82 s |
| `"population_diameter"` (one measure) | **4.11 s** |
| six measures | **14.62 s** |

Every measure is O(N²·D) and recomputed from scratch every generation; `PopulationDiameter`
and `AveragePairwiseDistance` each rebuild the *same* pairwise-distance list. Three fixes,
in order of value: (a) cache `_pairwise_distances` per generation across measures;
(b) add a `record_every: int = 1` thinning parameter; (c) note that
`AverageDistanceAroundAllIndividuals` is exactly `((N-1)/N)·AveragePairwiseDistance`, so it
carries no independent information and can be derived for free.

Also: none of the measures normalise by the box diagonal even though they all receive
`bounds`, so diversity values are not comparable across problems — which is precisely what a
diversity study needs.

### P4. `SHADE._sample_r2_vector` allocates a tagged candidate list per trial vector

`shade.py:148-157` builds `[("population", i), …] + [("archive", j), …]` — up to
`NP + 2.6·NP` tuples — for **every** trial vector. Replace with a single
`randrange(NP + len(archive))` and an index comparison.

### P5. `record_snapshots=True` has unbounded memory cost and no thinning

200 generations, `NP=100`, `D=30` → 201 snapshots holding 603 000 floats ≈ 4.8 MB of raw
data, ~39 MB as Python objects. SHADE additionally deep-copies the whole archive every
generation. No `snapshot_every`, no "best only" mode, no streaming-to-disk option.

### P6. `OptimizeResult.to_dict()` converts snapshots twice

`asdict(self)` already recurses into the `GenerationSnapshot` dataclasses; the result is then
thrown away and recomputed by `snapshots_to_dicts`.

---

## 6. Testing

64 tests, **0.11 s**, 88 % line coverage. The speed is the tell: nothing in the suite runs a
real optimisation long enough to expose a dynamics bug.

### T1. The three most error-prone initializers have **zero** coverage

`coverage report -m` on `initializers.py` (68 %):

```
initializers.py   128   41   68%   124-135, 154-165, 184-200, 264, 293
```

Those exact ranges are the bodies of `TentInitializer`, `OppositionInitializer` and
`QuasiOppositionInitializer` — i.e. C1 lives in code that no test executes. Only
`SobolInitializer` and `RandomInitializer` are tested. `benchmarks.py` is at 53 %.

### T2. No invariant / property tests

The suite verifies formulas against hand-computed values with a fake RNG — good, as far as it
goes — but never asserts the *invariants* that would have caught C1–C4 and H1–H10:

* the returned `x` lies inside `bounds` whenever a repairing handler is used;
* `nfev` equals the number of objective calls, for every initializer;
* `best_fitness_history` is monotone non-increasing;
* `len(diversity_history[k]) == nit + 1` for every `k`;
* the initial population contains `NP` distinct points;
* no `NaN`/`inf` ever reaches `memory_f` / `memory_cr`;
* two runs with the same seed agree, **across every (mutation × crossover × initializer ×
  handler × schedule) combination** — a parametrised sweep, which is cheap and would have
  caught H2 immediately.

Hypothesis (property-based testing) would be a strong fit and a nice sentence in the paper.

### T3. No reference-value regression tests

Nothing pins the implementation to published numbers. At minimum: jDE, SHADE and L-SHADE on
a handful of CEC functions at a fixed budget and seed, with tolerances, marked `slow`. This
is what makes the fidelity claim in §3 defensible.

### T4. No benchmark suite at all

For a software paper you will need CEC2014/2017/2022 or BBOB/COCO, 51 runs, and the standard
tables. Right now there are 15 mostly-2-D toy functions, twelve of which silently accept
wrong dimensions (C3). This is the largest single piece of missing work for the publication
plan.

---

## 7. Packaging and publication blockers

### K1. The PyPI name is taken

```
pypi.org/pypi/differential-evolution  ->  TAKEN, v1.12.0
    "Differential Evolution Algorithm with OpenMDAO Driver"
```

`differential_evolution` normalises to the same name. You cannot publish under it. Decide the
distribution name now, because it propagates into the paper, the docs, the DOI and the
citation. (`pydiffevo` was free at the time of writing; so are most `*-de` / `de-*`
compounds. Keep the *import* name `differential_evolution` if you like — distribution and
import names may differ.)

### K2. No LICENSE file

`pyproject.toml` says `license = { text = "MIT" }`, but there is no `LICENSE`, `LICENSE.txt`
or `COPYING` anywhere in the repository, and the built wheel contains no licence file. GitHub
shows no licence, PyPI shows no licence, and JOSS/SoftwareX will reject on this alone. Also,
the `{ text = ... }` table form is deprecated under PEP 639 for `setuptools>=77`; use
`license = "MIT"` plus `license-files = ["LICENSE"]`.

### K3. The README's compatibility claim is false for installed users

README: *"The original `main.py`, `population_initialization.py`, and `testing_functions.py`
remain as compatibility shims."* But `[tool.setuptools.packages.find] include =
["differential_evolution*"]` excludes them, and the built wheel contains only the package:

```
differential_evolution/*.py   +   dist-info      # main.py etc. are absent
```

So `from main import DifferentialEvolution` — the documented migration path, and the subject
of `MIGRATION.md` — works from a git checkout and fails after `pip install`. Either add them
via `py-modules`, or (better) delete them and update README/MIGRATION to point at
`differential_evolution.compat`.

### K4. Absolute local paths leak into the published README

Six links of the form `[ALGORITHM_AUDIT.md](/home/honza/PycharmProjects/differential_evolution/ALGORITHM_AUDIT.md)`
in `README.md` and `ALGORITHM_AUDIT.md`. These are broken on GitHub and end up verbatim in
the PyPI long description (the README is the `Description-Content-Type: text/markdown` body).
Use repo-relative links.

### K5. No CI, and none of the standard project files

Missing: `.github/workflows/`, `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`,
`CODE_OF_CONDUCT.md`, `py.typed`, `tox.ini`/`noxfile.py`, `.pre-commit-config.yaml`.
JOSS requires automated tests; SoftwareX reviewers look for CI badges. A matrix over
3.11/3.12/3.13 × Linux/macOS/Windows running pytest + ruff + mypy is an hour's work.

### K6. `pyproject.toml` is threadbare

No `classifiers`, no `keywords`, no `[project.urls]` (Homepage / Documentation / Issues), no
author email, no maintainer, no ORCID anywhere. No `[project.optional-dependencies].test`
even though the suite needs pytest. No `[tool.pytest.ini_options]`, `[tool.ruff]` or
`[tool.mypy]` sections. `requires-python = ">=3.11"` is asserted but nothing in the code
needs 3.11 — that gratuitously excludes 3.10 users.

### K7. `gpd_ll_function` should not ship as written

`benchmarks.py:108-116`:

```python
def gpd_ll_function(x, sample_path="gpd_sample"):
    with Path(sample_path).open("rb") as handle:
        data = pickle.load(handle)     # <- on EVERY objective evaluation
```

Three problems: it re-opens and **unpickles a file on every single evaluation** (catastrophic
for an optimizer); `pickle.load` on a path that can come from user data is arbitrary code
execution; and the default file `gpd_sample` does not exist in the repository, so the function
cannot run at all. It is also not exported from `__init__.py`, so it is effectively dead code
reachable only through the unpackaged `testing_functions.py` shim. Delete it, or turn it into
a class that loads the sample once in `__init__` and reads it from a documented, non-pickle
format (`.npy`, CSV).

Related: `math.log` in that function has no domain guard, so it raises for `shape <= 0` — and
combined with the default `NoBoundaryHandler` (H9) and no exception policy (H5), it will
reliably kill runs.

### K8. Repository junk

* `.gitignore` contains a line `/=8,` — the fossil of a mis-quoted `pip install "sphinx>=8,<9"`.
* `requirements.txt` contains only comments saying there are no requirements. Delete it or
  make it real; leaving it invites `pip install -r requirements.txt` confusion.
* `docs/api/generated/*.rst` are checked in **and** `autosummary_generate_overwrite = True`,
  so they are regenerated on every build — 60 files of guaranteed diff noise. Gitignore them.
* `docs/conf.py` sets `templates_path = ["_templates"]`, but `docs/_templates/` does not exist.
* `ALGORITHM_AUDIT.md` (26 KB) and `MIGRATION.md` are refactor-process artifacts. Keep them
  (they are genuinely good), but move them under `docs/` so the repository root reads as a
  library, not as a changelog of one refactor.

---

## 8. Lint summary (for the tidy-up pass)

`ruff check --select F,B,RUF,SIM,PERF differential_evolution` → 22 findings:

* **13 × `B905`** `zip()` without `strict=`. On Python ≥3.10 this is free safety, and it
  matters here: `ClipBoundaryHandler` silently *truncates* a trial vector whose length does
  not match `bounds`, turning a dimension bug into a confusing downstream error. Add
  `strict=True` everywhere.
* `F401` unused import `Bounds` in `optimizer.py:11`.
* `B004` `hasattr(x, "__call__")` → `callable(x)` (`diversity.py:258`).
* `RUF036` `None` not at the end of a union (`diversity.py:269`).
* `RUF022` `__all__` unsorted (`__init__.py:65`) — 60+ entries, currently grouped by nothing
  in particular.
* 2 × `SIM102`, 2 × `PERF401`, 1 × `SIM108`. **[2nd pass]** The `1 × E501` listed in the
  first draft was a stray from a separate default-rules run; under
  `--select F,B,RUF,SIM,PERF` the 22 findings are exactly
  13 `B905` + 1 `F401` + 1 `B004` + 1 `RUF022` + 1 `RUF036` + 2 `SIM102` + 1 `SIM108`
  + 2 `PERF401`. Verified independently on ruff 0.16.6.

`mypy differential_evolution` (default settings) → **63 errors in 10 files**, see D1.

---

## 9. What is genuinely good

Stated plainly, because it should go in the paper and because the rest of this document is
relentless:

* **The Sobol implementation is correct.** I cross-checked it against
  `scipy.stats.qmc.Sobol`. It legitimately differs from scipy for D ≥ 3 because scipy uses
  Joe–Kuo direction numbers and this uses Bratley–Fox — but it is a *valid* Sobol' sequence:
  the 1-D projections of the first 2^m points hit every one of the 2^m dyadic bins exactly
  once, for every dimension up to 40, and on 2-D adjacent-pair balance at D=40 it has **19
  failing pairs vs scipy's 22**. The primitive-polynomial and degree tables match Bratley–Fox
  exactly. This is a real piece of work and the audit's claim about the old implementation
  being broken is well supported.
* **L-SHADE actually works.** 10-D, 50 000 evaluations, `NP_init = 18D`, 5 seeds:
  sphere median `2.6e-48`, Rosenbrock median `2.5e-30` (one seed hit exactly `0.0`). That is
  reference-quality convergence.
* **The documentation builds clean under `sphinx -W --keep-going`** — no warnings at all.
  Rare. The numpydoc docstrings are consistent and the `docs/discrepancies.rst` page, which
  separates "deliberate project-specific choice" from "needs manual review", is an unusually
  honest and useful artifact. Keep it and expand it; it is a differentiator.
* **`ALGORITHM_AUDIT.md` is better than most published DE library documentation.** The
  operator-by-operator table with canonical source, base vector, exclusion rules and minimum
  population size is exactly the right structure. It should become a docs page and, edited
  down, a table in the paper.
* **The component decomposition is the right one.** Mutation / crossover / scale controller /
  crossover-rate controller / initializer / boundary handler / diversity measure /
  population schedule is a better factoring than either of the existing DE packages offers.
  The problems in §4 are about *typing and protocol width*, not about the decomposition.
* **Every benchmark formula is correct. [2nd pass]** All 15 exported functions were checked
  against their published global optima — zero mismatches (see C3). The 2-D restriction is a
  missing guard clause, not bad mathematics, which makes C3 much cheaper to fix than it looks.

* `twine check` passes; the deterministic-RNG test harness (`DeterministicRNG`) is a good
  pattern; RNG state is properly instance-scoped rather than global.

---

## 10. Suggested order of work

**Before any PyPI release**

1. C1 tent map, C2 SHADE `NaN`, C3 benchmark dimensions, C4 diversity aliasing.
2. H1 `-inf`, H2 double-`F`, H3 `success`/`message`, H9 default boundary handler,
   H10 JSON `inf`.
3. K1 pick a free distribution name, K2 add `LICENSE`, K3/K4 fix README claims and links,
   K7 remove `gpd_ll_function`.
4. Add CI (pytest + ruff + mypy, 3.11–3.13 × 3 OSes).

**Before 1.0 / before the API is frozen** (these are breaking changes)

5. D3 widen `BoundaryHandler` to receive the target vector; add reflection and
   midpoint/bounce-back handlers.
6. D1 annotate with the real Protocols, ship `py.typed`.
7. A batch-evaluation protocol (`objective(list_of_vectors) -> list[float]`) — without it,
   parallel evaluation can never be added.
8. A stopping-criterion protocol.
9. D2 unify the two `propose()` signatures; D4 formalise the optional lifecycle hooks.
10. H8 decide and document component ownership (deep-copy, or run-scoped state).

**Before the paper**

11. M1/M2 the SHADE/L-SHADE fidelity questions are now **settled against the papers**, so
    these are decisions to make rather than literature to check: `LSHADE` needs a fixed
    `p` (0.11) and `mean_WL` for `M_CR`; `SHADE` should drop the terminal-`M_CR` mechanism it
    borrowed from L-SHADE; the `r1` / `r2` / `p_best` exclusions and strict-vs-`≤` selection
    have to be either aligned with the papers or recorded in `discrepancies.rst`; midpoint
    repair needs D3 first. M10 belongs here too.
12. H6 fix the `DirectedMutation` coefficient (scale-dependence, silent fallback) and correct
    its citation. A candidate source exists but was not read, so this needs you to obtain the
    paper and compare its equation with `mutation.py`; the outcome decides between a
    bibliography fix and a rename.
13. T3/T4 reference-value regression tests + a real benchmark suite (CEC or BBOB) with the
    standard tables. This is the bulk of the remaining work.
14. P1–P3 performance pass, then publish an honest overhead table (µs/evaluation vs scipy) —
    owning the pure-Python trade-off explicitly is far stronger than letting a referee find it.
15. M5 fix the tanh schedule or its docstring, and justify it.

---

## 11. Evidence status of every finding

Nothing here is asserted from memory. This table says exactly how each finding was
established, so you can weight them accordingly — and so the three claims I am *not* fully
certain of are visible rather than buried.

| # | Finding | How it was verified |
|---|---|---|
| C1 | Tent map collapse | `repro.py` — runtime assertion |
| C2 | SHADE `NaN` memory | `repro.py` |
| C3 | 2-D benchmarks take *n*-D input | `repro.py` + `audit.py` (13/15 functions) |
| C4 | Aliased diversity names | `repro.py` |
| H1 | `-inf` flipped to `+inf` | `repro.py` |
| H2 | Two independent `F` values | `repro.py` |
| H3 | Constant `success` / `message` | `repro.py` |
| H4 | Budget ignored at init | `repro.py` |
| H5 | Objective exception kills the run | `repro.py` |
| H6 | `DirectedMutation` scale explosion | `repro.py` |
| H7 | Sobol ignores the RNG | `repro.py` |
| H8 | Shared component state wiped | `repro.py` |
| H9 | Default handler ignores bounds | `repro.py` |
| H10 | Invalid JSON | `repro.py` |
| M1a | SHADE `r1` excludes `p_best` | `repro2.py` — 4000 draws, index 4 never appears |
| M1b | Archive overshoots within a generation | `repro2.py` — reached 33 vs a limit of 20 |
| M1c | `_sample_r2_vector` reads `self.population` | **code reading only** — currently harmless, flagged as a latent hazard |
| M1d | `p_best` selection excludes the target | `verify_extra.py` — 5 000 draws, the best index never appears; L-SHADE 2014 Eq. (3) |
| M1e | Strict `<` selection vs the papers' `<=` | `verify_extra.py` + both papers' Eq. (6) |
| M1f | `SHADE` carries L-SHADE's terminal `M_CR` | `verify_extra.py` + "terminal" absent from SHADE 2013, present in L-SHADE 2014 Alg. 1 |
| M2 | `LSHADE` inherits SHADE's `p` and `M_CR` rule | code reading + **both papers read directly**: SHADE Eq. (17)/(18)/(20), L-SHADE Eq. (3)/(7) and the parameter table. Resolved, no longer uncertain |
| M3 | `Best*` excludes `best_index` | `repro2.py` — 3000 draws, index 0 never appears |
| M4 | Late population-size failure | `repro2.py` |
| M5 | tanh schedule vs its docstring | `repro.py` — 96 % of the reduction by half the run |
| M6 | QOBL branch is a no-op | `repro2.py` — both branches uniform on [M, O], means 0.498 / 0.496 |
| M7 | Trig mutation not shift-invariant | `repro2.py` |
| M8 | `run()` twice is a no-op; `population_size` mutated | `repro.py` |
| M9 | Initializer `nfev` accounting differs | `repro.py` |
| M10 | Default `LSHADE()` uses generation progress for LPSR | `verify_extra.py` — NP 100 → 4 in 50 generations on 2 748 evaluations |
| C3+ | All 15 benchmark formulas correct | `verify_extra.py` — evaluated at published optima, 0 mismatches |
| D1 | Protocols unused in annotations | `audit.py` — 7 fields typed `object`; mypy 56 errors |
| D2 | Clashing `propose()` signatures | `repro2.py` — `TypeError` inside generation 1 |
| D3 | `BoundaryHandler` cannot see the parent | `audit.py` — signature is `(trial_vector, bounds, rng)` |
| D4 | Half-declared lifecycle; no mutation protocol | `audit.py` — AST scan of the Protocol bodies |
| D5 | Components unusable standalone | `repro.py` |
| D6 | `== "none"` on an arbitrary object | `repro2.py` |
| D7 | `hasattr(x, "__call__")` | `repro2.py` — a non-callable object is accepted |
| D8 | Missing capabilities | `audit.py` — inventory against `__all__` and the constructor signature |
| P1 | Sampling hotspot | `audit.py` — timeit, 6.9 µs vs 0.8 µs |
| P2 | `dataclasses.fields()` per trial | `audit.py` — instrumented counter, 2.0 calls/trial |
| P3 | Diversity cost | `audit.py` — 1.0× / 5.3× / 18.6× |
| P4 | SHADE candidate-list allocation | `repro2.py` — instrumented RNG, lists of up to 203 tuples |
| P5 | Snapshot memory | `audit.py` — 603 000 floats, no thinning knob |
| P6 | Double snapshot conversion | `repro2.py` |
| T1 | Initializers uncovered | `audit.py` — `coverage report` |
| T2/T3 | No invariant or reference tests | `audit.py` — marker scan of the test file |
| T4 | No benchmark suite | `audit.py` — 15 functions, none from a standard suite |
| K1 | PyPI name taken | `audit.py` — live PyPI JSON query |
| K2 | No LICENSE | `audit.py` — filesystem + `pyproject` |
| K3 | Shims not packaged | `audit.py` + wheel contents listing |
| K4 | Absolute paths in README | `audit.py` — 4 in README.md, 2 in ALGORITHM_AUDIT.md |
| K5 | No CI or project files | `audit.py` |
| K6 | Threadbare `pyproject.toml` | `audit.py` |
| K7 | `gpd_ll_function` | `audit.py` — `pickle.load` in the body, data file absent, not exported |
| K8 | Repo junk | `audit.py` — `/=8,`, empty `requirements.txt`, 60 generated `.rst`, missing `_templates` |

### Where the first pass was not certain — now resolved

The first pass flagged three claims as resting on literature rather than execution. A second
pass pulled both primary papers and settled all three. Their text is quoted in M1, M2 and H6.

1. **The `M_CR` update rule in L-SHADE — CONFIRMED.** L-SHADE 2014 Eq. (7) defines a single
   weighted Lehmer mean and states that "`S` refers to either `S_CR` or `S_F`". SHADE 2013
   keeps them separate: Eq. (17) arithmetic for `M_CR`, Eq. (18) Lehmer for `M_F`. The library
   implements the SHADE 2013 rule in both classes, so `LSHADE` is wrong here.
2. **The fixed `p = 0.11` in L-SHADE — CONFIRMED.** Eq. (3) selects `x_pbest` from the top
   `N × p` members with `p` a fixed hyper-parameter; the paper's parameter table sweeps
   `{0.05, …, 0.15}` and reports 0.11. SHADE 2013 Eq. (20) is the randomised
   `p_i = rand[2/NP, 0.2]`. The library uses the randomised form in both classes.
3. **The `DirectedMutation` citation — STILL OPEN, but narrowed.** The DOI is definitely the
   trigonometric-mutation paper, so the citation as written is wrong; that part is certain.
   The first pass concluded that no primary source exists at all and that the operator should
   be renamed. That is now less likely — Fan & Lampinen do have a separate paper titled
   *"A directed mutation operation for the differential evolution algorithm"*, and there is
   also Fan, Lampinen & Dulikravich, *"Improvements to Mutation Donor Formulation of
   Differential Evolution"*, EUROGEN 2003 — but **the second pass could not read either
   paper**, so whether their equation is the one implemented here is unverified. Get the
   original before citing it.

### What remains unverified

* **Everything about the Fan & Lampinen directed-mutation paper except its existence.** The
  title and authorship are attested by a ResearchGate record and by secondary citations in DE
  surveys; the volume / issue / pages / DOI are not in Crossref, and neither pass could open
  the full text. So **the formula match is unverified** — this is the one place where the
  review points at a source it has not read, and the only recommendation that depends on
  reading it (fix the DOI vs. rename the operator) is left to you.
* **M1c** is still code reading only. It is currently harmless and flagged as a latent hazard,
  not as a live bug.
* The **`NeighborhoodSearchMutation`** citation is acknowledged as unverified in
  `docs/discrepancies.rst`; the second pass confirmed the implemented mathematics matches the
  NSDE definition as documented, but did not chase the primary source.

Everything else in the table above is machine-verified, quoted from a primary source, or a
direct checkable fact about the repository (a signature, an absent file, a missing `__all__`
entry).


---

## Appendix — reproducing this review

```bash
conda create -n differential_evolution python=3.12 pytest numpy scipy -y
conda activate differential_evolution
git clone https://github.com/vrbaj/differential_evolution && cd differential_evolution
pip install -e . ruff mypy coverage build twine
pytest tests -q                                   # 64 passed in 0.11s
ruff check --select F,B,RUF,SIM,PERF differential_evolution
mypy --ignore-missing-imports differential_evolution
coverage run -m pytest tests -q && coverage report -m --include="differential_evolution/*"
python -m sphinx -b html -W --keep-going docs docs/_build/html   # clean
```

Then, from the repository root:

```bash
python repro.py     # 19 runtime assertions: C1-C4, H1-H10, M5, M8, M8b, M9, D5
python repro2.py    # 11 runtime assertions: M1a, M1b, M3, M4, M6, M7, D2, D6, D7, P4, P6
python audit.py     # 14 repository/packaging/API checks (K1-K8, D1, D3, D4, D8, T1, T2, T4)
                    #  + the P1-P6 performance measurements and the lint/mypy summary
python verify_extra.py   # 5 second-pass checks: M1d, M1e, M1f, M10, plus the
                         # benchmark-formula sweep behind C3 and section 9
```

All four take well under a minute apart from the performance section of `audit.py`
(~30 s). `audit.py --offline` skips the live PyPI name query. Re-run them after fixing to
watch the FAIL lines turn into PASS.
